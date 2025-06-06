import random

from torch.utils.data import DataLoader
import numpy as np
from data import dataset
from entitites.BaseAgent import BaseAgent
import torch
import config
import copy
from tqdm import tqdm
from environments.QEnvironment import QEnvironment
from models.q_model import QModel
from entitites.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    def __init__(self, optimizer, criterion, early_stopping=10):
        self.device = config.get_device()
        self.model = QModel().to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.early_stopping = early_stopping

    def train(self, train_set, val_set):
        np.random.seed(
            config.RANDOM_SEED)  # TODO remodel to only jump action because jumping environment moves every state update by once to the right
        self.model.to(self.device)
        replay_buffer = ReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
        stop_counter = 0
        train_loss = 0
        best_val_loss = float('inf')
        train_loader = DataLoader(train_set, **config.PARAMS)
        val_loader = DataLoader(val_set, **config.PARAMS)

        self.warm_start_replay_buffer(replay_buffer, train_loader, self.device)
        for epoch in range(config.MAX_EPOCHS):
            self.model.train()
            batch_loss = 0
            epsilon = max(0.2 - epoch * 0.05, 0.01)  # epsilon decay (low because of warm start
            for environments, actions in tqdm(train_loader):
                expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy()) for env, env_actions in
                                zip(environments, actions)]
                agent_start_positions = []
                for expert_path in expert_paths:
                    start_idx = np.random.randint(config.ENV_SIZE - config.NUM_STEPS_ENV)
                    agent_start_positions.append(expert_path[start_idx])
                for env_id in range(config.BATCH_SIZE):
                    curr_env = QEnvironment(environment=environments[env_id].numpy(), size=config.ENV_SIZE,
                                            start_pos=agent_start_positions[env_id])
                    for step in range(config.NUM_STEPS_ENV):
                        state_tensor = torch.tensor(curr_env.state, dtype=torch.float32).unsqueeze(0).to(self.device)
                        action = self.epsilon_greedy_action(state_tensor, epsilon)
                        next_state, reward, done = curr_env.step(action)

                        replay_buffer.push(curr_env.state.copy(), action, reward, next_state.copy(), done)

                        if len(replay_buffer) >= 0:
                            states, actions, rewards, next_states, dones = replay_buffer.sample(
                                min(len(replay_buffer), config.NUM_REPLAY_SAMPLE))
                            next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
                            states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
                            actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
                            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

                            q_values = self.model(states_tensor)
                            q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                            with torch.no_grad():
                                next_q_values = self.model(next_states_tensor).max(1)[0]
                                targets = rewards_tensor + config.GAMMA * next_q_values * (1 - dones_tensor)

                            step_loss = self.criterion(q_values, targets)

                            self.optimizer.zero_grad()
                            step_loss.backward()
                            self.optimizer.step()

                            batch_loss += step_loss.item()

                        if done:
                            break
            batch_loss /= max(1, config.BATCH_SIZE)
            train_loss += batch_loss
            val_loss = self.loss(val_loader)
            print(f'Validation loss: {val_loss} at epoch {epoch + 1}/{config.MAX_EPOCHS}')
            if val_loss < best_val_loss:
                print(f'New best validation loss: {val_loss}\n old best validation loss: {best_val_loss}')
                best_val_loss = val_loss
                stop_counter = 0
                best_model = copy.deepcopy(self.model)
                config.save_model(self.model, name=f"Q_{epoch + 1}")
            else:
                stop_counter += 1

            if stop_counter >= self.early_stopping:
                epochs_ran = epoch + 1
                break
        config.save_model(best_model, name="final_Q")
        print(f'Final training loss: {train_loss / epochs_ran:.4f} after {epochs_ran} epochs')
        self.model = best_model

    def evaluate(self, environment):
        pass

    def loss(self, val_loader):
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0
        count = 0
        with torch.no_grad():
            for environments, actions in val_loader:
                for env, env_actions in zip(environments, actions):
                    environment = env.numpy()
                    expert_path = dataset.reconstruct_path(env.numpy(), env_actions.numpy())
                    start_idx = np.random.randint(config.ENV_SIZE - config.NUM_STEPS_ENV)
                    start_pos = expert_path[start_idx]
                    curr_env = QEnvironment(environment=environment, size=config.ENV_SIZE, start_pos=start_pos)
                    curr_env.reset()
                    steps_loss = 0
                    steps_taken = 0
                    for action_gt in env_actions[:config.NUM_STEPS_ENV]:
                        state_tensor = torch.tensor(curr_env.state, dtype=torch.float32).unsqueeze(0).to(self.device)
                        q_values = self.model(state_tensor)
                        action_gt = int(action_gt.item())
                        predicted_q = q_values[0, action_gt]
                        next_state, reward, done = curr_env.step(action_gt)
                        target_q = torch.tensor([reward], dtype=torch.float32).to(self.device)
                        step_loss = self.criterion(predicted_q.unsqueeze(0), target_q)
                        steps_loss += step_loss.item()
                        steps_taken += 1

                        if done:
                            break
                    if steps_taken > 0:
                        total_loss += steps_loss / steps_taken
                        count += 1
        return total_loss / max(1, count)

    def warm_start_replay_buffer(self, replay_buffer, data_loader, device, max_traverse_steps=config.NUM_STEPS_ENV,
                                 random_start=True):
        agent_start_positions = []
        max_traverse_steps = random.randint(1, config.WINDOW_LEN) * max_traverse_steps
        for environments, actions in data_loader:
            expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy()) for env, env_actions in
                            zip(environments, actions)]
            for expert_path in expert_paths:
                start_idx = np.random.randint(config.ENV_SIZE - config.NUM_STEPS_ENV)
                agent_start_positions.append(expert_path[start_idx])
            for env, env_actions, expert_path, start_pos in zip(environments, actions, expert_paths,
                                                                agent_start_positions):
                if random_start:
                    curr_env = QEnvironment(environment=env.numpy(), size=config.ENV_SIZE, start_pos=start_pos)
                else:
                    curr_env = QEnvironment(environment=env.numpy(), size=config.ENV_SIZE, start_pos=expert_path[0])
                for action in env_actions[:max_traverse_steps]:
                    next_state, reward, done = curr_env.step(action)
                    replay_buffer.push(curr_env.state.copy(), action, reward, next_state.copy(), done)
                    curr_env.state = next_state.copy()
                    if done:
                        break
        print(f'Replay buffer size: {len(replay_buffer)} with max_traverse_steps: {max_traverse_steps}')
        return len(replay_buffer)

    def epsilon_greedy_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            # Explore: choose a random action
            return random.randint(0, len(config.Actions) - 1)
        else:
            # Exploit: choose action with highest Q-value
            with torch.no_grad():
                actions = self.model(state)
                action = torch.argmax(actions, dim=1)
                return action.item()