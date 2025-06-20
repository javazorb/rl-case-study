import random
from collections import Counter
from itertools import islice
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


def warm_start_replay_buffer(replay_buffer, data_loader, device, max_traverse_steps=config.NUM_STEPS_ENV,
                             random_start=True, resample=False, max_trajectories=300, hybrid_mode=False, agent=None):
    max_resampling_steps = 0
    if resample:
        max_resampling_steps = config.REPLAY_BUFFER_SIZE * 0.1
    sampling_steps = 0
    agent_start_positions = []
    max_traverse_steps = random.randint(1, config.WINDOW_LEN) * max_traverse_steps

    random.shuffle(data_loader)
    hold_out_loader = islice(data_loader, int(0.1 * len(data_loader)))
    data_loader = islice(data_loader, int(0.3 * len(data_loader))) # using 30% of combined train and val set for warmstart
    for environments, actions in data_loader:
        expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy()) for env, env_actions in
                        zip(environments, actions)]
        if resample and sampling_steps > max_resampling_steps:
            break
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
                replay_buffer.push(curr_env.state.copy(), int(action), reward, next_state.copy(), done)
                curr_env.state = next_state.copy()
                if done:
                    break
        sampling_steps += 1
        if sampling_steps >= max_traverse_steps:
            break
    if hybrid_mode:
        for environments, actions in hold_out_loader:
            envs = environments
            for env in environments:
                curr_env = QEnvironment(environment=env.numpy(), size=config.ENV_SIZE, start_pos=None)
                state = curr_env.reset()
                done = False
                steps = 0
                while not done and steps < config.MAX_STEPS:
                    action = DQNAgent.epsilon_greedy_action(agent, state=state, epsilon=1)
                    next_state, reward, done = curr_env.step(action)
                    replay_buffer.push(curr_env.state.copy(), int(action), reward, next_state.copy(), done)
                    state = next_state
                    steps += 1

    print(f'Replay buffer size: {len(replay_buffer)} with max_traverse_steps: {max_traverse_steps}')
    return len(replay_buffer)


def validate_agent(agent, environments, agent_start_positions, max_steps=config.MAX_STEPS):
    total_rewards = []
    total_steps = []
    action_hist = Counter()
    successes = 0

    for env_id in range(len(environments)):
        curr_env = QEnvironment(environment=environments[env_id], size=config.ENV_SIZE,
                                start_pos=agent_start_positions[env_id])
        state = curr_env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            action = agent.epsilon_greedy_action(state, epsilon=0)
            action_hist[action] += 1

            next_state, reward, done = curr_env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
        if curr_env.current_position == curr_env.goal_position:
            successes += 1
        total_rewards.append(episode_reward)
        total_steps.append(steps)
    return {
        "avg_reward": np.mean(total_rewards),
        "avg_length": np.mean(total_steps),
        "success_rate": successes / len(environments),
        "action_dist": dict(action_hist)
    }


class DQNAgent(BaseAgent):
    def __init__(self, optimizer, criterion, early_stopping=10):
        self.device = config.get_device()
        self.model = QModel().to(self.device)
        self.target_model = QModel().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optimizer
        self.criterion = criterion
        self.early_stopping = early_stopping

        self.target_update_epoch = 5
        self.target_update_counter = 0

    def train_old(self, train_set, val_set):
        np.random.seed(
            config.RANDOM_SEED)  # TODO remodel to only jump action because jumping environment moves every state update by once to the right
        self.model.to(self.device)
        replay_buffer = ReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
        stop_counter = 0
        train_loss = 0
        best_val_loss = float('inf')
        train_loader = DataLoader(train_set, **config.PARAMS)
        val_loader = DataLoader(val_set, **config.PARAMS)

        warm_start_replay_buffer(replay_buffer, train_loader, self.device) # warm start
        for epoch in range(config.MAX_EPOCHS):
            self.model.train()
            batch_loss = 0
            #epsilon = max(0.2 - epoch * 0.05, 0.01)  # epsilon decay (low because of warm start
            epsilon = max(0.01, 0.5 * (0.9 ** epoch)) # smoother decay
            for environments, actions in tqdm(train_loader):
                expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy()) for env, env_actions in
                                zip(environments, actions)]
                agent_start_positions = []
                for expert_path in expert_paths:
                    start_idx = np.random.randint(config.ENV_SIZE - config.NUM_STEPS_ENV)
                    agent_start_positions.append(expert_path[start_idx])
                actual_batch_size = min(config.BATCH_SIZE, len(environments))
                indices = np.random.choice(len(environments), actual_batch_size, replace=False)
                for env_id in indices:
                    curr_env = QEnvironment(environment=environments[env_id].numpy(), size=config.ENV_SIZE,
                                            start_pos=agent_start_positions[env_id])
                    for step in range(config.NUM_STEPS_ENV):
                        state_tensor = torch.tensor(curr_env.state, dtype=torch.float32).unsqueeze(0).to(self.device)
                        action = self.epsilon_greedy_action(state_tensor, epsilon)
                        if action == 1:
                            action = 3
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
                                #next_q_values = self.model(next_states_tensor).max(1)[0]
                                next_q_values = self.target_model(next_states_tensor).max(1)[0]
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

            self.target_update_counter += 1
            if self.target_update_counter >= self.target_update_epoch:
                self.target_update_counter = 0
                self.target_model.load_state_dict(self.model.state_dict())

        config.save_model(best_model, name="final_Q")
        print(f'Final training loss: {train_loss / epochs_ran:.4f} after {epochs_ran} epochs')
        self.model = best_model

    def train(self, train_set, val_set):
        train_loader = DataLoader(train_set, **config.PARAMS)
        val_loader = DataLoader(val_set, **config.PARAMS)
        replay_buffer = ReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)

        warm_start_replay_buffer(replay_buffer, list(train_loader) + list(val_loader), self.device, hybrid_mode=True, agent=self)
        replay_buffer.visualize_content()
        best_model = copy.deepcopy(self.model)
        step_counter = 0
        best_val_loss = float('inf')
        best_avg_train_loss = float('inf')
        stop_counter = 0
        self.target_update_counter = 0
        val_batch_env = None
        val_batch_start_pos = None
        action_distribution = Counter()
        epoch_qs = []
        epoch_next_qs = []
        epoch_action_counts = []

        for epoch in range(config.MAX_EPOCHS):
            self.model.train()
            batch_loss = 0

            for environments, actions in tqdm(train_loader):
                expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy()) for env, env_actions in
                                zip(environments, actions)]
                agent_start_positions = []
                for expert_path in expert_paths:
                    start_idx = np.random.randint(config.ENV_SIZE - config.NUM_STEPS_ENV)
                    agent_start_positions.append(expert_path[start_idx])

                val_batch_env = environments
                val_batch_start_pos = agent_start_positions

                for env_id in range(len(environments)):
                    curr_env = QEnvironment(environment=environments[env_id].numpy(), size=config.ENV_SIZE,
                                            start_pos=agent_start_positions[env_id])

                    for _ in range(config.NUM_STEPS_ENV):
                        state_tensor = torch.tensor(curr_env.state, dtype=torch.float32).unsqueeze(0).to(self.device)

                        epsilon = max(config.MIN_EPSILON, 1.0 - step_counter / config.EPS_DECAY)
                        action = self.epsilon_greedy_action(state_tensor, epsilon)
                        if action == 1:
                            action = 3  # your remapping
                        action_distribution[action] += 1

                        next_state, reward, done = curr_env.step(action)
                        replay_buffer.push(curr_env.state.copy(), int(action), reward, next_state.copy(), done)
                        curr_env.state = next_state.copy()

                        if len(replay_buffer) >= config.NUM_REPLAY_SAMPLE:
                            states, actions_batch, rewards, next_states, dones = replay_buffer.sample(
                                config.NUM_REPLAY_SAMPLE)

                            states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
                            actions_tensor = torch.tensor(actions_batch, dtype=torch.long).to(self.device)
                            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                            next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
                            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

                            q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                            with torch.no_grad():
                                next_q_values = self.target_model(next_states_tensor).max(1)[0]
                                targets = rewards_tensor + config.GAMMA * next_q_values * (1 - dones_tensor)

                            loss = self.criterion(q_values, targets)

                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss.item()
                            epoch_qs.append(q_values.mean().item())
                            epoch_next_qs.append(next_q_values.mean().item())
                            batch_actions_counts = torch.bincount(torch.tensor(actions_batch))
                            epoch_action_counts.append(batch_actions_counts)

                        step_counter += 1
                        self.target_update_counter += 1

                        if done:
                            break

                        if self.target_update_counter >= self.target_update_epoch:
                            self.target_update_counter = 0
                            self.target_model.load_state_dict(self.model.state_dict())

            avg_train_loss = batch_loss / max(1, len(train_loader) * config.NUM_STEPS_ENV)
            val_loss = self.loss(val_loader)
            # Epoch summary
            avg_epoch_q = sum(epoch_qs) / len(epoch_qs)
            avg_epoch_next_q = sum(epoch_next_qs) / len(epoch_next_qs)
           # action_dist = {i: int(epoch_action_counts[i].item()) for i in range(len(epoch_action_counts)) if
           #                epoch_action_counts[i] > 0}
            print(
                f'Epoch {epoch + 1}/{config.MAX_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Epsilon: {epsilon:.4f}')
            print(f"Avg Q: {avg_epoch_q:.4f} | Avg Next Q: {avg_epoch_next_q:.4f}")
            #print(f"Action Distribution: {action_dist}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss
                best_avg_train_loss = avg_train_loss if avg_train_loss < best_avg_train_loss else best_avg_train_loss
                stop_counter = 0
                best_model = copy.deepcopy(self.model)
                config.save_model(self.model, name=f"Q_{epoch + 1}")
            else:
                stop_counter += 1

            if stop_counter >= self.early_stopping:
                print(f'Early stopping at epoch {epoch + 1}')
                break
            #val_stats = validate_agent(self, val_batch_env, val_batch_env, max_steps=config.MAX_STEPS)
            #print(val_stats)

        config.save_model(best_model, name="final_Q")
        self.model = copy.deepcopy(best_model)


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
                        if action_gt == 3: # for 2 states
                            action_gt = 1
                        #action_gt = int(action_gt.item())
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

    def epsilon_greedy_action(self, state, epsilon=0.1):
        selected_action = -1
        if random.random() < epsilon:
            # Explore: choose a random action
            selected_action = random.randint(0, len(config.QActions) - 1)
        else:
            # Exploit: choose action with highest Q-value
            with torch.no_grad():
                actions = self.model(state)
                action = torch.argmax(actions, dim=1)
                selected_action = action.item()
        if selected_action == 1:
            selected_action = 3
        return selected_action