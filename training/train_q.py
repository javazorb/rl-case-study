import random
from collections import deque
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
import numpy as np
import data.dataset as dataset
from environments.QEnvironment import QEnvironment


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def train(model, device, train_data, val_data, optimizer, criterion, early_stopping=10, epsilon=0.1):
    np.random.seed(config.RANDOM_SEED)
    model.to(device)
    replay_buffer = ReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
    stop_counter = 0
    train_loss = 0
    best_val_loss = float('inf')
    train_loader = DataLoader(train_data, **config.PARAMS)
    val_loader = DataLoader(val_data, **config.PARAMS)

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        batch_loss = 0
        for environments, actions in tqdm(train_loader):
            mini_batch_loss = 0
            expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy()) for env, env_actions in
                            zip(environments, actions)]
            agent_start_positions = []
            for expert_path in expert_paths:
                start_idx = np.random.randint(config.ENV_SIZE - config.NUM_STEPS_ENV)
                agent_start_positions.append(expert_path[start_idx])
            for env_id in range(config.BATCH_SIZE):
                curr_env = QEnvironment(environment=environments[env_id].numpy(), size=config.ENV_SIZE,
                                        start_pos=agent_start_positions[env_id])
                env_reward = 0
                for step in range(config.NUM_STEPS_ENV):
                    state_tensor = torch.tensor(curr_env.state, dtype=torch.float32).unsqueeze(0).to(device)
                    action = epsilon_greedy_action(model, state_tensor, epsilon)
                    next_state, reward, done = curr_env.step(action)

                    replay_buffer.push(curr_env.state.copy(), action, reward, next_state.copy(), done)

                    if len(replay_buffer) >= 0:
                        states, actions, rewards, next_states, dones = replay_buffer.sample(min(len(replay_buffer), config.NUM_REPLAY_SAMPLE))
                        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
                        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
                        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
                        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
                        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)

                        q_values = model(states_tensor)
                        q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                        with torch.no_grad():
                            next_q_values = model(next_states_tensor).max(1)[0]
                            targets = rewards_tensor + config.GAMMA * next_q_values * (1 - dones_tensor)

                        step_loss = criterion(q_values, targets)

                        optimizer.zero_grad()
                        step_loss.backward()
                        optimizer.step()

                        batch_loss += step_loss.item()

                    if done:
                        break

                    #next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
                    #with torch.no_grad():
                    #    next_max_Q = model(next_state).max().item()
                    #    target_Q = torch.tensor(reward + config.GAMMA * next_max_Q).to(device)
                    #predicted_Q = torch.max(model(next_state))
                    #step_loss = criterion(predicted_Q, target_Q)
                    #optimizer.zero_grad()
                    #step_loss.backward()
                    #optimizer.step()
                    #mini_batch_loss += step_loss.item()
                    #if done:
                    #    break
                #curr_env.reset()
                #mini_batch_loss = mini_batch_loss / config.NUM_STEPS_ENV
                #batch_loss += mini_batch_loss
            #batch_loss /= config.BATCH_SIZE
        batch_loss /= max(1, len(replay_buffer))
        train_loss += batch_loss
        val_loss = loss(model, device, val_loader, criterion)
        print(f'Validation loss: {val_loss.item()} at epoch {epoch + 1}/{config.MAX_EPOCHS}')
        if val_loss < best_val_loss:
            print(f'New best validation loss: {val_loss.item()}\n old best validation loss: {best_val_loss}')
            best_val_loss = val_loss
            stop_counter = 0
            best_model = model
            config.save_model(model, name=f"BC_{epoch + 1}")
        else:
            stop_counter += 1

        if stop_counter >= early_stopping:
            epochs_ran = epoch + 1
            break
    config.save_model(best_model, name="final_BC")
    print(f'Final training loss: {train_loss / epochs_ran:.4f} after {epochs_ran} epochs')


def epsilon_greedy_action(model, state, epsilon=0.1):
    if random.random() < epsilon:
        # Explore: choose a random action
        return random.randint(0, len(config.Actions) - 1)
    else:
        # Exploit: choose action with highest Q-value
        with torch.no_grad():
            actions = model(state)
            action = torch.argmax(actions, dim=1)
            return action.item()


def loss(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for states, actions in val_loader: #TODO change structure to resemble train method
            states = states.to(device)
            actions = actions.to(device)
            q_values = model(states)
            predicted = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            total_loss += predicted.sum().item()
    return total_loss / len(val_loader.dataset)

def evaluate_model(model, device, val_loader, num_episodes=5):
    model.eval()
    total_rewards = []

    for environments, actions in val_loader:
        for env, act in zip(environments, actions):
            environment = env.numpy()
            expert_path = dataset.reconstruct_path(environment, act.numpy())
            start_pos = expert_path[0]

            curr_env = QEnvironment(environment=environment, size=config.ENV_SIZE, start_pos=start_pos)
            curr_env.reset()
            cumulative_reward = 0

            for _ in range(config.NUM_STEPS_ENV):
                state_tensor = torch.tensor(curr_env.state, dtype=torch.float32).unsqueeze(0).to(device)
                action = epsilon_greedy_action(model, state_tensor, epsilon=0.0)  # greedy
                _, reward, done = curr_env.step(action)
                cumulative_reward += reward
                if done:
                    break

            total_rewards.append(cumulative_reward)

            if len(total_rewards) >= num_episodes:
                break
        if len(total_rewards) >= num_episodes:
            break

    avg_reward = np.mean(total_rewards)
    print(f'Average validation reward over {num_episodes} episodes: {avg_reward:.2f}')
    return avg_reward

