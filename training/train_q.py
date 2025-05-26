import random
from collections import deque
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
import numpy as np
import data.dataset as dataset
import copy
from environments.QEnvironment import QEnvironment
import matplotlib.pyplot as plt
import sys


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


def warm_start_replay_buffer(replay_buffer, data_loader, device, max_traverse_steps=config.NUM_STEPS_ENV, random_start=True):
    agent_start_positions = []
    max_traverse_steps = random.randint(1, config.WINDOW_LEN) * max_traverse_steps
    for environments, actions in data_loader:
        expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy()) for env,env_actions in zip(environments, actions)]
        for expert_path in expert_paths:
            start_idx = np.random.randint(config.ENV_SIZE - config.NUM_STEPS_ENV)
            agent_start_positions.append(expert_path[start_idx])
        for env, env_actions, expert_path, start_pos in zip(environments, actions, expert_paths, agent_start_positions):
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


def train(model, device, train_data, val_data, optimizer, criterion, early_stopping=10, epsilon=0.1):
    np.random.seed(config.RANDOM_SEED)
    model.to(device)
    replay_buffer = ReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
    stop_counter = 0
    train_loss = 0
    best_val_loss = float('inf')
    train_loader = DataLoader(train_data, **config.PARAMS)
    val_loader = DataLoader(val_data, **config.PARAMS)
    # Maybe TODO warm start for DQN with filling replay buffer with expert actions before training
    warm_start_replay_buffer(replay_buffer, train_loader, device)
    for epoch in range(config.MAX_EPOCHS):
        model.train()
        batch_loss = 0
        epsilon = max(0.9 - epoch * 0.05, 0.1) #epsilon decay
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
        batch_loss /= max(1, config.BATCH_SIZE)
        train_loss += batch_loss
        val_loss = loss(model, device, val_loader, criterion)
        print(f'Validation loss: {val_loss} at epoch {epoch + 1}/{config.MAX_EPOCHS}')
        if val_loss < best_val_loss:
            print(f'New best validation loss: {val_loss}\n old best validation loss: {best_val_loss}')
            best_val_loss = val_loss
            stop_counter = 0
            best_model = copy.deepcopy(model)
            config.save_model(model, name=f"Q_{epoch + 1}")
        else:
            stop_counter += 1

        if stop_counter >= early_stopping:
            epochs_ran = epoch + 1
            break
    config.save_model(best_model, name="final_Q")
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
    model.to(device)
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
                    state_tensor = torch.tensor(curr_env.state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = model(state_tensor)
                    action_gt = int(action_gt.item())
                    predicted_q = q_values[0, action_gt]
                    next_state, reward, done = curr_env.step(action_gt)
                    target_q = torch.tensor([reward], dtype=torch.float32).to(device)
                    step_loss = criterion(predicted_q.unsqueeze(0), target_q)
                    steps_loss += step_loss.item()
                    steps_taken += 1

                    if done:
                        break
                if steps_taken > 0:
                    total_loss += steps_loss / steps_taken
                    count += 1
    return total_loss / max(1, count)


def evaluate_model(model, device, val_loader, num_episodes=5):
    model.to(device)
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


def visualize_agent_path(env, path,  save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.axis('off')
    ax.imshow(env, cmap='gray', origin='lower', vmin=0, vmax=255)
    path_x = [p[0] for p in path]  # col
    path_y = [p[1] for p in path]  # row
    ax.plot(path_x, path_y, color='blue', marker='o', linewidth=2, label="Agent Path")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Agent Path Over Environment")
    ax.axis('off')
    plt.legend()
    plt.show()


import os

def evaluate_model_and_vis(model, device, val_loader, num_episodes=5, save_dir=None):
    model.to(device)
    model.eval()
    total_rewards = []

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    episode_idx = 0
    np.set_printoptions(threshold=sys.maxsize)
    for environments, actions in val_loader:
        for env, act in zip(environments, actions):
            #plt.imshow(env, cmap='gray', origin='lower')  # or origin='upper' if you prefer
            #plt.title("Environment with (row, col) labels")

            # BONUS: Overlay row,col labels
            #for i in range(env.shape[0]):
            #    for j in range(env.shape[1]):
            #        plt.text(j, i, f'{i},{j}', ha='center', va='center', fontsize=6, color='red')

            #plt.grid(False)
            #plt.show()
            print(env.numpy().tolist())
            environment = env.numpy()
            expert_path = dataset.reconstruct_path(environment, act.numpy())
            start_pos = expert_path[0]

            curr_env = QEnvironment(environment=environment, size=config.ENV_SIZE, start_pos=start_pos)
            curr_env.reset()
            cumulative_reward = 0

            # Track agent positions
            agent_path = [curr_env.current_position]

            for _ in range(config.NUM_STEPS_ENV*5):
                #visualize_agent_path(environment, agent_path, save_path=None,
                #                     title=f"Episode {episode_idx} | Reward: {cumulative_reward}")
                state_tensor = torch.tensor(curr_env.state, dtype=torch.float32).unsqueeze(0).to(device)
                action = epsilon_greedy_action(model, state_tensor, epsilon=0.0)  # greedy
                _, reward, done = curr_env.step(action)
                if reward == -1:
                    print(f'Episode obstacle at? {env[curr_env.current_position[0]][curr_env.current_position[1]]}')
                cumulative_reward += reward
                print(f'Agent position: {curr_env.current_position}\t action chosen: {action}\t reward: {reward}')
                agent_path.append(curr_env.current_position)
                if done:
                    break

            total_rewards.append(cumulative_reward)

            # Visualize and optionally save
            if save_dir:
                save_path = os.path.join(save_dir, f"episode_{episode_idx}.png")
                visualize_agent_path(environment, agent_path, save_path=save_path,title=f"Episode {episode_idx} | Reward: {cumulative_reward}")
            else:
                visualize_agent_path(environment, agent_path, title=f"Episode {episode_idx} | Reward: {cumulative_reward}")

            episode_idx += 1
            if episode_idx >= num_episodes:
                break

        if episode_idx >= num_episodes:
            break

    avg_reward = np.mean(total_rewards)
    print(f'Average validation reward over {num_episodes} episodes: {avg_reward:.2f}')
    return avg_reward
