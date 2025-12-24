import copy
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sympy.abc import alpha

import config
from environments.QEnvironment import QEnvironment
from models.base_model import BaseModel

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer, override oldest if full."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)



def train_dqn(train_envs, val_envs=None, steps=100000, batch_size=64,
              gamma=0.99, lr=1e-3, capacity=100000, update_target=10000):

    device = config.get_device()

    #obs_dim = train_envs[0].observation_space.shape[0]
    #n_actions = train_envs[0].action_space.n

    q_net = BaseModel(num_actions=len(config.QActions), input_shape=(1, 60, 60)).to(device)
    target_net = BaseModel(num_actions=len(config.QActions), input_shape=(1, 60, 60)).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    eval_every = 5000
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    buffer = ReplayBuffer(capacity)
    alpha = 100
    epsilon_start, epsilon_final, epsilon_decay = 1.0, 0.01, 50000
    steps_done = 0
    warm_start_replay_buffer(buffer, train_envs, jump_boost=3)

    #warm_start_replay_buffer(buffer, val_envs)
    print(f"[WARM START] Filled {len(buffer)} transitions")
    best_model = copy.deepcopy(q_net)
    best_model_success_rate = copy.deepcopy(q_net)
    best_score = -float('inf')
    best_success_rate = -float('inf')
    q_net.train()
    target_net.eval()
    #torch.backends.cudnn.enabled = False

    for step in range(steps):

        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states = states.to(device)
        next_states = next_states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        actions = torch.where(actions == 3, torch.tensor(1, device=actions.device), actions)
        # Q(s,a)
        q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = target_net(next_states).max(1)[0]
            q_target = rewards + gamma * next_q * (1 - dones)
            q_target = torch.clamp(q_target, min=-20.0, max=200.0)

        loss = F.mse_loss(q_values, q_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if step % update_target == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step % 1000 == 0:
            print(f"[OFFLINE] step {step} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.6f}")
        if val_envs is not None and step % eval_every == 0:
            mean_return, success_rate = evaluate_offline_policy(q_net, val_envs, device)
            print(f"[EVAL] step {step} | Return: {mean_return:.2f} | Success: {success_rate:.2%}")
            score = mean_return + alpha * success_rate
            if score > best_score:
                best_score = score
                best_model = copy.deepcopy(q_net)
                print(f"🔹 New best reward model saved | score: {best_score:.2f}")
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_model_success_rate = copy.deepcopy(q_net)
                print(f"🔹 New best success rate model saved | success rate: {success_rate:.2f}")
    config.save_model(best_model, name="final_DQN")
    if best_success_rate > 0:
         config.save_model(best_model_success_rate, name=f"final_DQN_best_success_rate_{best_success_rate:.2f}")


def evaluate_offline_policy(q_net, eval_envs, device):
    q_net.eval()
    returns = []
    successes = []
    for env, actions in eval_envs:
        env = QEnvironment(config.ENV_SIZE, env, None)
        state = env.reset()
        done = False
        ep_return = 0

        while not done:
            with torch.no_grad():
                action_idx = q_net(
                    torch.FloatTensor(state).unsqueeze(0).to(device)
                ).argmax(1).item()
            action = 0 if action_idx == 0 else 3
            state, reward, done, success = env.step(action)
            ep_return += reward

        returns.append(ep_return)
        successes.append(done and env.current_position[0] == env.goal_position[0])

    mean_return = np.mean(returns)
    success_rate = np.mean(successes)  # fraction of successful episodes
    return mean_return, success_rate


def warm_start_replay_buffer(
    replay_buffer,
    environments_data,
    jump_boost=1
):
    """
    expert_trajectories: list of trajectories
    each trajectory = list of (s, a, r, s', done)
    """
    count = 0
    for curr_env, actions in environments_data:
        env = QEnvironment(config.ENV_SIZE, curr_env, None)
        state = env.reset()
        done = False
        total_reward = 0
        for i, action in enumerate(actions):
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            if actions[i] == 3:
                for _ in range(jump_boost):
                    replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            count += 1
    print(f"[WARM START] Filled {count} transitions")



