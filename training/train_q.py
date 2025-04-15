import random

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
import numpy as np
import data.dataset as dataset
from environments.QEnvironment import QEnvironment


def train(model, device, train_data, val_data, optimizer, criterion, early_stopping=10, epsilon=0.1):
    np.random.seed(config.RANDOM_SEED)
    #torch.manual_seed(config.RANDOM_SEED)
    model.to(device)

    stop_counter = 0
    train_loss = 0
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
            # TODO maybe save env reward, env and actions for further specialised training where performance was bad
            for env_id in range(config.BATCH_SIZE):
                curr_env = QEnvironment(environment=environments[env_id].numpy(), size=config.ENV_SIZE)
                curr_env.start_positions = agent_start_positions[env_id]
                env_reward = 0
                for step in range(config.NUM_STEPS_ENV):
                    action = epsilon_greedy_action(model, curr_env, epsilon)
                    next_state, reward, done, _ = curr_env.step(action)
                    if done:
                        break
                    env_reward += reward
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        next_max_Q = torch.max(model(next_state))
                        target_Q = reward + config.GAMMA * (torch.max(model(next_state)))
                curr_env.reset()

def epsilon_greedy_action(model, state, epsilon=0.1):
    if random.random() < epsilon:
        # Explore: choose a random action
        return torch.randint(0, len(config.Actions), (1,), dtype=torch.long)
    else:
        # Exploit: choose action with highest Q-value
        with torch.no_grad():
            actions = model(state)
            action = torch.argmax(actions, dim=1)
            return action


def loss(model, device, val_loader, criterion):
    pass