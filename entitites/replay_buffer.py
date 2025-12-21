import random
from collections import deque, Counter
import numpy as np
import matplotlib.pyplot as plt
import torch

import config
from data import dataset


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

    def visualize_content(self):
        action_counts = Counter([a for (_, a, _, _, _) in self.buffer])
        plt.bar(list(map(str, action_counts.keys())), action_counts.values())
        plt.title("Replay Buffer Action Distribution")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()


class ReplayBufferBCDataset(torch.utils.data.Dataset):
    def __init__(self, buffer, window_len=9):
        self.buffer = buffer
        self.window_len = window_len

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        state, action, _, _, _ = self.buffer.buffer[idx]

        window = extract_window_from_state(state, self.window_len)

        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long)
        )


def extract_window_from_state(state, window_len, jump_bias=True):
    """
    Extracts a 60 x window_len window centered on the agent.
    Falls back to jump-relevant positions if agent is missing.
    """
    state = state.squeeze()
    agent_pos = np.argwhere(state == config.AGENT)

    floor_height = dataset.get_env_floor_height(state)
    obstacle_mask = state == config.WHITE
    obstacle_mask[floor_height:, :] = False

    cols = np.where(obstacle_mask.any(axis=0))[0]
    if len(agent_pos) > 0:
        x, y = agent_pos[0]
    elif len(cols) == 0:
        x = 0
        y = floor_height + 1
        print("No obstacle found")
    else:
        rows, cols = state.shape
        row_idx = np.arange(rows)[:, None]
        col_idx = np.arange(cols)[None, :]

        # basic free-space mask
        free = state != config.WHITE

        if jump_bias:
            # airborne (jump-relevant)
            airborne = row_idx < floor_height

            # near obstacle columns
            obstacle_cols = cols
            near_obstacle = np.zeros_like(state, dtype=bool)

            for c in obstacle_cols:
                near_obstacle[:, max(0, c - 2):min(cols, c + 3)] = True

            mask = free & airborne & near_obstacle

            # fallback if mask too strict
            if mask.sum() < 5:
                mask = free & airborne

        else:
            mask = free

        free_cells = np.argwhere(mask)

        if len(free_cells) == 0:
            raise ValueError("No valid fallback position found")

        x, y = free_cells[np.random.randint(len(free_cells))]

    window = dataset.extract_env_windows(
        state[np.newaxis, ...],
        [(x, y)],
        window_len
    )[0]

    return window

