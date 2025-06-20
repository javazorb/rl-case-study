import random
from collections import deque, Counter
import numpy as np
import matplotlib.pyplot as plt


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