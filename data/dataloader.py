from torch.utils.data import Dataset, DataLoader
import numpy as np


class EnvironmentDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = (
            data[0],
            np.array(data[1], dtype=np.int64),
        )  # tuple list length 2 env and opt_path
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        env = self.data[0][idx]
        path = self.data[1][idx]
        return env, path
