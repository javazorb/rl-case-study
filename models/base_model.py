import torch
from torch import nn
import torch.nn.functional as F
import config

class BaseModel(nn.Module):
    def __init__(self, num_actions=len(config.Actions), input_shape=(1, 60, 9)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 60, 9)
            n_flat = self.conv(dummy).view(1, -1).size(1)

        print(f"n_flat feature dim: {n_flat}")

        self.fc = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)