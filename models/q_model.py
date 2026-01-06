import torch
from torch import nn
import config

class QModel(nn.Module):
    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
    #    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
    #    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    #    self.fc1 = nn.Linear(64 * 60 * 60, out_features=128)
    #    self.fc2 = nn.Linear(in_features=128, out_features=len(config.QActions))
    #    self.relu = nn.ReLU()

    #def forward(self, x):
    #    x = self.relu(self.conv1(x))
    #    x = self.relu(self.conv2(x))
    #    x = x.view(x.size(0), -1)
    #    x = self.relu(self.fc1(x))
    #    x = self.fc2(x)
    #    return x
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
            dummy = torch.zeros(1, *input_shape)#torch.zeros(1, 1, 60, 9)
            self.n_flat = self.conv(dummy).view(1, -1).size(1)

        print(f"n_flat feature dim: {self.n_flat}")

        self.fc = nn.Sequential(
            nn.Linear(self.n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)