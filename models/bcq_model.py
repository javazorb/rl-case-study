from torch import nn
import torch.nn.functional as F
import config

class QModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)  # → (32, 30, 30)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # → (64, 15, 15)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # → (64, 15, 15)
        self.relu = nn.ReLU()

        #self.flat_dim = 64 * 15 * 15  # = 14400

        #self.fc1 = nn.Linear(self.flat_dim, 128)
        #self.fc2 = nn.Linear(128, len(config.Actions))

        # need imitation and q remodel
        self.q1 = nn.Linear(14400, 128)
        self.q2 = nn.Linear(128, len(config.QActions))
        self.i1 = nn.Linear(14400, 128)
        self.i2 = nn.Linear(128, len(config.QActions))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        #x = x.view(x.size(0), -1)
        #x = self.relu(self.fc1(x))
        #return self.fc2(x)
        q = F.relu(self.q1(x.reshape(-1, 14400)))
        i = F.relu(self.i1(x.reshape(-1, 14400)))
        i = self.i2(i)
        return self.q2(q), F.log_softmax(i, dim=1), i
