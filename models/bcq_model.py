import torch
from torch import nn
import torch.nn.functional as F
import config
from models.base_model import BaseModel


class BCQModel(nn.Module):
    def __init__(self, input_shape=(1, 60, 60), num_actions=len(config.QActions), *args, **kwargs):
        super().__init__()
        self.backbone = BaseModel(input_shape=input_shape, num_actions=num_actions)

        # IMPORTANT: get feature size from conv only
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            n_flat = self.backbone.conv(dummy).view(1, -1).size(1)

        self.q_head = nn.Sequential(
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self.i_head = nn.Sequential(
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
    #    self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)  # → (32, 30, 30)
    #    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # → (64, 15, 15)
    #    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # → (64, 15, 15)
    #    self.relu = nn.ReLU()

        #self.flat_dim = 64 * 15 * 15  # = 14400

        #self.fc1 = nn.Linear(self.flat_dim, 128)
        #self.fc2 = nn.Linear(128, len(config.Actions))

        # need imitation and q remodel
    #    self.q1 = nn.Linear(14400, 128)
    #    self.q2 = nn.Linear(128, len(config.QActions))
    #    self.i1 = nn.Linear(14400, 128)
    #    self.i2 = nn.Linear(128, len(config.QActions))

    def forward(self, x):
        features = self.backbone.conv(x)
        features = features.view(features.size(0), -1)

        q = self.q_head(features)
        logits = self.i_head(features)
        return q, F.log_softmax(logits, dim=1), logits
     #   x = self.relu(self.conv1(x))
     #   x = self.relu(self.conv2(x))
     #   x = self.relu(self.conv3(x))
        #x = x.view(x.size(0), -1)
        #x = self.relu(self.fc1(x))
        #return self.fc2(x)

        #q = F.relu(self.q1(x.reshape(-1, 14400)))
        #i = F.relu(self.i1(x.reshape(-1, 14400)))
     #   x = x.view(x.size(0), -1)
     #   q = F.relu(self.q1(x))
     #   i = F.relu(self.i1(x))
     #   i = self.i2(i)
     #   return self.q2(q), F.log_softmax(i, dim=1), i
