from torch import nn
import torch.nn.functional as F
import config

class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self):
        pass