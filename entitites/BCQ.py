import torch
import config

class BCQ(torch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass

    def train(self, train_set, val_set, device, optimizer, criterion):
        pass
