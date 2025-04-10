import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
import numpy as np
import data.dataset as dataset


def train(model, device, train_data, val_data, optimizer, criterion, early_stopping=10):
    pass

def loss(model, device, val_loader, criterion):
    pass