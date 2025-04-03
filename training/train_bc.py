import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
import numpy as np

def train(model, device, train_data, val_data, optimizer, criterion, early_stopping=5):
    np.random.seed(config.RANDOM_SEED)
    model.to(device)
    best_val_loss = float('inf')
    stop_counter = 0
    #train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    #val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_data, **config.PARAMS)
    val_loader = DataLoader(val_data, **config.PARAMS)

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        train_loss = 0

        for environments, actions in tqdm(train_loader, desc=f"Training Epoch: {epoch + 1}/{config.MAX_EPOCHS}"):
            environments = environments.to(device)
            actions = actions.to(device)
            # TODO now batch size is 10 meaning environments contains 10 60x60 envs and action 10 opt_paths with length 60
            # TODO reduce batch to value pairs for training eg a slice of env for example at position 15 and using action on pos 15 to let bc train on slices
            optimizer.zero_grad()
            output = model(environments)
            output = output.view(-1, len(config.Actions))
            cur_loss = criterion(output, actions)
            cur_loss.backward()
            optimizer.step()
            train_loss += cur_loss.item()
        val_loss = loss(model, device, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_counter = 0
            config.save_model(model, name=f"BC_{epoch + 1}")
        else:
            stop_counter += 1

        if stop_counter >= early_stopping:
            break
    config.save_model(model, name="final_BC")


def loss(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for environments, actions in val_loader:
            environments = environments.to(device)
            actions = actions.to(device)
            output = model(environments)
            val_loss += criterion(output, actions).item() * environments.size(0) / len(val_loader.dataset)

    return val_loss
