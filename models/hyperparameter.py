import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from training import train_bc


def search_hyperparameters(model, learning_rates, batch_sizes, optimizers, train_set, val_set):
    # Try different combinations of learning rate, batch size, optimizer
    best_val_loss = float('inf')
    best_params = None
    val_loader = DataLoader(val_set, **config.PARAMS)

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for optimizer_func in optimizers:
                # Initialize model, optimizer, and loss function
                optimizer = optimizer_func(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                # Train the model with the current configuration
                train_loss = train_bc.train(model, config.get_device(), train_set, val_set, optimizer, criterion)

                # Evaluate on validation set
                val_loss = train_bc.loss(model, config.get_device(), val_loader, criterion)

                # Check if this configuration gives the best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = (lr, batch_size, optimizer_func.__name__)

    print(f'Best Parameters: {best_params}, Validation Loss: {best_val_loss}')
    return best_params
