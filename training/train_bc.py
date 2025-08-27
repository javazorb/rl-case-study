from collections import Counter

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
import numpy as np
import copy
import data.dataset as dataset


def get_action_weights(train_data, device):
    counts = Counter()
    for _, actions in DataLoader(train_data, **config.PARAMS):
        for act in actions.numpy().flatten():
            counts[int(act)] += 1

    total = sum(counts.values())
    num_actions = len(config.Actions)
    weights = torch.zeros(num_actions, dtype=torch.float)
    for action in range(num_actions):
        if counts[action] > 0:
            weights[action] = total / (num_actions * counts[action])
        else:
            weights[action] = 0# inverse frequency


    alpha = 0.5
    weights[0] = (1 + alpha * (weights[0] - 1)) / 1.5
    print("Action counts:", dict(counts))
    print("Action weights:", weights.tolist())
    return weights.to(device)

def train(model, device, train_data, val_data, optimizer, criterion, early_stopping=10):
    np.random.seed(config.RANDOM_SEED) # TODO remodel to only jump action because jumping environment moves every state update by once to the rright
    model.to(device)
    best_val_loss = float('inf')
    stop_counter = 0
    best_model = model

    train_loader = DataLoader(train_data, **config.PARAMS)
    val_loader = DataLoader(val_data, **config.PARAMS)
    train_loss = 0
    epochs_ran = 0

    weights = get_action_weights(train_data, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        batch_loss = 0
        epoch_loss = 0
        for environments, actions in tqdm(train_loader, desc=f"Training Epoch: {epoch + 1}/{config.MAX_EPOCHS}"):
            expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy()) for env, env_actions in
                            zip(environments, actions)]
            agent_start_positions = []

            for expert_path in expert_paths:
                start_idx = np.random.randint(config.ENV_SIZE - config.NUM_STEPS_ENV)
                agent_start_positions.append(expert_path[start_idx])
            for step_idx in range(config.NUM_STEPS_ENV):
                state_batch = dataset.extract_env_windows(environments, agent_start_positions, config.WINDOW_LEN)
                state_batch = np.asarray(state_batch, dtype=np.int64)
                state_batch = torch.from_numpy(state_batch).float().to(device) # shape [1, 10, 60, 5]
                state_batch = state_batch.unsqueeze(1) # corrected shape [10, 1, 60, 5]

                optimizer.zero_grad()
                predicted_actions = model(state_batch.to(device))
                action_idxs = [x for x,y in agent_start_positions]
                correct_actions = [actions[i][action_idxs[i]] for i in range(len(action_idxs))]
                cur_loss = criterion(predicted_actions, torch.LongTensor(correct_actions).to(device))
                cur_loss.backward()
                optimizer.step()
                epoch_loss = cur_loss.item()
                agent_start_positions = dataset.update_agent_pos(agent_start_positions,
                                                                 expert_paths)  # updated along the expert path

        epoch_loss /= len(train_loader)
        val_loss = loss(model, device, val_loader, criterion)
        print(f"[Epoch {epoch+1}] Train loss: {epoch_loss:.4f} | Val loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            print(f'New best validation loss: {val_loss}\n old best validation loss: {best_val_loss}')
            best_val_loss = val_loss
            stop_counter = 0
            best_model = copy.deepcopy(model)
            config.save_model(model, name=f"BC_{epoch + 1}")
        else:
            stop_counter += 1

        if stop_counter >= early_stopping:
            epochs_ran = epoch + 1
            break
    config.save_model(best_model, name="final_BC")
    print(f'Final training loss: {train_loss/epochs_ran:.4f} after {epochs_ran} epochs')


def loss(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for environments, actions in val_loader:
            batch_loss = 0
            expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy()) for env, env_actions in
                            zip(environments, actions)]
            agent_start_positions = []

            for expert_path in expert_paths:
                start_idx = np.random.randint(config.ENV_SIZE - config.NUM_STEPS_ENV)
                agent_start_positions.append(expert_path[start_idx])

            for step_idx in range(config.NUM_STEPS_ENV):
                state_batch = dataset.extract_env_windows(environments, agent_start_positions, config.WINDOW_LEN)
                state_batch = np.asarray(state_batch, dtype=np.int64)
                state_batch = torch.from_numpy(state_batch).float().to(device)
                state_batch = state_batch.unsqueeze(1)  # Ensure the correct shape [batch_size, 1, 60, 5]

                predicted_actions = model(state_batch.to(device))

                action_idxs = [x for x, y in agent_start_positions]
                correct_actions = [actions[i][action_idxs[i]] for i in range(len(action_idxs))]

                # Compute the loss for this batch
                batch_loss += criterion(predicted_actions, torch.LongTensor(correct_actions).to(device)).item()
                agent_start_positions = dataset.update_agent_pos(agent_start_positions, expert_paths)
            batch_loss /= config.NUM_STEPS_ENV
        val_loss += batch_loss
    val_loss /= len(val_loader)
    return val_loss


def test_accuracy(model, device, test_data):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    total_predicted = []
    num_jump_right = 0
    test_loader = DataLoader(test_data, **config.PARAMS)
    with torch.no_grad():  # No need to track gradients during inference
        for environments, actions in test_loader:
            expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy()) for env, env_actions in
                            zip(environments, actions)]
            agent_start_positions = []

            for expert_path in expert_paths:
                start_idx = np.random.randint(config.ENV_SIZE - config.NUM_STEPS_ENV)
                agent_start_positions.append(expert_path[start_idx])
            for step_idx in range(config.NUM_STEPS_ENV):
                # Prepare the test batch
                state_batch = dataset.extract_env_windows(environments, agent_start_positions, config.WINDOW_LEN)
                state_batch = np.asarray(state_batch, dtype=np.int64)
                state_batch = torch.from_numpy(state_batch).float().to(device)
                state_batch = state_batch.unsqueeze(1)  # Ensure the correct shape [batch_size, 1, 60, 5]

                # Get the predicted actions from the model
                predicted_actions = model(state_batch)

                # Assuming predicted_actions is a tensor of probabilities, apply argmax to get the predicted class
                predicted_classes = torch.argmax(predicted_actions, dim=1)
                action_idxs = [x for x, y in agent_start_positions]
                correct_actions = [actions[i][action_idxs[i]] for i in range(len(action_idxs))]
                num_jump_right += correct_actions.count(config.Actions.JUMP_RIGHT.value)

                # Convert actions to a tensor on the same device
                correct_actions = torch.LongTensor(correct_actions).to(device)
                agent_start_positions = dataset.update_agent_pos(agent_start_positions, expert_paths)
                # Calculate how many predictions are correct
                correct += (predicted_classes == correct_actions).sum().item()
                total_predicted.append(predicted_classes.tolist())

                total += correct_actions.size(0)

    accuracy = correct / total  # Calculate accuracy
    return accuracy, total_predicted, num_jump_right
