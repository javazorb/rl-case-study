from collections import Counter
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
import numpy as np
import copy
import data.dataset as dataset


def get_action_weights(train_data, device):
    """
    Compute class weights for CrossEntropyLoss based on action frequencies.
    Rare actions (e.g. jumps) get higher weight.
    """
    counts = Counter()
    for _, actions in DataLoader(train_data, **config.PARAMS):
        counts.update(actions.numpy().flatten().astype(int))

    total = sum(counts.values())
    num_actions = len(config.Actions)
    weights = torch.zeros(num_actions, dtype=torch.float)

    for action in range(num_actions):
        if counts[action] > 0:
            weights[action] = total / (num_actions * counts[action])
        else:
            weights[action] = 0.0

    # Adjustment for run_right bias (action 0)
    alpha = 0.5
    weights[0] = (1 + alpha * (weights[0] - 1)) / 1.5

    print("Action counts:", dict(counts))
    print("Action weights:", weights.tolist())
    return weights.to(device)


def collect_training_windows(environments, actions, expert_paths, num_samples=5, force_jump=True):
    """
    Collect multiple state windows from expert paths.
    Optionally force inclusion of windows where a jump occurs.
    """
    batch_states, batch_labels = [], []

    for i, expert_path in enumerate(expert_paths):
        # sample random start points along the expert path
        sampled_idxs = np.random.choice(len(expert_path) - config.WINDOW_LEN, num_samples, replace=False)

        for idx in sampled_idxs:
            x, y = expert_path[idx]
            window = dataset.extract_env_windows(environments[i:i+1], [(x, y)], config.WINDOW_LEN)[0]
            batch_states.append(window)
            batch_labels.append(actions[i][x])

        if force_jump:
            # look for any jump actions along this path
            jump_idxs = np.where(np.isin(actions[i], [config.Actions.JUMP.value, config.Actions.JUMP_RIGHT.value]))[0]
            if len(jump_idxs) > 0:
                j = np.random.choice(jump_idxs)
                x, y = expert_path[min(j, len(expert_path)-1)]
                window = dataset.extract_env_windows(environments[i:i+1], [(x, y)], config.WINDOW_LEN)[0]
                batch_states.append(window)
                batch_labels.append(actions[i][x])

    return np.stack(batch_states), np.array(batch_labels)


def train(model, device, train_data, val_data, optimizer, criterion=None, early_stopping=10):
    """
    Train a behavioral cloning model with weighted CrossEntropy loss.
    Uses vectorized batch sampling instead of per-step loops.
    """
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    model.to(device)

    best_val_loss = float("inf")
    stop_counter = 0
    best_model = copy.deepcopy(model)

    train_loader = DataLoader(train_data, **config.PARAMS)
    val_loader = DataLoader(val_data, **config.PARAMS)

    if criterion is None:
        weights = get_action_weights(train_data, device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for environments, actions in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config.MAX_EPOCHS}"):
            # reconstruct expert paths for this batch
            expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy())
                            for env, env_actions in zip(environments, actions)]

            # collect windows (vectorized, includes jump bias)
            states, labels = collect_training_windows(environments, actions, expert_paths,
                                                      num_samples=config.NUM_STEPS_ENV, force_jump=True)

            # prepare tensors
            state_batch = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(1)
            correct_actions = torch.tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            predicted_actions = model(state_batch)
            loss = criterion(predicted_actions, correct_actions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        val_loss = evaluate_loss(model, device, val_loader, criterion)

        print(f"[Epoch {epoch+1}] Train loss: {epoch_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            print(f"New best validation loss: {val_loss:.4f} (prev {best_val_loss:.4f})")
            best_val_loss = val_loss
            stop_counter = 0
            best_model = copy.deepcopy(model)
            config.save_model(model, name=f"BC_{epoch+1}")
        else:
            stop_counter += 1

        if stop_counter >= early_stopping:
            print(f"Early stopping after {epoch+1} epochs.")
            break

    config.save_model(best_model, name="final_BC")
    print("Training complete.")


def evaluate_loss(model, device, val_loader, criterion):
    """
    Compute validation loss using vectorized window collection.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for environments, actions in val_loader:
            expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy())
                            for env, env_actions in zip(environments, actions)]

            states, labels = collect_training_windows(environments, actions, expert_paths,
                                                      num_samples=config.NUM_STEPS_ENV, force_jump=True)

            state_batch = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(1)
            correct_actions = torch.tensor(labels, dtype=torch.long, device=device)

            predicted_actions = model(state_batch)
            val_loss += criterion(predicted_actions, correct_actions).item()

    return val_loss / len(val_loader)


def test_accuracy(model, device, test_data):
    """
    Evaluate accuracy and action distribution on test set.
    """
    model.to(device)
    model.eval()

    correct, total = 0, 0
    total_predicted = []
    num_jump_right = 0
    num_expert_jump_right = 0

    test_loader = DataLoader(test_data, **config.PARAMS)

    with torch.no_grad():
        for environments, actions in test_loader:
            expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy())
                            for env, env_actions in zip(environments, actions)]

            states, labels = collect_training_windows(environments, actions, expert_paths,
                                                      num_samples=config.NUM_STEPS_ENV, force_jump=True)
            num_expert_jump_right += (actions == config.Actions.JUMP_RIGHT.value).sum().item()
            state_batch = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(1)
            correct_actions = torch.tensor(labels, dtype=torch.long, device=device)

            predicted_actions = model(state_batch)
            predicted_classes = torch.argmax(predicted_actions, dim=1)

            num_jump_right += (correct_actions.cpu().numpy() == config.Actions.JUMP_RIGHT.value).sum()

            correct += (predicted_classes == correct_actions).sum().item()
            total += correct_actions.size(0)

            total_predicted.extend(predicted_classes.tolist())


    accuracy = correct / total
    return accuracy, total_predicted, num_jump_right, num_expert_jump_right
