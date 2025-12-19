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
    weights[-1] = weights[-1] * 1.25
    print("Action counts:", dict(counts))
    print("Action weights:", weights.tolist())
    return weights.to(device)


def collect_training_windows(environments, actions, expert_paths, num_samples):
    """
    Collect (state_window, action_t) pairs with exact time alignment.
    """
    batch_states = []
    batch_labels = []

    for i, path in enumerate(expert_paths):
        T = len(path)

        # valid timesteps for a window
        valid_ts = np.arange(0, T - config.WINDOW_LEN)

        # sample timesteps
        sampled_ts = np.random.choice(
            valid_ts,
            size=min(num_samples, len(valid_ts)),
            replace=False
        )

        for t in sampled_ts:
            x, y = path[t]

            # extract window centered at (x, y) at time t
            window = dataset.extract_env_windows(
                environments[i:i+1],
                [(x, y)],
                9
            )[0]

            # label = action at SAME timestep t
            action_t = actions[i][t]

            batch_states.append(window)
            batch_labels.append(action_t)

    return np.stack(batch_states), np.array(batch_labels)


def collect_training_windows_old(environments, actions, expert_paths, num_samples=5, force_jump=True):
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
                picked_idx = np.random.choice(jump_idxs, int(len(jump_idxs) / 1.5), replace=False)
                for idx in range(len(picked_idx)):#range(int(len(jump_idxs) / 2)):
                    x, y = expert_path[min(idx, len(expert_path)-1)]
                    window = dataset.extract_env_windows(environments[i:i + 1], [(x, y)], config.WINDOW_LEN)[0]
                    batch_states.append(window)
                    batch_labels.append(actions[i][x])
                #j = np.random.choice(jump_idxs)
                #x, y = expert_path[min(j, len(expert_path)-1)]
                #window = dataset.extract_env_windows(environments[i:i+1], [(x, y)], config.WINDOW_LEN)[0]
                #batch_states.append(window)
                #batch_labels.append(actions[i][x])

    return np.stack(batch_states), np.array(batch_labels)


def collect_training_windows_stacked(environments, actions, expert_paths, num_samples=5, window_len=5, T=3, force_jump=True):
    """
    Collect stacked state windows for BC with temporal context.
    Returns states of shape (batch_size, T, 60, window_len)
    """
    batch_states, batch_labels = [], []

    for i, expert_path in enumerate(expert_paths):
        max_idx = len(expert_path) - window_len - (T-1)
        if max_idx <= 0:
            continue

        sampled_idxs = np.random.choice(max_idx, num_samples, replace=False)

        for idx in sampled_idxs:
            # stack T consecutive windows
            stacked = []
            for t in range(T):
                x, y = expert_path[idx + t]
                window = dataset.extract_env_windows(environments[i:i+1], [(x, y)], window_len)[0]
                stacked.append(window)
            stacked = np.stack(stacked, axis=0)  # shape (T, 60, window_len)
            batch_states.append(stacked)
            batch_labels.append(actions[i][idx + T - 1])

        if force_jump:
            jump_idxs = np.where(np.isin(actions[i], [config.Actions.JUMP.value, config.Actions.JUMP_RIGHT.value]))[0]
            if len(jump_idxs) > 0:
                picked_idx = np.random.choice(jump_idxs, max(1, int(len(jump_idxs)/1.5)), replace=False)
                for j_idx in picked_idx:
                    start_idx = max(0, j_idx - (T-1))
                    stacked = []
                    for t in range(T):
                        x, y = expert_path[start_idx + t]
                        window = dataset.extract_env_windows(environments[i:i+1], [(x, y)], window_len)[0]
                        stacked.append(window)
                    stacked = np.stack(stacked, axis=0)
                    batch_states.append(stacked)
                    batch_labels.append(actions[i][start_idx + T - 1])

    return np.stack(batch_states), np.array(batch_labels)



def train(model, device, train_data, val_data, optimizer, criterion=None, early_stopping=20):
    """
    Train a behavioral cloning model with weighted CrossEntropy loss.
    Uses vectorized batch sampling instead of per-step loops.
    """
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    model.to(device)
    losses = []
    best_val_loss = float("inf")
    stop_counter = 0
    best_model = copy.deepcopy(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=5e-6
    )

    train_loader = DataLoader(train_data, **config.PARAMS)
    val_loader = DataLoader(val_data, **config.PARAMS)

    if criterion is None:
        weights = get_action_weights(train_data, device)
        #criterion = torch.nn.CrossEntropyLoss(weight=weights)
        criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for environments, actions in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config.MAX_EPOCHS}"):
            # reconstruct expert paths for this batch
            expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy())
                            for env, env_actions in zip(environments, actions)]

            # collect windows (vectorized, includes jump bias)
            states, labels = collect_training_windows(environments, actions, expert_paths,
                                                      num_samples=4 * 9)

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
        losses.append(epoch_loss)
        val_loss = evaluate_loss(model, device, val_loader, criterion)
        scheduler.step(val_loss)
        print(f"[Epoch {epoch+1}] Train loss: {epoch_loss:.4f} | Val loss: {val_loss:.4f}")
        acc, actions = test_bc_accuracy(model, device, val_data)
        print(f"[Epoch {epoch+1}] Test accuracy: {acc:.4f}")
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
    return np.array(losses)


def evaluate_loss(model, device, val_loader, criterion, window_len=9, T=3):
    """
    Compute validation loss using stacked temporal windows.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for environments, actions in val_loader:
            expert_paths = [dataset.reconstruct_path(env.numpy(), env_actions.numpy())
                            for env, env_actions in zip(environments, actions)]

            # use the stacked window collection
            states, labels = collect_training_windows(environments, actions, expert_paths, 9*4)

            # tensor with shape (batch, T, 60, window_len)
            state_batch = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(1)
            correct_actions = torch.tensor(labels, dtype=torch.long, device=device)

            predicted_actions = model(state_batch)
            val_loss += criterion(predicted_actions, correct_actions).item()

    return val_loss / len(val_loader)



def test_accuracy_extensive(model, device, test_data):
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
                                                      num_samples=9*4)
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


def test_bc_accuracy(model, device, test_data):
    model.eval()
    model.to(device)

    correct, total = 0, 0
    pred_actions = []
    max_probs = []
    loader = DataLoader(test_data, **config.PARAMS)

    with torch.no_grad():
        for envs, acts in loader:
            expert_paths = [
                dataset.reconstruct_path(env.numpy(), act.numpy())
                for env, act in zip(envs, acts)
            ]

            states, labels = collect_training_windows(envs, acts, expert_paths, 9*4)

            states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(1)
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(states)
            probs = torch.softmax(logits, dim=1)
            max_probs.append(probs.max(dim=1).values.cpu())
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pred_actions.extend(preds.cpu().numpy())
    mean_max_prob = torch.cat(max_probs).mean().item()
    print(f"Mean max_prob: {mean_max_prob}")

    return correct / total, pred_actions

