import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import config
from data import dataset
from data.dataloader import EnvironmentDataset
from entitites.replay_buffer import ReplayBuffer
#from experiments.experiments_setup import evaluate
import training.train_bc_new as train_bc_new
import entitites.DQNAgent
import entitites.BCQ as bcq
from environments.QEnvironment import QEnvironment
from models.bc_model import BehavioralModel
from models.bcq_model import BCQModel
from models.q_model import QModel
import models
import matplotlib.pyplot as plt


def load_model(name, model):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # experiments/
    project_root = os.path.abspath(os.path.join(base_dir, '..'))  # move to root
    path = f'./trained_models/{name}.pt'
    model_path = os.path.join(project_root, path)
    model = model
    model.load_state_dict(torch.load(model_path), strict=False)
    return model


def predict_actions_window_model(model, device, env_np):
    """
    Predicts one action per column using sliding-window input.
    Works for both BC
    """
    model.eval()
    env_np = env_np[0].astype(np.float32)

    expert_actions = env_np[1]
    actions = []
    expert_path = dataset.reconstruct_path(env_np, expert_actions)

    for (x, y) in expert_path:
        window = dataset.extract_env_windows(
            np.expand_dims(env_np, 0),
            [(x, y)],
            config.WINDOW_LEN
        )[0]

        # Fix accidental 60×61
        if window.shape[1] == 61:
            window = window[:, :60]

        inp = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        logits = model(inp)
        actions.append(int(logits.argmax().item()))

    return np.array(actions)

def predict_actions_unified(model, device, env_np):
    """
    Predict predicted actions along the expert trajectory for BOTH:
    - Window-based models (BC, BCQ trained on windows)
    - Full-state models (DQN)

    Automatically detects which input shape the model expects.
    """

    model.eval()

    # env_np expected shape: (60, 60)
    env = env_np[0]

    # --------- Detect Model Type (DQN vs BC-windows) ----------
    # Inspect first conv layer
    first_weight = next(model.parameters())
    expects_window = (first_weight.shape[-1] == config.WINDOW_LEN)
    # True  → BC/BCQ window model
    # False → DQN full-env model

    # --------- Reconstruct expert path ----------
    # env_np passed in may include (env, expert_actions) in a tuple form
    if isinstance(env_np, tuple) or isinstance(env_np, list):
        env_img = env_np[0]
        env_actions = env_np[1]
    else:
        raise ValueError("env_np must be (env, expert_actions)")

    expert_path = dataset.reconstruct_path(env_img, env_actions)

    predicted_actions = []

    # --------- BC / BCQ WINDOW-BASED MODEL ---------- # TODO delete or rework and check for BCQ
    if expects_window:
        for (x, y) in expert_path:
            window = dataset.extract_env_windows(
                np.expand_dims(env_img, 0),
                [(x, y)],
                config.WINDOW_LEN
            )[0]  # shape 60×W

            # Fix accidental 60×61
            if window.shape[1] == config.ENV_SIZE + 1:
                window = window[:, :config.ENV_SIZE]

            inp = torch.tensor(window, dtype=torch.float32, device=device)
            inp = inp.unsqueeze(0).unsqueeze(1)  # → (1, 1, 60, 5)

            logits = model(inp)
            action = int(logits.argmax().item())
            predicted_actions.append(action)

        return np.array(predicted_actions)

    # --------- DQN FULL-STATE MODEL ----------
    else:
        # Build a temporary QEnvironment for stepping
        start_pos = expert_path[0]
        curr_env = QEnvironment(
            environment=env_img,
            size=config.ENV_SIZE,
            start_pos=start_pos
        )
        curr_env.reset()

        for (x, y) in expert_path:
            state_tensor = torch.tensor(curr_env.state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = int(q_values.argmax().item())

            predicted_actions.append(action)

            next_state, reward, done = curr_env.step(action)
            if done:
                break

        return np.array(predicted_actions)


def evaluate(envs, model):
    successes, rewards, lengths, trajectories = [], [], [], []
    device = config.get_device()
    model.to(device)

    model.eval()
    actions = None
    for env in envs:
        if isinstance(model, BehavioralModel):
            actions = predict_actions_window_model(model, device, env)
        elif isinstance(model, QModel):
            actions = predict_actions_unified(model, device, env)
        obs = env.reset()
        done = False
        total_reward = 0
        length = 0
        traj = [env.current_position]
        for action in actions:
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            length += 1
            traj.append(env.current_position)
        rewards.append(total_reward)
        lengths.append(length)
        trajectories.append(traj)
        successes.append(env.agent_pos[0] >= config.ENV_SIZE - 1)
    #    while not done:
    #        if isinstance(model, BehavioralModel):
    #            pass
    #        else:
    #            state_batch = torch.tensor(obs, dtype=torch.float32, device=config.get_device()).unsqueeze(1)
    #            logits = model(state_batch)
    #            action = logits.argmax().item()
    #        obs, reward, done = env.step(action)
    #        total_reward += reward
    #        length += 1
    #        traj.append(env.agent_pos)
    #    successes.append(env.agent_pos[1] >= config.env_size - 1)
    #    rewards.append(total_reward)
    #    lengths.append(length)
    #    trajectories.append(traj)
    return successes, rewards, lengths, trajectories


def compare_bc_dqn_bcq(bc_model, dqn_model, bcq_model, envs, threshold, save_dir):
    for i, env in enumerate(envs):
        expert_path = dataset.reconstruct_path(env[0], env[1])

        bc_path  = evaluate(env[0], bc_model)
        dqn_path = evaluate(env[0], dqn_model)
        bcq_path = evaluate(env[0], bcq_model)

        # 3-panel plot
        fig, axs = plt.subplots(1, 3, figsize=(16, 6))
        panels = [
            ("BC", bc_path[3][0]), # TODO debug because with evaluate above i get list of list of trajectories length 1
            ("DQN", dqn_path[3][0]),
            ("BCQ", bcq_path[3][0]),
        ]

        for ax, (name, path) in zip(axs, panels):
            ax.imshow(env, cmap="gray")
            ax.plot([x for _, x in expert_path], [y for y, _ in expert_path], "-g", label="Expert")
            ax.plot([x for _, x in path],       [y for y, _ in path],       "-r", label=name)
            ax.set_title(name)
            ax.legend()

        plt.savefig(f"{save_dir}/compare_bc_dqn_bcq_env_{i}.png", dpi=200, bbox_inches='tight')
        plt.close()



def run_experiment_1(agents, train_data, val_data, test_data, buffer, train=True):
    """
    Baseline Performance
    """
    print("Running Experiment 1: Baseline Performance")

    test_loader = DataLoader(test_data, **config.PARAMS)
    # Initialize models
    bc_agent = agents[0]
    dqn_agent = agents[1]
    bcq_agent = agents[2]
    # Train
    if train:
        bc_loss = train_bc_new.train(bc_agent.model, config.get_device(), train_data, val_data, bcq_agent.optimizer) #epoch losses
        dqn_loss, q_values, action_counts = dqn_agent.train(train_data, val_data) # also average train loss
        bcq_losses = bcq.train_bcq(bcq_agent, buffer) # losses dict shape
    else:
        bc_agent.model = load_model("final_BC_state_dict", BehavioralModel())
        dqn_agent.model = load_model("final_Q_state_dict", QModel())
        bcq_agent.model = load_model("final_BCQ_state_dict", BCQModel())

    # Evaluate
    #print(loss(bc_agent.model, config.get_device(), DataLoader(val_data, **config.PARAMS), nn.CrossEntropyLoss()))
    small_test_data = Subset(test_data, list(range(10)))
    for name, model in zip(["BC", "DQN", "BCQ"], [bc_agent.model, dqn_agent.model, bcq_agent.model]):
        successes, rewards, lengths, trajectories = evaluate(small_test_data, model)
        print(
            f"{name} Success Rate: {np.mean(successes):.2f}, Avg Reward: {np.mean(rewards):.2f}, Avg Length: {np.mean(lengths):.2f}")