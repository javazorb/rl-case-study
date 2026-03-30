import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
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
from models.base_model import BaseModel
import models
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_model(name, model):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # experiments/
    project_root = os.path.abspath(os.path.join(base_dir, '..'))  # move to root
    path = f'./trained_models/{name}.pt'
    model_path = os.path.join(project_root, path)
    model = model
    model.load_state_dict(torch.load(model_path), strict=False)
    return model


def predict_actions_window_model(model, device, env_np, crop_size=60):
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
       if crop_size < env_np.shape[0]:
           env_np = crop_env(env_np, (x, y), crop_size)
           center = (crop_size // 2, crop_size // 2)
       else:
           center = (x, y)
       window = dataset.extract_env_windows(
           np.expand_dims(env_np, 0),
           [center],#[(x, y)],
           9#config.WINDOW_LEN
       )[0]

       # Fix accidental 60×61
       if window.shape[1] == 61:
           window = window[:, :60]

       inp = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
       logits = model(inp)
       actions.append(int(logits.argmax().item()))

   return np.array(actions)


def predict_actions_window_model_crop(model, device, env_np, crop_size=60):
    """
    BC sliding-window prediction with environment-level cropping
    (shape-preserving for the CNN)
    """
    model.eval()

    env_img, expert_actions = env_np
    env_img = env_img.astype(np.float32)

    expert_path = dataset.reconstruct_path(env_img, expert_actions)
    actions = []

    for (x, y) in expert_path:
        if crop_size < env_img.shape[0]:
            cropped_env = crop_env(env_img, (x, y), crop_size)
            center = (crop_size // 2, y)
        else:
            cropped_env = env_img
            center = (x, y)

        window = dataset.extract_env_windows(
            np.expand_dims(cropped_env, 0),
            [center],
            9
        )[0]  # MUST be (60, 9)

        # 🔒 HARD invariant
        #assert window.shape == (60, 9), window.shape

        inp = (
            torch.tensor(window, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(1)
        )
        current_height = cropped_env.shape[1]
        pad_top = (config.ENV_SIZE - current_height) // 2
        pad_bottom = config.ENV_SIZE - current_height - pad_top
        padded_env = F.pad(inp, (0, 0, pad_top, pad_bottom), mode='constant', value=0)
        logits = model(padded_env)
        #logits = model(inp)
        actions.append(int(logits.argmax().item()))

    return np.array(actions)



def predict_actions_unified(model, device, env_np, crop_size=60):
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
            state = curr_env.state
            if crop_size < state.shape[0]:
                state = crop_env(
                    state,
                    curr_env.current_position,
                    crop_size
                )

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = int(q_values.argmax().item())
            if action == 1:
                action = 3
            predicted_actions.append(action)
            next_state, reward, done, _ = curr_env.step(action)
            if done:
                break

        return np.array(predicted_actions)


def predict_actions_bcq(model, device, env, env_actions, crop_size=60):
    expert_path = dataset.reconstruct_path(env, env_actions)
    agent = bcq.DiscreteBCQAgent(model=model, num_actions=100, threshold=0.1)
    curr_env = QEnvironment(
        environment=env,
        size=config.ENV_SIZE,
        start_pos=expert_path[0]
    )
    agent.model.eval()
    model.eval()
    actions = []
    with torch.no_grad():
        for _ in range(config.MAX_STEPS):
            state = curr_env.state
            if crop_size < state.shape[0]:
                state = crop_env(
                    state,
                    curr_env.current_position,
                    crop_size
                )
            action = agent.select_action(state)
            obs, reward, done, _ = curr_env.step(action)
            actions.append(action)
            if done:
                break
    return np.array(actions)

def evaluate(envs, model, crop_size=60, crop=False):
    successes, rewards, lengths, trajectories = [], [], [], []
    device = config.get_device()
    model.to(device)

    model.eval()
    actions = None
    all_actions = []
    for env, env_actions in envs:
        if isinstance(model, BaseModel):
            if crop_size < env.shape[0]:
                actions = predict_actions_window_model_crop(model, device, (env, env_actions), crop_size=crop_size)
            else:
                actions = predict_actions_window_model(model, device, (env, env_actions), crop_size=crop_size)
        elif isinstance(model, QModel):
            actions = predict_actions_unified(model, device, (env, env_actions), crop_size=crop_size)
        else:
            actions = predict_actions_bcq(model, device, env, env_actions, crop_size=crop_size)
        all_actions.append(actions)
        expert_path = dataset.reconstruct_path(env, env_actions)
        curr_env = QEnvironment(
            environment=env,
            size=config.ENV_SIZE,
            start_pos=expert_path[0],
            crop=crop
        )
        obs = curr_env.reset()
        done = False
        total_reward = 0
        length = 0
        traj = [curr_env.current_position]
        for action in actions:
            obs, reward, done, _ = curr_env.step(action)
            total_reward += reward
            length += 1
            traj.append(curr_env.current_position)
            if done:
                break
        rewards.append(total_reward)
        lengths.append(length)
        trajectories.append(traj)
        successes.append(curr_env.current_position[0] >= config.ENV_SIZE - 1)
    return successes, rewards, lengths, trajectories, Counter(np.concatenate(all_actions))


def compare_bc_dqn_bcq(bc_model, dqn_model, bcq_model, bc_jumpy_model, envs, save_dir, max_envs=10, experiment_name="base_performance"):
    os.makedirs(save_dir, exist_ok=True)

    device = config.get_device()
    bc_model.to(device).eval()
    bc_jumpy_model.to(device).eval()
    dqn_model.to(device).eval()
    bcq_model.to(device).eval()

    for i in range(min(max_envs, len(envs))):
        env, env_actions = envs[i]  # IMPORTANT
        expert_path = dataset.reconstruct_path(env, env_actions)

        # Wrap single env so evaluate() works
        single_env = [(env, env_actions)]

        bc_traj  = evaluate(single_env, bc_model)[3][0]
        bc_jumpy_traj = evaluate(single_env, bc_jumpy_model)[3][0]
        dqn_traj = evaluate(single_env, dqn_model)[3][0]
        bcq_traj = evaluate(single_env, bcq_model)[3][0]

        fig, axs = plt.subplots(1, 4, figsize=(18, 6))

        panels = [
            ("BC", bc_traj),
            ("BC_Oversampled_Jumps", bc_jumpy_traj),
            ("DQN", dqn_traj),
            ("BCQ", bcq_traj),
        ]

        for ax, (name, path) in zip(axs, panels):
            ax.imshow(env, cmap="gray", origin="lower")
            H = env.shape[0]
            # Expert path (green)
            #test1 = [y for y, x in expert_path][::-1]
            #test2 = [x for y, x in expert_path][::-1]
            ax.plot(
                [y for y, x in expert_path][::-1],
                [x for y, x in expert_path][::-1],
                "-g", label="Expert"
            )

            # Agent path (red)
            ax.plot(
                [y for y, x in path][::-1],
                [x for y, x in path][::-1],
                "-r", label=name
            )


            ax.set_title(name)
            ax.legend()
            ax.axis("off")

        plt.suptitle(f"Environment {i}", fontsize=14)
        plt.savefig(f"{save_dir}/{experiment_name}_compare_env_{i}.png", dpi=200, bbox_inches="tight")
        plt.close()



def plot_reward_length_scatter(rewards, lengths, label, save_path):
    plt.figure(figsize=(5, 4))
    plt.scatter(lengths, rewards, alpha=0.5)
    plt.xlabel("Trajectory length")
    plt.ylabel("Total reward")
    plt.title(label)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_length_hist(lengths, label, save_path):
    plt.figure(figsize=(5, 4))
    plt.hist(lengths, bins=20)
    plt.xlabel("Trajectory length")
    plt.ylabel("Count")
    plt.title(label)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_action_distribution(action_dist, title, save_path):
    # Normalize np.int64 → int
    action_dist = {int(k): int(v) for k, v in action_dist.items()}

    all_actions = list(range(len(config.QActions)))
    all_actions[1] = 3
    counts = [action_dist.get(a, 0) for a in all_actions]

    plt.figure(figsize=(5, 4))
    plt.bar(all_actions, counts)
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(all_actions)
    plt.savefig(save_path, dpi=200)
    plt.close()




def plot_success_rates(results, labels, save_path):
    plt.figure(figsize=(8, 4))
    offsets = {
        "BC": 0.0,
        "BC_Oversampled_Jumps": 0.0,
        "DQN": -0.02,
        "BCQ": 0.02,
    }

    for successes, label in zip(results, labels):
        y = np.array(successes, dtype=float) + offsets[label]
        plt.plot(y, label=label, marker="o", markersize=2)

    plt.xlabel("Environment index")
    plt.ylabel("Success (0/1)")
    plt.legend()
    plt.grid()
    plt.savefig(save_path, dpi=200)
    plt.close()


#def crop_env(env, center, crop_size, pad_val=0):
#    h, w = env.shape
#    half = crop_size // 2
#    cx, cy = center
#    cropped = np.full((crop_size, crop_size), pad_val, dtype=env.dtype)
#
#    for i in range(crop_size):
#        for j in range(crop_size):
#            x = cx - half + i
#            y = cy - half + j
#            if 0 <= x < h and 0 <= y < w:
#                cropped[i, j] = env[x, y]
#
#    return cropped
def crop_env(env, agent_pos, crop_size, pad_val=0):

    H, W = env.shape
    x, y = agent_pos
    half = crop_size // 2

    top = x - half
    bottom = x + half
    left = y - half
    right = y + half

    cropped = np.full((crop_size, crop_size), pad_val, dtype=env.dtype)

    for i in range(crop_size):
        for j in range(crop_size):

            src_x = top + i
            src_y = left + j

            if 0 <= src_x < H and 0 <= src_y < W:
                cropped[i, j] = env[src_x, src_y]

    return cropped


def crop_env_horizontally(env, start_x, goal_x):

    cropped_env = env[:, start_x:goal_x]

    return cropped_env


def plot_success_vs_crop(crop_sizes, success_rates, label, save_path):
    plt.figure(figsize=(5, 4))
    plt.plot(crop_sizes, success_rates, marker="o")
    plt.xlabel("Crop size")
    plt.ylabel("Success rate")
    plt.title(label)
    plt.grid()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_crop(env, agent_pos, crop_size, save_path):
    H, W = env.shape
    x, y = agent_pos
    half = crop_size // 2

    top = max(0, x - half)
    bottom = min(H, x + half)

    fig, ax = plt.subplots()

    ax.imshow(env, cmap="gray", origin="lower")

    rect = patches.Rectangle(
        (0, top),           # x,y
        W,                  # width
        bottom - top,       # height
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )

    ax.add_patch(rect)

    ax.scatter([y], [x], c="blue", label="Agent")

    ax.legend()
    ax.set_title(f"Crop size = {crop_size}")

    plt.savefig(save_path, dpi=200)
    plt.close()



def run_experiment_1(agents, train_data, val_data, test_data, buffer, train=True, experiment_name="base_performance"):
    """
    Baseline Performance
    """
    print(f"Running Experiment : {experiment_name}")

    test_loader = DataLoader(test_data, **config.PARAMS)
    # Initialize models
    bc_agent = agents[0]
    dqn_agent = agents[1]
    bcq_agent = agents[2]
    bc_agent_jumpy = agents[3]
    # Train
    if train:
        bc_loss = train_bc_new.train(bc_agent.model, config.get_device(), train_data, val_data, bcq_agent.optimizer) #epoch losses
        dqn_loss, q_values, action_counts = dqn_agent.train(train_data, val_data) # also average train loss
        bcq_losses = bcq.train_bcq(bcq_agent, buffer) # losses dict shape
    else:
        bc_agent.model = load_model("final_BC_state_dict", BaseModel())
        bc_agent_jumpy.model = load_model("final_BC_jumpy_state_dict", BaseModel())
        dqn_model = QModel(num_actions=len(config.QActions), input_shape=(1, 60, 60))
        dqn_agent.model = load_model("final_DQN_state_dict", dqn_model)
        bcq_agent.model = load_model("final_BCQ_state_dict", BCQModel())

    # Evaluate
    #print(loss(bc_agent.model, config.get_device(), DataLoader(val_data, **config.PARAMS), nn.CrossEntropyLoss()))
    #small_test_data = Subset(test_data, list(range(10)))
    if "cropped_environments" in experiment_name:
        crop_type = str.split(experiment_name, "_")[-1]

    for name, model in zip(["BC", "DQN", "BCQ", "BC_Oversampled_Jumps"], [bc_agent.model, dqn_agent.model, bcq_agent.model, bc_agent_jumpy.model]):
        success_rates = []

        if "cropped_environments" in experiment_name:
            successes, rewards, lengths, trajectories, action_dist = evaluate(test_data, model, crop=True)
            success_rates.append(np.mean(successes))
            print(
                f"{name} | crop={crop_type} | "
                f"Success: {np.mean(successes):.2f}, "
                f"Avg Reward: {np.mean(rewards):.2f}, "
                f"Avg Length: {np.mean(lengths):.2f}"
            )
        else:
            successes, rewards, lengths, trajectories, action_dist = evaluate(test_data, model)
        print(
            f"{name} Success Rate: {np.mean(successes):.2f}, Avg Reward: {np.mean(rewards):.2f}, Avg Length: {np.mean(lengths):.2f}, Action Counts: {action_dist}")
        plot_action_distribution(
            action_dist,
            f"{name} Action Distribution",
            f"plots/{experiment_name + name}_actions.png"
        )
        plot_length_hist(lengths, f"{name} Length Distribution", f"plots/{name}_lengths.png")
        plot_reward_length_scatter(rewards, lengths, f"{name} Reward Length Distribution", f"plots/{experiment_name + name}_reward_lengths.png")
        if experiment_name == "cropped_environments":
            plot_success_vs_crop(crop_type, success_rates, f"{name} Success Rates", f"plots/{experiment_name + name}.png")
    print("========================================== Compare Models ==========================================")
    if experiment_name == "cropped_environments":

        os.makedirs("plots/crop_debug", exist_ok=True)

        # visualize first few environments
        for i in range(min(5, len(test_data))):

            env, env_actions = test_data[i]
            expert_path = dataset.reconstruct_path(env, env_actions)

            # use first expert position as agent location
            agent_pos = expert_path[0]

            for crop_size in crop_sizes:
                plot_crop(
                    env,
                    agent_pos,
                    crop_size,
                    f"plots/crop_debug/env{i}_crop{crop_size}.png"
                )

    compare_bc_dqn_bcq(
        bc_agent.model,
        dqn_agent.model,
        bcq_agent.model,
        bc_agent_jumpy.model,
        test_data,
        save_dir="plots/path_comparisons",
        max_envs=5,
        experiment_name=experiment_name
    )
    print("========================================== Plot Success Rates ==========================================")
    #print(set(evaluate(test_data, dqn_agent.model)[0]))
    plot_success_rates(
        [
            evaluate(test_data, bc_agent.model)[0],
            evaluate(test_data, bc_agent_jumpy.model)[0],
            evaluate(test_data, dqn_agent.model)[0],
            evaluate(test_data, bcq_agent.model)[0],
        ],
        ["BC", "BC_Oversampled_Jumps", "DQN", "BCQ"],
        f"plots/{experiment_name}_success_per_env.png"
    )


