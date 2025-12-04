import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

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


def collect_all_windows_for_environment(env, actions, expert_path):
    """
    Collects ALL state windows (no sampling) for a single environment.
    Returns:
        states: [T, WINDOW_LEN, H, W]
        labels: [T] (expert action at each timestep)
    """

    states = []
    labels = []

    T = len(expert_path)

    for t in range(T - config.WINDOW_LEN):
        x, y = expert_path[t]

        # env must be wrapped in a batch dimension [1, H, W]
        window = dataset.extract_env_windows(env[None, ...], [(x, y)], config.WINDOW_LEN)[0]

        states.append(window)
        labels.append(actions[x])  # expert action at this time

    return np.stack(states), np.array(labels)

def load_model(name, model):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # experiments/
    project_root = os.path.abspath(os.path.join(base_dir, '..'))  # move to root
    path = f'./trained_models/{name}.pt'
    model_path = os.path.join(project_root, path)
    model = model
    model.load_state_dict(torch.load(model_path), strict=False)
    return model


def get_actions_bc(model, device, test_data):
    """
    Evaluate accuracy and action distribution on test set.
    """
    model.to(device)
    model.eval()
    total_predicted = []
    actions = test_data[1]
    expert_path = dataset.reconstruct_path(test_data[0], test_data[1])

    test_data = EnvironmentDataset(test_data)
    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 0}
    test_loader = DataLoader(test_data, batch_size=1)
    with torch.no_grad():
        for environments, _ in test_loader:
            states, labels = collect_all_windows_for_environment(environments, actions, expert_path)
            state_batch = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(1)
            actions = model(state_batch)
            print(actions.shape)

            #states, labels = train_bc_new.collect_training_windows(environments, actions, expert_paths,
            #                                                       num_samples=config.NUM_STEPS_ENV, force_jump=True)
            #state_batch = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(1)

            #predicted_actions = model(state_batch)
            #predicted_classes = torch.argmax(predicted_actions, dim=1)


            #total_predicted.extend(predicted_classes.tolist())

    return total_predicted

def get_all_bc_actions_for_batch(model, device, val_loader, criterion):
    model.to(device)
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
                agent_start_positions.append(expert_path[0])

            for step_idx in range(config.NUM_STEPS_ENV): # TODO do that for all windows, after the first 5 set start position plus 5 and repeat until you get 60 actions
                state_batch = dataset.extract_env_windows(environments, agent_start_positions, config.WINDOW_LEN)
                #state_batch = [arr[:, :-1] for arr in state_batch]
                state_batch = np.asarray(state_batch, dtype=np.int64)
                state_batch = torch.from_numpy(state_batch).float().to(device)
                state_batch = state_batch.unsqueeze(1)  # Ensure the correct shape [batch_size, 1, 60, 5]

                predicted_actions = model(state_batch.to(device))
                predicted_actions = torch.argmax(predicted_actions, dim=1).cpu().numpy()


    return val_loss



def evaluate(envs, model):
    successes, rewards, lengths, trajectories = [], [], [], []
    model.to(config.get_device())
    model.eval()
    if isinstance(model, BehavioralModel):
        actions_bc = get_actions_bc(model, config.get_device(), envs[0])
    for env in envs[0]:
        env = QEnvironment(size=config.ENV_SIZE, environment=env,
                                start_pos=None)
        obs = env.reset()
        done = False
        total_reward = 0
        length = 0
        traj = [env.current_position]
        while not done:
            if isinstance(model, BehavioralModel):
                pass
            else:
                state_batch = torch.tensor(obs, dtype=torch.float32, device=config.get_device()).unsqueeze(1)
                logits = model(state_batch)
                action = logits.argmax().item()
            obs, reward, done = env.step(action)
            total_reward += reward
            length += 1
            traj.append(env.agent_pos)
        successes.append(env.agent_pos[1] >= config.env_size - 1)
        rewards.append(total_reward)
        lengths.append(length)
        trajectories.append(traj)
    return successes, rewards, lengths, trajectories

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
    print(loss(bc_agent.model, config.get_device(), DataLoader(val_data, **config.PARAMS), nn.CrossEntropyLoss()))
    for name, model in zip(["BC", "DQN", "BCQ"], [bc_agent.model, dqn_agent.model, bcq_agent.model]):
        successes, rewards, lengths, trajectories = evaluate(test_data, model)
        print(
            f"{name} Success Rate: {np.mean(successes):.2f}, Avg Reward: {np.mean(rewards):.2f}, Avg Length: {np.mean(lengths):.2f}")