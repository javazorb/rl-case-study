import json
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import config
import data.generate_environment as generate_data
import data.dataset as data
import tqdm
import training.train_bc as train_bc
import models.bc_model as bc_model
import models.q_model as q_model
import models.bcq_model as bcq_model
from data.dataloader import EnvironmentDataset


def run():
    #envs = data_gen()
    #sets_generation()
    behavior_cloning = bc_model.BehavioralModel()
    train_data = data.load_dataset('train_data', 'data')
    test_data = data.load_dataset('test_data', 'data')
    val_data = data.load_dataset('val_data', 'data')
    train_set = EnvironmentDataset(train_data)
    val_set = EnvironmentDataset(val_data)
    test_set = EnvironmentDataset(test_data)
    train_bc.train(behavior_cloning, config.get_device(), train_set, val_set, nn.CrossEntropyLoss(), optim.Adam(behavior_cloning.parameters(), lr=0.001))


def sets_generation():
    envs = generate_data.load_environments()
    optimal_paths = load_optimal_paths()
    train_data, test_data, val_data = data.train_test_val_split(environments=envs, optimal_paths=optimal_paths)
    data.save_dataset(train_data, 'train_data')
    data.save_dataset(test_data, 'test_data')
    data.save_dataset(val_data, 'val_data')


def data_gen():
    #generate_data.generate_and_save_environments(num_environments=1000)
    envs = generate_data.load_environments()
    save_optimal_paths(envs)
    return envs


def save_optimal_paths(envs):
    agent_positions_all_envs = []
    for index, env in tqdm.tqdm(enumerate(envs), total=len(envs), desc="calculating optimal paths", unit="Environments"):
        _, agent_positions = data.calculate_optimal_trajectory(env, index)
        agent_positions_all_envs.append((index, sorted(list(set(agent_positions)), key=lambda x: x[1])))
    with open('data' + os.sep + 'optimal_paths.json', 'w') as file:
        json.dump(agent_positions_all_envs, file, indent=2)


def load_optimal_paths():
    with open('data' + os.sep + 'optimal_paths.json', 'r') as file:
        data = json.load(file)
    _, paths = map(list, zip(*data))
    return paths


if __name__ == '__main__':
    run()
