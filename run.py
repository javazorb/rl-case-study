import json
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
import data.generate_environment as generate_data
import data.dataset as data
import tqdm
import training.train_bc as train_bc
import training.train_q as train_q
import models.bc_model as bc_model
import models.q_model as q_model
import models.bcq_model as bcq_model
from data.dataloader import EnvironmentDataset
import torch
import models.hyperparameter as hyperparameter
from entitites.BCAgent import BCAgent
from training.train_q import ReplayBuffer


def load_model(name, model):
    path = f'./trained_models/{name}.pt'
    model = model
    model.load_state_dict(torch.load(path))
    return model


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
    weights = torch.tensor([1.0, 3.0]).to(config.get_device())
    #bc_agent = BCAgent(optimizer=optim.AdamW(behavior_cloning.parameters(), lr=0.001), criterion=nn.CrossEntropyLoss(weight=weights), early_stopping=10)
    #bc_agent.train(train_set, val_set)
    #acc = train_bc.test_accuracy(bc_agent.model, config.get_device(), test_set)
    #print(f'Test Accuracy: {acc}')
    #best_params = hyperparameter.search_hyperparameters(behavior_cloning, learning_rates=[0.001, 0.0005, 0.0001],
    #                                                   batch_sizes=[10, 32, 64, 128], optimizers=[optim.Adam, optim.SGD],
    #                                                   train_set=train_set, val_set=val_set)
    #print(best_params)
    #optimizer=optim.Adam(behavior_cloning.parameters(), lr=0.001), criterion=nn.CrossEntropyLoss())
    #train_bc.train(behavior_cloning, config.get_device(), train_set, val_set,
    #               optimizer=optim.AdamW(behavior_cloning.parameters(), lr=0.0001), criterion=nn.CrossEntropyLoss())

    #test_accuracy(behavior_cloning, test_set, 'final_BC_state_dict')
    #q_agent = q_model.QModel()
    #q_agent = load_model('final_Q_state_dict', q_agent)
    #buffer_len = train_q.warm_start_replay_buffer(ReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE), DataLoader(train_set, **config.PARAMS), config.get_device())
    #train_q.loss(q_agent, config.get_device(), DataLoader(val_set, **config.PARAMS), criterion=nn.MSELoss())
    #train_q.train(q_agent, config.get_device(), train_set, val_set, criterion=nn.MSELoss(), optimizer=optim.Adam(q_agent.parameters(), lr=0.0001))
    #train_q.evaluate_model_and_vis(q_agent, config.get_device(), DataLoader(train_set, **config.PARAMS), num_episodes=5)


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


def test_accuracy(model, test_set, name='final_BC_state_dict'):
    best_bc_model = load_model(name, model)
    acc = train_bc.test_accuracy(best_bc_model, config.get_device(), test_set)
    print(f'Test Accuracy: {acc}')


if __name__ == '__main__':
    run()
