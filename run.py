import copy
import json
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.ao.nn.quantized.functional import threshold
from torch.utils.data import DataLoader

import config
import data.generate_environment as generate_data
import data.dataset as data
import tqdm
import training.train_bc as train_bc
import training.train_bc_new as train_bc_new
import training.train_q as train_q
import models.bc_model as bc_model
import models.q_model as q_model
import models.bcq_model as bcq_model
import models.base_model as base_model
from data.dataloader import EnvironmentDataset
import torch
import models.hyperparameter as hyperparameter
from entitites.BCAgent import BCAgent
from entitites.DQNAgent import DQNAgent
from entitites.DQNAgent import warm_start_replay_buffer
from experiments.DQN_new_approach import train_dqn
from experiments.base_performance import run_experiment_1
from training.train_q import ReplayBuffer
from entitites.BCQ import DiscreteBCQAgent, fill_buffer
from entitites.BCQ import train_bcq


def load_model(name, model):
    path = f'./trained_models/{name}.pt'
    model = model
    model.load_state_dict(torch.load(path))
    return model


def run():
    # envs = data_gen()
    # sets_generation()
    behavior_cloning = bc_model.BehavioralModel()
    train_data = data.load_dataset('train_data', 'data')
    test_data = data.load_dataset('test_data', 'data')
    val_data = data.load_dataset('val_data', 'data')
    train_set = EnvironmentDataset(train_data)
    val_set = EnvironmentDataset(val_data)
    test_set = EnvironmentDataset(test_data)
    #weights = torch.tensor([1.0, 3.0]).to(config.get_device())
    #bc_agent = BCAgent(optimizer=optim.AdamW(behavior_cloning.parameters(), lr=0.001), criterion=nn.CrossEntropyLoss(weight=weights), early_stopping=10)
    #bc_agent.train(train_set, val_set)
    #acc = train_bc.test_accuracy(bc_agent.model, config.get_device(), test_set)
    #print(f'Test Accuracy: {acc}')
    #best_params = hyperparameter.search_hyperparameters(behavior_cloning, learning_rates=[0.001, 0.0005, 0.0001],
    #                                                   batch_sizes=[10, 32, 64, 128], optimizers=[optim.Adam, optim.SGD, optim.AdamW],
    #                                                   train_set=train_set, val_set=val_set)
    #print(best_params)
    #optimizer=optim.Adam(behavior_cloning.parameters(), lr=0.001, criterion=nn.CrossEntropyLoss())

    #train_bc.train(behavior_cloning, config.get_device(), train_set, val_set,
    #                optimizer=optim.AdamW(behavior_cloning.parameters(), lr=0.0001), criterion=nn.CrossEntropyLoss())
    #behavior_cloning = load_model('final_BC_state_dict', behavior_cloning)
    #acc, total_actions_predicted, num_counts_jump_right = train_bc.test_accuracy(behavior_cloning, config.get_device(), test_set)
    #train_dqn(train_set, val_set, steps=100000, batch_size=64, # TODO current used version
    #          gamma=0.99, lr=1e-4, capacity=100000, update_target=10000)
    behavior_cloning = base_model.BaseModel()
    #train_bc_new.train_only_jump(behavior_cloning, config.get_device(), train_set, val_set, # TODO both current used versions
    #                             optimizer=optim.AdamW(behavior_cloning.parameters(), lr=1e-4, weight_decay=1e-4),
    #                             criterion=None, early_stopping=3)
    #train_bc_new.train(behavior_cloning, config.get_device(), train_set, val_set, optimizer=optim.AdamW(behavior_cloning.parameters(), lr=1e-4, weight_decay=1e-4), criterion=None)

    #behavior_cloning = load_model("final_BC_state_dict", behavior_cloning)
    #acc, predicted_actions = train_bc_new.test_bc_accuracy(behavior_cloning, config.get_device(), val_set)
    #pred_array = np.array([p for batch in total_actions_predicted for p in batch])
    #jump_mask = (pred_array == config.Actions.JUMP_RIGHT.value)
    #jump_count = jump_mask.sum()
    #total_preds = pred_array.size
    #print(f'accuracy: {acc}')
    #print(f'predicted actions: {predicted_actions}')
    #print(f'jump count versus total predictions: {num_counts_jump_right} / {len(total_actions_predicted)}   Number of jump right of experts: {expert_jump_right}')
    q_net = q_model.QModel()
    #trained_q = load_model('final_Q_state_dict', q_agent)
    #buffer_len = train_q.warm_start_replay_buffer(ReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE), DataLoader(train_set, **config.PARAMS), config.get_device())
    #train_q.loss(q_agent, config.get_device(), DataLoader(val_set, **config.PARAMS), criterion=nn.MSELoss())
    #train_q.train(q_agent, config.get_device(), train_set, val_set, criterion=nn.MSELoss(), optimizer=optim.Adam(q_agent.parameters(), lr=0.0001))
    #train_q.evaluate_model_and_vis(q_agent, config.get_device(), DataLoader(train_set, **config.PARAMS), num_episodes=5)

    #q_agent = DQNAgent(optimizer=optim.Adam(q_net.parameters(), lr=0.001), criterion=nn.MSELoss())
    #q_agent.train(train_set, val_set)
    #train_q.evaluate_model_and_vis(q_agent.model, config.get_device(), DataLoader(train_set, **config.PARAMS), num_episodes=5)

    model = bcq_model.BCQModel(input_shape=(1, 60, 60), num_actions=len(config.QActions))
    agent = DiscreteBCQAgent(model=model, num_actions=100, threshold=0.1)
    buffer = ReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
    train_loader = DataLoader(train_set, **config.PARAMS)
    val_loader = DataLoader(val_set, **config.PARAMS)
    buffer = fill_buffer(list(train_loader))
    #train_bc_new.train_bc_from_buffer(behavior_cloning, buffer, config.get_device())
    #train_bc_new.evaluate_bc_jump_metrics(behavior_cloning, test_set, config.get_device())
    #os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    q_agent = DQNAgent(optimizer=optim.Adam(q_net.parameters(), lr=0.001), criterion=nn.MSELoss())
    bc_agent = BCAgent(optimizer=optim.AdamW(behavior_cloning.parameters(), lr=0.001),
                       criterion=nn.CrossEntropyLoss(), early_stopping=10)
    agents = [bc_agent, q_agent, agent, copy.deepcopy(bc_agent)]

    #run_experiment_1(agents, train_set, val_set, test_set, buffer, train=False)
    #data_gen(nr_obstacles=2, visualize=True, save_directory='data/multiple_obstacles')
    multiple_obst_envs = generate_data.load_environments('data/multiple_obstacles')
    optimal_paths = load_optimal_paths('data/multiple_obstacles')
    multiple_obst_data_set = data.train_test_val_split(environments=multiple_obst_envs, optimal_paths=optimal_paths, single=True)
    multiple_obst_data_set = dataset = list(zip(multiple_obst_data_set[0], multiple_obst_data_set[1]))
    #buffer = fill_buffer(list(DataLoader(multiple_obst_data_set, **config.PARAMS)))
    run_experiment_1(agents, None, None, multiple_obst_data_set, buffer, train=False)
    #train_bcq(agent, buffer, num_epochs=200, steps_per_epoch=1000, batch_size=32)


def sets_generation(single=False):
    envs = generate_data.load_environments()
    optimal_paths = load_optimal_paths()
    train_data, test_data, val_data = data.train_test_val_split(environments=envs, optimal_paths=optimal_paths, single=single)
    data.save_dataset(train_data, 'train_data')
    data.save_dataset(test_data, 'test_data')
    data.save_dataset(val_data, 'val_data')


def data_gen(nr_obstacles=1, visualize=True, save_directory='data/envs'):
    generate_data.generate_and_save_environments(save_directory=save_directory, num_environments=100, nr_obstacles=nr_obstacles, visualize=visualize)
    #generate_data.generate_and_save_environments(num_environments=1000)
    envs = generate_data.load_environments(save_directory)
    if nr_obstacles == 1:
        save_optimal_paths(envs)
    else:
        save_optimal_paths(envs, save_dir=save_directory)
    return envs


def save_optimal_paths(envs, save_dir='data'):
    agent_positions_all_envs = []
    for index, env in tqdm.tqdm(enumerate(envs), total=len(envs), desc="calculating optimal paths",
                                unit="Environments"):
        _, agent_positions = data.calculate_optimal_trajectory(env, index)
        agent_positions_all_envs.append((index, sorted(list(set(agent_positions)), key=lambda x: x[1])))
    with open(save_dir + os.sep + 'optimal_paths.json', 'w') as file:
        json.dump(agent_positions_all_envs, file, indent=2)


def load_optimal_paths(dir='data'):
    with open(dir + os.sep + 'optimal_paths.json', 'r') as file:
        data = json.load(file)
    _, paths = map(list, zip(*data))
    return paths


def test_accuracy(model, test_set, name='final_BC_state_dict'):
    best_bc_model = load_model(name, model)
    acc = train_bc.test_accuracy(best_bc_model, config.get_device(), test_set)
    print(f'Test Accuracy: {acc}')


if __name__ == '__main__':
    run()
