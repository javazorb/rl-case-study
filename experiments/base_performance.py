import numpy as np
from torch.utils.data import DataLoader

import config
from entitites.replay_buffer import ReplayBuffer
from experiments.experiments_setup import evaluate
import training.train_bc_new as train_bc_new
import entitites.DQNAgent
import entitites.BCQ as bcq
from models.bc_model import BehavioralModel
from models.bcq_model import BCQModel
from models.q_model import QModel
from run import load_model
import models


def run_experiment_1(agents, train_data, val_data, test_data, train=True):
    """
    Baseline Performance
    """
    print("Running Experiment 1: Baseline Performance")
    test_loader = DataLoader(test_data, **config.PARAMS)
    buffer = ReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
    buffer = bcq.fill_buffer(list(DataLoader(train_data, **config.PARAMS)))
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
    for name, model in zip(["BC", "DQN", "BCQ"], [bc_agent.model, dqn_agent.model, bcq_agent.model]):
        successes, rewards, lengths, trajectories = evaluate(test_loader, model)
        print(
            f"{name} Success Rate: {np.mean(successes):.2f}, Avg Reward: {np.mean(rewards):.2f}, Avg Length: {np.mean(lengths):.2f}")