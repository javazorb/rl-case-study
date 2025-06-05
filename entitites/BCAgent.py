from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from entitites.BaseAgent import BaseAgent
from models.bc_model import BehavioralModel
import config
import copy
import data.dataset as dataset

class BCAgent(BaseAgent):
    def __init__(self, optimizer, criterion, early_stopping=10):
        #super().__init__(optimizer, criterion)
        self.device = config.get_device()
        self.model = BehavioralModel().to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.early_stopping = early_stopping

    def train(self, train_set, val_set):
        self.model.train()
        np.random.seed(config.RANDOM_SEED)
        train_loader = DataLoader(train_set, **config.PARAMS)
        val_loader = DataLoader(val_set, **config.PARAMS)
        train_loss = 0
        val_loss = 0
        stop_counter = 0
        epochs_ran = 0
        best_val_loss = float('inf')
        best_model = self.model

        for epoch in range(config.MAX_EPOCHS):
            epoch_loss = 0
            for environments, actions in tqdm(train_loader, desc=f"Training Epoch: {epoch + 1}/{config.MAX_EPOCHS}"):
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
                    state_batch = torch.from_numpy(state_batch).float().to(self.device)  # shape [1, 10, 60, 5]
                    state_batch = state_batch.unsqueeze(1)  # corrected shape [10, 1, 60, 5]

                    self.optimizer.zero_grad()
                    predicted_actions = self.model(state_batch.to(self.device))
                    action_idxs = [x for x, y in agent_start_positions]
                    correct_actions = [actions[i][action_idxs[i]] for i in range(len(action_idxs))]
                    cur_loss = self.criterion(predicted_actions, torch.LongTensor(correct_actions).to(self.device))
                    cur_loss.backward()
                    self.optimizer.step()
                    batch_loss = cur_loss.item()
                    agent_start_positions = dataset.update_agent_pos(agent_start_positions,
                                                                     expert_paths)  # updated along the expert path
                batch_loss /= config.NUM_STEPS_ENV
                epoch_loss += batch_loss
            train_loss += epoch_loss
            val_loss = self.loss(val_loader)
            print(f'Validation loss: {val_loss} at epoch {epoch + 1}/{config.MAX_EPOCHS}')
            if val_loss < best_val_loss:
                print(f'New best validation loss: {val_loss}\n old best validation loss: {best_val_loss}')
                best_val_loss = val_loss
                stop_counter = 0
                best_model = copy.deepcopy(self.model)
                config.save_model(self.model, name=f"BC_{epoch}")
            else:
                stop_counter += 1

            if stop_counter >= self.early_stopping:
                epochs_ran = epoch + 1
                break
        config.save_model(best_model, name="final_BC")
        print(f'Final training loss: {train_loss / epochs_ran:.4f} after {epochs_ran} epochs')
        self.model = best_model

    def evaluate(self, environment):
        self.model.eval()
        pass

    def loss(self, val_loader):
        self.model.eval()
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
                    state_batch = torch.from_numpy(state_batch).float().to(self.device)
                    state_batch = state_batch.unsqueeze(1)  # Ensure the correct shape [batch_size, 1, 60, 5]

                    predicted_actions = self.model(state_batch.to(self.device))

                    action_idxs = [x for x, y in agent_start_positions]
                    correct_actions = [actions[i][action_idxs[i]] for i in range(len(action_idxs))]

                    # Compute the loss for this batch
                    batch_loss += self.criterion(predicted_actions, torch.LongTensor(correct_actions).to(self.device)).item()
                    agent_start_positions = dataset.update_agent_pos(agent_start_positions, expert_paths)
                batch_loss /= config.NUM_STEPS_ENV
            val_loss += batch_loss
        val_loss /= len(val_loader)
        return val_loss