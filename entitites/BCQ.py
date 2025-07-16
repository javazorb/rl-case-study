import torch
import torch.nn.functional as F
import copy

import config


def train_bcq(agent, replay_buffer, num_epochs=100, steps_per_epoch=1000, batch_size=32):
    for epoch in range(num_epochs):
        for _ in range(steps_per_epoch):
            logs = agent.train(replay_buffer, batch_size)

        print(f"[Epoch {epoch}] Total loss: {logs['total_loss']:.4f}, Q: {logs['q_loss']:.4f}, I: {logs['i_loss']:.4f}")


class DiscreteBCQAgent:
    def __init__(self, model, num_actions, threshold=0.3, gamma=0.99, lr=1e-4, device=config.get_device()):
        self.device = device
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.num_actions = num_actions
        self.threshold = threshold
        self.gamma = gamma
        self.update_counter = 0

    def select_action(self, state):
        self.model.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, 1, 60, 60]
            q, imt, _ = self.model(state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()
            final_q = imt * q + (1 - imt) * -1e8
            action = final_q.argmax(1).item()
        return action

    def train(self, replay_buffer, batch_size=32):
        self.model.train()
        self.model.to(self.device)

        # Sample
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        #target_qs = []
        #for i in range(batch_size):
        #    next_state = next_states.cpu().numpy()[i]
        #    next_state = torch.from_numpy(next_state).float().to(self.device)
        #    # Target Q
        #    with torch.no_grad():
        #        q_next, imt_next, _ = self.model(next_state)
        #        imt_next = imt_next.exp()
        #        imt_next = (imt_next / imt_next.max(1, keepdim=True)[0] > self.threshold).float()
        #        q_next = imt_next * q_next + (1 - imt_next) * -1e8
        #        next_actions = q_next.argmax(1, keepdim=True)
        #        target_q = rewards + self.gamma * (1 - dones) * self.target_model(next_states)[0].gather(1, next_actions)
        #        target_qs.append(target_q)
        with torch.no_grad():
            q_next, imt_next, _ = self.model(next_states)
            imt_next = imt_next.exp()  # log_probs → probs
            threshold_mask = (imt_next / imt_next.max(1, keepdim=True)[0] > self.threshold).float()
            q_next = threshold_mask * q_next + (1 - threshold_mask) * -1e8  # maskiere nicht-expert-Aktionen
            next_actions = q_next.argmax(1, keepdim=True)
            #next_actions[next_actions == 1] = 3
            mask = (next_actions == 1)
            next_actions = torch.where(mask, torch.tensor(3, device=next_actions.device), next_actions)
            target_q = rewards + self.gamma * (1 - dones) * self.target_model(next_states)[0].gather(1, next_actions)

        # Current Q + Imitation
        q_values, imt, i_logits = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1))

        q_loss = F.mse_loss(q_values, target_q)
        i_loss = F.nll_loss(imt, actions.squeeze())
        reg_loss = 1e-2 * i_logits.pow(2).mean()

        loss = q_loss + i_loss + reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update (or hard every N steps)
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return {
            "q_loss": q_loss.item(),
            "i_loss": i_loss.item(),
            "reg_loss": reg_loss.item(),
            "total_loss": loss.item(),
        }

