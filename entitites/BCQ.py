import torch
import torch.nn.functional as F
import copy
from PIL import Image
import config
from data.generate_environment import generate_environment
from environments.QEnvironment import QEnvironment
import os


def train_bcq(agent, replay_buffer, num_epochs=100, steps_per_epoch=1000, batch_size=32):
    EVAL_FREQUENCY = 10
    for epoch in range(num_epochs):
        for _ in range(steps_per_epoch):
            logs = agent.train(replay_buffer, batch_size)

        print(f"[Epoch {epoch}] Total loss: {logs['total_loss']:.4f}, Q: {logs['q_loss']:.4f}, I: {logs['i_loss']:.4f}")
        if epoch % EVAL_FREQUENCY == 0:
            eval_env = QEnvironment(size=config.ENV_SIZE, environment=generate_environment(), start_pos=None)

            gif_path = f"eval_outputs/epoch_{epoch}.gif"
            if not os.path.exists("eval_outputs"):
                os.makedirs("eval_outputs")
            evaluate_and_save_gif(agent, eval_env, gif_path)


def evaluate_and_save_gif(agent, env, gif_path, max_steps=config.MAX_STEPS):
    frames = []
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        frame = env.render(mode='rgb_array')
        frames.append(Image.fromarray(frame))
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            break
    frames.append(Image.fromarray(env.render(mode="rgb_array")))

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    print(f"[GIF] Gespeichert unter {gif_path} | Reward: {total_reward:.2f}")


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
            q_next, imt_next, _ = self.target_model(next_states)

            action_counts = torch.bincount(actions, minlength=self.num_actions).cpu().numpy()

            imt_next = imt_next.exp()  # log_probs → probs
            threshold_mask = (imt_next / imt_next.max(1, keepdim=True)[0] > self.threshold).float()
            q_next = threshold_mask * q_next + (1 - threshold_mask) * -1e8 #-1e8  # maskiere nicht-expert-Aktionen
            next_actions = q_next.argmax(1, keepdim=True)
            #next_actions[next_actions == 1] = 3
            mask = (next_actions == 1)
            next_actions = torch.where(mask, torch.tensor(3, device=next_actions.device), next_actions)
            target_q = rewards + self.gamma * (1 - dones) * self.target_model(next_states)[0].gather(1, next_actions)

        # Current Q + Imitation
        q_values, imt, i_logits = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1))

        q_loss = F.smooth_l1_loss(q_values, target_q)
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

