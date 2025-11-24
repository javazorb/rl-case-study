import random
from itertools import islice
import numpy as np
import torch
import torch.nn.functional as F
import copy
from PIL import Image
import config
from data import dataset
from data.generate_environment import generate_environment
from entitites.replay_buffer import ReplayBuffer
from environments.QEnvironment import QEnvironment
import os


def train_bcq(agent, replay_buffer, num_epochs=100, steps_per_epoch=1000, batch_size=32):
    EVAL_FREQUENCY = 10
    for epoch in range(num_epochs):
        for _ in range(steps_per_epoch):
            logs = agent.train(replay_buffer, batch_size)

        print(f"[Epoch {epoch}] Total loss: {logs['total_loss']:.4f}, Q: {logs['q_loss']:.4f}, I: {logs['i_loss']:.4f}")
        if epoch % EVAL_FREQUENCY == 0:
            pass
            eval_env = QEnvironment(size=config.ENV_SIZE, environment=generate_environment(), start_pos=None)

            gif_path = f"eval_outputs/epoch_{epoch}.gif"
            if not os.path.exists("eval_outputs"):
                os.makedirs("eval_outputs")
            evaluate_and_save_gif(agent, eval_env, gif_path)
    config.save_model(agent.model, name="final_BCQ")
    print("Training complete.")


def fill_buffer(data_loader, oversample_factor=config.OVERSAMPLE_FACTOR):
    """
        Fill a replay buffer with expert trajectories, oversampling jump actions to increase their ratio.

        Args:
            data_loader: iterable of (environments, actions)
            oversample_factor: int, how many extra times to push each jump action
        Returns:
            buffer: ReplayBuffer with filled transitions
        """
    buffer = ReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
    random.shuffle(data_loader)
    jump_counter_in_buffer = 0

    for envs, actions in data_loader:
        for env, env_actions in zip(envs, actions):
            environment = env.cpu().numpy()
            env_actions = env_actions.cpu().numpy()

            expert_path = dataset.reconstruct_path(environment, env_actions)
            random_start_idx = np.random.randint(len(expert_path))
            curr_env = QEnvironment(size=config.ENV_SIZE, environment=env.cpu().numpy(), start_pos=expert_path[random_start_idx])

            state = curr_env.state.copy()
            for action in env_actions:
                next_state, reward, done = curr_env.step(action)
                buffer.push(state, action, reward, next_state, done)
                state = next_state.copy()

                if action == 3:
                    jump_counter_in_buffer += 1
                if action == 3 and random.random() < 1:
                    for _ in range(oversample_factor):
                        buffer.push(state, action, reward, next_state, done)
                        jump_counter_in_buffer += 1
                if done:
                    break

    jump_count = sum(1 for t in buffer.buffer if t[1] == 3)  # action==3 is jump
    non_jump_count = len(buffer) - jump_count

    jump_ratio = jump_count / len(buffer)
    non_jump_ratio = non_jump_count / len(buffer)

    print(f"Total transitions: {len(buffer)}")
    print(f"Jump actions: {jump_count} ({jump_ratio:.2f})")
    print(f"Non-jump actions: {non_jump_count} ({non_jump_ratio:.2f})")
    return buffer

def generate_jumpy_actions_with_random_jumps(environment, env_actions, max_steps=config.ENV_SIZE, random_jump_prob=0.5):
    """
    generates jump actions with random jumps that jump before the obstacle occurs.
    The random jumps have a probabilit of occuring

    Args:
        environment (np.ndarray): 60x60 uint8 Environment
        actions (list[int]): list of actions
        max_steps (int): maximum number of steps preferably length of environment
        random_jump_prob (float): probability of random jump
    Returns:
        List[int]: of actions for the environment
    """
    actions = []
    lookahead = 5
    floor_height = dataset.get_env_floor_height(environment)
    agent_pos = (0, floor_height + 1)  # Startposition
    obst_start, obst_end = dataset.get_obst_positions(environment, floor_height)
    first_perfect_jump = next(i for i, a in enumerate(env_actions) if a == config.QActions.JUMP_RIGHT.value)
    agent_pos = (0, floor_height + 1)  # Startposition

    # Hindernisinformation
    obst_start, obst_end = dataset.get_obst_positions(environment, floor_height)

    # Berechne Index des ersten perfekten Sprungs
    # -> das ist die Position, an der normalerweise der Expert springen würde
    first_perfect_jump_index = first_perfect_jump
    jump_distance = obst_start - first_perfect_jump_index

    for step in range(max_steps):
        x, y = agent_pos

        # Sprung notwendig? (normale Hindernislogik)
        need_to_jump = (obst_start - jump_distance) <= x < obst_start and y == floor_height + 1

        # Frühzeitige zufällige Sprünge (exploration) — aber nicht im Landebereich
        before_jump_window = x < first_perfect_jump_index - lookahead
        do_random_jump = before_jump_window and random.random() < random_jump_prob and y == floor_height + 1

        # Sprungentscheidung
        if  do_random_jump :#or need_to_jump:
            action = 3  # jump
            agent_pos = (x + 1, y + 1)  # nach oben springen
        else:
            action = env_actions[step]  # do_nothing
            new_y = y - 1 if y > floor_height + 1 else y
            agent_pos = (x + 1, new_y)
        if x > obst_end and random.random() < random_jump_prob:
            action = 3
            agent_pos = (x + 1, y + 1)

        # Agent darf nicht unter Bodenhöhe sinken
        if agent_pos[1] < floor_height + 1:
            agent_pos = (agent_pos[0], floor_height + 1)

        actions.append(action)

        # Falls Agent am rechten Rand ist, abbrechen
        if agent_pos[0] >= environment.shape[1] - 1:
            break

    # Falls Aktionen kürzer als max_steps sind, mit do_nothing auffüllen
    while len(actions) < max_steps:
        actions.append(0)

    return actions


def evaluate_and_save_gif(agent, env, gif_path, max_steps=config.MAX_STEPS):
    frames = []
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        frame = env.render(mode='rgb_array')
        #frames.append(Image.fromarray(frame))
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            break
    #frames.append(Image.fromarray(env.render(mode="rgb_array")))

    #frames[0].save(
    #    gif_path,
    #    save_all=True,
    #    append_images=frames[1:],
    #    duration=100,
    #    loop=0
    #)
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
            if action == 1:
                action = 3
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
            mask = (next_actions == 1)
            next_actions = torch.where(mask, torch.tensor(3, device=next_actions.device), next_actions)
            #target_q = rewards + self.gamma * (1 - dones) * self.target_model(next_states)[0].gather(1, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * q_next.gather(1, next_actions)

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

