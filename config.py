import torch
from enum import Enum
import os

RANDOM_SEED = 42
ENV_SIZE = 60
OBSTACLE_WIDTH = 5
OBSTACLE_RANGE_START = 20
OBSTACLE_RANGE_END = 45
OBSTACLE_RANGE_HEIGHT_START = 1
OBSTACLE_RANGE_HEIGHT_END = 10
FLOOR_HEIGHT_RANGE_START = 1
FLOOR_HEIGHT_RANGE_END = 10
WHITE = 255
AGENT_START_POS = 0
AGENT_END_POS = 59
AGENT = 130
MAX_JUMP_HEIGHT = 20
WINDOW_LEN = 5
NUM_STEPS_ENV = WINDOW_LEN * 2
BATCH_SIZE = 10
PARAMS = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
MAX_EPOCHS = 100
MAX_STEPS = 100  # Maximum steps per episode
EPS_DECAY = 0.99  # Decay rate for epsilon in epsilon-greedy strategy
MIN_EPSILON = 0.01  # Minimum value for epsilon after decay
GAMMA = 0.99  # Discount factor for future rewards
REPLAY_BUFFER_SIZE = 10000
NUM_REPLAY_SAMPLE = 32
# RUN_RIGHT = 1
# RUN_LEFT = 2
# JUMP = 3
# JUMP_RIGHT = 4


class Actions(Enum):
    RUN_RIGHT = 0
    RUN_LEFT = 1
    JUMP = 2
    JUMP_RIGHT = 3


def get_device():
    use_cuda = torch.cuda.is_available()
    print(f'Using cuda: {use_cuda}')
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if torch.backends.mps.is_available():
        print(f'Using mps: {torch.backends.mps.is_available()}')
        device = torch.device("mps")
    return device

def save_model(model, name, path=os.getcwd() + os.sep + 'trained_models' + os.sep):
    if not os.path.exists(path):
        os.makedirs(path)
    path_model = os.path.join(path, f'{name}_model.pt')
    path_state_dict = os.path.join(path, f'{name}_state_dict.pt')
    torch.save(model.state_dict(), path_state_dict)
    torch.save(model, path_model)
    print(f'Model saved to {path_model}')
