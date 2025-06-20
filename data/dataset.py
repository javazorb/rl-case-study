import numpy as np
import config
import math
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from config import Actions
import pickle


def save_dataset(dataset, name):
    with open('data' + os.sep + name + '.pkl', 'wb') as file:
        pickle.dump(dataset, file)


def load_dataset(name, folder):
    with open(folder + os.sep + name + '.pkl', 'rb') as file:
        return pickle.load(file)


def train_test_val_split(environments, optimal_paths):
    """ Splits the environments into i.i.d training, validation and test sets.
    Utilizes the generate_synth_state_actions before splitting

    :param environments:
    :param optimal_paths:
    :return: tuple (train, validation, test)
    """

    envs_actions_list = generate_synth_state_actions(environments, optimal_paths)
    X, y = map(list, zip(*envs_actions_list))
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=config.RANDOM_SEED)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def convert_path_to_actions(env, path):
    actions = []
    for index, coord in enumerate(path):
        if index + 1 < len(path) and path[index + 1][0] > coord[0]:
            actions.append(Actions.JUMP_RIGHT.value)
        else:
            actions.append(Actions.RUN_RIGHT.value)
    return actions


def generate_synth_state_actions(environments, optimal_paths):
    """Generates synthetic state actions pairs for each environment based on a optimal calculated path,
    for each indivual environment

    :param environments: array of np.uint8 arrays
    :param optimal_paths of the environment
    :rtype: list of state action pairs
    :returns: a list of list containing the environments and the action corresponding to the optimal path
    """
    envs_state_action = []
    for index, env in enumerate(environments):
        env_actions = convert_path_to_actions(env, optimal_paths[index])
        envs_state_action.append((env, env_actions))

    return envs_state_action


def get_env_floor_height(environment):
    for index, row in enumerate(environment):
        if config.WHITE in row:
            if len(set(row)) == 1:
                return index


def get_obst_positions(environment, floor_height):
    obst_positions = np.ravel(np.where(environment[floor_height + 1] == config.WHITE)) # ravel converts the 1,d arr to 1d
    return obst_positions[0], obst_positions[-1]


def get_obstacle_height(environment, obst_start_pos):
    height = 0
    for index, row in enumerate(environment):
        if config.WHITE == row[obst_start_pos]:
            height += 1
    return height


def calculate_optimal_trajectory(environment, env_index):
    """
    Calculates the optimal trajectory to be used for the given environment and its id
    :param env_index:
    :param environment:
    :return: either a list of the actions or an array with the agent at every optimal position
    """
    obst_middle = math.ceil(config.OBSTACLE_WIDTH / 2)
    floor_height = get_env_floor_height(environment)
    obst_start_pos, obst_end_pos = get_obst_positions(environment, floor_height)
    previous_pos = 0
    jump_start = obst_start_pos - get_obstacle_height(environment, obst_start_pos)

    agent_positions = []  # To store agent positions for visualization or further processing
    reached_floor = False
    # Traverse environment rows
    for row_index, row in enumerate(environment[floor_height + 1:], start=floor_height + 1):
        if row_index == floor_height + 1:
            if previous_pos == 0:
                for i in range(len(row)):
                    if i < jump_start:
                        agent_positions.append((row_index, i))
                        row[i] = config.AGENT
                    else:
                        previous_pos = i
                        break
                for i in range(obst_end_pos + get_obstacle_height(environment, obst_start_pos), len(row)):
                    agent_positions.append((row_index, i))
                    row[i] = config.AGENT
        if previous_pos != 0 and previous_pos <= obst_start_pos + obst_middle:
            # Move 1 step right and 1 step up until middle of the obstacle
            start_pos = previous_pos
            inital_height = row_index
            for col_index in range(start_pos, len(row)):
                if col_index < obst_start_pos + obst_middle - 1:
                    agent_positions.append((inital_height, col_index))
                    environment[inital_height, col_index] = config.AGENT
                    previous_pos += 1
                    inital_height += 1
                else:
                    break  # Stop when reaching the middle of the obstacle
            # Decrease height and move 1 step right until back at floor height
            current_agent_height = inital_height
            start_pos = previous_pos
            for col_index in range(start_pos, len(row)):
                if current_agent_height > floor_height and not reached_floor:
                    agent_positions.append((current_agent_height, col_index))
                    environment[current_agent_height, col_index] = config.AGENT
                    current_agent_height -= 1
                else:
                    reached_floor = True
                    break  # Stop when back at floor height

    plt.imshow(environment, cmap='gray', origin='lower', vmin=0, vmax=255)
    plt.axis('off')
    #plt.show()
    if not os.path.exists('data/images/optimal_paths'):
        os.makedirs('data/images/optimal_paths')
    plt.savefig(os.path.join('data/images/optimal_paths', f'environment{env_index}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    if not os.path.exists('data/optimal_paths'):
        os.makedirs('data/optimal_paths')
    save_path = os.path.join('data/optimal_paths' + os.sep, f'environment{env_index}.npy')
    np.save(save_path, environment)

    return environment, agent_positions


def move_agent(pos, action):
    x, y = pos
    #if action == Actions.NOTHING.value:
    #    return x + 1, y
    #elif action == JUMP.value:
    #    return x + 1, y + 1
    if action == Actions.RUN_RIGHT.value:
        return x + 1, y
    elif action == Actions.RUN_LEFT.value:
        return x - 1, y
    elif action == Actions.JUMP.value:
        return x, y + 1
    elif action == Actions.JUMP_RIGHT.value:
        return x + 1, y + 1 # +1 on y because else the y pos gets into region < 0 when jumping
    else:
        return pos

def reconstruct_path(environment, actions):
    floor_height = get_env_floor_height(environment)
    start_pos = (0, floor_height + 1)
    path = [start_pos]
    pos = start_pos
    for action in actions:
        pos = move_agent(pos, action)
        x, y = pos
        if action == Actions.RUN_RIGHT.value and y > floor_height + 1: # Account for gravity after jumping
            pos = x, y - 1
        path.append(pos)
    return path


def extract_window(env, start_pos, window_len):
    row, col = start_pos
    half = window_len // 2
    start_idx = col - half
    end_idx = col + half + 1
    pad_left = max(0, -start_idx)
    pad_right = max(0, end_idx - env.shape[1])
    #env_window = env[:, start_idx:end_idx]
    slice_start = max(start_idx, 0)
    slice_end = min(end_idx, env.shape[1])
    env_window = env[:, slice_start:slice_end] # slicing column vectors
    env_padded = np.pad(env_window, pad_width=((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    return env_padded


def extract_env_windows(environments, agent_start_positions, window_len = 5):
    batch = []
    for idx, env in enumerate(environments):
        start_pos = agent_start_positions[idx]
        batch.append(extract_window(env, start_pos, window_len))
    return batch


def update_agent_pos(agent_start_positions, expert_paths):
    new_positions = []
    for i, pos in enumerate(agent_start_positions):
        expert_path = expert_paths[i]
        path_len = len(expert_path)
        curr_pos = pos
        closest_idx = np.argmin([np.linalg.norm(np.array(expert_path[j]) - np.array(curr_pos)) for j in range(path_len)])
        nex_pos_idx = min(closest_idx + 1, path_len - 1)
        next_pos = expert_path[nex_pos_idx]
        new_positions.append(next_pos)
    return new_positions