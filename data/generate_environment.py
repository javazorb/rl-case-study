import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import config


def generate_environment(size=(config.ENV_SIZE, config.ENV_SIZE),
                         obstacle_range=(config.OBSTACLE_RANGE_START, config.OBSTACLE_RANGE_END),
                         floor_height_range=(config.FLOOR_HEIGHT_RANGE_START, config.FLOOR_HEIGHT_RANGE_END),
                         obstacle_width=config.OBSTACLE_WIDTH,
                         obstacle_height_range=(config.OBSTACLE_RANGE_HEIGHT_START, config.OBSTACLE_RANGE_HEIGHT_END),
                         nr_obstacles=1):
    environment = np.zeros(size, dtype=np.uint8)
    floor_height = np.random.randint(*floor_height_range + (1,))
    environment[floor_height, :] = config.WHITE
    for _ in range(nr_obstacles):
        obstacle_top = floor_height + 1  # Assuming obstacle is one unit higher than the floor
        obstacle_bottom = obstacle_top + np.random.randint(*obstacle_height_range + (1,))
        obstacle_left = np.random.randint(*obstacle_range + (1,))
        obstacle_right = obstacle_left + obstacle_width

        i_top, i_bottom, i_left, i_right = map(int, [obstacle_top, obstacle_bottom, obstacle_left, obstacle_right])
        environment[i_top:i_bottom, i_left:i_right] = config.WHITE  # Use 255 for white

    return environment


def visualize_and_save_env(environment, save_path):
    plt.imshow(environment, cmap='gray', origin='lower', vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_and_save_environments(num_environments=100, save_directory='data/envs', visualize=True,
                                   floor_height_range=(config.FLOOR_HEIGHT_RANGE_START, config.FLOOR_HEIGHT_RANGE_END),
                                   obstacle_height_range=(config.OBSTACLE_RANGE_HEIGHT_START, config.OBSTACLE_RANGE_HEIGHT_END), nr_obstacles=1):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for i in tqdm(range(num_environments), desc='Generating Environments', unit='Environment'):
        environment = generate_environment(floor_height_range=floor_height_range, obstacle_height_range=obstacle_height_range, nr_obstacles=nr_obstacles)
        save_path = os.path.join(save_directory, f'environment{i}.npy')
        np.save(save_path, environment)
        if visualize:
            vis_path = ''
            if nr_obstacles > 1:
                vis_path = save_directory + '/images'
            else:
                vis_path = 'data/images'
            os.makedirs(vis_path, exist_ok=True)
            visualize_and_save_env(environment, save_path=os.path.join(vis_path, f'environment{i}.png'))


def load_environments(directory='data/envs'):
    environments = []
    #for filename in os.listdir(directory):
    #for filename in sorted(os.listdir(directory), key=lambda x:  int(''.join(filter(str.isdigit, x))) ): # Sorting by number
    for entry in sorted(
            (e for e in os.scandir(directory) if e.is_file()),
            key=lambda e: int(''.join(filter(str.isdigit, e.name)) or 0)
    ):
        if entry.name.endswith('.npy'):
            environment = np.load(entry.path)
            environments.append(environment)
    return environments

def crop_environment_old(env,
                     crop_top=0,
                     crop_right=0,
                     keep_size=60,
                     pad_value=255):

    h, w = env.shape
    cropped = env.copy()
    # crop ceiling
    if crop_top > 0:
        cropped = cropped[:h - crop_top, :]
        #cropped = cropped[crop_top:, :]

    # crop path length
    if crop_right > 0:
        cropped = cropped[:, :w - crop_right]

    new_h, new_w = cropped.shape

    padded = np.ones((keep_size, keep_size), dtype=env.dtype) * pad_value

    # IMPORTANT: place at bottom-left
    padded[keep_size - new_h:, :new_w] = cropped

    return padded


def crop_environment(env,
                     crop_top=0,
                     crop_right=0,
                     keep_size=60):
    h, w = env.shape
    cropped = env.copy()
    # ---------- find top of content ----------
    non_empty_rows = np.where(np.any(cropped != 0, axis=1))[0]
    if len(non_empty_rows) > 0:
        first_content_row = non_empty_rows[0]
    else:
        first_content_row = 0
    # ---------- crop ceiling relative to content ----------
    if crop_top > 0 >= crop_right:
        env[keep_size - crop_top:keep_size, :] = 255
        return env
    elif crop_top > 0 and crop_right > 0:
        cropped[keep_size - crop_top:keep_size, :] = 255
        #new_top = min(first_content_row + crop_top, h)
        #cropped = cropped[new_top:, :]
    # ---------- crop right ----------
    if crop_right > 0:
        cropped = cropped[:, :w - crop_right]
    new_h, new_w = cropped.shape
    # use wall padding
    padded = np.ones((keep_size, keep_size), dtype=env.dtype) * 255
    # bottom-left placement
    padded[keep_size - new_h:, :new_w] = cropped
    return padded

def crop_and_save_all_types(
        source_directory="data/envs",
        save_root="data/cropped_envs",
        crop_top=12,
        crop_right=12,
        visualize=True):

    files = [f for f in os.listdir(source_directory) if f.endswith(".npy")]

    types = {
        "top": (crop_top, 0),
        "side": (0, crop_right),
        "both": (crop_top, crop_right),
    }

    for t in types:
        npy_dir = os.path.join(save_root, t, "npy")
        img_dir = os.path.join(save_root, t, "images")
        os.makedirs(npy_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

    for i, file in enumerate(tqdm(files, desc="Cropping")):

        env = np.load(os.path.join(source_directory, file))

        for t, (ct, cr) in types.items():

            #pad_value = 255 if cr > 0 else 0
            pad_value = 255
            cropped = crop_environment(
                env,
                crop_top=ct,
                crop_right=cr#,
                #pad_value=pad_value
            )

            save_path = os.path.join(
                save_root,
                t,
                "npy",
                f"environment{i}.npy"
            )

            np.save(save_path, cropped)

            if visualize:

                vis_path = os.path.join(
                    save_root,
                    t,
                    "images",
                    f"environment{i}.png"
                )

                visualize_and_save_env(
                    cropped,
                    save_path=vis_path
                )
