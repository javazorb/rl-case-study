import numpy as np
import config
import data.dataset as dataset


class QEnvironment:
    def __init__(self, size=config.ENV_SIZE, environment=None, start_pos=None):
        self.good_action_cnt = 0
        self.jump_count = 0
        self.stuck_cnt = 0
        if environment is not None:
            environment = np.squeeze(environment)
        self.environment = environment
        self.size = size
        self.size_x, self.size_y = environment.shape
        floor_height = dataset.get_env_floor_height(self.environment)
        self.goal_position = (floor_height + 1, self.size - 1)
        #self.start_position = (floor_height + 1, 0)
        self.start_position = start_pos if start_pos is not None else (floor_height + 1, 0)
        #self.start_position = start_pos if start_pos is not None else (0, floor_height + 1)
        self.current_position = self.start_position
        self.state = np.zeros((1, self.size, self.size), dtype=np.float32)
        #self.state = np.zeros((1, self.size, self.size), dtype=np.uint8)
        self.done = False

        self.reset()

    def reset(self):
        self.state.fill(0)
        floor_height = dataset.get_env_floor_height(self.environment)
        self.start_position = (floor_height + 1, floor_height +1)
        self.current_position = self.start_position
        self.goal_position = (floor_height + 1, self.size - 1)
        self.done = False

        self.good_action_cnt = 0
        self.jump_count = 0
        self.stuck_cnt = 0

        self.state[0, floor_height, :] = self.environment[floor_height, :]
        self.state[0, self.goal_position[0], self.goal_position[1]] = 2  # Mark goal
        self.state[0, self.start_position[0], self.start_position[1]] = 1  # Mark start
        self.state[0, self.current_position[0], self.current_position[1]] = 3  # Agent marker
        return self.state

    def step_og(self, action):
        if self.current_position == self.goal_position:
            reward = 10
            self.done = True
            return self.current_position, reward, self.done
        x, y = self.current_position
        #if action == 0:  # RUN_RIGHT
        #    new_position = (x, min(y + 1, self.size - 1))
        #elif action == 1:  # RUN_LEFT
        #    new_position = (x, max(y - 1, 0))
        #elif action == 2:  # JUMP
        #    new_position = (min(x + 1, self.size - 1), y)
        #elif action == 3:  # JUMP_RIGHT
        #    new_position = (min(x + 1, self.size - 1), min(y + 1, self.size - 1))

        #if action == 0:  # RUN_RIGHT
        #    new_position = (x + 1, y)
        #elif action == 1:  # RUN_LEFT
        #    new_position = (max(x - 1, 0), y)
        #elif action == 2:  # JUMP
        #    new_position = (x, y + 1)
        #elif action == 3:  # JUMP_RIGHT
        #    new_position = (x + 1, y + 1)

        # Correct version with (row, col) indexing
        if action == 0:  # RUN_RIGHT
            new_position = (x, y + 1)
        elif action == 1:  # RUN_LEFT
            new_position = (x, y - 1)
        elif action == 2:  # JUMP (UP)
            new_position = (x - 1, y)
        elif action == 3:  # JUMP_RIGHT
            new_position = (x - 1, y + 1)

        new_x, new_y = new_position
        new_row = np.clip(new_x, 0, self.size - 1)
        new_col = np.clip(new_y, 0, self.size - 1)
        new_position = (new_row, new_col)
        new_x, new_y = new_position
        if y > dataset.get_env_floor_height(self.environment) + config.OBSTACLE_RANGE_HEIGHT_END + 5:
            new_y = dataset.get_env_floor_height(self.environment) + config.OBSTACLE_RANGE_HEIGHT_END + 5
        # Check if the agent is above the floor and if the next action is not jumping
       # if y > dataset.get_env_floor_height(self.environment) + 1 and action not in [2, 3]:
       #     new_y = min(y - 1, dataset.get_env_floor_height(self.environment) + 1)  # Move down, simulate gravity
       #     new_position = (new_x, new_y)

        # Gravity effect (falling down if no jump is made and in the air)
        if x < dataset.get_env_floor_height(self.environment) and action not in [2, 3]:
            new_x = min(x + 1, self.size - 1)
            new_position = (new_x, new_y)

        reward = 0

        if self.environment[new_x, new_y] == 255:
            reward = -1  # Hit obstacle
            next_position = self.current_position
            new_position = self.current_position
        elif new_position == self.goal_position:
            reward = 10
            next_position = new_position
            self.done = True
        else:
            next_position = new_position
            if action == 1:
                reward = -2
            else:
                reward = 1
            goal_x = self.goal_position[0]
            prev_dist = abs(goal_x - self.current_position[0])
            new_dist = abs(goal_x - new_position[0])
            if new_dist < prev_dist:
                reward += 0.5
            elif new_dist > prev_dist:
                reward -= 0.5
            if abs(new_col - self.goal_position[1]) < 5:
                reward += 1  # proximity bonus
            if self.environment[x, y] == 255 and self.environment[new_row, new_col] == 0:
                reward += 2  # successfully jumped over an obstacle

        self.current_position = new_position
        self.state.fill(0)
        self.state[0, dataset.get_env_floor_height(self.environment), :] = self.environment[dataset.get_env_floor_height(self.environment), :]
        self.state[0, self.goal_position[0], self.goal_position[1]] = 2  # Mark goal
        self.state[0, self.start_position[0], self.start_position[1]] = 1  # Mark start
        next_state = self.state
        return next_state, reward, self.done

    def step(self, action):
        if self.done:
            return self.state, 0, True

        x, y = self.current_position
        #print("Agent at:", (y, x), "→ Cell value:", self.environment[y, x], "Remember visualization is origin=lower so coordinates are switched")
        new_x, new_y = x, y

        # Actions: 0 = run right (↓), 1 = run left (↑), 2 = jump (→), 3 = jump right (↘)
        if action == 0:  # Run right # TODO first change to adapt to expert_path behaviour
            new_x += 1 #+
        elif action == 1:  # Run left
            new_x -= 1 #-
        elif action == 2:  # Jump
            new_y += 1 #-
            self.jump_count += 1
        elif action == 3:  # Jump right
            new_x += 1 #-
            new_y += 1 #+
            self.jump_count += 1

        # Clip to environment bounds
        new_x = np.clip(new_x, 0, self.size - 1)
        new_y = np.clip(new_y, 0, self.size - 1)
        reward = -0.1
        new_position = (new_x, new_y)
        floor_height = dataset.get_env_floor_height(self.environment)



        # Check collision
        if self.environment[new_position] == 255:
            reward += -1  # Obstacle hit
            new_position = self.current_position  # optionally stay in place
        elif new_position == self.goal_position:
            reward += 10
            self.done = True
        else:
            reward = 0

            # Encourage forward motion (i.e., right/down direction toward goal)
            if action == 1 or action == 2:
                reward -= 0.5  # discourage moving left (backwards)
            else:
                reward += 1  # small reward for valid forward action

            # Horizontal distance improvement bonus
            #old_dist = abs(self.goal_position[1] - y)
            #new_dist = abs(self.goal_position[1] - new_y)
            old_dist = abs(self.goal_position[1] - x)
            new_dist = abs(self.goal_position[1] - new_x)
            if new_dist < old_dist:
                self.good_action_cnt += 1
                #reward += 0.5
            elif new_dist > old_dist:
                reward -= 0.5

            # Bonus if close vertically to goal
            #if abs(new_x - self.goal_position[0]) < 3:
            #if abs(new_y - self.goal_position[1]) < 3:
            #    reward += 0.5

            # Bonus for jumping over an obstacle
            if self.environment[new_position[0] - 1, new_position[1] - 1] == 255 and self.environment[new_position] == 0:
                reward += 2

        if self.good_action_cnt > 3:
            reward += 2
            if self.jump_count < 15:
                reward += 1
                self.jump_count -= 3
        if self.current_position == new_position:
            self.stuck_cnt += 1
        if self.jump_count > 15:
            reward -= 2
        if self.jump_count > 20 or self.stuck_cnt > 10:
            done = True
            reward -= 10

        # Update position
        self.current_position = new_position
        floor_height = dataset.get_env_floor_height(self.environment)
        # Gravity: apply only if not jumping (actions 0 and 1)
        if y > dataset.get_env_floor_height(self.environment) and action not in [2, 3]:
            new_y = min( new_y - 1, floor_height + 1)
            new_position = (new_x, new_y)
        #print("Trying to move to:", (new_x, new_y), "→ Cell value:", self.environment[new_x, new_y])
        if action == 1 or action == 2:
            reward -= 1  # discourage moving left (backwards)
        # Update state representation
        self.state.fill(0)
        self.state[0, floor_height, :] = self.environment[floor_height, :]
        self.state[0, self.goal_position[0], self.goal_position[1]] = 2  # goal
        self.state[0, self.start_position[0], self.start_position[1]] = 1  # start
        self.state[0, self.current_position[0], self.current_position[1]] = 3  # agent

        return self.state, reward, self.done


