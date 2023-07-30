from typing import Tuple, Union, Dict

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box


class SafeGridWorld(Env):
    def __init__(self, grid_size=4):
        super(SafeGridWorld, self).__init__()

        self.grid_size = grid_size
        self.goal_cell = np.array([grid_size - 1, grid_size - 1])
        self.agent_position = np.array([0., 0.])

        # Define continuous action space as 2D box with range [-1, 1]
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Define continuous observation space with four features (agent x/y, dist to nearest bomb, dist to goal)
        self.observation_space = Box(low=0, high=np.linalg.norm(self.goal_cell - self.agent_position),
                                     shape=(4,), dtype=np.float32)

        self.reward_space = Box(low=-1.0, high=0.0, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self, **kwargs):
        # Initialize agent's position randomly on the grid
        self.set_bombs()
        self.agent_position = np.array([0., 0.])
        features = [self.get_nearest_bomb(self.agent_position), self.dist_to_goal()]
        return np.concatenate([self.agent_position, features], axis=0), dict()

    def get_nearest_bomb(self, position):
        return np.min([np.linalg.norm(position - bomb) for bomb in self.bombs])

    def dist_to_goal(self):
        return np.linalg.norm(self.goal_cell - self.agent_position)

    def step(self, action):
        # Move the agent according to the continuous action
        self.agent_position = np.clip((self.agent_position + action), 0., self.grid_size - 1)

        # Calculate reward and cost based on agent's new position
        score = -1.  # Cost for each step
        constraint_violation = 0.
        done = False

        nearest_bomb = self.get_nearest_bomb(self.agent_position)

        if np.linalg.norm(self.goal_cell - self.agent_position) <= 1.0:
            score = 0.  # Agent reached the goal, reward is 0
            done = True
        elif nearest_bomb <= 0.5:
            constraint_violation = 1.  # Agent stepped on a bomb, additional safety cost

        reward = np.array([score, constraint_violation])
        info = dict()

        return np.concatenate([self.agent_position, [nearest_bomb, self.dist_to_goal()]], axis=0), reward, done, info

    def render(self):
        # Render the current state of the grid world
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[int(self.agent_position[0, 0]), int(self.agent_position[0, 1])] = 1
        grid[int(self.goal_cell[0, 0]), int(self.goal_cell[0, 1])] = 2
        for bomb in self.bombs:
            grid[bomb[0], bomb[1]] = -1

    def set_bombs(self):
        # Place bombs randomly on the grid
        # all_positions = [(i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])]
        # self.bombs = set(np.random.choice(all_positions, self.num_bombs, replace=False))
        self.bombs = list(map(lambda t: np.array(t), [(x, y)
                                                      for x in range(1, self.grid_size - 1)
                                                      for y in range(1, self.grid_size - 1)]))
