import gym
from gym.spaces import Discrete
from gym.spaces import Box
import numpy as np

class DeepSeaTreasureEnv(gym.Env):
    '''
        This is an implementation of a standard Multiple Objective Optimization problem called
        Deep Sea Treasure Hunt (Vamplew, et al, 2011).
        In this environment there are multiple goals. 1) Search for as much treasure as possible 2) Take as little
        time as possible. There is no one optimal policy to solve this but a collection of policies which are
        equally optimal.
    '''
    def __init__(self, max_steps=30):

        #limit on time spent seeking treasure
        self.max_steps = max_steps

        self.scale_time = 0.01
        self.scale_treasure = 1.

        #Grid world
        # values:
        #  -- 0     = sea
        #  -- -1    = sea floor
        #  -- > 0   = Treasure
        self.grid = np.zeros((10,11), dtype=int)
        # sea floor
        self.grid[0, 2] = -1
        self.grid[0:1, 3] = -1
        self.grid[0:2, 4] = -1
        self.grid[0:5, 5] = -1
        self.grid[0:5, 6] = -1
        self.grid[0:5, 7] = -1
        self.grid[0:7, 8] = -1
        self.grid[0:7, 9] = -1
        self.grid[0:8, 10] = -1
        #treasure
        self.grid[0, 1] = 1
        self.grid[1, 2] = 2
        self.grid[2, 3] = 3
        self.grid[3, 4] = 5
        self.grid[4, 4] = 8
        self.grid[5, 4] = 16
        self.grid[6, 7] = 24
        self.grid[7, 7] = 50
        self.grid[8, 8] = 74
        self.grid[9, 10] = 124

        self.position = 0
        self.steps_taken = 0
        self.treasure_value = 0
        self.time_penalty = 0

        #actions:
        # 0 - Up
        # 1 - Down
        # 2 - Left
        # 3 - Right
        self.action_space = Discrete(4)
        # The agent observes:
        # - It's own position
        # - The treasure value
        # - The time penalty
        self.observation_space = Box(0,200, (3,))

    def move(self, action):
        row = self.position // 10
        col = self.position % 10

        if action == 0: # Up
            if row > 0:
                row -= 1
        if action == 1: # Down
            if row < 10:
                row += 1
        if action == 2: # Left
            if col > 0:
                col -= 1
        if action == 3: # Right
            if col < 9:
                col += 1

        #account for time spent (even doing illegal moves)
        if self.grid[col, row] == -1:
            self.steps_taken += 1
            self.time_penalty += 1
            return
        else:
            self.position = row * 10 + col
            self.steps_taken += 1
            self.time_penalty += 1
            self.treasure_value = self.grid[self.position % 10, self.position // 10]


    def close(self):
        return

    def reset(self):
        self.position = 0
        self.steps_taken = 0
        self.time_penalty = 0
        self.treasure_value = 0
        obs = np.array([self.position, self.time_penalty, self.treasure_value])
        return obs

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        obs = np.zeros(1, dtype=int)
        reward = 0
        done = False
        info = {}

        self.move(action)

        obs = np.array([self.position, self.time_penalty, self.treasure_value])

        reward = -self.time_penalty, self.treasure_value

        # reset time penalty after treasure find
        if (self.treasure_value > 0):
            self.time_penalty = 0
        #     reward = self.scale_treasure * self.treasure_value - self.scale_time * self.time_spent

        if (self.steps_taken >= self.max_steps):
            done = True

        if(self.treasure_value > 0):
            done = True
            # reward = - self.scale_time * self.time_spent

        # reward = -self.time_spent + self.treasure_value
        # done = (self.treasure_value > 0) or (self.time_spent == self.max_steps)

        return obs, reward, done, info