import numpy as np
import math
import gym
from enum import Enum

class EnvType(Enum):
    Mountain_Car = "MountainCar-v0"
    Cart_Pole = "CartPole-v1"
    Acrobot = "Acrobot-v1"
    Lunar_Lander = "LunarLander-v2"
    Car_Racing = "CarRacing-v2"

class BaseEnv:
    """
    This class defines a base environment with methods for initializing, seeding, stepping, and
    resetting the environment, as well as scaling states and converting between action indices and
    actions.

    :param env_type: The type of environment to use.
    :type env_type: EnvType
    :param state_size: The size of the state space.
    :type state_size: int
    :param use_gran: Whether to use granularity in the state space.
    :type use_gran: bool
    """
    def __init__(
            self,
            env_type : str,
            state_size : int = 1,
            use_gran : bool = True,
            render : bool = False):
        
        self._state_size = state_size

        if render:
            self._env = gym.make (env_type, render_mode='human')
        else:
            self._env = gym.make (env_type)
        self._action_space = self._env.action_space
        self._action_size = self._action_space.n
        self._n_state_variables = self._env.observation_space.shape[0]

        self._original_state_ranges = self._env.observation_space.low, self._env.observation_space.high
        
        self._gran = 0.001
        self._state_ranges = []

        if use_gran:
            for i in range (self._n_state_variables):
                low = math.floor(self._original_state_ranges[0][i] * 1/self._gran) 
                high = math.ceil(self._original_state_ranges[1][i] * 1/self._gran) + 1
                r = (low, high)
                self._state_ranges.append(r)
        else:
            for i in range (self._n_state_variables):
                low = self._original_state_ranges[0][i] 
                high = self._original_state_ranges[1][i] + 1
                r = (low, high)
                self._state_ranges.append(r)

        self._vars_split_allowed = [1 for i in range(len(self._state_ranges))]
        self.log_r = []
        
    def seed(self, seed):
        self._env.seed(seed)
    
    def step(self, action):
        return self._env.step(action)
    
    def reset(self):
        return self._env.reset()
    
    def reset (self):
        start_state = self._env.reset()
        return self.scale_state(start_state[0].tolist())

    def scale_state (self, state):
        for i in range (len(state)):
            state[i] = math.ceil(state[i] * 1/self._gran)
        return state

    # action_index into action
    def index_to_action (self, action_index):
        return self._action_space [action_index]

    # state to state_index
    def state_to_index (self, state):
        return state[0]*self._dimension[0] + state[1]
    
    def render(self):
        self._env.render()
