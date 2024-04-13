import numpy as np
import math
import gymnasium as gym
from enum import Enum
from gymnasium import spaces

class EnvType(Enum):
    Mountain_Car = "MountainCar-v0"
    Mountain_Car_Continuous = "MountainCarContinuous-v0"
    Cart_Pole = "CartPole-v1"
    Acrobot = "Acrobot-v1"
    Lunar_Lander = "LunarLander-v2"
    Pendulum = "Pendulum-v1"

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
            use_gran : bool = True,
            render : bool = False):
        

        if render:
            self._env = gym.make (env_type, render_mode='human')
        else:
            self._env = gym.make (env_type)

        self._state_size = self._env.observation_space.shape[0]
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

class ContinuousActionBaseEnv():
    """
    This class defines a base environment for continuous action spaces with methods for initializing, seeding, stepping, and 
    resetting the environment, as well as scaling states and converting between action indices and
    actions.

    :param env_type: The type of environment to use.
    :param K: The size of the action state space after discretizing.
    :type K: int
    :param use_gran: Whether to use granularity in the state space.
    :type use_gran: bool
    :param render: Whether to render the environment.
    :type render: bool
    """
    def __init__(
            self,
            env_type : str,
            K: int,
            use_gran : bool = True,
            render : bool = False):
        

        if render:
            self._env = gym.make (env_type, render_mode='human')
        else:
            self._env = gym.make (env_type)
        
        self._env = self.discretizing_wrapper(self._env, K)

        self._state_size = self._env.observation_space.shape[0]
        self._action_space = self._env.action_space
        self._action_size = K
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
    
    """
    wrapper for discretizing continuous action space
    """
    def discretizing_wrapper(self, env, K):
        """
        # discretize each action dimension to K bins
        # From https://github.com/robintyh1/onpolicybaselines/blob/master/onpolicyalgos/discrete_actions_space/acktr_discrete/wrapper.py
        # Yunhao Tang, Shipra Agrawal. "Discretizing Continuous Action Space for On-Policy Optimization"(AAAI, 2020)
        """
        unwrapped_env = env.unwrapped
        unwrapped_env.orig_step_ = unwrapped_env.step
        unwrapped_env.orig_reset_ = unwrapped_env.reset
        
        action_low, action_high = env.action_space.low, env.action_space.high
        naction = action_low.size

        action_table = np.reshape([np.linspace(action_low[i], action_high[i], K) for i in range(naction)], [naction, K])
        assert action_table.shape == (naction, K)



        # change observation space
        if naction == 1:
            env.action_space = spaces.Discrete(K)
        else:
            action_space = [K for _ in range(naction)]
            # print("This is the action space", action_space)
            a = []
            for _ in range(naction):
                a.append(K)
            action_space = spaces.MultiDiscrete(a) 
            action_space.sample()
        # env.action_space = spaces.MultiDiscrete([[1, K-1] for _ in range(naction)])

        unwrapped_env.step = self.discretizing_step
        unwrapped_env.reset = self.discretizing_reset

        self.unwrapped_env = unwrapped_env
        self.naction = naction
        self.action_table = action_table

        return env
    
    def discretizing_reset(self):
        obs = self.unwrapped_env.orig_reset_()
        return obs

    def discretizing_step(self, action):
        # action is a sequence of discrete indices
        action_cont = self.action_table[np.arange(self.naction), action]
        obs, rew, terminated, truncated, info = self.unwrapped_env.orig_step_(action_cont)
        
        return (obs, rew, terminated, truncated, info)