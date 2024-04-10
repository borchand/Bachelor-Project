import sys
import numpy as np
from baseenv import ContinuousActionBaseEnv, EnvType

class MountainCarContinuousEnv(ContinuousActionBaseEnv):
    """
        This class defines the Mountain Car Continuous environment, which is a subclass of the BaseEnv class.
        It initializes the environment, and defines the step methods.
    """
    def __init__(self, render=False):
        super().__init__(
            EnvType.Mountain_Car_Continuous.value,
            2,
            render=render)

    def step(self, action):  
        new_state, reward, terminated, truncated, _ = super().step(action)
        new_state = np.clip(new_state, a_min=self._env.observation_space.low, a_max=self._env.observation_space.high)
        self.log_r.append(reward)
        success = False
        done = False

        if terminated:
            success = True
            done = True

        if truncated:
            success = False
            done = True

        return self.scale_state(new_state.tolist()), reward, done, success