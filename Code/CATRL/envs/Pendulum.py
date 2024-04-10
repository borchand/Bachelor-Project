import sys
import numpy as np
from baseenv import ContinuousActionBaseEnv, EnvType

class PendulumEnv(ContinuousActionBaseEnv):
    """
        This class defines the Mountain Car Continuous environment, which is a subclass of the BaseEnv class.
        It initializes the environment, and defines the step methods.
    """
    def __init__(self, render=False):
        super().__init__(
            EnvType.Pendulum.value,
            40,
            render=render)

    def step(self, action):  
        new_state, reward, terminated, truncated, info = super().step(action)
        new_state = np.clip(new_state, a_min=self._env.observation_space.low, a_max=self._env.observation_space.high)
        self.log_r.append(reward)
        success = False
        done = False

        x, y, velocity = new_state
        
        if truncated:
            done = True

            # success if the pendulum is upright or close to upright and velocity is low
            if abs(y) < 0.1 and abs(velocity) < 0.1:
                success = True
                reward += 1000

        return self.scale_state(new_state.tolist()), reward, done, success