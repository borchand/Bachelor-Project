from baseenv import BaseEnv, EnvType
import numpy as np
import math

class CartPoleEnv(BaseEnv):
    """
        This class defines the Cart Pole environment, which is a subclass of the BaseEnv class.
        It initializes the environment, and defines the step methods.
    """

    def __init__(self, step_max, render=False):
        super().__init__(EnvType.Cart_Pole.value, True, render=render)
        self.steps = 0
        self.step_max = step_max

    def step(self, action):
        new_state, reward, terminated, truncated, info = super().step(action)
        new_state = np.clip(new_state, a_min=self._env.observation_space.low, a_max=self._env.observation_space.high)

        # The episode ends if any one of the following occurs:
        # Termination: Pole Angle is greater than ±12°
        # Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
        # Truncation: Episode length is greater than 500 (200 for v0)
        # So if it is truncated, we consider it a success

        self.steps += 1
        
        done = False
        success = False

        if self.steps >= self.step_max or terminated or truncated:
            done = True
            if not terminated:
                success = True
            else:
                reward = -1000

        return new_state.tolist(), reward, done, success
    
    def reset(self):
        self.steps = 0
        return super().reset()