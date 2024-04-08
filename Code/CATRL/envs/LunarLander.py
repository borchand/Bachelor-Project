import numpy as np
from baseenv import BaseEnv, EnvType

class LunarLanderEnv(BaseEnv):
    """
        This class defines the Mountain Car environment, which is a subclass of the BaseEnv class.
        It initializes the environment, and defines the step methods.
    """
    def __init__(self, render=False):
        super().__init__(EnvType.Lunar_Lander.value, 1, render=render)

    def step(self, action):  
        new_state, reward, terminated, _, _ = super().step(action)
        new_state = np.clip(new_state, a_min=self._env.observation_space.low, a_max=self._env.observation_space.high)
        self.log_r.append(reward)

        success = not self._env.lander.awake
            
        return self.scale_state(new_state.tolist()), reward, terminated, success