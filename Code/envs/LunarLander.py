import numpy as np
from baseenv import BaseEnv, EnvType

class LunarLanderEnv(BaseEnv):
    """
        This class defines the LunarLander environment, which is a subclass of the BaseEnv class.
        It initializes the environment, and defines the step methods.
    """
    def __init__(self, render=False):
        super().__init__(EnvType.Lunar_Lander.value, render=render)
        self.episode_reward = 0

    def reset(self):
        self.episode_reward = 0
        return super().reset()

    def step(self, action):  
        new_state, reward, terminated, truncated, _ = super().step(action)
        new_state = np.clip(new_state, a_min=self._env.observation_space.low, a_max=self._env.observation_space.high)
        self.log_r.append(reward)

        self.episode_reward += reward

        success = self.episode_reward >= 200

        return self.scale_state(new_state.tolist()), reward, terminated or truncated, success