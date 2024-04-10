import numpy as np
from baseenv import BaseEnv, EnvType

class AcrobotEnv(BaseEnv):
    """
        This class defines the Acrobot environment, which is a subclass of the BaseEnv class.
        It initializes the environment, and defines the step methods.
    """
    def __init__(self, step_max, render=False):
        super().__init__(EnvType.Acrobot.value, render=render)
        self.steps = 0
        self.step_max = step_max

    def step(self, action):  
        new_state, reward, terminated, truncated, _ = super().step(action)
        new_state = np.clip(new_state, a_min=self._env.observation_space.low, a_max=self._env.observation_space.high)
        self.log_r.append(reward)

        self.steps += 1

        done = False
        success = False

        if self.steps >= self.step_max or truncated:
            done = True

            success = False
        
        if terminated:
            done = True
            success = True
            reward = 1000

            
        return self.scale_state(new_state.tolist()), reward, done, success
    
    def reset(self):
        self.steps = 0
        return super().reset()