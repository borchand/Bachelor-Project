import os
import gymnasium as gym
from icml_2019_state_abstraction.mac.ActionWrapper import discretizing_wrapper
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, BaseCallback, CallbackList, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
import pandas as pd
import numpy as np
import argparse
import time
import random
# Allow the use of `pickle.load()` when downloading model from the hub
# Please make sure that the organization from which you download can be trusted
os.environ["TRUST_REMOTE_CODE"] = "True"

# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
# checkpoint = load_from_hub(
#     repo_id="sb3/dqn-MountainCar-v0",
#     filename="dqn-MountainCar-v0.zip",
# )
def _get_model_class(algo_name):
    if algo_name == 'ppo':
        return PPO
    elif algo_name == 'a2c':
        return A2C
    elif algo_name == 'dqn':
        return DQN
    elif algo_name == 'sac':
        return SAC
    elif algo_name == 'td3':
        return TD3
    elif algo_name == 'ddpg':
        return DDPG
    else:
        raise ValueError('Invalid algorithm name')
def get_save_name(env_name: str, algo_name: str, episodes: int, seed: int , k=None, verbose: bool = False):
    """
    Args:
        :param algo_name (str): Name of the algorithm
        :param env_name (str): Name of the environment
        :param timesteps (int): Number of timesteps to train the model
        :param k = 20 (int): Number of bins to discretize the action space
    Returns:
        save_name (str): Name of the file to save the model as
    """
    continuous_action_envs = ["MountainCarContinuous-v0", "Pendulum-v1", "Pendulum-v0", "LunarLanderContinuous-v2"]
    save_name = "rl-trained-agents/"+ str(episodes) + '/' 
    
    # If the action space is discretized, add the number of bins to the save name
    if k > 1 and env_name in continuous_action_envs:
        save_name += str(k) + "_"
    
    save_name += algo_name + "_" + env_name + "_" + str(seed)
    if verbose:
        print("this is the save name", save_name)
    return save_name

def get_vec_env(env_name, n_envs=8, k=1):

    wrapper_kwargs = {'k': k}
    vec_env = make_vec_env(env_id=env_name, n_envs=n_envs, wrapper_class=RewardShapingWrapper, wrapper_kwargs=wrapper_kwargs)
    return vec_env

def get_gym_env(env_name, render=False, k=1):
    """
    Args:
        :param env_name (str): Name of the environment
        :param render = True (bool): If True, render the environment
        :param discretetize = True (bool): If True, discretize the action space
        :param k = 20 (int): Number of bins to discretize the action space
    Returns:
        gym_env (gym.Env): Gym environment
    """
    gym_env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)

    if isinstance(gym_env.action_space, gym.spaces.Box): 
        gym_env = discretizing_wrapper(gym_env, k)

    return gym_env

class RewardShapingWrapper(gym.Wrapper):
    """
    Callback for logging the reward at each timestep
    """
    def __init__(self, env, k):
        if isinstance(env.action_space, gym.spaces.Box): 
            env = discretizing_wrapper(env, k)

        super().__init__(env)
        self.reward_shaping = self.get_reward_shaping()
        self.reward_shaping_end = self.get_reward_shaping_end()
    
    def reset(self, **kwargs):
        """
        Reset the environment
        """
        obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is this a final state (episode finished),
        is the max number of steps reached (episode finished artificially), additional informations
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.reward_shaping is not None:
            reward += self.reward_shaping(obs)
        
        if (terminated or truncated) and self.reward_shaping_end is not None:
            reward = self.reward_shaping_end(obs, terminated, truncated)

        return obs, reward, terminated, truncated, info
    
    def get_reward_shaping(self):
        """
        Args:
            state (np.ndarray): Current state
            reward (float): Reward from the environment
        Returns:
            reward (float): Reward after reward shaping
        """
        if self.env.unwrapped.spec.id == "MountainCar-v0":
            return self._MountainCar_reward_shaping
        
        if self.env.unwrapped.spec.id == "MountainCarContinuous-v0":
            return self._MountainCar_reward_shaping
        
        if self.env.unwrapped.spec.id == "Penulum-v1":
            return self._Pendulum_reward_shaping
        
        return None
    
    def get_reward_shaping_end(self):
        
        if self.env.unwrapped.spec.id == "MountainCar-v0":
            return self._MountainCar_reward_shaping_end
        
        if self.env.unwrapped.spec.id == "MountainCarContinuous-v0":
            return self._MountainCar_reward_shaping_end

        if self.env.unwrapped.spec.id == "Pendulum-v1":
            return self._Pendulum_reward_shaping_end
        
        if self.env.unwrapped.spec.id == "Acrobot-v1":
            return self._Acrobot_reward_shaping_end
        
        return None

    def _Acrobot_reward_shaping_end(self, state, terminated, truncated):
        if terminated:
            return 1000
        return 0
    
    def _Pendulum_reward_shaping(self, state):
        x, y, velocity = state
        if abs(y) < 0.2 and x > 0 and abs(velocity) < 0.2:
                return 10
        return 0

    def _Pendulum_reward_shaping_end(self, state, terminated, truncated):
        if abs(state[1]) < 0.1 and abs(state[2]) < 0.1:
            return 1000
        return 0
    
    def _MountainCar_reward_shaping(self, state):
        return 100*state[1] 
    
    def _MountainCar_reward_shaping_end(self, state, terminated, truncated):
        if terminated:
            return 1000
        return 0
    
    def _CartPole_reward_shaping_end(self, state, terminated, truncated):
        if terminated:
            return -1000
        return 1
    
def get_logger():
    """
    Get the logger
    """
    temp_dir = "logs/sb3/"
    return configure(temp_dir, ["csv"])

def from_config(config: dict, seed=None, verbose=False, time_limit_sec=None):
    
    main(
        env_name=config['env_name'],
        algo_name=config['algo'],
        episodes=config['policy_episodes'],
        k=config['k_bins'],
        render=config['render_policy'],
        save=config['save'],
        train=config['train'],
        debug=config['debug'],
        verbose=verbose,
        seed=seed)

def main(env_name: str, algo_name: str, episodes: int, k: int, seed: int, render=False, save=True, train=True, verbose=False, debug=False):
    """
    Args:
        :param env_name (str): Name of the environment
        :param algo_name (str): Name of the algorithm
            Choices: { 'ppo', 'a2c', 'dqn', 'sac', 'td3', 'ddpg' }
        :param timesteps (int): Number of timesteps to train the model
        :param render = False (bool): If True, render the environment
        :param save = True (bool): If True, save the trained model
        :param train = True (bool): If True, train the model
    """
    if debug == True:
        episodes = 3

    vec_env = get_vec_env(env_name, n_envs=8, k=k)
    
    env = get_gym_env(env_name, render=False, k=k)
    save_name = get_save_name(
        env_name=env_name,
        algo_name=algo_name,
        episodes=episodes,
        seed=seed,
        verbose=verbose,
        k=k)
    
    # env = RewardShapingWrapper(env)
    ## set random seed
    random.seed(seed)
    set_random_seed(seed)

    # get the model class from the algo name
    model_class = _get_model_class(algo_name)

    # create the model
    model = model_class("MlpPolicy", env=vec_env, verbose=1)
    logger = get_logger()
    model.set_logger(logger)
    
    if train:
        # callback to stop training after max episodes
        callback_progress_bar = ProgressBarCallback()
        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=episodes, verbose=1)
        callback_list = CallbackList([callback_max_episodes, callback_progress_bar])
        # get max number of steps in a episode
        max_timesteps = env._max_episode_steps * episodes
        
        # model learn the env
        if render:
            log_interval = 1
        state_time = time.time()
        model.learn(total_timesteps=max_timesteps, log_interval=5, callback=callback_list)
        end_time = time.time()

        training_time = end_time - state_time
        # This is the number o
        timesteps = model.num_timesteps
    # save the model
    if save and train:
        
        model.save(save_name)
        with open(save_name + "_time.txt", 'w') as f:
            f.write(str(training_time))
        

    if render:
        env = get_gym_env(env_name, render=True, k=k)
        model = model.load(save_name, env=env)

        obs, info = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                break
            env.render()

def show_model(env_name, algo_name, episodes=200, k=None):

    eval_env = gym.make(env_name, render_mode='human')
    save_name = get_save_name(env_name, algo_name, episodes, k=k)
    
    model = _get_model_class(algo_name)
    model = model.load(save_name, env=eval_env)

    obs, info = eval_env.reset()
    print("this is the obs", obs)
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            obs, info = eval_env.reset()
            break
        eval_env.render()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Set options for training and rendering CAT_RL')
    
    parser.add_argument('-e', '--env', default='CartPole-v1', help='Environment to train on')
    parser.add_argument('-a', '--algo', default='ppo', help='Algorithm to use when training')
    
    parser.add_argument('-ep', '--episodes', default=100, help='Number of episodes to train the model for', type=int)
    parser.add_argument('-t', '--time-steps', default=None, help='Number of time steps to train the model for', type=int)
    parser.add_argument('-k', '--k-bins',  default=1 , help='Discretize the action space into k bins', type=int)
    parser.add_argument('-seed', '--seed', default=42, help='Seed for reproducibility', type=int)

    parser.add_argument('-tr', '--train', choices=['t', 'f'], default='t', help='Train the model')
    parser.add_argument('-r', '--render', choices=['t', 'f'], default='t', help='Render the model')
    parser.add_argument('-s', '--save', choices=['t', 'f'], default='t', help='Save the model')
    
    parser.add_argument('-sh', '--show', choices=['t', 'f'], default='f', help='Show the model')

    args = parser.parse_args()
    
    # Render the model
    if args.show == 't':
        show_model(args.env, args.algo, episodes=args.episodes, k=args.k_bins)
        
    else:
        # Else just train the model
        main(
            env_name=args.env,
            algo_name=args.algo,
            episodes=args.episodes,
            render=args.render == 't',
            seed=args.seed,
            save=args.save == 't',
            train=args.train == 't',
            k=args.k_bins)