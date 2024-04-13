import os
import gymnasium as gym

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import numpy as np
import argparse
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

def main(env_name, algo_name, timesteps: int, render = False, save = True, train = True):
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
    save_name = "rl-trained-agents/"+ algo_name + "_" + env_name
    env = gym.make(env_name)

    # get the model class from the algo name
    model_class = _get_model_class(algo_name)

    # create the model
    model = model_class("MlpPolicy", env=env, verbose=1)

    # model learn the env
    if train:
        model.learn(total_timesteps=timesteps, log_interval=10)
    # save the model
    
    if save:
        model.save(save_name)


    if render:
        eval_env = gym.make(env_name, render_mode='human')
        
        model = model.load(save_name, env=eval_env)

        model.set_env(eval_env)
        obs, info = eval_env.reset()
        
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
    parser.add_argument('-a', '--algo', default='dqn', help='Algorithm to use when training')
    # parser.add_argument('-ep', '--episodes', default=100, help='Number of episodes to train the model for', type=int)
    parser.add_argument('-t', '--time-steps', default=None, help='Number of time steps to train the model for', type=int)
    parser.add_argument('-tr', '--train', choices=['t', 'f'], default='t', help='Train the model')
    
    parser.add_argument('-r', '--render', choices=['t', 'f'], default='f', help='Render the model')
    parser.add_argument('-s', '--save', choices=['t', 'f'], default='t', help='Save the model')
    
    parser.add_argument('-sh', '--show', choices=['t', 'f'], default='f', help='Show the model')

    args = parser.parse_args()
    
    # Render the model
    if args.show == 't':
        main(
            env_name=args.env,
            algo_name=args.algo,
            timesteps=0,
            render=True,
            save=False,
            train=False)
    
    # Else just train the model
    main(
        env_name=args.env,
        algo_name=args.algo,
        timesteps=args.time_steps,
        render=args.render == 't',
        save=args.save == 't',
        train=args.train == 't'
        )