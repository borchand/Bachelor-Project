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

def main(env_name, algo_name, render_mode, timesteps):
    
    env = gym.make(env_name, render_mode = "rgb_array")

    # get the model class from the algo name
    model_class = _get_model_class(algo_name)

    # create the model
    model = model_class("MlpPolicy", env=env, verbose=1)

    # model learn the env
    model.learn(total_timesteps=timesteps, log_interval=4)
    
    # save the model
    save_name = "rl-trained-agents/"+ algo_name + "_" + env_name
    model.save(save_name)

    # delete trained model to demonstrate loading
    # del model 

    # load the model
    # model = model_class.load(save_name)

    # obs, info = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         obs, info = env.reset()
    #         break
    #     env.render()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Set options for training and rendering CAT_RL')
    
    parser.add_argument('-e', '--env', default='CartPole-v1', help='Environment to train on')
    parser.add_argument('-a', '--algo', default='a2c', help='Algorithm to use')
    parser.add_argument('-t', '--time-steps', default=2000, help='Number of time steps to train the model for', type=int)
    parser.add_argument('-r', '--render', choices=['t', 'f'], default='t', help='Render the model')
    
    args = parser.parse_args()
    
    main(args.env, args.algo, args.render == 't', args.time_steps)