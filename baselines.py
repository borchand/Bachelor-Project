import os
import gymnasium as gym

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import numpy as np
# Allow the use of `pickle.load()` when downloading model from the hub
# Please make sure that the organization from which you download can be trusted
os.environ["TRUST_REMOTE_CODE"] = "True"

# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
checkpoint = load_from_hub(
    repo_id="sb3/dqn-MountainCar-v0",
    filename="dqn-MountainCar-v0.zip",
)
eval_env = gym.make("MountainCar-v0", render_mode="human")

# load the model with the environment
model = DQN.load(checkpoint, env=eval_env)
# Evaluate the agent and watch it
# episode_reward, episode_length = evaluate_policy(
#     model, eval_env, render=True, n_eval_episodes=5, deterministic=True, warn=False, return_episode_rewards=True
# )
def sample_action_from_model(model):
    vec_env = model.get_env()
    samples = []    
    for i in range(1000):
        ## Get random sample from the env
        sample = vec_env.sample()
        action, _states = model.predict(sample, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        
    return action
print(model._sample_action(), "this is the sample")
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
# print(np.mean(episode_reward, axis=0), np.mean(episode_length))
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")