import os
import gymnasium as gym

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import numpy as np
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

env_name = "CartPole-v1"

env = gym.make(env_name, render_mode = "rgb_array")

algo_name = "a2c"

save_name = algo_name + "_MountainCar-v0"

# create the model with the environment
model = A2C("MlpPolicy",env=env, verbose=1)

# learn the model
env_steps = 200
episodes = 1000
total_timesteps = env_steps * episodes
model.learn(total_timesteps=total_timesteps, log_interval=4)
# save the model
model.save(save_name)

del model # delete trained model to demonstrate loading

# load the model
model = A2C.load(save_name)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
        break
    env.render()