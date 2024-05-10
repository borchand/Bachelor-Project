# Disabling warnings. This only done because of the use of tqdm in the code.
# To see warnings, run the individual files
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import sys
from tqdm import tqdm
import argparse
import random
import os
import pandas as pd


sys.path.append('CATRL/')

from tileCoding import run_agent as run_tileCoding
from binQLearning import run_agent as run_binQ
from run_CATRL import main as run_CATRL
from run_icml import icml_from_config as run_icml
from stable_baselines3.common.utils import get_device

sys.path.append('Code/TileCoding/')
sys.path.append('Code/envs/')

from Code.TileCoding.config import Acrobot as TileAcrobot, CartPole as TileCartPole, MountainCar as TileMountainCar, MountainCarContinuous as TileMountainCarContinuous, LunarLander as TileLunarLander, Pendulum as TilePendulum
from Code.TileCoding.agent import Agent as TileCodingAgent

sys.path.append('Code/binQLearning/')

from Code.binQLearning.config import Acrobot as BinAcrobot, CartPole as BinCartPole, MountainCar as BinMountainCar, MountainCarContinuous as BinMountainCarContinuous, LunarLander as BinLunarLander, Pendulum as BinPendulum
from Code.binQLearning.agent import QLearningAgent as BinQLearningAgent

sys.path.append('Code/CATRL/')

from Code.CATRL.config import Acrobot as CATRLAcrobot, CartPole as CATRLCartPole, MountainCar as CATRLMountainCar, MountainCarContinuous as CATRLMountainCarContinuous, LunarLander as CATRLLunarLander, Pendulum as CATRLPendulum

sys.path.append('Code/icml/')
from Code.icml.training_config_mac import ACROBOT as IcmlAcrobotMAC, CARTPOLE as IcmlCartPoleMAC, MOUNTAIN_CAR as IcmlMountainCarMAC, MOUNTAIN_CAR_CONTINUOUS as IcmlMountainCarContinuousMAC, LUNAR_LANDER as IcmlLunarLanderMAC, PENDULUM as IcmlPendulumMAC
from Code.icml.training_config_ppo import ACROBOT as IcmlAcrobotPPO, CARTPOLE as IcmlCartPolePPO, MOUNTAIN_CAR as IcmlMountainCarPPO, MOUNTAIN_CAR_CONTINUOUS as IcmlMountainCarContinuousPPO, LUNAR_LANDER as IcmlLunarLanderPPO, PENDULUM as IcmlPendulumPPO

from Code.utils import load_model, load_abstraction

def save_log_2(log_data, agent, seed, env):
    """
    Save the log data
    """

    # create folder results if it does not exist
    if not os.path.exists("results-after-train/"):
        os.makedirs("results-after-train/")

    # create folder results/agent if it does not exist
    if not os.path.exists("results-after-train/" + agent):
        os.makedirs("results-after-train/" + agent)

    # create folder results/agent/env if it does not exist
    if not os.path.exists("results-after-train/" + agent + "/" + env):
        os.makedirs("results-after-train/" + agent + "/" + env)
    
    file_path = "results-after-train/" + agent + "/" + env + "/" + agent + "_" + str(seed) + ".csv"

    df = pd.DataFrame(log_data)

    df.to_csv(file_path)


def main():

    episodes = 1000
    seeds =  [224, 389, 405, 432, 521, 580, 639, 673, 803, 869]

    # run CAT-RL with trained models
    config = [CATRLAcrobot, CATRLCartPole, CATRLMountainCar, CATRLMountainCarContinuous, CATRLLunarLander, CATRLPendulum]

    agent_name = "CAT-RL"
    for env in config:

        print("Running CAT-RL for ", env['map_name'])

        for seed in tqdm(seeds):
            log_data = {
                "episode": [],
                "reward": [],
                "epochs": [],
                "success": []
            }

            abstract = load_abstraction(agent_name, env['map_name'], seed)
            agent = load_model(agent_name, env['map_name'], seed)
            _env = env['env']
            for j in range(episodes):
                total_reward = 0
                epochs = 0
                state = _env.reset()
                done = False
                while not done:
                    state_abs = abstract.state(state)
                    action = agent.policy(state_abs)
                    new_state, reward, done, success = _env.step(action) 
                    total_reward += reward
                    epochs += 1
                    state = new_state

                log_data["episode"].append(j)
                log_data["reward"].append(total_reward)
                log_data["epochs"].append(epochs)
                log_data["success"].append(success)
            save_log_2(log_data, "CAT-RL", seed, env['map_name'])

    # run TileCoding with trained models
    config = [TileAcrobot, TileCartPole, TileMountainCar, TileMountainCarContinuous, TileLunarLander, TilePendulum]
    agent_name = "TileCoding"
    for env in config:

        print("Running TileCoding for ", env['map_name'])

        for seed in tqdm(seeds):
            log_data = {
                "episode": [],
                "reward": [],
                "epochs": [],
                "success": []
            }

            agent = load_model(agent_name, env['map_name'], seed)
            _env = env['env']
            for j in range(episodes):
                total_reward = 0
                epochs = 0
                state = _env.reset()
                done = False
                while not done:
                    action = agent.choose_action(state)
                    new_state, reward, done, success = _env.step(action)
                    total_reward += reward
                    epochs += 1
                    state = new_state

                log_data["episode"].append(j)
                log_data["reward"].append(total_reward)
                log_data["epochs"].append(epochs)
                log_data["success"].append(success)
            save_log_2(log_data, "TileCoding", seed, env['map_name'])


    # run Bin Q Learning with trained models
    config = [BinAcrobot, BinCartPole, BinMountainCar, BinMountainCarContinuous, BinLunarLander, BinPendulum]
    agent_name = "binQ"
    for env in config:

        print("Running Bin Q Learning for ", env['map_name'])

        for seed in tqdm(seeds):
            log_data = {
                "episode": [],
                "reward": [],
                "epochs": [],
                "success": []
            }

            agent = load_model(agent_name, env['map_name'], seed)
            
            _env = env['env']
            for j in range(episodes):
                total_reward = 0
                epochs = 0
                state = _env.reset()
                done = False
                action = agent.reset_episode(state)
                while not done:
                    action = agent.act(state, mode="test")
                    new_state, reward, done, success = _env.step(action)
                    total_reward += reward
                    epochs += 1
                    state = new_state
                
                log_data["episode"].append(j)
                log_data["reward"].append(total_reward)
                log_data["epochs"].append(epochs)
                log_data["success"].append(success)
            save_log_2(log_data, "binQ", seed, env['map_name'])




if __name__ == "__main__":
    main()