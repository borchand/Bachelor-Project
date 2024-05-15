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

## icml stuff 
from icml_2019_state_abstraction.experiments.simple_rl.abstraction import AbstractionWrapper
from icml_2019_state_abstraction.experiments.simple_rl.agents import QLearningAgent as QLearningAgentIcml
from icml_2019_state_abstraction.experiments.run_learning_experiment import get_policy, load_agent_pytorch, Get_GymMDP, run_one_episode_q_function

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


def load_icml_q_learning(config: dict, gym_env, env_name, seed) -> tuple:
    
    k_bins = config['k_bins']
    algo = config['algo']
    policy_train_episodes = config['policy_episodes']
    experiment_episodes = config['experiment_episodes']
    
    print("Before get policy")
    policy = get_policy(gym_env, algo, policy_train_episodes, experiment_episodes, k_bins, seed)
    print("after get policy")
    actions = gym_env.get_actions()

    name_ext = "_phi_" + str(k_bins) + "_" + str(algo) + "_" + str(seed) if k_bins > 1 else "_phi_" + str(algo) + "_" + str(seed) 
    load_agent_path = "models/icml/" + env_name + "/" + str(experiment_episodes) + "/" "Q-learning" + name_ext
    
    abstraction_network, _ = load_agent_pytorch(policy, verbose=True)

    agent_params = {"alpha":0.0,"epsilon":0.1,"actions":actions,"load": True ,"load_path": load_agent_path}
    # include k_bins if the action space is discretized
    sa_agent = AbstractionWrapper(QLearningAgentIcml,
                                agent_params=agent_params,
                                state_abstr=abstraction_network,
                                name_ext=name_ext)
    
    return sa_agent 

def main():

    episodes = 1000
    seeds =  [224, 389, 405, 432, 521, 580, 639, 673, 803, 869]
    seeds_icml = [237, 379, 482, 672, 886]

    config = [IcmlAcrobotPPO, IcmlCartPolePPO, IcmlMountainCarPPO, IcmlMountainCarContinuousPPO, IcmlPendulumPPO, IcmlLunarLanderPPO]

    for env in config:

        print("Running PPO for ", env['gym_name'])

        for seed in tqdm(seeds_icml):
            log_data = {
                "episode": [],
                "reward": [],
                "epochs": [],
                "success": []
            }

            gym_env = Get_GymMDP(env['gym_name'], k=env['k_bins'], seed=seed)
            print("Loaded gym")
            sa_agent = load_icml_q_learning(config=env, gym_env=gym_env, env_name=env["gym_name"], seed=seed)
            print("Loaded abstract agent")
            # get env from 
            for episode in range(episodes):
                print("Running episode ", episode, " for ", env['gym_name'], " seed ", seed)
                total_reward, step, success = run_one_episode_q_function(gym_env=gym_env, sa_agent=sa_agent)
                
                log_data["episode"].append(episode)
                log_data["reward"].append(total_reward)
                log_data["epochs"].append(step)
                log_data["success"].append(success)

            save_log_2(log_data, "ppo", seed, env['gym_name'])
        print(f"Completed evironment {env['gym_name']} {seed}")

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

    # # run TileCoding with trained models
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