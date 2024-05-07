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


sys.path.append('CATRL/')

from tileCoding import run_agent as run_tileCoding
from binQLearning import run_agent as run_binQ
from run_CATRL import main as run_CATRL
from run_icml import icml_from_config as run_icml
from stable_baselines3.common.utils import get_device

sys.path.append('Code/TileCoding/')
sys.path.append('Code/envs/')

from Code.TileCoding.config import MountainCarContinuous as TileMountainCarContinuous, Pendulum as TilePendulum
from Code.TileCoding.agent import Agent as TileCodingAgent

sys.path.append('Code/binQLearning/')

from Code.binQLearning.config import MountainCarContinuous as BinMountainCarContinuous, Pendulum as BinPendulum
from Code.binQLearning.agent import QLearningAgent as BinQLearningAgent

sys.path.append('Code/CATRL/')

from Code.CATRL.config import MountainCarContinuous as CATRLMountainCarContinuous, Pendulum as CATRLPendulum

sys.path.append('Code/icml/')
from Code.icml.training_config_mac import MOUNTAIN_CAR_CONTINUOUS as IcmlMountainCarContinuousMAC, PENDULUM as IcmlPendulumMAC
from Code.icml.training_config_ppo import MOUNTAIN_CAR_CONTINUOUS as IcmlMountainCarContinuousPPO, PENDULUM as IcmlPendulumPPO

from Code.envs.MountainCarContinuous import MountainCarContinuousEnv
from Code.envs.Pendulum import PendulumEnv

import torch
is_cuda_available = torch.cuda.is_available()
print("Cuda available: ", is_cuda_available)
if is_cuda_available:
    print("Cuda device count: ", torch.cuda.device_count())
    print("Cuda Current device: ", torch.cuda.get_device_name(0))
    print("Threads: ", torch.get_num_threads())
    print("stable-baselines3 device:", get_device(device='auto'))



def main():
    k_bins = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    CartPoleEpisodes = 6000
    AcrobotEpisodes = 2000
    
    MountainCarEpisodes = 5000
    MountainCarContinuousEpisodes = 1000
    
    LunarLanderEpisodes = 6000
    PendulumEpisodes = 6000

    episodes_per_env = [AcrobotEpisodes, CartPoleEpisodes, MountainCarEpisodes, MountainCarContinuousEpisodes, LunarLanderEpisodes, PendulumEpisodes] 

    # test different bins for CATRL

    # print('\n' + '{:_^40}'.format("Running CAT-RL") + '\n')


    # configs = [CATRLMountainCarContinuous, CATRLPendulum]
    # for config, episodes in zip(configs, episodes_per_env):
    #     config['episode_max'] = episodes
    #     print("Running ", config['map_name'])

    #     for k in tqdm(k_bins):
    #         if config["map_name"] == "MountainCarContinuous-v0":
    #             config["env"] = MountainCarContinuousEnv(k_bins=k)
    #         else:
    #             config["env"] = PendulumEnv(k_bins=k)
    #         run_CATRL(config, verbose=False, model_save=False, log_folder="k_bins_results/")

    # test different bins for TileCoding

    # print('\n' + '{:_^40}'.format("Running TileCoding") + '\n')

    # configs = [TileMountainCarContinuous, TilePendulum]
    # for config, episodes in zip(configs, episodes_per_env):

    #     print("Running ", config['map_name'])
    #     tiling_specs = config['tiling_specs']

    #     for k in tqdm(k_bins):
    #         if config["map_name"] == "MountainCarContinuous-v0":
    #             config["env"] = MountainCarContinuousEnv(k_bins=k)
    #         else:
    #             config["env"] = PendulumEnv(k_bins=k)

    #         env = config['env']

    #         agent = TileCodingAgent((env._action_space.n, env._env.observation_space.low, env._env.observation_space.high), tiling_specs, verbose=False)
    #         run_tileCoding(env, agent, episodes, config['map_name'], verbose=False, model_save=False, log_folder="k_bins_results/")
    
    # test different bins for BinQ

    print('\n' + '{:_^40}'.format("Running BinQ") + '\n')

    configs = [BinMountainCarContinuous, BinPendulum]
    for config, episodes in zip(configs, episodes_per_env):

        print("Running ", config['map_name'])

        for k in tqdm(k_bins):
            if config["map_name"] == "MountainCarContinuous-v0":
                config["env"] = MountainCarContinuousEnv(k_bins=k)
            else:
                config["env"] = PendulumEnv(k_bins=k)

            env = config['env']
            
            agent = BinQLearningAgent(env._env, config["bins"], config["alpha"], config["gamma"], config["epsilon"], config["decay"], config["eps_min"], verbose=False)
            run_binQ(env, agent, episodes, config['map_name'], verbose=False, model_save=False, log_folder="k_bins_results/")

if __name__ == "__main__":
    main()