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

import torch
is_cuda_available = torch.cuda.is_available()
print("Cuda available: ", is_cuda_available)
if is_cuda_available:
    print("Cuda device count: ", torch.cuda.device_count())
    print("Cuda Current device: ", torch.cuda.get_device_name(0))
    print("Threads: ", torch.get_num_threads())
    print("stable-baselines3 device:", get_device(device='auto'))

def get_one_config_icml(env_name: str):
    
    if env_name.lower() == "cartpole":
        return [IcmlCartPolePPO]
    
    if env_name.lower() == "acrobot":
        return [IcmlAcrobotPPO]
    
    if env_name.lower() == "mountaincar":
        return [IcmlMountainCarPPO]
    
    if env_name.lower() == "mountaincarcontinuous":
        return [IcmlMountainCarContinuousPPO]
    
    if env_name.lower() == "pendulum":
        return [IcmlPendulumPPO]

    if env_name.lower() == "lunarlander":
        return [IcmlLunarLanderPPO]
    
    return None

def get_one_env_icml_episodes(env_name: str):
    CartPoleEpisodesIcml = 1000
    AcrobotEpisodesIcml = 2000
    MountainCarEpisodesIcml = 3000
    MountainCarContinuousEpisodesIcml = 1000
    LunarLanderEpisodesIcml = 3000
    PendulumEpisodesIcml = 3000 

    if env_name.lower() == "cartpole":
        return [CartPoleEpisodesIcml]
        # episodes_per_env = [LunarLanderEpisodes]
    
    if env_name.lower() == "acrobot":
        return [AcrobotEpisodesIcml]
    
    if env_name.lower() == "mountaincar":
        return [MountainCarEpisodesIcml]
    
    if env_name.lower() == "mountaincarcontinuous":
        return [MountainCarContinuousEpisodesIcml]
    
    if env_name.lower() == "pendulum":
        return [PendulumEpisodesIcml]

    if env_name.lower() == "lunarlander":
        return [LunarLanderEpisodesIcml]
    
    return None

def main(run_exp_num = 10, run_icml_code = False, run_rest = True, run_env: str = "all"):
    
    CartPoleEpisodes = 6000
    CartPoleEpisodesIcml = 1000
    
    AcrobotEpisodes = 2000
    
    
    MountainCarEpisodes = 5000
    MountainCarEpisodesIcml = 3000

    MountainCarContinuousEpisodes = 1000
    
    LunarLanderEpisodes = 6000
    LunarLanderEpisodesIcml = 6000
    
    PendulumEpisodes = 6000
    PendulumEpisodesIcml = 3000 
    
    episodes_per_env_imcl = []
    episodes_per_env = []
    # For icml to run one env at a time
    if run_env == "all" or run_env == None:
        episodes_per_env_imcl = [AcrobotEpisodes, CartPoleEpisodesIcml, MountainCarEpisodesIcml, MountainCarContinuousEpisodes, PendulumEpisodesIcml, LunarLanderEpisodesIcml] 
        # episodes_per_env = [AcrobotEpisodes, CartPoleEpisodes, MountainCarEpisodes, MountainCarContinuousEpisodes, PendulumEpisodes, LunarLanderEpisodes]
        episodes_per_env = [MountainCarContinuousEpisodes, PendulumEpisodes]
        ppo_configs = [IcmlAcrobotPPO, IcmlCartPolePPO, IcmlMountainCarPPO, IcmlMountainCarContinuousPPO, IcmlPendulumPPO, IcmlLunarLanderPPO]
    else: 
        # For icml 
        episodes_per_env_imcl = get_one_env_icml_episodes(run_env)
        ppo_configs = get_one_config_icml(run_env)
    
    print("run_env: ", run_env)
    # create seeds
    # seeds = random.sample(range(1000), run_exp_num)
    
    # Seeds for ICML
    seeds_icml = [237, 379, 482, 672, 886]
    # Seeds for TC CATRL and BinQ
    seeds =  [224, 389, 405, 432, 521, 580, 639, 673, 803, 869]

    
    if run_icml_code:

        print('\n' +'{:_^40}'.format("Running Icml PPO") + '\n')
        for ppo_config, episodes in zip(ppo_configs, episodes_per_env_imcl):

            print("Running: ", ppo_config['gym_name'], " for ", episodes, "total episodes")
            ppo_config['episode_max'] = episodes
            ppo_config['debug'] = False
            for i in tqdm(range(run_exp_num)):
                run_icml(ppo_config, seed=seeds_icml[i], verbose=False)
        
        # print('\n' +'{:_^40}'.format("Running Icml MAC") + '\n')
        # mac_configs = [ IcmlAcrobotMAC, IcmlCartPoleMAC, IcmlMountainCarMAC, IcmlMountainCarContinuousMAC, IcmlLunarLanderMAC, IcmlPendulumMAC]
        # for mac_config, episodes in zip(mac_configs, episodes_per_env):

        #     print("Running ", mac_config['gym_name'])
        #     mac_config['debug'] = False
        #     mac_config['episode_max'] = episodes
        #     for i in tqdm(range(run_exp_num)):
        #         run_icml(mac_config, seed=seeds[i], verbose=False)


    if run_rest:
        # CATRL
        # configs = [CATRLAcrobot, CATRLCartPole, CATRLMountainCar, CATRLMountainCarContinuous, CATRLPendulum, CATRLLunarLander]
        configs = [CATRLMountainCarContinuous, CATRLPendulum]

        print('\n' + '{:_^40}'.format("Running CAT-RL") + '\n')

        for config, episodes in zip(configs, episodes_per_env):

            print("Running ", config['map_name'])

            env = config['env']
            config['episode_max'] = episodes
            for i in tqdm(range(run_exp_num)):
                run_CATRL(config, seed=seeds[i], verbose=False)

        # Bin Q Learning
        # configs = [BinAcrobot, BinCartPole, BinMountainCar, BinMountainCarContinuous, BinPendulum, BinLunarLander]
        configs = [BinMountainCarContinuous, BinPendulum]


        print('\n' + '{:_^40}'.format("Running bins") + '\n')

        for config, episodes in zip(configs, episodes_per_env):
            print("Running ", config['map_name'])

            env = config['env']

            for i in tqdm(range(run_exp_num)):
                agent = BinQLearningAgent(env._env, config["bins"], config["alpha"], config["gamma"], config["epsilon"], config["decay"], config["eps_min"], seed=seeds[i], verbose=False)

                run_binQ(env, agent, episodes, config['map_name'], seed=seeds[i], verbose=False)

        
        # Tile Coding
        # configs = [TileAcrobot, TileCartPole, TileMountainCar, TileMountainCarContinuous, TilePendulum, TileLunarLander]
        configs = [TileMountainCarContinuous, TilePendulum]

        print('\n' + '{:_^40}'.format("Running Tile Coding") + '\n')
        for config, episodes in zip(configs, episodes_per_env):
            print("Running ", config['name'])

            env = config['env']
            tiling_specs = config['tiling_specs']

            for i in tqdm(range(run_exp_num)):
                agent = TileCodingAgent((env._action_space.n, env._env.observation_space.low, env._env.observation_space.high), tiling_specs, verbose=False)
                run_tileCoding(env, agent, episodes, config['map_name'], seed=seeds[i], verbose=False)
    
if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser(description='Run experiments')
    
    parser.add_argument('-n', '--num', type=int, default=20, help='Number of experiments to run')
    parser.add_argument('-icml', '--icml', default='f', choices=['f', 't'], help='Run ICML experiments')
    parser.add_argument('-r', '--rest', default='t', choices=['f', 't'], help='Run rest of the experiments')
    parser.add_argument('-e', '--env', default='all', type=str, help='Run experiments for specific environment')
    
    args = parser.parse_args()

    main(
        run_exp_num=args.num,
        run_icml_code=args.icml == 't',
        run_rest=args.rest == 't',
        run_env=args.env)