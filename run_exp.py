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
from run_icml import main_with_config as run_icml

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

def main(run_exp_num = 20, verbose = False):
    
    CartPoleEpisodes = 500
    AcrobotEpisodes = 1000
    
    MountainCarEpisodes = 1400
    MountainCarContinuousEpisodes = 600
    
    LunarLanderEpisodes = 1000
    PendulumEpisodes = 1000

    episodes_per_env = [AcrobotEpisodes, CartPoleEpisodes, MountainCarEpisodes, MountainCarContinuousEpisodes, LunarLanderEpisodes, PendulumEpisodes] 

    # create seeds
    seeds = random.sample(range(1000), run_exp_num)

    # CATRL
    congifs = [CATRLAcrobot, CATRLCartPole, CATRLMountainCar, CATRLMountainCarContinuous, CATRLPendulum]
    

    print('\n' +'{:_^40}'.format("Running CAT-RL") + '\n')

    for config, episodes in zip(congifs, episodes_per_env):

        print("Running ", config['map_name'])

        env = config['env']
        config['episode_max'] = episodes
        for i in tqdm(range(run_exp_num)):
            run_CATRL(config, seed=seeds[i], verbose=False)

    # Bin Q Learning
    congifs = [BinAcrobot, BinCartPole, BinMountainCar, BinMountainCarContinuous, BinPendulum]


    print('\n' + '{:_^40}'.format("Running bins") + '\n')

    for config, episodes in zip(congifs, episodes_per_env):
        print("Running ", config['map_name'])

        env = config['env']

        for i in tqdm(range(run_exp_num)):
            agent = BinQLearningAgent(env._env, config["bins"], config["alpha"], config["gamma"], config["epsilon"], config["decay"], config["eps_min"], seed=seeds[i], verbose=False)

            run_binQ(env, agent, episodes, config['map_name'], seed=seeds[i], verbose=False)


    
    
    # Tile Coding
    congifs = [TileAcrobot, TileCartPole, TileMountainCar, TileMountainCarContinuous, TilePendulum]

    print('\n' + '{:_^40}'.format("Running Tile Coding") + '\n')
    for config, episodes in zip(congifs, episodes_per_env):
        print("Running ", config['name'])

        env = config['env']
        tiling_specs = config['tiling_specs']

        for i in tqdm(range(run_exp_num)):
            agent = TileCodingAgent((env._action_space.n, env._env.observation_space.low, env._env.observation_space.high), tiling_specs, verbose=False)
            run_tileCoding(env, agent, episodes, config['map_name'], seed=seeds[i], verbose=False)


    print('\n' +'{:_^40}'.format("Running Icml PPO") + '\n')
    ppo_configs = [ IcmlAcrobotPPO, IcmlCartPolePPO, IcmlMountainCarPPO, IcmlMountainCarContinuousPPO, IcmlPendulumPPO]
    for ppo_config, episodes in zip(ppo_configs, episodes_per_env):

        print("Running ", ppo_config['gym_name'])

        ppo_config['episode_max'] = episodes
        for i in tqdm(range(run_exp_num)):
            run_icml(ppo_config, seed=seeds[i], verbose=False)

    
    print('\n' +'{:_^40}'.format("Running Icml MAC") + '\n')
    mac_configs = [ IcmlAcrobotMAC, IcmlCartPoleMAC, IcmlMountainCarMAC, IcmlMountainCarContinuousMAC, IcmlPendulumMAC]
    for mac_config, episodes in zip(mac_configs, episodes_per_env):

        print("Running ", mac_config['gym_name'])

        mac_config['episode_max'] = episodes
        for i in tqdm(range(run_exp_num)):
            run_icml(mac_config, seed=seeds[i], verbose=False)

    
if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser(description='Run experiments')
    
    parser.add_argument('-n', '--num', type=int, default=20, help='Number of experiments to run')
    parser.add_argument('-v', '--verbose', default='f', choices=['f', 't'], help='Verbose mode to display messages')

    args = parser.parse_args()

    main(args.num, args.verbose == 't')