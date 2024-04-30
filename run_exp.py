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

sys.path.append('Code/TileCoding/')
sys.path.append('Code/envs/')

from Code.TileCoding.config import Acrobot as TileAcrobot, CartPole as TileCartPole, MountainCar as TileMountainCar, MountainCarContinuous as TileMountainCarContinuous, LunarLander as TileLunarLander, Pendulum as TilePendulum
from Code.TileCoding.agent import Agent as TileCodingAgent

sys.path.append('Code/binQLearning/')

from Code.binQLearning.config import Acrobot as BinAcrobot, CartPole as BinCartPole, MountainCar as BinMountainCar, MountainCarContinuous as BinMountainCarContinuous, LunarLander as BinLunarLander, Pendulum as BinPendulum
from Code.binQLearning.agent import QLearningAgent as BinQLearningAgent

sys.path.append('Code/CATRL/')

from Code.CATRL.config import Acrobot as CATRLAcrobot, CartPole as CATRLCartPole, MountainCar as CATRLMountainCar, MountainCarContinuous as CATRLMountainCarContinuous, LunarLander as CATRLLunarLander, Pendulum as CATRLPendulum

from Code.icml.training_config_mac import ACROBOT as IcmlAcrobot, CARTPOLE as IcmlCartPole, MOUNTAIN_CAR as IcmlMountainCar, MOUNTAIN_CAR_CONTINUOUS as IcmlMountainCarContinuous, LUNAR_LANDER as IcmlLunarLander, PENDULUM as IcmlPendulum

def main(run_exp_num = 20):

    episodes_per_env = [10, 10, 10, 10, 10, 10] 

    # create seeds
    seeds = random.sample(range(1000), run_exp_num)
    
    # Tile Coding
    congifs = [TileAcrobot, TileCartPole, TileMountainCar, TileMountainCarContinuous, TileLunarLander, TilePendulum]

    print('\n' + '{:_^40}'.format("Running Tile Coding") + '\n')
    for config, episodes in zip(congifs, episodes_per_env):
        print("Running ", config['name'])

        env = config['env']
        tiling_specs = config['tiling_specs']

        for i in tqdm(range(run_exp_num)):
            agent = TileCodingAgent((env._action_space.n, env._env.observation_space.low, env._env.observation_space.high), tiling_specs, verbose=False)
            run_tileCoding(env, agent, episodes, config['map_name'], seed=seeds[i], verbose=False)

    # Bin Q Learning
    congifs = [BinAcrobot, BinCartPole, BinMountainCar, BinMountainCarContinuous, BinLunarLander, BinPendulum]


    print('\n' + '{:_^40}'.format("Running bins") + '\n')

    for config, episodes in zip(congifs, episodes_per_env):
        print("Running ", config['map_name'])

        env = config['env']

        for i in tqdm(range(run_exp_num)):
            agent = BinQLearningAgent(env._env, config["bins"], config["alpha"], config["gamma"], config["epsilon"], config["decay"], config["eps_min"], seed=seeds[i], verbose=False)

            run_binQ(env, agent, episodes, config['map_name'], seed=seeds[i], verbose=False)

    # CATRL
    congifs = [CATRLAcrobot, CATRLCartPole, CATRLMountainCar, CATRLMountainCarContinuous, CATRLLunarLander, CATRLPendulum]
    

    print('\n' +'{:_^40}'.format("Running CAT-RL") + '\n')

    for config, episodes in zip(congifs, episodes_per_env):

        print("Running ", config['map_name'])

        env = config['env']
        config['episode_max'] = episodes
        for i in tqdm(range(run_exp_num)):
            run_CATRL(config, seed=seeds[i], verbose=False)

    
    print('\n' +'{:_^40}'.format("Running Icml") + '\n')


    
if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('-n', '--num', type=int, default=20, help='Number of experiments to run')

    args = parser.parse_args()

    main(args.num)