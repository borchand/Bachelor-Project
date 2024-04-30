import sys
from tqdm import tqdm

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


def main():

    run_exp_num = 20
    episodes_per_env = [1000, 1000, 500, 500, 2000, 1000] 
    
    # Tile Coding
    congifs = [TileAcrobot, TileCartPole, TileMountainCar, TileMountainCarContinuous, TileLunarLander, TilePendulum]

    print("Running Tile Coding")
    for config, episodes in zip(congifs, episodes_per_env):
        print("Running ", config['name'])

        env = config['env']
        tiling_specs = config['tiling_specs']

        for i in tqdm(range(run_exp_num)):
            agent = TileCodingAgent((env._action_space.n, env._env.observation_space.low, env._env.observation_space.high), tiling_specs, verbose=False)
            run_tileCoding(env, agent, episodes, config['map_name'], verbose=False)

    # Bin Q Learning
    congifs = [BinAcrobot, BinCartPole, BinMountainCar, BinMountainCarContinuous, BinLunarLander, BinPendulum]

    for config, episodes in zip(congifs, episodes_per_env):
        print("Running ", config['map_name'])

        env = config['env']

        for i in tqdm(range(run_exp_num)):
            agent = BinQLearningAgent(env._env, config["bins"], config["alpha"], config["gamma"], config["epsilon"], config["decay"], config["eps_min"], verbose=False)

            run_binQ(env, agent, episodes, config['map_name'], verbose=False)

    # CATRL
    congifs = [CATRLAcrobot, CATRLCartPole, CATRLMountainCar, CATRLMountainCarContinuous, CATRLLunarLander, CATRLPendulum]
    
    for config, episodes in zip(congifs, episodes_per_env):

        print("Running ", config['map_name'])

        env = config['env']
        config['episodes'] = episodes
        for i in tqdm(range(run_exp_num)):
            run_CATRL(config, seed=None, verbose=False)

    print("Running Bin Q Learning")

if __name__ == "__main__":
    main()