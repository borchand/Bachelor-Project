# based on: https://github.com/lbarazza/Tile-Coding/blob/master/main.py
import sys
import random
import time
import argparse
from tqdm import tqdm
sys.path.append('Code/TileCoding/')
sys.path.append('Code/envs/')
sys.path.append("CATRL/")

from Code.binQLearning.config import *
from Code.utils import save_log, save_model, load_model
from Code.binQLearning.agent import QLearningAgent

def run_agent(env, agent, nEpisodes, env_name, seed=None, verbose=False, model_save=True, log_save=True, log_folder="results/"):

    log_data = {
        "episode": [],
        "reward": [],
        "epochs": [],
        "epsilon": [],
        "success": []
    }

    log_info = {
        "seed": [],
        "time": [],
        "agent": [],
        "env": [],
        "episodes": [],
        "epochs": [],
        "success_rate": [],
    }

    if seed is None:
        seed = random.randint(0, 1000)

    random.seed(seed)

    start_time = time.time()

    for i in range(nEpisodes):
        state = env.reset()
        action = agent.reset_episode(state)
        total_reward = 0
        epochs = 0
        done = False

        while not done:
            state, reward, done, success = env.step(action)
            total_reward += reward
            epochs += 1
            action = agent.act(state, reward, done)

        if verbose:
            print ("_______________________________")
            print ("Episode: " + str(i) + '\t' + "Reward: " + str (total_reward) + '\t' + "Epsilon: " + str(agent.epsilon) + '\t' + "epochs: " + str(epochs) + '\t' + "success: " + str(success))

        log_data["episode"].append(i)
        log_data["reward"].append(total_reward)
        log_data["epochs"].append(epochs)
        log_data["epsilon"].append(agent.epsilon)
        log_data["success"].append(success)

    end_time = time.time()

    log_info["seed"].append(seed)
    log_info["time"].append(end_time - start_time)
    log_info["agent"].append("binQ")
    log_info["env"].append(env_name)
    log_info["episodes"].append(nEpisodes)
    log_info["epochs"].append(sum(log_data["epochs"]))
    log_info["success_rate"].append(sum(log_data["success"])/nEpisodes)

    if model_save:
        save_model(agent, log_info)
    if log_save:
        save_log(log_data, log_info, folder=log_folder)

    return agent


def show_model(env, agent):
    for i in range(1):
        state = env.reset()
        done = False
        action = agent.reset_episode(state)
        while not done:
            env.render()
            action = agent.act(state, mode="test")
            new_state, reward, done, success = env.step(action)
            state = new_state


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set options for training and rendering TileCoding')
    parser.add_argument('-t', '--train', choices=['t', 'f'], default='t', help='Train the model')
    parser.add_argument('-r', '--render', choices=['t', 'f'], default='t', help='Render the model')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Seed for the model. If rendering, provide the seed of the model to render')
    parser.add_argument('-v', '--verbose', choices=['t', 'f'], default='t', help='Verbose mode')
    # choose the environment to train and render
    parser.add_argument('-e', '--env', default='MountainCar', choices=['MountainCar', 'MountainCarContinuous','CartPole', 'LunarLander', 'Acrobot', 'Pendulum'], help='Choose the environment to train and render')
    args = parser.parse_args()

    if args.env == 'MountainCar':
        config = MountainCar
    elif args.env == 'MountainCarContinuous':
        config = MountainCarContinuous
    elif args.env == 'CartPole':
        config = CartPole
    elif args.env == 'LunarLander':
        config = LunarLander
    elif args.env == 'Acrobot':
        config = Acrobot
    elif args.env == 'Pendulum':
        config = Pendulum
    else:
        print("Invalid environment")
        sys.exit()

    print("Environment: ", args.env)

    env = config['env']
    
    verbose = args.verbose == 't'

    if args.train == 't':
        print("Training the model")

        agent = QLearningAgent(env._env, config["bins"], config["alpha"], config["gamma"], config["epsilon"], config["decay"], config["eps_min"], seed=args.seed)

        trained_agent = run_agent(env, agent, config["episodes"], config['map_name'], args.seed, verbose=verbose)

    if args.render == 't':
        print("Rendering the model")
        if args.train != 't' and args.seed is None:
            print("Please provide a seed to render the model")
            sys.exit()

        if args.train != 't':
            trained_agent = load_model("binQ", config['map_name'], args.seed)

        env = config['renderEnv']
        show_model(env, trained_agent)
