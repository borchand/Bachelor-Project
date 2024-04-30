import sys
import os
import pickle
import pandas as pd
import argparse

sys.path.append('CATRL/')
sys.path.append('Code/CATRL/envs/')

from CATRL.method_catrl import train_CAT_RL
from Code.CATRL.config import *
from Code.utils import save_log, save_model, save_abstraction, load_model, load_abstraction

def show_model(agent, abstract, env):
    for i in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            state_abs = abstract.state(state)
            action = agent.policy(state_abs)
            new_state, reward, done, success = env.step(action)
            state = new_state

def main(config, seed=None, verbose=False):

    epsilon_min = config['epsilon_min']
    alpha = config['alpha']
    decay = config['decay']
    gamma = config['gamma']
    k_cap = config['k_cap']
    step_max = config['step_max']
    episode_max = config['episode_max']
    map_name = config['map_name']

    env = config['env']
    bootstrap = config['bootstrap'] 

    agent, abstraction, log_data, log_info = train_CAT_RL(
        map_name,
        step_max,
        episode_max,
        env,
        bootstrap,
        gamma,
        alpha,
        epsilon_min,
        decay,
        k_cap,
        seed=seed,
        verbose=verbose
    )

    save_model(agent, log_info)
    save_abstraction(abstraction, log_info)
    save_log(log_data, log_info)

    return agent, abstraction


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set options for training and rendering CAT_RL')
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

    verbose = args.verbose == 't'

    if args.train == 't':
        print("Training the model")
        agent, abstraction = main(config, seed=args.seed, verbose=verbose)

    if args.render == 't':
        print("Rendering the model")
        if args.train != 't' and args.seed is None:
            print("Please provide a seed to render the model")
            sys.exit()
        
        if args.train != 't':
            agent = load_model("CAT-RL", config['map_name'], args.seed)
            abstraction = load_abstraction("CAT-RL", config['map_name'], args.seed)
        
        show_model(agent, abstraction, config['renderEnv'])
