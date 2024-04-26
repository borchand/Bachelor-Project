import sys
import os
import pickle
import pandas as pd
import argparse

sys.path.append('CATRL/')
sys.path.append('Code/CATRL/envs/')

from CATRL.method_catrl import train_CAT_RL
from Code.CATRL.config import *

def save_log(log_data, log_info):
    # create folder results if it does not exist
    if not os.path.exists("results/"):
        os.makedirs("results/")

    # create folder results/agent if it does not exist
    if not os.path.exists("results/" + log_info["agent"][0]):
        os.makedirs("results/" + log_info["agent"[0]])

    # create folder results/agent/env if it does not exist
    if not os.path.exists("results/" + log_info["agent"][0] + "/" + log_info["env"][0]):
        os.makedirs("results/" + log_info["agent"][0] + "/" + log_info["env"][0])

    df = pd.DataFrame(log_data)

    df.to_csv("results/" + log_info["agent"][0] + "/" + log_info["env"][0] + "/" + log_info["agent"][0] + "_" + str(log_info["seed"][0]) + ".csv")

    df_info = pd.DataFrame(log_info)

    df_info.to_csv("results/" + log_info["agent"][0] + "/" + log_info["env"][0] + "/" + log_info["agent"][0] + "_" + str(log_info["seed"][0]) + "_info.csv")

def save_model(agent, abstract, log_info):
    # create folder models if it does not exist
    if not os.path.exists("models/"):
        os.makedirs("models/")

    # create folder models/agent if it does not exist
    if not os.path.exists("models/" + log_info["agent"][0]):
        os.makedirs("models/" + log_info["agent"][0])

    # create folder models/agent/env if it does not exist
    if not os.path.exists("models/" + log_info["agent"][0] + "/" + log_info["env"][0]):
        os.makedirs("models/" + log_info["agent"][0] + "/" + log_info["env"][0])

    file_name = log_info["agent"][0] + "/" + log_info["env"][0] + "/" + log_info["agent"][0] + "_" + str(log_info["seed"][0])

    # save the agent and abstraction
    pickle.dump(agent, open("models/" + file_name + "_agent.pkl", "wb"))
    pickle.dump(abstract, open("models/" + file_name + "_abs.pkl", "wb"))

def load_model(agent_name, env, seed):

    file_name = agent_name + "/" + env + "/" + agent_name + "_" + str(seed)

    # load the agent and abstraction
    agent = pickle.load(open("models/" + file_name + "_agent.pkl", "rb"))
    abstract = pickle.load(open("models/" + file_name + "_abs.pkl", "rb"))
    return agent, abstract

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

def main(config, seed=None):

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

    agent, abstract, log_data, log_info = train_CAT_RL(
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
        seed=seed
    )

    save_model(agent, abstract, log_info)
    save_log(log_data, log_info)

    return agent, abstract


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set options for training and rendering CAT_RL')
    parser.add_argument('-t', '--train', choices=['t', 'f'], default='t', help='Train the model')
    parser.add_argument('-r', '--render', choices=['t', 'f'], default='t', help='Render the model')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Seed for the model. If rendering, provide the seed of the model to render')
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

    if args.train == 't':
        print("Training the model")
        agent, abstract = main(config)

    if args.render == 't':
        print("Rendering the model")
        if args.train != 't' and args.seed is None:
            print("Please provide a seed to render the model")
            sys.exit()
        
        if args.train != 't':
            agent, abstract = load_model("CAT-RL", config['map_name'], args.seed)
        
        show_model(agent, abstract, config['renderEnv'])
