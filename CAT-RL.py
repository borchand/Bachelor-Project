import sys
import os
import pickle
import pandas as pd

sys.path.append('CATRL/')
sys.path.append('Code/CATRL/envs/')

from CATRL.method_catrl import train_CAT_RL
from Code.CATRL.envs.MountainCar import MountainCarEnv
from Code.CATRL.envs.config import *

def save_log(log_data, file_name):
    # create folder results if it does not exist
    if not os.path.exists("results/"):
        os.makedirs("results/")

    df = pd.DataFrame(log_data)

    df.to_csv("results/" + file_name + ".csv")

def save_model(agent, abstract, file_name):
    # create folder models if it does not exist
    if not os.path.exists("models/"):
        os.makedirs("models/")
    # save the agent and abstraction
    pickle.dump(agent, open("models/" + file_name + "_agent.pkl", "wb"))
    pickle.dump(abstract, open("models/" + file_name + "_abs.pkl", "wb"))

def load_model(file_name):
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

def main(config):

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

    agent, abstract, log_data = train_CAT_RL(
        map_name,
        step_max,
        episode_max,
        env,
        bootstrap,
        gamma,
        alpha,
        epsilon_min,
        decay,
        k_cap
    )

    save_model(agent, abstract, map_name)
    save_log(log_data, map_name)


if __name__ == "__main__":
    

    # config = CartPole
    # config = MountainCar
    # config = LunarLander
    config = Acrobot

    # main(config)

    
    agent, abstract = load_model(config['map_name'])
    show_model(agent, abstract, config['renderEnv'])
