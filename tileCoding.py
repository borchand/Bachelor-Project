# based on: https://github.com/lbarazza/Tile-Coding/blob/master/main.py
import sys
import gymnasium as gym
import pickle
import os
import argparse

sys.path.append('Code/TileCoding/')
sys.path.append('Code/envs/')

from Code.TileCoding.config import *
from Code.TileCoding.agent import *

def save_agent(agent, file_name):
    # create folder models if it does not exist
    if not os.path.exists("models/"):
        os.makedirs("models/")
    # save the agent and abstraction
    pickle.dump(agent, open("models/" + file_name + "_tileCoding_agent.pkl", "wb"))


def load_agent(file_name):
    # load the agent and abstraction
    agent = pickle.load(open("models/" + file_name + "_tileCoding_agent.pkl", "rb"))
    return agent

def run_agent(env, agent, nEpisodes, render=False):
    returns = [run_episode(env, agent, i) for i in range(nEpisodes)]
    save_agent(agent, "acrobot")
    return returns

def run_episode(env, agent, i):
    state = env.reset()
    ret = 0
    epochs = 0
    while True:
        action = agent.choose_action(state)

        new_state, reward, done, success = env.step(action)
        ret+=reward
        epochs+=1
        agent.train(state, action, reward, new_state, done)
        state = new_state
        if done:
            print("Episode: ", i, "Reward: ", ret, "Epsilon: ", agent.epsilon, "epochs: ", epochs, "success: ", success)

            break

    return ret


def show_model(env, agent):
    for i in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.choose_action(state)
            new_state, reward, done, success = env.step(action)
            state = new_state

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set options for training and rendering TileCoding')
    parser.add_argument('-t', '--train', choices=['t', 'f'], default='t', help='Train the model')
    parser.add_argument('-r', '--render', choices=['t', 'f'], default='t', help='Render the model')
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

    tiling_specs = config['tiling_specs']


    if args.train == 't':
        print("Training the model")
        agent = Agent((env._action_space.n, env._env.observation_space.low, env._env.observation_space.high), tiling_specs)
        run_agent(env, agent, config["episodes"])

    if args.render == 't':
        print("Rendering the model")
        agent = load_agent("acrobot")
        env = config['renderEnv']
        show_model(env, agent)







