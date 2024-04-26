# based on: https://github.com/lbarazza/Tile-Coding/blob/master/main.py
import sys
import gymnasium as gym
import pickle
import os
import argparse
from gymnasium import ObservationWrapper
import numpy as np

sys.path.append('Code/TileCoding/')
sys.path.append('Code/envs/')
sys.path.append("CATRL/")

from Code.binQLearning.config import *
class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, env, state_grid, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.env = env
        self.state_grid = state_grid
        print("State grid:", self.state_grid)
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
        self.seed = np.random.seed(seed)
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)
        
        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate # how quickly should we decrease epsilon
        self.min_epsilon = min_epsilon
        
        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        
        return tuple(self.discretize(state))
    
    def discretize(self, state):
        """Discretize a state to its corresponding grid cell."""
        return tuple(int(np.digitize(s, g)) for s, g in zip(state, self.state_grid))

    def reset_episode(self, state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action
    
    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode='train'):
        """Pick next action and update internal Q table (when mode != 'test')."""
        state = self.preprocess_state(state)
        if mode == 'test':
            # Test mode: Simply produce an action
            action = np.argmax(self.q_table[state])
        else:
            # Train mode (default): Update Q table, pick next action
            # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
                (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])

            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action


def save_agent(agent, file_name):
    # create folder models if it does not exist
    if not os.path.exists("models/"):
        os.makedirs("models/")
    # save the agent and abstraction
    pickle.dump(agent, open("models/" + file_name + "_binQLearning_agent.pkl", "wb"))


def load_agent(file_name):
    # load the agent and abstraction
    agent = pickle.load(open("models/" + file_name + "_binQLearning_agent.pkl", "rb"))
    return agent

def run_agent(env, agent, nEpisodes, render=False):
    returns = [run_episode(env, agent, i) for i in range(nEpisodes)]
    save_agent(agent, "acrobot")
    return returns

def run_episode(env, agent, i):
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

    print("Episode: ", i, "Reward: ", total_reward, "Epsilon: ", agent.epsilon, "epochs: ", epochs, "success: ", success)


    return total_reward


def show_model(env, agent):
    for i in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.reset_episode(state)
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

    def create_uniform_grid(low, high, bins=(10, 10)):
        grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
        return grid
    
    if args.train == 't':
        print("Training the model")
        # agent = qlearning (
        #     env, 
        #     epsilon=config["epsilon"],
        #     gamma=config["gamma"],
        #     alpha=config["alpha"],
        #     state_size = env._state_size, 
        #     action_size = env._action_size,
        #     eps_min=config["eps_min"],
        #     decay=config["decay"],)

        # create a state grid based on num of bins

        bins = 10
        state_grid = []
        for i in range(env._env.observation_space.shape[0]):
            state_grid.append(np.linspace(env._env.observation_space.low[i], env._env.observation_space.high[i], bins))

        print("State grid:", state_grid)
        agent = QLearningAgent(env._env, state_grid)

        run_agent(env, agent, config["episodes"])

    if args.render == 't':
        print("Rendering the model")
        agent = load_agent("acrobot")
        env = config['renderEnv']
        show_model(env, agent)
