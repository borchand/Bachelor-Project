import sys
# make imports work in the submodules
sys.path.append("./markovStateAbstractions/markov_abstr/gridworld/")

from markovStateAbstractions.markov_abstr.gridworld.models.nullabstraction import NullAbstraction
from markovStateAbstractions.markov_abstr.gridworld.agents.dqnagent import DQNAgent

import gymnasium as gym

env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
observation, info = env.reset(seed=42)

latent_dims = 2
n_actions = 5

phinet = NullAbstraction(-1, latent_dims)
learning_rate = 0.003
batch_size = 16
train_phi = False
gamma = .9


agent = DQNAgent(n_features=latent_dims,
                     n_actions=n_actions,
                     phi=phinet,
                     lr=learning_rate,
                     batch_size=batch_size,
                     train_phi=train_phi,
                     gamma=gamma,
                     factored=False)
print("test2")
for _ in range(1000):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    loss = agent.train(observation, action, reward, observation, terminated or truncated)

    if terminated or truncated:
        print("Episode terminated")
        agent.reset()
        observation, info = env.reset()

env.render()
env.close()