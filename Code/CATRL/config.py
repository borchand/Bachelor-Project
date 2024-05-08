import sys
sys.path.append('Code/envs/')
sys.path.append('Code/')

from envs.CartPole import CartPoleEnv
from envs.MountainCar import MountainCarEnv
from envs.MountainCarContinuous import MountainCarContinuousEnv
from envs.LunarLander import LunarLanderEnv
from envs.Acrobot import AcrobotEnv
from envs.Pendulum import PendulumEnv

CartPole = dict(
    epsilon_min = 0.05,
    alpha = 0.1,
    decay = 0.999,
    gamma = 0.95,
    k_cap = 1,
    step_max = 500,
    episode_max = 5000,
    map_name = "CartPole-v1",
    env = CartPoleEnv(step_max = 500),
    renderEnv = CartPoleEnv(step_max = 500, render=True),
    bootstrap = 'from_init',
)
MountainCar = dict(
    epsilon_min = 0.01,
    alpha = 0.05,
    decay = 0.9,
    gamma = 0.99,
    k_cap = 1,
    step_max = 200,
    episode_max = 500,
    map_name = "MountainCar-v0",
    env = MountainCarEnv(),
    renderEnv = MountainCarEnv(render=True),
    bootstrap = 'from_ancestor',
)

MountainCarContinuous = dict(
    epsilon_min = 0.01,
    alpha = 0.05,
    decay = 0.9,
    gamma = 0.99,
    k_cap = 1,
    step_max = 200,
    episode_max = 500,
    map_name = "MountainCarContinuous-v0",
    env = MountainCarContinuousEnv(k_bins=50),
    renderEnv = MountainCarContinuousEnv(render=True, k_bins=50),
    bootstrap = 'from_ancestor',
)

LunarLander = dict(
    epsilon_min = 0.05,
    alpha = 0.05,
    decay = 0.999,
    gamma = 0.95,
    k_cap = 2,
    step_max = 500,
    episode_max = 5000,
    map_name = "LunarLander-v2",
    env = LunarLanderEnv(),
    renderEnv = LunarLanderEnv(render=True),
    bootstrap = 'from_init',
)

Acrobot = dict(
    epsilon_min = 0.05,
    alpha = 0.05,
    decay = 0.999,
    gamma = 0.95,
    k_cap = 2,
    step_max = 500,
    episode_max = 3000,
    map_name = "Acrobot-v1",
    env = AcrobotEnv(500),
    renderEnv = AcrobotEnv(500, render=True),
    bootstrap = 'from_init',
)

Pendulum = dict(
    epsilon_min = 0.05,
    alpha = 0.05,
    decay = 0.999,
    gamma = 0.95,
    k_cap = 2,
    step_max = 200,
    episode_max = 6000,
    map_name = "Pendulum-v1",
    env = PendulumEnv(k_bins=8),
    renderEnv = PendulumEnv(render=True, k_bins=8),
    bootstrap = 'from_init',
)