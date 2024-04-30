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
    name = "CartPole",
    step_max = 500,
    map_name = "CartPole-v1",
    env = CartPoleEnv(step_max = 500),
    renderEnv = CartPoleEnv(step_max=500, render=True),
    episodes = 5000,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999,
    bins = [2, 3, 3, 5] # try 6
)

MountainCar = dict(
    name = "MountainCar",
    step_max = 200,
    map_name = "MountainCar-v0",
    env = MountainCarEnv(),
    renderEnv = MountainCarEnv(render=True),
    episodes = 5000,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999,
    bins = [3, 3]
)

MountainCarContinuous = dict(
    name = "MountainCarContinuous",
    step_max = 200,
    map_name = "MountainCarContinuous-v0",
    env = MountainCarContinuousEnv(),
    renderEnv = MountainCarContinuousEnv(render=True),
    episodes = 1000,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999,
    bins = [3, 3]
)

LunarLander = dict(
    name = "LunarLander",
    step_max = 500,
    map_name = "LunarLander-v2",
    env = LunarLanderEnv(),
    renderEnv = LunarLanderEnv(render=True),
    episodes = 6000,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999,
    bins = [3, 3, 3, 3, 5, 8, 2, 2]
)

Acrobot = dict(
    name = "Acrobot",
    step_max = 500,
    map_name = "Acrobot-v1",
    env = AcrobotEnv(step_max=500),
    renderEnv = AcrobotEnv(step_max=500, render=True),
    episodes = 1000,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999,
    bins = [2, 2, 2, 2, 3, 5]
)

Pendulum = dict(
    name = "Pendulum",
    step_max = 200,
    map_name = "Pendulum-v1",
    env = PendulumEnv(),
    renderEnv = PendulumEnv(render=True),
    episodes = 6000,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999,
    bins = [2, 4, 4]
)