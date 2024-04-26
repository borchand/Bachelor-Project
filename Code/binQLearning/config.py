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
    env = CartPoleEnv(step_max = 500),
    renderEnv = CartPoleEnv(step_max=500, render=True),
    episodes = 50000,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999
)

MountainCar = dict(
    name = "MountainCar",
    step_max = 200,
    env = MountainCarEnv(),
    renderEnv = MountainCarEnv(render=True),
    episodes = 25000,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999
)

MountainCarContinuous = dict(
    name = "MountainCarContinuous",
    step_max = 200,
    env = MountainCarContinuousEnv(),
    renderEnv = MountainCarContinuousEnv(render=True),
    episodes = 250,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999
)

LunarLander = dict(
    name = "LunarLander",
    step_max = 500,
    env = LunarLanderEnv(),
    renderEnv = LunarLanderEnv(render=True),
    episodes = 2000,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999
)

Acrobot = dict(
    name = "Acrobot",
    step_max = 500,
    env = AcrobotEnv(step_max=500),
    renderEnv = AcrobotEnv(step_max=500, render=True),
    episodes = 5000,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999,
)

Pendulum = dict(
    name = "Pendulum",
    step_max = 200,
    env = PendulumEnv(),
    renderEnv = PendulumEnv(render=True),
    episodes = 200,
    gamma = 0.99,
    alpha = 0.1,
    epsilon = 1.0,
    eps_min = 0.01,
    decay = 0.999
)