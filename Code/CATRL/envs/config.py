from CartPole import CartPoleEnv
from MountainCar import MountainCarEnv
from MountainCarContinuous import MountainCarContinuousEnv
from LunarLander import LunarLanderEnv
from Acrobot import AcrobotEnv
from Pendulum import PendulumEnv

CartPole = dict(
    epsilon_min = 0.05,
    alpha = 0.05,
    decay = 0.999,
    gamma = 0.95,
    k_cap = 1,
    step_max = 50,
    episode_max = 5000,
    map_name = "Cart_Pole",
    env = CartPoleEnv(step_max = 50),
    renderEnv = CartPoleEnv(step_max = 50, render=True),
    bootstrap = 'from_init',
)
MountainCar = dict(
    epsilon_min = 0.01,
    alpha = 0.05,
    decay = 0.9,
    gamma = 0.99,
    k_cap = 1,
    step_max = 200,
    episode_max = 2500,
    map_name = "Mountain_Car",
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
    episode_max = 2500,
    map_name = "Mountain_Car_Continuous",
    env = MountainCarContinuousEnv(),
    renderEnv = MountainCarContinuousEnv(render=True),
    bootstrap = 'from_ancestor',
)

LunarLander = dict(
    epsilon_min = 0.05,
    alpha = 0.05,
    decay = 0.999,
    gamma = 0.95,
    k_cap = 2,
    step_max = 500,
    episode_max = 20000,
    map_name = "lunar",
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
    episode_max = 5000,
    map_name = "acrobot",
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
    episode_max = 10000,
    map_name = "pendulum",
    env = PendulumEnv(),
    renderEnv = PendulumEnv(render=True),
    bootstrap = 'from_init',
)