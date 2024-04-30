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
    tiling_specs = [(tuple(9 for _ in range(4)), (-.8, -.4, .8, .1)),
                    (tuple(9 for _ in range(4)), (.8, .4, -.8, -.1)),
                    (tuple(9 for _ in range(4)), (-.4, .2, .4, 0))],
    episodes = 1000
)

MountainCar = dict(
    name = "MountainCar",
    step_max = 200,
    map_name = "MountainCar-v0",
    env = MountainCarEnv(),
    renderEnv = MountainCarEnv(render=True),
    tiling_specs = [(tuple(9 for _ in range(2)), (-1.2, 0.6)),
                    (tuple(9 for _ in range(2)), (-0.6, 1.2))],
    episodes = 1000
)

MountainCarContinuous = dict(
    name = "MountainCarContinuous",
    step_max = 200,
    map_name = "MountainCarContinuous-v0",
    env = MountainCarContinuousEnv(),
    renderEnv = MountainCarContinuousEnv(render=True),
    tiling_specs = [(tuple(9 for _ in range(2)), (-1.2, 0.6)),
                    (tuple(9 for _ in range(2)), (-0.6, 1.2))],
    episodes = 500
)

LunarLander = dict(
    name = "LunarLander",
    step_max = 500,
    map_name = "LunarLander-v2",
    env = LunarLanderEnv(),
    renderEnv = LunarLanderEnv(render=True),
    tiling_specs = [(tuple(10 for _ in range(8)), (-1.0, -1.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5)),
                    (tuple(10 for _ in range(8)), (1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)),
                    (tuple(10 for _ in range(8)), (-0.5, -0.5, -1.0, -1.0, -0.5, -0.5, -0.5, -0.5)),
                    (tuple(10 for _ in range(8)), (0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5))],
    episodes = 2000
)

Acrobot = dict(
    name = "Acrobot",
    step_max = 500,
    map_name = "Acrobot-v1",
    env = AcrobotEnv(step_max=500),
    renderEnv = AcrobotEnv(step_max=500, render=True),
    tiling_specs = [(tuple(10 for _ in range(6)), (-0.08, -0.06, -0.04, -0.02, 0.9, 0.6)),
                    (tuple(10 for _ in range(6)), (0.02, 0.0, -0.02, -0.04, -0.5, 0.6)),
                    (tuple(10 for _ in range(6)), (-0.06, -0.04, 0.0, -0.06, -0.8, -0.6))],
    episodes = 200
)

Pendulum = dict(
    name = "Pendulum",
    step_max = 200,
    map_name = "Pendulum-v1",
    env = PendulumEnv(),
    renderEnv = PendulumEnv(render=True),
    tiling_specs = [(tuple(3 for _ in range(3)), (-.1, 0, .1)),
                    (tuple(3 for _ in range(3)), (.1, 0, -.1)),
                    (tuple(3 for _ in range(3)), (.2, 0, -.2)),
                    (tuple(3 for _ in range(3)), (-.2, 0, .2)),],
    episodes = 6000
)