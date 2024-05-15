CARTPOLE = dict(
    gym_name='CartPole-v1',
    algo="ppo",
    policy_episodes=3600,
    experiment_episodes=2400,
    k_bins=1,
    train=False,
    run_experiment=True,
    abstraction=True,
    load_model=True,
    render_policy=False,
    render_experiment=False,
    save=True)

LUNAR_LANDER = dict(
    gym_name='LunarLander-v2',
    algo="ppo",
    policy_episodes=3600,
    experiment_episodes=2400,
    k_bins=1,
    train=False,
    run_experiment=True,
    abstraction=True,
    load_model=True,
    render_policy=False,
    render_experiment=False,
    save=True)

ACROBOT = dict(
    gym_name='Acrobot-v1',
    algo="ppo",
    policy_episodes=1200,
    experiment_episodes=800,
    k_bins=1,
    train=False,
    run_experiment=True,
    abstraction=True,
    load_model=True,
    render_policy=False,
    render_experiment=False,
    save=True)

MOUNTAIN_CAR = dict(
    gym_name='MountainCar-v0',
    algo="ppo",
    policy_episodes=3000,
    experiment_episodes=2000,
    k_bins=1,
    train=False,
    run_experiment=True,
    abstraction=True,
    load_model=True,
    render_policy=False,
    render_experiment=False,
    save=True)

PENDULUM = dict(
    gym_name='Pendulum-v1',
    algo="ppo",
    policy_episodes=3600,
    experiment_episodes=2400,
    k_bins=25,
    train=False,
    run_experiment=True,
    abstraction=True,
    load_model=True,
    render_policy=False,
    render_experiment=False,
    save=True)

MOUNTAIN_CAR_CONTINUOUS = dict(
    gym_name='MountainCarContinuous-v0',
    algo="ppo",
    policy_episodes=600,
    experiment_episodes=400,
    k_bins=2,
    train=False,
    run_experiment=True,
    abstraction=True,
    load_model=True,
    render_policy=False,
    render_experiment=False,
    save=True)
