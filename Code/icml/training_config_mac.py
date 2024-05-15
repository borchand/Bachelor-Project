CARTPOLE = dict(
    gym_name='CartPole-v1',
    algo="mac",
    policy_episodes=500,
    experiment_episodes=200,
    k_bins=1,
    train=True,
    run_experiment=True,
    abstraction=True,
    load_model=False,
    render_policy=False,
    render_experiment=False,
    max_buffer_size=10000,
    epsilon=0.3,
    actor_num_h=1,
    actor_h=64,
    actor_lr=0.001,
    critic_num_h=1,
    critic_h=32,
    critic_lr=0.01,
    critic_batch_size=32,
    critic_num_epochs=40,
    critic_target_net_freq=1,
    critic_train_type='model_free_critic_monte_carlo'
    )

LUNAR_LANDER = dict(
    gym_name='LunarLander-v2',
    algo="mac",
    policy_episodes=2000,
    experiment_episodes=200,
    k_bins=1,
    train=True,
    run_experiment=True,
    abstraction=True,
    load_model=False,
    render_policy=False,
    render_experiment=False,
    # Abstraction network
    max_buffer_size=10000,
    epsilon=0.3,
    actor_num_h=2,
    actor_h=128,
    actor_lr=0.00025,
    critic_num_h=2,
    critic_h=128,
    critic_lr=0.005,
    critic_batch_size=32,
    critic_num_epochs=10,
    critic_target_net_freq=1,
    critic_train_type='model_free_critic_monte_carlo'
    )

ACROBOT = dict(
    gym_name="Acrobot-v1",
    algo="mac",
    policy_episodes=1200,
    experiment_episodes=200,
    k_bins=1,
    train=True,
    run_experiment=True,
    abstraction=True,
    load_model=False,
    render_policy=False,
    render_experiment=False,
    max_buffer_size=10000,
    epsilon=0.6,
    actor_num_h=2,
    actor_h=128,
    actor_lr=0.00025,
    critic_num_h=2,
    critic_h=128,
    critic_lr=0.005,
    critic_batch_size=64,
    critic_num_epochs=10,
    critic_target_net_freq=1,
    critic_train_type='model_free_critic_monte_carlo'
    )

MOUNTAIN_CAR = dict(
    gym_name='MountainCar-v0',
    algo="mac",
    policy_episodes=1400,
    experiment_episodes=200,
    k_bins=1,
    train=True,
    run_experiment=True,
    abstraction=True,
    load_model=False,
    render_policy=False,
    render_experiment=False,
    max_buffer_size=10000,
    epsilon=0.3,
    actor_num_h=2,
    actor_h=128,
    actor_lr=0.001,
    critic_num_h=2,
    critic_h=128,
    critic_lr=0.001,
    critic_batch_size=32,
    critic_num_epochs=10,
    critic_target_net_freq=1,
    critic_train_type='model_free_critic_monte_carlo'
    )

PENDULUM = dict(
    gym_name='Pendulum-v1',
    algo="mac",
    policy_episodes=1200,
    experiment_episodes=200,
    k_bins=4,
    train=True,
    run_experiment=True,
    abstraction=True,
    load_model=False,
    render_policy=False,
    render_experiment=False,
    max_buffer_size=10000,
    epsilon=0.3,
    actor_num_h=2,
    actor_h=64,
    actor_lr=0.0001,
    critic_num_h=2,
    critic_h=64,
    critic_lr=0.001,
    critic_batch_size=32,
    critic_num_epochs=10,
    critic_target_net_freq=1,
    critic_train_type='model_free_critic_monte_carlo'
    )

MOUNTAIN_CAR_CONTINUOUS = dict(
    gym_name='MountainCarContinuous-v0',
    algo="mac",
    policy_episodes=600,
    experiment_episodes=200,
    k_bins=2,
    train=True,
    run_experiment=True,
    abstraction=True,
    load_model=False,
    render_policy=False,
    render_experiment=False,
    max_buffer_size=10000,
    epsilon=0.6,
    actor_num_h=2,
    actor_h=40,
    actor_lr=0.01,
    critic_num_h=2,
    critic_h=40,
    critic_lr=0.01,
    critic_batch_size=64,
    critic_num_epochs=10,
    critic_target_net_freq=1,
    critic_train_type='model_free_critic_monte_carlo')
