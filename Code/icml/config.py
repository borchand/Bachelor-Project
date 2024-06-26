import sys
sys.path.append('Code/envs/')
sys.path.append('Code/')
CARTPOLE = dict(
    max_buffer_size=5000,
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
    critic_train_type='model_free_critic_monte_carlo',
)
LUNAR_LANDER = dict(
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
    critic_train_type='model_free_critic_monte_carlo')

ACROBOT = dict(
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
    critic_train_type='model_free_critic_monte_carlo')

MOUNTAIN_CAR = dict(
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
    critic_train_type='model_free_critic_monte_carlo')

PENDULUM = dict(
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
    critic_train_type='model_free_critic_monte_carlo')

MOUNTAIN_CAR_CONTINUOUS = dict(
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