import training_config_mac
import training_config_ppo

def get_config(env_name: str, algo: str) -> dict:
    
    if algo == "mac":
        return _get_mac_config(env_name)
    elif algo == "ppo":
        return _get_ppo_config(env_name)
    else:
        raise ValueError("Invalid algorithm name")

def _get_mac_config(env_name: str) -> dict:
	
	# Get the config for the environment
	if env_name == "MountainCar-v0":
		return training_config_mac.MOUNTAIN_CAR
	if env_name == "CartPole-v0" or env_name == "CartPole-v1":
		return training_config_mac.CARTPOLE
	elif env_name == "Acrobot-v1":
		return training_config_mac.ACROBOT
	elif env_name == "LunarLander-v2":
		return training_config_mac.LUNAR_LANDER
	# Continuous action space
	elif env_name == "Pendulum-v1":
		return training_config_mac.PENDULUM
	elif env_name == "MountainCarContinuous-v0":
		return training_config_mac.MOUNTAIN_CAR_CONTINUOUS
	else:
		raise ValueError("Invalid environment name")
	
def _get_ppo_config(env_name: str) -> dict:
	# Get the config for the environment
	if env_name == "MountainCar-v0":
		return training_config_ppo.MOUNTAIN_CAR
	if env_name == "CartPole-v0" or env_name == "CartPole-v1":
		return training_config_ppo.CARTPOLE
	elif env_name == "Acrobot-v1":
		return training_config_ppo.ACROBOT
	elif env_name == "LunarLander-v2":
		return training_config_ppo.LUNAR_LANDER
	# Continuous action space
	elif env_name == "Pendulum-v1":
		return training_config_ppo.PENDULUM
	elif env_name == "MountainCarContinuous-v0":
		return training_config_ppo.MOUNTAIN_CAR_CONTINUOUS
	else:
		raise ValueError("Invalid environment name")