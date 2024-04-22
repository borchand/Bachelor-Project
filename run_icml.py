import sys
import os
sys.path.append("./icml_2019_state_abstraction/experiments")
sys.path.append("./icml_2019_state_abstraction/experiments/simple_rl")
sys.path.append("./icml_2019_state_abstraction/experiments/abstraction")
import icml_2019_state_abstraction.mac.run as run
import icml_2019_state_abstraction.experiments.run_learning_experiment as run_learning_experiment
from icml_2019_state_abstraction.experiments import run_learning_experiment
import baselines
import argparse


def main(gym_name: str, algo: str, time_steps: int, k_bins=1, train=True, run_experiment=True, abstraction=True, load_model=False, render=False, save=True):

    """
    Args:
        :param gym_name (str): Name of the environment
        :param algo (str): Name of the algorithm
        :param time_steps (int): Number of time steps to train the model for
        :param train = True (bool): If True, train the model
        :param run_experiment = True (bool): If True, run the learning experiment
        :param abstraction = True (bool): If True, use state abstraction
        :param discretize = True (bool): If True, discretize the action space
        :param load_model = False (bool): If True, load a pre-trained model
        :param render = False (bool): If True, render the model
        :param save = True (bool): If True, save the model
    Summary:
        Run the training of the model and the learning experiment
    """
    
    ## run training of policy
    if train and algo == 'mac':
        
        print("running training of algorithm: ", algo, "in environment: ", gym_name)
        run.main(
            env_name=gym_name,
            time_steps=time_steps,
            k_bins=k_bins)
    
    elif train:
        
        print("Running training of algorithm: ", algo, "in environment: ", gym_name, "for ", time_steps, "time steps.")
        baselines.main(
            env_name=gym_name,
            algo_name=algo,
            timesteps=time_steps,
            k=k_bins,
            render=render,
            save=save,
            train=train)
        
        print("Training complete.")
         
    ## run learning experiment
    if run_experiment or abstraction or load_model:
        run_learning_experiment.main(
            env_name=gym_name,
            algo=algo,
            k_bins=k_bins,
            abstraction=abstraction,
            load_model=load_model,
            policy_train_steps=time_steps,
            run_expiriment=run_experiment)

def show_model(gym_name: str, algo: str):
    pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Set options for training and rendering icml')
    
    parser.add_argument('-e', '--env', default='CartPole-v1', help='Environment to train on')
    parser.add_argument('-a', '--algo', default='ppo', choices=['mac', 'dqn', 'ppo', 'sac'], help='Algorithm to use when training')
    parser.add_argument('-t', '--time-steps', default=100_000, help='Number of time steps to train the model for', type=int)
    parser.add_argument('-k', '--k-bins', default=1, help='Number of bins to discretize the action space', type=int)

    parser.add_argument('-tr', '--train', choices=['t', 'f'], default='t', help='Train the model')
    parser.add_argument('-ex', '--experiment', choices=['t', 'f'], default='t', help='Run the learning experiment')
    parser.add_argument('-ab', '--abstraction', choices=['t', 'f'], default='t', help='Use state abstraction')

    parser.add_argument('-l', '--load', choices=['t', 'f'], default='f', help='Load a pre-trained model')
    parser.add_argument('-r', '--render', choices=['t', 'f'], default='f', help='Render the model')
    parser.add_argument('-s', '--save', choices=['t', 'f'], default='t', help='Save the model')
    parser.add_argument('-sh', '--show', choices=['t', 'f'], default='f', help='Show the model')

    args = parser.parse_args()
    
    main(
        gym_name=args.env,
        algo=args.algo,
        time_steps=args.time_steps,
        abstraction=args.abstraction == 't',
        train=args.train == 't',
        load_model=args.load == 't',
        render=args.render == 't',
        save=args.save == 't',
        run_experiment=args.experiment == 't',
        k_bins=args.k_bins)