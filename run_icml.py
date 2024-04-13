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


def main(gym_name: str, algo: str, time_steps: int, train = True, run_experiment = True, abstraction = True, render= False, save = True):

    
    ## run training of policy
    if train and algo == 'mac':
        print("running training of algorithm: ", algo, "in environment: ", gym_name)
            
        run.main(gym_name)
    elif train:
        print("Running training of algorithm: ", algo, "in environment: ", gym_name, "for ", time_steps, "time steps.")
        baselines.main(gym_name, algo, time_steps, render, save, train)
        print("Training complete.")
         
    ## run learning experiment
    if run_experiment:
        run_learning_experiment.main(
            gym_name,
            algo,
            abstraction=abstraction)

def show_model(gym_name: str, algo: str):
    pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Set options for training and rendering icml')
    
    parser.add_argument('-e', '--env', default='CartPole-v1', help='Environment to train on')
    parser.add_argument('-a', '--algo', default='dqn', choices=['mac', 'dqn', 'ppo', 'sac'], help='Algorithm to use when training')
    parser.add_argument('-t', '--time-steps', default=100_000, help='Number of time steps to train the model for', type=int)
    
    parser.add_argument('-tr', '--train', choices=['t', 'f'], default='t', help='Train the model')
    parser.add_argument('-ex', '--experiment', choices=['t', 'f'], default='t', help='Run the learning experiment')
    parser.add_argument('-ab', '--abstraction', choices=['t', 'f'], default='t', help='Use state abstraction')

    parser.add_argument('-r', '--render', choices=['t', 'f'], default='f', help='Render the model')
    parser.add_argument('-s', '--save', choices=['t', 'f'], default='t', help='Save the model')
    parser.add_argument('-sh', '--show', choices=['t', 'f'], default='f', help='Show the model')

    args = parser.parse_args()
    
    main(
        gym_name=args.env,
        algo=args.algo,
        time_steps=args.time_steps,
        train=args.train == 't',
        render=args.render == 't',
        save=args.save == 't')