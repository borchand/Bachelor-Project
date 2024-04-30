import sys
import os
sys.path.append("./icml_2019_state_abstraction/experiments")
sys.path.append("./icml_2019_state_abstraction/experiments/simple_rl")
sys.path.append("./icml_2019_state_abstraction/experiments/abstraction")
sys.path.append("./Code/icml/")
import icml_2019_state_abstraction.mac.run as run
import icml_2019_state_abstraction.experiments.run_learning_experiment as run_learning_experiment
from icml_2019_state_abstraction.experiments import run_learning_experiment
import baselines
import argparse
import Code.icml.config
def main(
        gym_name: str,
        algo: str,
        policy_episodes: int,
        experiment_episodes: int,
        k_bins: int,
        seed: int,
        train=True,
        run_experiment=True,
        abstraction=True,
        load_model=False,
        render_policy=False,
        render_experiment=False,
        save=True,
        verbose=False):

    """
    Args:
        :param gym_name (str): Name of the environment
        :param algo (str): Name of the algorithm
        :param policy_episodes (int): Number of episodes to train the model for
        :param experiment_episodes (int): Number of episodes to run the experiement for
        :param k_bins (int): Number of bins to discretize the action space
        :param seed (int): Seed for reproducibility
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
    
    continuous_action_envs = ['Pendulum-v1', 'MountainCarContinuous-v0', 'LunarLanderContinuous-v2']
    if gym_name in continuous_action_envs:
        assert k_bins > 1, "Action space must be discretized for continuous action environments."
    assert "-" in gym_name, "Remember to use the correct gym name. with version number."
    
    if algo == 'mac':
        
        if verbose:
            print("running training of algorithm: ", algo, "in environment: ", gym_name)
        
        run.main(
            env_name=gym_name,
            episodes=policy_episodes,
            k_bins=k_bins,
            seed=seed,
            train=train,
            verbose=verbose,
            render=render_policy)
    
    else:
        if verbose:
            print("Running training of algorithm: ", algo, "in environment: ", gym_name, "for ", policy_episodes, "episodes.")
        
        baselines.main(
            env_name=gym_name,
            algo_name=algo,
            episodes=policy_episodes,
            k=k_bins,
            seed=seed,
            render=render_policy,
            save=save,
            train=train)
        
        print("Training complete.")
         
    ## run learning experiment
    if run_experiment or abstraction or load_model:
        run_learning_experiment.main(
            env_name=gym_name,
            algo=algo,
            k_bins=k_bins,
            seed=seed,
            abstraction=abstraction,
            load_model=load_model,
            policy_train_episodes=policy_episodes,
            render=render_experiment,
            experiment_episodes=experiment_episodes,
            run_expiriment=run_experiment,
            verbose=verbose)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Set options for training and rendering icml')
    
    parser.add_argument('-e', '--env', default='CartPole-v1', help='Environment to train on')
    parser.add_argument('-a', '--algo', default='ppo', choices=['mac', 'dqn', 'ppo', 'sac'], help='Algorithm to use when training')
    parser.add_argument('-k', '--k-bins', default=1, help='Number of bins to discretize the action space', type=int)
    parser.add_argument('-pep', '--policy_episodes', default=100, help='Number of episodes to train the model for', type=int)
    parser.add_argument('-eep', '--experiment_episodes', default=100, help='Number of episodes to train the model for', type=int)
    parser.add_argument('-seed', '--seed', default=42, help='Seed for reproducibility', type=int)

    parser.add_argument('-tr', '--train', choices=['t', 'f'], default='t', help='Train the model')

    parser.add_argument('-ex', '--experiment', choices=['t', 'f'], default='t', help='Run the learning experiment')
    parser.add_argument('-ab', '--abstraction', choices=['t', 'f'], default='t', help='Use state abstraction')

    parser.add_argument('-l', '--load', choices=['t', 'f'], default='f', help='Load a pre-trained model')
    parser.add_argument('-s', '--save', choices=['t', 'f'], default='t', help='Save the model')
    parser.add_argument('-sh', '--show', choices=['t', 'f'], default='f', help='Show the model')
    parser.add_argument('-v', '--verbose', choices=['t', 'f'], default='t', help='Verbose output')
    
    parser.add_argument('-r', '--render', choices=['t', 'f'], default='t', help='Render the model')
    parser.add_argument('-rp', '--render-policy', choices=['t', 'f'], default=None, help='Render the policy')
    parser.add_argument('-re', '--render-experiment', choices=['t', 'f'], default=None, help='Render the policy')
    
    args = parser.parse_args()
    render_policy = args.render_policy if args.render_policy is not None else args.render
    render_experiment = args.render_experiment if args.render_experiment is not None else args.render
    
    main(
        gym_name=args.env,
        algo=args.algo,
        policy_episodes=args.policy_episodes,
        experiment_episodes=args.experiment_episodes,
        abstraction=args.abstraction == 't',
        seed=args.seed,
        train=args.train == 't',
        load_model=args.load == 't',
        render_policy=render_policy == 't',
        render_experiment=render_experiment == 't',
        save=args.save == 't',
        run_experiment=args.experiment == 't',
        k_bins=args.k_bins,
        verbose=args.verbose == 't')