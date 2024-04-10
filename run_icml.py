import sys
import os
sys.path.append("./icml_2019_state_abstraction/experiments")
sys.path.append("./icml_2019_state_abstraction/experiments/simple_rl")
sys.path.append("./icml_2019_state_abstraction/experiments/abstraction")
import icml_2019_state_abstraction.mac.run as run
import icml_2019_state_abstraction.experiments.run_learning_experiment as run_learning_experiment
from icml_2019_state_abstraction.experiments import run_learning_experiment


def main():

    gym_env = sys.argv[1]

    ## run training of policy
    # run.main(gym_env)

    ## run learning experiment
    run_learning_experiment.main(gym_env, abstraction=True)

if __name__ == "__main__":
    main()