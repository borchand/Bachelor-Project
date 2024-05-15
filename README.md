# Bachelor-Project


## Getting Started
The code in this repository uses python 3.10 or 3.11

Some tasks from OpenAI Gym uses `Box2D`. For this to work you need to isnstall swig. You can do this using brew with the following command:

```
brew install swig
```

To initialize submodules run this command
```
git submodule update --init --recursive
```

When you have swig installed, you can install the required packages using the following command:
```
pip install -r requirements.txt
```

## Environments
This repository contains the following environments:
- `CartPole-v1`
- `MountainCar-v0`
- `LunarLander-v2`
- `Acrobot-v1`
- `MountainCarContinuous-v0`
- `Pendulum-v1`

## Icml State abstraction 
To run the code, this will train a ppo policy with 1000 episodes and then run the experiments
```
python run_icml.py -a ppo -e Acrobot-v1 -pep 1000
```

To run a pre trained policy, you have to specify the seed 
```
python run_icml.py -a ppo -e Acroot-v1 -pep 1000 --seed 42 -tr f 
```
This has training false and will load a policy that have trained for 1000 episodes with the seed 42

## CAT_RL
about...

### Running the code
To run the code, you can use the following command:
```
python CAT-RL.py
```
This will run all the environments and render the model after training.

There are also some optional arguments you can use:
- `--env` or `-e` to specify the environment you want to run 
    - default: `MountainCar`
    - options: `MountainCar, MountainCarContinuous,CartPole, LunarLander, Acrobot, Pendulum`
- `--train` or `-t` to train the model
    - default: `t` (True)
    - options: `t, f` (True, False)
- `--render` or `-r` to render the model
    - default: `t` (True)
    - options: `t, f` (True, False)
- `--seed` or `-s` to specify the seed. If rendering without training, you need to set the seed of the trained model
    - default: `0`
- `--verbose` or `-v` to print the progress of the training
    - default: `t` (True)
    - options: `t, f` (True, False)
- `--help` or `-h` to get help

For example, to run the `CartPole-v1` environment without rendering the model, you can use the following command:
```
python CAT-RL.py -r f -e CartPole
```
or to just render a trained model with seed 123 from the `CartPole-v1` environment, you can use the following command:
```
python CAT-RL.py -t f -e CartPole -s 123
```

## Tile Coding
about...

### Running the code
To run the code, you can use the following command:
```
python tileCoding.py
```
This will run the code and render the model after training.

There are also some optional arguments you can use:
- `--env` or `-e` to specify the environment you want to run 
    - default: `MountainCar`
    - options: `MountainCar, MountainCarContinuous,CartPole, LunarLander, Acrobot, Pendulum`
- `--train` or `-t` to train the model
    - default: `t` (True)
    - options: `t, f` (True, False)
- `--render` or `-r` to render the model
    - default: `t` (True)
    - options: `t, f` (True, False)
- `--seed` or `-s` to specify the seed. If rendering without training, you need to set the seed of the trained model
    - default: `0`
- `--verbose` or `-v` to print the progress of the training
    - default: `t` (True)
    - options: `t, f` (True, False)
- `--help` or `-h` to get help

For example, to run the `CartPole-v1` environment without rendering the model, you can use the following command:
```
python tileCoding.py -r f -e CartPole
```

or to just render a trained model with seed 123 from the `CartPole-v1` environment, you can use the following command:
```
python tileCoding.py -t f -e CartPole -s 123
``` 

## Bins 
about...

### Running the code
To run the code, you can use the following command:
```
python binQlearning.py
```

This will run the code and render the model after training.

There are also some optional arguments you can use:
- `--env` or `-e` to specify the environment you want to run 
    - default: `MountainCar`
    - options: `MountainCar, MountainCarContinuous,CartPole, LunarLander, Acrobot, Pendulum`
- `--train` or `-t` to train the model
    - default: `t` (True)
    - options: `t, f` (True, False)
- `--render` or `-r` to render the model
    - default: `t` (True)
    - options: `t, f` (True, False)
- `--seed` or `-s` to specify the seed. If rendering without training, you need to set the seed of the trained model
    - default: `0`
- `--verbose` or `-v` to print the progress of the training
    - default: `t` (True)
    - options: `t, f` (True, False)
- `--help` or `-h` to get help

For example, to run the `CartPole-v1` environment without rendering the model, you can use the following command:
```
python binQlearning.py -r f -e CartPole
```

or to just render a trained model with seed 123 from the `CartPole-v1` environment, you can use the following command:
```
python binQlearning.py -t f -e CartPole -s 123
```

## Running experiments
To run the experiments, you can use the following command:
```
python run_exp.py
```
By default, this will run each algorithm 20 times for all the environments. The results will be saved in the `results` folder and the models will be saved in the `models` folder.

There are also some optional arguments you can use:
- `--num` or `-n`: specify the number of times to run each algorithm for each environment
    - default: `10`

# Trained models

The trained models for the different environments can be found in the `models` folder. The models are saved as `.pkl` files and can be loaded using the `pickle` library in Python.

<!-- # Docker stuff (might not need)
Make sure you have docker installed on your machine. If not, you can download it with brew using the following command:
```
brew install docker
```

Install the required packages using the following command:
```
pip install -r requirements.txt
```

## Docker
Make sure you have docker installed on your machine.

To run the application using docker, you can use the following command:
```
docker build -t bachelor-project .
```

Then you can run the application using the following command:
```
docker run --rm bachelor-project
```

or run it interactively using the following command:
```
docker run -it bachelor-project /bin/bash
```

To run the bash script `run.sh` in the interactive environment
```
./run.sh
```

The exit the interactive mode press `ctrl + p` followed by `ctrl + q`.
or press `ctrl + d` to exit the session -->