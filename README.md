# Bachelor-Project


## Getting Started
To run the code in this repository, you need to have Python...

Some tasks from OpenAI Gym uses `Box2D`. For this to work you need to isnstall swig. You can do this using brew with the following command:

```
brew install swig
```

When you have swig installed, you can install the required packages using the following command:
```
pip install -r requirements.txt
```


## CAT_RL
about...

### Environments
This repository contains the following environments:
- `CartPole-v1`
- `MountainCar-v0`
- `LunarLander-v2`
- `Acrobot-v1`

### Running the code
To run the code, you can use the following command:
```
python CAT-RL.py
```
This will run all the environments and render the model after training.

There are also some optional arguments you can use:
- `--env` or `-e` to specify the environment you want to run 
    - default: `MountainCar`
    - options: `CartPole, MountainCar, LunarLander, Acrobot, MountainCarContinuous`
- `--train` or `-t` to train the model
    - default: `t` (True)
    - options: `t, f` (True, False)
- `--render` or `-r` to render the model
    - default: `t` (True)
    - options: `t, f` (True, False)
- `--seed` or `-s` to specify the seed. If rendering without training, you need to set the seed of the trained model
    - default: `0
- `--help` or `-h` to get help

For example, to run the `CartPole-v1` environment without rendering the model, you can use the following command:
```
python CAT-RL.py -r f -e CartPole
```
or to just render a trained model with seed 123 from the `CartPole-v1` environment, you can use the following command:
```
python CAT-RL.py -t f -e CartPole -s 123
```

# Trained models

The trained models for the different environments can be found in the `models` folder. The models are saved as `.pkl` files and can be loaded using the `pickle` library in Python.

## CAT_RL

### CartPole-v1

### MountainCar-v0

### LunarLander-v2

### Acrobot-v1

### MountainCarContinuous-v0

### Pendulum-v0


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