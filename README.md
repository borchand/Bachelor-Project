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









# Docker stuff (might not need)
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
or press `ctrl + d` to exit the session