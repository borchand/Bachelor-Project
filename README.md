# Bachelor-Project


## Getting Started
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
docker run -rm bachelor-project
```

or run it interactively using the following command:
```
docker run -it bachelor-project /bin/bash
```

The exit the interactive mode press `ctrl + p` followed by `ctrl + q`.