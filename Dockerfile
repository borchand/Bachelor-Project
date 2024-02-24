# inspired by https://github.com/TTitcombe/docker_openai_gym/blob/master/Dockerfile
# dockerfile for python3.10
FROM python:3.10

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        curl \
        xvfb \
        ffmpeg \
        xorg-dev \
        libsdl2-dev \
        swig \
        cmake

RUN pip3 install --upgrade pip

# Move the requirements file into the image
COPY requirements.txt /temp/

# Install the python requirements on the image
RUN pip3 install -r /temp/requirements.txt

# Remove the requirements file - this is no longer needed
RUN rm /temp/requirements.txt

COPY . .

ENV XDG_RUNTIME_DIR=/

CMD ["python", "-u", "/testGym.py"]
