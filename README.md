# tiny-dqn

A very short & simple python implementation of Deep Q Networks using TensorFlow and OpenAI gym. This program learns to play MsPacman. With very little change it could be made to learn just about any other Atari game.

It is based on the 2013 paper by V. Mnih _et al._, "Playing Atari with Deep Reinforcement Learning": [arXiv:1312.5602](https://arxiv.org/pdf/1312.5602v1.pdf).

The two Q-networks (online DQN and target DQN) have 3 convolutional layers and two fully connected layers (including the output layer). This code implements a replay memory and ɛ-greedy policy for exploration.

## Requirements

* OpenAI gym + dependencies for the Atari environment
* TensorFlow 1.0+
* Numpy

## Installation

### On MacOSX

    $ brew install cmake boost boost-python sdl2 swig wget
    $ cd $your_work_directory
    $ git clone https://github.com/ageron/tiny-dqn.git
    $ cd tiny-dqn
    $ pip install --user --upgrade pip
    $ pip install --user --upgrade -r requirements.txt
    $ python tiny_dqn.py -v --render

### On Ubuntu 14.04

    $ apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
    $ cd $your_work_directory
    $ git clone https://github.com/ageron/tiny-dqn.git
    $ cd tiny-dqn
    $ vim requirements.txt
    $ pip install --user --upgrade pip
    $ pip install --user --upgrade -r requirements.txt
    $ python tiny_dqn.py -v --render

## Usage

To train the model:

    python tiny_dqn.py -v --number-steps 1000000

The model is saved to `my_dqn.ckpt` by default. To view it in action, run:

    python tiny_dqn.py --test --render

For more options:

    python tiny_dqn.py --help

## Disclaimer
This is a draft, I have not had the time to test it seriously yet. If you find any issue, please contact me or send a Pull Request.

Enjoy!