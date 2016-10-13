# tiny-dqn

A very short & simple python implementation of Deep Q Networks using TensorFlow and OpenAI gym. This program learns to play MsPacman. With very little change it could be made to learn just about any other Atari game.

It is based on the 2013 paper by V. Mnih _et al._, "Playing Atari with Deep Reinforcement Learning": [arXiv:1312.5602](https://arxiv.org/pdf/1312.5602v1.pdf).

The Q-networks (actor and critic) have 3 convolutional layers and two fully connected layers (including the output layer). This code implements a replay memory and ɛ-greedy policy for exploration.

## Requirements

* OpenAI gym + dependencies for the Atari environment ([installation instructions](https://github.com/openai/gym#installation))
* TensorFlow ([installation instructions](https://www.tensorflow.org/versions/master/get_started/os_setup.html))
* Numpy (`pip install numpy`)

## Usage

To train the model:

    python tiny_dqn.py -v --number-steps 10000

The model is saved to `my_dqn.ckpt` by default. To view it in action, run:

    python tiny_dqn.py --test --render

For more options:

    python tiny_dqn.py --help

## Disclaimer
This is a draft, I have not had the time to test it seriously yet. If you find any issue, please contact me or send a Pull Request.

Enjoy!