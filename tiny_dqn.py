import argparse
parser = argparse.ArgumentParser(description='Train a DQN net to play MsMacman.')
parser.add_argument('-i', '--iterations', type=int, help='number of training iterations', default=10000)
parser.add_argument('-l', '--learn-iterations', type=int, help='number of iterations between each training step', default=3)
parser.add_argument('-s', '--save-iterations', type=int, help='number of training iterations between saving each checkpoint', default=100)
parser.add_argument('-c', '--copy-iterations', type=int, help='number of training iterations between each copy of the critic to the actor', default=50)
parser.add_argument('-r', '--render', help='render training', action='store_true', default=False)
parser.add_argument('-p', '--path', help='path of the checkpoint file', default="my_dqn.ckpt")
parser.add_argument("-v", "--verbosity", action="count", help="increase output verbosity", default=0)
args = parser.parse_args()

from collections import deque
import gym
import numpy as np
import numpy.random as rnd
import os
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, fully_connected

env = gym.make("MsPacman-v0")
done = True  # env needs to be reset

# Construction phase
input_height = 80
input_width = 80
input_channels = 1  # we only look at one frame at a time, so ghosts and power pellets really are invisible when they blink
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3 
conv_activation = [tf.nn.relu] * 3
n_hidden_inputs = 64 * 10 * 10  # conv3 has 64 maps of 10x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # MsPacman has 9 actions: upper left, up, upper right, left, and so on.
initializer = tf.contrib.layers.variance_scaling_initializer() # He initialization

learning_rate = 0.01

def q_network(X_state, scope):
    prev_layer = X_state
    conv_layers = []
    with tf.variable_scope(scope) as scope:
        for n_maps, kernel_size, stride, padding, activation in zip(conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation):
            prev_layer = convolution2d(prev_layer, num_outputs=n_maps, kernel_size=kernel_size, stride=stride, padding=padding, activation_fn=activation, weights_initializer=initializer)
            conv_layers.append(prev_layer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_inputs])
        hidden = fully_connected(last_conv_layer_flat, n_hidden, activation_fn=hidden_activation, weights_initializer=initializer)
        outputs = fully_connected(hidden, n_outputs, activation_fn=None)
    trainable_vars = {var.name[len(scope.name):]: var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
    return outputs, trainable_vars

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
actor_q_values, actor_vars = q_network(X_state, scope="q_networks/actor")    # acts
critic_q_values, critic_vars = q_network(X_state, scope="q_networks/critic") # learns

copy_ops = [actor_var.assign(critic_vars[var_name])
            for var_name, actor_var in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None, 1])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = critic_q_values * tf.one_hot(X_action, n_outputs)
    cost = tf.reduce_mean(tf.square(y - q_value))
    global_step = tf.Variable(0, trainable=False, name='global_step')
    increment_global_step = tf.assign_add(global_step, 1)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(cost, global_step=global_step)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Replay memory, epsilon-greedy policy and observation preprocessing
replay_memory_size = 1000
replay_memory = deque()

def sample_memories():
    indices = rnd.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1].reshape(-1, 1), cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

epsilon_min = 0.05
epsilon_max = 1.0
epsilon_decay_steps = 100000
epsilon = epsilon_max

def epsilon_greedy(q_values, epsilon):
    if rnd.rand() < epsilon:
        return rnd.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

wall_color = np.array([228, 111, 111]).mean()
bg_color = np.array([0, 28, 136]).mean()
mspacman_color = np.array([210, 164, 74]).mean()

def preprocess_observation(obs):
    img = obs[6:166:2, ::2] # crop and downsize
    img = img.mean(axis=2) # to greyscale
    img[img == wall_color] = 0 # improve contrast
    img[img == bg_color] = 40
    img[img == mspacman_color] = 255
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    return img.reshape(80, 80, 1)

# Execution phase
n_iterations = args.iterations
learning_start_step = 1000
learning_every_n_steps = args.learn_iterations
batch_size = 50
gamma = 0.95
skip_start = 90  # skip boring iterations at the start of each game

with tf.Session() as sess:
    if os.path.isfile(args.path):
        saver.restore(sess, args.path)
    else:
        init.run()
    for iteration in range(n_iterations):
        if args.verbosity > 0:
            print("\rIteration {}/{} ({:.1f}%)\tepsilon={:.2f}\ttraining step={}".format(iteration, n_iterations, iteration * 100 / n_iterations, epsilon, global_step.eval()), end="")
        if done:
            obs = env.reset()
            for skip in range(skip_start):
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)
        if args.render:
            env.render()
        q_values = actor_q_values.eval(feed_dict={X_state: [state]})
        epsilon = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min) * global_step.eval() / epsilon_decay_steps)
        action = epsilon_greedy(q_values, epsilon)
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        if len(replay_memory)>replay_memory_size:
            replay_memory.popleft()
        state = next_state

        if iteration > learning_start_step and iteration % learning_every_n_steps == 0:
            X_state_val, X_action_val, rewards, X_next_state_val, continues = sample_memories()
            next_q_values = actor_q_values.eval(feed_dict={X_state: X_next_state_val})
            y_val = rewards + continues * gamma * np.max(next_q_values)
            training_op.run(feed_dict={X_state: X_state_val, X_action: X_action_val, y: y_val})
            if iteration % (args.save_iterations * learning_every_n_steps) == 0:
                saver.save(sess, args.path)
