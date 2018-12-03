import gym
import tensorflow as tf
import numpy as np
import math

# %matplotlib inline
# import matplotlib.pyplot as plt

# Create the Cart-Pole game environment
env = gym.make('MountainCar-v0')

# Environment parameters
state_size = 2
action_size = 3

hidden_layer_size = 128

batch_size = 25

learning_rate = 0.01

max_episodes = 100

max_steps = 200
percentile = 70

class Net:
    def __init__(self, 
                 state_size = state_size, 
                 action_size = action_size, 
                 hidden_layer_size = hidden_layer_size,
                 learning_rate = learning_rate, 
                 name = 'net'):
        
        with tf.variable_scope(name):
        
            ### Prediction part
        
            # Input layer, state s is input
            self.states = tf.placeholder(
                tf.float32, 
                [None, state_size])
            
            # Hidden layer, ReLU activation
            self.hidden_layer = tf.contrib.layers.fully_connected(
                self.states, 
                hidden_layer_size)
            
            # Hidden layer, linear activation, logits
            self.logits = tf.contrib.layers.fully_connected(
                self.hidden_layer, 
                action_size,
                activation_fn = None)
            
            # Output layer, softmax activation yields probability distribution for actions
            self.probabilities = tf.nn.softmax(self.logits)
    
            ### Training part 
    
            # Action a
            self.actions = tf.placeholder(
                tf.int32, 
                [None])
            
            # One-hot encoded action a 
            #
            # encoded_action_vector = [1, 0] if action a = 0
            # encoded_action_vector = [0, 1] if action a = 1
            self.one_hot_actions = tf.one_hot(
                self.actions, 
                action_size)

            # cross entropy
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits = self.logits, 
                labels = self.one_hot_actions)
            
            # cost
            self.cost = tf.reduce_mean(self.cross_entropy)
            
            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
            
    # get action chosen according to current probabilistic policy
    def get_action(self, state):
        feed_dict = { self.states : np.array([state]) } 
        probabilities = sess.run(self.probabilities, feed_dict = feed_dict)
        return np.random.choice(action_size, p=probabilities[0])
    
    # train based on batch
    def train(self, batch):
        states, actions = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        
        feed_dict = {
            self.states : states,
            self.actions : actions
        }
        
        sess.run(self.optimizer, feed_dict = feed_dict)

tf.reset_default_graph()
net = Net(name = 'net',
          hidden_layer_size = hidden_layer_size,
          learning_rate = learning_rate)

import random
import bisect
import time

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    start_index = int(max_episodes * percentile / 100)
    count = 0
    is_passing = False
    while True:
        count += 1
        total_reward_list = []
        trajectory_list = []

        for e in np.arange(max_episodes):
            total_reward = 0.0
            trajectory = []
            state = env.reset()
            prev_vel = 0.0
            for s in np.arange(max_steps):
                action = net.get_action(state)
                next_state, reward, done, _ = env.step(action)
                reward = abs(prev_vel-next_state[0])
                prev_vel = next_state[0]
                total_reward += reward
                trajectory.append((state, action))
                state = next_state
                if done: break

            index = bisect.bisect(total_reward_list, total_reward)
            total_reward_list.insert(index, total_reward)
            trajectory_list.insert(index, trajectory)
        
        # keep the elite episodes, that is, throw out the bad ones 
        # train on state action pairs extracted from the elite episodes
        # this code is not optimized, it can be cleaned up 
        state_action_pairs = []
        for trajectory in trajectory_list[start_index:]:
            for state_action_pair in trajectory:
                state_action_pairs.append(state_action_pair)
        # shuffle to avoid correlations between adjacent states
        random.shuffle(state_action_pairs) 
        n = len(state_action_pairs)
        batches = [state_action_pairs[k:k + batch_size] for k in np.arange(0, n, batch_size)]

        for batch in batches:
            net.train(batch)

        # test agent
        state = env.reset()
        # env.render()
        # time.sleep(0.05)
        total_reward = 0.0
        for s in np.arange(max_steps):
            action = net.get_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if is_passing:
                env.render()
            # time.sleep(0.05)
            if done: break

        env.close()
        print("Total reward:", total_reward, ' Count:', count)
        
        if total_reward > -200:
            is_passing = True
            print("Reached")   