import gym
import numpy as np
import time
import random
import math

env = gym.make("CartPole-v1")
reward_list = []

state = env.reset()
total_reward = 0
for _ in np.arange(2001):
    cart_pos, cart_vel, pole_ang, pole_vel = state
    total = cart_pos*-1 + cart_vel + pole_ang + pole_vel
    action = 0 if total < 0  else 1
    state, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()
    if done: 
        print('Reward:', total_reward)
        break
env.close()