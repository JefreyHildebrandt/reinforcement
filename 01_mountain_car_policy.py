import gym
import tensorflow as tf
import numpy as np
import time

env = gym.make('MountainCar-v0')
state = env.reset()
env.render()
time.sleep(0.1)

prev = -1
for _ in np.arange(200):
    action = 0 if prev < 0 else 2
    state, reward, done, other = env.step(action)
    prev = state[1]
    env.render()
    time.sleep(0.1)
    if done:
        break
env.close()