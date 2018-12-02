import gym
import numpy as np

env = gym.make("CartPole-v1")
state = env.reset()
env.render()

episodes = 1000
steps = 200

reward_list = []
for episode in range(1, episodes + 1):
    env.reset()
    total_reward = 0
    for step in range(1, steps + 1):
        next_state, reward, done, _ = env.step(env.action_space.sample())
        print('Sample:')
        print(env.action_space.sample())
        print('Stats:')
        print(next_state, reward, done)
        total_reward += reward
        if done: break
        state = next_state
    reward_list.append(total_reward)
env.close()