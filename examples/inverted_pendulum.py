import gym
import numpy as np

env = gym.make("CartPole-v0")
env.reset()

for i_episode in range(20):
    observation = env.reset()
    for t in range(20):
        env.render()
        print(observation)
        # action = env.action_space.sample()
        action = 1
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break