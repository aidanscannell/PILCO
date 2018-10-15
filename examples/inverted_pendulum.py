import gym
import numpy as np

env = gym.make("CartPole-v0")
env.reset()

def rollout(policy, timesteps):
    ########################################################################################
    # rollout - Performs a rollout of a given policy on the environment for a given number
    #           of timesteps
    #   Inputs:
    #       policy -> function representing policy to rollout
    #       timesteps - > the number of timesteps to perform
    #   Outputs:
    #      X -> list of training inputs - (x, u) where x=state and u=control
    #      Y -> list of training targets - differences y = x(t) - x(t-1)
    ########################################################################################
    X = []
    Y = []
    env.reset()
    x, _, _, _ = env.step(0)
    for t in range(timesteps):
        env.render()
        u = policy(x)
        x_new, _, done, _ = env.step(u)
        if done:
            break
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
    return np.stack(X), np.stack(Y)


def random_policy(x):
    ########################################################################################
    # random_policy - Calculates GP prior and returns sampled functions
    #   Inputs:
    #       policy -> function representing policy to rollout
    #       timesteps - > the number of timesteps to simulate
    #   Outputs:
    #      X -> list of training inputs - (x, u) where x=state and u=control
    #      Y -> list of training targets - differences y = x(t) - x(t-1)
    ########################################################################################
    return env.action_space.sample()


X, Y = rollout(policy=random_policy, timesteps=50)
for i in range(1, 3):
    X_, Y_ = rollout(policy=random_policy, timesteps=50)
    np.vstack((X, X_))
    np.vstack((Y, Y_))



# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(20):
#         env.render()
#         print(observation)
#         # action = env.action_space.sample()
#         action = 1
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break