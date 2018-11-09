import gym
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pilco.controllers import RBFNPolicy
from pilco.models.pilco import PILCO

env = gym.make('InvertedPendulum-v2')
env.reset()

def rollout(policy, timesteps):
    """
    Performs a rollout of a given policy on the environment for a given number of timesteps
    :param policy: function representing policy to rollout
    :param timesteps: the number of timesteps to perform
    :return: X -> list of training inputs - (x, u) where x=state and u=control
             Y -> list of training targets - differences y = x(t) - x(t-1)
    """
    X = []
    Y = []
    env.reset()
    x, _, _, _ = env.step(0)
    for t in range(timesteps):
        # env.render()
        u = policy(x)
        # print("Performing action: %f" % u)
        x_new, _, done, _ = env.step(u)
        # print(x_new)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
    return np.stack(X), np.stack(Y)


def random_policy(x):
    """
    Random policy that samples a random action from the action space
    :param x: state
    :return: action
    """
    return env.action_space.sample()

def pilco_policy(x):
    return pilco.compute_action(x)

X, Y = rollout(policy=random_policy, timesteps=40)
for i in range(1, 3):
    X_, Y_ = rollout(policy=random_policy, timesteps=40)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
policy = RBFNPolicy(state_dim, control_dim=control_dim, num_basis_fun=50, max_action=env.action_space.high[0])

pilco = PILCO(X, Y, policy=policy, horizon=40)

pilco.optimize()

rollout(policy=pilco_policy, timesteps=100)


# print(Y)
# predictions = pilco.policy.predict(X[0, :])
# print(predictions)

# for i_episode in range(20):
#     pilco.optimize()
#     X_new, Y_new = rollout(policy=pilco_policy, timesteps=100)
#     X_new = X_new.reshape((X_new.shape[0], X_new.shape[2]))
#     # print(X_new.shape)
#     Y_new = Y_new.reshape((Y_new.shape[0], Y_new.shape[2]))
#     # print(Y_new.shape)
#     X = np.vstack((X, X_new))
#     Y = np.vstack((Y, Y_new))
#
#
#     # Update dataset
#     # X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
#     pilco.dynamics_model.set_XY(X, Y)
#
#
