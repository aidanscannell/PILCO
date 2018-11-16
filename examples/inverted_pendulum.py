import gym
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pilco.controllers import RBFNPolicy
from pilco.models.pilco import PILCO
from pilco.cost_function import SaturatingCost

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
    x = np.append(x, np.sin(x[2]))
    x = np.append(x, np.cos(x[2]))
    x = np.delete(x, 2)
    for t in range(timesteps):
        # env.render()
        u = policy(x)
        # print("Performing action: %f" % u)
        x_new, reward, done, _ = env.step(u)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        x_new = np.append(x_new, np.sin(x_new[2]))
        x_new = np.append(x_new, np.cos(x_new[2]))
        x_new = np.delete(x_new, 2)
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


def cost(a, l):
    # TODO: make j_target dynamic (it is -1 because of choice of a and l)
    j_target = np.array([0, 0, -1])
    # j = np.array([x[0], np.sin(x[2]), np.cos(x[2])])
    C = np.array([[1., l, 0.0], [0., 0., l]])
    iT = a ** (-2) * np.dot(C.T, C)
    return j_target, iT

# def j(x):
#     return np.array([x[0, 0], np.sin(x[0, 2]), np.cos(x[0, 2])])

X, Y = rollout(policy=random_policy, timesteps=40)
for i in range(1, 3):
    X_, Y_ = rollout(policy=random_policy, timesteps=40)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
policy = RBFNPolicy(state_dim, control_dim=control_dim, num_basis_fun=50, max_action=env.action_space.high[0])

a = 0.25
l = 0.6
j_target, iT = cost(0.25, 0.6)
idxs = [0, 3, 4]
cost = SaturatingCost(j_target, iT, idxs)

pilco = PILCO(X, Y, policy=policy, cost=cost, horizon=40)

pilco.optimize1()

# pilco.predict()



# rollout(policy=pilco_policy, timesteps=100)


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
