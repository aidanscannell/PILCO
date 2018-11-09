import gpflow
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pilco.models.mgpr import MGPR
import time
import pandas as pd
import tensorflow as tf

float_type = gpflow.settings.dtypes.float_type


class PILCO:

    def __init__(self, X, Y, policy, horizon=30, name=None):
        # super(PILCO, self).__init__(name)
        self.state_dim = Y.shape[1]
        self.control_dim = X.shape[1] - Y.shape[1]
        self.horizon = horizon

        self.dynamics_model = MGPR(X, Y)
        self.policy = policy

    def optimize(self):
        print("Optimizing")
        start = time.time()
        self.dynamics_model.optimize()
        end = time.time()
        print("Optimized GP's for dynamics model in %.1f s" % (end-start))

        lengthscales = {}
        variances = {}
        noises = {}
        i = 0
        for model in self.dynamics_model.models:
            lengthscales['GP' + str(i)] = model.kern.lengthscales.value
            variances['GP' + str(i)] = np.array([model.kern.variance.value])
            noises['GP' + str(i)] = np.array([model.likelihood.variance.value])
            i += 1

        print('-----Learned models------')
        pd.set_option('precision', 3)
        print('---Lengthscales---')
        print(pd.DataFrame(data=lengthscales))
        print('---Variances---')
        print(pd.DataFrame(data=variances))
        print('---Noises---')
        print(pd.DataFrame(data=noises))

    def compute_action(self, m_x):
        """
        Computes the action according to PILCO
        :param m_x: mean of state; 1 x state_dim
        :return:
        """
        self.policy.compute_action(m_x, tf.zeros([self.state_dim, self.state_dim], float_type))
        return

        # def action(self, X):
    #     X.shape = (1, -1)
    #     return self.policy.predict(X)

    # def compute_action(self, x):
    #     return self.policy.compute_action(x, tf.zeros([self.state_dim, self.state_dim], float_type))


    # def plot(slef, m, X, Y):
    #     xx = np.linspace(X.min(), X.max(), 100).reshape(100, 1)
    #     # xx = np.linspace(Y.min(), Y.max(), 100).reshape(100, 1)
    #     # xx = np.linspace(-1.0, 1.0, 100).reshape(100, 1)
    #
    #     # xx = np.array(
    #     #     [np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
    #     #      np.linspace(X[:, 1].min(), X[:, 1].max(), 100),
    #     #      np.linspace(X[:, 2].min(), X[:, 2].max(), 100),
    #     #      np.linspace(X[:, 3].min(), X[:, 3].max(), 100),
    #     #      np.linspace(X[:, 4].min(), X[:, 4].max(), 100)]).T
    #
    #     print(xx.shape)
    #     mean, var = m.predict_y(xx)
    #     mean = mean + xx
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(X, Y, 'kx', mew=2, label='Data points')
    #     plt.plot(xx, mean, 'C0', lw=2, label='Mean of posterior')
    #     plt.fill_between(xx[:, 0],
    #                      mean[:, 0] - 2 * np.sqrt(var[:, 0]),
    #                      mean[:, 0] + 2 * np.sqrt(var[:, 0]),
    #                      color='C0', alpha=0.2,
    #                      label='$\pm$2 standard deviations of posterior')
    #     # plt.xlim(-1.0, 1.0)
    #     plt.legend(fontsize=15)
    #     plt.show()
    #
    # def plotDynamics(self, X, Y):
    #     for index, m in enumerate(self.dynamics_model.models):
    #         self.plot(m, X[:, index:index + 1], Y[:, index:index + 1])





# class Pilco(gpflow.models.Model):
#     def __init__(self, X, Y, num_induced_points=None, horizon=30, policy=None,
#                  reward=None, m_init=None, S_init=None, name=None):
#         super(Pilco, self).__init__(name)
#         if not num_induced_points:
#             self.mgpr = DynamicsModel(X, Y)
#         else:
#             self.mgpr = DynamicsModel(X, Y, num_induced_points)
#         self.state_dim = Y.shape[1]
#         self.control_dim = X.shape[1] - Y.shape[1]
#         self.horizon = horizon
#
#         # if policy is None:
#         #     # self.policy = policys.Linearpolicy(self.state_dim, self.control_dim)
#         # else:
#         self.policy = policy
#
#         if reward is None:
#             self.reward = ExponentialReward(self.state_dim)
#         else:
#             self.reward = reward
#
#         if m_init is None or S_init is None:
#             # If the user has not provided an initial state for the rollouts,
#             # then define it as the first state in the dataset.
#             self.m_init = X[0:1, 0:self.state_dim]
#             self.S_init = np.diag(np.ones(self.state_dim) * 0.1)
#         else:
#             self.m_init = m_init
#             self.S_init = S_init
#
#     @gpflow.name_scope('likelihood')
#     def _build_likelihood(self):
#         # This is for tuning policy's parameters
#         reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
#         return reward
#
#     def optimize(self):
#         '''
#         Optimizes both GP's and policy's hypeparamemeters.
#         '''
#         import time
#         start = time.time()
#         self.mgpr.optimize()
#         end = time.time()
#         print("Finished with GPs' optimization in %.1f seconds" % (end - start))
#
#     def predict(self, m_x, s_x, n):
#         loop_vars = [
#             tf.constant(0, tf.int32),
#             m_x,
#             s_x,
#             tf.constant([[0]], float_type)
#         ]
#
#         _, m_x, s_x, reward = tf.while_loop(
#             # Termination condition
#             lambda j, m_x, s_x, reward: j < n,
#             # Body function
#             lambda j, m_x, s_x, reward: (
#                 j + 1,
#                 *self.propagate(m_x, s_x),
#                 tf.add(reward, self.reward.compute_reward(m_x, s_x)[0])
#             ), loop_vars
#         )
#
#         return m_x, s_x, reward
#
#     def propagate(self, m_x, s_x):
#         m_u, s_u, c_xu = self.policy.compute_action(m_x, s_x)
#
#         m = tf.concat([m_x, m_u], axis=1)
#         s1 = tf.concat([s_x, s_x@c_xu], axis=1)
#         s2 = tf.concat([tf.transpose(s_x@c_xu), s_u], axis=1)
#         s = tf.concat([s1, s2], axis=0)
#
#         M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
#         M_x = M_dx + m_x
#         #TODO: cleanup the following line
#         S_x = S_dx + s_x + s1@C_dx + tf.matmul(C_dx, s1, transpose_a=True, transpose_b=True)
#
#         # While-loop requires the shapes of the outputs to be fixed
#         M_x.set_shape([1, self.state_dim]); S_x.set_shape([self.state_dim, self.state_dim])
#         return M_x, S_x







    # def optimize(self):
    #     optimizer = gpflow.train.ScipyOptimizer(options={'maxfun': 500})
    #     for model in self.models:
    #         model.likelihood.variance = 0.01
    #         optimizer.minimize(model)

    # def plotkernelsample(k, ax, xmin=-3, xmax=3):
    #     xx = np.linspace(xmin, xmax, 100)[:, None]
    #     K = k.compute_K_symm(xx)
    #     ax.plot(xx, np.random.multivariate_normal(np.zeros(100), K, 3).T)
    #     ax.set_title(k.__class__.__name__)
    #
    # def plotkernelfunction(K, ax, xmin=-3, xmax=3, other=0):
    #     xx = np.linspace(xmin, xmax, 100)[:, None]
    #     K = k.compute_K_symm(xx)
    #     ax.plot(xx, k.compute_K(xx, np.zeros((1, 1)) + other))
    #     ax.set_title(k.__class__.__name__ + ' k(x, %f)' % other)

    # def plot_kernels(self, X, Y):
    #     for m in self.models:
    #         xx = np.linspace(0, 1.1, 100).reshape(100, 1)
    #         mean, var = m.predict_y(xx)
    #         plt.plot(X, Y, 'kx', mew=2)
    #         line, = plt.plot(xx, mean, lw=2)
    #         _ = plt.fill_between(xx[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]), mean[:, 0] + 2 * np.sqrt(var[:, 0]),
    #                              color=line.get_color(), alpha=0.2)
    #         plt.show()
    #
    # def plot(self, X, Y):
    #     for m in self.models:
    #         xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)
    #         mean, var = m.predict_y(xx)
    #         plt.figure(figsize=(12, 6))
    #         plt.plot(X, Y, 'kx', mew=2)
    #         plt.plot(xx, mean, 'C0', lw=2)
    #         plt.fill_between(xx[:, 0],
    #                          mean[:, 0] - 2 * np.sqrt(var[:, 0]),
    #                          mean[:, 0] + 2 * np.sqrt(var[:, 0]),
    #                          color='C0', alpha=0.2)
    #         plt.xlim(-0.1, 1.1)
    #         plt.show()
