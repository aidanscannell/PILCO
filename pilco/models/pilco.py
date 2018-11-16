import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import time
from pilco.models.dynamics_model import DynamicsModel
from IPython.display import display
# import scipy.optimize as opt
from GPy import Model, Param


class PILCO(Model):

    def __init__(self, X, Y, policy, cost, horizon=30, name="RBFController", m_init=None, s_init=None):
        super(PILCO, self).__init__(name)
        self.state_dim = Y.shape[1]
        self.control_dim = X.shape[1] - Y.shape[1]
        self.horizon = horizon

        if m_init is None or s_init is None:
            self.m_init = X[0:1, 0:self.state_dim]
            self.s_init = np.diag(np.ones(self.state_dim) * 0.1)
        else:
            self.m_init = m_init
            self.s_init = s_init

        self.dynamics_model = DynamicsModel(X, Y)
        self.policy = policy
        self.cost = cost

    def optimize1(self):
        print("Optimizing")
        start = time.time()
        self.dynamics_model.optimize()
        end = time.time()
        print("Optimized GP's for dynamics model in %.1f s" % (end-start))

        start = time.time()
        self.optimize()
        end = time.time()
        print("Finished with Controller's optimization in %.1f seconds" % (end - start))

        # self.predict(self.m_init, self.s_init)
        # print("Optimizing policy")
        # opt.fmin_cg(self.objective_f, x0, fprime=self.objective_dfx, args=args)

        # print('\n-----Learned models------')
        # for model in self.dynamics_model.models:
        #     display(model)

    def compute_action(self, m_x):
        """
        Computes the action according to PILCO
        :param m_x: mean of state; 1 x state_dim
        :return:
        """
        return self.policy.compute_action(m_x, np.zeros((self.state_dim, self.state_dim)))[0][0][0]
        # self.policy.compute_action(m_x, tf.zeros([self.state_dim, self.state_dim], float_type))

    def log_likelihood(self):
        # This is for tuning controller's parameters
        return self.predict(self.m_init, self.s_init)[2]

    # def parameters_changed(self):


    # def parameters_changed(self):
    #     self.X.gradient = -scipy.optimize.rosen_der(self.X)

    # def objective_f(self, x, *args):
    #     self.policy.
    #     lengthscales, variances,
    #     return self.predict(self.m_init, self.s_init)
    #
    def objective_dfx(self, x, *args):
        return



    def dMdPhi(self):
        return

    def dSdPhi(self):
        return


    def predict(self, m_x, s_x):
        """
        Simulates policy on dynamics model and calculates the value function (sum of expected immediate costs)
        TODO: Add cost variance component to value function
        :param m_x: mean of state
        :param s_x: variance of state
        :return:
        """
        t = 0
        cost = 0
        # print("Starting internal simulation")
        while t < self.horizon:
            m_x, s_x = self.propagate(m_x, s_x)
            cost += self.cost.compute_cost(m_x, s_x)[0]
            # print("Cost at timestep %i is %f" % (t, cost))
            t += 1
        return m_x, s_x, cost

    def propagate(self, m_x, s_x):
        """
        Simulate the policy using the learnt dynamics model
        :param m_x: mean of state
        :param s_x: variance of state
        :return:
        """
        actions = self.policy.compute_action(m_x, s_x)

        m_u = actions[0][0]
        s_u = actions[0][1]

        m = np.concatenate((m_x, m_u), axis=None)
        m = m[np.newaxis]
        s1 = np.concatenate((s_x, np.zeros((1, s_x.shape[0]))), axis=0)
        s2 = np.concatenate((np.zeros((s_x.shape[0], 1)), s_u), axis=None)
        s = np.concatenate((s1.T, s2[np.newaxis]), axis=0)

        # predict new state distribution
        m_dx, s_dx = self.dynamics_model.predict(m, s)

        m_new = np.copy(m_x)
        s_new = np.copy(s_x)
        for i, m_ in enumerate(m_dx):
            m_new[:, i] += m_

        for i, s_ in enumerate(s_dx):
            s_new[i, i] += s_

        return m_new, s_new
