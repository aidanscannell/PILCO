import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import time
from pilco.models.dynamics_model import DynamicsModel
from IPython.display import display
# import scipy.optimize as opt
from GPy import Model, Param
import gpflow
float_type = gpflow.settings.dtypes.float_type
from decimal import *


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
        self.link_parameter(self.policy)
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

    # def log_likelihood(self):
    #     # This is for tuning controller's parameters
    #     return self.predict(self.m_init, self.s_init)[2]
    def objective_function(self):
        return self.predict(self.m_init, self.s_init)[2]

    def optimize(self):
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
        cost = Decimal(0)
        print("Starting internal simulation")
        while t < self.horizon:
            m_x, s_x = self.propagate(m_x, s_x)
            for i in range(m_x.shape[1]):
                print(m_x[0, i])
                m_x[0, i] = Decimal(m_x[0, i])
                for j in range(m_x.shape[1]):
                    s_x[i, j] = Decimal(s_x[i, j])
            # print("m_x")
            # print(m_x)
            # print("s_x")
            # print(s_x)
            # print(self.cost.compute_cost(m_x, s_x)[0][0])
            cost += Decimal(self.cost.compute_cost(m_x, s_x)[0][0][0])
            print("Cost at timestep %i is %f" % (t, cost))
            t += 1
        return m_x, s_x, cost

    def propagate(self, m_x, s_x):
        """
        Simulate the policy using the learnt dynamics model
        :param m_x: mean of state
        :param s_x: variance of state
        :return:
        """
        print("input proagate m/s")
        print(m_x)
        print(s_x)
        m_u, s_u, c_xu = self.policy.compute_action(m_x, s_x)

        print("C_XU")
        print(c_xu.shape)

        # m_u = actions[0][0]
        # s_u = actions[0][1]
        print("control m/s")
        print(m_u.shape)
        print(s_u.shape)

        m = np.concatenate((m_x, m_u), axis=None)
        print(m.shape)
        m = m[np.newaxis]
        print(m.shape)
        # s1 = np.concatenate((s_x, np.zeros((1, s_x.shape[0]))), axis=0)
        # print(s1.shape)
        # s2 = np.concatenate((np.zeros((s_x.shape[0], 1)), s_u), axis=None)
        # print(s2.shape)
        # s = np.concatenate((s1.T, s2[np.newaxis]), axis=0)
        s1 = np.concatenate((s_x, s_x @ c_xu), axis=1)
        s2 = np.concatenate(((s_x @ c_xu).T, s_u), axis=1)
        s = np.concatenate((s1, s2), axis=0)

        print("m/s for dynamics_model.predict")
        print(m.shape)
        print(s.shape)
        # predict new state distribution
        m_dx, s_dx, c_dx = self.dynamics_model.predict(m, s)

        print("diff m/s")
        print(m_dx)
        print(s_dx)

        m_new = np.copy(m_x)
        s_new = np.copy(s_x)
        print("m_new/s_new")
        print(m_new)
        print(s_new)
        M_x = m_x + m_dx
        S_x = s_x + s_dx + s1 @ c_dx + c_dx.T @ s1.T
        print("predicted m/s")
        print(M_x)
        print(S_x)
        return M_x, S_x
        # for i, m_ in enumerate(m_dx[0, :]):
        #     m_new[:, i] += m_
        # s_new += s_dx

        # print("predicted m/s")
        # print(m_new)
        # print(s_new)

        # return m_new, s_new
