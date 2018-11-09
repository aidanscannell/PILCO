import numpy as np
import tensorflow as tf
import gpflow
from pilco.models.mgpr import MGPR
float_type = gpflow.settings.dtypes.float_type


class LinearPolicy:
    """
    Linear Preliminary Policy
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 3.5.2 (pg 43)
    """

    def __init__(self, state_dim, control_dim, max_action):
        self.max_action = max_action
        self.Phi = np.random.rand(control_dim, state_dim)  # parameter matrix of weights (n, D)
        self.v = np.random.rand(1, control_dim)  # offset/bias vector (1, D )

    def predict(self, m, s):
        '''
        Predict Gaussian distribution for action given a state distribution input
        :param m: mean of the state
        :param s: variance of the state
        :return: mean (M) and variance (S) of action
        '''
        M = np.dot(self.Phi, m) + self.v
        S = np.dot(self.Phi, s).dot(self.Phi.T)
        return M, S

class RBFN(gpflow.Parameterized):
    def __init__(self, X, Y, kern):
        gpflow.Parameterized.__init__(self)
        self.X = gpflow.Param(X)
        self.Y = gpflow.Param(Y)
        self.kern = kern
        self.likelihood = gpflow.likelihoods.Gaussian()


class RBFNPolicy(MGPR):
    """
    Radial Basis Function Network Preliminary Policy
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 3.5.2 (pg 44)
    """

    def __init__(self, state_dim, control_dim, num_basis_fun, max_action):
        """
        Radial Basis Function Network
        :param state_dim: shape of input data (state_dim + control_dim)
        :param num_basis_fun: number of radial basis functions (no. hidden neurons)
        :param sigma:
        """
        MGPR.__init__(self,
                      np.random.randn(num_basis_fun, state_dim),
                      0.1 * np.random.randn(num_basis_fun, control_dim)
                      )
        # self.num_basis_fun = num_basis_fun
        # self.max_action = max_action
        for model in self.models:
            model.kern.variance = 1.0
            model.kern.variance.trainable = False
            self.max_action = max_action


    def create_models(self, X, Y):
        self.models = gpflow.params.ParamList([])
        for i in range(self.num_outputs):
            kern = gpflow.kernels.RBF(input_dim=X.shape[1], ARD=True)
            model = RBFN(X, Y, kern)
            self.models.append(model)

    def compute_action(self, m_x, s_x):
        """
        Computes action according to RBF Network policy
        :param m_x: mean of the state; 1 x state_dim
        :param s_x: variance of the state; state_dim x state_dim
        :return: mean (M) and variance (S) of action
        """
        M, S = self.predict(m_x, s_x)
        return M, S


    # def _kernel_function(self, center, data_input):
    #     """
    #     Squared Exponential Kernel
    #     :param center: center of Gaussian distribution
    #     :param data_input: a single data input (state_dim, 1)
    #     :return: Gaussian RBF
    #     """
    #     return np.exp(-self.sigma*np.linalg.norm(center-data_input)**2)
    #
    # def _calculate_interpolation_matrix(self, X):
    #     """
    #     Calculates interpolation matrix
    #     :param X: training inputs (num_samples, state_dim)
    #     :return: interpolation matrix
    #     """
    #     K = np.zeros((X.shape[0], self.num_basis_fun))
    #     for data_input_arg, data_input in enumerate(X):
    #         for center_arg, center in enumerate(self.centers):
    #             K[data_input_arg, center_arg] = self._kernel_function(center, data_input)
    #     return K
    #
    # def fit(self, X, Y):
    #     """
    #     Fits the weights (self.weights) using linear regression
    #     :param X: training inputs (num_samples, state_dim)
    #     :param Y: training targets (num_samples, control_dim)
    #     :return:
    #     """
    #     random_args = np.random.permutation(X.shape[0]).tolist()
    #     self.centers = [X[arg] for arg in random_args][:self.num_basis_fun]
    #     K = self._calculate_interpolation_matrix(X)
    #     batched_eye = tf.eye(tf.shape(X)[0], batch_shape=[Y.shape[1]], dtype=float_type)
    #     L = tf.cholesky(K + self.noise[:, None, None] * batched_eye)
    #     iK = tf.cholesky_solve(L, batched_eye)
    #     Y_ = tf.transpose(Y)[:, :, None]
    #     self.beta = tf.cholesky_solve(L, Y_)[:, :, 0]
    #     return iK, self.beta
    #     # self.weights = np.dot(np.linalg.pinv(K), Y)

    # def predict(self, m_x, s_x):
    #     """
    #
    #     :param m_x: mean of the state
    #     :param s_x: variance of the state
    #     :return:
    #     """
    #
    #     K = self._calculate_interpolation_matrix(X)
    #     # alpha**2 * np.eye()**(-0.5) * np.exp(-0.5 * (xi - mu).T  )
    #     return np.dot(K, self.weights)

    # def cetralise(self, m):
    #     return X - m

    # def predict(self, X):
    #     """
    #     Computes the predictive distribution of the policy p(π̃(x∗)) given an input state
    #     :param X: test data (num_test_samples, state_dim)
    #     :return: predictions (num_test_samples, control_dim)
    #     """
    #     K = self._calculate_interpolation_matrix(X)
    #     # alpha**2 * np.eye()**(-0.5) * np.exp(-0.5 * (xi - mu).T  )
    #     return np.dot(K, self.weights)

    # def predict(self, m_x, s_x):
    #
    #     return
    #
    # def predict(self, m_x, s_x):
    #     """
    #
    #     See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    #     Section 2.3.3 (pg 22)
    #     :param m_x: mean of the state
    #     :param s_x: variance of the state
    #     :return: predicted mean of actionM
    #     """
    #     Zeta_i =  # ζ_i := (x_i − μ)
    #     Zeta_j =
    #     iLambda_a =
    #     iLambda_b =
    #     R = np.dot(s_x, (iLambda_a + iLambda_b)) + np.eye(s_x.shape[0])  # R := Σ(Λ_a^(−1) +Λ_a^(−1))+I
    #     iR = np.linalg.inv(R)
    #     z_ij = np.dot(iLambda_a, Zeta_i) + np.dot(iLambda_a, Zeta_j)
    #
    #     n_ij_squared = 2 * (np.log(alpha_a) + np.log(alpha_b)) - \
    #                    (
    #                     np.dot(np.dot(Zeta_i.T, iLambda_a), Zeta_i) +
    #                     np.dot(np.dot(Zeta_j.T, iLambda_b), Zeta_j) +
    #                     np.dot(np.dot(np.dot(z_ij.T, iR), s_x), z_ij)
    #                    ) / 2
    #     Q_ij = np.exp(n_ij_squared) / np.sqrt(R)
    #
    #     return


    # def compute_action(self, m_x, s_x):
    #     """
    #
    #     See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    #     Section 2.3.3 (pg 22)
    #     :param m_x: mean of the state
    #     :param s_x: variance of the state
    #     :return: predicted mean of actionM
    #     """
    #     Zeta_i =  # ζ_i := (x_i − μ)
    #     Zeta_j =
    #     iLambda_a =
    #     iLambda_b =
    #     R = np.dot(s_x, (iLambda_a + iLambda_b)) + np.eye(s_x.shape[0])  # R := Σ(Λ_a^(−1) +Λ_a^(−1))+I
    #     iR = np.linalg.inv(R)
    #     z_ij = np.dot(iLambda_a, Zeta_i) + np.dot(iLambda_a, Zeta_j)
    #
    #     n_ij_squared = 2 * (np.log(alpha_a) + np.log(alpha_b)) - \
    #                    (
    #                     np.dot(np.dot(Zeta_i.T, iLambda_a), Zeta_i) +
    #                     np.dot(np.dot(Zeta_j.T, iLambda_b), Zeta_j) +
    #                     np.dot(np.dot(np.dot(z_ij.T, iR), s_x), z_ij)
    #                    ) / 2
    #     Q_ij = np.exp(n_ij_squared) / np.sqrt(R)
    #
    #     return


    # def predict(self, m, v):
    #     """
    #
    #     :param m: mean of input state (state_dim, 1)
    #     :param v: variance of input state (state_dim, 1)
    #     :return:
    #     """
    #     q = np.exp(-0.5 * (x - m).T * np.linalg.inv(v + lengthscales) * (x - m))
    #     return np.dot(self.weights, q)

    def squash(self, m, s):
        """

        :param m: mean of control input
        :param s: variance of control input
        :return: mean (M), variance (S) and input-output covariance of squashed control input
        """
        # self.max_action

        k = tf.shape(m)[1]
        max_action = self.max_action
        if max_action is None:
            max_action = tf.ones((1, k), dtype=float_type)  # squashes in [-1,1] by default
        else:
            max_action = max_action * tf.ones((1, k), dtype=float_type)

        M = max_action * tf.exp(-tf.diag_part(s) / 2) * tf.sin(m)

        lq = -(tf.diag_part(s)[:, None] + tf.diag_part(s)[None, :]) / 2
        q = tf.exp(lq)
        S = (tf.exp(lq + s) - q) * tf.cos(tf.transpose(m) - m) \
            - (tf.exp(lq - s) - q) * tf.cos(tf.transpose(m) + m)
        S = max_action * tf.transpose(max_action) * S / 2

        C = max_action * tf.diag(tf.exp(-tf.diag_part(s) / 2) * tf.cos(m))
        return M, S, tf.reshape(C, shape=[k, k])


    # def compute_action(self, m, s):
    #     """
    #
    #     :param m: mean of state x_(t-1)
    #     :param s: variance of state x_(t-1)
    #     :return: mean and variance of action p(π(x_(t−1)))
    #     """
    #     K = self._calculate_interpolation_matrix()
    #     return M, S
