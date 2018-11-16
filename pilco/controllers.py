import numpy as np
import tensorflow as tf
import GPy


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


class PseudGPR(GPy.Parameterized):

    def __init__(self, X, Y, kern, name="PseudoGPR"):
        GPy.Parameterized.__init__(self, name=name)
        self.X = GPy.Param("input", X)
        # self.add_parameter(self.X)
        self.Y = GPy.Param("target", Y)
        # self.add_parameter(self.Y)
        self.kern = kern
        self.likelihood = GPy.likelihoods.Gaussian()


class RBFNPolicy:
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
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.num_basis_fun = num_basis_fun
        self.max_action = max_action

        self.models = []
        self.create_models(np.random.randn(num_basis_fun, state_dim),
                           0.1 * np.random.randn(num_basis_fun, control_dim)
                           )

        for model in self.models:
            model.kern.variance = 1.0
            model.kern.variance.trainable = False
            self.max_action = max_action

    def create_models(self, X, Y):
        for i in range(self.control_dim):
            kernel = GPy.kern.RBF(input_dim=self.state_dim, ARD=1)
            model = PseudGPR(X, Y[:, i:i + 1], kernel)
            # model = GPy.models.GPRegression(X, Y[:, i:i+1], kernel)
            # model.likelihood = GPy.likelihoods.Gaussian()
            # model.X = GPy.Param(X)
            # model.Y = GPy.Param(Y)
            # model._add_parameter_name()
            self.models.append(model)

    def compute_action(self, m_x, s_x):
        """
        Computes action according to RBF Network policy
        :param m_x: mean of the state; 1 x state_dim
        :param s_x: variance of the state; state_dim x state_dim
        :return: mean (M) and variance (S) of action
        """
        actions = []
        # print(m_x)
        # print(m_x.shape)
        # print(m_x[np.newaxis].shape)
        for model in self.models:
            # m_u, s_u = model.predict(m_x)
            m_u, s_u = self.univariate_predict(m_x, s_x, model)
            actions.append([m_u, s_u])
        # M, S = self.predict(m_x, s_x)
        # TODO: squash using sin function
        return actions

    def calc_qi(self, m, x_i, alpha, inv, det):
        inp = x_i - m
        exp = -0.5 * np.dot(np.dot(inp, inv), inp.T)
        q = alpha * det ** (-0.5) * np.exp(exp)
        return q

    def calc_Qij(self, x_i, x_j, m, s, model, det, iLambda):
        frac = np.dot(model.kern.K(x_i, m), model.kern.K(x_j, m)) / np.sqrt(det)
        z_ij = 0.5 * (x_i + x_j)
        inp = z_ij - m
        exp1 = np.linalg.inv(s + 0.5 * iLambda)
        exp = np.dot(np.dot(np.dot(np.dot(inp, exp1), s), iLambda), inp.T)
        return frac * np.exp(exp)

    def univariate_predict(self, m, s, model):
        """

        :param m: mean of the input; (state_dim+control_dim x 1)
        :param s:
        :param model:
        :return:
        """
        # TODO - Check I have used the correct variance (it should be signal variance)
        alpha = model.kern.variance[0]  # signal variance
        iLambda = np.diag(model.kern.lengthscale)
        det = np.linalg.det(np.dot(s, iLambda) + np.eye(s.shape[0]))
        inv = np.linalg.inv(s + iLambda)
        q = np.empty(model.X.shape[0])
        for i, x in enumerate(model.X):
            q_i = self.calc_qi(m, x, alpha, inv, det)
            q[i] = q_i

        # TODO - Correct Cholesky Dec implementation
        eye = np.eye(model.X.shape[0])
        K = model.kern.K(model.X)
        L = np.linalg.cholesky(K + model.likelihood.variance * eye)
        iK = np.linalg.solve(L, eye)
        beta = np.linalg.solve(L, model.Y)

        m_star = np.dot(beta.T, q)

        Q = np.empty((model.X.shape[0], model.X.shape[0]))
        det = np.linalg.det(2 * np.dot(s, iLambda) + np.eye(s.shape[0]))
        for i, x_i in enumerate(model.X):
            for j, x_j in enumerate(model.X):
                Q_ij = self.calc_Qij(x_i[np.newaxis], x_j[np.newaxis], m, s, model, det, iLambda)
                Q[i, j] = Q_ij

        s_star = alpha - np.trace(np.dot(iK, Q)) + np.dot(np.dot(beta.T, Q), beta) - m_star ** 2

        return m_star, s_star

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

    # def squash(self, m, s):
    #     """
    #
    #     :param m: mean of control input
    #     :param s: variance of control input
    #     :return: mean (M), variance (S) and input-output covariance of squashed control input
    #     """
    #     # self.max_action
    #
    #     k = tf.shape(m)[1]
    #     max_action = self.max_action
    #     if max_action is None:
    #         max_action = tf.ones((1, k), dtype=float_type)  # squashes in [-1,1] by default
    #     else:
    #         max_action = max_action * tf.ones((1, k), dtype=float_type)
    #
    #     M = max_action * tf.exp(-tf.diag_part(s) / 2) * tf.sin(m)
    #
    #     lq = -(tf.diag_part(s)[:, None] + tf.diag_part(s)[None, :]) / 2
    #     q = tf.exp(lq)
    #     S = (tf.exp(lq + s) - q) * tf.cos(tf.transpose(m) - m) \
    #         - (tf.exp(lq - s) - q) * tf.cos(tf.transpose(m) + m)
    #     S = max_action * tf.transpose(max_action) * S / 2
    #
    #     C = max_action * tf.diag(tf.exp(-tf.diag_part(s) / 2) * tf.cos(m))
    #     return M, S, tf.reshape(C, shape=[k, k])


    # def compute_action(self, m, s):
    #     """
    #
    #     :param m: mean of state x_(t-1)
    #     :param s: variance of state x_(t-1)
    #     :return: mean and variance of action p(π(x_(t−1)))
    #     """
    #     K = self._calculate_interpolation_matrix()
    #     return M, S
