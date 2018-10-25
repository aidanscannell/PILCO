import numpy as np


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

class RBFNPolicy(object):
    """
    Radial Basis Function Network Preliminary Policy
    See Deisenroth et al 2015: Gaussian Processes for Data-Efficient Learning in Robotics and Control
    Section 3.5.2 (pg 44)
    """

    def __init__(self, input_dim, num_basis_fun, sigma=1.0):
        """
        Radial Basis Function Network
        :param input_dim: shape of input data (state_dim + control_dim)
        :param num_basis_fun: number of radial basis functions (no. hidden neurons)
        :param sigma:
        """
        self.input_dim = input_dim
        self.num_basis_fun = num_basis_fun
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_input):
        """
        Squared Exponential Kernel
        :param center: center of Gaussian distribution
        :param data_input: a single data input (input_dim, 1)
        :return: Gaussian RBF
        """
        return np.exp(-self.sigma*np.linalg.norm(center-data_input)**2)

    def _calculate_interpolation_matrix(self, X):
        """
        Calculates interpolation matrix
        :param X: training inputs (num_samples, input_dim)
        :return: interpolation matrix
        """
        K = np.zeros((X.shape[0], self.num_basis_fun))
        for data_input_arg, data_input in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                K[data_input_arg, center_arg] = self._kernel_function(center, data_input)
        return K

    def fit(self, X, Y):
        """
        Fits the weights (self.weights) using linear regression
        :param X: training inputs (num_samples, input_dim)
        :param Y: training targets (num_samples, output_dim)
        :return:
        """
        random_args = np.random.permutation(X.shape[0]).tolist()
        self.centers = [X[arg] for arg in random_args][:self.num_basis_fun]
        K = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(K), Y)

    def predict(self, X):
        """
        Computes the predictive distribution of the policy p(π̃(x∗)) given an input state
        :param X: test data (num_test_samples, input_dim)
        :return: predictions (num_test_samples, output_dim)
        """
        K = self._calculate_interpolation_matrix(X)
        return np.dot(K, self.weights)