import GPy
import numpy as np
from IPython.display import display


class DynamicsModel:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.state_dim = Y.shape[1]
        self.input_dim = X.shape[1]
        self.num_data_points = X.shape[0]

        self.models = []
        self.create_models()

    def create_models(self):
        for i in range(self.state_dim):
            kernel = GPy.kern.RBF(input_dim=self.input_dim, ARD=1)
            model = GPy.models.GPRegression(self.X, self.Y[:, i:i+1], kernel)
            self.models.append(model)
            display(model)

    def optimize(self):
        for model in self.models:
            model.optimize(messages=True, max_f_eval=1000)
            # display(model)

    def predict(self, m, s):
        M_x = []
        S_x = []
        for i, model in enumerate(self.models):
            m_x, s_x = self.univariate_predict(m, s, model)
            M_x.append(m_x)
            S_x.append(s_x)
        return M_x, S_x

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
        # iLambda = np.diag(model.kern.lengthscale)
        iLambda = np.linalg.inv(np.diag(model.kern.lengthscale))
        det = np.linalg.det(np.dot(s, iLambda) + np.eye(s.shape[0]))
        inv = np.linalg.inv(s + iLambda)
        q = np.empty(model.X.shape[0])
        for i, x in enumerate(model.X):
            q_i = self.calc_qi(m, x[np.newaxis], alpha, inv, det)
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

        s_star = alpha - np.trace(np.dot(iK, Q)) + np.dot(np.dot(beta.T, Q), beta) - m_star**2

        return m_star, s_star
