import GPy
import numpy as np
from IPython.display import display
from scipy.linalg import cho_factor, cho_solve


class DynamicsModel:

    def __init__(self, X, Y):
        # self.X = X
        # self.Y = Y
        self.state_dim = Y.shape[1]
        self.input_dim = X.shape[1]

        self.num_data_points = X.shape[0]

        self.models = []
        self.create_models(X, Y)

    def create_models(self, X, Y):
        for i in range(self.state_dim):
            kernel = GPy.kern.RBF(input_dim=self.input_dim, ARD=1)
            model = GPy.models.GPRegression(X, Y[:, i:i+1], kernel)
            self.models.append(model)
            # display(model)

    def optimize(self):
        for model in self.models:
            model.optimize(messages=True, max_f_eval=1000)
            # display(model)

    def predict(self, m, s):
        # print("m")
        # print(m.shape)
        # print("s")
        # print(s.shape)
        s = np.tile(s[None, None, :, :], [self.state_dim, self.state_dim, 1, 1])  # TODO: remove tile
        # print(s.shape)
        # print((s[0, ...]).shape)
        # print("X.shape")
        # print(self.X.shape)
        K = self.K(self.X)  # state_dim-by-N-by-N
        # print("noise")
        # print(self.noise.shape)
        # print(self.noise[:, None].shape)
        i = np.eye(self.X.shape[0])  # N-by-N
        batched_eye = np.tile(i[None, :, :], [self.state_dim, 1, 1]) # num_output-by-N-by-N
        # print("batched")
        # print(batched_eye.shape)
        # print("eye*noise")
        # print((batched_eye * self.noise[:, None]).shape)
        A = K + batched_eye * self.noise[:, None]
        iK = np.linalg.solve(A, batched_eye)
        # AA = np.array(A[0, :, :])
        # L = np.linalg.cholesky(AA)
        # print("L")
        # print(L.shape)
        # iK = np.linalg.solve(L, batched_eye)

        # print("iK")
        # print(iK.shape)
        # print("Y")
        # print(self.Y.shape)
        # beta = np.linalg.solve(L, self.Y)[:, :, 0]
        beta = np.linalg.solve(A, self.Y)[:, :, 0]
        # print("beta")
        # print(beta.shape)

        iL = 1/self.lengthscale  # TODO: should this be squared??
        iL[iL == np.inf] = 0

        inp = np.tile(self.centralised_inputs(m)[None, :, :], [self.state_dim, 1, 1]) # TODO: remove tile
        # print("iL")
        # print(iL.shape)
        # print("inp")
        # print(inp.shape)

        iN = inp @ iL
        # print("iN")
        # print(iN.shape)
        # print("iN.T")
        # print(np.transpose(iN, [0, 2, 1]).shape)

        B = iL @ s[0, ...] @ iL + np.eye(self.input_dim)
        # print("B")
        # print(B.shape)

        t = np.transpose(
            np.conjugate(np.linalg.solve(B, np.transpose(iN, [0, 2, 1]))),  # TODO: adjoint=True
            [0, 2, 1]
            )
        # print("t")
        # print(t.shape)

        lb = np.exp(-np.sum(iN * t, 2)/2) * beta

        # print("lb")
        # print(lb.shape)

        tiL = t @ iL
        # print("tiL")
        # print(tiL.shape)
        #
        # print("variance")
        # print(self.variance.shape)

        c = self.variance.flatten() / np.sqrt(np.linalg.det(B))
        # print("c")
        # print(c.shape)

        M = (np.sum(lb, 1) * c)[:, None]
        # print("M")
        # print(M.shape)
        # print("lb modified")
        # print(lb[:, :, None].shape)
        # print(np.matmul(np.transpose(np.conjugate(tiL), [0, 2, 1]), lb[:, :, None]).shape)
        V = np.matmul(np.transpose(np.conjugate(tiL), [0, 2, 1]), lb[:, :, None])[..., 0] * c[:, None]
        # print("V")
        # print(V.shape)
        # print("lengthscales recip")
        # print((self.lengthscale).shape)
        # print((self.lengthscale[None, :, :, :]).shape)
        # print((self.lengthscale[:, None, :, :]).shape)
        # print((self.lengthscale2).shape)
        # print((self.lengthscale2[None, :]).shape)
        # print((self.lengthscale2[:, None]).shape)

        iL1 = 1 / np.square(self.lengthscale[None, :, :, :])
        iL1[iL1 == np.inf] = 0
        iL2 = 1 / np.square(self.lengthscale[:, None, :, :])
        iL2[iL2 == np.inf] = 0
        R = s @ (
            iL1 +
            iL2
            ) + np.eye(self.input_dim)

        # print("R")
        # print(R.shape)
        # # print(R)
        #
        # print(self.lengthscale2[:, None, None, :].shape)
        # print(inp[None, :, :, :].shape)
        X = inp[None, :, :, :] / np.square(self.lengthscale2[:, None, None, :])
        # print("X")
        # print(X.shape)

        X2 = -inp[:, None, :, :] / np.square(self.lengthscale2[None, :, None, :])
        # print("X2")
        # print(X2.shape)

        Q = np.linalg.solve(R, s) / 2
        # print("Q")
        # print(Q.shape)

        Xs = np.sum(X @ Q * X, 3)  # TODO: what axis to sum over?
        # print("Xs")
        # print(Xs.shape)
        # print((X @ Q * X).shape)

        X2s = np.sum(X2 @ Q * X2, 3)  # TODO: what axis to sum over?
        # print("X2s")
        # print(X2s.shape)
        # print((X2 @ Q * X2).shape)
        #
        # print((X @ Q).shape)
        # print(X2.shape)

        maha = -2 * np.matmul(X @ Q, np.transpose(np.conjugate(X2), [0, 1, 3, 2])) + Xs[:, :, :, None] + X2s[:, :, None, :]  # TODO" transposed correct dimension?
        # print("maha")
        # print(maha.shape)
        #
        # print("variance")
        # print(self.variance.shape)

        k = np.log(self.variance) - np.sum(np.square(iN)) / 2
        # print("k")
        # print(k.shape)

        L = np.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        # print("L")
        # print(L.shape)

        S = (np.tile(beta[:, None, None, :], [1, self.state_dim, 1, 1])
             @ L @
             np.tile(beta[None, :, :, None], [self.state_dim, 1, 1, 1])
             )[:, :, 0, 0]
        # print("S")
        # print(S.shape)

        diagL = np.transpose(np.diagonal(L, 0, 0, 1))
        # print("diagL")
        # print(diagL.shape)
        #
        # print((np.multiply(iK, diagL)).shape)
        # print(np.sum(np.multiply(iK, diagL), 1).shape)
        sum = np.sum(np.multiply(iK, diagL), 1)
        sum2 = np.sum(sum, 1)
        # print(sum2.shape)

        S = S - np.diag(sum2)
        # print("S")
        # print(S.shape)

        S = S / np.sqrt(np.linalg.det(R))
        # print(S.shape)
        # print("variance")
        # print(self.variance.shape)
        S = S + np.diag(self.variance)
        # print(S.shape)
        S = S - M @ np.transpose(M)
        # print(S.shape)
        #
        # print("M.T")
        # print(np.transpose(M))
        # print("S")
        # print(S)
        # print("V")
        # print(V)

        return np.transpose(M), S, np.transpose(V)

    # def is_pos_def(self, x):
    #     return np.all(np.linalg.eigvals(x) > 0)

    def K(self, X1, X2=None):
        """
        :param X1:
        :param X2:
        :return: state_dim-by-N-by-N
        """
        return np.stack(
            [model.kern.K(X1, X2) for model in self.models]
        )

    @property
    def lengthscale(self):
        return np.stack(
            [np.diag(model.kern.lengthscale) for model in self.models]
        )

    @property
    def lengthscale2(self):
        return np.stack(
            [model.kern.lengthscale for model in self.models]
        )

    @property
    def X(self):
        """
        :return: N-by-state_dim
        """
        return self.models[0].X

    @property
    def Y(self):
        return np.stack(
            [model.Y for model in self.models]
        )

    @property
    def variance(self):
        return np.stack(
            [model.kern.variance for model in self.models]
        )

    @property
    def noise(self):
        return np.stack(
            [model.likelihood.variance for model in self.models]
        )

    def centralised_inputs(self, m):
        return self.X - m

    # def predict(self, m, s):
    #     M_x = []
    #     S_x = []
    #     for i, model in enumerate(self.models):
    #         m_x, s_x = self.univariate_predict(m, s, model)
    #         M_x.append(m_x)
    #         S_x.append(s_x)
    #     return M_x, S_x

    def calc_qi(self, m, x_i, alpha, inv, det):
        inp = x_i - m
        exp = -0.5 * np.dot(np.dot(inp, inv), inp.T)
        q = alpha**2 * det ** (-0.5) * np.exp(exp)
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

        # V = []
        # for i, model in enumerate(self.models):
        #     variance = model.kern.variance
        #     V.append(variance)
        # print("Variance")
        # print(V)
        # s_star = s_star - np.diag(np.array(V) - 1e-6)

        return m_star, s_star
