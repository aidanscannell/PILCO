import gpflow
import tensorflow as tf
import numpy as np
from gpflow import settings

float_type = settings.dtypes.float_type


class MGPR(gpflow.Parameterized):
    '''
    Multivariate Gaussian Process Regression
    '''

    def __init__(self, X, Y, name=None):
        super(MGPR, self).__init__(name)
        self.num_outputs = Y.shape[1]
        self.num_dims = X.shape[1]
        self.num_data_points = X.shape[0]

        self.models = []
        self.create_models(X, Y)

    def create_models(self, X, Y):
        for i in range(self.num_outputs):
            kern = gpflow.kernels.RBF(input_dim=X.shape[1], ARD=True)
            self.models.append(gpflow.models.GPR(X, Y[:, i:i+1], kern))
            # self.models[i].clear()
            self.models[i].compile()

    def set_XY(self, X, Y):
        for i in range(len(self.models)):
            self.models[i].X = X
            self.models[i].Y = Y[:, i:i + 1]

    def optimize(self):
        optimizer = gpflow.train.ScipyOptimizer(options={'maxfun': 500})
        for model in self.models:
            optimizer.minimize(model)
            # print("Constrained tensor")
            # print(model.kern.variance.constrained_tensor)

    def compute_factorizations(self):
        for model in self.models:
            eye = tf.eye(model.X.shape[0], dtype=float_type)  # identity matrix; n-by-n
            K = model.kern.K(model.X.parameter_tensor)  # kernel matrix K; n-by-n
            L = tf.cholesky(K + model.likelihood.variance.parameter_tensor * eye)
            iK = tf.cholesky_solve(L, eye)  # inverse kernel matrix inv(K); n-by-n
            beta = tf.cholesky_solve(L, model.Y.parameter_tensor)  # (K + sigma^2 I)^(-1) y; n-by-1
            model.iK = iK
            model.beta = beta

    def kern(self, X1, X2):
        print([model.kern.K(X1, X2) for model in self.models])
        return [model.kern.K(X1, X2) for model in self.models]

    def univariate_predict(self, m_x, s_x):

        return

    def predict(self, m_x, s_x):
        """
        Compute joint GP predictions (multivariate predictions) for an uncertain test input xstar N(m,S)
        :param m_x:
        :param s_x:
        :return:
        """
        print(self.kern(m_x, m_x))
        # K = self.K(self.X)
        print("here")
        # print(self.noise)
        # print(self.noise[:, None, None])

        self.compute_factorizations()

        # 1) compute predicted mean and covariance between input and prediction
        for model in self.models:
            iLambda = tf.matrix_diag(1/model.kern.lengthscales.constrained_tensor)  # inverse squared lengthscales; D-by-D
            inp = model.X.parameter_tensor - m_x  # subtract mean of test input from training input; n-by-D
            # print(inp)
            # print(iLambda)
            iN = inp @ iLambda  # n-by-D
            # print(iN)
            R = s_x + tf.matrix_diag(model.kern.lengthscales.constrained_tensor)
            iR = tf.cholesky
            tf.eye(tf.shape(s_x)[0], dtype=float_type) + s_x @ iLambda
            q = s_x @ iLambda + tf.eye(tf.shape(s_x)[0], dtype=float_type)
            # tf.matrix_determinant
            # print(inp)
            # print(s_x + iLambda)
            # print(q)
            # print(tf.exp(-1 / 2 * inp @ (s_x + iLambda) @ tf.transpose(inp)))
            # print(tf.exp(-1/2 * inp @ (s_x + iLambda) @ inp))
            # print(B)

        # for model in self.models:
        #     Zeta_i = inp
        #
        #     for j in range():
        #         Zeta_j =

        # batched_eye = tf.eye(tf.shape(self.X)[0], batch_shape=[self.num_outputs], dtype=float_type)
        # L = tf.cholesky(K + self.noise[:, None, None] * batched_eye)
        # iK = tf.cholesky_solve(L, batched_eye)
        # Y_ = tf.transpose(self.Y)[:, :, None]
        # # Why do we transpose Y? Maybe we need to change the definition of self.Y() or beta?
        # beta = tf.cholesky_solve(L, Y_)[:, :, 0]  # K/y; n-by-1

        return

    # def K(self, model, X1, X2=None):
    #     return model.kern.K(X1, X2)

    # def K(self, X1, X2=None):
    #     K = []
    #     for model in self.models:
    #         K.append(model.kern.K(X1, X2))
    #     return tf.stack(K)

    @property
    def X(self):
        return self.models[0].X.parameter_tensor

    @property
    def noise(self):
        noise = []
        for model in self.models:
            # noise.append(model.likelihood.variance.constrained_tensor)
            noise.append(model.likelihood.variance.value)

        return noise
        # return tf.stack(noise)

