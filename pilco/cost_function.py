import numpy as np
from abc import ABC, abstractmethod

class Cost(ABC):

    @abstractmethod
    def compute_cost(self, m_x, s_x):
        pass


class SaturatingCost(Cost):

    def __init__(self, x_target, iT, idxs=None):
        self.x_target = x_target
        self.iT = iT
        self.idxs = idxs

    def compute_cost(self, m_x, s_x):
        # det = np.linalg.det(np.eye(s_x.shape[0]) + np.dot(s_x, self.iT))
        # diff = j - self.x_target
        # return 1 - np.exp(-0.5 * np.dot(np.dot(diff, iT), diff.T))
        return self.expected_immediate_cost(m_x, s_x)

    def first_moment(self, m_x, s_x):
        diff = m_x[:, self.idxs] - self.x_target
        det = np.eye(s_x.shape[0]) + np.dot(s_x, self.iT)
        S1 = np.dot(self.iT, np.linalg.inv(det))
        E = np.linalg.det(det) ** (-0.5) * np.exp(-0.5 * np.dot(np.dot(diff, S1), diff.T))
        return E, S1

    def second_moment(self, m_x, s_x):
        diff = m_x[:, self.idxs] - self.x_target
        det = np.eye(s_x.shape[0]) + 2 * np.dot(s_x, self.iT)
        S2 = np.dot(self.iT, np.linalg.inv(det))
        E2 = np.linalg.det(det) ** (-0.5) * np.exp(np.dot(np.dot(diff, S2), diff.T))
        return E2, S2

    def expected_immediate_cost(self, m_x, s_x):
        s_x = s_x[:, self.idxs]
        s_x = s_x[self.idxs, :]
        E, _ = self.first_moment(m_x, s_x)
        E2, _ = self.second_moment(m_x, s_x)
        s_c = E2 - E**2
        return E, s_c

    def dEdM(self, m_x, s_x):
        s_x = s_x[:, self.idxs]
        s_x = s_x[self.idxs, :]
        diff = m_x[:, self.idxs] - self.x_target
        E, S1 = self.first_moment(m_x, s_x)
        return -E * np.dot(diff.T, S1)

    def dEdS(self, m_x, s_x):
        s_x = s_x[:, self.idxs]
        s_x = s_x[self.idxs, :]
        diff = m_x[:, self.idxs] - self.x_target
        E, S1 = self.first_moment(m_x, s_x)
        return 0.5 * E * np.dot( np.dot(np.dot(S1, diff), diff.T) - np.eye(s_x.shape[0]), S1)