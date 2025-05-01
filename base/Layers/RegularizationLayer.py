import numpy as np
from Kernels.GaussianKernel import GaussianKernel
from Kernels.LaplaceKernel import LaplaceKernel
from Kernels.BernoulliDropout import BernoulliDropout


class RegularizationLayer():

    def __init__(self,
                 regularization_type: str,
                 tau: float):

        self.__regularization_type = regularization_type

        if regularization_type == 'l1':
            self.__prior_kernel = LaplaceKernel(tau)
        if regularization_type == 'l2':
            self.__prior_kernel = GaussianKernel(tau)
        if 'dropout' in regularization_type:
            self.__dropout = BernoulliDropout()
            if regularization_type[1] == 'l1':
                self.__prior_kernel = LaplaceKernel(tau)
            if regularization_type[1] == 'l2':
                self.__prior_kernel = GaussianKernel(tau)
        self.distances = {}

    def __distance_prior(self, distance):
        k_value, _ = self.__prior_kernel(np.array([[distance]]),
                                         np.array([[0]]))
        return k_value[0]

    def forward(self, pattern_kernels, y, distances):
        priors = np.array([self.__distance_prior(d) for d in distances])
        if 'dropout' in self.__regularization_type:
            dropout = [self.__dropout(p) for p in priors]
            adjusted_kernels = pattern_kernels * dropout
            return adjusted_kernels, y, dropout
        else:
            adjusted_kernels = pattern_kernels * priors
            return adjusted_kernels, y, priors
