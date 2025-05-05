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

    def __distance_prior(self, distances):
        k_values, _ = self.__prior_kernel(distances.reshape(-1, 1),
                                          np.array([[0]]))
        return k_values

    def forward(self, pattern_kernels, y, distances):
        priors = self.__distance_prior(distances)
    
        if 'dropout' in self.__regularization_type:
            while True:
                dropout = np.array([self.__dropout(p) for p in priors])
                if np.any(dropout):
                    break
            adjusted_kernels = pattern_kernels * dropout
            return adjusted_kernels, y, dropout
        else:
            adjusted_kernels = pattern_kernels * priors
            return adjusted_kernels, y, priors