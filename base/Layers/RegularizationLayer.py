import numpy as np
from Kernels.GaussianKernel import GaussianKernel
from Kernels.LaplaceKernel import LaplaceKernel
from Kernels.BernoulliDropout import BernoulliDropout
from base.Priors import GaussianPrior
from base.Priors import LaplacePrior


class RegularizationLayer():

    def __init__(self,
                 regularization_type: str,
                 tau: float):

        self.__regularization_type = regularization_type

        if regularization_type == 'l1':
            self.__prior = LaplaceKernel(tau)
        if regularization_type == 'l2':
            self.__prior = GaussianKernel(tau)
        if 'dropout' in regularization_type:
            self.__dropout = BernoulliDropout()
            if regularization_type[1] == 'l1':
                self.__prior = LaplacePrior(tau)
            if regularization_type[1] == 'l2':
                self.__prior = GaussianPrior(tau)
        self.distances = {}

    def __prior_decay(self, distances):
        priors_values, _ = self.__prior(distances.reshape(-1, 1),
                                          np.array([[0]]))
        return priors_values

    def forward(self, pattern_kernels, y, distances):
        
        if 'dropout' in self.__regularization_type:
            priors = self.__prior(distances)
            i = 0
            while i <= 1000:
                dropout = np.array([self.__dropout(p) for p in priors])
                i += 1
                if np.any(dropout):
                    break
                    
            if not np.any(dropout):
                raise RuntimeError("Dropout sampling failed: all nodes dropped after 1000 attempts")
                
            adjusted_kernels = pattern_kernels * dropout
            return adjusted_kernels, y, dropout
        else:
            priors = self.__prior_decay(distances)
            adjusted_kernels = pattern_kernels * priors
            return adjusted_kernels, y, priors