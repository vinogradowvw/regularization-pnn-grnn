import numpy as np
from base.Kernel import Kernel


class GaussianKernel(Kernel):

    def __init__(self, sigma: float):
        super().__init__(sigma)

    def __call__(self, w, x):
        """
        Computes the Gaussian kernel value(s) between w and x.
        Assumes w and x are normalized to unit length.
        """
        return np.exp((np.dot(x, w.T)-1)/np.float_power(self._sigma, 2))[0]
