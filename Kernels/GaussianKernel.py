import numpy as np
from Distance.distance import distance_l2
from base.Kernel import Kernel


class GaussianKernel(Kernel):

    def __init__(self, sigma: float):
        super().__init__(sigma)
        self.distance = 'l2'

    def __call__(self, w, x):
        """
        Computes the Gaussian kernel value(s) between w and x.
        """
        dists = distance_l2(w, x)
        return np.exp(- (dists**2 / (2 * self._sigma**2))), dists
