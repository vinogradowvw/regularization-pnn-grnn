import numpy as np
from Distance.distance import distance_l1
from base.Kernel import Kernel


class LaplaceKernel(Kernel):

    def __init__(self, sigma: float):
        super().__init__(sigma)
        self.distance = 'l2'

    def __call__(self, w, x):
        """
        Computes the Laplace kernel value(s) between w and x.
        """
        dists = distance_l1(w, x)
        return np.exp(-dists / self._sigma), dists
