import numpy as np


class GaussianPrior():

    def __init__(self, tau: float):
        self.__tau = tau 

    def __call__(self, x):
        """
        Computes the Gaussian Prior value(s) at x.
        """
        return 2 / np.sqrt( 2 * np.pi  self.__tau**2) * np.exp( - (x**2 / (2 * self.__tau**2)))