import numpy as np
from scipy.stats import norm


class GaussianPrior():

    def __init__(self, tau: float, type: str = 'density'):
        self.__type = type
        self.__tau = tau 

    def __call__(self, x):
        """
        Computes the Gaussian Prior value(s) at x.
        """
        if self.__type == 'density':
            return 2 / np.sqrt( 2 * np.pi * self.__tau**2) * np.exp( - (x**2 / (2 * self.__tau**2)))
        elif self.__type == 'cumulative':
            return norm.cdf(x, loc=0, scale=self.__tau)
