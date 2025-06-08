import numpy as np
from scipy.stats import laplace


class LaplacePrior():

    def __init__(self, tau: float, type: str = 'density'):
        self.__type = type
        self.__tau = tau 

    def __call__(self, x):
        """
        Computes the Laplace Prior value(s) at x.
        """
        if self.__type == 'density':
            return np.exp(-x / self.__tau) / self.__tau
        elif self.__type == 'cumulative':
            return laplace.cdf(x, loc=0, scale=self.__tau) - laplace.cdf(0, loc=0, scale=self.__tau)
