import numpy as np


class LaplacePrior():

    def __init__(self, tau: float):
        self.__tau = tau 

    def __call__(self, x):
        """
        Computes the Laplace Prior value(s) at x.
        """
        return np.exp(-x / self.__tau) / self.__tau