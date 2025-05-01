import numpy as np


class BernoulliDropout():

    def __init__(self):
        pass

    def __call__(self, p):
        z = np.random.binomial(n=1, p=p)
        return z
