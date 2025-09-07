import numpy as np


def distance_l1(x, w):
    return np.sum(np.abs(w - x), axis=1)


def distance_l2(x, w):
    return np.sqrt(np.sum((w - x)**2, axis=1))
