import numpy as np


class SummationLayerGRNN:

    def __init__(self):
        pass

    def forward(self, inputs, y):
        marginal = np.sum(inputs)
        weighted_y = np.sum(inputs*y)
        return weighted_y, marginal
