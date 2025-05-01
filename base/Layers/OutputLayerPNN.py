import numpy as np

class OutputLayerPNN:

    def __init__(self, losses):
        self._losses = np.array(losses)

    def fit(self, X, y):
        unique = np.unique(y, return_counts=True)
        classes = unique[0]
        class_counts = unique[1]
        priors_class = class_counts / len(y)
        self._priors_class = priors_class

    def forward(self, likelihood):
        decision = np.argmax(
                likelihood
                * self._losses
                * self._priors_class
        )
        return decision