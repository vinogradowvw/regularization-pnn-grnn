import numpy as np

from base.Kernel import Kernel


class PatternLayer():

    def __init__(self, kernel: Kernel):
        self.__kernel = kernel
        self.__pattern_units = {'W': None, 'y': None}

    def __normalize_to_unit_len(self, x) -> np.ndarray:
        """Nomalize the feature vectors X to unit length

        Args:
            x (np.ndarray)

        Returns:
            np.ndarray: Normalized feature vectors with the same shape
        """
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / norms

    def fit(self, X, y):
        """Fitting the train data to the pattern layer."""

        X_normalized = self.__normalize_to_unit_len(X)
        self.__pattern_units = {'W': X_normalized, 'y': y}

    def forward(self, input):
        """Forwarding the pattern layer.

        Args:
            input (np.ndarray) (1, n_features)

        Returns:
            np.ndarray (2, number of patterns): Pairs of kernel funtion values
                and target values.
        """
        X_normalized = self.__normalize_to_unit_len(input)
        W = self.__pattern_units['W']
        kernel_values = self.__kernel(W, X_normalized)
        return [kernel_values, self.__pattern_units['y']]
