import numpy as np
from base.Kernel import Kernel


class PatternLayer():

    def __init__(self, kernel: Kernel, model: str):
        self.__kernel = kernel
        self.__pattern_units = {'W': None, 'y': None}
        self.__model = model

    def fit(self, X, y):
        """Fitting the train data to the pattern layer."""

        self.__pattern_units = {'W': np.array(X), 'y': np.array(y)}

    def forward(self, input):
        """Forwarding the pattern layer.

        Args:
            input (np.ndarray) (1, n_features)

        Returns:
            np.ndarray (2, number of patterns): Pairs of kernel funtion values
                and target values.
        """
        W = self.__pattern_units['W']
        y = self.__pattern_units['y']

        if self.__model == 'pnn':
            kernel_values_list = []
            distances_list = []
            y_list = []

            for class_label in np.unique(y):
                W_class = W[y == class_label]
                y_class = y[y == class_label]
                k_values, distances = self.__kernel(W_class, input)
                distances_list.append(distances)
                kernel_values_list.append(k_values)
                y_list.append(y_class)

            kernel_values = np.concatenate(kernel_values_list)
            y_values = np.concatenate(y_list)
            d_values = np.concatenate(distances_list)

        else:
            kernel_values, d_values = self.__kernel(W, input)
            kernel_values *= 1e10
            y_values = y
            
        return [kernel_values, y_values, d_values]
