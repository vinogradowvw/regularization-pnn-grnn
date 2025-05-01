import numpy as np


class SummationLayerPNN:

    def __init__(self, classes):
        self.__classes = classes

    def forward(self, inputs, y, weights):
        """Forwarding the summation layer.

        Args:
            input (np.ndarray) (number of patterns): Computed kernel function values
            y (np.array) (number of patterns): Pattern target values

        Returns:
            np.ndarray (n_classes): Sums of kernel function values for each class.
        """
        output = np.zeros(len(self.__classes))
        for c in self.__classes:
            class_mask = (y == c)
            sum_k = np.sum(inputs[class_mask])
            if weights is not None:
                normalization = np.sum(weights)
            else:
                normalization = len(inputs[class_mask])
            output[c] = sum_k / normalization
        return output
