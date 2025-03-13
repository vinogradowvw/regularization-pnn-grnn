import numpy as np


class SummationLayerPNN:

    def __init__(self, classes):
        self.__classes = classes

    def forward(self, inputs, y):
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
            output[c] = np.sum(inputs[class_mask])
        return output
