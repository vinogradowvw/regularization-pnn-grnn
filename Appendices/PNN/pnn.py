import numpy as np
from base.Estimator import Estimator
from base.Layers import PatternLayer
from base.Layers import SummationLayerPNN
from base.Layers import OutputLayerPNN


class PNN(Estimator):
    """Probabilistic Neural Network (PNN) for classification tasks.


    Attributes:
        __n_classes (int): Number of classes in the classification problem.
        __pattern_layer (PatternLayer): Layer for computing kernel values.
        __summation_layer (SummationLayerPNN): Layer for summing kernel values per class.
        __outpu_layer (OutputLayerPNN).
        __classes (np.ndarray): Unique class labels from the training data.
        __class_counts (np.ndarray): Number of training patterns per class.
    """

    def __init__(self,
                 kernel,
                 sigma,
                 n_classes,
                 losses
                 ):
        """Initialize the PNN model.

        Args:
            kernel (str): Type of kernel to use (e.g., 'gaussian').
            sigma (float): Kernel smoothing (bandwith) parameter.
            n_classes (int): Number of classes in the classification problem.
            losses (list): Loss weights for each class.
        """
        super().__init__(kernel, sigma)
        if (n_classes != len(losses)):
            raise ValueError("""Number of class must match
                                the length of loasses list""")
        self.__n_classes = n_classes
        self.__pattern_layer = PatternLayer(self._kernel, model='pnn')
        self.__summation_layer = SummationLayerPNN(list(range(self.__n_classes)))
        self.__output_layer = OutputLayerPNN(losses)

    def fit(self, X, y):
        """Train the PNN model on the provided data.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,), where each label is an
                integer from 0 to n_classes-1.
        """
        unique = np.unique(y, return_counts=True)   # storing classes
        self.__classes = unique[0]

        if (len(self.__classes) != self.__n_classes):
            raise ValueError(
                f"""Number of classes {self.__n_classes}
                doesn't match the number of class
                in target variable {len(self.__classes)}.
                """)
        elif not all(self.__classes == list(range(self.__n_classes))):
            raise ValueError(
                """Classes should be encoded as
                integers from 0 to number of classes - 1
                e.g.: [0, 1, 3].
                """)

        self.__pattern_layer.fit(X, y)
        self.__output_layer.fit(X, y)

    def predict(self, X):
        """Predict the class label for the input data.

        Args:
            X (np.ndarray): Input data of shape (1, n_features).

        Returns:
             np.ndarray (n_samples): Predicted labels.
        """

        k, y, d = self.__pattern_layer.forward(X)
        likelihood = self.__summation_layer.forward(k, y, None)
        decision = self.__output_layer.forward(likelihood)

        return decision
