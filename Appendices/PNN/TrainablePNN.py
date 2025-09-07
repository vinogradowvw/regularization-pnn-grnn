import numpy as np
from base.Estimator import Estimator
from base.Layers import SummationLayerPNN
from base.Layers import OutputLayerPNN
from base.Layers.TrainablePatternLayerPNN import TrainablePatternLayerPNN


class TrainablePNN(Estimator):
    """Trainable Probabilistic Neural Network for classification tasks.
    """

    def __init__(self,
                 sigma,
                 n_classes,
                 losses,
                 regularization,
                 tau=0.5):
        """Initialize the PNN model.

        Args:
            sigma (float): Kernel smoothing (bandwith) parameter.
            n_classes (int): Number of classes in the classification problem.
            losses (list): Loss weights for each class.
            regularization (str): 'l1' or 'l2' regularization
            tau (float): prior scale parameter
        """
        super().__init__(None, sigma)
        if (n_classes != len(losses)):
            raise ValueError("""Number of class must match
                                the length of loasses list""")
        self.__n_classes = n_classes

        self.pattern_layer = TrainablePatternLayerPNN(sigma, tau, regularization, n_classes)

        self.__summation_layer = SummationLayerPNN(list(range(self.__n_classes)))
        self.__output_layer = OutputLayerPNN(losses)

    def fit(self, X, y):
        """Train the PNN model on the provided data.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,), where each label is an
                integer from 0 to n_classes-1.
        """
        unique = np.unique(y, return_counts=True)
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

        self.pattern_layer.fit(X, y)
        self.__output_layer.fit(X, y)

    def predict(self, X):
        """Predict the class label for the input data.

        Args:
            X (np.ndarray): Input data of shape (1, n_features).

        Returns:
             np.ndarray (n_samples): Predicted labels.
        """

        weighed_N, y, beta = self.pattern_layer.forward(X)
        likelihood = self.__summation_layer.forward(weighed_N, y, beta)
        decision = self.__output_layer.forward(likelihood)

        return decision
