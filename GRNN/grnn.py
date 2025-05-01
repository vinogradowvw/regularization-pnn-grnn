from base.Estimator import Estimator
from base.Layers import PatternLayer
from base.Layers import SummationLayerGRNN
from base.Layers.RegularizationLayer import RegularizationLayer


class GRNN(Estimator):

    def __init__(self,
                 kernel,
                 sigma,
                 regularization=None,
                 tau=0.5):

        super().__init__(kernel, sigma)

        self.__pattern_layer = PatternLayer(self._kernel)

        self.__regularization_type = regularization

        if regularization:
            self.__regularization_layer = RegularizationLayer(regularization,
                                                              tau)
        self.__summation_layer = SummationLayerGRNN()

    def fit(self, X, y):
        self.__pattern_layer.fit(X, y)

    def predict(self, X):
        k, y, d = self.__pattern_layer.forward(X)
        if self.__regularization_type:
            k, y, weights = self.__regularization_layer.forward(k, y, d)
        nominator, denominator = self.__summation_layer.forward(k, y)
        estimation = nominator / denominator
        return estimation
