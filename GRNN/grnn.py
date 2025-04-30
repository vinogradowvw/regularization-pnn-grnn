from base.Estimator import Estimator
from base.Layers import PatternLayer
from base.Layers import SummationLayerGRNN


class GRNN(Estimator):

    def __init__(self, kernel, sigma):
        super().__init__(kernel, sigma)

        self.__pattern_layer = PatternLayer(self._kernel)
        self.__summation_layer = SummationLayerGRNN()

    def fit(self, X, y):
        self.__pattern_layer.fit(X, y)

    def predict(self, X):
        pattern_layer_output = self.__pattern_layer.forward(X)
        summation_layer_output = self.__summation_layer.forward(*pattern_layer_output)

        return summation_layer_output
