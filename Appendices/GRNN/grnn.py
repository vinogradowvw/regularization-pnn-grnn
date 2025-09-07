from base.Estimator import Estimator
from base.Layers import PatternLayer
from base.Layers import SummationLayerGRNN


class GRNN(Estimator):

    def __init__(self,
                 kernel,
                 sigma,
                 ):

        super().__init__(kernel, sigma)
        self.__pattern_layer = PatternLayer(self._kernel, model='grnn')
        self.__summation_layer = SummationLayerGRNN()

    def fit(self, X, y):
        self.__pattern_layer.fit(X, y)

    def predict(self, X):
        k, y, d = self.__pattern_layer.forward(X)
        nominator, denominator = self.__summation_layer.forward(k, y)
        estimation = nominator / denominator
        return estimation
