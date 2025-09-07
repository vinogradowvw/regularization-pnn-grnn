from base.Estimator import Estimator
from base.Layers import SummationLayerGRNN
from base.Layers.TrainablePatternLayerGRNN import TrainablePatternLayerGRNN


class TrainableGRNN(Estimator):

    def __init__(self,
                 sigma,
                 regularization,
                 tau=0.5):

        super().__init__(None, sigma)

        self.pattern_layer = TrainablePatternLayerGRNN(sigma=sigma,
                                                       tau=tau,
                                                       regularization=regularization)

        self.__summation_layer = SummationLayerGRNN()

    def fit(self, X, y):
        self.pattern_layer.fit(X, y)

    def predict(self, X):
        k, y, w = self.pattern_layer.forward(X)
        nominator, denominator = self.__summation_layer.forward(k, y)
        estimation = nominator / denominator
        return estimation
