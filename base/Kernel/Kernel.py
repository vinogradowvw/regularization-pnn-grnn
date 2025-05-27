from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, sigma: float):
        self._sigma = sigma

    @abstractmethod
    def __call__(self, W, X):
        raise NotImplementedError("'__call__' method is not implemented")