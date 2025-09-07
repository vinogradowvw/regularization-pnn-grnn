from abc import ABC, abstractmethod
from base.Kernels import GaussianKernel
from base.Kernels import Kernel
from typing import Dict, Optional


class Estimator(ABC):

    _kernel_conf: Dict[str, Kernel.__class__] = {
        'gaussian': GaussianKernel
    }

    def __init__(self, kernel: Optional[str], sigma: float):
        if kernel is not None:
            self._kernel = self._kernel_conf[kernel](sigma)

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("'fit' method is not implemented")

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("'prodict' method is not implemented")
