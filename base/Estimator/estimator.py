import numpy as np
from abc import ABC, abstractmethod
from Kernels import GaussianKernel
from base.Kernel import Kernel
from typing import Dict


class Estimator(ABC):

    _kernel_conf: Dict[str, Kernel.__class__] = {
        'gaussian': GaussianKernel
    }

    def __init__(self, kernel: str, sigma: float):
        self._kernel = self._kernel_conf[kernel](sigma)

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("'fit' method is not implemented")

    def _compute_kernel_values(self, x):
        return self._kernel(self._pattern_units, x)

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("'prodict' method is not implemented")
