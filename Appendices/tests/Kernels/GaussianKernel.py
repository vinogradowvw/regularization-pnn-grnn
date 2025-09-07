import numpy as np
import pytest

from Kernels import GaussianKernel


@pytest.fixture
def kernel():
    return GaussianKernel(0.5)


class TestGaussianKernel:

    def test_single_vectors(self, kernel):
        w = np.array([1, 0])
        x = np.array([0, 1])
        expected = np.exp((0 - 1) / (0.5 ** 2))
        result = kernel(w, x)
        assert np.isclose(result, expected, rtol=1e-7), f"Expected {expected}, got {result}"

    def test_parallel_vectors(self, kernel):
        w = np.array([1, 0])
        x = np.array([1, 0])
        expected = np.exp((1 - 1) / (0.5 ** 2))
        result = kernel(w, x)
        assert np.isclose(result, expected, rtol=1e-7), f"Expected {expected}, got {result}"

    def test_opposite_vectors(self, kernel):
        w = np.array([1, 0])
        x = np.array([-1, 0])
        expected = np.exp((-1 - 1) / (0.5 ** 2))
        result = kernel(w, x)
        assert np.isclose(result, expected, rtol=1e-7), f"Expected {expected}, got {result}"

    def test_invalid_dimensions(self, kernel):
        w = np.array([1, 0, 0])
        x = np.array([1, 0])
        with pytest.raises(ValueError):
            kernel(w, x)

    def test_non_normalized_warning(self, kernel):
        w = np.array([2, 0])
        x = np.array([0, 2])
        expected = np.exp((0 - 1) / (0.5 ** 2))
        result = kernel(w, x)
        assert np.isclose(result, expected, rtol=1e-7), "Result computed, but vectors should be normalized"
