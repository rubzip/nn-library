from ..base import Activation
import numpy as np


class Relu(Activation):
    def fn(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def fn_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
