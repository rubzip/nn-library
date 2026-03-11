from ..base import Activation
import numpy as np


class Relu6(Activation):
    def fn(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(0, x), 6)

    def fn_derivative(self, x: np.ndarray) -> np.ndarray:
        return ((x > 0) & (x < 6)).astype(float)
