from ..base import Activation
import numpy as np


class Tanh(Activation):
    def fn(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def fn_derivative(self, x: np.ndarray) -> np.ndarray:
        t = self.fn(x)
        return 1 - t**2
