from ..base import Activation
import numpy as np


class LeakyRelu(Activation):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def fn(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def fn_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)
