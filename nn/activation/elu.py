from ..base import Activation
import numpy as np


class Elu(Activation):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def fn(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def fn_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha * np.exp(x))
