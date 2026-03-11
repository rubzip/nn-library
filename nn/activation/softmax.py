from ..base import Activation
import numpy as np


class Softmax(Activation):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def fn(self, x: np.ndarray) -> np.ndarray:
        x = x / self.temperature
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def fn_derivative(self, x: np.ndarray) -> np.ndarray:
        pass
