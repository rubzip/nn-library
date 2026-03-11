from ..base import Activation

class Sigmoid(Activation):
    def fn(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def fn_derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.fn(x)
        return s * (1 - s)
