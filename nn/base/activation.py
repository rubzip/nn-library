from abc import abstractmethod
import numpy as np
from .layer import NonTrainableLayer


class Activation(NonTrainableLayer):
    def __init__(self):
        super().__init__()
        self._is_trainable = False

    @abstractmethod
    def fn(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fn_derivative(self, x: np.ndarray) -> np.ndarray:
        pass

    def forward(self, input_data, is_training=False) -> np.ndarray:
        self._validate_input(input_data)
        if is_training:
            self.input = input_data
        return self.fn(input_data)

    def backward(self, output_gradient, learning_rate) -> np.ndarray:
        self._validate_numpy(output_gradient)
        return output_gradient * self.fn_derivative(self.input)
