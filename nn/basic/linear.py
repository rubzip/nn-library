from ..base import TrainableLayer
import numpy as np


class Linear(TrainableLayer):
    def __init__(self, input_size: int, output_size: int, add_bias: bool = True):
        super().__init__()

        self._is_trainable = True
        self._expected_input_dim = 2
        self._expected_input_shape = (None, input_size)

        self.input_size = input_size
        self.output_size = output_size
        self.add_bias = add_bias

        self.weights: np.ndarray = np.random.randn(input_size, output_size) * np.sqrt(
            2.0 / input_size
        )
        if add_bias:
            self.biases: np.ndarray = np.zeros((1, output_size))

    def forward(self, input_data: np.ndarray, is_training: bool = False) -> np.ndarray:
        self.__validate_input(input_data)
        if is_training:
            self.input = input_data

        output = input_data @ self.weights
        if self.add_bias:
            output += self.biases

        return output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        input_gradient = output_gradient @ self.weights.T
        weights_gradient = self.input.T @ output_gradient

        self.weights -= learning_rate * weights_gradient
        if self.add_bias:
            biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
            self.biases -= learning_rate * biases_gradient

        return input_gradient

    def get_trainable_params(self) -> int:
        return self.weights.size + (self.biases.size if self.add_bias else 0)
