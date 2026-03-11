from .layer import Layer


class Activation(Layer):
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
        if is_training:
            self.input = input_data
        return self.fn(input_data)
    
    def backward(self, output_gradient, learning_rate) -> np.ndarray:
        return output_gradient * self.fn_derivative(self.input)

    def get_trainable_params(self) -> int:
        return 0
