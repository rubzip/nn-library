from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    def __init__(self):
        self._is_trainable = False
        self._expected_input_dim = None
        self._expected_input_shape = None

        self.input = None
        self.output = None

    @abstractmethod
    def forward(
        self, input_data: np.ndarray, is_training: bool = False
    ) -> np.ndarray: ...

    @abstractmethod
    def backward(
        self, output_gradient: np.ndarray, learning_rate: float
    ) -> np.ndarray: ...

    @abstractmethod
    def get_trainable_params(self) -> int: ...

    @abstractmethod
    def freeze(self): ...

    @abstractmethod
    def unfreeze(self): ...

    def clean(self):
        """Cleans memory"""
        self.input = None
        self.output = None

    def __call__(self, input_data: np.ndarray, is_training: bool = False) -> np.ndarray:
        return self.forward(input_data, is_training)

    def __validate_shape(self, input_data: np.ndarray):
        if self._expected_input_shape is None:
            return

        shape = input_data.shape
        for expected, actual in zip(self._expected_input_shape, shape):
            if expected is not None and expected != actual:
                raise ValueError(
                    f"Expected input shape {self._expected_input_shape}, got {shape}"
                )

    def __validate_dim(self, input_data: np.ndarray):
        if self._expected_input_dim is None:
            return

        dim = input_data.ndim
        if dim != self._expected_input_dim:
            raise ValueError(
                f"Expected input dimension {self._expected_input_dim}, got {dim}"
            )

    def __validate_numpy(self, input_data: np.ndarray):
        if not isinstance(input_data, np.ndarray):
            raise ValueError(
                f"Expected input to be a numpy array, got {type(input_data)}"
            )

    def __validate_input(self, input_data: np.ndarray):
        self.__validate_numpy(input_data)
        self.__validate_dim(input_data)
        self.__validate_shape(input_data)


class TrainableLayer(Layer):
    def __init__(self):
        super().__init__()
        self._is_trainable = True
    
    def freeze(self):
        self._is_trainable = False
    
    def unfreeze(self):
        self._is_trainable = True


class NonTrainableLayer(Layer):
    def __init__(self):
        super().__init__()
        self._is_trainable = False
    
    def freeze(self):
        self._is_trainable = False
    
    def unfreeze(self):
        self._is_trainable = False
