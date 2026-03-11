from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None
      
        self._is_trainable = False
        self._expected_dim = None
        self._expected_shape = None

    @abstractmethod
    def forward(self, input_data: np.ndarray, is_training: bool = False) -> np.ndarray: ...

    @abstractmethod
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray: ...

    @abstractmethod
    def get_trainable_params(self) -> int: ...
  
    def __call__(self, input_data: np.ndarray, is_training: bool = False) -> np.ndarray:
        return self.forward(input_data, is_training)

    def clean(self):
        """Cleans memory"""
        self.input = None
        self.output = None
    
    def __validate_input(self):
        pass
    
    def __validate_dim(self):
        pass
