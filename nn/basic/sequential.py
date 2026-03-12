from ..base import TrainableLayer, Layer
import numpy as np


class Sequential(TrainableLayer):
    def __init__(self, *layers: Layer):
        super().__init__()
        self.layers = layers

    def forward(self, input_data: np.ndarray, is_training: bool = False) -> np.ndarray:
        for layer in self.layers:
            input_data = layer.forward(input_data, is_training)
        return input_data
        
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)
        return output_gradient
        
    def get_trainable_params(self) -> int:
        return sum(layer.get_trainable_params() for layer in self.layers)
    
    def clean(self):
        for layer in self.layers:
            layer.clean()
    
    def freeze(self):
        self._is_trainable = False
        for layer in self.layers:
            layer.freeze()
    
    def unfreeze(self):
        self._is_trainable = True
        for layer in self.layers:
            layer.unfreeze()
