from abc import abstractmethod
from ..base import NonTrainableLayer
import numpy as np


class BaseScaler(NonTrainableLayer):
    @abstractmethod
    def fit(self, X): ...

class StandardScaler(BaseScaler):
    def __init__(self):
        super().__init__()
        self.mu, self.std = 0., 1.

    def fit(self, X):
        self.mu = X.mean()
        self.std = X.std()
    
    def forward(self, X):
        return (X - self.mu) / self.std
    
    def backward(self, X):
        return X * self.std + self.mu

class MinMaxScaler(BaseScaler):
    def __init__(self):
        super().__init__()
        self.min, self.max = 0., 0.

    def fit(self, X):
        self.min = X.min()
        self.max = X.max()
    
    def forward(self, X):
        return (X - self.min) / (self.max - self.min)
    
    def backward(self, X):
        return X * (self.max - self.min) + self.min
