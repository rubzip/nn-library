import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class MSE(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        delta = y_true - y_pred
        return np.mean(delta * delta)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        delta = y_true - y_pred
        return 2 * np.mean(delta)

class MAE(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        delta = y_true - y_pred
        return np.mean(np.abs(delta))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        delta = y_true - y_pred
        return np.mean(np.sign(delta))

class CrossEntropy(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Simplified backward for CE (often used with Softmax)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / y_true.shape[0]
