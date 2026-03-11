import unittest
import numpy as np
from nn.basic.linear import Linear
from nn.basic.sequential import Sequential
from nn.activation.relu import Relu
from nn.loss import MSE

class TestLayers(unittest.TestCase):
    def test_linear_forward(self):
        layer = Linear(3, 2)
        input_data = np.random.randn(5, 3)
        output = layer.forward(input_data)
        self.assertEqual(output.shape, (5, 2))

    def test_linear_backward(self):
        layer = Linear(3, 2)
        input_data = np.random.randn(5, 3)
        output = layer.forward(input_data, is_training=True)
        grad = np.random.randn(5, 2)
        input_grad = layer.backward(grad, 0.01)
        self.assertEqual(input_grad.shape, (5, 3))

    def test_sequential_forward(self):
        model = Sequential(
            Linear(3, 4),
            Relu(),
            Linear(4, 2)
        )
        input_data = np.random.randn(5, 3)
        output = model.forward(input_data)
        self.assertEqual(output.shape, (5, 2))

class TestLoss(unittest.TestCase):
    def test_mse_forward(self):
        loss = MSE()
        y_true = np.array([[1, 0]])
        y_pred = np.array([[0.5, 0.5]])
        val = loss.forward(y_true, y_pred)
        self.assertAlmostEqual(val, 0.25)

if __name__ == "__main__":
    unittest.main()
