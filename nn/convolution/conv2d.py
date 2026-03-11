from ..base import Layer
import numpy as np


class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self._is_trainable = True
        self._expected_input_dim = 4  # (batch_size, in_channels, height, width)
        self._expected_input_shape = (None, in_channels, None, None)
