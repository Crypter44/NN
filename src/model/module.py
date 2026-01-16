"""
Defines the base Module class and multiple neural network layers.
"""

import numpy as np

from src.model.initialization import XavierInitialization
from src.model.parameter import Parameter


class Module:
    """
    Abstract base class for all neural network modules.
    """
    TRAIN = 'train'
    EVAL = 'eval'

    def __init__(self):
        self._mode = Module.TRAIN  # default mode is training

    def forward(self, x):
        """
        Performs the forward pass of the module.
        :param x: Input data
        :return: Output data
        """
        raise NotImplementedError

    def backward(self, dout):
        """
        Performs the backward pass of the module.
        :param dout: Upstream gradient
        :return: Gradient with respect to input
        """
        raise NotImplementedError

    def parameters(self):
        """
        Returns the parameters of the module.
        :return: List of parameters
        """
        raise NotImplementedError

    def __call__(self, x):
        """
        Override for calling the module instance directly.
        :param x: Input data
        :return:  Output data
        """
        return self.forward(x)

    def train(self):
        """
        Sets the module to training mode.
        """
        self._mode = Module.TRAIN

    def eval(self):
        """
        Sets the module to evaluation mode.
        """
        self._mode = Module.EVAL


class LinearLayer(Module):
    def __init__(self, input_dim, output_dim, include_bias=True, init_method=XavierInitialization()):
        """
        Initializes the fully connected layer.
        :param input_dim: Input dimension D_in
        :param output_dim: Output dimension D_out
        :param include_bias: Whether to include bias term
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cache = None

        dim_w_in = self.input_dim
        dim_w_out = self.output_dim
        if include_bias:
            dim_w_in += 1  # for bias term
        self.W = Parameter(np.zeros((dim_w_in, dim_w_out)))  # weights and bias
        self.include_bias = include_bias

        self.init_method = init_method
        self.init_method(self.W)

    def forward(self, x):
        """
        Performs the forward pass of the fully connected layer.
        :param x: Input data of shape (N, D_in)
        :return:
        """
        if self.include_bias:
            # Add bias term to input
            N = x.shape[0]
            x = np.concatenate((x, np.ones((N, 1))), axis=1)  # shape (N, D_in + 1)
        z = x @ self.W
        self.cache = {'input': x, 'z': z}
        return z

    def backward(self, dout):
        """
        Performs the backward pass of the fully connected layer.

        Stores the gradient with respect to weights in self.W.gradient of shape (D_in[+1], D_out).
        :param dout: Upstream gradient of shape (N, D_out)
        :return: Gradient with respect to input of shape (N, D_in)
        """
        if self.cache is None:
            raise ValueError("No cache found. Perform forward pass before backward.")

        # Retrieve cached values
        x = self.cache['input']  # shape (N, D_in)
        N = x.shape[0]

        # gradient for weights
        self.W.update_grad((x.T @ dout) / N)  # (D_in+1, D_out)

        # gradient wrt input (for chaining)
        dX = dout @ self.W.T  # (N, D_in+1)
        if self.include_bias:
            dX = dX[:, :-1]  # drop bias part

        return dX

    def parameters(self):
        """
        Returns the parameters of the fully connected layer.
        :return: List of parameters
        """
        return [self.W]

class DropOut(Module):
    def __init__(self, drop_prob=0.5):
        """
        Initializes the Dropout layer.
        :param drop_prob: Probability of dropping a unit
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.mask = None

    def forward(self, x):
        """
        Performs the forward pass of the Dropout layer.
        :param x: Input data
        :return: Output data after applying dropout
        """
        if self._mode == Module.TRAIN:
            self.mask = (np.random.rand(*x.shape) >= self.drop_prob) / (1.0 - self.drop_prob)
            return x * self.mask
        else:
            return x

    def backward(self, dout):
        """
        Performs the backward pass of the Dropout layer.
        :param dout: Upstream gradient
        :return: Gradient with respect to input
        """
        if self._mode == Module.TRAIN:
            return dout * self.mask
        else:
            return dout

    def parameters(self):
        """
        Returns the parameters of the Dropout layer (none).
        :return: Empty list
        """
        return []
