import numpy as np

from src.model.activation_function import Sigmoid, ReLU

class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout, is_last_layer=False):
        raise NotImplementedError


class FullyConnectedLayer(Layer):
    def __init__(self, input_dim, output_dim, activation, include_bias=True):
        """
        Initializes the fully connected layer.
        :param input_dim: Input dimension D_in
        :param output_dim: Output dimension D_out
        :param activation: Activation function object
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cache = None
        self.W = None # weights and bias
        self.activation = activation
        self.grad = None
        self.include_bias = include_bias

        self.initialize_weights(include_bias)

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
        affine = x @ self.W
        z = self.activation(affine)
        self.cache = {'input': x, 'affine': affine, 'z': z}
        return z

    def backward(self, dout, is_last_layer=False):
        """
        Performs the backward pass of the fully connected layer.
        :param dout: Upstream gradient of shape (N, D_out)
        :param is_last_layer: Whether this layer is the last layer in the network
        :return: gradient for weights update of shape (D_in, D_out)
        """
        if self.cache is None:
            raise ValueError("No cache found. Perform forward pass before backward.")

        # Retrieve cached values
        x = self.cache['input'] # shape (N, D_in)
        affine = self.cache['affine']
        z = self.cache['z']
        N = x.shape[0]

        dZ = dout * self.activation.backward(z)  # (N, D_out)

        # gradient for weights
        self.grad = (x.T @ dZ) / N  # (D_in+1, D_out)

        # gradient wrt input (for chaining)
        dX = dZ @ self.W.T  # (N, D_in+1)
        if self.include_bias:
            dX = dX[:, :-1]  # drop bias part

        return dX

    def initialize_weights(self, include_bias=True):
        """
        Initialize the weight matrix W
        :param include_bias: Whether to include bias term in weights
        :return: None
        """
        input_dim = self.input_dim + 1 if include_bias else self.input_dim

        if isinstance(self.activation, Sigmoid):
            # xavier uniform initialization
            limit = np.sqrt(6 / (input_dim + self.output_dim))
            self.W = np.random.uniform(-limit, limit, (input_dim, self.output_dim))
        elif isinstance(self.activation, ReLU):
            # He initialization
            self.W = np.random.randn(input_dim, self.output_dim) * np.sqrt(2 / input_dim)
            # add small bias to avoid dead neurons
            if include_bias:
                self.W[-1, :] += 0.01
        else:
            # Default to small random values
            self.W = 0.01 * np.random.randn(input_dim, self.output_dim)
