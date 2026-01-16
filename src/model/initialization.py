import numpy as np

from src.model.parameter import Parameter


class Initialization:
    """
    Abstract base class for parameter initialization strategies.
    """

    def initialize(self, parameter: Parameter):
        """
        Initializes the given parameter.
        :param parameter: the parameter to initialize
        :return:
        """
        raise NotImplementedError

    def __call__(self, parameter: Parameter):
        return self.initialize(parameter)


class XavierInitialization(Initialization):
    """
    Xavier uniform initialization strategy.
    """

    def initialize(self, parameter: Parameter):
        """
        Xavier uniform initialization.

        Useful for layers with sigmoid or tanh activation functions.
        :param parameter:
        :return:
        """
        # xavier uniform initialization
        input_dim = parameter.shape[0]
        output_dim = parameter.shape[1]
        limit = np.sqrt(6 / (input_dim + output_dim))
        result = np.random.uniform(-limit, limit, (input_dim, output_dim))
        np.copyto(parameter, result)


class XavierInitializationNormal(Initialization):
    """
    Xavier normal initialization strategy.
    """
    def __init__(self, dead_relu_protection: bool = False):
        self.dead_relu_protection = dead_relu_protection

    def initialize(self, parameter: Parameter):
        """
        Xavier normal initialization with optional dead ReLU protection.

        Useful for layers with ReLU activation to prevent dead neurons.
        :param parameter: the parameter to initialize
        :param dead_relu_protection: whether to add small bias to avoid dead ReLU neurons
        :return:
        """
        # xavier normal initialization
        input_dim = parameter.shape[0]
        output_dim = parameter.shape[1]
        stddev = np.sqrt(2 / (input_dim + output_dim))
        result = np.random.normal(0, stddev, (input_dim, output_dim))
        if self.dead_relu_protection:
            # add small bias to avoid dead neurons
            result[-1, :] += 0.01
        np.copyto(parameter, result)
