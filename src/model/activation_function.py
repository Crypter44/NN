import numpy as np

from src.model.module import Module


class ActivationFunction(Module):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def parameters(self):
        return []

class ReLU(ActivationFunction):
    def forward(self, x):
        self.cache['input'] = x
        return np.maximum(0, x)

    def backward(self, dout):
        x = self.cache.get('input')
        if x is None:
            raise ValueError("No cached input found for backward pass. Perform forward pass first.")
        return (x > 0).astype(float) * dout


class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def backward(self, dout):
        return dout


class Sigmoid(ActivationFunction):
    def forward(self, x):
        self.cache['input'] = x
        z = 1 / (1 + np.exp(-x))
        self.cache['output'] = z
        return z

    def backward(self, dout):
        z = self.cache.get('output')
        if z is None:
            raise ValueError("No cached output found for backward pass. Perform forward pass first.")
        return z * (1 - z) * dout


class Tanh(ActivationFunction):
    def forward(self, x):
        self.cache['input'] = x
        z = np.tanh(x)
        self.cache['output'] = z
        return z

    def backward(self, dout):
        z = self.cache.get('output')
        if z is None:
            raise ValueError("No cached output found for backward pass. Perform forward pass first.")
        return (1 - np.square(z)) * dout

class Softmax(ActivationFunction):
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def backward(self, x):
        raise NotImplementedError("Softmax backward pass should be computed with the loss function.")
