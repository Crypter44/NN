import numpy as np


class ActivationFunction:
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, x):
        raise NotImplementedError("Backward method not implemented.")

    def __call__(self, x):
        return self.forward(x)


class ReLU(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return (x > 0).astype(float)  # elementwise derivative


class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return x * (1 - x)


class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - np.square(np.tanh(x))

class Softmax(ActivationFunction):
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def backward(self, x):
        raise NotImplementedError("Softmax backward pass should be computed with the loss function.")
