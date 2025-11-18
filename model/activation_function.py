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
        sig = self.forward(x)
        return sig * (1 - sig)  # elementwise derivative