import numpy as np


class Optimizer:
    def update(self, weights, grads):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, grads, learning_rate=None):
        learning_rate = learning_rate if learning_rate else self.learning_rate
        weights = weights - learning_rate * grads
        return weights

class SGDMomentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, weights, grads, learning_rate=None):
        learning_rate = learning_rate if learning_rate else self.learning_rate
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)
        self.velocity = self.momentum * self.velocity - learning_rate * grads
        weights = weights + self.velocity
        return weights

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, grads, learning_rate=None):
        learning_rate = learning_rate if learning_rate else self.learning_rate
        if self.m is None:
            self.m = np.zeros_like(weights)
        if self.v is None:
            self.v = np.zeros_like(weights)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return weights