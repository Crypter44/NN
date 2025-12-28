import numpy as np

from src.model.parameter import Parameter
from src.utils.utils import outdated


class Optimizer:
    def __init__(self):
        self.internal_state = {}
        self.registered_parameters: list[Parameter] = []

    def register_parameters(self, params: list[Parameter]):
        self.registered_parameters += params

    def zero_grad(self):
        for param in self.registered_parameters:
            param.zero_grad()

    def step(self, module):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.learning_rate = learning_rate

    def step(self, learning_rate=None):
        learning_rate = learning_rate if learning_rate else self.learning_rate

        for param in self.registered_parameters:
            param[...] -= learning_rate * param.gradient

class SGDMomentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum

    def step(self, learning_rate=None):
        learning_rate = learning_rate if learning_rate else self.learning_rate

        for param in self.registered_parameters:
            if id(param) not in self.internal_state:
                self.internal_state[id(param)] = np.zeros_like(param)
            velocity = self.internal_state[id(param)]
            grad = param.gradient
            velocity = self.momentum * velocity - learning_rate * grad
            param[...] += velocity
            self.internal_state[id(param)] = velocity

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def step(self, learning_rate=None):
        learning_rate = learning_rate if learning_rate else self.learning_rate

        for param in self.registered_parameters:
            if id(param) not in self.internal_state:
                self.internal_state[id(param)] = {
                    'm': np.zeros_like(param),
                    'v': np.zeros_like(param),
                    't': 0
                }
            state = self.internal_state[id(param)]
            m = state['m']
            v = state['v']
            t = state['t']

            grad = param.gradient

            t += 1
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            param[...] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.internal_state[id(param)]['m'] = m
            self.internal_state[id(param)]['v'] = v
            self.internal_state[id(param)]['t'] = t
