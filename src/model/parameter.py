import numpy as np


class Parameter(np.ndarray):
    def __new__(cls, input_array, gradient=None):
        # Create an instance of Parameter from the input array
        obj = np.asarray(input_array).view(cls)
        obj.gradient = np.zeros_like(input_array)
        return obj

    def zero_grad(self):
        # Reset the gradient to zero
        if self.gradient is not None:
            self.gradient.fill(0)

    def update_grad(self, grad):
        # Update the gradient with the provided value
        self.gradient[...] += grad

    def __array_finalize__(self, obj):
        # Called whenever a new Parameter is created from an existing one
        if obj is None: return
        self.gradient = getattr(obj, 'gradient', None)

