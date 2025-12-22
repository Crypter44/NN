from _pytest import unittest

from model.optimizer import Adam


class TestOptimizerDeepcopy(unittest.TestCase):

    def test_independent_velocity(self):
        opt = Adam(0.01)
        opt_copy = Adam(0.01)
        # Simulate gradients
        weights = 1.0
        grads = 0.1

        # Perform an update on the original optimizer
        print(opt.update(weights, grads))

        # Perform an update on the copied optimizer
        opt_copy.beta1 = 10000
        opt_copy.beta2 = 10000
        print(opt_copy.update(weights, grads))
        print(opt_copy.beta1)
        print(opt.beta1)

        # Check that the velocities are different
        self.assertNotEqual(opt.m, opt_copy.m)
        self.assertNotEqual(opt.v, opt_copy.v)