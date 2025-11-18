import numpy as np
import pytest
from _pytest import unittest

from model.network import NN
import model.layer as layer
import model.activation_function as activation
import model.loss_function as loss


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(521)


class TestNN(unittest.TestCase):
    def test_linear_function_fit(self):
        nn = NN(
            layer.FullyConnectedLayer(2, 2, activation.Linear()),
            layer.FullyConnectedLayer(2, 1, activation.Linear()),
            loss_function=loss.MeanSquaredError(),
            learning_rate=0.01
        )

        data = np.array([[0.0, 0.0],
                         [1.0, 1.0],
                         [2.0, 2.0],
                         [3.0, 3.0],
                         [4.0, 4.0],
                         [1.0, 0.0],
                         [2.0, 1.0],
                         [3.0, 2.0],
                         [4.0, 3.0],
                         [5.0, 4.0]])

        targets = np.array([[0.0],
                            [2.0],
                            [4.0],
                            [6.0],
                            [8.0],
                            [1.0],
                            [3.0],
                            [5.0],
                            [7.0],
                            [9.0]]) + 50

        losses = nn.train(data, targets, epochs=500)

        # plot the loss curve
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error Loss')
        plt.title('Linear Function Training Loss Curve')
        plt.show()

        assert losses[-1] < 1e-2, f"Final loss expected to be less than 1e-2, but got {losses[-1]}"

    def test_sign_function_fit(self):
        nn = NN(
            layer.FullyConnectedLayer(1, 1, activation.Sigmoid()),
            loss_function=loss.BinaryCrossEntropy(),
            learning_rate=0.1
        )

        data = np.array([[5],
                         [10],
                         [15],
                         [20],
                         [25],
                         [30],
                         [35],
                         [40],
                         [45],
                         [50],
                         [-5],
                         [-10],
                         [-15],
                         [-20],
                         [-25],
                         [-30],
                         [-35],
                         [-40],
                         [-45],
                         [-50]])

        targets = np.array([[1],
                            [1],
                            [1],
                            [1],
                            [1],
                            [1],
                            [1],
                            [1],
                            [1],
                            [1],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0],
                            [0]])

        losses = nn.train(data, targets, epochs=5_000)

        # plot the loss curve
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title('Binary Classification Training Loss Curve')
        plt.show()

        assert losses[-1] < 0.1, f"Final loss expected to be less than 0.1, but got {losses[-1]}"

        # Test that the sign function was successfully learned
        test_data = np.array([[526], [-342], [13], [-7], [0.5], [-0.5]])
        test_targets = np.array([[1], [0], [1], [0], [1], [0]])
        test_predictions = nn.predict(test_data)
        test_predictions = (test_predictions >= 0.5).astype(np.float32)
        assert np.array_equal(test_predictions, test_targets), \
            f"Predictions {test_predictions.flatten()} do not match targets {test_targets.flatten()}"

    def test_not_gate_fit(self):
        nn = NN(
            layer.FullyConnectedLayer(1, 2, activation.Sigmoid()),
            layer.FullyConnectedLayer(2, 1, activation.Sigmoid()),
            loss_function=loss.BinaryCrossEntropy(),
            learning_rate=0.1
        )

        data = np.array([[0],
                         [1]])

        targets = np.array([[1],
                            [0]])

        losses = nn.train(data, targets, epochs=10_000)

        # plot the loss curve
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title('NOT Gate Training Loss Curve')
        plt.show()

        assert losses[-1] < 0.1, f"Final loss expected to be less than 0.1, but got {losses[-1]}"

        # Test that the NOT gate was successfully learned
        test_data = np.array([[0], [1]])
        test_targets = np.array([[1], [0]])
        test_predictions = nn.predict(test_data)
        test_predictions = (test_predictions >= 0.5).astype(np.float32)
        assert np.array_equal(test_predictions, test_targets), \
            f"Predictions {test_predictions.flatten()} do not match targets {test_targets.flatten()}"

    def test_and_gate_fit(self):
        nn = NN(
            layer.FullyConnectedLayer(2, 2, activation.Sigmoid()),
            layer.FullyConnectedLayer(2, 1, activation.Sigmoid()),
            loss_function=loss.BinaryCrossEntropy(),
            learning_rate=0.1
        )

        data = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

        targets = np.array([[0],
                            [0],
                            [0],
                            [1]])

        losses = nn.train(data, targets, epochs=10_000)

        # plot the loss curve
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title('AND Gate Training Loss Curve')
        plt.show()

        # Test that the AND gate was successfully learned
        test_data = np.array([[0, 0],
                              [0, 1],
                              [1, 0],
                              [1, 1]])
        test_targets = np.array([[0],
                                 [0],
                                 [0],
                                 [1]])
        test_predictions = nn.predict(test_data)
        test_predictions = (test_predictions >= 0.5).astype(np.float32)
        assert np.array_equal(test_predictions, test_targets), \
            f"Predictions {test_predictions.flatten()} do not match targets {test_targets.flatten()}"

        assert losses[-1] < 0.1, f"Final loss expected to be less than 0.1, but got {losses[-1]}"

    def test_or_gate_fit(self):
        nn = NN(
            layer.FullyConnectedLayer(2, 2, activation.Sigmoid()),
            layer.FullyConnectedLayer(2, 1, activation.Sigmoid()),
            loss_function=loss.BinaryCrossEntropy(),
            learning_rate=0.5
        )

        data = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

        targets = np.array([[0],
                            [1],
                            [1],
                            [1]])

        losses = nn.train(data, targets, epochs=10_000)

        # plot the loss curve
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title('OR Gate Training Loss Curve')
        plt.show()

        assert losses[-1] < 0.1, f"Final loss expected to be less than 0.1, but got {losses[-1]}"

        # Test that the OR gate was successfully learned
        test_data = np.array([[0, 0],
                              [0, 1],
                              [1, 0],
                              [1, 1]])
        test_targets = np.array([[0],
                                 [1],
                                 [1],
                                 [1]])
        test_predictions = nn.predict(test_data)
        test_predictions = (test_predictions >= 0.5).astype(np.float32)
        assert np.array_equal(test_predictions, test_targets), \
            f"Predictions {test_predictions.flatten()} do not match targets {test_targets.flatten()}"

    def test_xor_gate_fit(self):
        nn = NN(
            layer.FullyConnectedLayer(2, 2, activation.ReLU()),
            layer.FullyConnectedLayer(2, 1, activation.Sigmoid()),
            loss_function=loss.BinaryCrossEntropy(),
            learning_rate=0.5
        )

        data = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

        targets = np.array([[0],
                            [1],
                            [1],
                            [0]])

        losses = nn.train(data, targets, epochs=10_000, shuffle=False)

        # plot the loss curve
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title('XOR Gate Training Loss Curve')
        plt.show()

        assert losses[-1] < 0.1, f"Final loss expected to be less than 0.1, but got {losses[-1]}"

        # Test that the XOR gate was successfully learned
        test_data = np.array([[0, 0],
                              [0, 1],
                              [1, 0],
                              [1, 1]])
        test_targets = np.array([[0],
                                 [1],
                                 [1],
                                 [0]])
        test_predictions = nn.predict(test_data)
        test_predictions = (test_predictions >= 0.5).astype(np.float32)
        assert np.array_equal(test_predictions, test_targets), \
            f"Predictions {test_predictions.flatten()} do not match targets {test_targets.flatten()}"


