import numpy as np
import pytest
from _pytest import unittest

from dataloader.datasets import CircleDataset
from model.network import NN
import model.layer as layer
import model.activation_function as activation
import model.loss_function as loss
import model.optimizer as optim


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(521)


class TestNN(unittest.TestCase):
    def test_linear_function_fit(self):
        nn = NN(
            layer.FullyConnectedLayer(2, 4, activation.Linear()),
            layer.FullyConnectedLayer(4, 1, activation.Linear()),
            optimizer=optim.SGD(learning_rate=0.01),
            loss_function=loss.MeanSquaredError(),
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

        losses, _ = nn.train((data, targets), epochs=500)

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
            optimizer=optim.SGD(learning_rate=0.1)
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

        losses, _ = nn.train((data, targets), epochs=5_000)

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
            optimizer=optim.SGD(learning_rate=0.1)
        )

        data = np.array([[0],
                         [1]])

        targets = np.array([[1],
                            [0]])

        losses, _ = nn.train((data, targets), epochs=10_000)

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
            optimizer=optim.SGD(learning_rate=0.1)
        )

        data = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

        targets = np.array([[0],
                            [0],
                            [0],
                            [1]])

        losses, _ = nn.train((data, targets), epochs=10_000)

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
            optimizer=optim.SGD(learning_rate=0.5)
        )

        data = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

        targets = np.array([[0],
                            [1],
                            [1],
                            [1]])

        losses, _ = nn.train((data, targets), epochs=10_000)

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
            layer.FullyConnectedLayer(2, 4, activation.ReLU()),
            layer.FullyConnectedLayer(4, 1, activation.Sigmoid()),
            loss_function=loss.BinaryCrossEntropy(),
            optimizer=optim.SGD(learning_rate=0.5)
        )

        data = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

        targets = np.array([[0],
                            [1],
                            [1],
                            [0]])

        losses, _ = nn.train((data, targets), epochs=10_000, shuffle=False)

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

    def test_xor_on_multiple_seeds(self):
        print()
        num_tests = 100
        fail = 0
        for seed in range(num_tests):
            np.random.seed(seed)
            nn = NN(
                layer.FullyConnectedLayer(2, 16, activation.ReLU()),
                layer.FullyConnectedLayer(16, 16, activation.ReLU()),
                layer.FullyConnectedLayer(16, 1, activation.Sigmoid()),
                loss_function=loss.BinaryCrossEntropy(),
                optimizer=optim.Adam(learning_rate=0.01)
            )

            data = np.array([[0, 0],
                             [0, 1],
                             [1, 0],
                             [1, 1]])

            targets = np.array([[0],
                                [1],
                                [1],
                                [0]])

            losses, _ = nn.train((data, targets), epochs=2_000, shuffle=False)

            if losses[-1] >= 0.1:
                print(f"Final loss expected to be less than 0.1, but got {losses[-1]} for seed {seed}")
                fail += 1
                continue

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

            if not np.array_equal(test_predictions, test_targets):
                print(f"Predictions do not match targets for seed {seed}")
                fail += 1

        if fail != 0:
            self.fail(f"XOR gate test failed at {fail} out of {num_tests} different random seeds")

    def test_circle_prediction(self):
        hidden_size = 64
        nn = NN(
            layer.FullyConnectedLayer(2, hidden_size, activation.ReLU()),
            layer.FullyConnectedLayer(hidden_size, hidden_size, activation.ReLU()),
            layer.FullyConnectedLayer(hidden_size, 1, activation.Sigmoid()),
            loss_function=loss.BinaryCrossEntropy(),
            optimizer=optim.Adam(learning_rate=0.01)
        )

        data = CircleDataset(0.6, 2000, batch_size=256, shuffle=True, drop_last=True, normalize_data=True)

        losses, _ = nn.train(data, epochs=250)

        # plot the loss curve
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.title('Circle Classification Training Loss Curve')
        plt.show()

        # plot random points, circle and predictions
        import matplotlib.pyplot as plt
        data = CircleDataset(0.3, 1024, batch_size=50, shuffle=False, drop_last=False, normalize_data=True)
        predictions, l = nn.evaluate(data.data, data.targets)

        print(l)
        print(predictions.flatten())
        predictions = (predictions >= 0.5).astype(np.float32)
        print(predictions.flatten())
        print(data.targets.flatten())
        plt.scatter(data.raw_data[:, 0], data.raw_data[:, 1], c=data.targets.flatten(), cmap='coolwarm', edgecolors='k')
        circle = plt.Circle((0.5, 0.5), 0.3, color='black', fill=False, linestyle='--')
        plt.gca().add_artist(circle)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Circle Classification Predictions')
        plt.show()

        assert losses[-1] < 0.1, f"Final loss expected to be less than 0.1, but got {losses[-1]}"
