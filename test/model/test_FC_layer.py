import numpy as np
from _pytest import unittest

from src.model import module as mdl
from src.model import optimizer as optim
from src.model.loss_function import MeanSquaredError


class TestFcLayer(unittest.TestCase):
    def test_FCL_forward(self):
        # Create a FullyConnectedLayer instance
        input_size = 2
        output_size = 1
        fc_layer = mdl.LinearLayer(input_size, output_size, include_bias=False)

        assert fc_layer.W.shape == (input_size, output_size), \
            f"Weight matrix shape expected to be {(input_size, output_size)}, but got {fc_layer.W.shape}"

        for i in range(len(fc_layer.W)):
            for j in range(len(fc_layer.W[0])):
                fc_layer.W[i][j] = i + j + 1

        # Define a simple input
        x = np.array([[1.0, 2.0],
                      [3.0, 4.0]])

        # Perform forward pass
        output = fc_layer.forward(x)
        expected_output_shape = (x.shape[0], output_size)
        assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {output.shape}"
        assert np.allclose(output, np.array([[5.0], [11.0]])), \
            f"Forward pass output expected to be [[5.0], [11.0]], but got {output}"

    def test_FCL_approximate_linear_function(self):
        # Create a FullyConnectedLayer instance
        input_size = 2
        output_size = 1
        fc_layer = mdl.LinearLayer(input_size, output_size, include_bias=True)

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

        learning_rate = 0.01
        loss_function = MeanSquaredError()
        epochs = 50000
        training_loss = []
        optimizer = optim.Adam(learning_rate=learning_rate)
        optimizer.register_parameters(fc_layer.parameters())
        for epoch in range(epochs):
            # Forward pass
            predictions = fc_layer.forward(data)

            # Compute Mean Squared Error loss
            loss = loss_function.forward(predictions, targets, elementwise=True)
            training_loss.append(np.mean(loss))

            # Backward pass
            fc_layer.backward(loss_function.backward(predictions, targets))
            # Update weights
            optimizer.step()
            optimizer.zero_grad()

        # plot loss curve
        import matplotlib.pyplot as plt
        plt.plot(training_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error Loss')
        plt.title('Training Loss Curve')
        plt.show()

        # print final weights
        print("\nFinal weights after training:")
        print(fc_layer.W)

        # Final evaluation
        final_predictions = fc_layer.forward(data)
        final_loss = loss_function.forward(final_predictions, targets)
        assert final_loss < 0.1, \
            f"Final loss expected to be less than 0.1, but got {final_loss}"

        test_data = np.array([[6.0, 5.0],
                              [7.0, 6.0]])
        test_targets = np.array([[11.0], [13.0]]) + 50
        test_predictions = fc_layer.forward(test_data)
        test_loss = loss_function.forward(test_predictions, test_targets)
        assert test_loss < 0.1, \
            f"Test loss expected to be less than 0.1, but got {test_loss}"