import numpy as np


class NN:
    def __init__(self, *layers, loss_function, learning_rate=0.01):
        self.layers = layers
        self.loss_function = loss_function
        self.learning_rate = learning_rate


    def train(self, data, targets, epochs=100, shuffle=False):
        loss_list = []
        for epoch in range(epochs):
            # Forward pass
            output = data
            if shuffle:
                # Shuffle data and targets in unison
                perm = np.random.permutation(data.shape[0])
                output = data[perm]
                targets = targets[perm]
            for layer in self.layers:
                output = layer.forward(output)

            # Compute loss
            loss = self.loss_function(output, targets)
            loss_list.append(loss)

            # Backward pass
            grad = self.loss_function.backward(output, targets)
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
                layer.update(self.learning_rate)

        return loss_list

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def evaluate(self, data, targets):
        predictions = self.predict(data)
        loss = self.loss_function(predictions, targets)
        return loss