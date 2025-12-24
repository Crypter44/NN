import warnings
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from src.dataloader.dataloader import Dataloader


class NN:
    def __init__(self, *layers, loss_function, optimizer):
        """
        Initializes the neural network with given layers, loss function and optimizer.

        At the moment, the optimizer is copied for each layer, so each layer has its own optimizer instance.
        This allows for optimizers that maintain state (e.g. momentum, Adam, etc.).
        Later on, this may be changed to allow for more flexible optimizer assignment.
        :param layers: layers of the neural network (in order and as separate arguments)
        :param loss_function: loss function to use
        :param optimizer: optimizer to use for all layers
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizers = {l: deepcopy(optimizer) for l in self.layers}

    def train(self, data, epochs=100, **kwargs):
        """
        Trains the neural network on the given data for a number of epochs.

        If data is not a Dataloader, it will be converted to one with batch_size=len(data),
        shuffle=False, drop_last=False, to simulate full-batch training on the provided data.
        :param data: data to train on (Dataloader or tuple of (data, targets))
        :param epochs: number of epochs to train for
        :kwargs: additional arguments (not used, but currently needed for backwards compatibility from older versions)
        :return: loss_list: list of losses for each batch,
                 grad_norm_list: list of gradient norms for each layer and batch
        """
        if not isinstance(data, Dataloader):
            warnings.warn(
                "Data is not a Dataloader, will try to convert it to one with batch_size=len(data), shuffle=False, drop_last=False.")
            if not isinstance(data, tuple) or len(data) != 2:
                raise ValueError("Data must be a tuple of (data, targets) to convert to Dataloader.")
            batch_size = len(data[0])
            data = Dataloader(data[0], data[1], batch_size=batch_size, shuffle=False, drop_last=False)
            data.print(True)
            print(f"Data converted to Dataloader with batch_size={batch_size}, shuffle=False, drop_last=False.")

        loss_list = []
        grad_norm_list = []

        pbar = tqdm(range(epochs), desc="Training", unit="epoch")

        for epoch in pbar:
            batch_loss_list = []
            batch_grad_norm_list = []
            for batch in data:
                data_batch, targets_batch = batch
                # Forward pass
                output = data_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss
                loss = self.loss_function(output, targets_batch)
                batch_loss_list.append(loss)

                # Backward pass
                grad = self.loss_function.backward(output, targets_batch)
                last_layer = True
                for layer in reversed(self.layers):
                    layer.backward(grad, is_last_layer=last_layer)
                    if last_layer:
                        last_layer = False
                    grad = layer.backward(grad)
                    batch_grad_norm_list.append(np.linalg.norm(grad))
                    layer.W = self.optimizers[layer].update(layer.W, layer.grad)
            loss_list.append(np.mean(batch_loss_list))
            grad_norm_list.append(np.mean(batch_grad_norm_list))

        return loss_list, grad_norm_list

    def predict(self, input_data):
        """
        Predicts the output for the given input data.
        :param input_data: the input data
        :return: the predicted output
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def __call__(self, *args, **kwargs):
        """
        Override for calling the network instance directly.
        :param args: positional arguments
        :param kwargs: additional arguments
        :return: predicted output
        """
        return self.predict(*args, **kwargs)

    def evaluate(self, data, targets):
        """
        Evaluates the network on the given data and targets.
        :param data: the input data
        :param targets: the target outputs
        :return: tuple of (predictions, loss)
        """
        predictions = self.predict(data)
        loss = self.loss_function(predictions, targets)
        return predictions, loss
