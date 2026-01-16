import warnings
from copy import deepcopy
import pickle
import numpy as np
from tqdm import tqdm

from src.dataloader.dataloader import Dataloader
from src.utils.utils import outdated


class NN:
    def __init__(self, *layers, loss_function, optimizer):
        """
        Initializes the neural network with given layers, loss function and optimizer.

        :param layers: layers of the neural network (in order and as separate arguments)
        :param loss_function: loss function to use
        :param optimizer: optimizer to use for all layers
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer

        for layer in self.layers:
            self.optimizer.register_parameters(layer.parameters())

    def train(self):
        """
        Sets all layers to training mode.
        """
        for layer in self.layers:
            layer.train()

    def eval(self):
        """
        Sets all layers to evaluation mode.
        """
        for layer in self.layers:
            layer.eval()

    def run_training(self, data, epochs=100, eval_data: Dataloader = None, **kwargs) -> dict:
        """
        Trains the neural network on the given data for a number of epochs.

        If data is not a Dataloader, it will be converted to one with batch_size=len(data),
        shuffle=False, drop_last=False, to simulate full-batch training on the provided data.
        :param data: data to train on (Dataloader or tuple of (data, targets))
        :param epochs: number of epochs to train for
        :param eval_data: optional evaluation data to evaluate on after each epoch
        :kwargs: additional arguments (not used, but currently needed for backwards compatibility from older versions)
        :return: info dictionary containing various training statistics
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

        # set up info dictionary
        info = {
            'loss_list': [],
            'grad_norm_list': []
        }
        if eval_data is not None:
            info['eval_loss_list'] = []
        patience = kwargs.get('patience', 10)
        best_loss = np.inf

        # Training loop
        for epoch in range(epochs):
            self.train()
            batch_loss = 0
            batch_grad_norm = 0

            pbar = tqdm(data, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            for batch in pbar:
                data_batch, targets_batch = batch
                # Forward pass
                output = self.forward(data_batch)
                # Compute loss
                loss = self.loss_function(output, targets_batch)
                batch_loss += loss
                # Backward pass
                self.optimizer.zero_grad()
                grad_norm = self.backward(output, targets_batch)
                batch_grad_norm += grad_norm
                # optimize
                self.optimizer.step()

            info["loss_list"].append(batch_loss / data.num_batches())
            info["grad_norm_list"].append(batch_grad_norm / data.num_batches())

            # Evaluation for early stopping or monitoring
            if eval_data is not None:
                self.eval()
                eval_predictions, eval_loss = self.evaluate(eval_data)
                info["eval_loss_list"].append(eval_loss)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.save_weights("best_model_weights")
                    patience = kwargs.get('patience', 10)
                else:
                    patience -= 1
                    if patience == 0:
                        print("Early stopping triggered.")
                        self.load_weights("best_model_weights")
                        return info


            tqdm.write(
                f"Epoch {epoch + 1}/{epochs}: "
                f"          Loss: {info['loss_list'][-1]:.4f}, "
                f"          Grad Norm: {info['grad_norm_list'][-1]:.4f}"
                f"          Eval Loss: {info['eval_loss_list'][-1]:.4f}" if eval_data is not None else "N/A"
                f"          Patience: {patience}" if eval_data is not None else "N/A"
            )

        return info

    def forward(self, input_data):
        """
        Performs a forward pass through the network.
        :param input_data: the input data
        :return: the output after passing through all layers
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output, targets):
        """
        Performs a backward pass through the network.
        :param input_data: the gradient of the loss with respect to the output
        :return: the norm of the gradients across all layers
        """
        grad = self.loss_function.backward(output, targets)
        grad_norm = np.linalg.norm(grad)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_norm += np.linalg.norm(grad)

        return grad_norm / len(self.layers)

    def predict(self, input_data):
        """
        Predicts the output for the given input data.
        :param input_data: the input data
        :return: the predicted output
        """
        return self.forward(input_data)

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

    def save_weights(self, filepath):
        weights = [layer.parameters() for layer in self.layers]
        with open(filepath + ".pkl", "wb") as f:
            pickle.dump(weights, f)

    def load_weights(self, filepath):
        with open(filepath + ".pkl", "rb") as f:
            weights = pickle.load(f)
        for parameters, layer in zip(weights, self.layers):
            for p, lp in zip(parameters, layer.parameters()):
                lp[...] = deepcopy(p)
