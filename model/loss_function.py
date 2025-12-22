import numpy as np

class LossFunction:
    def forward(self, y_pred, y_true):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, y_pred, y_true):
        raise NotImplementedError("Backward method not implemented.")

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

class MeanSquaredError(LossFunction):
    def forward(self, y_pred, y_true, elementwise=False):
        """
        Compute the Mean Squared Error loss.

        :param elementwise: If True, return the loss for each element; otherwise, return the mean loss.
        :param y_pred: Predicted values, shape (N, 1)
        :param y_true: True values, shape (N, 1)
        :return: Mean Squared Error loss
        """
        loss = (y_true - y_pred) ** 2
        if not elementwise:
            loss = np.mean(loss)
        return loss

    def backward(self, y_pred, y_true):
        """
        Compute the gradient of the Mean Squared Error loss with respect to y_pred.

        :param y_pred: Predicted values, shape (N, 1)
        :param y_true: True values, shape (N, 1)
        :return: Gradient of the loss with respect to y_pred, shape (N, 1)
        """
        N = y_true.shape[0]
        grad = (2 / N) * (y_pred - y_true)
        return grad

class BinaryCrossEntropy(LossFunction):
    def forward(self, y_pred, y_true, elementwise=False):
        """
        Compute the Binary Cross Entropy loss.

        :param elementwise: If True, return the loss for each element; otherwise, return the mean loss.
        :param y_pred: Predicted probabilities, shape (N, 1)
        :param y_true: True binary labels, shape (N, 1)
        :return: Binary Cross Entropy loss
        """
        epsilon = 1e-15  # to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        if not elementwise:
            loss = np.mean(loss)
        return loss

    def backward(self, y_pred, y_true):
        """
        Compute the gradient of the Binary Cross Entropy loss with respect to y_pred.

        :param y_pred: Predicted probabilities, shape (N, 1)
        :param y_true: True binary labels, shape (N, 1)
        :return: Gradient of the loss with respect to y_pred, shape (N, 1)
        """
        epsilon = 1e-15  # to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        N = y_true.shape[0]
        grad = - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        grad = grad / N
        return grad

class BinaryCrossEntropyLossFromLogits(LossFunction):
    def forward(self, y_pred, y_true, elementwise=False):
        loss = np.maximum(y_pred, 0) - y_pred * y_true + np.log(1 + np.exp(-np.abs(y_pred)))
        if not elementwise:
            loss = np.mean(loss)
        return loss

    def backward(self, y_pred, y_true):
        N = y_true.shape[0]
        sigmoid = 1 / (1 + np.exp(-y_pred))
        grad = (sigmoid - y_true) / N
        return grad