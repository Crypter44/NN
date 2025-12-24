import numpy as np

from src.dataloader.datasets import MNISTDataset
from src.model.activation_function import ReLU, Linear
from src.model.layer import FullyConnectedLayer
from src.model.loss_function import SoftmaxCrossEntropy
from src.model.network import NN
from src.model.optimizer import Adam
from src.utils.image_classification import plot_images_with_colored_labels
from src.utils.utils import plot_loss_curve, set_seed

set_seed(526)

data = MNISTDataset(
    train=True,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    normalize_data=True,
)

print("MNIST train dataset loaded.")

nn = NN(
    FullyConnectedLayer(784, 512, activation=ReLU()),
    FullyConnectedLayer(512, 64, activation=ReLU()),
    FullyConnectedLayer(64, 10, activation=Linear()),
    loss_function=SoftmaxCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
)

loss_list, grad_norm_list = nn.train(data, epochs=20)

print("Training completed.")

# plot loss curve
plot_loss_curve(loss_list)

images, labels = next(iter(data))

# Visualize some predictions
plot_images_with_colored_labels(
    images.reshape(-1, 28, 28),
    nn(images).argmax(axis=1).astype(int),
    labels.argmax(axis=1).astype(int),
)

# calculate accuracy on train set
train_predictions = nn(data.data).argmax(axis=1).astype(int)
train_accuracy = np.mean(train_predictions == data.targets.argmax(axis=1).astype(int))
print(f"Train accuracy: {train_accuracy * 100:.2f}%")

# test set
test_data = MNISTDataset(
    train=False,
    batch_size=-1,
    shuffle=False,
    drop_last=False,
    normalize_data=True,
)

test_images, test_labels = next(iter(test_data))
test_predictions = nn(test_images).argmax(axis=1).astype(int)

# calculate test accuracy
test_accuracy = np.mean(test_predictions == test_labels.argmax(axis=1).astype(int))
print(f"Test accuracy: {test_accuracy * 100:.2f}%")