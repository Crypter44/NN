import numpy as np

from src.dataloader.datasets import MNISTDataset
from src.dataloader.transformation import RandomTranslationWithPadding, ChainTransformation, Flatten
from src.model import activation_function as af
from src.model.initialization import XavierInitializationNormal
from src.model.module import LinearLayer, DropOut
from src.model.loss_function import SoftmaxCrossEntropy
from src.model.network import NN
from src.model.optimizer import Adam
from src.utils.image_classification import plot_images_with_colored_labels
from src.utils.utils import plot_loss_curve, set_seed

set_seed(526)

image_size = 40

data = MNISTDataset(
    train=True,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    transformation=ChainTransformation(
        RandomTranslationWithPadding((image_size, image_size)),
        Flatten()
    )

)

test_data = MNISTDataset(
    train=False,
    batch_size=-1,
    shuffle=False,
    drop_last=False,
    transformation=ChainTransformation(
        RandomTranslationWithPadding((image_size, image_size)),
        Flatten()
    )
)

print("MNIST train dataset loaded.")

nn = NN(
    DropOut(0.05),
    LinearLayer(image_size ** 2, 512, init_method=XavierInitializationNormal(True)),
    af.ReLU(),
    DropOut(0.2),
    LinearLayer(512, 64, init_method=XavierInitializationNormal(True)),
    af.ReLU(),
    LinearLayer(64, 10, init_method=XavierInitializationNormal(True)),
    af.Linear(),
    loss_function=SoftmaxCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
)

nn.train()
info = nn.run_training(data, epochs=100, eval_data=test_data, patience=5)

loss_list = info["train_loss"]
eval_loss_list = info["eval_loss"]

print("Training completed.")

nn.eval()
plot_loss_curve(loss_list, eval_loss_list)

images, labels = next(iter(data))

# Visualize some predictions
plot_images_with_colored_labels(
    images.reshape(-1, image_size, image_size),
    nn(images).argmax(axis=1).astype(int),
    labels.argmax(axis=1).astype(int),
)

# calculate accuracy on train set
train_predictions = nn(data.data).argmax(axis=1).astype(int)
train_accuracy = np.mean(train_predictions == data.targets.argmax(axis=1).astype(int))
print(f"Train accuracy: {train_accuracy * 100:.4f}%")

test_images, test_labels = next(iter(test_data))
test_predictions = nn(test_images).argmax(axis=1).astype(int)

# calculate test accuracy
test_accuracy = np.mean(test_predictions == test_labels.argmax(axis=1).astype(int))
print(f"Test accuracy: {test_accuracy * 100:.4f}%")

nn.save_weights("weights")