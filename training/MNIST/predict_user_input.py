from src.model.activation_function import Linear, ReLU
from src.model.layer import FullyConnectedLayer
from src.model.loss_function import SoftmaxCrossEntropy
from src.model.network import NN
from src.model.optimizer import Adam
from src.utils.number_painter_MNIST import MNISTDrawer

nn = NN(
    FullyConnectedLayer(784, 512, activation=ReLU()),
    FullyConnectedLayer(512, 64, activation=ReLU()),
    FullyConnectedLayer(64, 10, activation=Linear()),
    loss_function=SoftmaxCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
)

nn.load_weights("weights")
print("Model weights loaded.")

drawer = MNISTDrawer(nn)
drawer.run()
