import numpy as np

from src.model.network import NN
from src.model import optimizer as optim, loss_function as loss, layer, activation_function as activation
from src.dataloader.datasets import CircleDataset, PolygonDataset, SpiralDataset

from src.utils import binary_classification_problem_2d as bcp2d

hidden_size = 64
radius = 0.5
nn = NN(
    layer.FullyConnectedLayer(2, hidden_size, activation.ReLU()),
    layer.FullyConnectedLayer(hidden_size, hidden_size, activation.ReLU()),
    layer.FullyConnectedLayer(hidden_size, hidden_size, activation.ReLU()),
    layer.FullyConnectedLayer(hidden_size, 1, activation.Sigmoid()),
    loss_function=loss.BinaryCrossEntropy(),
    optimizer=optim.Adam(learning_rate=0.001)
)

circle_data = CircleDataset(radius, 2000, batch_size=1, shuffle=True, drop_last=True, normalize_data=False)

polygon_vertices = np.array(
    [[0.05956074, -0.97678535],
     [0.42807192, -0.40685457],
     [0.84576034, -0.0403644],
     [0.04659918, 0.05901351],
     [0.01321098, 0.19017006],
     [-0.34544488, 0.40700824],
     [-0.22050716, -0.00302628],
     [-0.41997357, -0.49790762]])
polygon_data = PolygonDataset(
    polygon_vertices=polygon_vertices,
    num_points=10_000,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    normalize_data=False
)
print("Polygon vertices:", polygon_data.polygon_vertices)

spiral_data = SpiralDataset(
    num_points=4_000,
    turns=3,
    noise=0.1,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    normalize_data=False
)

problem = bcp2d.BinaryClassification2DProblem(
    train_set=spiral_data,
    test_set=None,
    model=nn,
)

losses, grad_norms = problem.run_training_and_animate(
    epochs=750,
    interval=1,
    include_test_set=False,
    length_of_frame=100,
)

print("Final Loss:", losses[-1])
