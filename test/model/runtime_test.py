import time
import unittest

import numpy as np

from src.model.activation_function import ReLU, Linear
from src.model.loss_function import MeanSquaredError
from src.model.network import NN
from src.model.layer import FullyConnectedLayer
from src.model.optimizer import Adam


class RuntimeTest(unittest.TestCase):
    def test_fit_runtime(self):
        nn = NN(
            FullyConnectedLayer(65, 256, activation=ReLU()),
            FullyConnectedLayer(256, 64, activation=ReLU()),
            FullyConnectedLayer(64, 1, activation=Linear()),
            loss_function=MeanSquaredError(),
            optimizer=Adam(learning_rate=0.01)
        )

        epochs = 1

        random_input_data = np.random.randn(1, 65)
        random_target_data = np.random.randn(1, 1)
        start_time = time.time()
        for _ in range(10_000 // 150):
            for i in range(6):
                for j in range(25):
                    loss_list, grad_norm_list = nn.train((random_input_data, random_target_data), epochs=1)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Training runtime for 5 epochs on 10,000 samples: {runtime:.2f} seconds")
        print(f"Average training time per epoch: {runtime / epochs:.2f} seconds")