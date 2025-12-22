import numpy as np
from _pytest import unittest

from dataloader.dataloader import Dataloader
from dataloader.datasets import CircleDataset
from model import layer, activation_function as activation, loss_function as loss, optimizer as optim
from model.network import NN


class OptimizerTest(unittest.TestCase):
    def generate_linear_function(self, n_points=200):
        data = np.random.rand(n_points, 2)
        function = lambda x: 2 * x[:, 0] + 3 * x[:, 1] + 1
        targets = function(data).reshape(-1, 1)
        return data, targets

    def test_compare_optimizers(self):
        optimizers = [
            optim.SGD(learning_rate=0.01),
            optim.SGDMomentum(learning_rate=0.01, momentum=0.9),
            optim.Adam(learning_rate=0.01)
        ]

        results_loss = {}
        results_gradnorm = {}
        num_tests = 100
        n_points = 200
        for opt in optimizers:
            results_loss[type(opt).__name__] = []
            results_gradnorm[type(opt).__name__] = []
            for x in range(num_tests):
                nn = NN(
                    layer.FullyConnectedLayer(2, 4, activation.ReLU()),
                    layer.FullyConnectedLayer(4, 1, activation.Linear()),
                    loss_function=loss.MeanSquaredError(),
                    optimizer=opt
                )

                data, targets = self.generate_linear_function()
                losses, grad_norms = nn.train((data, targets), epochs=5_000, shuffle=False)
                results_loss[type(opt).__name__].append(losses)
                results_gradnorm[type(opt).__name__].append(grad_norms)

        # Plot mean loss curves and min/max bands
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for opt_name, loss_curves in results_loss.items():
            loss_curves = np.array(loss_curves)
            mean_loss = np.mean(loss_curves, axis=0)
            min_loss = np.min(loss_curves, axis=0)
            max_loss = np.max(loss_curves, axis=0)

            plt.plot(mean_loss, label=f'{opt_name} Mean Loss')
            plt.fill_between(range(len(mean_loss)), min_loss, max_loss, alpha=0.2)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

        # plot all individual loss curves
        plt.figure(figsize=(12, 8))
        for opt_name, loss_curves in results_loss.items():
            # get next color from the current colormap
            color = plt.get_cmap('tab10')(list(results_loss.keys()).index(opt_name))
            for i, losses in enumerate(loss_curves):
                plt.plot(losses, color=color, alpha=0.3)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()

        # also plot gradient norms for all individual curves
        plt.figure(figsize=(12, 8))
        for opt_name, gradnorm_curves in results_gradnorm.items():
            # get next color from the current colormap
            color = plt.get_cmap('tab10')(list(results_gradnorm.keys()).index(opt_name))
            for i, grad_norms in enumerate(gradnorm_curves):
                plt.plot(grad_norms, color=color, alpha=0.3)
        plt.xlabel('Epochs')
        plt.ylabel('Gradient Norm')
        plt.grid()
        plt.show()

        # calculate statistics
        print()
        at_least_one_fail = False
        for opt_name, loss_curves in results_loss.items():
            fails = sum(1 for losses in loss_curves if losses[-1] > 0.1)
            if fails > 0:
                at_least_one_fail = True
            print(f"{opt_name}: {fails}/{num_tests} failed to converge (final loss > 0.1)")
            print(f"  Min final loss: {np.min([losses[-1] for losses in loss_curves]):.6f}")
            print(f"  Max final loss: {np.max([losses[-1] for losses in loss_curves]):.6f}")
            print(f"  Mean final loss: {np.mean([losses[-1] for losses in loss_curves]):.6f}")
            print(f"  Std final loss: {np.std([losses[-1] for losses in loss_curves]):.6f}")
            print()

        self.assertFalse(at_least_one_fail, "At least one optimizer failed to converge in one of the tests.")

    def test_compare_optimizers_on_circle(self):
        optimizers = [
            # optim.SGD(learning_rate=0.001),
            # optim.SGDMomentum(learning_rate=0.001, momentum=0.9),
            optim.Adam(learning_rate=0.001)
        ]

        results_loss = {}
        results_gradnorm = {}

        num_tests = 5
        n_points = 2000
        hidden_size = 64

        dataloader = CircleDataset(0.3, n_points, batch_size=64, shuffle=True, drop_last=True, normalize_data=True)

        for opt in optimizers:
            results_loss[type(opt).__name__] = []
            results_gradnorm[type(opt).__name__] = []
            for x in range(num_tests):
                nn = NN(
                    layer.FullyConnectedLayer(2, hidden_size, activation.ReLU()),
                    layer.FullyConnectedLayer(hidden_size, hidden_size, activation.ReLU()),
                    layer.FullyConnectedLayer(hidden_size, 1, activation.Linear()),
                    loss_function=loss.BinaryCrossEntropyLossFromLogits(),
                    optimizer=opt
                )

                losses, grad_norms = nn.train(dataloader, epochs=500, shuffle=True)
                results_loss[type(opt).__name__].append(losses)
                results_gradnorm[type(opt).__name__].append((losses, grad_norms))

        # Plot mean loss curves and min/max bands
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for opt_name, loss_curves in results_loss.items():
            loss_curves = np.array(loss_curves)
            mean_loss = np.mean(loss_curves, axis=0)
            min_loss = np.min(loss_curves, axis=0)
            max_loss = np.max(loss_curves, axis=0)

            plt.plot(mean_loss, label=f'{opt_name} Mean Loss')
            plt.fill_between(range(len(mean_loss)), min_loss, max_loss, alpha=0.2)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

        # plot all individual loss curves
        plt.figure(figsize=(12, 8))
        for opt_name, loss_curves in results_loss.items():
            # get next color from the current colormap
            color = plt.get_cmap('tab10')(list(results_loss.keys()).index(opt_name))
            for i, losses in enumerate(loss_curves):
                plt.plot(losses, color=color, alpha=0.3)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()

        # also plot gradient norms for all individual curves
        plt.figure(figsize=(12, 8))
        for opt_name, loss_gradcurves in results_gradnorm.items():
            # get next color from the current colormap
            color = plt.get_cmap('tab10')(list(results_gradnorm.keys()).index(opt_name))
            for i, (losses, grad_norms) in enumerate(loss_gradcurves):
                plt.plot(grad_norms, color=color, alpha=0.3)
        plt.xlabel('Epochs')
        plt.ylabel('Gradient Norm')
        plt.grid()
        plt.show()

        # calculate statistics
        print()
        at_least_one_fail = False
        for opt_name, loss_curves in results_loss.items():
            fails = sum(1 for losses in loss_curves if losses[-1] > 0.1)
            if fails > 0:
                at_least_one_fail = True
            print(f"{opt_name}: {fails}/{num_tests} failed to converge (final loss > 0.1)")
            print(f"  Min final loss: {np.min([losses[-1] for losses in loss_curves]):.6f}")
            print(f"  Max final loss: {np.max([losses[-1] for losses in loss_curves]):.6f}")
            print(f"  Mean final loss: {np.mean([losses[-1] for losses in loss_curves]):.6f}")
            print(f"  Std final loss: {np.std([losses[-1] for losses in loss_curves]):.6f}")
            print()

        self.assertFalse(at_least_one_fail, "At least one optimizer failed to converge in one of the tests.")
