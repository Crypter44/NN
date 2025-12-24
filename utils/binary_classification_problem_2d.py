import numpy as np
import tqdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from dataloader.dataloader import Dataloader
from model.network import NN


class BinaryClassification2DProblem:
    def __init__(self, train_set: Dataloader, test_set: Dataloader | None, model: NN):
        self.train_set = train_set
        self.test_set = test_set

        self.model = model

    def run_training_and_animate(self, epochs: int, interval: int, include_test_set: bool = False, length_of_frame: int = 750, cmap: str = 'managua'):
        predictions_train = {}
        if include_test_set:
            if self.test_set is not None:
                predictions_test = {}
            else:
                raise ValueError("Test set is None, cannot include test set in animation.")


        resolution = 1000
        heatmap_grid = np.array(
            [[x, y] for y in np.linspace(-1, 1, resolution) for x in np.linspace(-1, 1, resolution)]
        )

        losses, grad_norms = [], []

        print("Starting training...")

        predictions_train[0] = self.model(heatmap_grid).reshape(resolution, resolution)
        if include_test_set:
            predictions_test[0] = self.model(heatmap_grid).reshape(resolution, resolution)

        pbar = tqdm.tqdm(range(1, epochs + 1))
        for i in pbar:
            l, g = self.model.train(
                data=self.train_set,
                epochs=1,
            )
            losses.append(l[-1])
            grad_norms.append(g[-1])

            if i % interval == 0:
                predictions_train[i] = self.model(heatmap_grid).reshape(resolution, resolution)
                if include_test_set:
                    predictions_test[i] = self.model(heatmap_grid).reshape(resolution, resolution)

        print("Finished training.")
        print("Generating decision boundary animation...")
        print("Generating train set animation...")

        self.plot_decision_boundary(predictions_train, train=True, length_of_frame=length_of_frame, cmap=cmap)
        if include_test_set:
            print("Generating test set animation...")
            self.plot_decision_boundary(predictions_test, train=False, length_of_frame=length_of_frame, cmap=cmap)

        print("Finished animations.")

        return losses, grad_norms

    def plot_decision_boundary(self, predictions_train: dict, train=False, length_of_frame= 750, cmap='managua'):
        fig, ax = plt.subplots()

        num_frames = len(predictions_train)
        pbar = tqdm.tqdm(range(num_frames))

        def update(frame):
            pbar.update(1)
            ax.clear()

            probs = list(predictions_train.values())[frame]

            ax.imshow(
                probs,
                extent=(-1, 1, -1, 1),
                origin='lower',
                cmap=cmap,
                alpha=1,
                vmin=0,
                vmax=1
            )

            ax.scatter(
                self.train_set.data[:, 0],
                self.train_set.data[:, 1],
                c=self.train_set.targets,
                cmap=cmap,
                edgecolor='k',
                linewidth=0.1,
                s=3,
                alpha=1
            )

            title = "Train Set" if train else "Test Set"
            ax.set_title(f"{title}: Epoch {list(predictions_train.keys())[frame]}")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.grid(False)
            ax.set_aspect('equal')
            return ax,

        ani = FuncAnimation(fig, update, frames=num_frames, blit=False, repeat=False, interval=length_of_frame)
        pbar.close()
        ani.save('decision_boundary' + ('_train' if train else '_test') + '.mp4', dpi=150)



