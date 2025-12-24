import matplotlib.pyplot as plt
import numpy as np


def plot_images_with_colored_labels(images, labels, ground_truth=None):
    """
    Plots a grid of images with their labels colored based on correctness.

    :param images: A numpy array of shape (N, H, W) or (N, H, W, C) containing the images to plot.
    :param labels: A numpy array of shape (N,) containing the predicted labels for the images.
    :param ground_truth: (Optional) A numpy array of shape (N,) containing the ground truth labels.
                         If provided, labels will be colored green for correct predictions and red for incorrect ones.
                         If not provided, all labels will be colored black.
    """

    num_images = images.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_images)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(grid_size * grid_size):
        ax = axes[i]
        ax.axis('off')

        if i < num_images:
            img = images[i]
            label = labels[i]

            if ground_truth is not None:
                gt_label = ground_truth[i]
                color = 'green' if label == gt_label else 'red'
            else:
                color = 'black'

            if img.ndim == 2:  # Grayscale image
                ax.imshow(img, cmap='gray')
            else:  # Color image
                ax.imshow(img)

            ax.set_title(f'Label: {label}', color=color)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    plt.show()
