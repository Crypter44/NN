def plot_loss_curve(loss_list):
    """
    Plots the loss curve over training iterations.

    :param loss_list: List of loss values recorded during training.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')  # Use logarithmic scale for better visibility
    plt.legend()
    plt.grid(True)
    plt.show()

def set_seed(seed):
    """
    Sets the random seed for reproducibility.

    :param seed: The seed value to set.
    """
    import numpy as np
    import random

    np.random.seed(seed)
    random.seed(seed)