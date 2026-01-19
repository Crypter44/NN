def plot_loss_curve(loss_list, eval_loss_list=None):
    """
    Plots the loss curve over training iterations.

    :param loss_list: List of loss values recorded during training.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Loss', color='blue')
    if eval_loss_list is not None:
        plt.plot(eval_loss_list, label='Eval Loss', color='orange')
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

def outdated(obj):
    if isinstance(obj, type):  # class
        class Wrapper(obj):
            def __init__(self, *args, **kwargs):
                raise RuntimeError(f"{obj.__name__} is outdated and must not be used.")
        return Wrapper
    else:  # function
        def wrapper(*args, **kwargs):
            raise RuntimeError(f"{obj.__name__} is outdated and must not be used.")
        return wrapper