import numpy as np

from dataloader.dataloader import Dataloader


class CircleDataset(Dataloader):
    """
    Dataset of points in a 2D unit square, labeled based on whether they are inside or outside a circle of given radius.
    This is a binary classification problem that can be used to test the capabilities of neural networks to learn non-linear decision boundaries.
    The circle is centered at (0.5, 0.5) in the unit square [0, 1] x [0, 1].
    Points inside the circle are labeled as 0, points outside the circle are labeled as 1.
    """
    def __init__(self, radius, num_points, batch_size=64, shuffle=True, drop_last=True, normalize_data=True):
        self.radius = radius
        self.num_points = num_points

        data, targets = self.generate_circle_data(radius, num_points)
        super().__init__(data, targets, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, normalize_data=normalize_data)

    @staticmethod
    def generate_circle_data(radius, n_points=200):
        """
        Generates data points and targets for the circle dataset.
        :param radius: radius of the circle
        :param n_points: number of data points to generate
        :return: data: numpy array of shape (n_points, 2), targets: numpy array of shape (n_points, 1)
        """
        data = np.random.rand(n_points, 2)
        center = np.array([0.5, 0.5])
        distances = np.linalg.norm(data - center, axis=1)
        targets = (distances > radius).astype(np.float32).reshape(-1, 1)
        return data, targets

