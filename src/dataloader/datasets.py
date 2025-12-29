from pathlib import Path

import numpy as np
from matplotlib.path import Path as MatplotlibPath
from torchvision import datasets

from src.dataloader.dataloader import Dataloader


class CircleDataset(Dataloader):
    """
    Dataset of points in a 2D unit square, labeled based on whether they are inside or outside a circle of given radius.
    This is a binary classification problem that can be used to test the capabilities of neural networks to learn non-linear decision boundaries.
    The circle is centered at (0.5, 0.5) in the unit square [0, 1] x [0, 1].
    Points inside the circle are labeled as 0, points outside the circle are labeled as 1.
    """

    def __init__(self, radius, num_points, batch_size=64, shuffle=True, drop_last=True, transformation=None):
        self.radius = radius
        self.num_points = num_points

        data, targets = self.generate_circle_data(radius, num_points)
        super().__init__(data, targets, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, transformation=transformation)

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


class PolygonDataset(Dataloader):
    """
    Dataset of points in a 2D square, labeled based on whether they are inside or outside a polygon of given number of sides.
    This is a binary classification problem that can be used to test the capabilities of neural networks to learn non-linear decision boundaries.
    The polygon is centered at (0, 0) in the square [-1, 1] x [-1, 1].
    Points inside the polygon are labeled as 0, points outside the polygon are labeled as 1.
    """

    def __init__(self, num_points, polygon_vertices=None, num_sides=8, batch_size=64, shuffle=True, drop_last=True, transformation=None):
        self.num_sides = num_sides
        self.num_points = num_points

        data, targets, polygon = self.generate_polygon_data(polygon_vertices, num_sides, num_points)
        self.polygon_vertices = polygon
        super().__init__(data, targets, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, transformation=transformation)

    @staticmethod
    def generate_polygon_data(polygon_vertices=None, num_sides=8, n_points=200):
        """
        Generates data points and targets for the polygon dataset.
        :param num_sides: number of sides of the regular polygon
        :param n_points: number of data points to generate
        :return: data: numpy array of shape (n_points, 2), targets: numpy array of shape (n_points, 1)
        """
        if polygon_vertices is None:
            rng = np.random.default_rng()
            angle = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)

            # random radial variation per vertex (adjust min/max to control irregularity)
            radial_variation = rng.uniform(0, 2, size=num_sides)
            base_radius = 0.5
            radii = base_radius * radial_variation

            # small angular jitter and optional random rotation
            angle += rng.uniform(-0.12, 0.12, size=num_sides)
            angle += rng.uniform(0, 2 * np.pi)  # random global rotation

            # ensure vertices are in angle order to avoid self-intersections
            order = np.argsort(angle)
            angle = angle[order]
            radii = radii[order]

            polygon_vertices = np.column_stack((radii * np.cos(angle), radii * np.sin(angle)))
        polygon_path = MatplotlibPath(polygon_vertices)

        data = np.random.rand(n_points, 2) * 2 - 1  # Points in [-1, 1] x [-1, 1]
        inside_polygon = polygon_path.contains_points(data)
        targets = (~inside_polygon).astype(np.float32).reshape(-1, 1)

        return data, targets, polygon_vertices


class SpiralDataset(Dataloader):
    """
    Dataset of points in a 2D square, labeled based on whether they are inside or outside a spiral shape.
    This is a binary classification problem that can be used to test the capabilities of neural networks to learn non-linear decision boundaries.
    The spiral is centered at (0, 0) in the square [-1, 1] x [-1, 1].
    Points inside the spiral are labeled as 0, points outside the spiral are labeled as 1.
    """

    def __init__(self, num_points, turns=3, noise=0.1, batch_size=64, shuffle=True, drop_last=True, transformation=None):
        self.turns = turns
        self.num_points = num_points
        self.noise = noise

        data, targets = self.generate_spiral_data(turns, num_points)
        super().__init__(data, targets, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, transformation=transformation)

    @staticmethod
    def generate_spiral_data(turns=3, n_points=200, scale=0.9, width=0.1):
        """
        Generates points in [-1, 1] x [-1, 1] and labels based on an Archimedean spiral.
        :param turns: number of spiral turns
        :param n_points: number of points
        :param scale: max spiral radius (0 < scale <= 1)
        :param width: thickness of the spiral (distance tolerance)
        """
        data = np.random.rand(n_points, 2) * 2 - 1
        targets = np.ones((n_points, 1), dtype=np.float32)  # 1 = outside, 0 = on/inside spiral

        max_theta = 2 * np.pi * turns
        c = scale / max_theta  # r = c * theta_total

        for i in range(n_points):
            x, y = data[i]
            r = np.hypot(x, y)
            theta = np.arctan2(y, x)
            if theta < 0:
                theta += 2 * np.pi  # normalize to [0, 2*pi)

            # check all equivalent angles for each turn: theta + 2*pi*k
            inside = False
            for k in range(turns):
                theta_total = theta + 2 * np.pi * k
                r_spiral = c * theta_total
                if r_spiral > scale:
                    break
                if abs(r - r_spiral) <= 0.5 * width:
                    inside = True
                    break

            if inside:
                targets[i] = 0.0

        return data, targets


class MNISTDataset(Dataloader):
    """
    MNIST handwritten digit dataset wrapped into the custom Dataloader class.
    Loads the train or test split, flattens to vectors of size 784, and normalizes to [0, 1].
    """

    def __init__(self, train=True, root=None, batch_size=64, shuffle=True, drop_last=True, transformation=None):
        if root is None:
            BASE_DIR = Path(__file__).parent
            DATA_DIR = BASE_DIR / "downloaded_data"
            root = DATA_DIR.as_posix()

        self.train = train

        data, targets = self.load_mnist(root, train=train)
        super().__init__(data, targets, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, transformation=transformation)

    @staticmethod
    def load_mnist(root, train=True):
        """
        Download (if needed) and load the MNIST dataset into numpy arrays.
        :return:
            data: shape (N, 784), float32
            targets: shape (N, 1), float32
        """
        dataset = datasets.MNIST(root=root, train=train, download=True)

        # Tensor -> numpy
        data = dataset.data.numpy().astype(np.float32)
        targets = dataset.targets.numpy().astype(np.int64)

        # Make targets one-hot encoded
        targets_one_hot = np.zeros((len(targets), 10), dtype=np.float32)
        targets_one_hot[np.arange(len(targets)), targets] = 1

        return data, targets_one_hot
