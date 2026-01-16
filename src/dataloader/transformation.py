import numpy as np


class Transformation:
    def apply(self, data):
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, data):
        return self.apply(data)


class ChainTransformation(Transformation):
    def __init__(self, *transformations):
        self.transformations = transformations

    def apply(self, data):
        for transform in self.transformations:
            data = transform.apply(data)
        return data


class Normalize(Transformation):
    def __init__(self, mean=None, std=None, axis=(0,)):
        self.mean = mean
        self.std = std
        self.axis = axis

    def apply(self, data):
        if self.mean is None or self.std is None:
            self.fit(data)
        return (data - self.mean) / self.std

    def fit(self, data):
        self.mean = np.mean(data, axis=self.axis, keepdims=True)
        self.std = np.std(data, axis=self.axis, keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

class RandomTranslationWithPadding(Transformation):
    def __init__(self, dims_after_transform):
        self.dims_after_transform = dims_after_transform

    def apply(self, data):
        new_shape = (data.shape[0],) + self.dims_after_transform
        empty = np.zeros(new_shape)
        data_shape = data.shape
        #max_translations = [self.dims_after_transform[i-1] - data_shape[i] for i in range(1, len(data_shape))]
        max_translations = [self.dims_after_transform[i] - data_shape[i + 1] for i in range(len(self.dims_after_transform))]
        for j in range(data_shape[0]):
            translations = [np.random.randint(0, max_translations[i-1] + 1) for i in range(1, len(data_shape))]
            slices = tuple(slice(translations[i-1], translations[i-1] + data_shape[i]) for i in range(1, len(data_shape)))
            slices = (j,) + slices
            empty[slices] = data[j]
        return empty


class Flatten(Transformation):
    def apply(self, data):
        return data.reshape(data.shape[0], -1)
