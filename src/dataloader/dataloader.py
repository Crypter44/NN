from copy import deepcopy

import numpy as np

from src.dataloader.transformation import Transformation


class Dataloader:
    def __init__(
            self,
            data,
            targets,
            batch_size=8,
            shuffle=True,
            drop_last=False,
            transformation: Transformation | None = None
    ):
        """
        Initializes the dataloader with data.

        :param data: data in form of numpy array of shape (B, N, D),
        where B is batch size, N is number of samples, D is number of features
        :param targets: targets in form of numpy array of shape (B, M),
        where B is batch size, M is number of target features
        :param batch_size: size of each batch
        :param shuffle: whether to shuffle the data at the start of each epoch
        :param drop_last: whether to drop the last batch if it's smaller than batch_size
        """
        self.raw_data = deepcopy(data)
        self.data = data
        self.targets = targets

        self.transformation = transformation
        if self.transformation:
            self.data = self.transformation(self.data)

        self.index = 0
        self.indices = list(range(len(self.data)))
        self.batch_size = batch_size if batch_size > 0 else len(self.data)
        self.shuffle = shuffle
        self.drop_last = drop_last

        if self.shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        """
        Shuffles the data indices, so that data is accessed in random order.
        :return: None
        """
        self.indices = np.random.permutation(self.indices)

    def __iter__(self):
        """
        Initializes the iterator.
        :return: self
        """
        self.index = 0
        if self.shuffle:
            self.shuffle_data()
        return self

    def __next__(self):
        """
        Returns the next batch of data.

        This method contains the logic of the dataloader.
        Based on the current index and batch size, it selects the appropriate
        indices from the shuffled indices list, retrieves the corresponding data
        and targets, and returns them as a batch. If the end of the data is reached,
        it raises StopIteration to signal the end of the epoch.
        :return: batch of data and targets
        """
        if self.index >= len(self.indices):
            raise StopIteration
        if self.index + self.batch_size > len(self.indices):
            if self.drop_last:
                raise StopIteration
            else:
                batch_indices = self.indices[self.index:]
        else:
            batch_indices = self.indices[self.index:self.index + self.batch_size]
        batch = self.data[batch_indices]
        targets = self.targets[batch_indices]
        batch = (batch, targets)
        self.index += self.batch_size
        return batch

    def __getitem__(self, item):
        """
        Gets the data and targets at the given index or slice.
        :param item: index or slice
        :return: data and targets at the given index or slice
        """
        batch = self.data[item]
        targets = self.targets[item]
        return batch, targets

    @staticmethod
    def train_and_eval_split(data, targets, train_fraction=0.8, **dataloader_kwargs):
        """
        Splits the data into training and evaluation sets, and returns two Dataloaders.
        :param data: data in form of numpy array
        :param targets: targets in form of numpy array
        :param train_fraction: fraction of data to use for training
        :param dataloader_kwargs: additional arguments to pass to the Dataloader constructor
        :return: train_dataloader, eval_dataloader
        """
        num_train = int(len(data) * train_fraction)
        train_data = data[:num_train]
        train_targets = targets[:num_train]
        eval_data = data[num_train:]
        eval_targets = targets[num_train:]

        train_dataloader = Dataloader(train_data, train_targets, **dataloader_kwargs)
        eval_dataloader = Dataloader(eval_data, eval_targets, **dataloader_kwargs)

        return train_dataloader, eval_dataloader

    def print(self, include_data=False):
        """
        Prints the dataloader information.
        :param include_data: whether to include the data in the printout
        :return: None
        """
        print(f"Dataloader:")
        print(f"  Number of samples: {len(self.data)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Shuffle: {self.shuffle}")
        print(f"  Drop last: {self.drop_last}")
        print(f"  Data shape: {self.data.shape}")
        print(f"  Targets shape: {self.targets.shape}")
        if include_data:
            print(f"  Data: \n{self.data}")
            print(f"  Targets: \n{self.targets}")
