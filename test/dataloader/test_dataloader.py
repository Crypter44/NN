from _pytest import unittest
import numpy as np

from src.dataloader.dataloader import Dataloader


class TestDataLoader(unittest.TestCase):
    def test_dataloader_init(self):

        data = np.array(list(range(10)))
        targets = np.array(list(range(10, 20)))

        dataloader = Dataloader(data, targets, batch_size=1, shuffle=False, drop_last=False, normalize_data=False)

    def test_dataloader_iteration(self):

        data = np.array(list(range(10)))
        targets = np.array(list(range(10, 20)))

        dataloader = Dataloader(data, targets, batch_size=1, shuffle=False, drop_last=False, normalize_data=False)

        for i, batch in enumerate(dataloader):
            data, targets = batch
            assert data.shape == (1,)
            assert targets.shape == (1,)
            assert data[0] == i
            assert targets[0] == i + 10

    def test_dataloader_iteration_with_batch_size(self):

        data = np.array(list(range(10)))
        targets = np.array(list(range(10, 20)))

        dataloader = Dataloader(data, targets, batch_size=3, shuffle=False, drop_last=False, normalize_data=False)

        expected_batches = [
            (np.array([0, 1, 2]), np.array([10, 11, 12])),
            (np.array([3, 4, 5]), np.array([13, 14, 15])),
            (np.array([6, 7, 8]), np.array([16, 17, 18])),
            (np.array([9]), np.array([19])),
        ]

        for i, batch in enumerate(dataloader):
            data, targets = batch
            expected_data, expected_targets = expected_batches[i]
            assert np.array_equal(data, expected_data)
            assert np.array_equal(targets, expected_targets)

    def test_dataloader_iteration_with_drop_last(self):

        data = np.array(list(range(10)))
        targets = np.array(list(range(10, 20)))

        dataloader = Dataloader(data, targets, batch_size=3, shuffle=False, drop_last=True, normalize_data=False)

        expected_batches = [
            (np.array([0, 1, 2]), np.array([10, 11, 12])),
            (np.array([3, 4, 5]), np.array([13, 14, 15])),
            (np.array([6, 7, 8]), np.array([16, 17, 18])),
        ]

        for i, batch in enumerate(dataloader):
            data, targets = batch
            expected_data, expected_targets = expected_batches[i]
            assert np.array_equal(data, expected_data)
            assert np.array_equal(targets, expected_targets)

    def test_dataloader_normalization(self):

        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        targets = np.array([[10.0], [11.0], [12.0], [13.0], [14.0]])

        dataloader = Dataloader(data, targets, batch_size=1, shuffle=False, drop_last=False, normalize_data=True)

        mean = np.mean(data)
        std = np.std(data)

        for i, batch in enumerate(dataloader):
            data_batch, targets_batch = batch
            expected_data = (data[i] - mean) / std
            assert np.allclose(data_batch, expected_data, atol=1e-6)
            assert np.array_equal(targets_batch.flatten(), targets[i]), \
                f"Targets do not match for batch {i} expected {targets[i]}, got {targets_batch}"

    def test_dataloader_shuffle(self):

        data = np.array(list(range(100)))
        targets = np.array(list(range(100, 200)))

        dataloader = Dataloader(data, targets, batch_size=10, shuffle=True, drop_last=False, normalize_data=False)

        all_data = set()
        all_targets = set()
        print("\nShuffled Batches:")
        for batch in dataloader:
            data_batch, targets_batch = batch
            for d in data_batch:
                all_data.add(d)
            for t in targets_batch:
                all_targets.add(t)

            assert np.array_equal(targets_batch, data_batch+100), "Targets do not match data + 100"
            print(f"    Batch data:     {data_batch},\n"
                  f"    Batch targets:  {targets_batch}")

        assert all_data == set(range(100)), "Not all data samples were seen during iteration"
        assert all_targets == set(range(100, 200)), "Not all target samples were seen during iteration"
