import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler


class TwoStreamBatchSampler(Sampler):
    """Samples labeled and unlabeled data in fixed proportions"""

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        """
        Parameters
        ----------
        primary_indices : list
            Indices for labeled data
        secondary_indices : list
            Indices for unlabeled data
        batch_size : int
            Total batch size
        secondary_batch_size : int
            Number of unlabeled samples per batch
        """
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.batch_size = batch_size
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert self.primary_batch_size > 0, "Labeled batch size must be positive"

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)

        return (
            self.primary_indices[
            next(primary_iter) * self.primary_batch_size:(next(primary_iter) + 1) * self.primary_batch_size]
            + self.secondary_indices[
              next(secondary_iter) * self.secondary_batch_size:(next(secondary_iter) + 1) * self.secondary_batch_size]
            for (primary_iter, secondary_iter) in zip(
            grouper(iterate_once(self.primary_indices), self.primary_batch_size),
            grouper(iterate_eternally(self.secondary_indices), self.secondary_batch_size)
        )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    """Helper function for single iteration"""
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    """Helper function for infinite iteration"""

    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """Collect data into fixed-length chunks"""
    args = [iter(iterable)] * n
    return zip(*args)