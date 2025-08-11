"""
Two-stream batch sampler for semi-supervised learning.

This sampler creates batches containing both labeled and unlabeled data,
following the SSL4MIS implementation pattern.
"""

import itertools
import numpy as np
from torch.utils.data import Sampler


class TwoStreamBatchSampler(Sampler):
    """
    Iterate two sets of indices
    
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    
    This matches the SSL4MIS implementation to ensure consistent
    data sampling behavior for uncertainty-aware mean teacher training.
    """
    
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        """
        Parameters
        ----------
        primary_indices : list
            Indices of labeled samples (will be iterated once per epoch)
        secondary_indices : list
            Indices of unlabeled samples (will be cycled infinitely)
        batch_size : int
            Total batch size
        secondary_batch_size : int
            Number of unlabeled samples per batch
        """
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        
        assert len(self.primary_indices) >= self.primary_batch_size > 0, \
            f"Not enough labeled samples: {len(self.primary_indices)} < {self.primary_batch_size}"
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0, \
            f"Not enough unlabeled samples: {len(self.secondary_indices)} < {self.secondary_batch_size}"
    
    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )
    
    def __len__(self):
        """Number of batches per epoch (based on labeled data)"""
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    """Create a random permutation of the iterable"""
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    """Infinitely cycle through random permutations of indices"""
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3) --> ABC DEF
    args = [iter(iterable)] * n
    return zip(*args)