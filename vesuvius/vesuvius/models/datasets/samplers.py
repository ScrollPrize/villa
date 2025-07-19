import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler
import itertools

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
        self.primary_indices = list(primary_indices)
        self.secondary_indices = list(secondary_indices)
        self.batch_size = batch_size
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert self.primary_batch_size > 0, "Labeled batch size must be positive"

    def __iter__(self):
        # Shuffle primary indices
        primary_indices_shuffled = np.random.permutation(self.primary_indices)
        
        # Create batches
        for i in range(0, len(primary_indices_shuffled), self.primary_batch_size):
            # Get primary batch
            primary_batch = primary_indices_shuffled[i:i + self.primary_batch_size].tolist()
            
            # If we don't have enough primary samples, skip this batch
            if len(primary_batch) < self.primary_batch_size:
                break
            
            # Get secondary batch (sample with replacement if needed)
            if len(self.secondary_indices) >= self.secondary_batch_size:
                secondary_batch = np.random.choice(
                    self.secondary_indices, 
                    size=self.secondary_batch_size, 
                    replace=False
                ).tolist()
            else:
                # If we have fewer secondary indices than needed, sample with replacement
                secondary_batch = np.random.choice(
                    self.secondary_indices, 
                    size=self.secondary_batch_size, 
                    replace=True
                ).tolist()
            
            # Yield combined batch
            yield primary_batch + secondary_batch

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    """Helper function for single iteration"""
    return iter(np.random.permutation(iterable))


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