from __future__ import annotations

import itertools
import random

from torch.utils.data import Sampler


class TwoStreamBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        primary_indices,
        secondary_indices,
        batch_size,
        secondary_batch_size,
        *,
        seed=0,
    ):
        self.primary_indices = [int(idx) for idx in primary_indices]
        self.secondary_indices = [int(idx) for idx in secondary_indices]
        self.secondary_batch_size = int(secondary_batch_size)
        self.primary_batch_size = int(batch_size) - self.secondary_batch_size
        self.seed = int(seed)
        self._epoch = 0

        if not (len(self.primary_indices) >= self.primary_batch_size > 0):
            raise ValueError(
                f"Not enough labeled samples: {len(self.primary_indices)} < {self.primary_batch_size}"
            )
        if not (len(self.secondary_indices) >= self.secondary_batch_size > 0):
            raise ValueError(
                f"Not enough unlabeled samples: {len(self.secondary_indices)} < {self.secondary_batch_size}"
            )

    def __len__(self):
        return (len(self.primary_indices) + self.primary_batch_size - 1) // self.primary_batch_size

    def __iter__(self):
        epoch_seed = self.seed + self._epoch
        self._epoch += 1
        primary_iter = _iterate_once(self.primary_indices, seed=epoch_seed)
        secondary_iter = _iterate_eternally(self.secondary_indices, seed=epoch_seed + 1)
        pad_rng = random.Random(epoch_seed + 2)

        for primary_batch in _grouper(primary_iter, self.primary_batch_size):
            if len(primary_batch) < self.primary_batch_size:
                primary_batch = primary_batch + tuple(
                    pad_rng.choice(self.primary_indices)
                    for _ in range(self.primary_batch_size - len(primary_batch))
                )
            secondary_batch = tuple(itertools.islice(secondary_iter, self.secondary_batch_size))
            yield list(primary_batch) + list(secondary_batch)


def _iterate_once(indices, *, seed):
    shuffled = list(indices)
    random.Random(int(seed)).shuffle(shuffled)
    return iter(shuffled)


def _iterate_eternally(indices, *, seed):
    epoch = 0
    while True:
        shuffled = list(indices)
        random.Random(int(seed) + epoch).shuffle(shuffled)
        epoch += 1
        for idx in shuffled:
            yield idx


def _grouper(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, int(n)))
        if not chunk:
            return
        yield chunk


__all__ = ["TwoStreamBatchSampler"]
