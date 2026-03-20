from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def _close_dataloader(dataloader: DataLoader) -> None:
    iterator = getattr(dataloader, "_iterator", None)
    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown_workers):
        shutdown_workers()
    dataset = getattr(dataloader, "dataset", None)
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


@dataclass
class _LoaderState:
    dataloader: DataLoader
    epoch: int = 0
    iterator: Iterator[Any] | None = None


class _WeightedCombinedLoaderIterator:
    def __init__(self, loader: "WeightedCombinedLoader") -> None:
        self.loader = loader
        self.rng = random.Random(loader.seed)
        self.states = [_LoaderState(dataloader=dataloader) for dataloader in loader.dataloaders]
        for state in self.states:
            _set_dataloader_epoch(state.dataloader, state.epoch)
            state.iterator = iter(state.dataloader)

    def __iter__(self) -> "_WeightedCombinedLoaderIterator":
        return self

    def __next__(self) -> Any:
        index = self.rng.choices(range(len(self.states)), weights=self.loader.weights, k=1)[0]
        state = self.states[index]
        while True:
            try:
                assert state.iterator is not None
                return next(state.iterator)
            except StopIteration:
                state.epoch += 1
                _set_dataloader_epoch(state.dataloader, state.epoch)
                state.iterator = iter(state.dataloader)


class WeightedCombinedLoader:
    """Sample batches from multiple dataloaders using fixed weights."""

    def __init__(
        self,
        dataloaders: Sequence[DataLoader],
        *,
        weights: Sequence[float],
        seed: int = 0,
    ) -> None:
        if not dataloaders:
            raise ValueError("WeightedCombinedLoader requires at least one dataloader.")
        if len(dataloaders) != len(weights):
            raise ValueError("weights must match the number of dataloaders.")
        normalized_weights = tuple(float(weight) for weight in weights)
        if any(weight <= 0.0 for weight in normalized_weights):
            raise ValueError(f"weights must be positive, got {normalized_weights}")
        self.dataloaders = tuple(dataloaders)
        self.weights = normalized_weights
        self.seed = int(seed)

    def __iter__(self) -> _WeightedCombinedLoaderIterator:
        return _WeightedCombinedLoaderIterator(self)

    def set_epoch(self, epoch: int) -> None:
        for dataloader in self.dataloaders:
            _set_dataloader_epoch(dataloader, epoch)

    def close(self) -> None:
        for dataloader in self.dataloaders:
            _close_dataloader(dataloader)
