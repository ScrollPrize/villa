from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler


@dataclass(frozen=True)
class ShuffleSampler:
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", int(self.seed))

    def build_loader(
        self,
        dataset,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn,
        shuffle: bool,
    ) -> DataLoader:
        generator = None
        if len(dataset) > 0 and bool(shuffle):
            generator = torch.Generator()
            generator.manual_seed(int(self.seed))
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False if len(dataset) <= 0 else bool(shuffle),
            drop_last=True,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=collate_fn,
            generator=generator,
        )


@dataclass(frozen=True)
class GroupBalancedSampler:
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", int(self.seed))

    def build_loader(
        self,
        dataset,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn,
        shuffle: bool,
    ) -> DataLoader:
        del shuffle
        if len(dataset) <= 0:
            return DataLoader(
                dataset,
                batch_size=int(batch_size),
                shuffle=False,
                drop_last=True,
                num_workers=int(num_workers),
                pin_memory=bool(pin_memory),
                collate_fn=collate_fn,
            )

        group_array = torch.as_tensor(dataset.sample_groups, dtype=torch.long)
        n_groups = int(group_array.max().item()) + 1
        group_counts = torch.bincount(group_array, minlength=n_groups).float()
        group_weights = len(dataset) / group_counts.clamp_min(1)
        sample_weights = group_weights[group_array]
        generator = torch.Generator()
        generator.manual_seed(int(self.seed))
        weighted_sampler = WeightedRandomSampler(
            sample_weights,
            len(dataset),
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            sampler=weighted_sampler,
            drop_last=True,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=collate_fn,
        )


@dataclass(frozen=True)
class GroupStratifiedSampler:
    batch_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "seed", int(self.seed))

    def build_loader(
        self,
        dataset,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        collate_fn,
        shuffle: bool,
    ) -> DataLoader:
        del batch_size, shuffle
        if len(dataset) <= 0:
            return DataLoader(
                dataset,
                batch_size=int(self.batch_size),
                shuffle=False,
                drop_last=True,
                num_workers=int(num_workers),
                pin_memory=bool(pin_memory),
                collate_fn=collate_fn,
            )
        group_indices = [int(group_idx) for group_idx in dataset.sample_groups]
        batch_size = int(self.batch_size)
        seed = int(self.seed)

        class _BatchSampler(Sampler[list[int]]):
            def __init__(self):
                self.batch_size = batch_size
                self.drop_last = True
                self._rng = random.Random(seed)

                indices_by_group: dict[int, list[int]] = {}
                for sample_idx, group_idx in enumerate(group_indices):
                    indices_by_group.setdefault(group_idx, []).append(int(sample_idx))

                self.groups = sorted(indices_by_group.keys())
                self.n_groups = len(self.groups)
                if self.batch_size < self.n_groups or self.batch_size % self.n_groups:
                    raise ValueError(
                        f"group_stratified batch_size={self.batch_size} must cover and divide n_groups={self.n_groups}"
                    )
                self.per_group = self.batch_size // self.n_groups
                self._indices_by_group = indices_by_group
                self._epoch_batches = len(group_indices) // self.batch_size

            def __len__(self) -> int:
                return int(self._epoch_batches)

            def __iter__(self) -> Iterator[list[int]]:
                order_by_group: dict[int, list[int]] = {}
                cursor_by_group: dict[int, int] = {}
                for group_idx in self.groups:
                    order = list(self._indices_by_group[group_idx])
                    self._rng.shuffle(order)
                    order_by_group[group_idx] = order
                    cursor_by_group[group_idx] = 0

                for _ in range(len(self)):
                    batch: list[int] = []
                    for group_idx in self.groups:
                        order = order_by_group[group_idx]
                        cursor = cursor_by_group[group_idx]
                        for _ in range(self.per_group):
                            if cursor >= len(order):
                                order = list(self._indices_by_group[group_idx])
                                self._rng.shuffle(order)
                                order_by_group[group_idx] = order
                                cursor = 0
                            batch.append(order[cursor])
                            cursor += 1
                        cursor_by_group[group_idx] = cursor
                    self._rng.shuffle(batch)
                    yield batch

        return DataLoader(
            dataset,
            batch_sampler=_BatchSampler(),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=collate_fn,
        )


__all__ = [
    "GroupBalancedSampler",
    "GroupStratifiedSampler",
    "ShuffleSampler",
]
