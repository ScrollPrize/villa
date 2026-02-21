import math
import random
from typing import Dict, Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class GroupStratifiedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        group_indices: Sequence[int],
        *,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = True,
    ):
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.drop_last = bool(drop_last)
        self._rng = random.Random(int(seed))

        group_indices_list = [int(x) for x in group_indices]
        if not group_indices_list:
            raise ValueError("group_indices is empty")

        indices_by_group: Dict[int, List[int]] = {}
        for idx, g in enumerate(group_indices_list):
            indices_by_group.setdefault(int(g), []).append(int(idx))

        self.groups = sorted(indices_by_group.keys())
        self.n_groups = len(self.groups)
        if self.batch_size < self.n_groups:
            raise ValueError(
                f"batch_size={self.batch_size} is smaller than n_groups={self.n_groups}. "
                "Group-stratified batches require batch_size >= n_groups."
            )
        if self.batch_size % self.n_groups != 0:
            raise ValueError(
                f"batch_size={self.batch_size} must be divisible by n_groups={self.n_groups} "
                "for group-stratified batches."
            )
        self.per_group = self.batch_size // self.n_groups

        empty_groups = [g for g, idxs in indices_by_group.items() if len(idxs) == 0]
        if empty_groups:
            raise ValueError(f"Some groups have 0 samples: {sorted(empty_groups)}")

        self._indices_by_group = indices_by_group
        self._epoch_batches = (
            len(group_indices_list) // self.batch_size
            if self.drop_last
            else int(math.ceil(len(group_indices_list) / self.batch_size))
        )

    def __len__(self) -> int:
        return int(self._epoch_batches)

    def __iter__(self) -> Iterator[List[int]]:
        order_by_group: Dict[int, List[int]] = {}
        ptr_by_group: Dict[int, int] = {}

        for g in self.groups:
            order = list(self._indices_by_group[g])
            self._rng.shuffle(order)
            order_by_group[g] = order
            ptr_by_group[g] = 0

        for _ in range(len(self)):
            batch: List[int] = []
            for g in self.groups:
                order = order_by_group[g]
                ptr = ptr_by_group[g]
                for _ in range(self.per_group):
                    if ptr >= len(order):
                        order = list(self._indices_by_group[g])
                        self._rng.shuffle(order)
                        order_by_group[g] = order
                        ptr = 0
                    batch.append(order[ptr])
                    ptr += 1
                ptr_by_group[g] = ptr

            # mix groups within the batch
            self._rng.shuffle(batch)
            yield batch

