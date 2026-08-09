from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from collections import Counter, defaultdict
from typing import Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler


def _stable_child_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class _RecyclingShuffledQueue:
    """Shuffle once, consume without replacement, then reshuffle on recycle."""

    def __init__(self, values: Sequence, *, seed: int):
        if not values:
            raise ValueError("recycling queue requires at least one value")
        self._source = list(values)
        self._rng = random.Random(int(seed))
        self._order: list = []
        self._cursor = 0
        self._recycles = -1
        self._recycle()

    @property
    def recycles(self) -> int:
        return max(0, int(self._recycles))

    def _recycle(self) -> None:
        self._order = list(self._source)
        self._rng.shuffle(self._order)
        self._cursor = 0
        self._recycles += 1

    def pop(self):
        if self._cursor >= len(self._order):
            self._recycle()
        value = self._order[self._cursor]
        self._cursor += 1
        return value


class FixedScrollPriorStratifiedBatchSampler(Sampler[list[int]]):
    """Draw exact scroll quotas and balance physical segments hierarchically.

    Each scroll owns a shuffled physical-segment queue. Each physical segment
    owns a shuffled representation queue, and each representation owns a
    shuffled patch-index queue. Every queue is consumed without replacement
    until exhausted and is reshuffled only when it recycles. This gives every
    physical segment equal long-run mass within its scroll and divides the
    segment mass among duplicate representations instead of double-counting
    them.

    The mapping is deliberately config-driven: every dataset must map every
    representation ``segment_relpath`` to explicit physical-segment and
    representation keys. No naming or shared-volume heuristic is applied at
    runtime.
    """

    def __init__(
        self,
        patches: Sequence,
        datasets: Sequence[dict],
        *,
        batch_quotas: Mapping[str, int],
        batch_size: int,
        seed: int,
    ) -> None:
        self.batch_size = int(batch_size)
        self.drop_last = True
        self.seed = int(seed)
        self.batch_quotas = {
            str(scroll): int(quota) for scroll, quota in batch_quotas.items()
        }
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.batch_quotas or any(quota <= 0 for quota in self.batch_quotas.values()):
            raise ValueError("fixed scroll quotas must all be positive")
        if sum(self.batch_quotas.values()) != self.batch_size:
            raise ValueError(
                f"fixed scroll quotas sum to {sum(self.batch_quotas.values())}, "
                f"expected batch_size={self.batch_size}"
            )
        if not patches:
            raise ValueError("fixed-prior sampling requires at least one patch")

        dataset_contracts: dict[int, dict] = {}
        for dataset_idx, dataset in enumerate(datasets):
            scroll = str(dataset.get("sampling_scroll", "")).strip()
            physical_keys = dataset.get("sampling_physical_segment_keys")
            representation_keys = dataset.get("sampling_representation_keys")
            if not scroll:
                raise ValueError(
                    f"datasets[{dataset_idx}] must define non-empty sampling_scroll"
                )
            if not isinstance(physical_keys, dict) or not physical_keys:
                raise ValueError(
                    f"datasets[{dataset_idx}] must define sampling_physical_segment_keys"
                )
            if not isinstance(representation_keys, dict) or not representation_keys:
                raise ValueError(
                    f"datasets[{dataset_idx}] must define sampling_representation_keys"
                )
            dataset_contracts[dataset_idx] = {
                "scroll": scroll,
                "physical": {str(k): str(v) for k, v in physical_keys.items()},
                "representation": {
                    str(k): str(v) for k, v in representation_keys.items()
                },
            }

        indices_by_representation: dict[str, list[int]] = defaultdict(list)
        representation_metadata: dict[str, tuple[str, str]] = {}
        patch_metadata: list[tuple[str, str, str]] = []
        for sample_idx, patch in enumerate(patches):
            dataset_idx = int(patch.segment.dataset_idx)
            segment_relpath = str(patch.segment.segment_relpath)
            try:
                contract = dataset_contracts[dataset_idx]
            except KeyError as exc:
                raise ValueError(
                    f"patch references unknown dataset_idx={dataset_idx}"
                ) from exc
            try:
                physical_key = contract["physical"][segment_relpath]
                representation_key = contract["representation"][segment_relpath]
            except KeyError as exc:
                raise ValueError(
                    "patch representation is missing an explicit sampling mapping: "
                    f"dataset_idx={dataset_idx}, segment_relpath={segment_relpath!r}"
                ) from exc
            scroll = contract["scroll"]
            previous = representation_metadata.setdefault(
                representation_key, (scroll, physical_key)
            )
            if previous != (scroll, physical_key):
                raise ValueError(
                    f"representation_key={representation_key!r} maps to conflicting "
                    f"scroll/physical keys: {previous!r} vs {(scroll, physical_key)!r}"
                )
            indices_by_representation[representation_key].append(int(sample_idx))
            patch_metadata.append((scroll, physical_key, representation_key))

        observed_scrolls = {scroll for scroll, _, _ in patch_metadata}
        if observed_scrolls != set(self.batch_quotas):
            raise ValueError(
                "fixed scroll quota keys must exactly match patch scrolls: "
                f"quotas={sorted(self.batch_quotas)}, patches={sorted(observed_scrolls)}"
            )

        representations_by_physical: dict[tuple[str, str], set[str]] = defaultdict(set)
        physicals_by_scroll: dict[str, set[str]] = defaultdict(set)
        physical_scroll: dict[str, str] = {}
        for representation, (scroll, physical) in representation_metadata.items():
            previous_scroll = physical_scroll.setdefault(physical, scroll)
            if previous_scroll != scroll:
                raise ValueError(
                    f"physical_segment_key={physical!r} crosses scrolls: "
                    f"{previous_scroll!r} vs {scroll!r}"
                )
            representations_by_physical[(scroll, physical)].add(representation)
            physicals_by_scroll[scroll].add(physical)

        self._patch_metadata = patch_metadata
        self._indices_by_representation = {
            key: list(values) for key, values in indices_by_representation.items()
        }
        self._representations_by_physical = {
            key: sorted(values) for key, values in representations_by_physical.items()
        }
        self._physicals_by_scroll = {
            scroll: sorted(values) for scroll, values in physicals_by_scroll.items()
        }
        self._epoch_batches = len(patches) // self.batch_size
        if self._epoch_batches <= 0:
            raise ValueError(
                f"fixed-prior sampler has {len(patches)} patches, fewer than one batch"
            )

        self._physical_queues = {
            scroll: _RecyclingShuffledQueue(
                physicals,
                seed=_stable_child_seed(self.seed, f"physical:{scroll}"),
            )
            for scroll, physicals in self._physicals_by_scroll.items()
        }
        self._representation_queues = {
            key: _RecyclingShuffledQueue(
                representations,
                seed=_stable_child_seed(
                    self.seed, f"representation:{key[0]}:{key[1]}"
                ),
            )
            for key, representations in self._representations_by_physical.items()
        }
        self._patch_queues = {
            representation: _RecyclingShuffledQueue(
                indices,
                seed=_stable_child_seed(self.seed, f"patch:{representation}"),
            )
            for representation, indices in self._indices_by_representation.items()
        }
        self._batch_order_rng = random.Random(
            _stable_child_seed(self.seed, "final-batch-order")
        )
        self._yielded_batches = 0
        self._observed_scrolls: Counter[str] = Counter()
        self._observed_physicals: Counter[str] = Counter()
        self._observed_representations: Counter[str] = Counter()

    def __len__(self) -> int:
        return int(self._epoch_batches)

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(len(self)):
            batch: list[int] = []
            batch_scrolls: Counter[str] = Counter()
            batch_physicals: Counter[str] = Counter()
            batch_representations: Counter[str] = Counter()
            for scroll, quota in self.batch_quotas.items():
                for _ in range(quota):
                    physical = self._physical_queues[scroll].pop()
                    representation = self._representation_queues[
                        (scroll, physical)
                    ].pop()
                    sample_idx = int(self._patch_queues[representation].pop())
                    sample_scroll, sample_physical, sample_representation = (
                        self._patch_metadata[sample_idx]
                    )
                    if (sample_scroll, sample_physical, sample_representation) != (
                        scroll,
                        physical,
                        representation,
                    ):
                        raise RuntimeError("fixed-prior sampler hierarchy became inconsistent")
                    batch.append(sample_idx)
                    batch_scrolls[scroll] += 1
                    batch_physicals[physical] += 1
                    batch_representations[representation] += 1
            if dict(batch_scrolls) != self.batch_quotas:
                raise RuntimeError(
                    f"fixed-prior batch quota violation: {dict(batch_scrolls)!r}"
                )
            self._batch_order_rng.shuffle(batch)
            self._yielded_batches += 1
            self._observed_scrolls.update(batch_scrolls)
            self._observed_physicals.update(batch_physicals)
            self._observed_representations.update(batch_representations)
            yield batch

    def definition_audit(self) -> dict:
        patch_counts_scroll = Counter(scroll for scroll, _, _ in self._patch_metadata)
        patch_counts_physical = Counter(
            physical for _, physical, _ in self._patch_metadata
        )
        patch_counts_representation = Counter(
            representation for _, _, representation in self._patch_metadata
        )
        return {
            "strategy": "fixed_scroll_prior_stratified",
            "seed": self.seed,
            "batch_size": self.batch_size,
            "batches_per_loader_epoch": len(self),
            "target_per_batch": dict(self.batch_quotas),
            "target_fraction": {
                scroll: quota / self.batch_size
                for scroll, quota in self.batch_quotas.items()
            },
            "source_patches": len(self._patch_metadata),
            "source_patches_by_scroll": dict(sorted(patch_counts_scroll.items())),
            "source_patches_by_physical_segment": dict(
                sorted(patch_counts_physical.items())
            ),
            "source_patches_by_representation": dict(
                sorted(patch_counts_representation.items())
            ),
            "physical_segments_by_scroll": {
                scroll: list(physicals)
                for scroll, physicals in sorted(self._physicals_by_scroll.items())
            },
            "representations_by_physical_segment": {
                physical: list(self._representations_by_physical[(scroll, physical)])
                for scroll, physicals in sorted(self._physicals_by_scroll.items())
                for physical in physicals
            },
        }

    def observed_audit(self) -> dict:
        return {
            "strategy": "fixed_scroll_prior_stratified",
            "seed": self.seed,
            "batches_yielded_to_dataloader": int(self._yielded_batches),
            "samples_yielded_to_dataloader": int(
                self._yielded_batches * self.batch_size
            ),
            "observed_by_scroll": dict(sorted(self._observed_scrolls.items())),
            "observed_by_physical_segment": dict(
                sorted(self._observed_physicals.items())
            ),
            "observed_by_representation": dict(
                sorted(self._observed_representations.items())
            ),
            "patch_queue_recycles": {
                key: queue.recycles
                for key, queue in sorted(self._patch_queues.items())
            },
        }


def hierarchical_scroll_segment_weights(
    patches: Sequence,
    datasets: Sequence[dict],
) -> tuple[torch.Tensor, dict]:
    """Equalize scroll mass, then physical-segment mass within each scroll.

    Multiple representations of the same physical segment (for example native
    9.362 um and pooled 2.399 um PHerc0139) intentionally share one segment
    budget.  Their patch windows divide that budget instead of counting as
    independent segments.
    """

    if not patches:
        raise ValueError("hierarchical sampling requires at least one patch")

    dataset_scrolls: dict[int, str] = {}
    for dataset_idx, dataset in enumerate(datasets):
        scroll = str(dataset.get("sampling_scroll", "")).strip()
        if not scroll:
            raise ValueError(
                f"datasets[{dataset_idx}] must define non-empty sampling_scroll"
            )
        dataset_scrolls[dataset_idx] = scroll

    segment_keys: list[tuple[str, str]] = []
    for patch in patches:
        dataset_idx = int(patch.segment.dataset_idx)
        try:
            scroll = dataset_scrolls[dataset_idx]
        except KeyError as exc:
            raise ValueError(f"patch references unknown dataset_idx={dataset_idx}") from exc
        segment = str(patch.segment.segment_relpath)
        segment_keys.append((scroll, segment))

    patch_counts = Counter(segment_keys)
    segments_by_scroll: dict[str, set[str]] = defaultdict(set)
    for scroll, segment in patch_counts:
        segments_by_scroll[scroll].add(segment)
    scroll_count = len(segments_by_scroll)

    weights = []
    for scroll, segment in segment_keys:
        weights.append(
            1.0
            / (
                scroll_count
                * len(segments_by_scroll[scroll])
                * patch_counts[(scroll, segment)]
            )
        )
    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    weights_tensor /= weights_tensor.sum()

    audit = {
        "strategy": "scroll_segment_balanced",
        "scrolls": scroll_count,
        "segments": len(patch_counts),
        "patches": len(segment_keys),
        "segments_per_scroll": {
            scroll: len(segments)
            for scroll, segments in sorted(segments_by_scroll.items())
        },
        "patches_per_scroll": {
            scroll: sum(
                count
                for (candidate_scroll, _), count in patch_counts.items()
                if candidate_scroll == scroll
            )
            for scroll in sorted(segments_by_scroll)
        },
    }
    return weights_tensor, audit


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
        multiprocessing_context=None,
    ) -> DataLoader:
        generator = None
        if len(dataset) > 0 and bool(shuffle):
            generator = torch.Generator()
            generator.manual_seed(int(self.seed))
        persistent_workers = int(num_workers) > 0
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": int(batch_size),
            "shuffle": False if len(dataset) <= 0 else bool(shuffle),
            "drop_last": True,
            "num_workers": int(num_workers),
            "persistent_workers": persistent_workers,
            "pin_memory": bool(pin_memory),
            "collate_fn": collate_fn,
            "generator": generator,
        }
        if multiprocessing_context is not None:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context
        return DataLoader(**loader_kwargs)


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
        multiprocessing_context=None,
    ) -> DataLoader:
        del shuffle
        if len(dataset) <= 0:
            persistent_workers = int(num_workers) > 0
            loader_kwargs = {
                "dataset": dataset,
                "batch_size": int(batch_size),
                "shuffle": False,
                "drop_last": True,
                "num_workers": int(num_workers),
                "persistent_workers": persistent_workers,
                "pin_memory": bool(pin_memory),
                "collate_fn": collate_fn,
            }
            if multiprocessing_context is not None:
                loader_kwargs["multiprocessing_context"] = multiprocessing_context
            return DataLoader(**loader_kwargs)

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
        persistent_workers = int(num_workers) > 0
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": int(batch_size),
            "shuffle": False,
            "sampler": weighted_sampler,
            "drop_last": True,
            "num_workers": int(num_workers),
            "persistent_workers": persistent_workers,
            "pin_memory": bool(pin_memory),
            "collate_fn": collate_fn,
        }
        if multiprocessing_context is not None:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context
        return DataLoader(**loader_kwargs)


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
        multiprocessing_context=None,
    ) -> DataLoader:
        del batch_size, shuffle
        if len(dataset) <= 0:
            persistent_workers = int(num_workers) > 0
            loader_kwargs = {
                "dataset": dataset,
                "batch_size": int(self.batch_size),
                "shuffle": False,
                "drop_last": True,
                "num_workers": int(num_workers),
                "persistent_workers": persistent_workers,
                "pin_memory": bool(pin_memory),
                "collate_fn": collate_fn,
            }
            if multiprocessing_context is not None:
                loader_kwargs["multiprocessing_context"] = multiprocessing_context
            return DataLoader(**loader_kwargs)
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

        persistent_workers = int(num_workers) > 0
        loader_kwargs = {
            "dataset": dataset,
            "batch_sampler": _BatchSampler(),
            "num_workers": int(num_workers),
            "persistent_workers": persistent_workers,
            "pin_memory": bool(pin_memory),
            "collate_fn": collate_fn,
        }
        if multiprocessing_context is not None:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context
        return DataLoader(**loader_kwargs)


__all__ = [
    "FixedScrollPriorStratifiedBatchSampler",
    "GroupBalancedSampler",
    "GroupStratifiedSampler",
    "ShuffleSampler",
    "hierarchical_scroll_segment_weights",
]
