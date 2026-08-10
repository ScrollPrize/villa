"""Uniform, scroll/segment-balanced, and fixed-scroll-prior sampling policies."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
import hashlib
import random

import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.types import Patch, SamplingPolicy


def _stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class _RecyclingQueue:
    def __init__(self, values: Sequence, *, seed: int) -> None:
        if not values:
            raise ValueError("recycling queue requires at least one value")
        self._source = list(values)
        self._rng = random.Random(int(seed))
        self._order = []
        self._cursor = 0
        self._recycles = -1
        self._recycle()

    @property
    def recycles(self) -> int:
        return max(0, self._recycles)

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
    """Draw exact scroll quotas, then balance physical/representation/patch queues."""

    def __init__(
        self,
        patches: Sequence[Patch],
        config: InkDataConfig,
        *,
        batch_size: int,
    ) -> None:
        self.batch_size = int(batch_size)
        self.drop_last = True
        self.seed = config.sampling.seed
        self.batch_quotas = dict(config.sampling.fixed_batch_quotas)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.batch_quotas or any(value <= 0 for value in self.batch_quotas.values()):
            raise ValueError("fixed scroll quotas must all be positive")
        if sum(self.batch_quotas.values()) != self.batch_size:
            raise ValueError(
                f"fixed scroll quotas sum to {sum(self.batch_quotas.values())}, "
                f"expected batch_size={self.batch_size}"
            )
        if not patches:
            raise ValueError("fixed-prior sampling requires at least one patch")

        for dataset_idx, source in enumerate(config.datasets):
            if not source.sampling_scroll:
                raise ValueError(
                    f"datasets[{dataset_idx}] must define non-empty sampling_scroll"
                )
            if not source.sampling_physical_segment_keys:
                raise ValueError(
                    f"datasets[{dataset_idx}] must define sampling_physical_segment_keys"
                )
            if not source.sampling_representation_keys:
                raise ValueError(
                    f"datasets[{dataset_idx}] must define sampling_representation_keys"
                )

        metadata: list[tuple[str, str, str]] = []
        indices_by_representation: dict[str, list[int]] = defaultdict(list)
        representation_metadata: dict[str, tuple[str, str]] = {}
        for sample_idx, patch in enumerate(patches):
            source = config.datasets[patch.segment.dataset_idx]
            relpath = patch.segment.segment_relpath
            if not source.sampling_scroll:
                raise ValueError(
                    f"datasets[{patch.segment.dataset_idx}] must define non-empty sampling_scroll"
                )
            try:
                physical = source.sampling_physical_segment_keys[relpath]
                representation = source.sampling_representation_keys[relpath]
            except KeyError as exc:
                raise ValueError(
                    "patch representation is missing an explicit sampling mapping: "
                    f"dataset_idx={patch.segment.dataset_idx}, segment_relpath={relpath!r}"
                ) from exc
            previous = representation_metadata.setdefault(
                representation, (source.sampling_scroll, physical)
            )
            if previous != (source.sampling_scroll, physical):
                raise ValueError(
                    f"representation_key={representation!r} maps to conflicting metadata"
                )
            indices_by_representation[representation].append(sample_idx)
            metadata.append((source.sampling_scroll, physical, representation))
        observed_scrolls = {scroll for scroll, _, _ in metadata}
        if observed_scrolls != set(self.batch_quotas):
            raise ValueError(
                "fixed scroll quota keys must exactly match patch scrolls: "
                f"quotas={sorted(self.batch_quotas)}, patches={sorted(observed_scrolls)}"
            )
        representations_by_physical: dict[tuple[str, str], set[str]] = defaultdict(set)
        physicals_by_scroll: dict[str, set[str]] = defaultdict(set)
        physical_scroll: dict[str, str] = {}
        for representation, (scroll, physical) in representation_metadata.items():
            previous = physical_scroll.setdefault(physical, scroll)
            if previous != scroll:
                raise ValueError(f"physical_segment_key={physical!r} crosses scrolls")
            representations_by_physical[(scroll, physical)].add(representation)
            physicals_by_scroll[scroll].add(physical)
        self._metadata = metadata
        self._physicals_by_scroll = {
            key: sorted(values) for key, values in physicals_by_scroll.items()
        }
        self._representations_by_physical = {
            key: sorted(values) for key, values in representations_by_physical.items()
        }
        self._epoch_batches = len(patches) // self.batch_size
        if self._epoch_batches <= 0:
            raise ValueError(
                f"fixed-prior sampler has {len(patches)} patches, fewer than one batch"
            )
        self._physical_queues = {
            scroll: _RecyclingQueue(
                values, seed=_stable_seed(self.seed, f"physical:{scroll}")
            )
            for scroll, values in self._physicals_by_scroll.items()
        }
        self._representation_queues = {
            key: _RecyclingQueue(
                values,
                seed=_stable_seed(self.seed, f"representation:{key[0]}:{key[1]}"),
            )
            for key, values in self._representations_by_physical.items()
        }
        self._patch_queues = {
            key: _RecyclingQueue(
                values, seed=_stable_seed(self.seed, f"patch:{key}")
            )
            for key, values in indices_by_representation.items()
        }
        self._batch_rng = random.Random(_stable_seed(self.seed, "final-batch-order"))
        self._yielded = 0
        self._observed_scrolls: Counter[str] = Counter()
        self._observed_physicals: Counter[str] = Counter()
        self._observed_representations: Counter[str] = Counter()

    def __len__(self) -> int:
        return self._epoch_batches

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(len(self)):
            batch: list[int] = []
            batch_scrolls: Counter[str] = Counter()
            batch_physicals: Counter[str] = Counter()
            batch_representations: Counter[str] = Counter()
            for scroll, quota in self.batch_quotas.items():
                for _ in range(quota):
                    physical = self._physical_queues[scroll].pop()
                    representation = self._representation_queues[(scroll, physical)].pop()
                    sample_idx = int(self._patch_queues[representation].pop())
                    if self._metadata[sample_idx] != (scroll, physical, representation):
                        raise RuntimeError("fixed-prior sampler hierarchy became inconsistent")
                    batch.append(sample_idx)
                    batch_scrolls[scroll] += 1
                    batch_physicals[physical] += 1
                    batch_representations[representation] += 1
            if dict(batch_scrolls) != self.batch_quotas:
                raise RuntimeError(f"fixed-prior batch quota violation: {dict(batch_scrolls)!r}")
            self._batch_rng.shuffle(batch)
            self._yielded += 1
            self._observed_scrolls.update(batch_scrolls)
            self._observed_physicals.update(batch_physicals)
            self._observed_representations.update(batch_representations)
            yield batch

    def definition_audit(self) -> dict:
        scroll_counts = Counter(scroll for scroll, _, _ in self._metadata)
        physical_counts = Counter(physical for _, physical, _ in self._metadata)
        representation_counts = Counter(representation for _, _, representation in self._metadata)
        return {
            "strategy": "fixed_scroll_prior_stratified",
            "seed": self.seed,
            "batch_size": self.batch_size,
            "batches_per_loader_epoch": len(self),
            "target_per_batch": dict(self.batch_quotas),
            "target_fraction": {
                scroll: quota / self.batch_size for scroll, quota in self.batch_quotas.items()
            },
            "source_patches": len(self._metadata),
            "source_patches_by_scroll": dict(sorted(scroll_counts.items())),
            "source_patches_by_physical_segment": dict(sorted(physical_counts.items())),
            "source_patches_by_representation": dict(sorted(representation_counts.items())),
            "physical_segments_by_scroll": self._physicals_by_scroll,
            "representations_by_physical_segment": {
                physical: self._representations_by_physical[(scroll, physical)]
                for scroll, physicals in sorted(self._physicals_by_scroll.items())
                for physical in physicals
            },
        }

    def observed_audit(self) -> dict:
        return {
            "strategy": "fixed_scroll_prior_stratified",
            "seed": self.seed,
            "batches_yielded_to_dataloader": self._yielded,
            "samples_yielded_to_dataloader": self._yielded * self.batch_size,
            "observed_by_scroll": dict(sorted(self._observed_scrolls.items())),
            "observed_by_physical_segment": dict(
                sorted(self._observed_physicals.items())
            ),
            "observed_by_representation": dict(
                sorted(self._observed_representations.items())
            ),
            "patch_queue_recycles": {
                key: queue.recycles for key, queue in sorted(self._patch_queues.items())
            },
        }


def hierarchical_scroll_segment_weights(
    patches: Sequence[Patch], config: InkDataConfig
) -> tuple[torch.Tensor, dict]:
    """Equalize scroll mass, then representation-segment mass within each scroll."""
    if not patches:
        raise ValueError("hierarchical sampling requires at least one patch")
    keys: list[tuple[str, str]] = []
    for patch in patches:
        source = config.datasets[patch.segment.dataset_idx]
        if not source.sampling_scroll:
            raise ValueError(
                f"datasets[{patch.segment.dataset_idx}] must define non-empty sampling_scroll"
            )
        keys.append((source.sampling_scroll, patch.segment.segment_relpath))
    patch_counts = Counter(keys)
    segments_by_scroll: dict[str, set[str]] = defaultdict(set)
    for scroll, segment in patch_counts:
        segments_by_scroll[scroll].add(segment)
    scroll_count = len(segments_by_scroll)
    weights = torch.as_tensor(
        [
            1.0
            / (
                scroll_count
                * len(segments_by_scroll[scroll])
                * patch_counts[(scroll, segment)]
            )
            for scroll, segment in keys
        ],
        dtype=torch.double,
    )
    weights /= weights.sum()
    return weights, {
        "strategy": "scroll_segment_balanced",
        "scrolls": scroll_count,
        "segments": len(patch_counts),
        "patches": len(keys),
        "segments_per_scroll": {
            scroll: len(segments) for scroll, segments in sorted(segments_by_scroll.items())
        },
        "patches_per_scroll": {
            scroll: sum(
                count
                for (candidate, _), count in patch_counts.items()
                if candidate == scroll
            )
            for scroll in sorted(segments_by_scroll)
        },
    }


def build_sampling_policy(
    patches: Sequence[Patch], config: InkDataConfig, *, batch_size: int
) -> SamplingPolicy:
    """Resolve exactly one of the three training DataLoader sampling seams."""
    strategy = config.sampling.strategy
    if strategy == "uniform":
        generator = torch.Generator()
        generator.manual_seed(config.sampling.seed)
        return SamplingPolicy(
            strategy=strategy,
            shuffle=True,
            generator=generator,
            audit={"strategy": strategy, "patches": len(patches)},
        )
    if strategy == "scroll_segment_balanced":
        weights, audit = hierarchical_scroll_segment_weights(patches, config)
        generator = torch.Generator()
        generator.manual_seed(config.sampling.seed)
        sampler = WeightedRandomSampler(
            weights,
            len(patches),
            replacement=True,
            generator=generator,
        )
        return SamplingPolicy(
            strategy=strategy,
            generator=generator,
            sampler=sampler,
            audit=audit,
        )
    batch_sampler = FixedScrollPriorStratifiedBatchSampler(
        patches, config, batch_size=batch_size
    )
    return SamplingPolicy(
        strategy=strategy,
        batch_sampler=batch_sampler,
        audit=batch_sampler.definition_audit(),
    )
