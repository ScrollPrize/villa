"""Deterministic continuous sampling without control-point-density bias."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiberLocation:
    """An interpolated base-volume location, deliberately not a CP index."""

    segment_index: int
    fraction: float
    xyz: tuple[float, float, float]


class MixedSourceArcSampler:
    """Exactly half of every logical batch from each supervision source.

    Scheduling depends only on global sample index and seed, never on loader
    order, rank, microbatch size or pool size. Reuse the ordinary scroll/arc
    sampler within each source; dataset size does not change source balance.
    """

    def __init__(self, records, *, global_batch_size, seed):
        if type(global_batch_size) is not int or global_batch_size <= 0 or global_batch_size % 2:
            raise ValueError("mixed supervision needs a positive even global batch size")
        self.global_batch_size = global_batch_size
        self.seed = int(seed)
        self.indices = {}
        for index, record in enumerate(records):
            mode = record.dataset_config.get("supervision_mode", "precise")
            if mode not in ("precise", "coarse_area"):
                raise ValueError(f"unknown supervision_mode: {mode!r}")
            self.indices.setdefault(mode, []).append(index)
        if set(self.indices) != {"precise", "coarse_area"}:
            raise ValueError("mixed supervision requires both precise and coarse_area records")
        self.samplers = {mode: ScrollArcSampler([records[i] for i in indices])
                         for mode, indices in self.indices.items()}
        self._batch = None
        self._schedule = None

    def source(self, sample_index):
        if sample_index < 0:
            raise ValueError("sample index must be nonnegative")
        batch, offset = divmod(int(sample_index), self.global_batch_size)
        if self._batch != batch:
            # A separate deterministic stream leaves ordinary location RNG
            # independent of how often this schedule is requested/recreated.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, batch, 0x4D4958]))
            self._schedule = rng.permutation(self.global_batch_size) < self.global_batch_size // 2
            self._batch = batch
        return "coarse_area" if self._schedule[offset] else "precise"

    def sample(self, sample_index, rng):
        mode = self.source(sample_index)
        record, local_index, location = self.samplers[mode].sample(rng)
        return record, self.indices[mode][local_index], location


class ScrollArcSampler:
    def __init__(self, records):
        self.records = records
        self.segment_cdfs = []
        grouped = {}
        for index, record in enumerate(records):
            scroll = record.dataset_config.get("scroll_id")
            if not isinstance(scroll, str) or not scroll.strip():
                raise ValueError("scroll_balanced_arclength requires a nonempty dataset scroll_id")
            points = np.asarray(record.fiber.line_points_xyz, dtype=np.float64)
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            cdf = np.cumsum(lengths)
            if not np.isfinite(points).all() or not len(cdf) or not np.isfinite(cdf).all() or cdf[-1] <= 0:
                raise ValueError(f"scroll {scroll}: fiber {record.fiber.path} needs positive finite line length")
            self.segment_cdfs.append(cdf)
            grouped.setdefault(scroll, []).append(index)
        self.scrolls = tuple(sorted(grouped))
        self.record_indices = [np.asarray(grouped[s], dtype=np.int64) for s in self.scrolls]
        self.fiber_cdfs = [np.cumsum([self.segment_cdfs[i][-1] for i in ids])
                           for ids in self.record_indices]
        self.distance_scales = [None] * len(self.scrolls)
        for scroll_index, ids in enumerate(self.record_indices):
            configs = [records[i].dataset_config for i in ids]
            if not any('sampling_group' in c or 'sampling_group_weight' in c for c in configs):
                continue
            weights, totals, groups = {}, {}, []
            for index, config in zip(ids, configs):
                group = config.get('sampling_group')
                weight = config.get('sampling_group_weight')
                if not isinstance(group, str) or not group.strip():
                    raise ValueError('every dataset in a grouped scroll needs a sampling_group')
                if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
                    raise ValueError('sampling_group_weight must be positive and finite')
                if group in weights and weights[group] != weight:
                    raise ValueError('datasets sharing a sampling_group must agree on its weight')
                weights[group] = weight
                totals[group] = totals.get(group, 0.0) + self.segment_cdfs[index][-1]
                groups.append(group)
            # Reweight fibers by group mass / group length. One uniform distance
            # draw still selects both fiber and position; no extra RNG calls.
            lengths = np.asarray([self.segment_cdfs[i][-1] for i in ids])
            masses = np.asarray([length * weights[group] / totals[group]
                                 for length, group in zip(lengths, groups)])
            cdf = np.cumsum(masses)
            if not np.isfinite(cdf).all() or np.any(np.diff(np.r_[0.0, cdf]) <= 0):
                raise ValueError('sampling group masses are not numerically representable')
            self.fiber_cdfs[scroll_index] = cdf
            self.distance_scales[scroll_index] = lengths / masses

    def sample(self, rng):
        scroll_index = int(rng.integers(len(self.scrolls)))
        fibers = self.fiber_cdfs[scroll_index]
        distance = min(float(rng.random() * fibers[-1]), np.nextafter(fibers[-1], 0.0))
        local_index = int(np.searchsorted(fibers, distance, side="right"))
        record_index = int(self.record_indices[scroll_index][local_index])
        distance -= float(fibers[local_index - 1]) if local_index else 0.0
        scales = self.distance_scales[scroll_index]
        if scales is not None:
            distance *= float(scales[local_index])
        return self._location(record_index, distance)

    def sample_within(self, rng, record_indices):
        """Conditional arclength draw for replacement in a fixed scroll/group."""
        lengths = np.asarray([self.segment_cdfs[i][-1] for i in record_indices])
        cdf = np.cumsum(lengths)
        distance = min(float(rng.random()*cdf[-1]), np.nextafter(cdf[-1], 0.0))
        local = int(np.searchsorted(cdf, distance, side="right"))
        distance -= float(cdf[local-1]) if local else 0.0
        return self._location(int(record_indices[local]), distance)

    def _location(self, record_index, distance):
        segments = self.segment_cdfs[record_index]
        distance = min(distance, np.nextafter(segments[-1], 0.0))
        segment = int(np.searchsorted(segments, distance, side="right"))
        previous = float(segments[segment - 1]) if segment else 0.0
        fraction = (distance - previous) / (float(segments[segment]) - previous)
        points = self.records[record_index].fiber.line_points_xyz
        p0, p1 = np.asarray(points[segment], dtype=np.float64), np.asarray(points[segment + 1], dtype=np.float64)
        location = FiberLocation(segment, fraction, tuple(p0 + fraction * (p1 - p0)))
        return self.records[record_index], record_index, location
