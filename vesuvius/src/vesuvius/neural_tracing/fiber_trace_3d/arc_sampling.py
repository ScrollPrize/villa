"""Deterministic continuous sampling without control-point-density bias."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiberLocation:
    """An interpolated base-volume location, deliberately not a CP index."""

    segment_index: int
    fraction: float
    xyz: tuple[float, float, float]


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

    def sample(self, rng):
        scroll_index = int(rng.integers(len(self.scrolls)))
        fibers = self.fiber_cdfs[scroll_index]
        distance = min(float(rng.random() * fibers[-1]), np.nextafter(fibers[-1], 0.0))
        local_index = int(np.searchsorted(fibers, distance, side="right"))
        record_index = int(self.record_indices[scroll_index][local_index])
        distance -= float(fibers[local_index - 1]) if local_index else 0.0
        segments = self.segment_cdfs[record_index]
        distance = min(distance, np.nextafter(segments[-1], 0.0))
        segment = int(np.searchsorted(segments, distance, side="right"))
        previous = float(segments[segment - 1]) if segment else 0.0
        fraction = (distance - previous) / (float(segments[segment]) - previous)
        points = self.records[record_index].fiber.line_points_xyz
        p0, p1 = np.asarray(points[segment], dtype=np.float64), np.asarray(points[segment + 1], dtype=np.float64)
        location = FiberLocation(segment, fraction, tuple(p0 + fraction * (p1 - p0)))
        return self.records[record_index], record_index, location
