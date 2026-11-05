"""CPU-only exact reference prefetch geometry, shared with process workers."""
import math
import itertools
import numpy as np

def reference_chunks(points_base, factor, shape, chunk):
    """Exact segment crossings of interpolation-support chunk boundaries.

    Both a chunk boundary and its preceding voxel can change the corner-chunk
    set. Visit each interval's midpoint and endpoints, not just annotations.
    """
    points = np.asarray(points_base, dtype=np.float64) / factor
    result = set()
    for left, right in zip(points[:-1], points[1:]):
        delta = right - left
        breaks = {0.0, 1.0}
        for xyz in range(3):
            if delta[xyz] == 0:
                continue
            lo, hi = sorted((left[xyz], right[xyz]))
            # Clip traversal ranges to the actual array, not out-of-volume
            # annotation extents. Shape is ZYX; points are XYZ.
            n = shape[2 - xyz]
            for boundary in (0, n - 1):
                t = (boundary - left[xyz]) / delta[xyz]
                if 0 < t < 1:
                    breaks.add(float(t))
            for k in range(max(0, math.floor(lo / chunk)), min(math.ceil(hi / chunk) + 2, math.ceil(n / chunk) + 1)):
                for boundary in (k * chunk, k * chunk - 1):
                    t = (boundary - left[xyz]) / delta[xyz]
                    if 0 < t < 1:
                        breaks.add(float(t))
        ordered = sorted(breaks)
        samples = ordered + [(a + b) * 0.5 for a, b in zip(ordered[:-1], ordered[1:])]
        for t in samples:
            p = (left + t * delta)[::-1]
            if np.any(p < 0) or np.any(p > np.asarray(shape) - 1):
                continue
            floor = np.floor(p).astype(np.int64)
            for corner in itertools.product((0, 1), repeat=3):
                index = np.minimum(floor + corner, np.asarray(shape) - 1) // chunk
                result.add(tuple(int(v) for v in index))
    return result

def reference_chunk_job(arguments):
    return reference_chunks(*arguments)
