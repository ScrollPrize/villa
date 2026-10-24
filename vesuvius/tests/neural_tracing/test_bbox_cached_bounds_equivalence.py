"""Regression checks for the cached per-record extrema used by bbox optimization."""
import unittest
import numpy as np

from vesuvius.neural_tracing.inference.generate_segment_cover_bboxes import _points_within_bbox


def _cached_accepts(points, bbox):
    """Mirror the O(1) bounds check in _optimize_band_bboxes, not an independent implementation."""
    if not len(points):
        return True
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    return bool(
        lo[0] >= bbox[0] and hi[0] < bbox[1] + 1
        and lo[1] >= bbox[2] and hi[1] < bbox[3] + 1
        and lo[2] >= bbox[4] and hi[2] < bbox[5] + 1
    )


class CachedBboxBoundsEquivalence(unittest.TestCase):
    def test_half_open_edges_and_degenerate_inputs(self):
        box = (0, 127, -25, 75, 100, 228)
        cases = [
            np.empty((0, 3), dtype=float),
            np.array([[0, -25, 100], [127.999, 75.999, 228.999]]),
            np.array([[128, 0, 150]]),
            np.array([[-0.001, 0, 150]]),
            np.array([[42, 76, 150]]),
            np.array([[42, 0, 229]]),
            np.array([[42, 0, np.nan]]),
            np.array([[np.inf, 0, 100]]),
            np.array([[-np.inf, 0, 100]]),
            np.array([[42, 0, 100], [42, np.nan, 100]]),
        ]
        for pts in cases:
            with self.subTest(points=repr(pts)):
                expected = bool(np.all(_points_within_bbox(pts, box)))
                self.assertEqual(_cached_accepts(pts, box), expected)

    def test_randomized_finite_and_boundary_points(self):
        rng = np.random.default_rng(2054)
        for _ in range(300):
            start = rng.integers(-500, 500, size=3)
            width = rng.integers(1, 300, size=3)
            bbox = tuple(int(v) for pair in zip(start, start + width - 1) for v in pair)
            points = rng.uniform(start - 2, start + width + 2, size=(int(rng.integers(1, 50)), 3))
            expected = bool(np.all(_points_within_bbox(points, bbox)))
            self.assertEqual(_cached_accepts(points, bbox), expected)


if __name__ == "__main__":
    unittest.main()
