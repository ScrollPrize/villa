import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from dinovol_2.dataset import object_labels
from dinovol_2.dataset.object_labels import _mean_nn_distance_at_most, fiber_objects


def exact_mean_nn_distance_at_most(tree, points, limit, bound):
    distances, _ = tree.query(points, k=1)
    return float(distances.mean()) <= limit


def line(start, direction, length, offset=(0, 0, 0)):
    t = np.arange(length, dtype=np.float32)[:, None]
    return np.asarray(start, np.float32) + np.asarray(offset, np.float32) + t * np.asarray(direction, np.float32)


class FiberDedupTests(unittest.TestCase):
    def test_capped_decision_matches_exact_mean(self):
        rng = np.random.default_rng(0)
        for trial in range(200):
            tree_points = line(rng.uniform(0, 20, 3), rng.normal(size=3) / 3, int(rng.integers(5, 200)))
            tree_points += rng.normal(scale=rng.uniform(0, 2), size=tree_points.shape).astype(np.float32)
            query = tree_points[rng.choice(len(tree_points), int(rng.integers(1, 100)))]
            query = query + rng.normal(scale=rng.uniform(0, 8), size=query.shape).astype(np.float32)
            tree = cKDTree(tree_points)
            for limit in (0.5, 1.5, 3.0, 10.0):
                for bound in (limit, limit * 4, 1e9):
                    self.assertEqual(_mean_nn_distance_at_most(tree, query, limit, bound),
                                     exact_mean_nn_distance_at_most(tree, query, limit, bound),
                                     (trial, limit, bound))

    def test_fiber_objects_match_exact_dedup(self):
        shape = (40, 40, 40)
        present = np.zeros(shape, bool)
        direction = np.zeros(shape + (3,), np.float32)
        # Parallel fibers along x: exact duplicates one voxel apart, near ones, and distinct ones.
        for z, y, length in ((5, 5, 35), (5, 6, 30), (6, 6, 25), (5, 9, 34), (20, 20, 38), (21, 22, 20),
                             (30, 10, 12), (30, 30, 36), (31, 30, 3)):
            present[z, y, 1:1 + length] = True
            direction[z, y, 1:1 + length] = (0, 0, 1)
        # A diagonal fiber crossing the others' bounding boxes.
        for i in range(30):
            present[5 + i, 5 + i // 2, 10] = True
            direction[5 + i, 5 + i // 2, 10] = (0.894, 0.447, 0)
        got = fiber_objects(present, direction, min_voxels=3, dedup_distance=1.5)
        with patch.object(object_labels, "_mean_nn_distance_at_most", exact_mean_nn_distance_at_most):
            want = fiber_objects(present, direction, min_voxels=3, dedup_distance=1.5)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
        self.assertGreater(int(got[0].max()), 2)


if __name__ == "__main__":
    unittest.main()
