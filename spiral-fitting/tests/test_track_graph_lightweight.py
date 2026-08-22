"""The fitter's CSR-only TrackGraph must match the full topology wrapper."""
import unittest

import numpy as np

from track_graph import TrackGraph


def crossing_cache():
    # Two undirected edges: 0--1 and 1--2, stored as symmetric CSR records.
    return {
        "source_ids": np.array([10, 20, 30], dtype=np.uint64),
        "offsets": np.array([0, 1, 3, 4], dtype=np.int64),
        "partners": np.array([1, 0, 2, 1], dtype=np.int32),
        "self_local": np.array([1, 2, 1, 2], dtype=np.int32),
        "partner_local": np.array([2, 1, 2, 1], dtype=np.int32),
        "positions": np.array([1.0, 2.0, 3.0, 4.0]),
        "clearances": np.array([0.1, 0.1, 0.2, 0.2]),
    }


class LightweightTrackGraphTests(unittest.TestCase):

    def setUp(self):
        self.full = TrackGraph(crossing_cache())
        self.light = TrackGraph(crossing_cache(), build_topology=False)

    def assert_csr_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)

    def test_counts_do_not_require_duplicate_topology(self):
        self.assertIsNone(self.light.graph)
        self.assertEqual(len(self.light), len(self.full))
        self.assertEqual(self.light.edge_count, self.full.edge_count)

    def test_restricted_csr_is_bitwise_identical(self):
        selected = np.array([10, 20], dtype=np.uint64)
        self.assert_csr_equal(
            self.light.restricted_csr(selected),
            self.full.restricted_csr(selected))

    def test_clipped_csr_is_bitwise_identical(self):
        selected = np.array([10, 20, 30], dtype=np.uint64)
        input_offsets = np.array([0, 3, 6, 9], dtype=np.int64)
        surviving = np.array([0, 1, 2], dtype=np.int64)
        point_map = np.arange(9, dtype=np.int32)
        output_offsets = input_offsets.copy()
        self.assert_csr_equal(
            self.light.clipped_csr(
                selected, input_offsets, surviving, point_map, output_offsets),
            self.full.clipped_csr(
                selected, input_offsets, surviving, point_map, output_offsets))

    def test_traversal_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "without traversal topology"):
            self.light.gated_random_walk(0, 2)


if __name__ == "__main__":
    unittest.main()
