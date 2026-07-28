import unittest

import numpy as np
from scipy.spatial import cKDTree

from track_graph import TrackGraph
from tracks import (
    _build_crossing_partner_csr,
    prepare_main_phase_tracks,
)


def small_crossing_cache():
    return {
        "source_ids": np.array([10, 20, 30], dtype=np.uint64),
        "offsets": np.array([0, 1, 3, 4], dtype=np.int64),
        "partners": np.array([1, 0, 2, 1], dtype=np.int32),
        "self_local": np.array([2, 1, 4, 3], dtype=np.int32),
        "partner_local": np.array([1, 2, 3, 4], dtype=np.int32),
        "positions": np.array([2.0, 1.0, 4.0, 3.0]),
        "clearances": np.ones(4, dtype=np.float64),
    }


class TrackGraphTests(unittest.TestCase):
    def test_builds_undirected_rustworkx_topology(self):
        graph = TrackGraph(small_crossing_cache(), track_chunk_size=1)

        self.assertEqual(len(graph), 3)
        self.assertEqual(graph.edge_count, 2)
        self.assertEqual(graph.graph.degree(1), 2)

    def test_restricts_crossings_by_stable_source_identity(self):
        graph = TrackGraph(small_crossing_cache())
        restricted = graph.restricted_csr(
            np.array([10, 20], dtype=np.uint64))

        np.testing.assert_array_equal(restricted["offsets"], [0, 1, 2])
        np.testing.assert_array_equal(restricted["partners"], [1, 0])
        np.testing.assert_array_equal(restricted["self_local"], [2, 1])
        np.testing.assert_array_equal(restricted["partner_local"], [1, 2])

    def test_prepare_uses_native_graph_cache_without_clipping(self):
        graph = TrackGraph(small_crossing_cache())
        tracks = [
            np.arange(15, dtype=np.float32).reshape(5, 3),
            np.arange(15, 30, dtype=np.float32).reshape(5, 3),
        ]
        prepared = prepare_main_phase_tracks(
            tracks,
            None,
            0.0,
            "cpu",
            sampling_config={
                "track_crossing_precompute_max": 1,
                "track_max_track_crossing_per_step": 1,
            },
            track_source_ids=np.array([10, 20], dtype=np.uint64),
            track_graph=graph,
        )

        self.assertEqual(
            int(prepared["crossing_index_stats"]["directed_crossings"]), 2)

    def test_remaps_crossings_and_drops_excluded_crossing_points(self):
        graph = TrackGraph(small_crossing_cache())
        input_offsets = np.array([0, 5, 11, 16], dtype=np.int64)
        kept = np.ones(16, dtype=bool)
        kept[1] = False       # Before track 0's crossing: shifts its index.
        kept[5] = False       # Before track 1's first crossing.
        kept[9] = False       # Track 1 local 4: removes its track-2 crossing.
        old_to_new = np.full(16, -1, dtype=np.int32)
        old_to_new[kept] = np.arange(np.count_nonzero(kept), dtype=np.int32)
        output_offsets = np.array([0, 4, 8, 13], dtype=np.int64)

        clipped = graph.clipped_csr(
            graph.source_ids,
            input_offsets,
            np.array([0, 1, 2]),
            old_to_new,
            output_offsets,
        )

        np.testing.assert_array_equal(clipped["offsets"], [0, 1, 2, 2])
        np.testing.assert_array_equal(clipped["partners"], [1, 0])
        np.testing.assert_array_equal(clipped["self_local"], [1, 0])
        np.testing.assert_array_equal(clipped["partner_local"], [0, 1])

    def test_prepare_reuses_graph_after_point_exclusion(self):
        horizontal = np.array([
            [10, 10, 8], [10, 10, 9], [10, 10, 10],
            [10, 10, 11], [10, 10, 12],
        ], dtype=np.float32)
        vertical = np.array([
            [10, 8, 10], [10, 9, 10], [10, 10, 10],
            [10, 11, 10], [10, 12, 10],
        ], dtype=np.float32)
        source_ids = np.array([10, 20], dtype=np.uint64)
        cache = _build_crossing_partner_csr(
            [horizontal, vertical],
            ["horizontal", "vertical"],
            source_ids=source_ids,
        )
        graph = TrackGraph(cache)
        # Remove a non-crossing point before each crossing, forcing both cached
        # local indices to change while leaving the crossing itself active.
        anchors = cKDTree(np.array([
            horizontal[0], vertical[0],
        ], dtype=np.float32))

        prepared = prepare_main_phase_tracks(
            [horizontal, vertical],
            None,
            0.1,
            "cpu",
            anchor_tree=anchors,
            sampling_config={
                "track_crossing_precompute_max": 1,
                "track_max_track_crossing_per_step": 1,
            },
            track_families=["horizontal", "vertical"],
            track_source_ids=source_ids,
            track_graph=graph,
        )

        self.assertEqual(prepared["lengths"].tolist(), [4, 4])
        if "crossing_index_stats" in prepared:
            self.assertEqual(
                int(prepared["crossing_index_stats"]["directed_crossings"]), 2)
        else:
            self.assertEqual(len(prepared["crossing_csr"]["partners"]), 2)


if __name__ == "__main__":
    unittest.main()
