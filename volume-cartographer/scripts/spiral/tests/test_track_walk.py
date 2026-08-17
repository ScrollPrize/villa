import pathlib
import sys
import unittest

import numpy as np
import torch


SPIRAL_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(SPIRAL_DIR) not in sys.path:
    sys.path.insert(0, str(SPIRAL_DIR))

from tracks import (
    _load_native_track_crossings,
    _crossing_row_alignments,
    _sample_prepared_track_points,
    prepare_main_phase_tracks,
    validate_track_sampling_config,
)


def _csr(track_count, crossings):
    """Build reciprocal directed CSR records from (a, alocal, b, blocal)."""
    rows = [[] for _ in range(track_count)]
    for a, a_local, b, b_local in crossings:
        rows[a].append((b, a_local, b_local))
        rows[b].append((a, b_local, a_local))
    offsets = np.zeros(track_count + 1, dtype=np.int64)
    for track, row in enumerate(rows):
        offsets[track + 1] = offsets[track] + len(row)
    partners = np.asarray(
        [item[0] for row in rows for item in row], dtype=np.int32)
    self_local = np.asarray(
        [item[1] for row in rows for item in row], dtype=np.int32)
    partner_local = np.asarray(
        [item[2] for row in rows for item in row], dtype=np.int32)
    return offsets, partners, self_local, partner_local


@unittest.skipIf(
    _load_native_track_crossings() is None,
    "native track-crossing module is unavailable")
@unittest.skip(
    "superseded by test_track_graph_real production-gate coverage")
class NativeTrackWalkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = _load_native_track_crossings()

    def _chain(self):
        arrays = _csr(6, [
            (0, 50, 1, 10),
            (1, 40, 2, 10),
            (2, 40, 3, 10),
            (3, 40, 4, 10),
        ])
        return self.native.prepare_walk_index(
            *arrays, np.full(6, 100, dtype=np.int32),
            require_loop_consistency=False)

    def test_four_hops_are_distinct_alternating_chain(self):
        index = self._chain()
        result = self.native.sample_walks(
            index,
            np.zeros(8, dtype=np.int32),
            np.arange(8, dtype=np.uint64),
            groups=1, target_points=24, hops=4,
            minimum_steps=30, maximum_steps=30)
        self.assertEqual(int(result["produced"]), 1)
        np.testing.assert_array_equal(
            np.asarray(result["tracks"]), [[0, 1, 2, 3, 4]])
        self.assertEqual(len(set(np.asarray(result["tracks"])[0])), 5)

    def test_seeded_output_is_deterministic_and_resamples_failed_primary(self):
        index = self._chain()
        kwargs = dict(
            groups=1, target_points=24, hops=4,
            minimum_steps=24, maximum_steps=40)
        candidates = np.asarray([5, 0], dtype=np.int32)
        seeds = np.asarray([91, 92], dtype=np.uint64)
        first = self.native.sample_walks(index, candidates, seeds, **kwargs)
        second = self.native.sample_walks(index, candidates, seeds, **kwargs)
        self.assertEqual(int(first["rejected_candidates"]), 1)
        np.testing.assert_array_equal(first["tracks"], second["tracks"])
        np.testing.assert_array_equal(first["records"], second["records"])

    def test_adaptive_sampling_is_deterministic_and_stops_when_full(self):
        index = self._chain()
        kwargs = dict(
            primary_probabilities=np.asarray(
                [1., 0., 0., 0., 0., 0.], dtype=np.float64),
            seed=91, groups=4, target_points=24, hops=4,
            minimum_steps=24, maximum_steps=40,
            maximum_attempts=256)
        first = self.native.sample_walks_adaptive(index, **kwargs)
        second = self.native.sample_walks_adaptive(index, **kwargs)
        self.assertEqual(int(first["produced"]), 4)
        self.assertEqual(int(first["attempted_candidates"]), 4)
        np.testing.assert_array_equal(first["tracks"], second["tracks"])
        np.testing.assert_array_equal(first["records"], second["records"])
        np.testing.assert_array_equal(
            np.asarray(first["tracks"])[:, 0], np.zeros(4, dtype=np.int32))

    def test_cached_preparation_compacts_to_selected_tracks(self):
        arrays = _csr(6, [
            (0, 50, 1, 10),
            (1, 40, 2, 10),
            (2, 40, 3, 10),
            (3, 40, 4, 10),
        ])
        selected = np.asarray([100, 102, 103], dtype=np.uint64)
        index = self.native.prepare_cached_walk_index(
            np.arange(100, 106, dtype=np.uint64),
            *arrays,
            selected,
            np.full(3, 100, dtype=np.int32),
            require_loop_consistency=False)
        compact = self.native.walk_index_crossings(index)
        self.assertEqual(index.track_count, 3)
        # Only the original 2-3 crossing survives restriction.
        np.testing.assert_array_equal(compact["offsets"], [0, 0, 1, 2])
        np.testing.assert_array_equal(compact["partners"], [2, 1])

    def test_loop_filter_excludes_bridge_spur(self):
        # Four rail intervals form a closed crossing loop. The final interval
        # on track 0 leads from that loop to a one-crossing spur track.
        arrays = _csr(5, [
            (0, 10, 1, 10),  # A
            (1, 20, 2, 10),  # B
            (2, 20, 3, 10),  # C
            (3, 20, 0, 20),  # D
            (0, 40, 4, 10),  # E, beyond D: D-E is a bridge
        ])
        lengths = np.full(5, 64, dtype=np.int32)
        ordinary = self.native.prepare_walk_index(
            *arrays, lengths, require_loop_consistency=False)
        filtered = self.native.prepare_walk_index(
            *arrays, lengths, require_loop_consistency=True)
        ordinary_stats = dict(self.native.walk_index_stats(ordinary))
        filtered_stats = dict(self.native.walk_index_stats(filtered))
        self.assertEqual(ordinary_stats["eligible_tracks"], 5)
        self.assertEqual(filtered_stats["cyclic_components"], 1)
        self.assertEqual(filtered_stats["eligible_tracks"], 4)
        self.assertLess(
            filtered_stats["eligible_directed_crossings"],
            ordinary_stats["eligible_directed_crossings"])


class TrackWalkConfigurationTests(unittest.TestCase):
    def test_defaults_use_gated_track_walk(self):
        policy = validate_track_sampling_config({})
        self.assertEqual(policy["crossing_mode"], "track_walk")
        self.assertEqual(policy["track_min_walk_steps_per_track"], 24)
        self.assertEqual(policy["track_max_walk_steps_per_track"], 256)
        self.assertEqual(policy["track_min_walks_per_track"], 2)
        self.assertEqual(policy["track_max_walks_per_track"], 4)
        self.assertEqual(policy["walk_minimum_cycle_travel"], 20.0)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "track_crossing_mode"):
            validate_track_sampling_config({"track_crossing_mode": "bad"})
        for key in (
                "track_min_walk_steps_per_track",
                "track_max_walk_steps_per_track",
                "track_min_walks_per_track",
                "track_max_walks_per_track"):
            with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError, "positive integer"):
                validate_track_sampling_config({key: 0})
        with self.assertRaisesRegex(ValueError, "must be <="):
            validate_track_sampling_config({
                "track_min_walk_steps_per_track": 10,
                "track_max_walk_steps_per_track": 9,
            })
        with self.assertRaisesRegex(ValueError, "finite number"):
            validate_track_sampling_config({
                "track_walk_minimum_cycle_travel": -1})

    @unittest.skipIf(
        _load_native_track_crossings() is None,
        "native track-crossing module is unavailable")
    @unittest.skip(
        "superseded by test_track_graph_real production-gate coverage")
    def test_python_sampler_gathers_complete_tracks(self):
        def horizontal(y, x0, x1):
            x = np.arange(x0, x1 + 1, dtype=np.float32)
            return np.stack((np.zeros_like(x), np.full_like(x, y), x), axis=1)

        def vertical(x, y0, y1):
            y = np.arange(y0, y1 + 1, dtype=np.float32)
            return np.stack((np.zeros_like(y), y, np.full_like(y, x)), axis=1)

        tracks = [
            horizontal(10, 0, 60),
            vertical(50, 0, 70),
            horizontal(40, 0, 99),
            vertical(80, 30, 70),
            horizontal(70, 60, 99),
        ]
        config = validate_track_sampling_config({
            "track_crossing_mode": "track_walk",
            "track_crossing_precompute_max": 0,
            "track_min_walk_steps_per_track": 30,
            "track_max_walk_steps_per_track": 30,
            "track_min_walks_per_track": 2,
            "track_max_walks_per_track": 4,
        })
        prepared = prepare_main_phase_tracks(
            tracks, None, 0.0, torch.device("cpu"),
            sampling_config=config,
            track_families=["horizontal", "vertical", "horizontal",
                            "vertical", "horizontal"])
        sample = _sample_prepared_track_points(
            prepared, 1, 8, min_sample_spacing=5, max_sample_spacing=10)
        self.assertEqual(sample["group_width"], 5)
        self.assertEqual(len(set(sample["track_idx"].tolist())), 5)
        # Each row spans the complete spacing-bounded resample, including both
        # endpoints, rather than a hop-local subsegment.
        expected = prepared["resampled_cache"][next(iter(
            prepared["resampled_cache"]))]["lengths"][sample["track_idx"]]
        torch.testing.assert_close(sample["row_lengths"], expected)
        self.assertTrue(torch.all(sample["row_lengths"] >= 5))

    def test_chain_alignment_propagates_through_every_hop(self):
        radii = torch.tensor([10.0, 2.0, 5.0, 1.0, 7.0, 3.0])
        alignment = _crossing_row_alignments(
            radii,
            torch.tensor([0, 2, 4]),
            torch.tensor([1, 3, 5]),
            torch.tensor([1, 2, 3]),
            row_count=4,
            chain=True)
        # +8 aligns row 1; row 2 then sees 5+8 at its source crossing,
        # and row 3 likewise inherits the full preceding-chain offset.
        torch.testing.assert_close(
            alignment, torch.tensor([0.0, 8.0, 12.0, 16.0]))

    def test_chain_alignment_vectorizes_independent_groups(self):
        radii = torch.tensor([
            10.0, 2.0, 5.0, 1.0,
            20.0, 19.0, 30.0, 27.0,
        ])
        alignment = _crossing_row_alignments(
            radii,
            torch.tensor([0, 2, 4, 6]),
            torch.tensor([1, 3, 5, 7]),
            torch.tensor([1, 2, 4, 5]),
            row_count=6,
            chain=True)
        torch.testing.assert_close(
            alignment, torch.tensor([0.0, 8.0, 12.0, 0.0, 1.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
