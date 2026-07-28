import os
from pathlib import Path
import unittest

import numpy as np

from track_graph import TrackGraph
from tracks import _load_native_track_crossings


DEFAULT_TRACK_ROOT = Path(
    "/home/sean/Desktop/spiral_dataset/to_hf/tracks")
TRACK_BASENAME = "2um_ds2_ps256_surf_v2.dbm"
KNOWN_CYCLE_SOURCE_ID = 8065948596289


class RealTrackGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ.get(
            "VC_SPIRAL_REAL_TRACKS", DEFAULT_TRACK_ROOT))
        cache_path = root / f"{TRACK_BASENAME}.crossings.npz"
        if not cache_path.is_file():
            raise unittest.SkipTest(
                f"real 2um crossing cache is unavailable: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as stored:
            cache = {
                name: stored[name]
                for name in (
                    "source_ids", "offsets", "partners", "self_local",
                    "partner_local", "positions", "clearances")
            }
        cls.graph = TrackGraph(cache)

    def test_full_real_graph_dimensions(self):
        self.assertEqual(len(self.graph), 19_746_134)
        self.assertEqual(self.graph.edge_count, 50_399_694)

    def test_real_cycle_gate_and_gated_walk(self):
        root = self.graph.node_for_source_id(KNOWN_CYCLE_SOURCE_ID)
        cycles = self.graph.short_cycles_through(root, limit=100)
        self.assertTrue(cycles)
        self.assertTrue(all(
            len(cycle) == 4 and len(set(cycle)) == 4
            for cycle in cycles))

        passing_candidate = None
        witness = None
        for cycle in cycles:
            for candidate in (cycle[1], cycle[-1]):
                witness = self.graph.transition_return_cycle_witness(
                    root, root, candidate)
                if witness is not None:
                    passing_candidate = candidate
                    break
            if passing_candidate is not None:
                break
        self.assertIsNotNone(
            passing_candidate,
            "known real cycles contain no candidate with a 20-voxel exit")

        entry = self.graph.crossing_record(passing_candidate, root)
        exit_record = self.graph.crossing_record(
            passing_candidate, witness[2])
        separation = abs(
            float(self.graph.positions[exit_record])
            - float(self.graph.positions[entry]))
        self.assertGreaterEqual(separation, 20.0)
        self.assertEqual(witness[0], root)
        self.assertEqual(witness[-1], root)
        self.assertEqual(len(set(witness[:-1])), len(witness) - 1)
        candidate_begin = int(self.graph.offsets[passing_candidate])
        candidate_end = int(self.graph.offsets[passing_candidate + 1])
        candidate_positions = self.graph.positions[
            candidate_begin:candidate_end]
        self.assertFalse(self.graph.transition_has_return_cycle(
            root, root, passing_candidate,
            minimum_candidate_travel=float(np.ptp(candidate_positions)) + 1.0))
        self.assertFalse(self.graph.transition_has_return_cycle(
            root, root, passing_candidate, max_cycle_tracks=3))

        walk = self.graph.gated_random_walk(
            root, 3, rng=np.random.default_rng(7))
        self.assertGreaterEqual(len(walk), 2)
        self.assertEqual(len(set(walk)), len(walk))
        for step in range(1, len(walk)):
            self.assertTrue(self.graph.transition_has_return_cycle(
                root,
                walk[step - 1],
                walk[step],
                visited=walk[:step],
            ))

    def test_native_production_walk_uses_root_return_gate(self):
        native = _load_native_track_crossings()
        if native is None:
            self.skipTest("native track-crossing module is unavailable")
        root = self.graph.node_for_source_id(KNOWN_CYCLE_SOURCE_ID)
        produced = None
        chosen_csr = None
        for cycle in self.graph.short_cycles_through(root, limit=100):
            selected = self.graph.source_ids[np.asarray(cycle)]
            csr = self.graph.restricted_csr(selected)
            lengths = np.maximum.reduceat(
                csr["self_local"], csr["offsets"][:-1]).astype(
                    np.int32) + 1
            index = native.prepare_walk_index(
                csr["offsets"], csr["partners"], csr["self_local"],
                csr["partner_local"], csr["positions"], lengths)
            result = native.sample_walks(
                index,
                np.zeros(128, dtype=np.int32),
                np.arange(128, dtype=np.uint64),
                groups=1,
                target_points=24,
                minimum_hops=2,
                maximum_hops=8,
                minimum_steps=1,
                maximum_steps=int(lengths.max()),
                minimum_candidate_travel=20.0,
            )
            if int(result["produced"]) == 1:
                produced = result
                chosen_csr = csr
                break
        self.assertIsNotNone(
            produced, "no real short cycle produced a gated native walk")
        walk_hops = int(np.asarray(produced["walk_hops"])[0])
        self.assertGreaterEqual(walk_hops, 2)
        self.assertLessEqual(walk_hops, 3)
        self.assertLess(walk_hops, 8)
        walked = np.asarray(produced["tracks"])[0][:walk_hops + 1]
        self.assertEqual(len(set(walked.tolist())), len(walked))

        records = np.asarray(produced["records"])[0][:2 * walk_hops]
        for step in range(1, walk_hops):
            entry_record = int(records[2 * (step - 1) + 1])
            exit_record = int(records[2 * step])
            self.assertGreaterEqual(
                abs(
                    float(chosen_csr["positions"][exit_record])
                    - float(chosen_csr["positions"][entry_record])
                ),
                20.0,
            )
        final_entry = int(records[-1])
        final_track = int(walked[-1])
        final_begin = int(chosen_csr["offsets"][final_track])
        final_end = int(chosen_csr["offsets"][final_track + 1])
        valid_closures = [
            record
            for record in range(final_begin, final_end)
            if int(chosen_csr["partners"][record]) == int(walked[0])
            and abs(
                float(chosen_csr["positions"][record])
                - float(chosen_csr["positions"][final_entry])
            ) >= 20.0
        ]
        self.assertTrue(valid_closures)


if __name__ == "__main__":
    unittest.main()
