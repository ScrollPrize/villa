"""Coverage and padding exclusion for the reader comparison workload planner."""
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from bench_pipeline import coordinate_box, sampling_box, surface_workload
from reader_coverage import box_chunks, sampled_chunks, spread_workload


class ReaderCoverageTests(unittest.TestCase):
    def test_interpolation_coverage_matches_explicit_neighbors(self):
        chunks = np.array([4, 5, 6])
        origin = np.array([2**24 + 3, 13, 25])
        coords = np.array([[-.25, 0, 1.5, 4], [1, -2.5, 2, 7.25],
                           [-1.75, 2, 5, 10.5]], dtype=np.float32)
        expected = set()
        for point in coords.T:
            for neighbor in itertools.product(*[(int(np.floor(v)), int(np.ceil(v))) for v in point]):
                expected.add(tuple((np.array(neighbor) + origin) // chunks))
        self.assertEqual(sampled_chunks(coords, origin, chunks, (2**25, 128, 128)), expected)

    def test_spread_excludes_padding_and_counts_union_deterministically(self):
        class Reader:
            chunks = (4, 4, 4)
            shape = (64, 32, 32)

            def read_region(self, origin, shape):
                self.last = origin
                # A zero slab at both Z ends, plus padding on the low X side.
                valid = origin[0] >= 8 and origin[0] + shape[0] <= 56 and origin[2] >= 8
                return np.full(shape, int(valid), dtype=np.uint8)

        args = SimpleNamespace(seed=123, min_unique_chunks=100, crops=2, xy_radius=8)
        templates = [[((5, 5, 5), [24, 12, 12], np.eye(3, dtype=np.float32))]]
        first, coverage, _ = spread_workload('ml_batch', templates, Reader(), args, sampling_box, coordinate_box)
        second, other, _ = spread_workload('ml_batch', templates, Reader(), args, sampling_box, coordinate_box)
        self.assertEqual([g[0][1] for g in first], [g[0][1] for g in second])
        self.assertEqual(coverage, other)
        union = set()
        for batch in first:
            for shape, origin, axes in batch:
                self.assertTrue(np.all(Reader().read_region(origin, shape)))
                union.update(box_chunks(origin, shape, np.array(Reader.chunks)))
        self.assertEqual(len(union), coverage['unique_chunks'])
        self.assertGreaterEqual(len(union), 100)
        self.assertGreater(coverage['rejected_padding_regions'], 0)

    def test_surface_translation_preserves_geometry_and_coverage(self):
        class Reader:
            chunks = (4, 4, 4)
            shape = (64, 32, 32)

            def read_region(self, origin, shape):
                assert all(0 <= o and o + n <= v for o, n, v in zip(origin, shape, self.shape))
                return np.ones(shape, dtype=np.uint8)

        points = np.array([[[0.25, 0.5, 0.75], [4.25, 3.5, 4.75]],
                           [[1.75, 4.5, 0.75], [3.25, 5.5, 4.75]]], dtype=np.float32)
        args = SimpleNamespace(seed=123, min_unique_chunks=100, crops=2, xy_radius=8)
        requests, coverage, descriptions = spread_workload('surface', [[((2, 2), [24, 12, 12], points)]],
                                                          Reader(), args, sampling_box, coordinate_box)
        union = set()
        for batch, desc in zip(requests, descriptions):
            shape, origin, coords = batch[0]
            self.assertIs(coords, points)
            self.assertEqual(origin, desc['coordinate_origin'])
            union.update(sampled_chunks(np.moveaxis(coords, -1, 0), origin, Reader.chunks, Reader.shape))
        self.assertEqual(len(union), coverage['unique_chunks'])
        self.assertGreaterEqual(len(union), 100)

    def test_surface_fixture_report_is_json_serializable(self):
        args = SimpleNamespace(surface=Path(__file__).resolve().parents[1] / 'benchmarks/fixtures/fiber_132487',
                               surface_size=(96, 512), surface_coordinate_scale=1., crops=2)
        templates, description = surface_workload(args, (18946, 8174, 8174))
        json.dumps(description, allow_nan=False)
        json.dumps([dict(shape=batch[0][0], coordinate_origin=batch[0][1]) for batch in templates])

    def test_padding_only_input_cannot_satisfy_coverage(self):
        reader = SimpleNamespace(chunks=(4, 4, 4), shape=(64, 32, 32),
                                 read_region=lambda origin, shape: np.zeros(shape, dtype=np.uint8))
        args = SimpleNamespace(seed=123, min_unique_chunks=10, crops=1, xy_radius=8)
        templates = [[((2, 2, 2), [24, 12, 12], np.eye(3, dtype=np.float32))]]
        with self.assertRaisesRegex(ValueError, 'only 0 nonzero-region chunks'):
            spread_workload('ml_batch', templates, reader, args, sampling_box, coordinate_box)

    def test_impossible_target_fails_before_reads(self):
        reader = SimpleNamespace(chunks=(32, 32, 32), shape=(64, 64, 64))
        args = SimpleNamespace(min_unique_chunks=5000)
        with self.assertRaisesRegex(ValueError, 'fewer than 5000'):
            spread_workload('ml_batch', [], reader, args, sampling_box, coordinate_box)


if __name__ == '__main__':
    unittest.main()
