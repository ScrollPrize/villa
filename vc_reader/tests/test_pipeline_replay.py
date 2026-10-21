"""Exact workload replay and transform output validation."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from pipeline_replay import image_shape, restore_requests, validate_outputs


class PipelineReplayTests(unittest.TestCase):
    def test_replays_recorded_grids_without_replacing_locations(self):
        grid = [[8, 9, 10], [1234.25, 2345.5, 3456.75], np.eye(3).tolist()]
        source = dict(suite='readers', schema_version=3,
                      workloads={'rotated_box': dict(requests=[[grid]], coverage=dict(unique_chunks=5001, requests=1))})
        snapshot = copy.deepcopy(source)
        restored = restore_requests(source, ['rotated_box'], 5000)['rotated_box']
        self.assertEqual(restored[0][0][0], (8, 9, 10))
        self.assertEqual(restored[0][0][1], grid[1])
        np.testing.assert_array_equal(restored[0][0][2], np.eye(3, dtype=np.float32))
        self.assertEqual(restored[0][0][2].dtype, np.float32)
        self.assertEqual(source, snapshot)
        with self.assertRaisesRegex(ValueError, 'below 5002'):
            restore_requests(source, ['rotated_box'], 5002)
        with self.assertRaisesRegex(ValueError, 'absent'):
            restore_requests(source, ['surface'], 5000)

    def test_surface_uses_saved_translation_and_checks_geometry_hashes(self):
        points = np.ones((2, 3, 3), dtype=np.float32)
        description = dict(files_sha256={'x.tif': 'fixture'}, tiles=[dict(coordinate_origin=(10, 20, 30),
                                                                          local_coordinates_sha256='coordinates')])
        source = dict(suite='readers', schema_version=3, config=dict(surface='/fixture'),
                      input=dict(shape=[100, 100, 100]), surface=json.loads(json.dumps(description)),
                      workloads={'surface': dict(coverage=dict(unique_chunks=5000, requests=1),
                          requests=[dict(shape=[2, 3], template=0, coordinate_origin=[40, 50, 60])])})
        with patch('bench_pipeline.surface_workload', return_value=([[((2, 3), (10, 20, 30), points)]], description)):
            batch = restore_requests(source, ['surface'], 5000)['surface'][0]
            self.assertEqual(batch[0][1], [40, 50, 60])
            self.assertIs(batch[0][2], points)
            source['surface']['tiles'][0]['local_coordinates_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'geometry differs'):
                restore_requests(source, ['surface'], 5000)

    def test_plane_and_surface_are_2d_but_slabs_remain_3d(self):
        self.assertEqual(image_shape('plane', (1, 512, 512)), (1, 512, 512))
        self.assertEqual(image_shape('surface', (96, 512)), (1, 96, 512))
        self.assertEqual(image_shape('slab', (16, 512, 512)), (1, 16, 512, 512))
        self.assertEqual(image_shape('ml_batch', (128, 128, 128)), (1, 128, 128, 128))

    def test_validation_rejects_nonfinite_and_material_differences(self):
        grids = [((2, 3), (0, 0, 0), None)]
        a = np.ones((1, 2, 3), dtype=np.float32)
        result = validate_outputs(dict(cpu=[a], cuda=[a + np.float32(1e-6)]), grids, 'surface')
        self.assertGreater(result['max_abs'], 0)
        with self.assertRaisesRegex(AssertionError, 'differ'):
            validate_outputs(dict(cpu=[a], cuda=[a + np.float32(.01)]), grids, 'surface')
        with self.assertRaisesRegex(ValueError, 'invalid'):
            validate_outputs(dict(cpu=[a * np.float32(np.nan)]), grids, 'surface')
        with self.assertRaisesRegex(ValueError, 'output count'):
            validate_outputs(dict(cpu=[]), grids, 'surface')


if __name__ == '__main__':
    unittest.main()
