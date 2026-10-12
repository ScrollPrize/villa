import json
import os
import tempfile
import unittest

import numpy as np

import merge_concat_runs
import tifxyz
from tifxyz_spacing import grid_spacing_and_area


def plane(rows, cols, du, dv):
    """A tilted plane sampled every du voxels along columns and dv along rows, ZYX."""
    j, i = np.meshgrid(np.arange(cols, dtype=np.float64), np.arange(rows, dtype=np.float64))
    u, v = j * du, i * dv
    zyxs = np.stack([1000.0 + v, 2000.0 + 0.6 * u, 3000.0 + 0.8 * u], axis=-1)       # |d/du| = 1: spacing du along columns
    return zyxs.astype(np.float32)


class GridSpacingAndAreaTests(unittest.TestCase):
    def test_measures_spacing_and_area_not_the_step(self):
        zyxs = plane(6, 9, du=24.0, dv=30.0)
        valid = np.ones(zyxs.shape[:2], bool)
        cols, rows, area = grid_spacing_and_area(zyxs, valid, step_size=20.0)
        self.assertAlmostEqual(cols, 24.0, places=3)
        self.assertAlmostEqual(rows, 30.0, places=3)
        self.assertEqual(area, round(5 * 8 * 24.0 * 30.0))

    def test_invalid_vertices_are_left_out(self):
        zyxs = plane(6, 9, du=24.0, dv=30.0)
        zyxs[2, 3] = -1.0
        valid = np.any(zyxs != -1, axis=-1)
        cols, rows, area = grid_spacing_and_area(zyxs, valid, step_size=20.0)
        self.assertAlmostEqual(cols, 24.0, places=3)
        self.assertEqual(area, round((5 * 8 - 4) * 24.0 * 30.0))           # the four quads around the hole

    def test_axis_without_neighbours_falls_back_to_the_step(self):
        zyxs = plane(1, 9, du=24.0, dv=30.0)
        cols, rows, area = grid_spacing_and_area(zyxs, np.ones((1, 9), bool), step_size=20.0)
        self.assertEqual((rows, area), (20.0, 0))

    def test_writers_record_the_measured_scale(self):
        zyxs = plane(6, 9, du=24.0, dv=30.0)
        with tempfile.TemporaryDirectory() as temporary:
            tifxyz.save_tifxyz(zyxs, temporary, 'w000', 20.0, 9.362, 'test')
            with open(os.path.join(temporary, 'w000', 'meta.json')) as f:
                meta = json.load(f)
            merge_concat_runs.save_tifxyz(zyxs[..., ::-1], os.path.join(temporary, 'merged'), 'w000', 20.0, 9.362, 'test')
            with open(os.path.join(temporary, 'merged', 'meta.json')) as f:
                merged = json.load(f)
        for m in (meta, merged):
            self.assertAlmostEqual(m['scale'][0], 1 / 24.0, places=6)
            self.assertAlmostEqual(m['scale'][1], 1 / 30.0, places=6)
            self.assertEqual(m['area_vx2'], round(5 * 8 * 24.0 * 30.0))
        self.assertAlmostEqual(meta['area_cm2'], meta['area_vx2'] * 9.362 ** 2 / 1e8, places=9)


if __name__ == '__main__':
    unittest.main()
