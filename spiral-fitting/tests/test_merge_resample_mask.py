import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import merge_concat_runs


class ResampleMaskTests(unittest.TestCase):
    def test_downsampling_masks_a_hole_skipped_by_nearest_neighbor(self):
        pts = np.full((5, 20, 3), 5000.0, np.float32)
        pts[2, 9] = -1.0

        out, valid = merge_concat_runs.resample_to(pts, (5, 10))

        expected_valid = np.ones((5, 10), dtype=bool)
        expected_valid[1:4, 3:6] = False
        np.testing.assert_array_equal(valid, expected_valid)
        self.assertTrue(np.all(out[~valid] == -1.0))
        self.assertTrue(np.all(out[valid] == 5000.0))

    def test_zero_weight_nonfinite_input_does_not_leave_valid_nonfinite_output(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                pts = np.full((5, 7, 3), 5000.0, np.float32)
                pts[0, 4] = value

                out, valid = merge_concat_runs.resample_to(pts, (2, 3))

                # The center column samples source column 3 exactly, but OpenCV
                # can still evaluate the zero-weight contribution from column 4.
                self.assertTrue(np.all(np.isfinite(out[valid])))
                self.assertTrue(np.all(out[~valid] == -1.0))
                self.assertTrue(valid[1, 0])
                np.testing.assert_array_equal(out[1, 0], [5000.0] * 3)

    def test_downsampling_never_keeps_a_vertex_blended_with_a_missing_one(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            h, w = int(rng.integers(5, 12)), int(rng.integers(12, 40))
            pts = np.stack(np.meshgrid(np.arange(w, dtype=np.float32) * 20.0, np.arange(h, dtype=np.float32) * 20.0), axis=-1)
            pts = np.concatenate([pts, np.full((h, w, 1), 5000.0, np.float32)], axis=-1)
            holes = rng.random((h, w)) < 0.05
            pts[holes] = -1.0
            th, tw = h, max(2, int(w * rng.uniform(0.5, 0.95)))
            out, valid = merge_concat_runs.resample_to(pts, (th, tw))
            self.assertTrue(np.all(out[~valid] == -1.0))
            self.assertTrue(np.all(out[valid][:, 2] > 4999.0), "a vertex kept as valid was interpolated toward the -1 marker")

    def test_same_size_only_masks_invalid_vertices(self):
        pts = np.full((4, 5, 3), 7.0, np.float32)
        pts[1, 2] = -1.0
        out, valid = merge_concat_runs.resample_to(pts, (4, 5))
        self.assertEqual(int((~valid).sum()), 1)
        self.assertTrue(np.all(out[1, 2] == -1.0))


if __name__ == "__main__":
    unittest.main()
