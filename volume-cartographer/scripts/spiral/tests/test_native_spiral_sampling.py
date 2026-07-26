import unittest
from types import SimpleNamespace

import numpy as np

import dt_targets
from native_spiral import load_native_spiral_sampling


native = load_native_spiral_sampling()


@unittest.skipUnless(native is not None, "vc.spiral_sampling is not built")
class NativePatchSamplingTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((31, 37), dtype=bool)
        self.mask[8:14, 10:22] = False
        self.atlas = native.PatchSamplingAtlas([self.mask])

    def test_straight_strip_samples_are_deterministic_and_valid(self):
        indices = np.zeros(24, dtype=np.int64)
        first = np.asarray(self.atlas.sample_patch_strips(indices, 40, 1234))
        second = np.asarray(self.atlas.sample_patch_strips(indices, 40, 1234))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 24, 40, 2))
        floors = np.floor(first).astype(np.int64)
        self.assertTrue(self.mask[floors[..., 0], floors[..., 1]].all())
        # Horizontal strips share i, vertical strips share j.
        self.assertTrue(np.allclose(first[0, :, :, 0], first[0, :, :1, 0]))
        self.assertTrue(np.allclose(first[1, :, :, 1], first[1, :, :1, 1]))

    def test_l_shapes_report_invalid_anchors_and_stay_on_mask(self):
        anchors = np.array([[4, 4], [10, 12], [20, 25]], dtype=np.int64)
        result = self.atlas.sample_l_shapes(
            np.zeros(len(anchors), dtype=np.int64), anchors, 32, 99)
        valid = np.asarray(result["valid"], dtype=bool)
        np.testing.assert_array_equal(valid, [True, False, True])
        ijs = np.asarray(result["ijs"])[valid]
        floors = np.floor(ijs).astype(np.int64)
        self.assertTrue(self.mask[floors[..., 0], floors[..., 1]].all())

    def test_append_preserves_patch_index_order(self):
        second_mask = np.ones((9, 11), dtype=bool)
        self.atlas.append([second_mask])
        result = np.asarray(self.atlas.sample_patch_strips(
            np.array([1], dtype=np.int64), 16, 7))
        self.assertLess(result[..., 0].max(), 9)
        self.assertLess(result[..., 1].max(), 11)


@unittest.skipUnless(native is not None, "vc.spiral_sampling is not built")
class NativeDtTargetTests(unittest.TestCase):
    def setUp(self):
        self.previous_native = dt_targets._native_spiral_sampling

    def tearDown(self):
        dt_targets._native_spiral_sampling = self.previous_native

    def test_dt_sample_preparation_matches_python(self):
        mask = np.ones((53, 71), dtype=bool)
        mask[7:19, 21:36] = False
        mask[40:, :12] = False
        patch = SimpleNamespace(
            _sampling_valid_quad_mask_np=mask,
            scale=np.array([0.25, 0.5]),
        )
        dt_targets._native_spiral_sampling = None
        dt_targets.prepare_patch_dt_target_samples([patch], 256, 128)
        expected = (
            patch._dt_target_ijs.copy(),
            patch._dt_target_block_rc.copy(),
            patch._dt_target_block_shape,
            patch._dt_target_anchor_max_dist_sq,
        )
        dt_targets._native_spiral_sampling = native
        dt_targets.prepare_patch_dt_target_samples([patch], 256, 128)
        np.testing.assert_array_equal(patch._dt_target_ijs, expected[0])
        np.testing.assert_array_equal(patch._dt_target_block_rc, expected[1])
        self.assertEqual(patch._dt_target_block_shape, expected[2])
        self.assertEqual(patch._dt_target_anchor_max_dist_sq, expected[3])

    def test_block_unwrap_matches_python(self):
        rows, columns = 17, 19
        all_rc = np.stack(np.unravel_index(
            np.arange(rows * columns), (rows, columns)), axis=1).astype(np.int32)
        keep = ~((all_rc[:, 0] == 8) & (all_rc[:, 1] > 3))
        block_rc = all_rc[keep]
        rng = np.random.default_rng(12)
        theta = rng.uniform(-np.pi, np.pi, len(block_rc)).astype(np.float32)
        dt_targets._native_spiral_sampling = None
        expected = dt_targets._unwrap_block_samples(
            theta, block_rc, (rows, columns))
        dt_targets._native_spiral_sampling = native
        actual = dt_targets._unwrap_block_samples(
            theta, block_rc, (rows, columns))
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
