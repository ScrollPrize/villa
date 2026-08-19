import unittest
from types import SimpleNamespace

import numpy as np

import dt_targets
from spiral_sampling import load_spiral_sampling


spiral_sampling = load_spiral_sampling()


class PatchSamplingBindingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            spiral_sampling, 'vc.spiral_sampling is required for patch sampling')
        self.mask = np.ones((31, 37), dtype=bool)
        self.mask[8:14, 10:22] = False
        self.atlas = spiral_sampling.PatchSamplingAtlas([self.mask])

    def test_patch_points_are_deterministic_distinct_and_valid(self):
        indices = np.zeros(24, dtype=np.int64)
        first = self.atlas.sample_patch_points(indices, 40, 1234)
        second = self.atlas.sample_patch_points(indices, 40, 1234)
        np.testing.assert_array_equal(first['ijs'], second['ijs'])
        np.testing.assert_array_equal(first['counts'], second['counts'])
        ijs = np.asarray(first['ijs'])
        counts = np.asarray(first['counts'])
        self.assertEqual(ijs.shape, (24, 40, 2))
        np.testing.assert_array_equal(counts, 40)
        floors = np.floor(ijs).astype(np.int64)
        self.assertTrue(self.mask[floors[..., 0], floors[..., 1]].all())
        for row in floors:
            self.assertEqual(len(np.unique(row, axis=0)), 40)

    def test_small_patch_uses_every_cell_and_pads_with_valid_geometry(self):
        mask = np.zeros((4, 5), dtype=bool)
        mask[[0, 1, 3], [1, 4, 2]] = True
        atlas = spiral_sampling.PatchSamplingAtlas([mask])
        result = atlas.sample_patch_points(np.array([0]), 8, 77)
        ijs = np.asarray(result['ijs'])[0]
        self.assertEqual(int(np.asarray(result['counts'])[0]), 3)
        cells = np.floor(ijs).astype(np.int64)
        self.assertEqual(len(np.unique(cells[:3], axis=0)), 3)
        self.assertTrue(mask[cells[:, 0], cells[:, 1]].all())
        np.testing.assert_array_equal(ijs[3:], np.repeat(ijs[:1], 5, axis=0))

    def test_native_strip_apis_are_absent(self):
        self.assertFalse(hasattr(self.atlas, 'sample_patch_strips'))
        self.assertFalse(hasattr(self.atlas, 'sample_patch_walks'))
        self.assertFalse(hasattr(self.atlas, 'sample_l_shapes'))

    def test_append_preserves_patch_index_order(self):
        second_mask = np.ones((9, 11), dtype=bool)
        self.atlas.append([second_mask])
        result = np.asarray(self.atlas.sample_patch_points(
            np.array([1], dtype=np.int64), 16, 7)['ijs'])
        self.assertLess(result[..., 0].max(), 9)
        self.assertLess(result[..., 1].max(), 11)


@unittest.skipUnless(spiral_sampling is not None, "vc.spiral_sampling is not built")
class DtTargetBindingTests(unittest.TestCase):
    def setUp(self):
        self.previous_binding = dt_targets._spiral_sampling

    def tearDown(self):
        dt_targets._spiral_sampling = self.previous_binding

    def test_dt_sample_preparation_matches_python(self):
        mask = np.ones((53, 71), dtype=bool)
        mask[7:19, 21:36] = False
        mask[40:, :12] = False
        patch = SimpleNamespace(
            _sampling_valid_quad_mask_np=mask,
            scale=np.array([0.25, 0.5]),
        )
        dt_targets._spiral_sampling = None
        dt_targets.prepare_patch_dt_target_samples([patch], 256, 128)
        expected = (
            patch._dt_target_ijs.copy(),
            patch._dt_target_block_rc.copy(),
            patch._dt_target_block_shape,
            patch._dt_target_anchor_max_dist_sq,
        )
        dt_targets._spiral_sampling = spiral_sampling
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
        dt_targets._spiral_sampling = None
        expected = dt_targets._unwrap_block_samples(
            theta, block_rc, (rows, columns))
        dt_targets._spiral_sampling = spiral_sampling
        actual = dt_targets._unwrap_block_samples(
            theta, block_rc, (rows, columns))
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
