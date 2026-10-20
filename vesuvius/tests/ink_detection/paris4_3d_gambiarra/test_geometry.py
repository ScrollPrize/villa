"""View geometry (point coordinates vs crop voxels) and the point losses. CPU."""
import itertools
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from vesuvius.ink_detection.paris4_3d_gambiarra import geometry as g, losses, rotation_aug, views


class CubeViews(unittest.TestCase):
    def test_points_follow_the_crop(self):
        """A parent voxel q lands at view voxel points(q) of the cube_crop for all 48 symmetries."""
        rng = np.random.default_rng(0)
        parent = rng.integers(0, 256, (256,) * 3, dtype=np.uint8)
        v = np.concatenate([g.CORNERS, rng.integers(0, 128, (200, 3))]).astype(np.float64)
        for perm, flips in itertools.product(g.PERMS, itertools.product((False, True), repeat=3)):
            item = dict(offset=rng.integers(16, 113, 3).tolist(), permutation=list(perm), flips=list(flips))
            t = g.construct(item['offset'], perm, flips, [0, 0, 0])
            crop = views.cube_crop(parent, item)
            q = g.inverse(v, t)
            np.testing.assert_array_equal(q, np.round(q))
            np.testing.assert_array_equal(g.points(q, t), v)
            qi = q.astype(np.int64)
            vi = v.astype(np.int64)
            np.testing.assert_array_equal(crop[vi[:, 0], vi[:, 1], vi[:, 2]], parent[qi[:, 0], qi[:, 1], qi[:, 2]])
            # The Vesuvius transforms' own keypoint rule maps crop points to the same view coordinates.
            out = views.TRANSPOSE.apply({'keypoints': torch.from_numpy(q - item['offset'])},
                                        axis_order=[0, *(a + 1 for a in perm)])
            out = views.MIRROR.apply(out, axes=[a for a, f in enumerate(flips) if f], crop_shape=(128,) * 3)
            np.testing.assert_array_equal(out['keypoints'].numpy(), v)

    def test_valid(self):
        def t(offset):
            return g.construct(offset, (0, 1, 2), (0, 0, 0), [0, 0, 0])
        self.assertTrue(g.valid(t([64, 64, 64]), np.array([128., 128., 128.])))
        self.assertTrue(g.valid(t([128, 64, 64]), np.array([192., 128., 128.])))
        self.assertFalse(g.valid(t([129, 64, 64]), np.array([192., 128., 128.])))  # crop leaves the parent
        self.assertFalse(g.valid(t([126, 64, 64]), np.array([128., 128., 128.])))  # anchor 2 voxels from the view edge


class RotatedViews(unittest.TestCase):
    def context(self, a, c):
        z = torch.arange(320, dtype=torch.float32) - 32  # context voxel k <-> parent coordinate k - 32
        return (a[0] * z[:, None, None] + a[1] * z[None, :, None] + a[2] * z[None, None, :] + c)[None, None]

    def test_linear_field(self):
        """The resampled view of a linear field f(q) equals f(inverse(v)) at every view voxel v (trilinear is exact)."""
        a, c = (1e-3, 2e-3, -1.5e-3), .4
        x = self.context(a, c)
        v = np.stack(np.meshgrid(*[np.arange(128.)] * 3, indexing='ij'), -1).reshape(-1, 3)
        for offset, perm, flips, angles in (([64, 64, 64], (2, 0, 1), (1, 0, 1), [20., 30., 40.]),
                                            ([40, 70, 90], (0, 1, 2), (0, 0, 0), [0., 0., 0.])):
            t = g.construct(offset, perm, flips, angles)
            out = views.resample(x[0], t['pull_matrix'], t['input_center'] + 32)[0].numpy().reshape(-1)
            ref = g.inverse(v, t) @ np.array(a) + c
            np.testing.assert_allclose(out, ref, rtol=0, atol=2e-5)

    def test_discrete_equals_cube_crop(self):
        rng = np.random.default_rng(1)
        parent = rng.integers(0, 256, (256,) * 3, dtype=np.uint8)
        context = np.zeros((320,) * 3, np.float32)
        context[32:288, 32:288, 32:288] = parent / np.float32(255)
        item = dict(offset=[30, 77, 101], permutation=[1, 2, 0], flips=[True, False, True])
        t = g.construct(item['offset'], item['permutation'], item['flips'], [0, 0, 0])
        out = views.resample(torch.from_numpy(context)[None], t['pull_matrix'], t['input_center'] + 32)[0].numpy()
        np.testing.assert_allclose(out, views.cube_crop(parent, item) / np.float32(255), rtol=0, atol=1e-4)

    def test_stage4_frame_linear_field(self):
        """A rotated, zoomed stage-4 frame samples the parent256 where to_parent puts each view voxel, and to_view
        inverts it (the label coordinates of the view)."""
        a, c = (2e-3, -1e-3, 1.5e-3), .3
        z = torch.arange(256, dtype=torch.float32)
        x = (a[0] * z[:, None, None] + a[1] * z[None, :, None] + a[2] * z[None, None, :] + c)[None]
        t = g.construct([40, 20, 70], (1, 2, 0), (True, False, True), [0, 0, 0])
        f = rotation_aug.frame(t, rotation_aug.uniform_rotation(np.random.default_rng(5)), 1.1)
        v = np.stack(np.meshgrid(*[np.arange(128.)] * 3, indexing='ij'), -1).reshape(-1, 3)
        out = views.resample(x, f['M'], f['center'])[0].numpy().reshape(-1)
        np.testing.assert_allclose(out, rotation_aug.to_parent(v, f) @ np.array(a) + c, rtol=0, atol=2e-5)
        plan_v = g.points(rotation_aug.to_parent(v[::997], f), t)
        np.testing.assert_allclose(rotation_aug.to_view(plan_v, f), v[::997], rtol=0, atol=1e-9)


class Losses(unittest.TestCase):
    def test_point_weights(self):
        labels = np.array([0, 1, 0, 1, 1])
        ids = np.array([0, 0, 0, 1, 2, 2, 3, 4, 4, 4])
        w, unique = losses.point_weights(ids, labels)
        np.testing.assert_array_equal(unique, [0, 1, 2, 3, 4])
        self.assertAlmostEqual(w[labels[ids] == 0].sum(), .5)
        self.assertAlmostEqual(w[labels[ids] == 1].sum(), .5)
        for i in unique:  # equal share per point of a class, split over its occurrences
            self.assertAlmostEqual(w[ids == i].sum(), .5 / np.sum(labels == labels[i]))

    def test_balanced_point_bce(self):
        ids = torch.tensor([3, 3, 5, 7, 7, 7])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        logits = torch.tensor([.3, -1., 2., .5, -.2, 1.])
        loss, unique = losses.balanced_point_bce(logits, ids, labels)
        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        ref = .5 * ((bce[:2].mean() + bce[2]) / 2 + bce[3:].mean())
        torch.testing.assert_close(loss, ref)
        self.assertEqual(unique.tolist(), [3, 5, 7])

    def test_weak_bce_one_class(self):
        ids = torch.tensor([1, 1, 2])
        labels = torch.tensor([1, 1, 1])
        logits = torch.tensor([.1, .2, -.3])
        loss, n = losses.weak_bce(logits, ids, labels)
        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        torch.testing.assert_close(loss, .5 * (bce[:2].mean() + bce[2]) / 2)
        self.assertEqual(n, 2)


if __name__ == '__main__':
    unittest.main()
