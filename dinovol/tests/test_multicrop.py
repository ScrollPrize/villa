import random
import unittest
import numpy as np
import torch

from dinovol_2.ops.collate import collate_multicrop_batch
from dinovol_2.methods.dino_data import build_dino_ibot_collate_fn
from dinovol_2.ops.point_embeddings import interpolation_support_indices


class MultiCropTests(unittest.TestCase):
    def samples(self, global_count=3, local_count=2):
        return [dict(global_views=[torch.full((1,8,8,8), 10*v+s) for v in range(global_count)],
                     local_views=[torch.full((1,4,4,4), 100+10*v+s) for v in range(local_count)]) for s in range(2)]

    def test_view_order_and_independent_sizes(self):
        batch = collate_multicrop_batch(self.samples())
        self.assertEqual(batch['collated_global_crops'].shape, (6,1,8,8,8))
        self.assertEqual(batch['collated_local_crops'].shape, (4,1,4,4,4))
        self.assertEqual(batch['collated_global_crops'][:,0,0,0,0].tolist(), [0,1,10,11,20,21])
        self.assertEqual(batch['collated_local_crops'][:,0,0,0,0].tolist(), [100,101,110,111])
        self.assertNotIn('collated_masks', batch)

    def test_zero_locals_and_bad_view_counts(self):
        batch = collate_multicrop_batch(self.samples(local_count=0))
        self.assertEqual(batch['n_local_views'], 0)
        self.assertEqual(batch['collated_local_crops'].shape[0], 0)
        samples = self.samples()
        samples[1]['global_views'].pop()
        with self.assertRaisesRegex(ValueError, 'same global/local'):
            collate_multicrop_batch(samples)

    def test_ibot_protects_points_and_preserves_crop_order(self):
        samples = self.samples(global_count=2)
        for sample in samples:
            sample['global_point_coordinates'] = [torch.tensor([[1.5,1.5,1.5]])]*2
            sample['global_point_type_ids'] = [torch.tensor([7])]*2
        random.seed(5); np.random.seed(5)
        batch = build_dino_ibot_collate_fn(dict(global_crop_size=[8]*3,patch_size=[4]*3,
            mask_ratio_min_max=(1.,1.),mask_sample_probability=1.))(samples)
        support = interpolation_support_indices(batch['collated_point_coordinates'], (4,4,4), (2,2,2))
        self.assertFalse(batch['collated_masks'][batch['collated_point_rows'][:,None], support].any())
        self.assertEqual(batch['collated_point_rows'].tolist(), [0,1,2,3])
        expected = collate_multicrop_batch(samples)
        torch.testing.assert_close(batch['collated_global_crops'],expected['collated_global_crops'])
        indices = batch['mask_indices_list']
        torch.testing.assert_close(indices, batch['collated_masks'].flatten().nonzero().flatten())
        for row in range(4):
            weights = batch['masks_weight'][indices // 8 == row]
            if weights.numel(): self.assertAlmostEqual(float(weights.sum()),1.)
