"""CT/presence blur contracts; also runnable with the standard unittest runner."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.regression.data import (
    IdentityObservationBuilder, IdentitySampling, ObservationBuilder,
    augment_image_pair, photometric,
)
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig
from vesuvius.neural_tracing.fiber_follow.regression.train import build_parser


class BlurAugmentationTests(unittest.TestCase):
    def test_shared_blur_preserves_channel_alignment_and_mass(self):
        image = torch.zeros(2, 17, 17, 17)
        image[:, 8, 8, 8] = 1
        augment_image_pair(image, (1., 0., 0.), np.random.default_rng(1), blur_sigma=1.)
        # CT's existing mean-centered photometric calculation adds FP32 rounding.
        torch.testing.assert_close(image[0], image[1], rtol=0, atol=1e-8)
        self.assertGreater(image[0, 8, 8, 9].item(), 0.)
        self.assertLess(image[0, 8, 8, 8].item(), 1.)
        torch.testing.assert_close(image.sum((1, 2, 3)), torch.ones(2))
        self.assertEqual(image[0].argmax().item(), 8 * 17 * 17 + 8 * 17 + 8)

    def test_channels_do_not_mix_and_constant_boundaries_stay_constant(self):
        image = torch.zeros(2, 9, 9, 9)
        image[1].fill_(.75)
        augment_image_pair(image, (1., 0., 0.), np.random.default_rng(2), blur_sigma=1.25)
        self.assertEqual(image[0].count_nonzero().item(), 0)
        torch.testing.assert_close(image[1], torch.full_like(image[1], .75))

    def test_disabled_blur_preserves_existing_photometric_values_and_rng(self):
        image = torch.from_numpy(np.random.default_rng(0).random((2, 9, 9, 9), dtype=np.float32))
        expected = image.clone()
        a, b = np.random.default_rng(3), np.random.default_rng(3)
        params = (1.2, -.05, .02)
        expected[0] = torch.from_numpy(photometric(expected[0].numpy(), params, a))
        augment_image_pair(image, params, b, blur_sigma=0.)
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
        self.assertEqual(a.random(), b.random())

    def test_presence_dropout_remains_exactly_zero_after_blur(self):
        image = torch.ones(2, 9, 9, 9)
        augment_image_pair(image, (1., 0., 0.), np.random.default_rng(4),
                           blur_sigma=1., drop_presence=True)
        self.assertEqual(image[1].count_nonzero().item(), 0)
        torch.testing.assert_close(image[0], torch.ones_like(image[0]))

    def test_defaults_and_invalid_settings(self):
        self.assertEqual(IdentitySampling().blur_probability, .25)
        self.assertEqual(build_parser().get_default('blur_probability'), .25)
        self.assertEqual(IdentitySampling(blur_sigma=[.5, 1.25]).blur_sigma, (.5, 1.25))
        for kwargs in (dict(blur_probability=-.1), dict(blur_probability=1.1),
                       dict(blur_probability=float('nan')), dict(blur_sigma=(-1., 1.)),
                       dict(blur_sigma=(2., 1.)), dict(blur_sigma=(0., float('inf')))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                IdentitySampling(**kwargs)

    def test_builder_blurs_current_memory_and_seed_without_changing_labels(self):
        cfg = DirectConfig(memory_slots=2, memory_steps=2, memory_patch_size=9)
        builder = IdentityObservationBuilder(cfg, augment=True)
        pair = torch.zeros(2, 9, 9, 9)
        pair[:, 4, 4, 4] = 1
        padding = torch.from_numpy(np.random.default_rng(2).random(pair.shape, dtype=np.float32))
        original = dict(x=dict(fine=pair[None].clone(), seed_mask=torch.ones(1, 1),
                              memory_patches=torch.stack((pair, padding))[None].clone(),
                              memory_mask=torch.tensor([[True, False]]),
                              memory_seed_patch=pair[None].clone(), memory_seed_valid=torch.tensor([True])),
                        dense_ab=torch.tensor([[[1., 2.]]]))
        training_batch = copy.deepcopy(original)
        item = dict(photometric=(1., 0., 0.), identity_seed=1, blur_sigma=1., drop_presence=False)
        def targets(items):
            # Supervision is constructed before image augmentation.
            self.assertEqual(training_batch['x']['fine'][0, 0, 4, 4, 4].item(), 1.)
            return dict(presence_dropped=torch.zeros(1))
        with patch.object(ObservationBuilder, '__call__', return_value=training_batch), \
             patch.object(builder, 'identity_targets', side_effect=targets), \
             patch('vesuvius.neural_tracing.fiber_follow.regression.memory_data.memory_targets', return_value={}):
            result = builder([item], None)
        x = result['x']
        self.assertEqual(result['blurred'].tolist(), [1.])
        self.assertLess(x['fine'][0, 0, 4, 4, 4].item(), 1.)
        torch.testing.assert_close(x['fine'][0], x['memory_patches'][0, 0], rtol=0, atol=0)
        torch.testing.assert_close(x['fine'][0], x['memory_seed_patch'][0], rtol=0, atol=0)
        torch.testing.assert_close(x['memory_patches'][0, 1], padding, rtol=0, atol=0)
        torch.testing.assert_close(result['dense_ab'], original['dense_ab'], rtol=0, atol=0)
        # Evaluation takes the ordinary unaugmented path even with blur defaults.
        builder.augment = False
        with patch.object(ObservationBuilder, '__call__', return_value=copy.deepcopy(original)):
            evaluated = builder([item], None)
        torch.testing.assert_close(evaluated['x']['fine'], original['x']['fine'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
