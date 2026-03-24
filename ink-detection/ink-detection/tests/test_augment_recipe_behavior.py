from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from ink.recipes.augment import TrainAugment
from ink.recipes.data.normalization import (
    ClipMaxDiv255Normalization,
    FoldForegroundClipZScoreNormalization,
)
from ink.recipes.data.transforms import (
    apply_eval_sample_transforms,
    apply_train_sample_transforms,
    build_joint_transform,
)

class TrainAugmentContractTests(unittest.TestCase):
    def test_build_binds_patch_size_and_drops_legacy_config_surface(self):
        recipe = TrainAugment()

        bound = recipe.build(patch_size=256)

        self.assertEqual(bound.size, 256)
        self.assertFalse(hasattr(bound, "config"))
        self.assertFalse(hasattr(bound, "normalization_mode"))

    def test_build_train_ops_returns_only_augment_policy(self):
        recipe = TrainAugment(
            size=16,
            horizontal_flip_p=0.2,
            vertical_flip_p=0.0,
            brightness_contrast_p=0.5,
            brightness_limit=(0.1, 0.3),
            contrast_limit=(0.0, 0.25),
            brightness_by_max=False,
            random_gamma_p=0.4,
            random_gamma_limit=(70, 130),
            multiplicative_noise_p=1.0,
            multiplicative_noise_multiplier=(0.8, 1.2),
            multiplicative_noise_per_channel=True,
            shift_scale_rotate_p=0.6,
            rotate_limit=45,
            shift_limit=0.1,
            scale_limit=0.2,
            blur_p=0.2,
            use_motion_blur=False,
            coarse_dropout_p=0.3,
            coarse_dropout_max_holes=4,
            coarse_dropout_max_width_ratio=0.25,
            coarse_dropout_max_height_ratio=0.5,
        )

        ops = recipe.build_train_ops()

        self.assertEqual(
            [type(op).__name__ for op in ops],
            [
                "HorizontalFlip",
                "VerticalFlip",
                "RandomBrightnessContrast",
                "RandomGamma",
                "MultiplicativeNoise",
                "ShiftScaleRotate",
                "OneOf",
                "CoarseDropout",
            ],
        )
        self.assertAlmostEqual(ops[0].p, 0.2)
        self.assertAlmostEqual(ops[1].p, 0.0)
        self.assertEqual([type(op).__name__ for op in ops[6].transforms], ["GaussNoise", "GaussianBlur"])

    def test_data_layer_builds_joint_transform_with_resize_normalization_and_tensor(self):
        transform = build_joint_transform(
            "train",
            augment=TrainAugment(size=16),
            normalization=ClipMaxDiv255Normalization(),
            patch_size=16,
            in_channels=4,
        )

        self.assertEqual(
            [type(op).__name__ for op in transform.transforms],
            [
                "Resize",
                "HorizontalFlip",
                "VerticalFlip",
                "RandomBrightnessContrast",
                "RandomGamma",
                "MultiplicativeNoise",
                "ShiftScaleRotate",
                "OneOf",
                "CoarseDropout",
                "Normalize",
                "ToTensorV2",
            ],
        )

    def test_eval_sample_transforms_apply_data_owned_normalization_and_label_resize(self):
        stats = {
            "percentile_00_5": 2.0,
            "percentile_99_5": 10.0,
            "mean": 6.0,
            "std": 2.0,
        }
        image = np.stack(
            [
                np.linspace(0.0, 14.0, num=64, dtype=np.float32).reshape(8, 8),
                np.linspace(2.0, 16.0, num=64, dtype=np.float32).reshape(8, 8),
            ],
            axis=-1,
        )
        label = np.full((8, 8, 1), 255, dtype=np.uint8)
        transform = build_joint_transform(
            "valid",
            augment=TrainAugment(size=8),
            normalization=FoldForegroundClipZScoreNormalization(stats=stats),
            patch_size=8,
            in_channels=2,
        )

        image_out, label_out = apply_eval_sample_transforms(
            image,
            label,
            patch_size=8,
            transform=transform,
        )

        expected = image.astype(np.float32).copy()
        np.clip(expected, 2.0, 10.0, out=expected)
        expected -= 6.0
        expected /= 2.0

        self.assertEqual(tuple(image_out.shape), (1, 2, 8, 8))
        self.assertEqual(tuple(label_out.shape), (1, 2, 2))
        torch.testing.assert_close(image_out.squeeze(0), torch.from_numpy(expected.transpose(2, 0, 1)))
        torch.testing.assert_close(label_out, torch.ones((1, 2, 2), dtype=torch.float32))

    def test_train_sample_transforms_apply_fourth_and_invert_before_joint_transform(self):
        image = np.arange(2 * 2 * 4, dtype=np.uint8).reshape(2, 2, 4)
        label = np.ones((2, 2, 1), dtype=np.float32)
        augment = TrainAugment(
            size=8,
            fourth_augment_p=1.0,
            fourth_augment_min_crop_ratio=0.5,
            fourth_augment_max_crop_ratio=0.5,
            fourth_augment_cutout_max_count=0,
            fourth_augment_cutout_p=0.0,
            invert_p=1.0,
        )

        with (
            mock.patch("ink.recipes.augment.default.random.random", side_effect=[0.0, 1.0, 0.0]),
            mock.patch("ink.recipes.augment.default.random.randint", side_effect=[2, 1, 0, 0]),
            mock.patch("ink.recipes.augment.default.np.random.shuffle", side_effect=lambda _: None),
            mock.patch(
                "ink.recipes.data.transforms._apply_joint_transform",
                side_effect=lambda transform, image, label, *, patch_size: (image, label),
            ),
        ):
            image_out, label_out = apply_train_sample_transforms(
                image,
                label,
                augment=augment,
                patch_size=8,
                transform="joint",
            )

        expected_fourth = np.zeros_like(image)
        expected_fourth[..., 0:2] = image[..., 1:3]
        expected = (expected_fourth.min().astype(np.int64) + expected_fourth.max().astype(np.int64)) - expected_fourth.astype(np.int64)
        expected = expected.astype(np.uint8)

        np.testing.assert_array_equal(image_out, expected)
        np.testing.assert_array_equal(label_out, label)


if __name__ == "__main__":
    unittest.main()
