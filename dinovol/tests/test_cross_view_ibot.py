from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from scipy import ndimage as ndi

from dinovol_2.augmentation.pipelines.training_transforms import create_training_transforms
from dinovol_2.augmentation.transforms.spatial.mirroring import MirrorTransform
from dinovol_2.augmentation.transforms.spatial.rot90 import Rot90Transform
from dinovol_2.augmentation.transforms.spatial.transpose import TransposeAxesTransform
from dinovol_2.augmentation.transforms.utils.compose import ComposeTransforms
from dinovol_2.dataset.ssl_zarr_dataset import SampledCropRegion, SSLZarrDataset
from dinovol_2.model.model import DinoVitStudentTeacher
from dinovol_2.pretrain import DinoIBOTPretrainer
import tests.test_object_points as object_point_tests


def _flatten(transform):
    inner = getattr(transform, "transform", None)
    if inner is not None:
        return _flatten(inner)
    children = getattr(transform, "list_of_transforms", None) or getattr(transform, "transforms", None)
    if children:
        return [leaf for child in children for leaf in _flatten(child)]
    return [transform]


class CrossViewTokenTargetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.volume = object_point_tests._make_volume(self.root)

    def tearDown(self):
        self.tempdir.cleanup()

    def dataset_config(self, **overrides):
        return {
            "epoch_length": 4, "vol_trim_pct": 1.0, "nonzero_threshold": 0.3,
            "global_crop_size": [32, 32, 32], "local_crop_size": [16, 16, 16],
            "global_crop_scale": [0.5, 1.0], "local_crop_scale": [0.3, 0.6],
            "num_local_crops": 1, "source_sampling_size": [64, 64, 64],
            "datasets": [self.volume], "normalization_scheme": "none",
            "cross_view_token_patch_size": [8, 8, 8],
            **overrides,
        }

    def check_targets(self, sample):
        targets = sample["global_token_targets"]
        self.assertEqual(tuple(targets.shape), (2, 64, 3))
        axes = [torch.arange(4, dtype=torch.float32) * 8 + 3.5] * 3
        centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).numpy()
        matched = 0
        for view in range(2):
            target = targets[view].numpy()
            interior = np.all((target > 1) & (target < 30), axis=1)
            matched += int(interior.sum())
            # The ramp is affine in source position: a token and its target in the other
            # view must read the same value.
            here = ndi.map_coordinates(sample["global_views"][view][0].numpy(), centers[interior].T, order=1)
            there = ndi.map_coordinates(sample["global_views"][1 - view][0].numpy(), target[interior].T, order=1)
            np.testing.assert_allclose(here, there, atol=1e-2)
        self.assertGreater(matched, 0)

    def test_targets_follow_crops_rotations_flips_and_transposes(self):
        spatial = ComposeTransforms([
            Rot90Transform(allowed_axes={0, 1, 2}),
            MirrorTransform(allowed_axes=(0, 1, 2)),
            TransposeAxesTransform(allowed_axes={0, 1, 2}),
        ])
        for overrides in ({}, {"object_points": {
            "enabled": True, "points_per_sample": 96, "max_labeled_points": 48,
            "points_per_object": 6, "max_objects": 12, "slab_thickness": 8,
        }}):
            dataset = SSLZarrDataset(self.dataset_config(**overrides), do_augmentations=True)
            dataset.global_transforms = [spatial, spatial]
            for seed in range(4):
                np.random.seed(seed)
                torch.manual_seed(seed)
                self.check_targets(dataset[0])
            dataset.close()

    def test_targets_reject_augmentations_that_move_token_centers(self):
        dataset = SSLZarrDataset(self.dataset_config(), do_augmentations=True)
        original = dataset._apply_paired_global_transform_with_points

        def shifted(transform, global_view, gram_teacher_view, keypoints):
            view, gram, moved = original(transform, global_view, gram_teacher_view, keypoints)
            return view, gram, torch.as_tensor(moved) + 0.3

        dataset._apply_paired_global_transform_with_points = shifted
        with self.assertRaises(ValueError):
            dataset[0]
        dataset.close()

    def test_teacher_tokens_are_read_from_the_other_view(self):
        tokens = torch.randn(4, 64, 5)
        axes = [torch.arange(4, dtype=torch.float32) * 8 + 3.5] * 3
        centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(1, -1, 3).expand(4, -1, -1)
        branch = torch.nn.ModuleDict({"backbone": torch.nn.Module()})
        branch.backbone.patch_size = (8, 8, 8)
        rows = torch.tensor([2, 3, 0, 1])
        permutation = torch.randperm(64)
        result = DinoVitStudentTeacher._cross_view_tokens(branch, tokens, centers[:, permutation], rows, (32, 32, 32))
        torch.testing.assert_close(result, tokens[rows][:, permutation])


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.volume = object_point_tests._make_volume(self.root)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_full_preset_has_every_family_and_artifacts_before_reorientation(self):
        leaves = _flatten(create_training_transforms((32, 32, 32), gaussian_blur=True))
        names = [type(leaf).__name__ for leaf in leaves]
        for expected in (
            "GaussianBlurTransform", "GaussianNoiseTransform", "SharpeningTransform", "ContrastTransform",
            "MultiplicativeBrightnessTransform", "BrightnessAdditiveTransform", "SimulateLowResolutionTransform",
            "GammaTransform", "InvertImageTransform", "SmearTransform", "InhomogeneousSliceIlluminationTransform",
            "BrightnessGradientAdditiveTransform", "LocalGammaTransform",
            "Rot90Transform", "MirrorTransform", "TransposeAxesTransform",
        ):
            self.assertIn(expected, names)
        first_reorientation = min(names.index(n) for n in ("Rot90Transform", "MirrorTransform", "TransposeAxesTransform"))
        self.assertLess(names.index("SmearTransform"), first_reorientation)
        self.assertLess(names.index("InhomogeneousSliceIlluminationTransform"), first_reorientation)
        for leaf in leaves:
            if isinstance(leaf, (Rot90Transform, TransposeAxesTransform)):
                self.assertEqual(set(leaf.allowed_axes), {0, 1, 2})

    def test_each_global_crop_gets_its_own_augmentation(self):
        config = {
            "epoch_length": 4, "vol_trim_pct": 1.0, "nonzero_threshold": 0.3,
            "global_crop_size": [32, 32, 32], "local_crop_size": [16, 16, 16],
            "global_crop_scale": [0.5, 1.0], "local_crop_scale": [0.3, 0.6],
            "num_local_crops": 2, "source_sampling_size": [64, 64, 64], "datasets": [self.volume],
            "augmentation": "full", "gaussian_blur": True,
        }
        dataset = SSLZarrDataset(config, do_augmentations=True)
        # Both global crops come from the same region: only their augmentations can differ.
        dataset._sample_global_crop_region = lambda *args, **kwargs: SampledCropRegion((8, 8, 8), (40, 40, 40))
        for seed in range(6):
            np.random.seed(seed)
            torch.manual_seed(seed)
            sample = dataset[0]
            first, second = sample["global_views"]
            self.assertFalse(torch.equal(first, second), f"seed {seed}: both global crops augmented identically")
            self.assertFalse(torch.equal(*sample["local_views"]))
        dataset.close()


class CrossViewTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.volume = object_point_tests._make_volume(self.root)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_training_step_with_cross_view_targets(self):
        torch.manual_seed(0)
        np.random.seed(0)
        config = object_point_tests.ObjectLossTrainingTests.config(self, objects=True, output="cross_view")
        config["ibot_targets"] = "cross_view"
        config["mask_sample_probability"] = 1.0
        trainer = DinoIBOTPretrainer(config)
        loader = trainer.build_dataloader()
        try:
            iterator = iter(loader)
            batches = [next(iterator), next(iterator)]
            self.assertEqual(tuple(batches[0]["global_token_targets"].shape), (4, 64, 3))
            metrics = trainer.train_step(batches, 0)
            for key in ("loss", "ibot_loss", "object_point_loss"):
                self.assertTrue(np.isfinite(metrics[key]), key)
            self.assertGreater(metrics["ibot_target_fraction"], 0.0)
            self.assertLessEqual(metrics["ibot_target_fraction"], 1.0)
            self.assertGreater(metrics["ibot_loss"], 0.0)
        finally:
            trainer._close_dataloader(loader)

    def test_validation_loader_has_no_token_targets(self):
        config = object_point_tests.ObjectLossTrainingTests.config(self, objects=True, output="cross_view_val")
        config["ibot_targets"] = "cross_view"
        trainer = DinoIBOTPretrainer(config)
        self.assertIn("cross_view_token_patch_size", trainer._prepare_dataset_config(config["dataset"], enable_point_supervision=True))
        self.assertNotIn("cross_view_token_patch_size", trainer._prepare_dataset_config(config["dataset"]))


if __name__ == "__main__":
    unittest.main()
