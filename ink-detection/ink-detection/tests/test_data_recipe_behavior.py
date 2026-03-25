from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr
from torch.utils.data import DataLoader

from ink.core import DataBundle, Experiment, assemble_experiment, build_experiment_data
from ink.recipes.augment import TrainAugment
from ink.recipes.data.normalization import (
    ClipMaxDiv255Normalization,
    FoldForegroundClipZScoreNormalization,
)
from ink.recipes.data.patch_bundle import (
    GeneratedPatchBundleDataRecipe,
    PatchBundleWriter,
)
import ink.recipes.data.zarr_io as zarr_io_module
import ink.recipes.data.zarr_data as zarr_data_module
from ink.recipes.data.zarr_data import ZarrPatchDataRecipe
from ink.recipes.trainers import PatchTrainer
from tests.support.build_recipes import StaticBuildRecipe


def _neutral_augment(size: int = 8) -> TrainAugment:
    return TrainAugment(
        size=size,
        horizontal_flip_p=0.0,
        vertical_flip_p=0.0,
        brightness_contrast_p=0.0,
        random_gamma_p=0.0,
        multiplicative_noise_p=0.0,
        shift_scale_rotate_p=0.0,
        blur_p=0.0,
        coarse_dropout_p=0.0,
        fourth_augment_p=0.0,
        invert_p=0.0,
    )


def _write_zarr_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(path), mode="w", shape=tuple(array.shape), dtype=array.dtype)
    store[:] = array


def _make_canonical_segment(
    root: Path,
    *,
    group_name: str,
    segment_id: str,
    volume: np.ndarray,
    label: np.ndarray,
    supervision_mask: np.ndarray,
    validation_mask: np.ndarray | None = None,
    write_validation_mask: bool = True,
    dataset_suffix: str = "",
) -> dict[str, object]:
    segment_dir = root / group_name / segment_id
    _write_zarr_array(segment_dir / f"{segment_id}.zarr", np.asarray(volume))
    _write_zarr_array(segment_dir / f"{segment_id}_inklabels{dataset_suffix}.zarr", np.asarray(label))
    _write_zarr_array(segment_dir / f"{segment_id}_supervision_mask{dataset_suffix}.zarr", np.asarray(supervision_mask))
    if write_validation_mask:
        _write_zarr_array(
            segment_dir / f"{segment_id}_validation_mask{dataset_suffix}.zarr",
            np.asarray(supervision_mask if validation_mask is None else validation_mask),
        )
    return {}


class ZarrPatchDataRecipeTests(unittest.TestCase):
    def test_zarr_patch_data_uses_dataset_version_for_labels_and_split_specific_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
                validation_mask=np.pad(
                    np.full((4, 4), 255, dtype=np.uint8),
                    ((0, 4), (0, 4)),
                    mode="constant",
                    constant_values=0,
                ),
                dataset_suffix="_v2",
            )

            recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=4,
                tile_size=4,
                stride=4,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
                dataset_version="v2",
            )

            bundle = recipe.build(augment=_neutral_augment(size=4))
            train_batch = next(iter(bundle.train_loader))
            val_batch = next(iter(bundle.eval_loader))

            self.assertEqual(recipe.dataset_version, "v2")
            self.assertEqual(recipe.label_suffix, "_v2")
            self.assertEqual(recipe.mask_suffix, "_v2")
            self.assertEqual(bundle.train_loader.dataset.mask_name, "supervision_mask")
            self.assertEqual(bundle.eval_loader.dataset.mask_name, "validation_mask")
            self.assertEqual(len(bundle.train_loader.dataset), 4)
            self.assertEqual(len(bundle.eval_loader.dataset), 1)
            self.assertEqual(train_batch.meta.patch_xyxy[0].tolist(), [0, 0, 4, 4])
            self.assertEqual(val_batch.meta.patch_xyxy[0].tolist(), [0, 0, 4, 4])
            self.assertTrue(bool(np.allclose(train_batch.meta.valid_mask.numpy(), 1.0)))
            self.assertTrue(bool(np.allclose(val_batch.meta.valid_mask.numpy(), 1.0)))

    def test_zarr_patch_data_val_only_segments_union_supervision_and_validation_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segTrain",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segVal",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
                validation_mask=np.pad(
                    np.full((4, 4), 255, dtype=np.uint8),
                    ((0, 4), (0, 4)),
                    mode="constant",
                    constant_values=0,
                ),
            )

            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segTrain": train_segment,
                    "segVal": val_segment,
                },
                train_segment_ids=("segTrain",),
                val_segment_ids=("segVal",),
                in_channels=4,
                patch_size=4,
                tile_size=4,
                stride=4,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            ).build(augment=_neutral_augment(size=4))

            self.assertEqual(len(bundle.eval_loader.dataset), 4)

    def test_zarr_patch_data_val_only_segment_without_validation_mask_uses_supervision_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segTrain",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segVal",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
                write_validation_mask=False,
            )

            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segTrain": train_segment,
                    "segVal": val_segment,
                },
                train_segment_ids=("segTrain",),
                val_segment_ids=("segVal",),
                in_channels=4,
                patch_size=4,
                tile_size=4,
                stride=4,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            ).build(augment=_neutral_augment(size=4))

            self.assertEqual(len(bundle.eval_loader.dataset), 4)

    def test_zarr_patch_data_uses_latest_available_label_and_mask_versions_when_unspecified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(
                segment_dir / "segA.zarr",
                np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
            )
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", np.zeros((8, 8), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels_v2.zarr", np.full((8, 8), 255, dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((8, 8), 255, dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_supervision_mask_v2.zarr", np.full((8, 8), 255, dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_validation_mask.zarr", np.full((8, 8), 255, dtype=np.uint8))
            _write_zarr_array(
                segment_dir / "segA_validation_mask_v2.zarr",
                np.pad(
                    np.full((4, 4), 255, dtype=np.uint8),
                    ((0, 4), (0, 4)),
                    mode="constant",
                    constant_values=0,
                ),
            )

            recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": {}},
                train_segment_ids=("segA",),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=4,
                tile_size=4,
                stride=4,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            )

            bundle = recipe.build(augment=_neutral_augment(size=4))
            train_batch = next(iter(bundle.train_loader))

            self.assertEqual(recipe.dataset_version, "")
            self.assertEqual(recipe.label_suffix, "")
            self.assertEqual(recipe.mask_suffix, "")
            self.assertEqual(len(bundle.eval_loader.dataset), 1)
            self.assertTrue(bool(np.allclose(train_batch.y.numpy(), 1.0)))

    def test_zarr_patch_data_builds_real_loaders_with_bound_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segA": train_segment,
                    "segB": val_segment,
                },
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=FoldForegroundClipZScoreNormalization(),
                extras={
                    "normalization_stats": {
                        "percentile_00_5": 2.0,
                        "percentile_99_5": 10.0,
                        "mean": 6.0,
                        "std": 2.0,
                    },
                },
            )

            bundle = recipe.build(augment=_neutral_augment())

            self.assertIsInstance(bundle, DataBundle)
            self.assertIsInstance(bundle.train_loader, DataLoader)
            self.assertIsInstance(bundle.eval_loader, DataLoader)
            self.assertEqual(bundle.in_channels, 4)
            self.assertIsInstance(bundle.augment, TrainAugment)
            self.assertEqual(bundle.train_loader.dataset.normalization.stats["mean"], 6.0)
            self.assertEqual(bundle.eval_loader.dataset.normalization.stats["mean"], 6.0)
            self.assertEqual(bundle.extras, {})
            self.assertNotIn("stitch", bundle.extras)
            self.assertEqual(len(bundle.train_loader), 1)
            self.assertEqual(len(bundle.eval_loader), 1)

    def test_zarr_patch_data_loaders_use_flat_train_and_val_contracts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 0, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segA": train_segment,
                    "segB": val_segment,
                },
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            )

            bundle = recipe.build(augment=_neutral_augment())
            train_batch = next(iter(bundle.train_loader))
            val_batch = next(iter(bundle.eval_loader))

            train_x = train_batch.x
            train_y = train_batch.y
            train_valid_mask = train_batch.meta.valid_mask
            val_x = val_batch.x
            val_y = val_batch.y
            val_valid_mask = val_batch.meta.valid_mask
            val_xyxy = val_batch.meta.patch_xyxy

            self.assertEqual(tuple(train_x.shape), (1, 1, 4, 8, 8))
            self.assertEqual(tuple(train_y.shape), (1, 1, 2, 2))
            self.assertEqual(tuple(train_valid_mask.shape), (1, 1, 2, 2))
            self.assertAlmostEqual(float(train_x[0, 0, 0, 0, 0].item()), 10.0 / 255.0, places=5)
            self.assertTrue(bool(np.allclose(train_valid_mask.numpy(), 1.0)))

            self.assertEqual(tuple(val_x.shape), (1, 1, 4, 8, 8))
            self.assertEqual(tuple(val_y.shape), (1, 1, 2, 2))
            self.assertEqual(tuple(val_valid_mask.shape), (1, 1, 2, 2))
            self.assertEqual(tuple(val_xyxy.shape), (1, 4))
            self.assertEqual(val_xyxy[0].tolist(), [0, 0, 8, 8])
            self.assertTrue(bool(np.allclose(val_valid_mask.numpy(), 1.0)))

    def test_zarr_patch_data_can_skip_train_valid_mask_while_keeping_eval_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 0, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segA": train_segment,
                    "segB": val_segment,
                },
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
                include_train_valid_mask=False,
            )

            bundle = recipe.build(augment=_neutral_augment())
            train_batch = next(iter(bundle.train_loader))
            val_batch = next(iter(bundle.eval_loader))

            self.assertIsNone(train_batch.meta.valid_mask)
            self.assertEqual(tuple(train_batch.x.shape), (1, 1, 4, 8, 8))
            self.assertEqual(tuple(train_batch.y.shape), (1, 1, 2, 2))
            self.assertIsNotNone(val_batch.meta.valid_mask)
            self.assertEqual(tuple(val_batch.meta.valid_mask.shape), (1, 1, 2, 2))

    def test_zarr_patch_data_skip_train_valid_mask_uses_label_only_reads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": train_segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
                include_train_valid_mask=False,
            )

            bundle = recipe.build(augment=_neutral_augment())
            with mock.patch.object(
                zarr_io_module,
                "read_label_region",
                wraps=zarr_io_module.read_label_region,
            ) as label_read, mock.patch.object(
                zarr_io_module,
                "read_label_and_supervision_mask_region",
                wraps=zarr_io_module.read_label_and_supervision_mask_region,
            ) as full_read:
                train_batch = next(iter(bundle.train_loader))

            self.assertIsNone(train_batch.meta.valid_mask)
            self.assertGreater(label_read.call_count, 0)
            self.assertEqual(full_read.call_count, 0)

    def test_zarr_patch_data_enumerates_patch_coordinates_when_segments_omit_xyxys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segA": train_segment,
                    "segB": val_segment,
                },
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=4,
                tile_size=8,
                stride=4,
                train_batch_size=2,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            )

            bundle = recipe.build(augment=_neutral_augment(size=4))
            train_batch = next(iter(bundle.train_loader))
            val_batch = next(iter(bundle.eval_loader))

            train_x = train_batch.x
            train_y = train_batch.y
            train_valid_mask = train_batch.meta.valid_mask
            val_x = val_batch.x
            val_y = val_batch.y
            val_valid_mask = val_batch.meta.valid_mask
            val_xyxy = val_batch.meta.patch_xyxy

            self.assertEqual(len(bundle.train_loader.dataset), 4)
            self.assertEqual(len(bundle.eval_loader.dataset), 4)
            self.assertEqual(tuple(train_x.shape), (2, 1, 4, 4, 4))
            self.assertEqual(tuple(train_y.shape), (2, 1, 1, 1))
            self.assertEqual(tuple(train_valid_mask.shape), (2, 1, 1, 1))
            self.assertEqual(tuple(val_x.shape), (1, 1, 4, 4, 4))
            self.assertEqual(tuple(val_y.shape), (1, 1, 1, 1))
            self.assertEqual(tuple(val_valid_mask.shape), (1, 1, 1, 1))
            self.assertEqual(val_xyxy[0].tolist(), [0, 0, 4, 4])

    def test_zarr_patch_data_reuses_shared_volume_cache_across_train_and_val(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            )

            bundle = recipe.build(augment=_neutral_augment())

            self.assertIs(bundle.train_loader.dataset._volume_cache, bundle.eval_loader.dataset._volume_cache)
            self.assertEqual(len(bundle.train_loader.dataset._volume_cache), 1)

    def test_zarr_patch_data_can_lazily_cache_train_patches_in_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": train_segment, "segB": val_segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
                cache_train_patches_in_memory=True,
            ).build(augment=_neutral_augment())

            train_dataset = bundle.train_loader.dataset
            eval_dataset = bundle.eval_loader.dataset

            self.assertTrue(train_dataset.cache_patches_in_memory)
            self.assertFalse(eval_dataset.cache_patches_in_memory)
            self.assertEqual(len(train_dataset._volume_cache), 2)
            self.assertEqual(len(train_dataset._patch_cache), 0)

            train_dataset._load_item(0)

            self.assertEqual(len(train_dataset._patch_cache), 1)
            self.assertEqual(len(train_dataset._volume_cache), 2)

    def test_zarr_patch_data_preloads_memory_cached_train_patches_for_worker_processes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": train_segment, "segB": val_segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                num_workers=2,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
                cache_train_patches_in_memory=True,
            ).build(augment=_neutral_augment())

            train_dataset = bundle.train_loader.dataset
            eval_dataset = bundle.eval_loader.dataset

            self.assertEqual(len(train_dataset._patch_cache), len(train_dataset))
            self.assertEqual(len(eval_dataset._patch_cache), 0)
            cached_image, cached_label, cached_valid_mask, cached_xyxy, cached_segment_id = train_dataset._patch_cache[0]
            self.assertTrue(cached_image.flags.writeable)
            self.assertTrue(cached_label.flags.writeable)
            self.assertTrue(cached_valid_mask.flags.writeable)
            self.assertTrue(cached_xyxy.flags.writeable)
            self.assertEqual(cached_segment_id, "segA")

    def test_zarr_patch_data_reads_runtime_masks_from_cached_roi_regions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=4,
                tile_size=8,
                stride=4,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            ).build(augment=_neutral_augment(size=4))

            dataset = bundle.train_loader.dataset
            label_mask_store = dataset._label_mask_store_cache[("segA", "", "", ("supervision_mask",))]
            self.assertEqual(len(label_mask_store._bbox_cache), 0)

            with mock.patch(
                "ink.recipes.data.zarr_data.read_supervision_mask_for_shape",
                side_effect=AssertionError("runtime patch reads should not reload the full supervision mask"),
            ), mock.patch(
                "ink.recipes.data.zarr_io.read_label_and_supervision_mask_region",
                wraps=zarr_io_module.read_label_and_supervision_mask_region,
            ) as region_mock:
                dataset._load_item(0)

            self.assertEqual(region_mock.call_count, 1)
            self.assertEqual(len(label_mask_store._bbox_cache), 1)

    def test_zarr_patch_data_derives_default_train_and_eval_stitch_segments_from_split_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segA": train_segment,
                    "segB": val_segment,
                },
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            )

            bundle = recipe.build(augment=_neutral_augment())

            self.assertNotIn("stitch", bundle.extras)

    def test_zarr_patch_data_derives_group_indices_from_parent_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seg_a = _make_canonical_segment(
                root,
                group_name="0139",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            seg_b = _make_canonical_segment(
                root,
                group_name="0500p2",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segA": seg_a,
                    "segB": seg_b,
                },
                train_segment_ids=("segA", "segB"),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=8,
                train_batch_size=2,
                valid_batch_size=1,
                shuffle=False,
            ).build(augment=_neutral_augment())

            train_batch = next(iter(bundle.train_loader))

            self.assertEqual(train_batch.meta.group_idx.tolist(), [0, 1])
            self.assertIsNone(bundle.group_counts)

    def test_zarr_patch_data_reuses_cached_patch_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_root = root / "patch_index_cache"
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": train_segment, "segB": val_segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=4,
                tile_size=8,
                stride=4,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                patch_index_cache_dir=str(cache_root),
            )

            first_bundle = recipe.build(augment=_neutral_augment(size=4))
            self.assertEqual(len(first_bundle.train_loader.dataset), 4)
            self.assertTrue(any(cache_root.rglob("*.npy")))
            self.assertTrue(any(cache_root.rglob("*.json")))

            with mock.patch(
                "ink.recipes.data.zarr_data.build_patch_index",
                side_effect=AssertionError("patch index cache should avoid re-extracting coordinates"),
            ), mock.patch(
                "ink.recipes.data.zarr_data.read_supervision_mask_for_shape",
                side_effect=AssertionError("patch index cache should avoid re-reading the full supervision mask"),
            ):
                second_bundle = recipe.build(augment=_neutral_augment(size=4))

            self.assertEqual(len(second_bundle.train_loader.dataset), 4)
            self.assertEqual(len(second_bundle.eval_loader.dataset), 4)

    def test_patch_bundle_writer_round_trips_patch_recipe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundle"
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 0, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            source_recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": train_segment, "segB": val_segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            )
            source_bundle = source_recipe.build(augment=_neutral_augment())
            generated_bundle = GeneratedPatchBundleDataRecipe(
                bundle_root=str(bundle_root),
                source=source_recipe,
            ).build(augment=_neutral_augment())
            segment_manifest = yaml.safe_load((bundle_root / "group_a" / "segA" / "manifest.yaml").read_text(encoding="utf-8"))

            source_train = next(iter(source_bundle.train_loader))
            bundle_train = next(iter(generated_bundle.train_loader))
            source_eval = next(iter(source_bundle.eval_loader))
            bundle_eval = next(iter(generated_bundle.eval_loader))

            self.assertTrue(torch.equal(bundle_train.x, source_train.x))
            self.assertTrue(torch.equal(bundle_train.y, source_train.y))
            self.assertTrue(torch.equal(bundle_train.meta.valid_mask, source_train.meta.valid_mask))
            self.assertEqual(bundle_train.meta.segment_ids, source_train.meta.segment_ids)
            self.assertTrue(torch.equal(bundle_train.meta.patch_xyxy, source_train.meta.patch_xyxy))

            self.assertTrue(torch.equal(bundle_eval.x, source_eval.x))
            self.assertTrue(torch.equal(bundle_eval.y, source_eval.y))
            self.assertTrue(torch.equal(bundle_eval.meta.valid_mask, source_eval.meta.valid_mask))
            self.assertEqual(bundle_eval.meta.segment_ids, source_eval.meta.segment_ids)
            self.assertTrue(torch.equal(bundle_eval.meta.patch_xyxy, source_eval.meta.patch_xyxy))
            self.assertEqual(segment_manifest["schema_version"], 3)
            self.assertEqual(segment_manifest["recipe_family"], "masked_zarr_segment")
            self.assertEqual(segment_manifest["segment_id"], "segA")
            self.assertEqual(segment_manifest["group_name"], "group_a")

    def test_generated_patch_bundle_recipe_ensures_bundle_before_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundle"
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 0, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            source_recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": train_segment, "segB": val_segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            )

            with (
                mock.patch("ink.recipes.data.patch_bundle.recipe.PatchBundleWriter") as writer_cls,
                mock.patch.object(ZarrPatchDataRecipe, "build", return_value="bundle") as build_mock,
            ):
                out = GeneratedPatchBundleDataRecipe(
                    bundle_root=str(bundle_root),
                    source=source_recipe,
                ).build(augment=_neutral_augment())

            writer_cls.assert_called_once_with(source_recipe)
            writer_cls.return_value.ensure.assert_called_once_with(out_root=bundle_root.resolve())
            build_mock.assert_called_once()
            self.assertEqual(out, "bundle")

    def test_patch_bundle_writer_ensure_reuses_existing_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundle"
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 0, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            source_recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": train_segment, "segB": val_segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            )

            writer = PatchBundleWriter(source_recipe)
            first_written = writer.ensure(out_root=bundle_root)
            self.assertTrue((bundle_root / "group_a" / "segA" / "manifest.yaml").exists())
            self.assertTrue((bundle_root / "group_a" / "segB" / "manifest.yaml").exists())

            with mock.patch.object(
                PatchBundleWriter,
                "_write_segment",
                side_effect=AssertionError("existing bundle should be reused"),
            ):
                second_written = writer.ensure(out_root=bundle_root)

            self.assertEqual(first_written, second_written)

    def test_patch_bundle_writer_ensure_rebuilds_only_stale_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundle"
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 0, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            source_recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": train_segment, "segB": val_segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            )

            writer = PatchBundleWriter(source_recipe)
            writer.ensure(out_root=bundle_root)

            inklabels_attrs = root / "group_a" / "segA" / "segA_inklabels.zarr" / ".zattrs"
            inklabels_attrs.write_text('{"bundle_test":"stale"}', encoding="utf-8")

            written_segments = []
            original_write_segment = PatchBundleWriter._write_segment

            def _record_write_segment(writer_self, **kwargs):
                written_segments.append(kwargs["segment_id"])
                return original_write_segment(writer_self, **kwargs)

            with mock.patch.object(
                PatchBundleWriter,
                "_write_segment",
                autospec=True,
                side_effect=_record_write_segment,
            ):
                rewritten = writer.ensure(out_root=bundle_root)

            self.assertEqual(rewritten, str(bundle_root.resolve()))
            self.assertEqual(written_segments, ["segA"])

    def test_patch_bundle_writer_masks_data_outside_supervision_or_validation_region(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundle"
            segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.pad(
                    np.full((4, 4), 255, dtype=np.uint8),
                    ((0, 4), (0, 4)),
                    mode="constant",
                    constant_values=0,
                ),
                validation_mask=np.pad(
                    np.full((4, 4), 255, dtype=np.uint8),
                    ((4, 0), (4, 0)),
                    mode="constant",
                    constant_values=0,
                ),
            )
            source_recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=4,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
            )

            PatchBundleWriter(source_recipe).write(out_root=bundle_root)

            label = np.asarray(zarr.open(str(bundle_root / "group_a" / "segA" / "segA_inklabels.zarr"), mode="r"))
            volume = np.asarray(zarr.open(str(bundle_root / "group_a" / "segA" / "segA.zarr"), mode="r"))

            self.assertEqual(int(label[4:, :4].sum()), 0)
            self.assertEqual(int(volume[:, 4:, :4].sum()), 0)
            self.assertGreater(int(label[:4, :4].sum()), 0)
            self.assertGreater(int(label[4:, 4:].sum()), 0)

    def test_patch_bundle_writer_outputs_dataset_root_reusable_with_new_split_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundle"
            seg_a = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            seg_b = _make_canonical_segment(
                root,
                group_name="group_b",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            source_recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a, "segB": seg_b},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
            )

            PatchBundleWriter(source_recipe).write(out_root=bundle_root)

            reused_bundle = ZarrPatchDataRecipe(
                dataset_root=str(bundle_root),
                segments={"segA": {}, "segB": {}},
                train_segment_ids=("segB",),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
            ).build(augment=_neutral_augment())

            train_batch = next(iter(reused_bundle.train_loader))
            eval_batch = next(iter(reused_bundle.eval_loader))

            self.assertEqual(train_batch.meta.segment_ids, ["segB"])
            self.assertEqual(eval_batch.meta.segment_ids, ["segA"])

    def test_generated_patch_bundle_val_only_segment_without_validation_mask_uses_supervision_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundle"
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segTrain",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segVal",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
                write_validation_mask=False,
            )
            source_recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segTrain": train_segment, "segVal": val_segment},
                train_segment_ids=("segTrain",),
                val_segment_ids=("segVal",),
                in_channels=4,
                patch_size=4,
                tile_size=4,
                stride=4,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
            )

            bundle = GeneratedPatchBundleDataRecipe(
                bundle_root=str(bundle_root),
                source=source_recipe,
            ).build(augment=_neutral_augment(size=4))

            self.assertTrue((bundle_root / "group_a" / "segVal" / "segVal_supervision_mask.zarr").exists())
            self.assertFalse((bundle_root / "group_a" / "segVal" / "segVal_validation_mask.zarr").exists())
            self.assertEqual(len(bundle.eval_loader.dataset), 4)

    def test_patch_bundle_writer_round_trips_patch_recipe_with_group_idx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundle"
            seg_a = _make_canonical_segment(
                root,
                group_name="0139",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            seg_b = _make_canonical_segment(
                root,
                group_name="0500p2",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            source_recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a, "segB": seg_b},
                train_segment_ids=("segA", "segB"),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=8,
                train_batch_size=2,
                valid_batch_size=1,
                shuffle=False,
            )
            bundle = GeneratedPatchBundleDataRecipe(
                bundle_root=str(bundle_root),
                source=source_recipe,
            ).build(augment=_neutral_augment())

            train_batch = next(iter(bundle.train_loader))

            self.assertEqual(train_batch.meta.group_idx.tolist(), [0, 1])
            self.assertIsNone(bundle.group_counts)

    def test_patch_bundle_writer_round_trips_explicit_group_counts_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundle"
            seg_a = _make_canonical_segment(
                root,
                group_name="0139",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            seg_b = _make_canonical_segment(
                root,
                group_name="0500p2",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            PatchBundleWriter(
                ZarrPatchDataRecipe(
                    dataset_root=str(root),
                    segments={"segA": seg_a, "segB": seg_b},
                    train_segment_ids=("segA", "segB"),
                    val_segment_ids=("segA",),
                    in_channels=4,
                    patch_size=8,
                    train_batch_size=2,
                    valid_batch_size=1,
                    shuffle=False,
                    extras={"group_counts": [3, 4]},
                )
            ).write(out_root=bundle_root)

            bundle = GeneratedPatchBundleDataRecipe(
                bundle_root=str(bundle_root),
                source=ZarrPatchDataRecipe(
                    dataset_root=str(root),
                    segments={"segA": seg_a, "segB": seg_b},
                    train_segment_ids=("segA", "segB"),
                    val_segment_ids=("segA",),
                    in_channels=4,
                    patch_size=8,
                    train_batch_size=2,
                    valid_batch_size=1,
                    shuffle=False,
                    extras={"group_counts": [3, 4]},
                ),
            ).build(augment=_neutral_augment())

            self.assertEqual(bundle.group_counts, [3, 4])

    def test_build_experiment_data_accepts_zarr_patch_data_recipe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            recipe = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": segment},
                train_segment_ids=("segA",),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
            )

            bundle = build_experiment_data(recipe, augment=_neutral_augment())

            self.assertIsInstance(bundle.train_loader, DataLoader)
            self.assertIsInstance(bundle.eval_loader, DataLoader)
            self.assertIsInstance(bundle.augment, TrainAugment)
            self.assertEqual(bundle.extras, {})
            self.assertNotIn("stitch", bundle.extras)

class DeclarativeDataAssemblyTests(unittest.TestCase):
    def test_assemble_experiment_binds_patch_trainer_with_zarr_data_recipe_when_bundle_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            val_segment = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )

            data = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segA": train_segment,
                    "segB": val_segment,
                },
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            )
            experiment = Experiment(
                name="erm_baseline",
                data=data,
                model=StaticBuildRecipe("model"),
                loss="loss",
                objective=StaticBuildRecipe("objective"),
                runtime=StaticBuildRecipe("runtime"),
                augment=_neutral_augment(),
                trainer=PatchTrainer(),
            )

            bundle = build_experiment_data(
                experiment.data,
                runtime=experiment.runtime,
                augment=experiment.augment,
            )
            bound = assemble_experiment(experiment, bundle)
            training = bound.trainer

            self.assertIsInstance(training.train_loader, DataLoader)
            self.assertIsInstance(training.eval_loader, DataLoader)
            train_batch = next(iter(training.train_loader))
            val_batch = next(iter(training.eval_loader))
            self.assertIsNotNone(train_batch.y)
            self.assertIsNotNone(train_batch.meta.patch_xyxy)
            self.assertIsNotNone(train_batch.meta.valid_mask)
            self.assertIsNotNone(val_batch.meta.patch_xyxy)
            self.assertIsNotNone(val_batch.meta.valid_mask)
            self.assertIs(training.experiment.data, data)


if __name__ == "__main__":
    unittest.main()
