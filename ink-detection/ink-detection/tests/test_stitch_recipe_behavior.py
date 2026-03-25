from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import zarr

from ink.core import Batch, BatchMeta, DataBundle
from ink.recipes.augment import TrainAugment
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.patch_bundle import GeneratedPatchBundleDataRecipe
from ink.recipes.data.zarr_data import ZarrPatchDataRecipe
from ink.recipes import stitch as stitch_pkg
from ink.recipes.stitch import loaders as stitch_loaders
from ink.recipes.stitch.plan_from_zarr import build_zarr_stitch_segment_specs
from ink.recipes.stitch import train_component_runtime as stitch_train_component_runtime
from ink.recipes.stitch import train_runtime as stitch_train_runtime
from ink.recipes.stitch import (
    EvalStitchConfig,
    StitchData,
    StitchLayout,
    StitchRuntime,
    StitchRuntimeRecipe,
    TrainStitchConfig,
    TrainStitchLossConfig,
    TrainStitchVizConfig,
    allocate_segment_buffers,
    build_segment_roi_meta,
)


class _LossComponentTerm:
    requires_boundary_dist_map = False

    def compute(self, batch):
        valid_logits = batch.logits[batch.valid_mask]
        valid_targets = batch.targets[batch.valid_mask]
        return {
            "loss": valid_logits.mean(),
            "demo_component": valid_targets.mean(),
            "ignored_component": torch.tensor(9.0),
        }


class _IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x


class _PatchLoss:
    def __call__(self, logits, targets, *, valid_mask=None):
        del targets, valid_mask
        return torch.tensor(2.0, dtype=torch.float32, device=logits.device)

    def loss_values(self, logits, targets, *, valid_mask=None):
        del targets, valid_mask
        batch = int(logits.shape[0])
        return torch.full((batch,), 2.0, dtype=torch.float32, device=logits.device)

    def training_outputs(self, logits, targets, *, valid_mask=None):
        del targets, valid_mask
        device = logits.device
        return {
            "loss": torch.tensor(2.0, dtype=torch.float32, device=device),
            "components": {
                "dice": torch.tensor(3.0, dtype=torch.float32, device=device),
                "bce": torch.tensor(4.0, dtype=torch.float32, device=device),
                "dice_loss": torch.tensor(5.0, dtype=torch.float32, device=device),
            },
        }


_patch_loss = _PatchLoss()


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
) -> dict[str, object]:
    segment_dir = root / group_name / segment_id
    _write_zarr_array(segment_dir / f"{segment_id}.zarr", np.asarray(volume))
    _write_zarr_array(segment_dir / f"{segment_id}_inklabels.zarr", np.asarray(label))
    _write_zarr_array(segment_dir / f"{segment_id}_supervision_mask.zarr", np.asarray(supervision_mask))
    if write_validation_mask:
        _write_zarr_array(
            segment_dir / f"{segment_id}_validation_mask.zarr",
            np.asarray(supervision_mask if validation_mask is None else validation_mask),
        )
    return {
        "layer_range": (0, int(volume.shape[0] if volume.shape[0] <= volume.shape[-1] else volume.shape[-1])),
        "reverse_layers": False,
    }


def _single_patch_batch():
    return (
        torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32),
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32),
        [(0, 0, 2, 2)],
        torch.tensor([0], dtype=torch.long),
    )


class StitchRoiLayoutTests(unittest.TestCase):
    def test_build_segment_roi_meta_matches_expected_shapes(self):
        meta = build_segment_roi_meta((10, 12), [(1, 3, 2, 5), (3, 5, 0, 2)], 2, use_roi=True)

        self.assertEqual(meta["full_shape"], (5, 6))
        self.assertEqual(
            meta["rois"],
            [
                {"offset": (1, 2), "buffer_shape": (2, 3)},
                {"offset": (3, 0), "buffer_shape": (2, 2)},
            ],
        )

    def test_allocate_segment_buffers_returns_float32_numpy_pairs(self):
        meta = {
            "rois": [
                {"offset": (1, 2), "buffer_shape": (2, 3)},
            ]
        }
        buffers = allocate_segment_buffers(meta)

        self.assertEqual(len(buffers), 1)
        pred_buf, count_buf, offset = buffers[0]
        self.assertEqual(pred_buf.shape, (2, 3))
        self.assertEqual(count_buf.shape, (2, 3))
        self.assertEqual(offset, (1, 2))


class StitchRuntimeDataTests(unittest.TestCase):
    def test_stitch_data_reads_explicit_config(self):
        term = _LossComponentTerm()
        stitch_data = StitchData.from_config(
            {
                "downsample": 2,
                "use_roi": True,
                "eval": {
                    "segments": [
                        {"segment_id": "segA", "shape": (8, 8), "bbox": (1, 3, 2, 4)},
                    ],
                },
                "train": {
                    "segment_ids": ["segA"],
                    "segments": [{"segment_id": "segA", "shape": (8, 8)}],
                    "components": [
                        {"component_key": ("segA", 0), "shape": (4, 4), "bbox": (0, 2, 0, 2)},
                    ],
                    "viz": {
                        "enabled": True,
                        "loss_components": ["loss"],
                    },
                    "loss": {
                        "patch_batch_size": 3,
                        "valid_batch_size": 5,
                        "patch_loss_weight": 0.75,
                        "gradient_checkpointing": True,
                        "save_on_cpu": True,
                        "terms": (term,),
                    },
                }
            }
        )

        self.assertEqual(stitch_data.layout.downsample, 2)
        self.assertTrue(stitch_data.layout.use_roi)
        self.assertEqual([spec.segment_id for spec in stitch_data.eval.segments], ["segA"])
        self.assertEqual(stitch_data.train.segment_ids, ["segA"])
        self.assertEqual(stitch_data.train.component_keys, [("segA", 0)])
        self.assertTrue(stitch_data.train.viz.enabled)
        self.assertEqual(stitch_data.train.viz.loss_components, ("loss",))
        self.assertEqual(stitch_data.train.loss.patch_batch_size, 3)
        self.assertEqual(stitch_data.train.loss.patch_loss_weight, 0.75)
        self.assertTrue(stitch_data.train.loss.gradient_checkpointing)
        self.assertTrue(stitch_data.train.loss.save_on_cpu)
        self.assertEqual(stitch_data.train.loss.terms, (term,))

    def test_stitch_data_accepts_log_only_segment_lane(self):
        stitch_data = StitchData.from_config(
            {
                "log_only": {
                    "segment_ids": ["segL"],
                    "every_n_epochs": 5,
                }
            }
        )

        self.assertEqual(stitch_data.log_only.segment_ids, ["segL"])
        self.assertEqual(stitch_data.log_only.every_n_epochs, 5)

    def test_stitch_data_rejects_unknown_top_level_keys(self):
        with self.assertRaisesRegex(ValueError, "stitch.extra"):
            StitchData.from_config(
                {
                    "extra": {
                        "segments": [{"segment_id": "segL", "shape": (8, 8)}],
                    }
                }
            )

    def test_stitch_data_rejects_dead_eval_middle_layer_keys(self):
        with self.assertRaisesRegex(ValueError, "stitch.eval.loader_to_segment"):
            StitchData.from_config(
                {
                    "eval": {
                        "loader_to_segment": {0: "segA"},
                    }
                }
            )

    def test_stitch_data_rejects_flat_train_loss_keys(self):
        with self.assertRaisesRegex(ValueError, "stitch.train.patch_loss_weight"):
            StitchData.from_config(
                {
                    "train": {
                        "patch_loss_weight": 0.75,
                    }
                }
            )

    def test_train_stitch_accepts_explicit_segment_id_overrides(self):
        stitch_data = StitchData.from_config(
            {
                "train": {
                    "segment_ids": ["segA", "segB"],
                },
                "eval": {
                    "segment_ids": ["segC"],
                },
            }
        )

        self.assertEqual(stitch_data.train.segment_ids, ["segA", "segB"])
        self.assertEqual(stitch_data.eval.segment_ids, ["segC"])

    def test_train_stitch_viz_config_is_cadence_only(self):
        stitch_data = StitchData.from_config(
            {
                "train": {
                    "viz": {
                        "enabled": True,
                    },
                }
            }
        )

        self.assertFalse(hasattr(stitch_data.train.viz, "segment_ids"))

    def test_stitch_runtime_recipe_derives_train_and_eval_segments_from_zarr_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a, "segB": seg_b},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            ).build(augment=_neutral_augment())

            runtime = StitchRuntimeRecipe().build(bundle)

            self.assertTrue(runtime.data.layout.use_roi)
            self.assertEqual([spec.segment_id for spec in runtime.data.train.segments], ["segA"])
            self.assertEqual(runtime.data.train.segments[0].bbox, ((0, 8, 0, 8),))
            self.assertEqual([spec.segment_id for spec in runtime.data.eval.segments], ["segB"])
            self.assertEqual(runtime.data.eval.segments[0].bbox, ((0, 8, 0, 8),))

    def test_stitch_runtime_recipe_derives_train_and_eval_segments_from_patch_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                group_name="group_a",
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
                normalization=ClipMaxDiv255Normalization(),
            )
            bundle = GeneratedPatchBundleDataRecipe(
                bundle_root=str(root / "bundle"),
                source=source_recipe,
            ).build(augment=_neutral_augment())

            runtime = StitchRuntimeRecipe().build(bundle)

            self.assertTrue(runtime.data.layout.use_roi)
            self.assertEqual([spec.segment_id for spec in runtime.data.train.segments], ["segA"])
            self.assertEqual(runtime.data.train.segments[0].bbox, ((0, 8, 0, 8),))
            self.assertEqual([spec.segment_id for spec in runtime.data.eval.segments], ["segB"])
            self.assertEqual(runtime.data.eval.segments[0].bbox, ((0, 8, 0, 8),))

    def test_build_stitch_runtime_loaders_supports_patch_bundle_train_viz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seg_a = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
                write_validation_mask=False,
            )
            seg_b = _make_canonical_segment(
                root,
                group_name="group_a",
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
                normalization=ClipMaxDiv255Normalization(),
            )
            bundle = GeneratedPatchBundleDataRecipe(
                bundle_root=str(root / "bundle"),
                source=source_recipe,
            ).build(augment=_neutral_augment())

            runtime = StitchRuntimeRecipe(config={"train": {"viz": {"enabled": True}}}).build(bundle)
            train_viz_loaders, log_only_loaders = stitch_loaders.build_stitch_runtime_loaders(
                stitch_data=runtime.data,
                train_loader=bundle.train_loader,
                eval_loader=bundle.eval_loader,
            )

            self.assertEqual(len(train_viz_loaders), 1)
            self.assertEqual(len(log_only_loaders), 0)
            batch = next(iter(train_viz_loaders[0]))
            self.assertEqual(batch.meta.segment_ids, ["segA"])
            self.assertEqual(batch.meta.patch_xyxy[0].tolist(), [0, 0, 8, 8])

    def test_build_stitch_runtime_loaders_train_viz_uses_train_mask_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seg_a = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
                write_validation_mask=False,
            )
            seg_b = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a, "segB": seg_b},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            ).build(augment=_neutral_augment())

            runtime = StitchRuntimeRecipe(config={"train": {"viz": {"enabled": True}}}).build(bundle)
            train_viz_loaders, _log_only_loaders = stitch_loaders.build_stitch_runtime_loaders(
                stitch_data=runtime.data,
                train_loader=bundle.train_loader,
                eval_loader=bundle.eval_loader,
            )

            self.assertEqual(len(train_viz_loaders), 1)
            self.assertEqual(train_viz_loaders[0].dataset.split, "valid")
            self.assertEqual(train_viz_loaders[0].dataset.mask_split_name, "train")
            self.assertEqual(train_viz_loaders[0].dataset.mask_names_for_segment("segA"), ("supervision_mask",))
            batch = next(iter(train_viz_loaders[0]))
            self.assertEqual(batch.meta.segment_ids, ["segA"])

    def test_stitch_runtime_recipe_uses_explicit_train_and_eval_segment_id_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a, "segB": seg_b},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            ).build(augment=_neutral_augment())

            runtime = StitchRuntimeRecipe(
                config={
                    "train": {
                        "segment_ids": ["segB"],
                        "viz": {"enabled": True},
                    },
                    "eval": {
                        "segment_ids": ["segA"],
                    },
                }
            ).build(bundle)

            self.assertEqual(runtime.data.train.segment_ids, ["segB"])
            self.assertEqual([spec.segment_id for spec in runtime.data.train.segments], ["segB"])
            self.assertEqual(runtime.data.eval.segment_ids, ["segA"])
            self.assertEqual([spec.segment_id for spec in runtime.data.eval.segments], ["segA"])

    def test_stitch_runtime_recipe_derives_log_only_segments_from_explicit_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a, "segB": seg_b},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            ).build(augment=_neutral_augment())

            runtime = StitchRuntimeRecipe(
                config={"log_only": {"segment_ids": ["segB"], "every_n_epochs": 4}}
            ).build(bundle)

            self.assertEqual(runtime.data.log_only.segment_ids, ["segB"])
            self.assertEqual(runtime.data.log_only.every_n_epochs, 4)
            self.assertEqual([spec.segment_id for spec in runtime.data.log_only.segments], ["segB"])
            self.assertEqual(runtime.data.log_only.segments[0].bbox, ((0, 8, 0, 8),))

    def test_stitch_runtime_recipe_log_only_can_fall_back_to_full_segment_without_supervision_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            seg_c_dir = root / "group_b" / "segC"
            _write_zarr_array(
                seg_c_dir / "segC.zarr",
                np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (90, 100, 110, 120)], axis=0),
            )
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segA": seg_a,
                    "segB": seg_b,
                    "segC": {"layer_range": (0, 4), "reverse_layers": False},
                },
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            ).build(augment=_neutral_augment())

            runtime = StitchRuntimeRecipe(
                config={"log_only": {"segment_ids": ["segC"]}}
            ).build(bundle)

            self.assertEqual(runtime.data.log_only.segment_ids, ["segC"])
            self.assertEqual([spec.segment_id for spec in runtime.data.log_only.segments], ["segC"])
            self.assertIsNone(runtime.data.log_only.segments[0].bbox)

    def test_stitch_runtime_build_exposes_explicit_train_runtime(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                layout=StitchLayout(),
                train=TrainStitchConfig(
                    components=[{"component_key": ("segA", 0), "shape": (4, 4)}],
                    loss=TrainStitchLossConfig(
                        patch_batch_size=2,
                        valid_batch_size=4,
                        patch_loss_weight=0.25,
                        gradient_checkpointing=True,
                        save_on_cpu=True,
                    ),
                ),
            )
        )

        self.assertIs(runtime.data, runtime.train.data)
        self.assertIs(runtime.state, runtime.train.state)
        self.assertEqual(runtime.train.patch_loss_weight, 0.25)
        self.assertTrue(runtime.train.gradient_checkpointing)
        self.assertTrue(runtime.train.save_on_cpu)

    def test_runtime_accepts_explicit_stitch_data_object(self):
        stitch_data = StitchData(
            layout=StitchLayout(),
            eval=EvalStitchConfig(
                segments=[{"segment_id": "segA", "shape": (4, 4)}],
            ),
        )

        runtime = StitchRuntime._from_config(stitch_data)

        self.assertIs(runtime.data, stitch_data)
        self.assertIn("segA", runtime.state.roi_buffers_by_split["eval"])

    def test_runtime_exposes_explicit_segment_layout(self):
        runtime = StitchRuntime._from_config(
            {
                "downsample": 2,
                "use_roi": True,
                "eval": {
                    "segments": [
                        {"segment_id": "segA", "shape": (8, 12), "bbox": [(1, 3, 2, 5), (3, 4, 0, 2)]},
                    ],
                },
            }
        )

        layout = runtime.segment_layout()

        self.assertEqual(layout.downsample, 2)
        self.assertEqual(layout.segment_shapes, {"segA": (8, 12)})
        self.assertEqual(layout.segment_rois, {"segA": ((1, 3, 2, 5), (3, 4, 0, 2))})

    def test_train_runtime_shares_runtime_state(self):
        runtime = StitchRuntime._from_config()

        self.assertIs(runtime.state, runtime.train.state)


class ZarrStitchLoaderPrepTests(unittest.TestCase):
    def test_build_stitch_inference_loaders_use_union_of_supervision_and_validation_masks_when_roi_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seg_a = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((12, 12), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((12, 12), 255, dtype=np.uint8),
                supervision_mask=np.pad(
                    np.full((4, 4), 255, dtype=np.uint8),
                    ((0, 8), (0, 8)),
                    mode="constant",
                    constant_values=0,
                ),
                validation_mask=np.pad(
                    np.full((4, 4), 255, dtype=np.uint8),
                    ((8, 0), (8, 0)),
                    mode="constant",
                    constant_values=0,
                ),
            )
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a},
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
            ).build(augment=_neutral_augment(size=4))

            loaders = stitch_loaders.build_stitch_inference_loaders(
                stitch_data=StitchData.from_config(
                    {
                        "use_roi": True,
                        "eval": {"segment_ids": ["segA"]},
                    }
                ),
                eval_loader=bundle.eval_loader,
            )

            self.assertEqual(len(loaders), 1)
            self.assertEqual(
                {
                    (str(segment_id), tuple(int(value) for value in xyxy))
                    for segment_id, xyxy in loaders[0].dataset._samples
                },
                {
                    ("segA", (0, 0, 4, 4)),
                    ("segA", (8, 8, 12, 12)),
                },
            )

    def test_build_stitch_inference_loaders_use_full_grid_when_roi_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seg_a = _make_canonical_segment(
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
            )
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a},
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
            ).build(augment=_neutral_augment(size=4))

            loaders = stitch_loaders.build_stitch_inference_loaders(
                stitch_data=StitchData.from_config(
                    {
                        "use_roi": False,
                        "eval": {"segment_ids": ["segA"]},
                    }
                ),
                eval_loader=bundle.eval_loader,
            )

            self.assertEqual(len(loaders), 1)
            self.assertEqual(len(loaders[0].dataset), 4)
            batch = next(iter(loaders[0]))
            self.assertIsNone(batch.y)
            self.assertIsNone(batch.meta.valid_mask)
            self.assertEqual(batch.meta.segment_ids, ["segA"])
            self.assertEqual(batch.meta.patch_xyxy[0].tolist(), [0, 0, 4, 4])

    def test_build_stitch_inference_loaders_skip_empty_volume_patches_when_roi_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            volume = np.zeros((4, 8, 8), dtype=np.uint8)
            volume[:, 0:4, 0:4] = 64
            seg_a = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=volume,
                label=np.zeros((8, 8), dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a},
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
            ).build(augment=_neutral_augment(size=4))

            loaders = stitch_loaders.build_stitch_inference_loaders(
                stitch_data=StitchData.from_config(
                    {
                        "use_roi": False,
                        "eval": {"segment_ids": ["segA"]},
                    }
                ),
                eval_loader=bundle.eval_loader,
            )

            self.assertEqual(len(loaders), 1)
            non_empty_batches = [batch for batch in loaders[0] if batch is not None]
            self.assertEqual(len(non_empty_batches), 1)
            self.assertEqual(non_empty_batches[0].meta.patch_xyxy[0].tolist(), [0, 0, 4, 4])

    def test_build_zarr_stitch_segment_specs_use_union_of_supervision_and_validation_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seg_a = _make_canonical_segment(
                root,
                group_name="group_a",
                segment_id="segA",
                volume=np.stack([np.full((12, 12), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
                label=np.full((12, 12), 255, dtype=np.uint8),
                supervision_mask=np.pad(
                    np.full((4, 4), 255, dtype=np.uint8),
                    ((0, 8), (0, 8)),
                    mode="constant",
                    constant_values=0,
                ),
                validation_mask=np.pad(
                    np.full((4, 4), 255, dtype=np.uint8),
                    ((8, 0), (8, 0)),
                    mode="constant",
                    constant_values=0,
                ),
            )
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a},
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
            ).build(augment=_neutral_augment(size=4))

            segment_specs = build_zarr_stitch_segment_specs(
                bundle.eval_loader.dataset,
                segment_ids=("segA",),
                downsample=1,
                mode="eval",
                use_roi=True,
            )

            self.assertEqual(len(segment_specs), 1)
            self.assertEqual(
                segment_specs[0].bbox,
                (
                    (0, 4, 0, 4),
                    (8, 12, 8, 12),
                ),
            )

    def test_build_zarr_segment_eval_loaders_reuses_existing_dataset_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a, "segB": seg_b},
                train_segment_ids=("segA", "segB"),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            ).build(augment=_neutral_augment())

            dataset = bundle.train_loader.dataset
            expected_samples = [sample for sample in dataset._samples if str(sample[0]) == "segB"]

            with mock.patch.object(stitch_loaders, "build_zarr_split_samples") as build_mock:
                loaders = stitch_loaders.build_zarr_segment_eval_loaders(
                    dataset,
                    segment_ids=("segB",),
                    batch_size=1,
                )

            build_mock.assert_not_called()
            self.assertEqual(len(loaders), 1)
            self.assertEqual(loaders[0].dataset._samples, expected_samples)
            self.assertEqual(loaders[0].dataset.segment_ids, ("segB",))
            self.assertIs(loaders[0].dataset._volume_cache, dataset._volume_cache)
            self.assertIs(loaders[0].dataset._label_mask_store_cache, dataset._label_mask_store_cache)

    def test_build_zarr_segment_eval_loaders_falls_back_to_rebuild_with_patch_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            cache_root = root / "patch_cache"
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a, "segB": seg_b},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
                patch_index_cache_dir=str(cache_root),
            ).build(augment=_neutral_augment())

            dataset = bundle.train_loader.dataset
            rebuilt_samples = [("segB", (0, 0, 8, 8), 0)]

            with mock.patch.object(
                stitch_loaders,
                "build_zarr_split_samples",
                return_value=rebuilt_samples,
            ) as build_mock:
                loaders = stitch_loaders.build_zarr_segment_eval_loaders(
                    dataset,
                    segment_ids=("segB",),
                    batch_size=1,
                )

            build_mock.assert_called_once()
            self.assertEqual(build_mock.call_args.args[0].patch_index_cache_dir, str(cache_root))
            self.assertEqual(len(loaders), 1)
            self.assertEqual(loaders[0].dataset._samples, rebuilt_samples)
            self.assertEqual(loaders[0].dataset.segment_ids, ("segB",))

    def test_build_zarr_segment_eval_loaders_rebuilds_only_missing_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                group_name="group_a",
                segment_id="segB",
                volume=np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (50, 60, 70, 80)], axis=0),
                label=np.full((8, 8), 255, dtype=np.uint8),
                supervision_mask=np.full((8, 8), 255, dtype=np.uint8),
            )
            bundle = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={"segA": seg_a, "segB": seg_b},
                train_segment_ids=("segA",),
                val_segment_ids=("segB",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
                normalization=ClipMaxDiv255Normalization(),
            ).build(augment=_neutral_augment())

            dataset = bundle.train_loader.dataset
            expected_seg_a_samples = [sample for sample in dataset._samples if str(sample[0]) == "segA"]
            rebuilt_seg_b_samples = [("segB", (0, 0, 8, 8), 0)]

            with mock.patch.object(
                stitch_loaders,
                "build_zarr_split_samples",
                return_value=rebuilt_seg_b_samples,
            ) as build_mock:
                loaders = stitch_loaders.build_zarr_segment_eval_loaders(
                    dataset,
                    segment_ids=("segA", "segB"),
                    batch_size=1,
                )

            build_mock.assert_called_once()
            self.assertEqual(build_mock.call_args.kwargs["segment_ids"], ("segB",))
            self.assertEqual(len(loaders), 2)
            self.assertEqual(loaders[0].dataset._samples, expected_seg_a_samples)
            self.assertEqual(loaders[0].dataset.segment_ids, ("segA",))
            self.assertEqual(loaders[1].dataset._samples, rebuilt_seg_b_samples)
            self.assertEqual(loaders[1].dataset.segment_ids, ("segB",))


class StitchRuntimeMutationTests(unittest.TestCase):
    def test_set_train_component_datasets_returns_matching_dataset_and_reorders_specs(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                train=TrainStitchConfig(
                    components=[
                        {"component_key": ("segA", 0), "shape": (4, 4)},
                        {"component_key": ("segB", 1), "shape": (4, 4)},
                    ]
                )
            )
        )

        with mock.patch.object(runtime.state, "clear_boundary_caches") as clear_caches:
            runtime.train.set_component_datasets(
                ["train_dataset_b", "train_dataset_a"],
                [("segB", 1), ("segA", 0)],
            )

        clear_caches.assert_called_once_with()
        self.assertEqual(runtime.train.dataset_for_component(("segA", 0)), "train_dataset_a")
        self.assertEqual(runtime.train.dataset_for_component(("segB", 1)), "train_dataset_b")
        self.assertIsNone(runtime.train.dataset_for_component(("segC", 0)))
        self.assertEqual(runtime.data.train.component_keys, [("segB", 1), ("segA", 0)])

    def test_set_train_loaders_updates_loaders_only(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                train=TrainStitchConfig(
                    segment_ids=["segA"],
                    viz=TrainStitchVizConfig(enabled=True),
                )
            )
        )

        runtime.train.set_loaders(["loader_a", "loader_b"])

        self.assertEqual(runtime.train.loaders, ["loader_a", "loader_b"])
        self.assertEqual(runtime.data.train.segment_ids, ["segA"])

    def test_duplicate_train_component_keys_raise_at_stitch_data_boundary(self):
        with self.assertRaises(ValueError):
            StitchRuntime._from_config(
                StitchData(
                    train=TrainStitchConfig(
                        components=[
                            {"component_key": ("segA", 0), "shape": (2, 2)},
                            {"component_key": ("segA", 0), "shape": (2, 2)},
                        ]
                    )
                )
            )


class StitchRuntimeBehaviorTests(unittest.TestCase):
    def test_saved_tensors_warning_uses_structured_gradient_checkpointing_key(self):
        owner = types.SimpleNamespace(
            gradient_checkpointing=True,
            save_on_cpu=True,
            _warned_checkpoint_vs_offload=False,
        )

        with mock.patch("ink.recipes.stitch.train_runtime._log") as log_mock:
            stitch_train_component_runtime.stitch_saved_tensors_context(owner, log=stitch_train_runtime._log)

        log_mock.assert_called_once()
        message = log_mock.call_args.args[1]
        self.assertIn("stitch.train.loss.gradient_checkpointing=true", message)
        self.assertNotIn("training.stitch_gradient_checkpointing", message)

    def test_save_on_cpu_runtime_error_uses_structured_save_on_cpu_key(self):
        owner = types.SimpleNamespace(
            gradient_checkpointing=False,
            save_on_cpu=True,
            _warned_checkpoint_vs_offload=False,
        )

        with mock.patch("ink.recipes.stitch.train_component_runtime.torch.autograd", new=types.SimpleNamespace()):
            with self.assertRaises(RuntimeError) as raised:
                stitch_train_component_runtime.stitch_saved_tensors_context(owner, log=stitch_train_runtime._log)

        message = str(raised.exception)
        self.assertIn("stitch.train.loss.save_on_cpu=true", message)
        self.assertNotIn("training.stitch_save_on_cpu", message)

    def test_compute_train_stitch_loss_uses_config_owned_stitch_terms(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                train=TrainStitchConfig(
                    components=[{"component_key": ("segA", 0), "shape": (2, 2)}],
                    loss=TrainStitchLossConfig(terms=(_LossComponentTerm(),)),
                )
            ),
            patch_loss=_patch_loss,
        )
        runtime.train.set_component_datasets(
            {
                ("segA", 0): [
                    (
                        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32),
                        torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                        (0, 0, 2, 2),
                        0,
                    )
                ]
            }
        )

        out = runtime.train.compute_component_loss(_IdentityModel(), component_key=("segA", 0))

        self.assertEqual(out["component_key"], ("segA", 0))
        self.assertEqual(out["patch"]["count"], 1)
        self.assertIn("loss", out["stitch"]["components"])
        self.assertIn("demo_component", out["stitch"]["components"])

    def test_compute_train_stitch_loss_with_zero_patch_weight_skips_patch_components(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                train=TrainStitchConfig(
                    components=[{"component_key": ("segA", 0), "shape": (2, 2)}],
                    loss=TrainStitchLossConfig(
                        patch_loss_weight=0.0,
                        terms=(_LossComponentTerm(),),
                    ),
                )
            )
        )
        runtime.train.set_component_datasets(
            {
                ("segA", 0): [
                    (
                        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32),
                        torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                        (0, 0, 2, 2),
                        0,
                    )
                ]
            }
        )

        out = runtime.train.compute_component_loss(_IdentityModel(), component_key=("segA", 0))

        self.assertEqual(out["patch"]["count"], 1)
        self.assertEqual(out["patch"]["components"], {})
        self.assertIn("loss", out["stitch"]["components"])
        self.assertIn("demo_component", out["stitch"]["components"])

    def test_compute_train_stitch_loss_raises_when_component_has_no_stitched_pixels(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                train=TrainStitchConfig(
                    components=[{"component_key": ("segA", 0), "shape": (2, 2)}],
                    loss=TrainStitchLossConfig(terms=(_LossComponentTerm(),)),
                )
            ),
            patch_loss=_patch_loss,
        )
        runtime.train.set_component_datasets(
            {
                ("segA", 0): [
                    (
                        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32),
                        torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                        (8, 8, 10, 10),
                        0,
                    )
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "produced zero stitched components"):
            runtime.train.compute_component_loss(_IdentityModel(), component_key=("segA", 0))

    def test_run_train_viz_pass_without_loss_components_returns_segment_viz_only(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                train=TrainStitchConfig(
                    segments=[{"segment_id": "segA", "shape": (2, 2)}],
                    viz=TrainStitchVizConfig(enabled=True),
                )
            )
        )
        runtime.train.set_loaders([[_single_patch_batch()]])

        out = runtime.train.run_viz_pass(_IdentityModel(), epoch=0)

        self.assertIn("segA", out)
        self.assertIn("img_u8", out["segA"])
        self.assertNotIn("loss_components", out["segA"])

    def test_run_train_viz_pass_accepts_batch_objects_from_real_dataloaders(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                train=TrainStitchConfig(
                    segments=[{"segment_id": "segA", "shape": (2, 2)}],
                    viz=TrainStitchVizConfig(enabled=True),
                )
            )
        )
        batch = Batch(
            x=torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float32),
            y=torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32),
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 2, 2]], dtype=torch.long),
            ),
        )
        runtime.train.set_loaders([[batch]])

        out = runtime.train.run_viz_pass(_IdentityModel(), epoch=0)

        self.assertIn("segA", out)
        self.assertIn("img_u8", out["segA"])

    def test_run_train_viz_pass_with_loss_components_returns_filtered_components(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                train=TrainStitchConfig(
                    segments=[{"segment_id": "segA", "shape": (2, 2)}],
                    viz=TrainStitchVizConfig(
                        enabled=True,
                        loss_components=("loss", "demo_component"),
                    ),
                    loss=TrainStitchLossConfig(terms=(_LossComponentTerm(),)),
                )
            )
        )
        runtime.train.set_loaders([[_single_patch_batch()]])

        out = runtime.train.run_viz_pass(_IdentityModel(), epoch=0)

        self.assertIn("segA", out)
        self.assertIn("loss_components", out["segA"])
        self.assertIn("covered_px", out["segA"]["loss_components"])
        self.assertIn("loss", out["segA"]["loss_components"])
        self.assertIn("demo_component", out["segA"]["loss_components"])
        self.assertNotIn("ignored_component", out["segA"]["loss_components"])
        self.assertAlmostEqual(float(out["segA"]["loss_components"]["loss"]), 0.25, places=6)
        self.assertAlmostEqual(float(out["segA"]["loss_components"]["demo_component"]), 0.5, places=6)

    def test_run_train_viz_pass_filter_excludes_unrequested_loss_keys(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                train=TrainStitchConfig(
                    segments=[{"segment_id": "segA", "shape": (2, 2)}],
                    viz=TrainStitchVizConfig(
                        enabled=True,
                        loss_components=("demo_component",),
                    ),
                    loss=TrainStitchLossConfig(terms=(_LossComponentTerm(),)),
                )
            )
        )
        runtime.train.set_loaders([[_single_patch_batch()]])

        out = runtime.train.run_viz_pass(_IdentityModel(), epoch=0)

        self.assertIn("segA", out)
        self.assertIn("loss_components", out["segA"])
        self.assertIn("covered_px", out["segA"]["loss_components"])
        self.assertIn("demo_component", out["segA"]["loss_components"])
        self.assertNotIn("loss", out["segA"]["loss_components"])
        self.assertNotIn("ignored_component", out["segA"]["loss_components"])

    def test_run_train_viz_pass_uses_bound_precision_context(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                train=TrainStitchConfig(
                    segments=[{"segment_id": "segA", "shape": (2, 2)}],
                    viz=TrainStitchVizConfig(enabled=True),
                )
            )
        )
        runtime.train.set_loaders([[_single_patch_batch()]])
        calls = []

        def precision_context(*, device=None):
            calls.append(device)
            return nullcontext()

        runtime.train.precision_context = precision_context

        runtime.train.run_viz_pass(_IdentityModel(), epoch=0)

        self.assertEqual(calls, [torch.device("cpu")])

    def test_compute_train_stitch_loss_uses_bound_precision_context(self):
        runtime = StitchRuntime._from_config(
            StitchData(
                train=TrainStitchConfig(
                    components=[{"component_key": ("segA", 0), "shape": (2, 2)}],
                    loss=TrainStitchLossConfig(terms=(_LossComponentTerm(),)),
                )
            ),
            patch_loss=_patch_loss,
        )
        runtime.train.set_component_datasets(
            {
                ("segA", 0): [
                    (
                        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32),
                        torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                        (0, 0, 2, 2),
                        0,
                    )
                ]
            }
        )
        calls = []

        def precision_context(*, device=None):
            calls.append(device)
            return nullcontext()

        runtime.train.precision_context = precision_context

        runtime.train.compute_component_loss(_IdentityModel(), component_key=("segA", 0))

        self.assertEqual(calls, [torch.device("cpu")])


class StitchRuntimeContractTests(unittest.TestCase):
    def test_package_exports_train_and_eval_helpers(self):
        self.assertTrue(hasattr(stitch_pkg, "TrainStitchRuntime"))
        self.assertTrue(hasattr(stitch_pkg, "StitchInference"))
        self.assertFalse(hasattr(stitch_pkg, "run_log_only_stitch_pass"))

    def test_runtime_recipe_builds_stitch_runtime(self):
        bundle = DataBundle(train_loader="train", eval_loader="val", in_channels=8)

        runtime = StitchRuntimeRecipe(
            config={
                "eval": {
                    "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                }
            }
        ).build(bundle, patch_loss=_patch_loss)

        self.assertIsInstance(runtime, StitchRuntime)
        self.assertEqual([spec.segment_id for spec in runtime.data.eval.segments], ["segA"])
        self.assertIs(runtime.train.patch_loss, _patch_loss)

    def test_boundary_and_stitch_modules_import_cleanly_in_both_orders(self):
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        pythonpath = str(repo_root)
        if env.get("PYTHONPATH"):
            pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath

        orders = (
            ("ink.recipes.losses.boundary", "ink.recipes.stitch"),
            ("ink.recipes.stitch", "ink.recipes.losses.boundary"),
        )
        for first, second in orders:
            cmd = [
                sys.executable,
                "-c",
                f"import {first}; import {second}",
            ]
            completed = subprocess.run(
                cmd,
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"failed import order {first} -> {second}:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
