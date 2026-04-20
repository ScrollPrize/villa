"""End-to-end smoke for BaseTrainer + CrossFrameZarrDataset."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader

from vesuvius.data import affine
from vesuvius.models.training.train import BaseTrainer


def _write_zarr(path: Path, arr: np.ndarray, chunks: tuple) -> None:
    z = zarr.open(str(path), mode="w", shape=arr.shape, dtype=arr.dtype, chunks=chunks)
    z[...] = arr


def _write_identity_transform(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": affine.SCHEMA_VERSION,
                "fixed_volume": "synthetic_fixed",
                "transformation_matrix": np.eye(4)[:3, :].tolist(),
                "fixed_landmarks": [],
                "moving_landmarks": [],
            }
        )
    )


def _make_synthetic_cross_frame(root: Path) -> SimpleNamespace:
    """Write a small image + binary label pair and an identity transform."""
    rng = np.random.default_rng(0)
    shape = (32, 32, 32)
    image = rng.integers(0, 255, size=shape, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.uint8)
    # 8x8x8 label region, aligned with the 16^3 patch grid, fully inside image.
    labels[8:24, 8:24, 8:24] = 255

    image_path = root / "image.zarr"
    labels_path = root / "labels.zarr"
    tform_path = root / "transform.json"
    _write_zarr(image_path, image, chunks=(16, 16, 16))
    _write_zarr(labels_path, labels, chunks=(16, 16, 16))
    _write_identity_transform(tform_path)
    return SimpleNamespace(
        image_path=image_path, labels_path=labels_path, tform_path=tform_path,
    )


def _make_mgr(paths: SimpleNamespace, cache_dir: Path) -> SimpleNamespace:
    """ConfigManager-compatible SimpleNamespace matching BaseTrainer's expectations."""
    patch_size = (16, 16, 16)
    return SimpleNamespace(
        gpu_ids=None,
        use_ddp=False,
        targets={
            "fibers": {
                "out_channels": 2,
                "activation": "none",
                "losses": [{"name": "nnUNet_DC_and_CE_loss", "weight": 1.0}],
            }
        },
        tr_configs={},
        model_name="fibers_smoke",
        train_patch_size=patch_size,
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        enable_deep_supervision=False,
        spacing=(1.0, 1.0, 1.0),
        data_path=cache_dir,
        min_labeled_ratio=0.001,
        min_bbox_percent=0.0,
        skip_patch_validation=False,
        allow_unlabeled_data=False,
        normalization_scheme="zscore",
        intensity_properties={},
        skip_intensity_sampling=True,
        no_spatial_augmentation=True,
        no_scaling_augmentation=True,
        cache_valid_patches=False,
        dataset_config={
            "dataset_type": "cross_frame",
            "image_zarr_url": str(paths.image_path),
            "labels_zarr_url": str(paths.labels_path),
            "transform_json_url": str(paths.tform_path),
            "targets": {"fibers": {"out_channels": 2, "activation": "none"}},
            "cache_dir": str(cache_dir),
        },
        ome_zarr_resolution=0,
        valid_patch_find_resolution=0,
        bg_sampling_enabled=False,
        bg_to_fg_ratio=0.5,
        unlabeled_foreground_enabled=False,
        unlabeled_foreground_threshold=0.05,
        unlabeled_foreground_bbox_threshold=0.15,
        unlabeled_foreground_volumes=None,
        profile_augmentations=False,
        guide_loss_weight=0.0,
        guide_supervision_target=None,
        train_num_dataloader_workers=0,
        model_config={
            "features_per_stage": [8, 16, 32],
            "n_stages": 3,
            "n_blocks_per_stage": [1, 1, 1],
            "n_conv_per_stage_decoder": [1, 1],
            "kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            "basic_encoder_block": "ConvBlock",
            "basic_decoder_block": "ConvBlock",
            "bottleneck_block": "BasicBlockD",
            "separate_decoders": True,
            "input_shape": list(patch_size),
        },
    )


def test_cross_frame_base_trainer_smoke_runs_three_training_steps(tmp_path: Path):
    paths = _make_synthetic_cross_frame(tmp_path)
    mgr = _make_mgr(paths, cache_dir=tmp_path)
    trainer = BaseTrainer(mgr=mgr, verbose=False)

    dataset = trainer._configure_dataset(is_training=True)
    assert len(dataset) > 0

    model = trainer._build_model().to(trainer.device)
    loss_fns = trainer._build_loss()
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        _inputs, targets_dict, outputs = trainer._get_model_outputs(model, batch)
        loss, _task_losses = trainer._compute_train_loss(outputs, targets_dict, loss_fns)
        assert torch.isfinite(loss)
        losses.append(float(loss.item()))
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
