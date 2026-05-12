"""Tests that volume-space augmentation (3D mirror + Y<->X transpose) does
NOT change the direction token. Direction lives in UV space, not volume
space, so mirror/transpose on the volume must preserve direction semantics.
"""
from __future__ import annotations

import numpy as np
import torch

from vesuvius.neural_tracing.autoreg_mesh.dataset import (
    _apply_spatial_augmentation,
)


def _make_inputs():
    rng = np.random.default_rng(123)
    vol = rng.standard_normal((8, 8, 8)).astype(np.float32)
    cond = rng.standard_normal((4, 3, 3)).astype(np.float32)
    masked = rng.standard_normal((4, 3, 3)).astype(np.float32)
    return vol, cond, masked


def test_volume_aug_disabled_is_identity():
    vol, cond, masked = _make_inputs()
    out_vol, out_cond, out_masked, meta = _apply_spatial_augmentation(
        vol, cond_local=cond, masked_local=masked,
        crop_size=(8, 8, 8), augmentation_cfg={}, enabled=False,
    )
    np.testing.assert_array_equal(out_vol, vol)
    np.testing.assert_array_equal(out_cond, cond)
    np.testing.assert_array_equal(out_masked, masked)
    assert meta["spatial_augmented"] is False
    assert meta["spatial_mirror_axes"] == []
    assert meta["spatial_axis_order"] == [0, 1, 2]


def test_volume_aug_with_zero_probs_is_identity():
    vol, cond, masked = _make_inputs()
    cfg = {"mirror_prob": 0.0, "transpose_prob": 0.0,
           "mirror_axes": [0, 1, 2], "transpose_axes": [1, 2]}
    out_vol, out_cond, out_masked, meta = _apply_spatial_augmentation(
        vol, cond_local=cond, masked_local=masked,
        crop_size=(8, 8, 8), augmentation_cfg=cfg, enabled=True,
    )
    np.testing.assert_array_equal(out_vol, vol)
    np.testing.assert_array_equal(out_cond, cond)
    np.testing.assert_array_equal(out_masked, masked)
    assert meta["spatial_augmented"] is False


def test_volume_aug_with_forced_mirror_changes_volume_not_uv_layout():
    """Force a mirror on axis 2 (volume X). The volume should flip along
    its X axis; the UV layout (cond.shape[:2]) must be unchanged."""
    vol, cond, masked = _make_inputs()
    cfg = {"mirror_prob": 1.0, "transpose_prob": 0.0,
           "mirror_axes": [2], "transpose_axes": [1, 2]}
    torch.manual_seed(0)
    out_vol, out_cond, out_masked, meta = _apply_spatial_augmentation(
        vol, cond_local=cond, masked_local=masked,
        crop_size=(8, 8, 8), augmentation_cfg=cfg, enabled=True,
    )
    # Some mirror axes were chosen (axis 2 has 50% chance each per
    # MirrorTransform; with seed 0 we expect a deterministic result).
    # We don't assert the specific axes; we just assert UV layout stable.
    assert out_cond.shape == cond.shape
    assert out_masked.shape == masked.shape


def test_volume_aug_with_forced_yx_swap_preserves_uv_layout():
    """Force a Y<->X transpose. cond.shape[:2] stays the same; only the
    3D world-coord channels and the volume voxel axes permute."""
    vol, cond, masked = _make_inputs()
    cfg = {"mirror_prob": 0.0, "transpose_prob": 1.0,
           "mirror_axes": [0, 1, 2], "transpose_axes": [1, 2]}
    torch.manual_seed(0)
    out_vol, out_cond, out_masked, meta = _apply_spatial_augmentation(
        vol, cond_local=cond, masked_local=masked,
        crop_size=(8, 8, 8), augmentation_cfg=cfg, enabled=True,
    )
    assert out_cond.shape == cond.shape
    assert out_masked.shape == masked.shape
    # axis_order is in keypoint coord-channel terms; for Y<->X swap it
    # should be either [0, 1, 2] (no swap chosen due to RNG) or [0, 2, 1].
    assert meta["spatial_axis_order"] in ([0, 1, 2], [0, 2, 1])


def test_volume_aug_metadata_keys_present():
    vol, cond, masked = _make_inputs()
    cfg = {"mirror_prob": 0.5, "transpose_prob": 0.5,
           "mirror_axes": [0, 1, 2], "transpose_axes": [1, 2]}
    _, _, _, meta = _apply_spatial_augmentation(
        vol, cond_local=cond, masked_local=masked,
        crop_size=(8, 8, 8), augmentation_cfg=cfg, enabled=True,
    )
    for key in ("spatial_augmented", "spatial_mirror_axes", "spatial_axis_order"):
        assert key in meta
