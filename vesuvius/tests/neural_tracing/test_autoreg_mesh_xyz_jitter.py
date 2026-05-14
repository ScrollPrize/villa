"""Tests for the XYZ-jitter drift-recovery augmentation.

The helper `_maybe_apply_xyz_jitter` in train.py samples per-element uniform
noise in [-J, +J] voxels and ADDS it to the previous-token xyz INPUTS
(`batch["target_xyz"]` and `batch["prompt_tokens"]["xyz"]`). It must NOT
mutate the original batch dict so the loss computation later in the same
training step reads the unjittered ground-truth targets.
"""
from __future__ import annotations

import pytest
import torch

from vesuvius.neural_tracing.autoreg_mesh.train import _maybe_apply_xyz_jitter


def _make_batch(B: int = 2, target_len: int = 16, prompt_len: int = 8):
    target_xyz = torch.rand(B, target_len, 3) * 100.0
    sup_mask = torch.ones(B, target_len, dtype=torch.bool)
    # mark the last 3 target positions invalid (padding-like)
    sup_mask[:, -3:] = False
    prompt_xyz = torch.rand(B, prompt_len, 3) * 100.0
    prompt_valid = torch.ones(B, prompt_len, dtype=torch.bool)
    prompt_valid[:, -2:] = False
    prompt_tokens = {
        "xyz": prompt_xyz,
        "valid_mask": prompt_valid,
    }
    batch = {
        "target_xyz": target_xyz,
        "target_supervision_mask": sup_mask,
        "prompt_tokens": prompt_tokens,
    }
    return batch


def test_xyz_jitter_disabled_is_identity():
    batch = _make_batch()
    cfg = {"xyz_jitter_enabled": False, "xyz_jitter_max_voxels": 1.5}
    out = _maybe_apply_xyz_jitter(batch, cfg, global_step=0)
    # Disabled: returns the SAME dict object (not even a copy).
    assert out is batch


def test_xyz_jitter_off_before_start_step():
    batch = _make_batch()
    cfg = {"xyz_jitter_enabled": True, "xyz_jitter_max_voxels": 1.5,
           "xyz_jitter_start_step": 5000}
    out = _maybe_apply_xyz_jitter(batch, cfg, global_step=4999)
    assert out is batch
    # After start_step it kicks in (different object, modified xyz).
    out2 = _maybe_apply_xyz_jitter(batch, cfg, global_step=5000)
    assert out2 is not batch
    assert not torch.equal(out2["target_xyz"], batch["target_xyz"])


def test_xyz_jitter_preserves_original_batch():
    batch = _make_batch()
    target_xyz_before = batch["target_xyz"].clone()
    prompt_xyz_before = batch["prompt_tokens"]["xyz"].clone()
    cfg = {"xyz_jitter_enabled": True, "xyz_jitter_max_voxels": 1.5,
           "xyz_jitter_start_step": 0}
    out = _maybe_apply_xyz_jitter(batch, cfg, global_step=1000)
    # Returned dict is NEW.
    assert out is not batch
    assert out["prompt_tokens"] is not batch["prompt_tokens"]
    # Original tensors UNTOUCHED.
    assert torch.equal(batch["target_xyz"], target_xyz_before)
    assert torch.equal(batch["prompt_tokens"]["xyz"], prompt_xyz_before)
    # Returned values differ from originals at supervised positions.
    assert not torch.equal(out["target_xyz"], batch["target_xyz"])
    assert not torch.equal(out["prompt_tokens"]["xyz"], batch["prompt_tokens"]["xyz"])


def test_xyz_jitter_magnitude_bounded():
    batch = _make_batch(B=4, target_len=64, prompt_len=16)
    cfg = {"xyz_jitter_enabled": True, "xyz_jitter_max_voxels": 1.5,
           "xyz_jitter_start_step": 0}
    out = _maybe_apply_xyz_jitter(batch, cfg, global_step=0)
    target_diff = (out["target_xyz"] - batch["target_xyz"]).abs()
    prompt_diff = (out["prompt_tokens"]["xyz"] - batch["prompt_tokens"]["xyz"]).abs()
    # Per-element absolute deviation must be <= max_voxels.
    assert target_diff.max().item() <= 1.5 + 1e-6
    assert prompt_diff.max().item() <= 1.5 + 1e-6


def test_xyz_jitter_zero_at_invalid_positions():
    batch = _make_batch()
    cfg = {"xyz_jitter_enabled": True, "xyz_jitter_max_voxels": 1.5,
           "xyz_jitter_start_step": 0}
    out = _maybe_apply_xyz_jitter(batch, cfg, global_step=0)
    sup_mask = batch["target_supervision_mask"]
    invalid_target = ~sup_mask
    invalid_prompt = ~batch["prompt_tokens"]["valid_mask"]
    # Invalid positions have ZERO jitter (jitter scaled by mask).
    target_diff = (out["target_xyz"] - batch["target_xyz"])[invalid_target]
    prompt_diff = (out["prompt_tokens"]["xyz"] - batch["prompt_tokens"]["xyz"])[invalid_prompt]
    assert torch.all(target_diff == 0.0)
    assert torch.all(prompt_diff == 0.0)


def test_xyz_jitter_resamples_each_call():
    """Two calls with the same step should produce different noise (the
    helper does NOT seed for reproducibility; the dataloader/sampler does)."""
    batch = _make_batch()
    cfg = {"xyz_jitter_enabled": True, "xyz_jitter_max_voxels": 1.5,
           "xyz_jitter_start_step": 0}
    torch.manual_seed(0)
    out_a = _maybe_apply_xyz_jitter(batch, cfg, global_step=42)
    # Different seed -> different noise.
    torch.manual_seed(1)
    out_b = _maybe_apply_xyz_jitter(batch, cfg, global_step=42)
    assert not torch.equal(out_a["target_xyz"], out_b["target_xyz"])


def test_xyz_jitter_zero_max_voxels_is_noop():
    batch = _make_batch()
    cfg = {"xyz_jitter_enabled": True, "xyz_jitter_max_voxels": 0.0,
           "xyz_jitter_start_step": 0}
    out = _maybe_apply_xyz_jitter(batch, cfg, global_step=0)
    assert out is batch


def test_config_validator_rejects_bad_xyz_jitter():
    from vesuvius.neural_tracing.autoreg_mesh.config import (
        DEFAULT_AUTOREG_MESH_CONFIG,
        validate_autoreg_mesh_config,
    )
    # Negative voxels
    bad = {**DEFAULT_AUTOREG_MESH_CONFIG, "datasets": [],
           "xyz_jitter_max_voxels": -1.0}
    with pytest.raises(ValueError, match="xyz_jitter_max_voxels must be >= 0"):
        validate_autoreg_mesh_config(bad)
    # Fat-finger cap
    bad2 = {**DEFAULT_AUTOREG_MESH_CONFIG, "datasets": [],
            "xyz_jitter_max_voxels": 50.0}
    with pytest.raises(ValueError, match="fat-finger"):
        validate_autoreg_mesh_config(bad2)
    # Non-bool enabled
    bad3 = {**DEFAULT_AUTOREG_MESH_CONFIG, "datasets": [],
            "xyz_jitter_enabled": "yes"}
    with pytest.raises(ValueError, match="xyz_jitter_enabled must be a boolean"):
        validate_autoreg_mesh_config(bad3)
