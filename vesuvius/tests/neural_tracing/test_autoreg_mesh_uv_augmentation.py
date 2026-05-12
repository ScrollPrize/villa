"""Tests for the UV-grid (2D parametrization) augmentation in autoreg_mesh.

The UV transpose flips the rows<->cols layout of the conditioning surface
WITHOUT touching the 3D volume voxels or the world-coord channel order.
Direction tokens index UV axes (left/right = UV-cols, up/down = UV-rows),
so they must be remapped consistently: left<->up, right<->down.
"""
from __future__ import annotations

import numpy as np
import pytest

from vesuvius.neural_tracing.autoreg_mesh.dataset import (
    _UV_TRANSPOSE_DIRECTION_REMAP,
    _apply_uv_grid_augmentation,
)
from vesuvius.neural_tracing.autoreg_mesh.serialization import (
    extract_frontier_prompt_band,
)


def _square_cond_masked(direction: str, full_size: int, cond_size: int):
    """Build cond + masked arrays so that the full UV surface is square."""
    surface = np.zeros((full_size, full_size, 3), dtype=np.float32)
    rows, cols = np.indices((full_size, full_size), dtype=np.float32)
    surface[..., 0] = 100.0 + rows               # Z channel (arbitrary fill)
    surface[..., 1] = 200.0 + cols               # Y channel
    surface[..., 2] = 300.0 + (rows * cols)      # X channel
    if direction == "left":
        cond = surface[:, :cond_size, :]
        masked = surface[:, cond_size:, :]
    elif direction == "right":
        cond = surface[:, -cond_size:, :]
        masked = surface[:, :-cond_size, :]
    elif direction == "up":
        cond = surface[:cond_size, :, :]
        masked = surface[cond_size:, :, :]
    elif direction == "down":
        cond = surface[-cond_size:, :, :]
        masked = surface[:-cond_size, :, :]
    else:
        raise AssertionError(direction)
    return cond.copy(), masked.copy(), surface


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_uv_transpose_swaps_direction_for_all(direction: str):
    cond, masked, _ = _square_cond_masked(direction, full_size=8, cond_size=3)
    cfg = {"uv_transpose_prob": 1.0}
    out_cond, out_masked, new_dir, meta = _apply_uv_grid_augmentation(
        cond, masked, direction=direction, augmentation_cfg=cfg, enabled=True
    )
    assert meta["uv_transposed"] is True
    assert new_dir == _UV_TRANSPOSE_DIRECTION_REMAP[direction]
    # UV layout is transposed (rows<->cols); world-coord channels unchanged.
    assert out_cond.shape == (cond.shape[1], cond.shape[0], 3)
    assert out_masked.shape == (masked.shape[1], masked.shape[0], 3)
    np.testing.assert_array_equal(out_cond, cond.swapaxes(0, 1))
    np.testing.assert_array_equal(out_masked, masked.swapaxes(0, 1))


def test_uv_transpose_world_coords_unchanged():
    """World-coord channel values must be permuted in UV layout but identical
    in value (no Z/Y/X swap in the channel dim)."""
    cond, masked, _ = _square_cond_masked("left", full_size=6, cond_size=2)
    cfg = {"uv_transpose_prob": 1.0}
    out_cond, out_masked, _, _ = _apply_uv_grid_augmentation(
        cond, masked, direction="left", augmentation_cfg=cfg, enabled=True
    )
    # For each (r, c) in cond -> swapped (c, r); the 3-channel ZYX tuple
    # at that position must match elementwise.
    for r in range(cond.shape[0]):
        for c in range(cond.shape[1]):
            np.testing.assert_array_equal(out_cond[c, r, :], cond[r, c, :])
    for r in range(masked.shape[0]):
        for c in range(masked.shape[1]):
            np.testing.assert_array_equal(out_masked[c, r, :], masked[r, c, :])


def test_uv_transpose_skipped_when_non_square():
    """If the full UV grid (cond + masked) is non-square along the split
    axis, the transpose must be a no-op."""
    rng = np.random.default_rng(0)
    # direction=left -> split on cols: full = (H, cond_cols + masked_cols)
    # Pick H != cond_cols + masked_cols so the full surface is non-square.
    cond = rng.standard_normal((5, 2, 3)).astype(np.float32)
    masked = rng.standard_normal((5, 4, 3)).astype(np.float32)
    cfg = {"uv_transpose_prob": 1.0}
    out_cond, out_masked, new_dir, meta = _apply_uv_grid_augmentation(
        cond, masked, direction="left", augmentation_cfg=cfg, enabled=True
    )
    assert meta["uv_transposed"] is False
    assert new_dir == "left"
    np.testing.assert_array_equal(out_cond, cond)
    np.testing.assert_array_equal(out_masked, masked)


def test_uv_transpose_disabled_is_noop():
    cond, masked, _ = _square_cond_masked("up", full_size=4, cond_size=1)
    cfg = {"uv_transpose_prob": 1.0}
    out_cond, out_masked, new_dir, meta = _apply_uv_grid_augmentation(
        cond, masked, direction="up", augmentation_cfg=cfg, enabled=False
    )
    assert meta["uv_transposed"] is False
    assert new_dir == "up"
    assert out_cond is cond
    assert out_masked is masked


def test_uv_transpose_prob_zero_is_noop():
    cond, masked, _ = _square_cond_masked("down", full_size=4, cond_size=1)
    cfg = {"uv_transpose_prob": 0.0}
    _, _, new_dir, meta = _apply_uv_grid_augmentation(
        cond, masked, direction="down", augmentation_cfg=cfg, enabled=True
    )
    assert meta["uv_transposed"] is False
    assert new_dir == "down"


def test_uv_transpose_frequency_matches_prob():
    """Empirical: forcing prob=0.5 should give ~50% transpose over many trials."""
    cond, masked, _ = _square_cond_masked("right", full_size=4, cond_size=1)
    cfg = {"uv_transpose_prob": 0.5}
    n_trials = 2000
    n_swapped = 0
    import torch
    torch.manual_seed(0)
    for _ in range(n_trials):
        _, _, _, meta = _apply_uv_grid_augmentation(
            cond, masked, direction="right", augmentation_cfg=cfg, enabled=True
        )
        if bool(meta["uv_transposed"]):
            n_swapped += 1
    rate = n_swapped / n_trials
    assert 0.45 <= rate <= 0.55, f"expected ~0.5 swap rate over {n_trials} trials, got {rate}"


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_frontier_band_after_uv_transpose(direction: str):
    """The frontier band extracted from the transposed cond with the remapped
    direction should equal the transposed band extracted from the original
    cond with the original direction.
    """
    cond, masked, _ = _square_cond_masked(direction, full_size=8, cond_size=3)
    cfg = {"uv_transpose_prob": 1.0}
    out_cond, _, new_dir, meta = _apply_uv_grid_augmentation(
        cond, masked, direction=direction, augmentation_cfg=cfg, enabled=True
    )
    assert meta["uv_transposed"] is True
    band_original = extract_frontier_prompt_band(cond, direction=direction, frontier_band_width=1)
    band_after = extract_frontier_prompt_band(out_cond, direction=new_dir, frontier_band_width=1)
    # Both bands are slabs of width=1; same set of cells, just laid out under
    # the swapped UV axes. The "after" band should be the original band
    # transposed across UV axes.
    np.testing.assert_array_equal(band_after, band_original.swapaxes(0, 1))


def test_uv_transpose_direction_remap_is_4cycle():
    """Applying the remap twice should return the original direction
    (because swapaxes is its own inverse on a 2D UV grid)."""
    for d in ("left", "right", "up", "down"):
        once = _UV_TRANSPOSE_DIRECTION_REMAP[d]
        twice = _UV_TRANSPOSE_DIRECTION_REMAP[once]
        assert twice == d, f"direction remap is not involutive: {d} -> {once} -> {twice}"


def test_uv_transpose_metadata_present_when_disabled():
    """When disabled, metadata still reports uv_transposed=False (consumers
    rely on this key existing)."""
    cond, masked, _ = _square_cond_masked("left", full_size=4, cond_size=1)
    _, _, _, meta = _apply_uv_grid_augmentation(
        cond, masked, direction="left", augmentation_cfg={"uv_transpose_prob": 1.0}, enabled=False
    )
    assert "uv_transposed" in meta
    assert meta["uv_transposed"] is False
