"""End-to-end smoke tests for the CT-gated 3D ink label generator.

These tests verify the contract reachable without ground-truth ink labels:

- A bright ridge in a synthetic CT volume produces non-zero labels along the
  ridge when the 2D ink prediction agrees with the location.
- The surface-window refinement shrinks the labelled volume relative to
  baseline.
- The connected-component filter drops below-threshold specks.
- Output zarr matches the input spatial shape and uint8 dtype.

These are smoke tests, not quality tests. The script's status is "sketch"
until validated against real annotation in #192.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import generate_3d_ink_labels as gli  # noqa: E402  (path injected by conftest)


def _write_zarr(path: str, arr: np.ndarray) -> None:
    import zarr

    z = zarr.open(path, mode="w", shape=arr.shape, chunks=arr.shape, dtype=arr.dtype, fill_value=0)
    z[:] = arr


def _make_inputs(tmp_path, ct_volume: np.ndarray, ink_pred_2d: np.ndarray) -> tuple[str, str]:
    """Write a CT volume + matching 2D ink prediction; return (ct_path, ink_path).

    The production contract is that the ink prediction's (H, W) equals the
    bbox's (h, w). These fixtures honour that.
    """
    assert ink_pred_2d.shape == ct_volume.shape[1:], (
        f"ink prediction {ink_pred_2d.shape} must match CT yx {ct_volume.shape[1:]}"
    )
    ct_path = str(tmp_path / "ct.zarr")
    ink_path = str(tmp_path / "ink.zarr")
    _write_zarr(ct_path, ct_volume)
    _write_zarr(ink_path, ink_pred_2d)
    return ct_path, ink_path


def test_baseline_produces_nonzero_labels(tmp_path):
    rng = np.random.default_rng(0)
    ct_volume = rng.integers(0, 40, size=(32, 32, 32), dtype=np.uint8)
    # Bright ridge along x at z=16, y=14..18, x=8..24 — simulates ink material
    ct_volume[14:18, 14:18, 8:24] = 220
    ink_pred = np.zeros((32, 32), dtype=np.uint8)
    ink_pred[12:20, 6:26] = 200  # >= 0.5 when scaled
    ct_path, ink_path = _make_inputs(tmp_path, ct_volume, ink_pred)
    out_path = str(tmp_path / "labels.zarr")

    result = gli.generate_3d_ink_labels(
        ct_path,
        ink_path,
        out_path,
        z0=0, z1=32, y0=0, y1=32, x0=0, x1=32,
        ink_threshold=0.5,
        ct_percentile=85.0,
    )
    assert os.path.exists(out_path)
    assert result["labeled_voxels"] > 0
    assert 0.0 < result["label_fraction"] < 1.0
    assert result["surface_active_fraction"] > 0.0


def test_surface_window_reduces_labels(tmp_path):
    rng = np.random.default_rng(1)
    ct_volume = rng.integers(0, 40, size=(32, 32, 32), dtype=np.uint8)
    # Add several scattered "ink" voxels in z to make the bulk-vs-surface
    # distinction observable.
    ct_volume[16, 14:18, 10:22] = 220
    ct_volume[4, 14:18, 10:22] = 220   # far from z=16 — bulk noise
    ct_volume[28, 14:18, 10:22] = 220  # far from z=16 — bulk noise
    ink_pred = np.zeros((32, 32), dtype=np.uint8)
    ink_pred[12:20, 8:24] = 200
    ct_path, ink_path = _make_inputs(tmp_path, ct_volume, ink_pred)

    base = gli.generate_3d_ink_labels(
        ct_path, ink_path, str(tmp_path / "base.zarr"),
        z0=0, z1=32, y0=0, y1=32, x0=0, x1=32,
        ink_threshold=0.5, ct_percentile=70.0,
    )
    sw = gli.generate_3d_ink_labels(
        ct_path, ink_path, str(tmp_path / "sw.zarr"),
        z0=0, z1=32, y0=0, y1=32, x0=0, x1=32,
        ink_threshold=0.5, ct_percentile=70.0, surface_window=2,
    )
    assert sw["labeled_voxels"] <= base["labeled_voxels"]
    # The "bulk" voxels at z=4 and z=28 should be dropped.
    assert sw["voxels_dropped_off_surface"] > 0


def test_cc_filter_drops_isolated_voxels(tmp_path):
    """Sparse single-voxel specks should be dropped by a min-component filter.

    We need realistic background noise so the per-column CT-percentile gate is
    meaningful — without it, an all-zero column makes any threshold trivially
    met everywhere, defeating the test.
    """
    rng = np.random.default_rng(7)
    # Realistic background: low-amplitude noise everywhere
    ct_volume = rng.integers(0, 30, size=(16, 16, 16), dtype=np.uint8)
    # Three isolated bright "ink" voxels at z=8
    ct_volume[8, 4, 4] = 250
    ct_volume[8, 12, 12] = 250
    ct_volume[8, 8, 8] = 250
    # Ink prediction positive only at those three (y, x) locations
    ink_pred = np.zeros((16, 16), dtype=np.uint8)
    ink_pred[4, 4] = 200
    ink_pred[12, 12] = 200
    ink_pred[8, 8] = 200
    ct_path, ink_path = _make_inputs(tmp_path, ct_volume, ink_pred)

    result = gli.generate_3d_ink_labels(
        ct_path, ink_path, str(tmp_path / "labels.zarr"),
        z0=0, z1=16, y0=0, y1=16, x0=0, x1=16,
        ink_threshold=0.5, ct_percentile=99.0,
        min_component_voxels=4,
    )
    assert result["labeled_voxels"] == 0
    assert result["connected_components_total"] > 0
    assert result["connected_components_kept"] == 0


def test_output_zarr_matches_input_shape(tmp_path):
    import zarr

    rng = np.random.default_rng(2)
    ct_volume = rng.integers(0, 40, size=(32, 32, 32), dtype=np.uint8)
    ct_volume[16, 14:18, 8:24] = 220
    # The bbox is 12x12 in y/x, so the ink prediction must be 12x12.
    ink_pred = np.full((12, 12), 200, dtype=np.uint8)
    ct_path = str(tmp_path / "ct.zarr")
    ink_path = str(tmp_path / "ink.zarr")
    _write_zarr(ct_path, ct_volume)
    _write_zarr(ink_path, ink_pred)

    gli.generate_3d_ink_labels(
        ct_path, ink_path, str(tmp_path / "labels.zarr"),
        z0=8, z1=20, y0=8, y1=20, x0=8, x1=20,
        ink_threshold=0.5,
    )
    out = zarr.open(str(tmp_path / "labels.zarr"), mode="r")
    assert tuple(out.shape) == (32, 32, 32)
    assert out.dtype == np.uint8
    # Unwritten region must remain zero.
    assert int(np.asarray(out[:8, :, :]).sum()) == 0
