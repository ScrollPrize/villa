"""Tests for Volume.meta() across the store layouts Volume accepts."""

from __future__ import annotations

import numpy as np
import zarr

from vesuvius.data.volume import Volume


def _add_level(group, name: str, shape) -> None:
    """Add a resolution level to a group under either supported zarr version.

    zarr 3 renamed Group.create_dataset to Group.create_array; 2.x only has the
    former, and CI runs both.
    """
    create = getattr(group, "create_array", None) or group.create_dataset
    create(name, shape=shape, chunks=(8, 8, 8), dtype="u1")


def _make_volume(data) -> Volume:
    """Build a Volume without running __init__ (which needs network/config)."""
    vol = Volume.__new__(Volume)
    vol.type = "zarr"
    vol.scroll_id = None
    vol.segment_id = None
    vol.energy = None
    vol.resolution = None
    vol.url = "memory://test.zarr"
    vol.dtype = np.dtype("uint8")
    vol.normalization_scheme = "none"
    vol.global_mean = None
    vol.global_std = None
    vol.return_as_type = "np.float32"
    vol.return_as_tensor = False
    vol.inklabel = None
    vol.data = data
    return vol


def test_meta_reports_single_array_without_len(tmp_path, capsys) -> None:
    arr = zarr.open(str(tmp_path / "single.zarr"), mode="w",
                    shape=(16, 16, 16), chunks=(8, 8, 8), dtype="u1")
    _make_volume(arr).meta()
    out = capsys.readouterr().out
    assert "Number of Resolution Levels: 1" in out
    assert "Level 0 Shape: (16, 16, 16)" in out


def test_meta_reports_multiscale_group(tmp_path, capsys) -> None:
    root = zarr.open_group(str(tmp_path / "multi.zarr"), mode="w")
    _add_level(root, "0", (16, 16, 16))
    _add_level(root, "1", (8, 8, 8))
    _make_volume(root).meta()
    out = capsys.readouterr().out
    assert "Number of Resolution Levels: 2" in out
    assert "Level 0 Shape: (16, 16, 16)" in out
    assert "Level 1 Shape: (8, 8, 8)" in out


def test_meta_reports_sequence_of_arrays(tmp_path, capsys) -> None:
    arrays = [
        zarr.open(str(tmp_path / "a.zarr"), mode="w", shape=(16, 16, 16),
                  chunks=(8, 8, 8), dtype="u1"),
        zarr.open(str(tmp_path / "b.zarr"), mode="w", shape=(8, 8, 8),
                  chunks=(8, 8, 8), dtype="u1"),
    ]
    _make_volume(arrays).meta()
    out = capsys.readouterr().out
    assert "Number of Resolution Levels: 2" in out
