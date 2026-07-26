"""Tests for the VCDataset bbox (region-of-interest) sliding-window restriction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.data.vc_dataset import VCDataset

PATCH = (64, 64, 64)
SHAPE = (300, 280, 260)


@pytest.fixture(scope="module")
def volume_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("vcds_bbox") / "vol.zarr"
    arr = zarr.open(str(path), mode="w", shape=SHAPE, chunks=(64, 64, 64), dtype="u1")
    rng = np.random.default_rng(97)
    arr[:] = rng.integers(0, 255, size=SHAPE, dtype=np.uint8)
    return str(path)


def _dataset(volume_path: str, **kwargs) -> VCDataset:
    return VCDataset(
        input_path=volume_path,
        patch_size=PATCH,
        mode="infer",
        skip_empty_patches=False,
        verbose=False,
        **kwargs,
    )


def test_bbox_positions_stay_global_and_inside_roi(volume_path: str) -> None:
    bbox = (100, 200, 50, 180, 0, 130)
    ds = _dataset(volume_path, bbox=bbox)
    assert ds.all_positions, "bbox run produced no patch positions"
    z0, z1, y0, y1, x0, x1 = bbox
    for pz, py, px in ds.all_positions:
        assert z0 <= pz and pz + PATCH[0] <= z1
        assert y0 <= py and py + PATCH[1] <= y1
        assert x0 <= px and px + PATCH[2] <= x1


def test_bbox_covers_roi_ends(volume_path: str) -> None:
    bbox = (100, 200, 50, 180, 0, 130)
    ds = _dataset(volume_path, bbox=bbox)
    assert min(p[0] for p in ds.all_positions) == bbox[0]
    assert max(p[0] for p in ds.all_positions) + PATCH[0] == bbox[1]
    assert max(p[1] for p in ds.all_positions) + PATCH[1] == bbox[3]
    assert max(p[2] for p in ds.all_positions) + PATCH[2] == bbox[5]


def test_bbox_reduces_patch_count(volume_path: str) -> None:
    full = _dataset(volume_path)
    roi = _dataset(volume_path, bbox=(100, 200, 50, 180, 0, 130))
    assert len(roi.all_positions) < len(full.all_positions)


def test_open_bounds_resolve_to_volume_edges(volume_path: str) -> None:
    ds = _dataset(volume_path, bbox=(100, None, None, None, None, 130))
    assert min(p[0] for p in ds.all_positions) == 100
    assert max(p[1] for p in ds.all_positions) + PATCH[1] == SHAPE[1]
    assert max(p[2] for p in ds.all_positions) + PATCH[2] == 130


def test_sub_patch_roi_grows_to_patch_and_stays_in_volume(volume_path: str) -> None:
    ds = _dataset(volume_path, bbox=(290, 300, 0, 10, 250, 260))
    assert len(ds.all_positions) == 1
    pz, py, px = ds.all_positions[0]
    assert 0 <= pz and pz + PATCH[0] <= SHAPE[0]
    assert 0 <= py and py + PATCH[1] <= SHAPE[1]
    assert 0 <= px and px + PATCH[2] <= SHAPE[2]


def test_num_parts_partitions_roi_without_loss(volume_path: str) -> None:
    bbox = (100, 200, 50, 180, 0, 130)
    single = _dataset(volume_path, bbox=bbox)
    union: list = []
    for part_id in range(3):
        part = _dataset(volume_path, bbox=bbox, num_parts=3, part_id=part_id)
        union.extend(part.all_positions)
    assert sorted(union) == sorted(single.all_positions)


@pytest.mark.parametrize(
    "bbox",
    [
        (200, 100, 0, 10, 0, 10),  # empty range
        (-5, 100, 0, 10, 0, 10),  # negative bound
        (100, 200, 0, 10),  # wrong arity
    ],
)
def test_invalid_bbox_rejected(volume_path: str, bbox) -> None:
    with pytest.raises(ValueError):
        _dataset(volume_path, bbox=bbox)


def test_bbox_outside_volume_rejected(volume_path: str) -> None:
    with pytest.raises(ValueError):
        _dataset(volume_path, bbox=(500, 600, 0, 10, 0, 10))
