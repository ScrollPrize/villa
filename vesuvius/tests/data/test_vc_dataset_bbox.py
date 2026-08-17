"""Tests for the VCDataset bbox (region-of-interest) sliding-window restriction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.data.vc_dataset import VCDataset
from vesuvius.utils.models.helpers import compute_steps_for_sliding_window

PATCH = (64, 64, 64)
SHAPE = (300, 280, 260)
STEP = 0.5


def _full_grid(axis: int) -> list[int]:
    """The positions a full-volume run places along one axis."""
    return list(compute_steps_for_sliding_window(SHAPE[axis], PATCH[axis], STEP))


def _covers(positions, axis: int, lo: int, hi: int) -> bool:
    """True when the patches leave no gap over [lo, hi)."""
    spans = sorted((p, p + PATCH[axis]) for p in {q[axis] for q in positions})
    reach = lo
    for start, end in spans:
        if start > reach:
            return False
        reach = max(reach, end)
    return reach >= hi


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


def test_bbox_positions_come_from_the_full_volume_grid(volume_path: str) -> None:
    # The point of the flag is to redo one region of a full run. That only holds
    # if the patches are the ones the full run would have placed there; a grid
    # anchored to the ROI tiles at a different stride and blends differently.
    bbox = (100, 200, 50, 180, 0, 130)
    ds = _dataset(volume_path, bbox=bbox)
    assert ds.all_positions, "bbox run produced no patch positions"
    for axis in range(3):
        allowed = set(_full_grid(axis))
        assert {p[axis] for p in ds.all_positions} <= allowed


def test_bbox_selects_exactly_the_patches_that_touch_the_roi(volume_path: str) -> None:
    bbox = (100, 200, 50, 180, 0, 130)
    ds = _dataset(volume_path, bbox=bbox)
    got = sorted(ds.all_positions)
    expected = sorted(
        (z, y, x)
        for z in _full_grid(0) if z < bbox[1] and z + PATCH[0] > bbox[0]
        for y in _full_grid(1) if y < bbox[3] and y + PATCH[1] > bbox[2]
        for x in _full_grid(2) if x < bbox[5] and x + PATCH[2] > bbox[4]
    )
    assert got == expected


def test_bbox_positions_are_a_subset_of_a_full_run(volume_path: str) -> None:
    bbox = (100, 200, 50, 180, 0, 130)
    full = set(_dataset(volume_path).all_positions)
    roi = set(_dataset(volume_path, bbox=bbox).all_positions)
    assert roi and roi <= full


def test_bbox_covers_the_whole_roi(volume_path: str) -> None:
    # Patches may overhang the bbox - they have to, since a voxel's blended
    # value depends on every patch covering it - but no gap may be left inside.
    bbox = (100, 200, 50, 180, 0, 130)
    ds = _dataset(volume_path, bbox=bbox)
    assert _covers(ds.all_positions, 0, bbox[0], bbox[1])
    assert _covers(ds.all_positions, 1, bbox[2], bbox[3])
    assert _covers(ds.all_positions, 2, bbox[4], bbox[5])


def test_bbox_reduces_patch_count(volume_path: str) -> None:
    full = _dataset(volume_path)
    roi = _dataset(volume_path, bbox=(100, 200, 50, 180, 0, 130))
    assert len(roi.all_positions) < len(full.all_positions)


def test_open_bounds_resolve_to_volume_edges(volume_path: str) -> None:
    ds = _dataset(volume_path, bbox=(100, None, None, None, None, 130))
    assert _covers(ds.all_positions, 0, 100, SHAPE[0])
    assert _covers(ds.all_positions, 1, 0, SHAPE[1])
    assert _covers(ds.all_positions, 2, 0, 130)
    # An axis left open must reproduce the full-volume grid on that axis.
    assert sorted({p[1] for p in ds.all_positions}) == _full_grid(1)


def test_sub_patch_roi_stays_in_volume_and_on_the_grid(volume_path: str) -> None:
    roi = (290, 300, 0, 10, 250, 260)
    ds = _dataset(volume_path, bbox=roi)
    assert ds.all_positions
    for axis in range(3):
        allowed = set(_full_grid(axis))
        for pos in ds.all_positions:
            assert pos[axis] in allowed
            assert 0 <= pos[axis] and pos[axis] + PATCH[axis] <= SHAPE[axis]
    assert _covers(ds.all_positions, 0, roi[0], roi[1])
    assert _covers(ds.all_positions, 1, roi[2], roi[3])
    assert _covers(ds.all_positions, 2, roi[4], roi[5])


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
