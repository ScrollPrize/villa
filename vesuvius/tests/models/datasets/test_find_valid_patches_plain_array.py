"""A plain zarr array has no downsampled levels, so find_valid_patches must fall
back to full resolution instead of scaling full-resolution positions (#1970)."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from vesuvius.models.datasets.find_valid_patches import find_valid_patches

SHAPE = (40, 24, 24)
PATCH = (16, 16, 16)


def _labels() -> np.ndarray:
    data = np.zeros(SHAPE, dtype=np.uint8)
    data[:20] = 1  # only the patch at the origin is fully labeled
    return data


def _plain_array(data: np.ndarray):
    arr = zarr.open(mode="w", shape=data.shape, chunks=PATCH, dtype=data.dtype)
    arr[...] = data
    return arr


def _group(levels: dict):
    group = zarr.open_group(mode="w")
    for key, data in levels.items():
        create = getattr(group, "create_array", None) or group.create_dataset
        arr = create(key, shape=data.shape, chunks=PATCH, dtype=data.dtype)
        arr[...] = data
    return group


def _starts(label_obj, level: int):
    result = find_valid_patches(
        [label_obj], ["vol"], PATCH, bbox_threshold=0.15, label_threshold=0.5,
        num_workers=1, valid_patch_find_resolution=level,
    )
    return sorted(tuple(int(v) for v in p["start_pos"]) for p in result["fg_patches"])


@pytest.mark.parametrize("level", [1, 2])
def test_plain_array_falls_back_to_full_resolution(level):
    starts = _starts(_plain_array(_labels()), level)
    assert starts == [(0, 0, 0)]
    for start in starts:
        assert all(s + p <= n for s, p, n in zip(start, PATCH, SHAPE))


def test_plain_array_at_level_zero_is_unchanged():
    assert _starts(_plain_array(_labels()), 0) == [(0, 0, 0)]


def test_multiscale_group_still_uses_requested_level():
    data = _labels()
    group = _group({"0": data, "1": data[::2, ::2, ::2]})
    assert _starts(group, 1) == [(0, 0, 0)]
