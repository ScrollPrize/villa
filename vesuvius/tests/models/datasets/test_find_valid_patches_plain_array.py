"""A plain zarr array has no downsampled levels, so find_valid_patches must fall
back to full resolution instead of scaling full-resolution positions (#1970)."""

from __future__ import annotations

import numpy as np
import pytest
from vc_reader.reader import ZarrWriter, create_group, open_group, open_zarr

from vesuvius.models.datasets.find_valid_patches import find_valid_patches

SHAPE = (40, 24, 24)
PATCH = (16, 16, 16)


def _labels() -> np.ndarray:
    data = np.zeros(SHAPE, dtype=np.uint8)
    data[:20] = 1  # only the patch at the origin is fully labeled
    return data


def _plain_array(path, data: np.ndarray):
    ZarrWriter.create(path / "plain.zarr", shape=data.shape, chunks=PATCH, dtype=data.dtype)[...] = data
    return open_zarr(path / "plain.zarr")


def _group(path, levels: dict):
    group = create_group(path / "group.zarr")
    for key, data in levels.items():
        group.create_array(key, shape=data.shape, chunks=PATCH, dtype=data.dtype)[...] = data
    return open_group(path / "group.zarr")


def _starts(label_obj, level: int):
    result = find_valid_patches(
        [label_obj], ["vol"], PATCH, bbox_threshold=0.15, label_threshold=0.5,
        num_workers=1, valid_patch_find_resolution=level,
    )
    return sorted(tuple(int(v) for v in p["start_pos"]) for p in result["fg_patches"])


@pytest.mark.parametrize("level", [1, 2])
def test_plain_array_falls_back_to_full_resolution(tmp_path, level):
    starts = _starts(_plain_array(tmp_path, _labels()), level)
    assert starts == [(0, 0, 0)]
    for start in starts:
        assert all(s + p <= n for s, p, n in zip(start, PATCH, SHAPE))


def test_plain_array_at_level_zero_is_unchanged(tmp_path):
    assert _starts(_plain_array(tmp_path, _labels()), 0) == [(0, 0, 0)]


def test_multiscale_group_still_uses_requested_level(tmp_path):
    data = _labels()
    group = _group(tmp_path, {"0": data, "1": data[::2, ::2, ::2]})
    assert _starts(group, 1) == [(0, 0, 0)]
