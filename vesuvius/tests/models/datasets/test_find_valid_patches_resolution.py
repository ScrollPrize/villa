"""Tests for downsample-level resolution in find_valid_patches (issue #1970).

A plain zarr array has no downsampled levels, so asking for
``valid_patch_find_resolution >= 1`` must fall back to full resolution
instead of scanning the full-resolution grid as if it were downsampled.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import zarr

from vesuvius.models.datasets.find_valid_patches import find_valid_patches


# Shape of the PHerc0500P2 label volume cited in issue #1970.
SHAPE_3D = (221, 132, 132)
PATCH_3D = (64, 64, 64)


def _plain_array(data: np.ndarray, chunks: Tuple[int, ...]):
    arr = zarr.create_array(store={}, shape=data.shape, chunks=chunks, dtype=data.dtype)
    arr[...] = data
    return arr


def _multiscale_group(levels: dict, chunks: Tuple[int, ...]):
    group = zarr.open_group(store={}, mode="w")
    for key, data in levels.items():
        arr = group.create_array(key, shape=data.shape, chunks=chunks, dtype=data.dtype)
        arr[...] = data
    return group


def _starts(result, kind: str = "fg_patches"):
    return sorted(tuple(int(v) for v in p["start_pos"]) for p in result[kind])


def _assert_in_bounds(starts, shape, patch):
    for start in starts:
        for axis, (s, p, n) in enumerate(zip(start, patch, shape)):
            assert 0 <= s and s + p <= n, (
                f"patch start {start} extends past axis {axis} (size {n}) with patch {patch}"
            )


def _expected_grid(shape, patch):
    axes = [range(0, n - p + 1, p) for n, p in zip(shape, patch)]
    return sorted(tuple(int(v) for v in pos) for pos in np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(shape), -1).T)


@pytest.mark.parametrize(
    "shape,patch,chunks",
    [
        (SHAPE_3D, PATCH_3D, (64, 64, 64)),
        (SHAPE_3D[1:], PATCH_3D[1:], (64, 64)),
    ],
    ids=["3d", "2d"],
)
def test_plain_array_level1_falls_back_to_full_resolution(shape, patch, chunks):
    labels = _plain_array(np.ones(shape, dtype=np.uint8), chunks)

    result = find_valid_patches(
        [labels], ["plain"], patch, valid_patch_find_resolution=1, num_workers=1
    )
    starts = _starts(result)

    _assert_in_bounds(starts, shape, patch)
    assert (0,) * len(shape) in starts
    assert starts == _expected_grid(shape, patch)


def test_plain_array_level1_matches_level0():
    labels = _plain_array(np.ones(SHAPE_3D, dtype=np.uint8), (64, 64, 64))

    level0 = find_valid_patches(
        [labels], ["plain"], PATCH_3D, valid_patch_find_resolution=0, num_workers=1
    )
    level1 = find_valid_patches(
        [labels], ["plain"], PATCH_3D, valid_patch_find_resolution=1, num_workers=1
    )

    assert _starts(level1) == _starts(level0)


def test_multiscale_group_still_uses_requested_level():
    full = np.ones(SHAPE_3D, dtype=np.uint8)
    labels = _multiscale_group({"0": full, "1": full[::2, ::2, ::2]}, (32, 32, 32))

    result = find_valid_patches(
        [labels], ["multiscale"], PATCH_3D, valid_patch_find_resolution=1, num_workers=1
    )
    starts = _starts(result)

    _assert_in_bounds(starts, SHAPE_3D, PATCH_3D)
    # Level 1 is 111x66x66, scanned with 32^3 patches -> z starts 0,32,64 (x2 = 0,64,128).
    assert starts == _expected_grid(SHAPE_3D, PATCH_3D)


def test_unlabeled_fg_image_follows_label_fallback():
    """A plain label array falls back to level 0; a multiscale image must follow it.

    Image data occupies z < 110 at full resolution, so the patches at z=0 and
    z=64 contain image data and z=128 does not. Reading the image's level 1 on
    the full-resolution label grid would put the data at z < 55 instead.
    """
    ignore_value = 2
    labels = _plain_array(np.full(SHAPE_3D, ignore_value, dtype=np.uint8), (64, 64, 64))
    image_full = np.zeros(SHAPE_3D, dtype=np.uint8)
    image_full[:110] = 1
    image = _multiscale_group({"0": image_full, "1": image_full[::2, ::2, ::2]}, (64, 64, 64))

    result = find_valid_patches(
        [labels],
        ["plain_labels"],
        PATCH_3D,
        valid_patch_find_resolution=1,
        num_workers=1,
        ignore_labels=[ignore_value],
        image_arrays=[image],
        collect_unlabeled_fg=True,
    )
    starts = _starts(result, "unlabeled_fg_patches")

    _assert_in_bounds(starts, SHAPE_3D, PATCH_3D)
    assert sorted({s[0] for s in starts}) == [0, 64]
    assert len(starts) == 2 * 2 * 2


def test_unlabeled_only_volume_with_plain_image_array():
    image = _plain_array(np.ones(SHAPE_3D, dtype=np.uint8), (64, 64, 64))

    result = find_valid_patches(
        [None],
        ["image_only"],
        PATCH_3D,
        valid_patch_find_resolution=1,
        num_workers=1,
        image_arrays=[image],
        collect_unlabeled_fg=True,
    )
    starts = _starts(result, "unlabeled_fg_patches")

    _assert_in_bounds(starts, SHAPE_3D, PATCH_3D)
    assert starts == _expected_grid(SHAPE_3D, PATCH_3D)
