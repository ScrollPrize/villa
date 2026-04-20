"""Network-marked preflight: open real S3 + HTTPS + transform.json and
enumerate a small slice of foreground patches. Skipped by default; run with:

    uv run pytest -m 'slow and network' \
        tests/models/datasets/test_cross_frame_realdata.py -v -s

The test is deliberately small-footprint: it caps the label-volume scan to
a few chunk rows so it completes in a minute or two even with moderate
network latency.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pytest

from vesuvius.data import affine
from vesuvius.data.utils import open_zarr
from vesuvius.models.datasets.cross_frame_dataset import (
    CrossFrameZarrDataset,
    _resolve_zarr_array,
)


IMAGE_URL = "s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0"
LABELS_URL = "https://dl.ash2txt.org/other/dev/meshes/s1a-fibers-230125-ome.zarr/0"
TRANSFORM_URL = "s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/transform.json"


pytestmark = [pytest.mark.slow, pytest.mark.network]


def _make_mgr(cache_dir: Path, *, patch_size=(128, 128, 128)) -> SimpleNamespace:
    return SimpleNamespace(
        train_patch_size=patch_size,
        targets={"fibers": {"out_channels": 2, "activation": "none"}},
        min_labeled_ratio=0.01,
        min_bbox_percent=0.15,
        normalization_scheme="zscore",
        intensity_properties={},
        no_spatial_augmentation=True,
        no_scaling_augmentation=True,
        data_path=cache_dir,
        dataset_config={
            "dataset_type": "cross_frame",
            "image_zarr_url": IMAGE_URL,
            "labels_zarr_url": LABELS_URL,
            "transform_json_url": TRANSFORM_URL,
            "storage_options": {"anon": True},
            "cache_dir": str(cache_dir / ".cross_frame_cache"),
        },
    )


def test_remote_zarrs_and_transform_are_reachable():
    image = _resolve_zarr_array(open_zarr(IMAGE_URL, mode="r", storage_options={"anon": True}))
    labels = _resolve_zarr_array(open_zarr(LABELS_URL, mode="r"))
    doc = affine.read_transform_json(TRANSFORM_URL)

    assert image.shape == (75784, 32693, 32693), f"unexpected image shape {image.shape}"
    assert labels.shape == (14376, 7888, 8096), f"unexpected labels shape {labels.shape}"
    assert doc.matrix_xyz.shape == (4, 4)
    print(f"\nimage  shape={image.shape} dtype={image.dtype}")
    print(f"labels shape={labels.shape} dtype={labels.dtype}")
    print(f"transform_xyz=\n{doc.matrix_xyz}")


def test_foreground_scan_over_small_region(tmp_path: Path):
    """Scan ~1% of the label volume for FG chunks; report counts and timing.

    We hack the scan by monkeypatching the labels volume view to a small
    z-slice so the test doesn't touch the whole 900 GB array.
    """
    cache_dir = tmp_path / "cache"
    mgr = _make_mgr(cache_dir=tmp_path)

    # Build dataset normally to open arrays + transform, then swap the labels
    # array for a small z-slab before calling the scanner.
    # We construct the dataset shell manually to avoid its auto-scan.
    ds = object.__new__(CrossFrameZarrDataset)
    ds.mgr = mgr
    ds.is_training = False
    ds.patch_size = tuple(mgr.train_patch_size)
    ds.stride = ds.patch_size
    ds.target_names = ["fibers"]
    ds.target_name = "fibers"
    ds.targets = mgr.targets
    ds.valid_patch_value = None
    ds.min_labeled_ratio = mgr.min_labeled_ratio
    ds.min_bbox_percent = mgr.min_bbox_percent
    ds.image_zarr_url = IMAGE_URL
    ds.labels_zarr_url = LABELS_URL
    ds.transform_json_url = TRANSFORM_URL
    ds.storage_options_image = {"anon": True}
    ds.storage_options_labels = {}

    ds._image_array = _resolve_zarr_array(
        open_zarr(IMAGE_URL, mode="r", storage_options={"anon": True})
    )
    ds._labels_array = _resolve_zarr_array(open_zarr(LABELS_URL, mode="r"))
    ds._image_shape = tuple(int(v) for v in ds._image_array.shape[-3:])
    ds._labels_shape = tuple(int(v) for v in ds._labels_array.shape[-3:])

    doc = affine.read_transform_json(TRANSFORM_URL)
    ds._transform_doc = doc
    ds._matrix_xyz = doc.matrix_xyz
    ds._matrix_zyx_label_to_image = affine.label_to_image_zyx_matrix(doc.matrix_xyz)

    # Limit the scan to roughly 1% of the label volume: 2 patch-rows on z.
    dz = ds.patch_size[0]
    limited_shape = (min(2 * dz, ds._labels_shape[0]),) + ds._labels_shape[1:]
    print(f"\nScanning labels region {limited_shape} (of {ds._labels_shape})")
    ds._labels_shape = limited_shape

    t0 = time.perf_counter()
    positions = ds._scan_foreground_positions()
    elapsed = time.perf_counter() - t0
    print(f"Scan found {len(positions)} FG patches in {elapsed:.1f} s")
    # Do not hard-assert a specific count — fiber density varies per z.
    assert isinstance(positions, list)


def test_single_patch_fetch_end_to_end(tmp_path: Path):
    """Open, find ONE FG patch, fetch both label and resampled image."""
    cache_dir = tmp_path / "cache"
    mgr = _make_mgr(cache_dir=tmp_path)

    labels = _resolve_zarr_array(open_zarr(LABELS_URL, mode="r"))
    image = _resolve_zarr_array(open_zarr(IMAGE_URL, mode="r", storage_options={"anon": True}))
    matrix = affine.label_to_image_zyx_matrix(
        affine.read_transform_json(TRANSFORM_URL).matrix_xyz
    )
    ps = (128, 128, 128)

    # Walk through the label volume on a coarse stride to find one FG patch
    # whose image AABB is inside bounds. The transform can translate/rotate
    # heavily, so scan a wider area before giving up.
    image_shape = tuple(int(v) for v in image.shape)
    dz, dy, dx = ps
    stride = (4 * dz, 4 * dy, 4 * dx)  # sample every 512 voxels coarsely
    found: Tuple[int, int, int] | None = None
    for z in range(0, labels.shape[0] - dz, stride[0]):
        for y in range(0, labels.shape[1] - dy, stride[1]):
            for x in range(0, labels.shape[2] - dx, stride[2]):
                start, stop = affine.label_patch_image_aabb(
                    matrix, (z, y, x), ps, image_shape_zyx=None, margin=0,
                )
                if any(s < 0 for s in start):
                    continue
                if any(e > limit for e, limit in zip(stop, image_shape)):
                    continue
                slab = np.asarray(labels[z:z + dz, y:y + dy, x:x + dx])
                if slab.size == 0 or not np.any(slab > 0):
                    continue
                found = (z, y, x)
                break
            if found:
                break
        if found:
            break

    if found is None:
        pytest.skip("No FG+in-bounds patch found in the sampled region")

    t0 = time.perf_counter()
    image_patch = affine.resample_image_to_label_grid(image, matrix, found, ps)
    elapsed = time.perf_counter() - t0
    assert image_patch.shape == ps
    print(f"\nFound FG patch at {found}; resample took {elapsed*1000:.1f} ms")
