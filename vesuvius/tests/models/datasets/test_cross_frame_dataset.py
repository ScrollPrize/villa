"""Tests for vesuvius.models.datasets.cross_frame_dataset."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pytest
import zarr

from vesuvius.data import affine
from vesuvius.models.datasets.cross_frame_dataset import CrossFrameZarrDataset


def _write_identity_transform(path: Path) -> None:
    payload = {
        "schema_version": affine.SCHEMA_VERSION,
        "fixed_volume": "synthetic_fixed",
        "transformation_matrix": np.eye(4)[:3, :].tolist(),
        "fixed_landmarks": [],
        "moving_landmarks": [],
    }
    path.write_text(json.dumps(payload))


def _write_transform(path: Path, matrix_xyz: np.ndarray) -> None:
    payload = {
        "schema_version": affine.SCHEMA_VERSION,
        "fixed_volume": "synthetic_fixed",
        "transformation_matrix": matrix_xyz[:3, :].tolist(),
        "fixed_landmarks": [],
        "moving_landmarks": [],
    }
    path.write_text(json.dumps(payload))


def _write_zarr_array(path: Path, arr: np.ndarray, chunks: Tuple[int, int, int]) -> None:
    z = zarr.open(str(path), mode="w", shape=arr.shape, dtype=arr.dtype, chunks=chunks)
    z[...] = arr


def _make_mgr(
    *,
    image_url: str,
    labels_url: str,
    transform_url: str,
    patch_size: Tuple[int, int, int] = (16, 16, 16),
    min_labeled_ratio: float = 0.001,
    min_bbox_percent: float = 0.0,
    cache_dir: Path | None = None,
) -> SimpleNamespace:
    mgr = SimpleNamespace()
    mgr.train_patch_size = patch_size
    mgr.targets = {"fibers": {"out_channels": 2, "activation": "none"}}
    mgr.min_labeled_ratio = min_labeled_ratio
    mgr.min_bbox_percent = min_bbox_percent
    mgr.normalization_scheme = "none"
    mgr.intensity_properties = {}
    mgr.no_spatial_augmentation = True
    mgr.no_scaling_augmentation = True
    mgr.data_path = cache_dir if cache_dir is not None else Path("/tmp")
    mgr.dataset_config = {
        "image_zarr_url": image_url,
        "labels_zarr_url": labels_url,
        "transform_json_url": transform_url,
    }
    return mgr


@pytest.fixture
def identity_pair(tmp_path: Path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 64, 64), dtype=np.uint8)
    labels = np.zeros((64, 64, 64), dtype=np.uint8)
    # two non-overlapping 16^3 chunks with fibers
    labels[16:32, 16:32, 16:32] = 255
    labels[32:48, 32:48, 32:48] = 1

    image_path = tmp_path / "image.zarr"
    labels_path = tmp_path / "labels.zarr"
    tform_path = tmp_path / "transform.json"

    _write_zarr_array(image_path, image, chunks=(16, 16, 16))
    _write_zarr_array(labels_path, labels, chunks=(16, 16, 16))
    _write_identity_transform(tform_path)

    return SimpleNamespace(
        image=image, labels=labels,
        image_url=str(image_path), labels_url=str(labels_path),
        tform_url=str(tform_path), root=tmp_path,
    )


def test_dataset_enumerates_foreground_only(identity_pair):
    mgr = _make_mgr(
        image_url=identity_pair.image_url,
        labels_url=identity_pair.labels_url,
        transform_url=identity_pair.tform_url,
        cache_dir=identity_pair.root,
    )
    ds = CrossFrameZarrDataset(mgr, is_training=False)
    assert len(ds) == 2  # exactly the two FG chunks
    positions = set(tuple(p) for p in ds._patches)
    assert (16, 16, 16) in positions
    assert (32, 32, 32) in positions


def test_identity_image_patch_matches_slab(identity_pair):
    mgr = _make_mgr(
        image_url=identity_pair.image_url,
        labels_url=identity_pair.labels_url,
        transform_url=identity_pair.tform_url,
        cache_dir=identity_pair.root,
    )
    ds = CrossFrameZarrDataset(mgr, is_training=False)
    sample = ds[0]
    pos = sample["patch_info"]["position"]
    expected = identity_pair.image[
        pos[0]:pos[0] + 16, pos[1]:pos[1] + 16, pos[2]:pos[2] + 16
    ].astype(np.float32)
    got = sample["image"].numpy().squeeze(0)
    np.testing.assert_allclose(got, expected, atol=1e-5)

    # label should be binary {0, 1}
    label = sample["fibers"].numpy().squeeze(0)
    assert set(np.unique(label).tolist()).issubset({0.0, 1.0})
    assert label.sum() > 0


def test_cache_roundtrip(identity_pair, tmp_path):
    cache_dir = tmp_path / "cache"
    mgr = _make_mgr(
        image_url=identity_pair.image_url,
        labels_url=identity_pair.labels_url,
        transform_url=identity_pair.tform_url,
        cache_dir=identity_pair.root,
    )
    mgr.dataset_config["cache_dir"] = str(cache_dir)

    ds1 = CrossFrameZarrDataset(mgr, is_training=False)
    first = list(ds1._patches)

    assert any(cache_dir.rglob("fibers_patches_*.npz"))

    # Replace labels with all zeros on disk: index must still load from cache.
    zeroed = np.zeros_like(identity_pair.labels)
    _write_zarr_array(Path(identity_pair.labels_url), zeroed, chunks=(16, 16, 16))
    ds2 = CrossFrameZarrDataset(mgr, is_training=False)
    assert list(ds2._patches) == first


def test_translation_transform_resamples_correctly(tmp_path: Path):
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=(64, 64, 64), dtype=np.uint8)
    # Label FG placed at (16, 16, 16); labels and image overlap except for translation.
    labels = np.zeros((64, 64, 64), dtype=np.uint8)
    labels[20:36, 20:36, 20:36] = 1

    image_path = tmp_path / "image.zarr"
    labels_path = tmp_path / "labels.zarr"
    tform_path = tmp_path / "transform.json"
    _write_zarr_array(image_path, image, chunks=(16, 16, 16))
    _write_zarr_array(labels_path, labels, chunks=(16, 16, 16))

    # Transform: image_xyz = label_xyz - [1, 2, 3] (ZYX). Schema is XYZ so translation
    # vector in XYZ form is [-3, -2, -1] -> M = I with m[:3,3] = [-3, -2, -1], and we
    # want label_xyz = M_fwd @ image_xyz, so M_fwd translates image to label. Because
    # the dataset inverts M_fwd, with M_fwd translating by +[3,2,1] in XYZ the inverse
    # translates by -[3,2,1] in XYZ (i.e. subtract from label to get image).
    m = np.eye(4)
    m[:3, 3] = [3.0, 2.0, 1.0]  # XYZ translation fixed<-moving
    _write_transform(tform_path, m)

    mgr = _make_mgr(
        image_url=str(image_path),
        labels_url=str(labels_path),
        transform_url=str(tform_path),
        patch_size=(16, 16, 16),
        min_labeled_ratio=0.01,
        min_bbox_percent=0.0,
        cache_dir=tmp_path,
    )
    ds = CrossFrameZarrDataset(mgr, is_training=False)
    assert len(ds) >= 1
    sample = ds[0]
    pos = sample["patch_info"]["position"]
    image_patch = sample["image"].numpy().squeeze(0)

    # With M (XYZ) translating moving -> fixed by +[3,2,1],
    # the inverse maps label ZYX -> image ZYX via shift by -[1,2,3] (after swap).
    # So image coord = label coord - [1, 2, 3].
    z, y, x = pos
    expected = image[z - 1:z - 1 + 16, y - 2:y - 2 + 16, x - 3:x - 3 + 16].astype(np.float32)
    np.testing.assert_allclose(image_patch, expected, atol=1e-4)


def test_empty_labels_raise(tmp_path):
    image = np.zeros((32, 32, 32), dtype=np.uint8)
    labels = np.zeros((32, 32, 32), dtype=np.uint8)
    image_path = tmp_path / "image.zarr"
    labels_path = tmp_path / "labels.zarr"
    tform_path = tmp_path / "transform.json"
    _write_zarr_array(image_path, image, chunks=(16, 16, 16))
    _write_zarr_array(labels_path, labels, chunks=(16, 16, 16))
    _write_identity_transform(tform_path)
    mgr = _make_mgr(
        image_url=str(image_path),
        labels_url=str(labels_path),
        transform_url=str(tform_path),
        patch_size=(16, 16, 16),
        cache_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="0 foreground patches"):
        CrossFrameZarrDataset(mgr, is_training=False)


def test_out_of_bounds_patches_skipped(tmp_path):
    """Label patches whose image-frame AABB lies outside the image are dropped."""
    rng = np.random.default_rng(2)
    image = rng.integers(0, 255, size=(32, 32, 32), dtype=np.uint8)
    labels = np.ones((64, 64, 64), dtype=np.uint8)  # every patch is FG
    image_path = tmp_path / "image.zarr"
    labels_path = tmp_path / "labels.zarr"
    tform_path = tmp_path / "transform.json"
    _write_zarr_array(image_path, image, chunks=(16, 16, 16))
    _write_zarr_array(labels_path, labels, chunks=(16, 16, 16))
    _write_identity_transform(tform_path)

    mgr = _make_mgr(
        image_url=str(image_path),
        labels_url=str(labels_path),
        transform_url=str(tform_path),
        patch_size=(16, 16, 16),
        cache_dir=tmp_path,
    )
    ds = CrossFrameZarrDataset(mgr, is_training=False)
    # labels is 64^3 of ones -> 4^3 = 64 candidate patches, but image is 32^3 so only
    # positions in [0..16] x 3 are valid -> 2^3 = 8 patches
    assert len(ds) == 8
