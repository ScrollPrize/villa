"""Tests for how the raw==0 zero mask is produced and transported for
empty-input masking (issue #1114): Volume computes it on RAW data before
normalization, VCDataset attaches it per patch (padding regions True,
multi-channel voxels masked only when all channels are 0), and collate_fn
stacks it across the batch.
"""

import numpy as np
import torch
import zarr

from vesuvius.data.vc_dataset import VCDataset
from vesuvius.data.volume import Volume


def _write_zarr(path, arr):
    z = zarr.open(str(path), mode="w", shape=arr.shape, chunks=arr.shape, dtype=arr.dtype)
    z[:] = arr
    return arr


def test_volume_zero_mask_computed_on_raw_values(tmp_path):
    """The mask must mark raw==0 voxels even though normalization maps them to
    a nonzero value - a post-normalization input==0 check would miss them."""
    rng = np.random.default_rng(0)
    raw = rng.integers(50, 200, size=(16, 16, 16), dtype=np.uint8)
    raw[:4] = 0
    _write_zarr(tmp_path / "vol.zarr", raw)

    volume = Volume(
        type="zarr",
        path=str(tmp_path / "vol.zarr"),
        normalization_scheme="instance_zscore",
        return_as_tensor=True,
        return_zero_mask=True,
    )
    data, mask = volume[:, :, :]

    np.testing.assert_array_equal(mask.numpy(), raw == 0)
    assert (np.abs(data.numpy()[raw == 0]) > 1e-6).all()


def test_vcdataset_patch_mask_and_collate(tmp_path):
    """Per-patch masks match raw==0 at the patch position; a fully-zero patch
    is flagged empty with an all-True mask; collate stacks mixed patches."""
    rng = np.random.default_rng(1)
    raw = rng.integers(50, 200, size=(32, 32, 64), dtype=np.uint8)
    raw[:, :, :32] = 0  # first patch fully zero
    raw[5:8, 5:8, 40:44] = 0  # small zero pocket inside the second patch
    _write_zarr(tmp_path / "vol.zarr", raw)

    ds = VCDataset(
        input_path=str(tmp_path / "vol.zarr"),
        patch_size=(32, 32, 32),
        step_size=1.0,
        normalization_scheme="instance_zscore",
        mode="infer",
        skip_empty_patches=True,
        return_zero_mask=True,
        verbose=False,
    )
    assert len(ds) == 2

    items = {ds[i]["pos"]: ds[i] for i in range(len(ds))}
    empty_item = items[(0, 0, 0)]
    data_item = items[(0, 0, 32)]

    assert empty_item["is_empty"] is True
    assert bool(empty_item["zero_mask"].all())

    assert data_item["is_empty"] is False
    expected = raw[:, :, 32:64] == 0
    np.testing.assert_array_equal(data_item["zero_mask"][0].numpy(), expected)

    batch = VCDataset.collate_fn([empty_item, data_item])
    assert tuple(batch["zero_mask"].shape) == (2, 1, 32, 32, 32)
    assert batch["zero_mask"].dtype == torch.bool
    assert bool(batch["zero_mask"][0].all())
    np.testing.assert_array_equal(batch["zero_mask"][1, 0].numpy(), expected)


def test_vcdataset_multichannel_mask_requires_all_channels_zero(tmp_path):
    """4D input: a voxel is masked only when ALL input channels read raw 0."""
    raw = np.full((2, 8, 8, 8), 100, dtype=np.uint8)
    raw[:, 0] = 0  # both channels zero at z=0 -> masked
    raw[0, 1] = 0  # only channel 0 zero at z=1 -> not masked
    _write_zarr(tmp_path / "vol4d.zarr", raw)

    ds = VCDataset(
        input_path=str(tmp_path / "vol4d.zarr"),
        patch_size=(8, 8, 8),
        step_size=1.0,
        normalization_scheme="instance_zscore",
        mode="infer",
        skip_empty_patches=False,
        return_zero_mask=True,
        verbose=False,
    )
    assert len(ds) == 1

    zero_mask = ds[0]["zero_mask"]
    assert tuple(zero_mask.shape) == (1, 8, 8, 8)
    assert bool(zero_mask[0, 0].all())
    assert not bool(zero_mask[0, 1:].any())
