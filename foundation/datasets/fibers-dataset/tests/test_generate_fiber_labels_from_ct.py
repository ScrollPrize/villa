"""End-to-end test for the CT-derived fiber label generator.

Builds a synthetic CT zarr containing a vessel-like ridge structure, runs
``generate_fiber_labels_from_ct`` on a bbox covering the structure, and
verifies the output zarr is created with the expected shape and contains
labels along the ridge.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import generate_fiber_labels_from_ct as glabels  # noqa: E402  (path injected by conftest)


@pytest.fixture
def synthetic_ct_volume():
    """A (32, 32, 32) uint8 volume with a bright horizontal ridge through the middle."""
    rng = np.random.default_rng(0)
    vol = (rng.integers(0, 30, size=(32, 32, 32), dtype=np.uint8))
    # Draw a horizontal bright ridge along axis 2 at z=16, y=16, x=8..24
    vol[14:18, 14:18, 8:24] = 220
    return vol


def _write_zarr(path, arr):
    import zarr

    z = zarr.open(path, mode="w", shape=arr.shape, chunks=arr.shape, dtype=arr.dtype, fill_value=0)
    z[:] = arr


def test_generates_zarr_with_labeled_ridge(tmp_path, synthetic_ct_volume):
    input_path = str(tmp_path / "ct.zarr")
    output_path = str(tmp_path / "fiber.zarr")
    _write_zarr(input_path, synthetic_ct_volume)

    result = glabels.generate_fiber_labels_from_ct(
        input_path,
        output_path,
        z0=8,
        z1=24,
        y0=8,
        y1=24,
        x0=4,
        x1=28,
        threshold=0.5,
        margin=4,
    )

    assert os.path.exists(output_path)
    assert result["bbox"] == (8, 24, 8, 24, 4, 28)
    assert result["voxels"] == 16 * 16 * 24
    # The synthetic ridge should produce a non-zero number of labels.
    assert result["fiber_voxels"] > 0
    assert 0.0 < result["fiber_fraction"] < 1.0
    assert result["elapsed_s"] > 0.0


def test_output_zarr_matches_input_shape(tmp_path, synthetic_ct_volume):
    import zarr

    input_path = str(tmp_path / "ct.zarr")
    output_path = str(tmp_path / "fiber.zarr")
    _write_zarr(input_path, synthetic_ct_volume)

    glabels.generate_fiber_labels_from_ct(
        input_path,
        output_path,
        z0=0,
        z1=16,
        y0=0,
        y1=16,
        x0=0,
        x1=16,
        threshold=0.5,
        margin=2,
    )

    out_arr = zarr.open(output_path, mode="r")
    assert tuple(out_arr.shape) == synthetic_ct_volume.shape
    assert out_arr.dtype == np.uint8
    # The unwritten region (z>=16) should remain zero.
    assert int(np.asarray(out_arr[16:, :, :]).sum()) == 0


def test_probability_output_is_optional(tmp_path, synthetic_ct_volume):
    import zarr

    input_path = str(tmp_path / "ct.zarr")
    label_path = str(tmp_path / "fiber.zarr")
    prob_path = str(tmp_path / "fiber_prob.zarr")
    _write_zarr(input_path, synthetic_ct_volume)

    glabels.generate_fiber_labels_from_ct(
        input_path,
        label_path,
        z0=4,
        z1=20,
        y0=4,
        y1=20,
        x0=4,
        x1=28,
        threshold=0.5,
        margin=4,
        probability_output=prob_path,
    )

    assert os.path.exists(prob_path)
    prob_arr = zarr.open(prob_path, mode="r")
    assert prob_arr.dtype == np.float32
    region = np.asarray(prob_arr[4:20, 4:20, 4:28])
    assert region.min() >= 0.0
    assert region.max() <= 1.0
