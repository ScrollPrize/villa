"""Geometry surface consistency post-processing: transform properties and sparse pyramid parity."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from vesuvius.ink_detection.postprocessing import surface_consistency as gsc


def test_sheet_is_kept_and_isolated_voxel_is_attenuated():
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[8, :, :] = 1.0
    volume[2, 2, 2] = 1.0
    out = gsc.enhance(volume, radius=2)
    assert out.shape == volume.shape and out.dtype == np.float32
    assert np.isclose(out[8, 8, 8], 1.0)
    assert out[2, 2, 2] < 0.3


def test_orientation_robust_across_axes():
    base = np.zeros((16, 16, 16), dtype=np.float32)
    base[8, :, :] = 1.0
    for axes in ((0, 1, 2), (1, 0, 2), (2, 1, 0)):
        volume = np.transpose(base, axes)
        out = gsc.enhance(volume, radius=2)
        assert np.isclose(out.max(), 1.0) and np.isclose(out[volume == 1.0].min(), 1.0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        gsc.enhance(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        gsc.enhance(np.zeros((4, 4, 4), dtype=np.float32), radius=0)
    bad = np.zeros((4, 4, 4), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        gsc.enhance(bad)


def test_uint8_round_trip_matches_native_encoding():
    assert np.isclose(gsc.to_unit(np.array([[[255]]], dtype=np.uint8))[0, 0, 0], 1.0)
    assert gsc.encode_uint8(np.array([[[0.5]]], dtype=np.float32))[0, 0, 0] == 128


def _native_style_pyramid(tmp_path, shape=(20, 30, 40), chunks=(8, 16, 16), levels=3):
    rng = np.random.default_rng(7)
    dense = np.zeros(shape, dtype=np.float32)
    dense[2:12, 4:20, 3:30] = rng.random((10, 16, 27), dtype=np.float32)
    dense[15, 20:28, 30:38] = 1.0
    encoded = gsc.encode_uint8(dense)
    path = tmp_path / "prediction.ome.zarr"
    arrays = gsc.create_pyramid(path, shape, chunks, levels)
    arrays[0][:] = encoded
    level = encoded
    for index in range(1, levels):
        level = gsc.downsample_mean_3d(level)
        arrays[index][:] = level
    return path, encoded


def test_sparse_pyramid_matches_dense_transform_and_keeps_support(tmp_path):
    path, encoded = _native_style_pyramid(tmp_path)
    out = tmp_path / "enhanced.ome.zarr"
    info = gsc.enhance_pyramid(path, out, radius=2)
    source = zarr.open(str(path), mode="r")
    result = zarr.open(str(out), mode="r")
    assert gsc.multiscale_paths(result) == ["0", "1", "2"]
    assert tuple(result["0"].chunks) == tuple(source["0"].chunks)
    assert result["0"].dtype == np.uint8

    got = np.asarray(result["0"])
    expected = gsc.encode_uint8(gsc.enhance(encoded.astype(np.float32) / 255.0, 2))
    chunks = tuple(int(c) for c in source["0"].chunks)
    mask = np.zeros(encoded.shape, dtype=bool)
    occupied = gsc.occupied_chunks(source["0"], path / "0")
    assert info["occupied_chunks"] == len(occupied) > 0
    for index in occupied:
        mask[tuple(slice(index[a] * chunks[a], (index[a] + 1) * chunks[a]) for a in range(3))] = True
    assert np.array_equal(got[mask], expected[mask])
    assert not got[~mask].any()
    assert np.array_equal(np.asarray(result["1"]), gsc.downsample_mean_3d(got))
    assert np.array_equal(np.asarray(result["2"]), gsc.downsample_mean_3d(gsc.downsample_mean_3d(got)))


def test_occupied_chunks_listing_matches_brute_force(tmp_path):
    path, _ = _native_style_pyramid(tmp_path)
    array = zarr.open(str(path), mode="r")["0"]
    listed = gsc.occupied_chunks(array, path / "0")
    assert listed == gsc.occupied_chunks(array, None) and listed


def test_cli_writes_receipt_and_refuses_overwrite(tmp_path):
    path, _ = _native_style_pyramid(tmp_path)
    out = tmp_path / "cli.ome.zarr"
    receipt = tmp_path / "receipt.json"
    assert gsc.main([str(path), str(out), "--receipt", str(receipt)]) == 0
    assert receipt.exists()
    with pytest.raises(FileExistsError):
        gsc.main([str(path), str(out)])
    assert gsc.main([str(path), str(out), "--overwrite"]) == 0
