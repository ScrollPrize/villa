"""Prepared geometry uses the same native renderer as surface and plane rasters."""
import numpy as np
import pytest

from build_support import load_build
load_build()

from vc_reader.reader import ZarrWriter, open_zarr


@pytest.mark.parametrize("fill", [-7, 7])
def test_prepared_geometry_layers_masks_and_volume_fill(tmp_path, fill):
    z, y, x = np.mgrid[:8, :9, :10]
    values = (100 * z + 10 * y + x).astype(np.float32)
    path = tmp_path / "ct.zarr"
    ZarrWriter.create(path, shape=values.shape, chunks=(3, 4, 5), dtype="f4", fill_value=fill)[:] = values
    reader = open_zarr(path)
    # Singleton row, isolated valid points, anisotropic level placement, and
    # directions whose length is meaningful (not renormalized by the renderer).
    xyz = np.array([[[3, 4, 8], [4, 6, 12], [100, 100, 100], [2, 2, 4]]], np.float32)
    normals = np.zeros_like(xyz)
    normals[..., 2] = 4
    valid = np.array([[True, False, True, True]])
    rendered = reader.render_points(xyz, normals, valid, offsets=(-1, 0, 1), scale_zyx=(.25, .5, 1))
    np.testing.assert_allclose(rendered[0, [0, 2, 3]], [323, np.nan if fill < 0 else fill, 212])
    assert np.isnan(rendered[0, 1])
    assert rendered.dtype == np.float32


def test_prepared_fractional_positions_and_default_offset(tmp_path):
    values = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    path = tmp_path / "ct.zarr"
    ZarrWriter.create(path, shape=values.shape, chunks=(2, 2, 2), dtype="u1")[:] = values
    reader = open_zarr(path)
    xyz = np.array([[[1.5, 1.5, 1.5], [4, 4, 4], [np.nan, 1, 1]]], np.float32)
    result = reader.render_points(xyz, np.zeros_like(xyz), np.ones((1, 3), bool))
    np.testing.assert_allclose(result[0, :2], [31.5, 0])
    assert np.isnan(result[0, 2])
    with pytest.raises(ValueError, match="H,W"):
        reader.render_points(xyz, np.zeros((1, 2, 3)), np.ones((1, 3), bool))


def test_prepared_geometry_spans_native_tiles(tmp_path):
    z, y, x = np.mgrid[:4, :4, :4]
    path = tmp_path / "ct.zarr"
    ZarrWriter.create(path, shape=z.shape, chunks=(2, 2, 2), dtype="f4")[:] = 100 * z + 10 * y + x
    row, col = np.mgrid[:67, :69].astype(np.float32)
    xyz = np.stack((col / 40, row / 40, np.ones_like(row)), -1)
    result = open_zarr(path).render_points(xyz, np.zeros_like(xyz), np.ones(row.shape, bool))
    np.testing.assert_allclose(result, 100 + row / 4 + col / 40, atol=2e-5)
