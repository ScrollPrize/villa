"""interpolate_at_points must map nominal (voxel) queries to grid indices the way
the rest of the tifxyz package does: ``scale`` counts grid cells per voxel (a
segment sampled every 20 voxels stores 0.05), so grid index = voxel * scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesuvius.tifxyz import Tifxyz
from vesuvius.tifxyz.upsampling import interpolate_at_points

METHODS = ("catmull_rom", "linear", "bspline")


def _planar_grid(h: int, w: int, step_y: float, step_x: float):
    """A flat surface whose vertex (r, c) sits at voxel (x=c*step_x, y=r*step_y, z=100)."""
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float32)
    x = cols * np.float32(step_x)
    y = rows * np.float32(step_y)
    z = np.full((h, w), 100.0, dtype=np.float32)
    mask = np.ones((h, w), dtype=bool)
    return x, y, z, mask


@pytest.mark.parametrize("method", METHODS)
def test_voxel_queries_with_stored_scale(method: str) -> None:
    x, y, z, mask = _planar_grid(31, 31, 20.0, 20.0)
    query_y = np.array([[300.0, 340.0, 370.0]])
    query_x = np.array([[300.0, 360.0, 330.0]])

    xi, yi, zi, valid = interpolate_at_points(
        x, y, z, mask, query_y, query_x, scale=(0.05, 0.05), method=method
    )

    assert valid.all()
    np.testing.assert_allclose(xi, query_x, atol=1e-3)
    np.testing.assert_allclose(yi, query_y, atol=1e-3)
    np.testing.assert_allclose(zi, 100.0, atol=1e-3)


@pytest.mark.parametrize("method", METHODS)
def test_anisotropic_scale_is_applied_per_axis(method: str) -> None:
    # rows every 10 voxels (scale_y 0.1), columns every 25 voxels (scale_x 0.04)
    x, y, z, mask = _planar_grid(31, 31, 10.0, 25.0)
    query_y = np.array([[150.0, 175.0]])
    query_x = np.array([[375.0, 437.5]])

    xi, yi, _, valid = interpolate_at_points(
        x, y, z, mask, query_y, query_x, scale=(0.1, 0.04), method=method
    )

    assert valid.all()
    np.testing.assert_allclose(xi, query_x, atol=1e-3)
    np.testing.assert_allclose(yi, query_y, atol=1e-3)


@pytest.mark.parametrize("method", METHODS)
def test_unit_scale_queries_are_grid_indices(method: str) -> None:
    x, y, z, mask = _planar_grid(31, 31, 20.0, 20.0)
    query_y = np.array([[15.0, 17.5]])
    query_x = np.array([[16.0, 15.0]])

    xi, yi, _, valid = interpolate_at_points(
        x, y, z, mask, query_y, query_x, scale=(1.0, 1.0), method=method
    )

    assert valid.all()
    np.testing.assert_allclose(xi, [[320.0, 300.0]], atol=1e-3)
    np.testing.assert_allclose(yi, [[300.0, 350.0]], atol=1e-3)


@pytest.mark.parametrize("method", METHODS)
def test_matches_tifxyz_full_resolution_indexing(method: str) -> None:
    x, y, z, mask = _planar_grid(11, 11, 20.0, 20.0)
    surface = Tifxyz(
        _x=x, _y=y, _z=z, _scale=(0.05, 0.05), _mask=mask, path=None,
        interp_method=method, resolution="full",
    )
    fx, fy, fz, fvalid = surface[40:43, 60:63]

    rows, cols = np.mgrid[40:43, 60:63].astype(np.float64)
    xi, yi, zi, valid = interpolate_at_points(
        x, y, z, mask, rows, cols, scale=(0.05, 0.05), method=method
    )

    np.testing.assert_array_equal(valid, fvalid)
    np.testing.assert_allclose(xi, fx, atol=1e-4)
    np.testing.assert_allclose(yi, fy, atol=1e-4)
    np.testing.assert_allclose(zi, fz, atol=1e-4)
