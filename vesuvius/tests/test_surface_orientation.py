from __future__ import annotations

import numpy as np
import pytest

from vesuvius.surface_orientation import (
    depth_orientation,
    grid_normals,
    voxel_size_from_name,
)

VOXEL_UM = 10.0
SCALE = 0.05  # one grid cell every 20 voxels, as published meshes


def cylinder(*, ccw: bool, radius_vx: float = 500.0, arc_deg: float = 180.0, noise: float = 0.0):
    """Sheet on a cylinder around the z axis; columns follow the angle, rows follow +z."""
    step = 1.0 / SCALE
    cols = int(np.deg2rad(arc_deg) * radius_vx / step)
    theta = np.arange(cols) * step / radius_vx
    if not ccw:
        theta = theta[::-1]
    z = 1000.0 + np.arange(60)[:, None] * step + 0 * theta[None]
    x = 2000.0 + radius_vx * np.cos(theta)[None] + 0 * z
    y = 2000.0 + radius_vx * np.sin(theta)[None] + 0 * z
    rng = np.random.default_rng(0)
    if noise:
        x = x + rng.normal(0, noise, x.shape)
        y = y + rng.normal(0, noise, y.shape)
        z = z + rng.normal(0, noise, z.shape)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def test_grid_normal_matches_vc_convention() -> None:
    # Columns counter-clockwise, rows along +z: dP/dU x dP/dV points away from the axis.
    x, y, z = cylinder(ccw=True)
    points = np.stack([x, y, z], -1).astype(np.float64)
    normals = grid_normals(points, np.ones(x.shape, bool))
    radial = points[..., :2] - 2000.0
    radial /= np.linalg.norm(radial, axis=-1, keepdims=True)
    dot = np.einsum("ijc,ijc->ij", normals[1:-1, 1:-1, :2], radial[1:-1, 1:-1])
    assert np.all(dot > 0.99)


@pytest.mark.parametrize("ccw, flip", [(True, True), (False, False)])
def test_cylinder_verdict_follows_grid_handedness(ccw: bool, flip: bool) -> None:
    x, y, z = cylinder(ccw=ccw)
    report = depth_orientation(x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM)
    assert report["grid_normal"] == ("outward" if flip else "inward")
    assert report["vc_render_flip_normals"] is flip
    assert report["decisiveness"] > 0.99


def test_mirrored_grid_reverses_verdict() -> None:
    x, y, z = cylinder(ccw=True, noise=2.0)
    forward = depth_orientation(x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM)
    mirrored = depth_orientation(
        x[:, ::-1], y[:, ::-1], z[:, ::-1], scale=SCALE, voxel_size_um=VOXEL_UM
    )
    assert forward["inward_fraction"] == pytest.approx(1.0 - mirrored["inward_fraction"], abs=1e-5)
    assert forward["vc_render_flip_normals"] is not mirrored["vc_render_flip_normals"]


def test_noise_shorter_than_stencil_does_not_change_verdict() -> None:
    # 2 voxels of vertex jitter swamps a one-cell stencil but not a 4 mm one.
    x, y, z = cylinder(ccw=True, noise=2.0)
    report = depth_orientation(x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM)
    assert report["vc_render_flip_normals"] is True
    assert report["stencil_cells"] == [20, 20]


def test_flat_sheet_is_undetermined() -> None:
    rows, cols = np.mgrid[0:60, 0:80].astype(np.float32) * 20.0
    rng = np.random.default_rng(1)
    x = cols + rng.normal(0, 1.0, cols.shape)
    y = 500.0 + rng.normal(0, 1.0, cols.shape)
    z = rows + 100.0
    report = depth_orientation(x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM)
    assert report["grid_normal"] == "undetermined"
    assert report["vc_render_flip_normals"] is None


def test_invalid_vertices_are_ignored() -> None:
    x, y, z = cylinder(ccw=False)
    x[:, :10] = -1
    y[:, :10] = -1
    z[:, :10] = -1
    report = depth_orientation(x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM)
    assert report["vc_render_flip_normals"] is False


def test_too_small_for_stencil_is_undetermined() -> None:
    x, y, z = cylinder(ccw=True, arc_deg=40.0)
    report = depth_orientation(x[:20], y[:20], z[:20], scale=SCALE, voxel_size_um=VOXEL_UM)
    assert report["vertices_used"] == 0
    assert report["status"] == "too_small"
    assert report["vc_render_flip_normals"] is None


def test_voxel_size_from_published_mesh_name() -> None:
    assert voxel_size_from_name("x/20260317000000-on-20250728140407-9.362um.tifxyz") == 9.362
    assert voxel_size_from_name("auto_grown_20250703034159599") is None


def test_spikes_do_not_dilute_the_verdict() -> None:
    # Two columns pushed 600 voxels off the sheet along the normal:
    # stencils touching a spike measure the spike, not the curvature, and with
    # balanced signs they would drown the verdict.
    x, y, z = cylinder(ccw=True, radius_vx=900.0, arc_deg=200.0)
    radial = np.stack([x - 2000.0, y - 2000.0], -1)
    radial /= np.linalg.norm(radial, axis=-1, keepdims=True)
    spikes = np.isin(np.arange(x.shape[1]), (40, 120))[None]
    x = x + np.where(spikes, 600.0 * radial[..., 0], 0.0).astype(np.float32)
    y = y + np.where(spikes, 600.0 * radial[..., 1], 0.0).astype(np.float32)
    report = depth_orientation(x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM)
    assert report["vc_render_flip_normals"] is True
    assert report["decisiveness"] > 0.9


def test_large_grid_is_decimated_with_the_same_verdict() -> None:
    x, y, z = cylinder(ccw=False)
    full = depth_orientation(x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM)
    small = depth_orientation(x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM, max_vertices=1000)
    assert small["decimation"] > 1
    assert small["stencil_cells"][0] < full["stencil_cells"][0]
    assert small["vc_render_flip_normals"] is full["vc_render_flip_normals"] is False


def test_anisotropic_scale_sets_each_stencil() -> None:
    x, y, z = cylinder(ccw=True)
    report = depth_orientation(x, y, z, scale=(0.05, 0.1), voxel_size_um=VOXEL_UM)
    assert report["stencil_cells"] == [20, 40]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_invalid_voxel_size_raises(bad: float) -> None:
    x, y, z = cylinder(ccw=True)
    with pytest.raises(ValueError):
        depth_orientation(x, y, z, scale=SCALE, voxel_size_um=bad)


def test_modest_step_and_hole_are_not_measured() -> None:
    # Half the sheet 250 voxels off along the normal (under the 1.5x chord
    # limit) and a one-column hole: stencils across either measure the defect.
    x, y, z = cylinder(ccw=True, radius_vx=900.0, arc_deg=200.0)
    radial = np.stack([x - 2000.0, y - 2000.0], -1)
    radial /= np.linalg.norm(radial, axis=-1, keepdims=True)
    step = (np.arange(x.shape[1]) >= x.shape[1] // 2)[None]
    x = x + np.where(step, 250.0 * radial[..., 0], 0.0).astype(np.float32)
    y = y + np.where(step, 250.0 * radial[..., 1], 0.0).astype(np.float32)
    valid = np.ones(x.shape, bool)
    valid[:, x.shape[1] // 4] = False
    report = depth_orientation(x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM, valid=valid)
    assert report["vc_render_flip_normals"] is True
    assert report["decisiveness"] > 0.9


def test_decimation_does_not_step_over_holes() -> None:
    # Every third column invalid: each full-resolution stencil crosses a hole,
    # and decimating by 3 must not sample around them into a clean grid.
    x, y, z = cylinder(ccw=True)
    valid = np.ones(x.shape, bool)
    valid[:, 1::3] = False
    full = depth_orientation(x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM, valid=valid)
    decimated = depth_orientation(
        x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM, valid=valid, max_vertices=1000
    )
    assert decimated["decimation"] == 3
    assert full["status"] == decimated["status"] == "too_small"


def test_too_few_vertices_is_too_small() -> None:
    x, y, z = cylinder(ccw=True)
    report = depth_orientation(
        x, y, z, scale=SCALE, voxel_size_um=VOXEL_UM, min_vertices=10**6
    )
    assert report["vertices_used"] > 0
    assert report["status"] == "too_small"
