"""Regression tests for the blend's core rules on a small synthetic prediction (no vc_reader needed).

The real-data check is in the README: on the 16 held-out PHerc. Paris. 4 regions and on a full PHerc. 125 mesh.
"""
import numpy as np

from segmentation.growpatch_blend.blend import tiled_blend

SHEET_Z = 120.0


def prediction(shape=(240, 260, 260), sheets=(SHEET_Z,), width=1.6):
    """ZYX uint8 cube with flat sheets z = const (Gaussian ridges across z)."""
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    out = np.zeros(shape, np.float32)
    for s in sheets:
        out = np.maximum(out, np.exp(-0.5 * ((z - s) / width) ** 2))
    return np.broadcast_to((out * 255).astype(np.uint8), shape).copy()


def mesh(z, step=8.0, n=26, x0=30.0, y0=30.0):
    """A flat GrowPatch-like vertex grid at depth z (rows along y, columns along x)."""
    yy, xx = np.meshgrid(y0 + step * np.arange(n), x0 + step * np.arange(n), indexing="ij")
    xyz = np.stack([xx, yy, np.full(xx.shape, z)], -1).astype(np.float64)
    return xyz, np.ones(xx.shape, bool)


def test_moves_halfway_onto_a_lone_sheet():
    xyz, valid = mesh(SHEET_Z + 3.0)
    out, ok, move, report = tiled_blend(xyz, valid, prediction(), (0.0, 0.0, 0.0))
    assert report["tiles_solved"] > 0
    z = out[ok][:, 2]
    moved = np.abs(move[ok][:, 2]) > 1e-6
    assert moved.mean() > 0.5
    np.testing.assert_allclose(z[moved], SHEET_Z + 1.5, atol=0.35)


def test_leaves_the_mesh_alone_when_the_sheet_is_far():
    xyz, valid = mesh(SHEET_Z + 12.0)
    out, ok, move, report = tiled_blend(xyz, valid, prediction(), (0.0, 0.0, 0.0))
    assert np.abs(move).max() == 0.0
