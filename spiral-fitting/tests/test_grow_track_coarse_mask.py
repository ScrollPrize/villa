"""Tests for grow_track_graph.py --coarse-mask-mode support."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grow_track_graph as ggt  # noqa: E402


def _plane(valid):
    rows, cols = np.indices(valid.shape).astype(np.float64)
    flat = np.full_like(rows, 100.0)
    grid = np.stack((cols + 100.0, rows + 100.0, flat), -1)
    grid[~valid] = -1.0
    return grid


def _finalize(fine, factor, mode):
    coarse = ggt.resample_grid(fine, factor)
    keep = None
    if mode == "support":
        keep = ggt.support_coarse_mask(coarse, fine[..., 0] >= 0, factor)
    return coarse, ggt.finalize_coarse_grid(coarse, keep)


@pytest.mark.parametrize("factor", [1.0005, 2.0, 2.5, 4.0])
def test_full_plane_keeps_rim_that_erosion_drops(factor):
    fine = _plane(np.ones((17, 19), bool))
    coarse, support = _finalize(fine, factor, "support")
    _, erode = _finalize(fine, factor, "erode")
    np.testing.assert_array_equal(support, coarse)
    assert (erode[..., 0] >= 0).sum() < (support[..., 0] >= 0).sum()


def test_interior_fine_hole_removes_one_corner():
    valid = np.ones((13, 13), bool)
    valid[2, 2] = False  # inside quad (0, 0) at factor 4; all corners valid
    coarse, out = _finalize(_plane(valid), 4.0, "support")
    assert (coarse[..., 0] >= 0).all()
    kept = out[..., 0] >= 0
    # Dropping corner (0, 0) loses no other supported quad.
    assert not kept[0, 0] and kept.sum() == kept.size - 1
    np.testing.assert_array_equal(out[kept], coarse[kept])


def test_every_kept_quad_has_full_fine_support():
    rng = np.random.default_rng(0)
    for _ in range(20):
        valid = rng.random((41, 37)) > 0.03
        fine = _plane(valid)
        factor = rng.uniform(2.0, 4.0)
        _, out = _finalize(fine, factor, "support")
        kept = out[..., 0] >= 0
        quads = kept[:-1, :-1] & kept[1:, :-1] & kept[:-1, 1:] & kept[1:, 1:]
        rows = np.arange(0, 41 - 1 + 1e-9, factor)
        cols = np.arange(0, 37 - 1 + 1e-9, factor)
        for r, c in zip(*np.nonzero(quads)):
            r1 = min(int(np.ceil(rows[r + 1] - 1e-9)), 40)
            c1 = min(int(np.ceil(cols[c + 1] - 1e-9)), 36)
            lo_r, lo_c = int(rows[r]), int(cols[c])
            assert valid[lo_r:r1 + 1, lo_c:c1 + 1].all()


@pytest.mark.parametrize("output", ["7.5", "2.5", "0"])
def test_support_rejects_non_multiple_output_spacing(output):
    with pytest.raises(SystemExit):
        ggt.main([
            "t", "c", "o", "--seeds", "1", "--coarse-mask-mode", "support",
            "--resample-spacing", "5", "--output-spacing", output,
        ])


@pytest.mark.parametrize("output", ["5", "10", "20"])
def test_support_accepts_multiple_output_spacing(output, monkeypatch):
    class Reached(Exception):
        pass

    def stop(*_args, **_kwargs):
        raise Reached

    monkeypatch.setattr(ggt, "PackedTracks", stop)
    with pytest.raises(Reached):
        ggt.main([
            "t", "c", "o", "--seeds", "1", "--coarse-mask-mode", "support",
            "--resample-spacing", "5", "--output-spacing", output,
        ])
