"""vc_tomography.MosaicCanvas (vc_tomography/mosaic.hpp): the tiles of a mosaic joined in one sinogram and reconstructed by
one filtered backprojection. A sample of Gaussians wider than any tile, recorded by five tiles whose rotation axis lies
near the end of their rows (half acquisitions), each with a level of its own: the slice holds every Gaussian where it
is and as dense as it is, the tiles' levels are found, and the CPU and CUDA kernels agree."""
import numpy as np
import pytest

import vc_tomography
from vc_tomography import _core

NG, COLS, RC, PIXEL_M = 720, 160, 140.0, 1e-5
TILES = [(0.0, 0.0), (120.0, 0.0), (-60.0, 104.0), (-60.0, -104.0), (60.0, 104.0)]       # where each tile's axis passes through the canvas
BLOBS = [(10.0, -20.0, 0.6, 6.0), (95.0, 30.0, 0.5, 5.0), (-70.0, 80.0, 0.4, 7.0), (-40.0, -110.0, 0.7, 5.0), (150.0, -10.0, 0.3, 6.0),
         (40.0, 150.0, 0.5, 6.0)]                                                         # x, y, amplitude of the line integral, sigma
LEVELS = [0.0, 0.02, -0.015, 0.03, 0.01]
ANGLES = (-2 * np.pi * np.arange(NG) / NG).astype(np.float32)                             # the exact backprojector's, as a pipeline's bins have them
DEVICES = ["cpu"] + (["cuda"] if vc_tomography.cuda_available() else [])


def _rows(ax, ay, level=0.0, angles=ANGLES):
    """a tile's rows: column c of the row at angle a records the canvas column c - RC + ax cos a + ay sin a from the middle"""
    a = angles.astype(np.float64)[:, None]
    t = np.arange(COLS)[None, :] - RC + ax * np.cos(a) + ay * np.sin(a)
    out = np.zeros((len(angles), COLS))
    for x, y, amplitude, sigma in BLOBS:
        out += amplitude * np.exp(-0.5 * ((t - (x * np.cos(a) + y * np.sin(a))) / sigma) ** 2)
    return (out + level).astype(np.float32)


def _canvas(device, levels=LEVELS, **kw):
    c = _core.MosaicCanvas(NG, [(RC, ax, ay, COLS) for ax, ay in TILES], PIXEL_M, device=device, taper=10, level_step=5, **kw)
    c.begin(1)
    for k, ((ax, ay), level) in enumerate(zip(TILES, levels)):
        c.add(k, 0, _rows(ax, ay, level), np.ones((NG, COLS), np.float32), ANGLES)
    return c


@pytest.mark.parametrize("device", DEVICES)
def test_the_tiles_of_a_mosaic_give_one_slice(device):
    c = _canvas(device)
    assert c.size == 2 * c.half + 1 and c.half == 262          # what the tiles reach: 120 px to the furthest axis, 140 px of its rows, a pixel to spare
    got = c.slice(0, debug=True)
    img, h = got["slice"], c.half
    assert img.shape == (c.size, c.size) and np.isfinite(img).all()
    yy, xx = np.mgrid[:c.size, :c.size]
    away = np.ones(img.shape, bool)
    for x, y, amplitude, sigma in BLOBS:
        want = amplitude / (sigma * np.sqrt(2 * np.pi)) / (PIXEL_M * 100)                 # the Gaussian's density at its middle [1 / cm]
        r, col = int(round(h + y)), int(round(h + x))
        box = img[r - 6:r + 7, col - 6:col + 7]
        assert np.unravel_index(np.argmax(box), box.shape) == (6, 6) and box.max() == pytest.approx(want, rel=0.01)
        away &= np.hypot(xx - h - x, yy - h - y) > 5 * sigma
    assert np.abs(img[away]).max() < 0.005 * img.max()                                    # nothing between them: no row was cut inside the sample
    # a tile's level is what brings its values to the anchor's
    assert np.allclose(c.levels(0), -np.array(LEVELS), atol=2e-5) and c.angle_spread(0) < 1e-3
    # the sinogram as it is filtered: whole rows, the turn's second half the mirror of its first
    sino = got["sino"]
    assert sino.shape == (NG, c.size) and got["filtered"].shape == (NG // 2, c.size) and got["angles"].shape == (NG // 2,)
    assert np.allclose(sino[NG // 2:], sino[:NG // 2, ::-1], atol=1e-6)
    # a region is the same pixels (on CUDA to the rounding of the backprojector's positions: 5e-9 of the largest value here)
    roi = (100, 230, 180, 300)
    part, same = c.slice(0, roi=roi), img[roi[2]:roi[3], roi[0]:roi[1]]
    assert np.array_equal(part, same) if device == "cpu" else np.abs(part - same).max() < 1e-7 * img.max()


def test_without_the_levels_the_tiles_disagree():
    plain = _core.MosaicCanvas(NG, [(RC, ax, ay, COLS) for ax, ay in TILES], PIXEL_M, device="cpu", taper=10, levels=False)
    plain.begin(1)
    for k, ((ax, ay), level) in enumerate(zip(TILES, LEVELS)):
        plain.add(k, 0, _rows(ax, ay, level), np.ones((NG, COLS), np.float32), ANGLES)
    img = plain.slice(0)
    assert plain.levels(0) == [0.0] * len(TILES)
    h = plain.half
    yy, xx = np.mgrid[:plain.size, :plain.size]
    away = np.ones(img.shape, bool)
    for x, y, _, sigma in BLOBS:
        away &= np.hypot(xx - h - x, yy - h - y) > 5 * sigma
    # steps where one tile's rows end: what lies between the Gaussians is further from nothing than with the levels found
    assert np.abs(img[away]).max() > 1.5 * np.abs(_canvas("cpu").slice(0)[away]).max()


def test_a_tile_that_shares_no_ray_keeps_its_level():
    far = TILES + [(900.0, 0.0)]
    c = _core.MosaicCanvas(NG, [(RC, ax, ay, COLS) for ax, ay in far], PIXEL_M, device="cpu", level_step=5)
    c.begin(1)
    for k, (ax, ay) in enumerate(far):
        c.add(k, 0, _rows(ax, ay, 0.01 * k), np.ones((NG, COLS), np.float32), ANGLES)
    c.slice(0, roi=(0, 8, 0, 8))
    assert c.levels(0)[-1] == 0.0 and np.allclose(c.levels(0)[1:5], -0.01 * np.arange(1, 5), atol=2e-5)


@pytest.mark.skipif(not vc_tomography.cuda_available(), reason="no CUDA")
def test_cpu_and_cuda_agree():
    a, b = _canvas("cpu").slice(0), _canvas("cuda").slice(0)
    assert np.abs(a - b).max() < 1e-5 * np.abs(a).max()


def test_what_the_canvas_refuses():
    with pytest.raises(ValueError, match="even number"):
        _core.MosaicCanvas(NG + 1, [(RC, 0.0, 0.0, COLS)], PIXEL_M)
    with pytest.raises(ValueError, match="one width"):
        _core.MosaicCanvas(NG, [(RC, 0.0, 0.0, COLS), (RC, 50.0, 0.0, COLS + 2)], PIXEL_M)
    c = _core.MosaicCanvas(NG, [(RC, 0.0, 0.0, COLS)], PIXEL_M)
    c.begin(1)
    rows, w = _rows(0.0, 0.0), np.ones((NG, COLS), np.float32)
    c.add(0, 0, rows, w, ANGLES)
    with pytest.raises(ValueError, match="added once"):
        c.add(0, 0, rows, w, ANGLES)
    c.slice(0, roi=(0, 4, 0, 4))
    with pytest.raises(ValueError, match="not inside the canvas"):
        c.slice(0, roi=(0, c.size + 1, 0, 4))
