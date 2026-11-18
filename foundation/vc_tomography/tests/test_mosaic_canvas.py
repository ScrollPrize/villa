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
    c = _core.MosaicCanvas(NG, [(RC, ax, ay, COLS) for ax, ay in TILES], PIXEL_M, device=device, taper=10, level_step=5, rebin_angles=False, **kw)
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("kernel", ["two_pass", "linear", "lanczos3", "measured"])
def test_rebinning_recovers_parallel_projections(device, direction, kernel):
    # An analytic phantom in sample coordinates. The detector records a
    # different direction in every column, and each tile has its own view times.
    nviews = 480
    step = direction * 2 * np.pi / nviews
    nominal = step * np.arange(nviews)
    fan = (np.arange(COLS) - 0.5 * (COLS - 1)) * 0.00027
    tiles = [(RC, ax, ay, COLS) for ax, ay in TILES]
    args = dict(device=device, taper=12, levels=False, fans=[fan.tolist()] * len(tiles))
    direct = _core.MosaicCanvas(nviews, tiles, PIXEL_M, rebin_angles=False, **args)
    aligned = _core.MosaicCanvas(nviews, tiles, PIXEL_M, rebin_angles=True, rebin_kernel=kernel, **args)

    def project(a, offset):
        return sum(amp * np.exp(-0.5 * ((offset - x * np.cos(a) - y * np.sin(a)) / sig) ** 2)
                   for x, y, amp, sig in BLOBS)

    for c in (direct, aligned):
        c.begin(1)
    for k, (_, ax, ay, _) in enumerate(tiles):
        angles = nominal + step * (0.23 + 0.38 * np.sin(nominal + k))
        physical = angles[:, None] - fan
        offset = np.arange(COLS) - RC + ax * np.cos(physical) + ay * np.sin(physical)
        vals = project(physical, offset).astype(np.float32)
        weights = np.ones_like(vals)
        weights[::7, 28:49] = 0
        vals[weights == 0] = np.nan
        for c in (direct, aligned):
            c.add(k, 0, vals, weights, angles.astype(np.float32))
    errors, image_errors = [], []
    for c in (direct, aligned):
        got = c.slice(0, debug=True)
        expected = project(got["angles"][:, None], np.arange(c.size) - c.half)
        count = len(got["angles"])
        measured = got["weights"][:count] > 0.5
        r = (got["sino"][:count] - expected)[measured]
        errors.append(np.sqrt(np.mean(r ** 2)))
        yy, xx = np.indices(got["slice"].shape) - c.half
        density = sum(amp / (sig * np.sqrt(2 * np.pi) * PIXEL_M * 100) *
                      np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sig ** 2)
                      for x, y, amp, sig in BLOBS)
        signal = density > .01 * density.max()
        image_errors.append(np.sqrt(np.mean((got["slice"][signal] - density[signal]) ** 2)))
    assert image_errors[1] < 0.5 * image_errors[0]
    assert errors[1] < 0.35 * errors[0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel", ["two_pass", "linear", "lanczos3", "measured"])
def test_rebin_wraps_the_turn_and_masks_empty_rows(device, kernel):
    n = 32
    phase = 0.37
    a = np.arange(n) * 2 * np.pi / n + phase
    # The first tile sets the grid; only tile 1 contributes. Its last and
    # first rows bracket grid row zero and land on opposite sides of 2 pi.
    c = _core.MosaicCanvas(n, [(8, 0, 0, 17)] * 2, PIXEL_M, levels=False,
                           device=device, rebin_angles=True, rebin_kernel=kernel)
    c.begin(1)
    v = np.zeros((n, 17), np.float32)
    w = np.zeros_like(v)
    c.add(0, 0, v, w, a.astype(np.float32))
    v.fill(np.nan)
    v[-1] = 3
    w[-1] = 0.8
    b = np.mod(a + 0.25 * 2 * np.pi / n, 2 * np.pi)
    c.add(1, 0, v, w, b.astype(np.float32))
    got = c.slice(0, roi=(0, 1, 0, 1), debug=True)
    assert got["weights"][0, c.half] == pytest.approx(.2, abs=1e-5)
    assert got["sino"][0, c.half] == pytest.approx(3., abs=1e-5)


@pytest.mark.skipif(not vc_tomography.cuda_available(), reason="no CUDA")
@pytest.mark.parametrize("kernel", ["two_pass", "linear", "lanczos3", "measured"])
def test_rebinned_cpu_and_cuda_agree(kernel):
    projections = _rows(*TILES[1])
    weights = np.ones_like(projections)
    fan = np.linspace(-.001, .001, COLS).tolist()
    outputs = []
    for device in ("cpu", "cuda"):
        c = _core.MosaicCanvas(NG, [(RC, *TILES[1], COLS)], PIXEL_M, levels=False,
                               device=device, rebin_angles=True, rebin_kernel=kernel, fans=[fan])
        c.begin(1)
        c.add(0, 0, projections, weights, ANGLES + .001 * np.sin(ANGLES))
        outputs.append(c.slice(0, roi=(200, 275, 225, 300), debug=True))
    for key in ("weights", "sino"):
        np.testing.assert_allclose(outputs[0][key], outputs[1][key], rtol=2e-5, atol=2e-5)
    # Backprojector parity uses the peak of the slice, including near-zero pixels.
    a, b = outputs[0]["slice"], outputs[1]["slice"]
    assert np.max(np.abs(a - b)) < 1e-5 * np.max(np.abs(a))


@pytest.mark.parametrize("device", DEVICES)
def test_lanczos_retains_angular_modulation_and_exact_views(device):
    # An even harmonic survives folding. Interpolate a fractional view delay
    # while keeping two conjugate irregularly spaced views exactly on the grid.
    n, cols = 64, 9
    grid = 2 * np.pi * np.arange(n) / n
    a = grid + 0.35 * 2 * np.pi / n
    a[::n // 2] = grid[::n // 2]
    amplitude = np.cos(10 * a).astype(np.float32)
    rows = np.repeat(amplitude[:, None], cols, axis=1)
    w = np.ones_like(rows)
    errors = []
    for kernel in ("linear", "lanczos3"):
        c = _core.MosaicCanvas(n, [(4, 0, 0, cols)] * 2, PIXEL_M, device=device,
                               levels=False, rebin_angles=True, rebin_kernel=kernel)
        c.begin(1)
        c.add(0, 0, rows, np.zeros_like(w), grid.astype(np.float32))
        c.add(1, 0, rows, w, a.astype(np.float32))
        result = c.slice(0, roi=(0, 1, 0, 1), debug=True)
        middle = result["sino"][:n // 2, c.half]
        assert middle[0] == pytest.approx(amplitude[0], abs=1e-6)
        errors.append(np.linalg.norm(middle - np.cos(10 * grid[:n // 2])))
    assert errors[1] < 0.4 * errors[0]


@pytest.mark.parametrize("device", DEVICES)
def test_measured_grid_preserves_each_half_and_normalizes_the_full_turn(device):
    n, cols = 50, 17
    phase = 2 * np.pi * np.arange(n) / n
    angles = (phase + 0.018 * np.sin(phase)).astype("f")
    c = _core.MosaicCanvas(n, [(8, 0, 0, cols)] * 2, PIXEL_M, device=device,
                           rebin_angles=True, rebin_kernel="measured", levels=False,
                           ramp_filter_name="none", bp_interp="linear", bp_oversample=1)
    c.begin(2)
    for y in range(2):
        for sign in (-1, 1):
            c.add_view_angles(y, np.mod(angles + sign * .003 + y * .1, 2 * np.pi).astype("f"))
        # Odd modulation deliberately distinguishes the two half-turns.
        target = (angles + y * .1).astype("f")
        row = (1 + .3 * np.cos(target)).astype("f")
        data = np.repeat(row[:, None], cols, axis=1)
        c.add(0, y, data, np.ones_like(data), target)
    for y in (1, 0):
        got = c.slice(y, roi=(c.half, c.half+1, c.half, c.half+1), debug=True)
        a = np.unwrap(got["angles"].astype(float))
        target = angles.astype(float) + y * .1
        np.testing.assert_allclose(np.sin(a-target), 0, atol=5e-7)
        values = got["sino"][:, c.half]
        np.testing.assert_allclose(values, 1 + .3 * np.cos(target), atol=1e-6)
        # With the identity ramp its spectrum is exactly the angular scale.
        gaps = np.diff(np.r_[a, a[0]+2*np.pi])
        q = .25 * (gaps + np.roll(gaps,1))
        np.testing.assert_allclose(got["filtered"][:, c.half], q*values, atol=2e-6)
        np.testing.assert_allclose(got["slice"][0,0], np.sum(q*values)/(100*PIXEL_M), rtol=2e-6)
        assert got["opposite_only_fraction"] == 0
    with pytest.raises(ValueError, match="slice is made"):
        c.add(0, 0, data, np.ones_like(data), target)


@pytest.mark.parametrize("device", DEVICES)
def test_default_exposure_blend_matches_independent_regional_fbps(device):
    # Use unequal gates, overlapping discs, and intentionally different fine detail
    # per tile. Construct the reference by weighting the input views before rebinning.
    views, cols = 48, 41
    angles = (-2 * np.pi * np.arange(views) / views).astype("f")
    tiles = [(24, -9, 2, cols), (24, 9, -3, cols)]
    gates = [.09, .13]
    rng = np.random.default_rng(937)
    signals = rng.normal(size=(len(tiles), views, cols)).astype("f")
    weights = np.ones_like(signals)
    weights[:, ::7, 5:12] = 0

    def canvas(gate, mode=None, focus=None):
        options = {} if mode is None else dict(rebin_kernel=mode)
        c = _core.MosaicCanvas(views, tiles, PIXEL_M, device=device, levels=False, taper=4,
                               exposure_gates=gate, bp_oversample=1, bp_interp="linear", **options)
        c.begin(1)
        for k, (_, x, y, _) in enumerate(tiles):
            w = weights[k]
            if focus is not None:
                _, cx, cy, _ = tiles[focus]
                q = gates[k] * ((cx - x) * np.sin(angles) - (cy - y) * np.cos(angles)) / 2
                w = w * (.01 + .99 / (1 + q**4))[:, None]
            c.add(k, 0, signals[k], w, angles)
        return c

    c = canvas(gates)
    assert c.exposure_weighted
    base = canvas(gates, "measured").slice(0, debug=True)
    fallback = canvas([gates[0], 0])
    assert not fallback.exposure_weighted
    np.testing.assert_array_equal(fallback.slice(0), base["slice"])
    yy, xx = np.indices((c.size, c.size)) - c.half
    accum, support = np.zeros_like(xx, float), np.zeros_like(xx, float)
    padded = vc_tomography.roundfft(c.size + 64)
    left = (padded - c.size) // 2
    band = .5 - .5 * np.cos(np.pi * np.clip((np.fft.rfftfreq(padded) - .04) / .06, 0, 1))
    for k, (radius, x, y, _) in enumerate(tiles):
        fine = canvas(gates, "measured", focus=k).slice(0, debug=True)["filtered"]
        diff = np.pad(fine - base["filtered"], ((0, 0), (left, padded - c.size - left)), mode="edge")
        high = np.fft.irfft(np.fft.rfft(diff) * band, n=padded)[:, left:left+c.size]
        image = vc_tomography.backproject((base["filtered"] + high).astype("f"), base["angles"], c.size, c.size,
                                          rot_center=c.half, center=c.half, scale=1/(100*PIXEL_M),
                                          device=device, interp=0)
        w = .5 - .5 * np.cos(np.pi * np.clip(1 - np.hypot(xx - x, yy - y) / radius, 0, 1))
        accum += w * (image - base["slice"])
        support += w
    expected = base["slice"] + accum / np.maximum(1, support)
    got = c.slice(0)
    peak = np.abs(expected).max()
    assert np.max(np.abs(expected - base["slice"])) > .05 * peak
    np.testing.assert_allclose(got, expected, atol=4e-6 * peak, rtol=0)
    np.testing.assert_allclose(c.slice(0, roi=(10, 43, 21, 51)), got[21:51, 10:43], atol=2e-6 * peak, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_exposure_reuses_ordinary_tile_levels(device):
    angles = ANGLES[::8].copy()
    tiles = [(RC, *xy, COLS) for xy in TILES[:2]]
    rows = [_rows(*xy, offset, angles) for xy, offset in zip(TILES[:2], LEVELS[:2])]
    solved = []
    for mode in ("exposure", "measured"):
        c = _core.MosaicCanvas(len(angles), tiles, PIXEL_M, device=device, rebin_kernel=mode,
                               exposure_gates=[.07, .05], level_step=3, taper=8)
        c.begin(1)
        for k, values in enumerate(rows):
            c.add(k, 0, values, np.ones_like(values), angles)
        c.slice(0, roi=(c.half - 1, c.half + 1, c.half - 1, c.half + 1))
        solved.append(c.levels(0))
    assert abs(solved[0][1]) > .01
    np.testing.assert_array_equal(*solved)
