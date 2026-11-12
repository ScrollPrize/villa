"""Detector map of the reconstruction (detector_map.hpp): the rows read at non-uniform positions without loss
(resample_rows against the direct sum of the trigonometric interpolant), the inversion of the map and its sign
conventions, the projections' own scales, and scans gridded through a map against the same data on the ideal grid."""
import h5py
import numpy as np
import pytest

import helirec
from helirec import _core
from test_sirt_pipeline import recon

DEVICES = helirec.devices()


def fft_length(n):
    """upsample_fft_length: the smallest 2^a 3^b 5^c 7^d that holds n + 64"""
    best = None
    for a in range(16):
        for b in range(11):
            for c in range(8):
                for d in range(7):
                    v = 2 ** a * 3 ** b * 5 ** c * 7 ** d
                    if v >= n + 64 and (best is None or v < best):
                        best = v
    return best


def interpolant(rows, pos):
    """The band-limited interpolant of shift_rows at the positions pos, by its direct sum in double precision: the
    rows (float32) bridged to the transform's length as shift_rows bridges them (in single precision), every bin of
    the transform at every position, the Nyquist bin of an even length as cos(pi p)."""
    nr, n = rows.shape
    L = fft_length(n)
    f32 = np.float32
    t = (np.arange(n, L) - n + 1).astype(f32) / f32(L - n + 1)
    w = f32(0.5) - f32(0.5) * np.cos(f32(3.14159265358979) * t, dtype=f32)
    bridge = rows[:, -1:] + (rows[:, :1] - rows[:, -1:]) * w
    assert bridge.dtype == f32
    A = np.fft.rfft(np.concatenate([rows, bridge], axis=1).astype(np.float64), axis=1)
    k = np.arange(L // 2 + 1)
    weight = np.full(L // 2 + 1, 2.0)
    weight[0] = 1.0
    out = np.empty(pos.shape)
    for r in range(nr):
        c = A[r] * weight
        phase = np.exp(2j * np.pi * np.outer(pos[r], k) / L)
        if L % 2 == 0:
            out[r] = ((phase[:, :-1] * c[:-1]).real.sum(1) + 0.5 * c[-1].real * np.cos(np.pi * pos[r])) / L
        else:
            out[r] = (phase * c).real.sum(1) / L
    return out


def test_fft_length_is_the_library_s():
    # (the direct sum must bridge the rows to the library's length: checked through a delay by the whole length,
    # which is the identity on that length alone)
    rng = np.random.default_rng(0)
    for n in (77, 161, 2528):
        rows = rng.standard_normal((1, n)).astype(np.float32)
        pos = np.arange(n, dtype=np.float64)[None] + fft_length(n)
        np.testing.assert_allclose(_core.resample_rows(rows, pos), rows, atol=2e-6 * np.abs(rows).max())
    assert fft_length(2528) == 2592 and fft_length(161) % 2 == 1          # (an odd length: no Nyquist bin)


def sample_rows(n, rng):
    """rows with content up to the Nyquist frequency, of the row and of its transform"""
    L, x = fft_length(n), np.arange(n)
    rows = [rng.standard_normal(n), rng.standard_normal(n),               # white: every frequency
            np.cos(np.pi * x),                                            # the row's Nyquist frequency
            np.cos(2 * np.pi * (L // 2) / L * x + 0.3),                   # the transform's highest bins
            np.cos(2 * np.pi * (L // 2 - 1) / L * x + 1.0),
            1 + 0.05 * np.cos(2 * np.pi * 0.47 * x + 0.3),                # a projection: a pattern on a level
            np.full(n, 2.5), rng.uniform(0.5, 1.5, n)]
    return np.stack(rows).astype(np.float32)



def smooth_displacements(n_rows, n, rng, reach=2.0):
    x = np.arange(n)
    d = np.stack([np.sin(2 * np.pi * rng.uniform(0.2, 1.5) * x / n + rng.uniform(0, 6)) * rng.uniform(0.3, 1) +
                  rng.uniform(-1, 1) for _ in range(n_rows)])
    return d * (reach / np.abs(d).max())


@pytest.mark.parametrize("n", [2528, 161, 160, 77])     # the detector; an odd transform length; small rows
def test_resample_rows_is_the_interpolant(n):
    """resample_rows against the direct sum, at positions displaced smoothly by up to 2 samples: within 1e-6 of
    each row's amplitude (its largest sample), content up to Nyquist included; the two backends agree as closely."""
    rng = np.random.default_rng(n)
    rows = sample_rows(n, rng)
    amp = np.abs(rows).max(1, keepdims=True)
    x = np.arange(n, dtype=np.float64)
    pos = x[None] + smooth_displacements(len(rows), n, rng)
    assert 1.99 < np.abs(pos - x).max() <= 2.0
    want = interpolant(rows, pos)
    got = {d: _core.resample_rows(rows, pos, device=d) for d in DEVICES}
    for d in DEVICES:
        assert got[d].dtype == np.float32 and got[d].shape == rows.shape
        assert (np.abs(got[d] - want) / amp).max() <= 1e-6, d
    if len(DEVICES) > 1:
        assert (np.abs(got["cpu"] - got["cuda"]) / amp).max() <= 1e-6
    # positions beyond the row and far beyond it: the bridge, and the interpolant's period
    far = pos + np.linspace(-40, 40, len(rows))[:, None] + 3 * fft_length(n)
    for d in DEVICES:
        assert (np.abs(_core.resample_rows(rows, far, device=d) - interpolant(rows, far)) / amp).max() <= 1e-6, d


@pytest.mark.parametrize("device", DEVICES)
def test_resample_rows_and_shift_rows(device):
    """at the positions shift_rows reads, resample_rows gives shift_rows' rows (both within rounding of the
    interpolant): no displacement, a constant one, and each row its own"""
    n = 2528
    rng = np.random.default_rng(7)
    rows = sample_rows(n, rng)
    amp = np.abs(rows).max(1, keepdims=True)
    x = np.arange(n, dtype=np.float64)
    same = lambda a, b: (np.abs(a - b) / amp).max()
    still = _core.resample_rows(rows, np.repeat(x[None], len(rows), 0), device=device)
    assert same(still, rows) <= 1e-6 and same(still, _core.shift_rows(rows, 0.0, device=device)) <= 1e-6
    for shift in (0.37, -1.5, 2.0):
        s = np.float64(np.float32(shift))
        moved = _core.resample_rows(rows, np.repeat((x - s)[None], len(rows), 0), device=device)
        assert same(moved, _core.shift_rows(rows, shift, device=device)) <= 1.5e-6        # (two roundings)
        assert same(moved, interpolant(rows, np.repeat((x - s)[None], len(rows), 0))) <= 1e-6
        # the same through the row descriptions (no map: X = j - shift is read at X)
        np.testing.assert_array_equal(_core.resample_rows(rows, shift=shift, device=device), moved)
    assert same(_core.resample_rows(rows, np.repeat((x - 1)[None], len(rows), 0), device=device)[:, 1:], rows[:, :-1]) <= 1e-6
    # more rows than one transform batch (CUDA: 4096), the last batch incomplete
    many = np.repeat(rows, 600, 0)
    pos = np.repeat(x[None] + smooth_displacements(len(rows), n, rng), 600, 0)
    np.testing.assert_array_equal(_core.resample_rows(many, pos, device=device),
                                  np.repeat(_core.resample_rows(rows, pos[::600], device=device), 600, 0))


# a map like a detector's: terms of total order <= 3, up to about 2 px, changing by about 1e-3 px per px
TERMS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]


def detector_map(rng, n=2528, det_rows=1200, size=0.5):
    return dict(map_i=[t[0] for t in TERMS], map_j=[t[1] for t in TERMS],
                map_eu=list(rng.uniform(-size, size, len(TERMS))), map_ev=list(rng.uniform(-size, size, len(TERMS))),
                map_c0=(n - 1) / 2, map_cs=n / 2, map_v0=(det_rows - 1) / 2, map_vs=det_rows / 2, det_rows=det_rows)


def displacement(m, c, v, n):
    """(Eu, Ev) of calibration's DetectorMap::displacement, held inside the detector (detector_map.hpp)"""
    c = np.clip(c, -32, n - 1 + 32)
    v = np.clip(v, 0, m["det_rows"] - 1)
    x, y = (c - m["map_c0"]) / m["map_cs"], (v - m["map_v0"]) / m["map_vs"]
    mono = [x ** i * y ** j for i, j in zip(m["map_i"], m["map_j"])]
    return sum(a * q for a, q in zip(m["map_eu"], mono)), sum(a * q for a, q in zip(m["map_ev"], mono))


def source(m, X, Y, n, row_slope=0.0, slope_col=0.0, col_scale=0.0, scale_col=0.0, row_scale=0.0, scale_row=0.0):
    """DetectorMap::source with the roll and the scales, to convergence in extended precision"""
    p, v = X.astype(np.longdouble), Y.astype(np.longdouble)
    for _ in range(60):
        du, dv = displacement(m, p, v, n)
        p = (X + col_scale * scale_col - du) / (1 + col_scale)
        v = (Y + row_slope * (p - slope_col) + row_scale * scale_row - dv) / (1 + row_scale)
    return p, v


def test_detector_map_conventions():
    """pixel (c, v) looks at the ideal position (c + Eu, v + Ev): Eu = +2 reads the ideal column X at detector
    column X - 2, Ev = -1 the ideal row Y at detector row Y + 1 (calibration's DetectorMap::source, as
    test_calibration.py::test_session_detector_map reads whole pixels); the scales enter like terms of the map"""
    one = dict(map_i=[0], map_j=[0], map_c0=0., map_cs=1., map_v0=0., map_vs=1., det_rows=1200, det_cols=2528)
    X, Y = np.array([100.0, 2000.25]), np.array([50.0, 700.5])
    c, v = _core.detector_map_source(X, Y, map_eu=[2.0], map_ev=[-1.0], **one)
    np.testing.assert_array_equal(c, X - 2)
    np.testing.assert_array_equal(v, Y + 1)
    # col_scale: c + col_scale (c - scale_col) = X; row_scale: v + row_scale (v - scale_row) = Y; with or without a map
    none = ([], [], [], [], 0., 1., 0., 1., 1200, 2528)
    c, v = _core.detector_map_source(X, Y, *none, col_scale=1e-2, scale_col=1000.0, row_scale=-2e-2, scale_row=500.0)
    np.testing.assert_allclose(c, (X + 1e-2 * 1000.0) / (1 + 1e-2), rtol=0, atol=1e-11)
    np.testing.assert_allclose(v, (Y - 2e-2 * 500.0) / (1 - 2e-2), rtol=0, atol=1e-11)
    c, v = _core.detector_map_source(X, Y, map_eu=[2.0], map_ev=[-1.0], **one, col_scale=1e-2, scale_col=1000.0,
                                     row_scale=-2e-2, scale_row=500.0)
    np.testing.assert_allclose(c + 2 + 1e-2 * (c - 1000.0), X, rtol=0, atol=1e-11)
    np.testing.assert_allclose(v - 1 - 2e-2 * (v - 500.0), Y, rtol=0, atol=1e-11)
    # the roll keeps its meaning: the ideal row of a stack row is y + row_slope (detector column - slope_col)
    c, v = _core.detector_map_source(X, Y, map_eu=[2.0], map_ev=[-1.0], **one, row_slope=1e-2, slope_col=1263.5)
    np.testing.assert_allclose(v, Y + 1e-2 * (c - 1263.5) + 1, rtol=0, atol=1e-11)
    # the rows of the gridding (single precision): the detector row for the ideal row base + pos, minus base
    cols, base = np.array([5, 900, 2527], np.int32), np.array([3, 600, 1100], np.int32)
    pos = np.array([0.25, 0.5, 0.75], np.float32)
    one.pop("det_cols")
    row = lambda **k: _core.detector_map_source_row(cols, base, pos, det_cols=2528, **k)
    np.testing.assert_array_equal(row(map_eu=[2.0], map_ev=[-1.0], **one), pos + 1)
    np.testing.assert_array_equal(row(map_eu=[0.0], map_ev=[0.0], **one), pos)
    scaled = row(map_eu=[], map_ev=[], **{**one, "map_i": [], "map_j": []}, row_scale=-8e-5, scale_row=599.5)
    np.testing.assert_allclose(scaled, (base + pos.astype(np.float64) - 8e-5 * 599.5) / (1 - 8e-5) - base, rtol=0, atol=5e-7)
    assert abs(scaled[0] - pos[0] - 8e-5 * (3.2023 - 599.5)) < 1e-6    # (v = Y - row_scale (v - scale_row): 0.048 rows lower)


def test_detector_map_inversion():
    """the positions solved for a map with a roll and both scales: the equations hold to 3e-8 px (the columns of
    resample_rows, double precision) and to 1e-6 rows (the rows of the gridding, single precision)"""
    n, det_rows = 2528, 1200
    rng = np.random.default_rng(11)
    m = detector_map(rng, n, det_rows)
    reg = dict(row_slope=1e-3, slope_col=1263.5, col_scale=1e-4, scale_col=1200.0, row_scale=-8e-5, scale_row=599.5)
    plan = _core.detector_map_plan(**m, det_cols=n, row_slope=1e-3, col_scale=1e-4, row_scale=8e-5)
    cc, vv = np.meshgrid(np.linspace(-32, n + 31, 300), np.linspace(0, det_rows - 1, 300))
    du, dv = displacement(m, cc, vv, n)
    assert np.abs(du).max() <= plan["max_eu"] <= np.abs(du).max() + 0.2          # bounds, and close ones
    assert np.abs(dv).max() <= plan["max_ev"] <= np.abs(dv).max() + 0.2
    assert 1e-3 < plan["gradient"] < 1e-2 and plan["passes"] in (3, 4) and plan["tolerance"] == 3e-8
    X = rng.uniform(-10, n + 10, 20000)
    Y = rng.uniform(-20, det_rows + 20, 20000)                                   # (rows beyond the detector too)
    c, v = _core.detector_map_source(X, Y, **m, det_cols=n, **reg)
    du, dv = displacement(m, c, v, n)
    assert np.abs(c + du + reg["col_scale"] * (c - reg["scale_col"]) - X).max() < 3e-8
    assert np.abs(v + dv + reg["row_scale"] * (v - reg["scale_row"]) - Y - reg["row_slope"] * (c - reg["slope_col"])).max() < 3e-8
    pr, vr = source(m, X, Y, n, **reg)
    assert np.abs(c - pr).max() < 3e-8 and np.abs(v - vr).max() < 3e-8
    assert 0.5 < np.abs(c - X).max() < 4
    # the rows: every detector column, rows across the detector, positions within a few rows of the stack row
    cols = rng.integers(0, n, 200000).astype(np.int32)
    base = rng.integers(0, det_rows, 200000).astype(np.int32)
    pos = rng.uniform(-3, 4, 200000).astype(np.float32)
    got = _core.detector_map_source_row(cols, base, pos, **m, det_cols=n, row_scale=reg["row_scale"], scale_row=reg["scale_row"])
    Y = base + pos.astype(np.longdouble)
    want = Y.copy()
    for _ in range(60):          # v + Ev(c, v) + row_scale (v - scale_row) = Y at the detector column c
        want = (Y + reg["row_scale"] * reg["scale_row"] - displacement(m, cols.astype(np.float64), want, n)[1]) / (1 + reg["row_scale"])
    assert 0.3 < np.abs(want - Y).max() < 2 and np.abs(got - (want - base)).max() < 1e-6
    # a map the fixed point cannot invert, a scale that is none, powers and sizes that are not a map's
    steep = dict(m, map_eu=[200 * a for a in m["map_eu"]])
    for bad in (steep, dict(m, map_cs=0.0), dict(m, map_i=[9] + m["map_i"][1:]), dict(m, map_ev=m["map_ev"][:-1])):
        with pytest.raises(Exception):
            _core.detector_map_plan(**bad, det_cols=n)
    with pytest.raises(Exception):
        _core.detector_map_plan(**m, det_cols=n, col_scale=0.6)


@pytest.mark.parametrize("device", DEVICES)
def test_resample_rows_through_a_map(device):
    """rows read through a detector map: the interpolant at the detector column of every ideal column, each row at
    its own height, delay and scales; whole pixels read the neighbouring samples"""
    n, det_rows, nr = 2528, 1200, 8
    rng = np.random.default_rng(13)
    m = detector_map(rng, n, det_rows)
    rows = sample_rows(n, rng)[:nr]
    amp = np.abs(rows).max(1, keepdims=True)
    x = np.arange(n, dtype=np.float64)
    ys = np.linspace(-3.3, det_rows + 2.2, nr)
    shifts = rng.uniform(-2, 2, nr).astype(np.float32)
    slope, col_scale, row_scale = rng.uniform(-2e-3, 2e-3, nr).astype(np.float32), rng.uniform(-2e-4, 2e-4, nr).astype(np.float32), \
        rng.uniform(-2e-4, 2e-4, nr).astype(np.float32)
    got = _core.resample_rows(rows, shift=0.25, shifts=list(shifts), row_y=list(ys), row_slope=list(slope), slope_col=1263.5,
                              col_scale=list(col_scale), scale_col=1200.0, row_scale=list(row_scale), scale_row=599.5, **m,
                              device=device)
    X = x[None] - (np.float32(0.25) + shifts)[:, None].astype(np.float64)       # what shift_rows would read
    pos = np.stack([source(m, X[r], np.full(n, ys[r]), n, float(slope[r]), 1263.5, float(col_scale[r]), 1200.0,
                           float(row_scale[r]), 599.5)[0] for r in range(nr)]).astype(np.float64)
    assert 0.5 < np.abs(pos - X).max() < 4
    assert (np.abs(got - interpolant(rows, pos)) / amp).max() <= 1e-6
    assert (np.abs(got - _core.resample_rows(rows, pos, device=device)) / amp).max() <= 2e-7
    # Eu = +2: the ideal column X is the detector's column X - 2
    whole = dict(map_i=[0], map_j=[0], map_eu=[2.0], map_ev=[-1.0], det_rows=det_rows)
    out = _core.resample_rows(rows, row_y=list(ys), **whole, device=device)
    assert (np.abs(out[:, 2:] - rows[:, :-2]) / amp).max() <= 1e-6
    # a map of zeros, and none: shift_rows' positions
    zero = dict(whole, map_eu=[0.0], map_ev=[0.0])
    np.testing.assert_array_equal(_core.resample_rows(rows, shift=0.37, row_y=list(ys), **zero, device=device),
                                  _core.resample_rows(rows, shift=0.37, device=device))
    assert (np.abs(_core.resample_rows(rows, shift=0.37, device=device) - _core.shift_rows(rows, 0.37, device=device)) / amp).max() <= 1.5e-6
    # col_scale without a map: p = (X + col_scale scale_col) / (1 + col_scale)
    out = _core.resample_rows(rows, col_scale=[1e-3] * nr, scale_col=1000.0, device=device)
    assert (np.abs(out - interpolant(rows, np.repeat(((x + 1.0) / 1.001)[None], nr, 0))) / amp).max() <= 1e-6


def test_resample_rows_backends_agree_through_a_map():
    if len(DEVICES) < 2:
        pytest.skip("one backend")
    n, det_rows, nr = 2528, 1200, 8
    rng = np.random.default_rng(17)
    m = detector_map(rng, n, det_rows)
    rows = sample_rows(n, rng)[:nr]
    kw = dict(shift=0.25, row_y=list(np.linspace(0, det_rows - 1, nr)), row_slope=[1e-3] * nr, slope_col=1263.5,
              col_scale=[1e-4] * nr, scale_col=1263.5, row_scale=[-8e-5] * nr, scale_row=599.5, **m)
    a, b = (_core.resample_rows(rows, **kw, device=d) for d in ("cpu", "cuda"))
    assert (np.abs(a - b) / np.abs(rows).max(1, keepdims=True)).max() <= 1e-6


# ---------------------------------------------------------------- through the pipeline
NR, ND, NPROJ, V0, C0 = 40, 160, 129, 19.5, 79.5


def ideal(X, Y):
    """the scene on the ideal grid: a column pattern near the top of the band that fades towards the detector's ends
    (a row that ends in a pattern is not its band-limited interpolant there: test_tomo.py::test_shift_rows_exact),
    on a ramp along the rows (which two rows interpolate exactly; the lanczos3 row kernel does not quite)"""
    return 1 + 0.05 * np.exp(-0.5 * ((X - C0) / 20) ** 2) * np.cos(2 * np.pi * 0.31 * X + 0.4) + 0.004 * (Y - V0)


def frame_scan(tmp_path, name, eu=None, ev=None, offset=0.0, stretch_cols=0.0, stretch_rows=0.0, **kw):
    """a stationary scan whose frames show `ideal` through a detector: pixel (c, v) records the scene at
    (c + eu(c, v) + stretch_cols (c - C0), v + ev(c, v) + stretch_rows (v - V0)); kw: what make_scan_data is told"""
    v, c = np.mgrid[:NR, :ND].astype(np.float64)
    X = c + (eu(c, v) if eu else 0.0) + stretch_cols * (c - C0)
    Y = v + (ev(c, v) if ev else 0.0) + stretch_rows * (v - V0)
    frame = np.rint(40000 * ideal(X, Y)).astype(np.uint16)
    path = tmp_path / f"{name}.h5"
    if not path.exists():
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=np.broadcast_to(frame, (NPROJ, NR, ND)).copy(), chunks=(1, NR, ND))
    return _core.make_scan_data(np.arange(NPROJ) * (360 / 64), np.zeros(NPROJ), .01, NR, ND, 0, -offset, 0., np.full(NPROJ, .2),
                                np.array([0], np.float32), np.zeros((NR, ND), np.float32),
                                np.full((1, NR, ND), 50000, np.float32), np.ones((NR, ND), np.float32),
                                _core.RawReader([str(path)], "data", NR, ND), np.zeros(NPROJ, np.int32),
                                np.arange(NPROJ, dtype=np.int32), **kw)


def gridded(scan, device, **k):
    cfg = dict(backprojector="fbp", rot_center=C0, grid_size=33, z_interp="linear")
    cfg.update(k)
    return recon(scan, device, **cfg).reconstruct(8, 32, debug=True)["sino"][:, :, 40:120]


# a smooth map of this small detector: up to 1.6 px along the columns, 0.5 px along the rows
MAP = dict(map_i=[0, 2, 1, 1, 0], map_j=[0, 0, 1, 0, 2], map_eu=[0.3, 0.8, -0.3, 0.0, 0.2], map_ev=[-0.2, 0.25, 0.2, 0.3, 0.0],
           map_c0=C0, map_cs=80.0, map_v0=V0, map_vs=20.0)


def map_terms(coefficients):
    def f(c, v):
        x, y = (c - MAP["map_c0"]) / MAP["map_cs"], (v - MAP["map_v0"]) / MAP["map_vs"]
        return sum(a * x ** i * y ** j for a, i, j in zip(coefficients, MAP["map_i"], MAP["map_j"]))
    return f


# what the frames' 16 bits leave of the gridded values (40000 counts: 1.3e-5 each, a few of them in a sample)
BITS = 6e-5


@pytest.mark.parametrize("device", DEVICES)
def test_scan_through_a_detector_map(tmp_path, device):
    """a scan recorded through a detector map grids like the same scene recorded on the ideal grid, when the map is
    given: whole pixels to rounding, a smooth map to the frames' 16 bits; left out, the map's displacement shows"""
    plain = gridded(frame_scan(tmp_path, "plain"), device)
    assert np.ptp(plain.mean((0, 1))) > 0.04 and np.ptp(plain.mean((0, 2))) > 0.08      # both patterns are there
    # whole pixels: Eu = +2, Ev = -1: the pixel two columns lower and one row higher looks at the ideal position
    whole = dict(map_i=[0], map_j=[0], map_eu=[2.0], map_ev=[-1.0])
    scan = lambda told: frame_scan(tmp_path, "whole", lambda c, v: 2.0 + 0 * c, lambda c, v: -1.0 + 0 * c, **told)
    assert np.abs(gridded(scan(whole), device) - plain).max() < 3e-6
    assert np.abs(gridded(scan({}), device) - plain).max() > 0.03
    # a smooth map
    eu, ev = map_terms(MAP["map_eu"]), map_terms(MAP["map_ev"])
    v, c = np.mgrid[:NR, :ND].astype(np.float64)
    assert 1.0 < np.abs(eu(c, v)).max() < 2.0 and 0.4 < np.abs(ev(c, v)).max() < 1.0
    mapped = gridded(frame_scan(tmp_path, "smooth", eu, ev, **MAP), device)
    assert np.abs(mapped - plain).max() < BITS
    assert np.abs(gridded(frame_scan(tmp_path, "smooth", eu, ev), device) - plain).max() > 0.03
    # each part alone: the rows wherever frames are gridded, the columns where they are read exactly
    rows_only, cols_only = dict(MAP, map_eu=[0.0] * 5), dict(MAP, map_ev=[0.0] * 5)
    assert np.abs(gridded(frame_scan(tmp_path, "rows", None, ev, **rows_only), device) - plain).max() < BITS
    assert np.abs(gridded(frame_scan(tmp_path, "rows", None, ev), device) - plain).max() > 1e-3
    assert np.abs(gridded(frame_scan(tmp_path, "cols", eu, None, **cols_only), device) - plain).max() < BITS
    assert np.abs(gridded(frame_scan(tmp_path, "cols", eu, None), device) - plain).max() > 0.03
    other = dict(x_interp="lanczos3")
    assert np.abs(gridded(frame_scan(tmp_path, "rows", None, ev, **rows_only), device, **other) -
                  gridded(frame_scan(tmp_path, "plain"), device, **other)).max() < BITS
    # the lanczos3 row kernel at the rows the map gives: its own error on a ramp, 2 % of a row at worst
    kernel = dict(z_interp="lanczos3")
    assert np.abs(gridded(frame_scan(tmp_path, "smooth", eu, ev, **MAP), device, **kernel) -
                  gridded(frame_scan(tmp_path, "plain"), device, **kernel)).max() < 3e-4
    # with a fractional column offset of the scan: the delay and the map in one resampling
    shifted = gridded(frame_scan(tmp_path, "plain", offset=0.5), device)
    assert np.abs(gridded(frame_scan(tmp_path, "smooth", eu, ev, offset=0.5, **MAP), device) - shifted).max() < BITS
    # a map of zeros changes nothing, bit for bit
    zeros = dict(MAP, map_eu=[0.0] * 5, map_ev=[0.0] * 5)
    np.testing.assert_array_equal(gridded(frame_scan(tmp_path, "plain", **zeros), device), plain)
    if device == "cuda":
        assert np.abs(mapped - gridded(frame_scan(tmp_path, "smooth", eu, ev, **MAP), "cpu")).max() < 3e-6


@pytest.mark.parametrize("device", DEVICES)
def test_scan_with_frame_scales(tmp_path, device):
    """row_scale, col_scale: frames stretched along the rows (pixel (c, v) records the scene at row v + s (v - V0))
    or along the columns grid like the plain ones when their scale is given, with or without a detector map"""
    plain = gridded(frame_scan(tmp_path, "plain"), device)
    rs, cs = 0.05, 0.01                                     # 0.6 rows, 0.4 columns at the ends of what is compared
    rows = lambda **told: gridded(frame_scan(tmp_path, "row_scale", stretch_rows=rs, **told), device)
    assert np.abs(rows(row_scale=[rs] * NPROJ, scale_row=V0) - plain).max() < 3e-6       # (a ramp: no rounding to 16 bits shows)
    assert np.abs(rows() - plain).max() > 2e-3
    assert np.abs(rows(row_scale=[-rs] * NPROJ, scale_row=V0) - plain).max() > 4e-3      # (the sign matters)
    assert np.abs(rows(row_scale=[rs] * NPROJ, scale_row=V0 + 4) - plain).max() > 5e-4   # (and the row it is about)
    cols = lambda **told: gridded(frame_scan(tmp_path, "col_scale", stretch_cols=cs, **told), device)
    assert np.abs(cols(col_scale=[cs] * NPROJ, scale_col=C0) - plain).max() < BITS
    assert np.abs(cols() - plain).max() > 0.01
    assert np.abs(cols(col_scale=[-cs] * NPROJ, scale_col=C0) - plain).max() > 0.01
    # per projection: a scale given to half the projections only leaves the other half stretched
    half = [rs] * (NPROJ // 2) + [0.0] * (NPROJ - NPROJ // 2)
    assert np.abs(rows(row_scale=half, scale_row=V0) - plain).max() > 1e-3
    # both scales and a map together
    eu, ev = map_terms(MAP["map_eu"]), map_terms(MAP["map_ev"])
    told = dict(MAP, row_scale=[rs] * NPROJ, scale_row=V0, col_scale=[cs] * NPROJ, scale_col=C0)
    assert np.abs(gridded(frame_scan(tmp_path, "all", eu, ev, stretch_cols=cs, stretch_rows=rs, **told), device) - plain).max() < BITS
    # the rows are scaled in the gridding of measured pixels as well (the iterative methods)
    it = dict(backprojector="sirt", tomo_params={"num_iter": "1"})
    measured = gridded(frame_scan(tmp_path, "plain"), device, **it)
    assert np.abs(gridded(frame_scan(tmp_path, "row_scale", stretch_rows=rs, row_scale=[rs] * NPROJ, scale_row=V0), device, **it) - measured).max() < 3e-6
    assert np.abs(gridded(frame_scan(tmp_path, "row_scale", stretch_rows=rs), device, **it) - measured).max() > 2e-3


@pytest.mark.parametrize("device", DEVICES)
def test_detector_map_is_refused_where_it_cannot_be_applied(tmp_path, device):
    lateral = [dict(MAP), dict(col_scale=[0.01] * NPROJ, scale_col=C0)]
    vertical = [dict(MAP, map_eu=[0.0] * 5), dict(row_scale=[0.01] * NPROJ, scale_row=V0)]
    for told in lateral + vertical:
        with pytest.raises(Exception, match="legacy reference"):
            recon(frame_scan(tmp_path, "plain", **told), device, backprojector="fbp", legacy_reference=True)
    for told in lateral:   # a lateral part needs the exact column shift: not dropped silently
        with pytest.raises(Exception, match="x_interp = sinc"):
            recon(frame_scan(tmp_path, "plain", **told), device, backprojector="fbp", x_interp="lanczos3")
        with pytest.raises(Exception, match="measured pixels"):
            recon(frame_scan(tmp_path, "plain", **told), device, backprojector="sirt", tomo_params={"num_iter": "1"})
    for told in vertical:  # the rows are solved wherever frames are gridded
        recon(frame_scan(tmp_path, "plain", **told), device, backprojector="fbp", x_interp="lanczos3")
        recon(frame_scan(tmp_path, "plain", **told), device, backprojector="sirt", tomo_params={"num_iter": "1"})
    for bad in (dict(MAP, map_ev=MAP["map_ev"][:-1]), dict(row_scale=[0.01] * 3), dict(col_scale=[0.01] * 3),
                dict(MAP, map_ev=[30.0] + MAP["map_ev"][1:]), dict(MAP, map_cs=0.0), dict(row_scale=[0.6] * NPROJ)):
        with pytest.raises(Exception):
            recon(frame_scan(tmp_path, "plain", **bad), device, backprojector="fbp")
