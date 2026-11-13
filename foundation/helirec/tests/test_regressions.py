"""Regression tests for corrected numerical conventions (gridrec centring, projector adjointness, output scale,
Paganin DC, rebalance, ramp padding, mending, calibration cache) and the legacy reference mode."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import helirec
from helirec import _core, calibration
from helirec.recon import CALIBRATION_KEYS, DEFAULTS, LEGACY_REFERENCE, HelicalRecon, legacy_reference_config

DEVICES = helirec.devices()


# ---------------------------------------------------------------- gridrec bounds and even-grid centring
def gaussian_sino(nd, ang, x0, y0, s=2.0):
    rot = (nd - 1) / 2
    u = np.arange(nd)[None, :] - rot
    p = (x0 * np.cos(ang) + y0 * np.sin(ang))[:, None]
    return (np.sqrt(2 * np.pi) * s * np.exp(-(u - p) ** 2 / (2 * s * s))).astype(np.float32), rot


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("N,nd", [(32, 32), (33, 32), (64, 64), (65, 65), (64, 40), (128, 100)])
def test_gridrec_centre_and_bounds(device, N, nd):
    ang = np.linspace(0, 2 * np.pi, 180, endpoint=False).astype(np.float32)
    x0, y0 = 5.0, -3.0
    sino, rot = gaussian_sino(nd, ang, x0, y0)
    g = _core.tomo_reconstruct(sino, np.full_like(sino, 0.5), ang, N, rot, "gridrec", {"filter_name": "ramlak"},
                               device=device)
    yy, xx = np.mgrid[:N, :N] - (N - 1) / 2
    m = np.clip(g, 0, None) * ((xx - x0) ** 2 + (yy - y0) ** 2 < 36)
    cx, cy = (m * xx).sum() / m.sum(), (m * yy).sum() / m.sum()
    assert abs(cx - x0) < 0.05 and abs(cy - y0) < 0.05, (cx, cy)
    # N = pdim (32 x 32): the de-apodisation table must cover row 0 / column 0
    assert np.any(g[0] != 0) and np.any(g[:, 0] != 0)


@pytest.mark.parametrize("device", DEVICES)
def test_gridrec_scale_is_uniform(device):
    """a small feature reconstructs at its true value at any radius: the de-apodisation is the transform of the
    gridding kernel in use (the closed form of the matched kernel gave 1.12 at the centre, 1.25 at radius 55 here)"""
    N = nd = 129
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False).astype(np.float32)
    c = (N - 1) // 2
    for r in (0, 30, 55):
        sino, rot = gaussian_sino(nd, ang, float(r), 0.0, s=3.0)
        g = _core.tomo_reconstruct(sino, np.full_like(sino, 0.5), ang, N, rot, "gridrec", {"filter_name": "ramlak"},
                                   device=device)
        assert abs(g[c, c + r] - 1.0) < 0.01, (r, g[c, c + r])


# ---------------------------------------------------------------- chord adjointness on boundary rays
def explicit_matrices(N, nd, rot, ang, device, projector):
    A = np.zeros((len(ang) * nd, N * N))
    for j in range(N * N):
        e = np.zeros(N * N, np.float32)
        e[j] = 1
        A[:, j] = _core.chord_project(e.reshape(N, N), ang, nd, rot, device=device, projector=projector).ravel()
    B = np.zeros((N * N, len(ang) * nd))
    for i in range(len(ang) * nd):
        e = np.zeros(len(ang) * nd, np.float32)
        e[i] = 1
        B[:, i] = _core.chord_backproject(e.reshape(len(ang), nd), ang, N, rot, device=device, projector=projector).ravel()
    return A, B


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("projector", ["chord", "sf"])
def test_projector_transpose_on_boundaries(device, projector):
    zero = np.zeros(1, np.float32)
    for N, nd, rot in ((4, 5, 2.0), (5, 6, 2.5)):    # rays exactly on pixel boundaries
        x, y = np.ones((N, N), np.float32), np.ones((1, nd), np.float32)
        ax = _core.chord_project(x, zero, nd, rot, device=device, projector=projector)
        aty = _core.chord_backproject(y, zero, N, rot, device=device, projector=projector)
        assert np.isclose((ax * y).sum(), N * N) and np.isclose((x * aty).sum(), N * N)
    ang = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, np.pi / 4, 3 * np.pi / 4, 0.3, 2.0], np.float32)
    for N in (4, 5, 8):
        for nd in (N, N + 1):
            for rot in ((N - 1) / 2, (N - 1) / 2 + 0.5, (N - 1) / 2 + 0.25):
                A, B = explicit_matrices(N, nd, rot, ang, device, projector)
                assert np.abs(A - B.T).max() < 1e-5, (N, nd, rot)


# ---------------------------------------------------------------- the merged frame
def test_merged_frame_from_the_geometry_calibration():
    """the frame the scans are merged in: one axis and one vector from the poses (medians over the helix, on a
    quarter-pixel grid), and the placement that follows"""
    from helirec import geometry
    n = 40
    cal = dict(layout=dict(frames=[dict(rest1=10, helix=20, rest2=10)] * 2, row_sign=-1.0))
    H = [np.linspace(0, 30, n), np.linspace(0, 30, n) + 0.5]                 # (the record: the second scan half a row on)
    z_pix = [-(h - h[0]) for h in H]
    poses = [dict(axis=np.full(n, 1684.81), row=-z_pix[0] + 0.02 * np.arange(n)),
             dict(axis=np.full(n, 2528.28), row=-z_pix[1] + 0.02 * np.arange(n) - 0.97)]
    for p_ in poses:                                                         # (the rest turns are not what it is taken from)
        p_["axis"][:10] += 3.0
        p_["row"][30:] -= 2.0
    lay = geometry.merged_layout(cal, poses, H)
    assert lay["cor_main_detector"] == 1684.75 and lay["vector"] == (1.0, -843.5)
    assert abs(lay["measured"]["second_scan_rows"] + 0.97) < 1e-9 and abs(lay["measured"]["axis_second_scan"] - 2528.28) < 1e-9
    pl, width, anchor = calibration.placement_from_vector(lay["vector"], 2528)
    assert (pl[0].shift_x, pl[0].x_readjustment, pl[0].z_readjustment) == (843, -0.5, 0.0)
    assert (pl[1].shift_x, pl[1].x_readjustment, pl[1].z_readjustment) == (0, 0.0, 1.0)
    assert width == 3372 and lay["cor_main_detector"] + anchor == 2528.25
    assert geometry.merged_layout(cal, poses, H, quantum=0.5)["vector"] == (1.0, -843.5)


def test_calibration_parameters():
    """the geometry calibration is named by parameters every mode keeps; those of the calibration it replaced are
    refused by name"""
    r = bare_recon()
    assert DEFAULTS["geometry"] == "" and set(CALIBRATION_KEYS) == {"geometry", "row_time_us"}
    assert not set(CALIBRATION_KEYS) & set(LEGACY_REFERENCE)
    r._update({"geometry": "/somewhere/geometry.json", "row_time_us": "-19.87"})
    r._update({"mode": "legacy_reference"})
    assert r.params["geometry"] == "/somewhere/geometry.json" and r.params["row_time_us"] == "-19.87"
    r._update({"mode": "corrected"})
    assert r.params["geometry"] == "/somewhere/geometry.json" and r.params["backprojector"] == "fbp"
    for gone in ("z_translation_scale", "companion_registration", "stationary_registration", "axis_refinement", "rolling_shutter"):
        with pytest.raises(KeyError, match="geometry calibration"):
            bare_recon()._update({gone: "auto"})
    with pytest.raises(RuntimeError, match="geometry"):      # neither given nor a cache directory to make it in
        _core.geometry_calibration_file("", "", ["/data/HA/HA.h5", "/data/TA/TA.h5"])


def test_geometry_calibration_is_made_once_and_kept(tmp_path):
    """no geometry given: geometry.json of the cache directory, made by the geometry calibration step when it is not
    there or is other scans', and read from then on; a given file is used as it is and nothing is made
    (helirec::geometry_calibration_file)"""
    made = []

    def step(scans, row_time, device):
        made.append((list(scans), row_time, device))
        return json.dumps(dict(scans=list(scans), row_time=dict(seconds_per_row=row_time)))

    cache = str(tmp_path / "cache")
    pair = ["/data/HA/HA.h5", "/data/TA/TA.h5"]
    path, raw = _core.geometry_calibration_file("", cache, pair, "", "cpu", None, step)
    assert path == str(tmp_path / "cache" / "geometry.json") and json.loads(raw)["scans"] == pair
    assert made == [(pair, None, "cpu")]
    assert _core.geometry_calibration_file("", cache, pair, "", "cpu", None, step) == (path, raw) and len(made) == 1   # kept: not made again
    other = ["/data/HA/HA.h5", "/data/TB/TB.h5"]                                   # other scans: made again, replaced
    path2, raw2 = _core.geometry_calibration_file("", cache, other, "-19.87", "cpu", None, step)
    assert path2 == path and json.loads(raw2)["scans"][1] == "/data/TB/TB.h5" and len(made) == 2
    assert made[1][1] == pytest.approx(-19.87e-6)                                 # (row_time_us: only for making it)
    given = tmp_path / "given.json"
    given.write_text('{"scans": ["x.h5", "y.h5"]}')
    assert _core.geometry_calibration_file(str(given), cache, pair, "", "cpu", None, step) == (str(given), given.read_text()) and len(made) == 2


# ---------------------------------------------------------------- output scale (attenuation)
@pytest.mark.parametrize("device", DEVICES)
def test_ramp_scale_attenuation(device):
    n_proj, nb, R, mu = 720, 160, 50.0, 0.3
    rot = (nb - 1) / 2 + 0.25
    ang = np.arange(n_proj) * 2 * np.pi / n_proj
    sub = (np.arange(8) + 0.5) / 8 - 0.5
    u = np.arange(nb)[None, :] + sub[:, None] - rot
    row = (2 * mu * np.sqrt(np.clip(R * R - u * u, 0, None))).mean(0)
    S = np.tile(row, (n_proj, 1)).astype(np.float32)
    W = np.ones((n_proj, 1, nb), np.float32)
    N = 161
    for legacy, expect in ((False, mu), (True, mu / 2)):
        _, Wb = _core.rebalance(S[:, None, :].copy(), W, rot, legacy, device=device)
        if not legacy:
            assert np.allclose(Wb[:, 0, 60:100], 0.5, atol=1e-3)   # full turn: every ray twice, pair-balanced
        q = _core.ramp_filter(S, rot, 1, True, legacy, device=device) * Wb[:, 0, :]
        img = _core.backproject(q, (-ang).astype(np.float32), N, N, rot, 1.0, (N - 1) / 2, device=device)
        c = img[70:91, 70:91].mean()
        assert abs(c - expect) < 0.01 * mu, (legacy, c)


# ---------------------------------------------------------------- Paganin DC and chunk independence
PH = dict(energy_kev=116.0, distance_m=1.1919883738942316, pixel_size_m=8.64e-6, det_rows=1200, det_cols=2528)
PAG = [{"type": "paganin", "delta_beta": 1000}, {"type": "unsharp", "sigma": 1.2, "coeff": 4}]


@pytest.mark.parametrize("device", DEVICES)
def test_paganin_constant_and_chunk_invariance(device):
    assert helirec.filter_margin(PAG, legacy_reference=True, **PH) == 20       # the reference's margin
    m = helirec.filter_margin(PAG, **PH)
    assert m == 20 + 11                  # fused Paganin + unsharp: the sum of the kernels' supports
    W = 400
    gains = {}
    for legacy, mm in ((False, m), (True, 20)):
        for h in (1, 2, 8, 64):
            a = np.ones((1, h + 2 * mm, W), np.float32)
            out = helirec.apply_filters(a, PAG, mm, device=device, legacy_reference=legacy, **PH)[0, mm:mm + h]
            gains[legacy, h] = float(out[:, W // 2].mean())
    for h in (1, 2, 8, 64):
        assert abs(gains[False, h] - 1) < 2e-6, gains
    assert gains[True, 1] < 0.999 and gains[True, 64] < gains[True, 1]    # the reference's DC loss
    # chunk invariance: rows of one frame, filtered in chunks of different heights, agree
    rng = np.random.default_rng(0)
    frame = (1 + 0.05 * rng.standard_normal((80 + 2 * m, W))).astype(np.float32)
    big = helirec.apply_filters(frame[None], PAG, m, device=device, **PH)[0, m:m + 80]
    for h in (1, 4, 16):
        for r0 in (0, 33):
            part = helirec.apply_filters(frame[None, r0:r0 + h + 2 * m].copy(), PAG, m, device=device, **PH)[0, m:m + h]
            assert np.abs(part - big[r0:r0 + h]).max() < 1e-6, (h, r0)


# ---------------------------------------------------------------- rebalance partition of unity
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [16, 8000])
def test_rebalance_partition_of_unity(device, n):
    cols, rot = 5, 2.0                                   # exact reflection: column 2 maps onto itself
    R = np.zeros((n, 1, cols), np.float32)
    W = np.ones((n, 1, cols), np.float32)
    _, Wb = _core.rebalance(R, W, rot, False, device=device)
    pair = Wb[:, 0, 2] + np.roll(Wb[:, 0, 2], -n // 2)
    np.testing.assert_allclose(pair, 1.0, rtol=1e-6)
    _, Wl = _core.rebalance(R, W, rot, True, device=device)    # the reference's endpoint defect
    assert np.isclose(Wl[0, 0, 2], 1 / 3) and np.isclose(Wl[n // 2, 0, 2], 1 / 2)
    W2 = W.copy()
    W2[0], W2[n // 2], W2[n - 1] = 2, 3, 7
    _, Wb = _core.rebalance(R, W2, rot, False, device=device)
    assert np.isclose(Wb[0, 0, 2], 2 / 5) and np.isclose(Wb[n // 2, 0, 2], 3 / 5)
    _, Wl = _core.rebalance(R, W2, rot, True, device=device)
    assert np.isclose(Wl[0, 0, 2], 1 / 6) and np.isclose(Wl[n // 2, 0, 2], 3 / 10)


# ---------------------------------------------------------------- ramp padding reflection about the true axis
def test_ramp_padding_reflection():
    g = _core.ramp_geometry(3372, 8000, 2528.25)
    gl = _core.ramp_geometry(3372, 8000, 2528.25, legacy_reference=True)
    assert gl["two_rots"] == 5057 and gl["legacy"] and not g["legacy"]
    # extended field of view: rows measured on [0, nb) with the axis at rot (2 rot > nb); the ramp needs the row
    # beyond nb, which the padding takes from the opposite view. A linear profile f(u) = a + b u (u = x - rot) has the
    # opposite view f(-u); with an exact continuation, filtering the truncated row equals filtering the analytic one
    # (up to the far padding), so compare the filtered rows near the right edge with a long-detector reference.
    nb, n_proj, rot = 160, 8, 120.25
    def prof(u):
        return np.clip(60 - np.abs(u + 7.3), 0, None) * 0.01   # a tent, off-axis (asymmetric under u -> -u)
    x = np.arange(nb)
    S = np.zeros((n_proj, nb), np.float32)
    S[:n_proj // 2] = prof(x - rot)
    S[n_proj // 2:] = prof(rot - x)                      # theta + pi: u -> -u
    nl = 2 * 256
    xl = np.arange(nl) - (nl - 1) / 2
    long_rows = np.zeros((n_proj, nl), np.float32)
    long_rows[:n_proj // 2] = prof(xl)
    long_rows[n_proj // 2:] = prof(-xl)
    ref = _core.ramp_filter(long_rows, (nl - 1) / 2, 1, False)
    sel = slice(nb - 30, nb)                             # the last bins, where the continuation matters
    err = {}
    for legacy in (False, True):
        q = _core.ramp_filter(S, rot, 1, False, legacy)
        refq = ref[:, np.round(x[sel] - rot + (nl - 1) / 2).astype(int)]
        err[legacy] = np.abs(q[:, sel] - refq).max()
    assert err[False] < err[True], err


# ---------------------------------------------------------------- ramp filters (windowed ramps of fbp / ghbp)
def ramp_window(name, x, p0=0.5, p1=8.0):
    """the window of each ramp filter at x cycles per pixel"""
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"ramlak": lambda: np.ones_like(x),
                "shepp": lambda: np.sinc(x),
                "cosine": lambda: np.cos(np.pi * x),
                "hann": lambda: np.where(x <= p0, 0.5 * (1 + np.cos(np.pi * x / p0)), 0),      # zero beyond the cutoff p0
                "hamming": lambda: np.where(x <= p0, 0.54 + 0.46 * np.cos(np.pi * x / p0), 0),
                "parzen": lambda: np.where(x <= p0, (1 - np.abs(x) / p0) ** 3, 0),
                "butterworth": lambda: 1 / (1 + (x / p0) ** (2 * p1))}[name]()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("par", [None, (0.3, 4.0), (1.0,)])
def test_ramp_filter_windows(device, par):
    assert set(_core.ramp_filter_names()) == {"none", "shepp", "cosine", "hann", "hamming", "ramlak", "parzen", "butterworth"}
    # an impulse on a centred axis has zero padding: the filtered row is the filter's kernel around the impulse
    nb, n_proj, c = 96, 4, 40
    rot = (nb - 1) / 2
    npad = _core.ramp_geometry(nb, n_proj, rot)["npad"]
    S = np.zeros((n_proj, nb), np.float32)
    S[:, c] = 1
    h = np.zeros(npad)
    h[0] = 0.25
    n = np.arange(1, npad // 2, 2)
    h[n] = h[-n] = -1 / (np.pi * n) ** 2
    H, x = np.fft.rfft(h).real, np.arange(npad // 2 + 1) / npad
    scale, text = _core.ramp_scale(n_proj), "" if par is None else ":".join(map(str, par))
    np.testing.assert_array_equal(_core.ramp_filter(S, rot, device=device), _core.ramp_filter(S, rot, device=device, filter_name="ramlak", filter_par=text))
    for name in _core.ramp_filter_names():
        spectrum = np.ones_like(H) if name == "none" else H * ramp_window(name, x, *(par or ()))
        expect = np.roll(np.fft.irfft(spectrum * scale, npad), c)[:nb]
        q = _core.ramp_filter(S, rot, device=device, filter_name=name, filter_par=text)
        np.testing.assert_allclose(q, np.tile(expect, (n_proj, 1)), atol=2e-6 * scale, err_msg=name)
    with pytest.raises(Exception, match="unknown ramp filter"):
        _core.ramp_filter(S, rot, device=device, filter_name="lanczos")
    with pytest.raises(Exception, match="cutoff"):
        _core.ramp_filter(S, rot, device=device, filter_name="hann", filter_par="0")


# ---------------------------------------------------------------- mending domain
@pytest.mark.parametrize("device", DEVICES)
def test_mending_domain(device):
    NX = 3372
    R = np.ones((1, NX), np.float32)
    RW = np.ones((1, NX), np.float32)
    R[0, 1000], RW[0, 1000] = 99, 0
    legacy = _core.mend_null_weights(R, RW, 1e-4, 1, NX - 2527, device=device)   # index 2527 read as a width
    assert legacy[0, 1000] == 99
    fixed = _core.mend_null_weights(R, RW, 1e-4, 1, 3371, device=device)
    assert fixed[0, 1000] == 1


# ---------------------------------------------------------------- legacy reference mode (parameters)
def bare_recon():
    r = object.__new__(HelicalRecon)
    r.params = {**DEFAULTS, "filters": []}
    return r


def test_legacy_mode_pins_parameters():
    r = bare_recon()
    r._update({"mode": "legacy_reference"})
    assert r.legacy and {k: r.params[k] for k in LEGACY_REFERENCE} == LEGACY_REFERENCE
    r._update({"mode": "legacy_reference", "backprojector": "ghbp", "filters": "paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)"})
    for bad in ({"backprojector": "fbp"}, {"filters": []}, {"paganin.delta_beta": 500}, {"margin": 30}):
        with pytest.raises(ValueError, match="pins"):
            r._update(bad)
    with pytest.raises(ValueError, match="pins"):
        bare_recon()._update({"mode": "legacy_reference", "filters": "paganin(delta_beta=500)"})
    r._update({"mode": "corrected"})
    assert r.params["filters"] == [] and r.params["mode"] == "corrected" and r.params["backprojector"] == "fbp"
    with pytest.raises(ValueError):
        r._update({"mode": "nabu"})
    cfg = legacy_reference_config()
    assert cfg["legacy_reference"] is True and cfg["backprojector"] == "ghbp" and len(cfg["filters"]) == 2


def test_default_params_are_corrected():
    assert DEFAULTS["mode"] == "corrected" and DEFAULTS["filters"] == []
    assert os.path.basename(_core.__file__).startswith("_core")


# ---------------------------------------------------------------- detector blur from the noise of flats
def test_detector_blur_field_from_noise_spectra():
    """noise blurred along the columns by 4 px per 1000 columns (0.5 px at the centre) on a white floor"""
    rng = np.random.default_rng(11)
    rows, cols, tile, a11, c1 = 384, 1024, 128, 4.0, 0.5
    ys, xs = list(range(0, rows, tile)), list(range(0, cols, tile))
    f = np.fft.rfftfreq(cols)
    power = np.zeros((len(ys), len(xs), tile, tile // 2 + 1))
    for _ in range(6):
        # (what passed the optics is band-limited before it is blurred: little of it is left at the white floor's corner)
        g = lambda v, ax: _core.gaussian_filter1d(np.ascontiguousarray(v, np.float32), ax, 0.8, "reflect")
        w = np.fft.rfft(np.stack([g(g(v, 0), 1) for v in rng.standard_normal((16, rows, cols))]), axis=-1)
        img = 0.5 * rng.standard_normal((16, rows, cols))
        for x0 in xs:
            s = a11 * (x0 + tile / 2 - 0.5 * (cols - 1)) / 1000 + c1
            img[:, :, x0:x0 + tile] += np.fft.irfft(w * np.sinc(s * f), cols, axis=-1)[:, :, x0:x0 + tile]
        power += _core.tile_power_spectra(np.ascontiguousarray(img, np.float32), tile, ys, xs)
    assert power.shape == (3, 8, 128, 65) and (power > 0).all()
    field, rms_none, rms_fit, solver = _core.detector_blur_field_fit(power, ys, xs, rows, cols)
    assert solver["usable"] and solver["solver_version"]
    assert abs(field[0] - a11) < 0.5 and abs(field[4] - c1) < 0.15 and rms_fit < 0.6 * rms_none
    assert abs(field[2]) * 0.5 + abs(field[3]) * 0.2 + abs(field[5]) < 0.6       # no blur along the rows
    # white noise everywhere: nothing to fit
    white = _core.tile_power_spectra(rng.standard_normal((40, rows, cols)).astype(np.float32), tile, ys, xs)
    with pytest.raises(Exception):
        _core.detector_blur_field_fit(white, ys, xs, rows, cols)
