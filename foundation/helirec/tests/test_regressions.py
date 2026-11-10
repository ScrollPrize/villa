"""Regression tests for corrected numerical conventions (gridrec centring, projector adjointness, output scale,
Paganin DC, rebalance, ramp padding, mending, calibration cache) and the legacy reference mode."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import helirec
from helirec import _core, calibration
from helirec.recon import DEFAULTS, LEGACY_REFERENCE, HelicalRecon, legacy_reference_config

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


# ---------------------------------------------------------------- calibration cache identity
def fake_scan(tmp_path, name, nproj=10):
    f = tmp_path / f"{name}.h5"
    f.write_bytes(b"x" * 100)
    refs = SimpleNamespace(files=[str(f)], file_index=np.zeros(nproj, np.int32), __len__=lambda: nproj)
    sc = SimpleNamespace(name=name, master=str(f), frame_shape=(4, 6), dtype=np.uint16,
                         projections=list(range(nproj)), darks=[0], flats=[[0, 1]], energy_kev=116.0, distance_m=1.19,
                         pixel_size_m=8.64e-6, estimated_cor_from_motor_pix=0.0,
                         unwrapped_angles_deg=np.arange(nproj, dtype=float), z_translation_m=np.zeros(nproj),
                         control_A=np.ones(nproj), flats_unwrapped_angles_deg=np.zeros(1),
                         referenced_files=lambda: [str(f)])
    return sc, f


def test_calibration_cache_fingerprint(tmp_path):
    m, fm = fake_scan(tmp_path, "HA")
    c, _ = fake_scan(tmp_path, "TA")
    fp = calibration.acquisition_fingerprint(m, c)
    assert calibration.acquisition_fingerprint(m, c) == fp
    fm.write_bytes(b"y" * 101)                           # replaced data under the same name
    fp2 = calibration.acquisition_fingerprint(m, c)
    assert fp2 != fp
    cache = tmp_path / "cache"
    cache.mkdir()
    arrays = {k: np.zeros(3, np.float32) for k in calibration.REDUCTION_KEYS}
    with open(cache / "calibration_arrays.npz", "wb") as fh:
        np.savez(fh, _fingerprint=np.array(fp), **arrays)
    meas = dict(source="measured", fingerprint=fp, version=calibration.CALIBRATION_VERSION, rot_center=2528.25)
    (cache / "calibration.json").write_text(json.dumps(meas))
    log = []
    cal, arr = calibration._load_cache(str(cache), fp, log.append)
    assert cal is not None and arr is not None
    cal, arr = calibration._load_cache(str(cache), fp2, log.append)     # another acquisition: nothing reused
    assert cal is None and arr is None
    (cache / "calibration.json").write_text(json.dumps(dict(meas, source="override")))
    cal, arr = calibration._load_cache(str(cache), fp, log.append)      # an override is never a calibration
    assert cal is None and arr is not None
    (cache / "calibration.json").write_text(json.dumps({k: v for k, v in meas.items() if k != "fingerprint"}))
    assert calibration._load_cache(str(cache), fp, log.append)[0] is None   # pre-fingerprint caches are stale


# ---------------------------------------------------------------- helical pitch from opposite projections
def test_pitch_scale_from_opposite_projections():
    """a texture seen by projections half a turn apart, mirrored about the axis and displaced by the true travel"""
    rng = np.random.default_rng(5)
    rows, cols, axis, per_turn = 160, 300, 171.25, 200
    g = lambda a, ax: _core.gaussian_filter1d(np.ascontiguousarray(a, np.float32), ax, 1.5, "reflect")
    tex = g(g(rng.standard_normal((1, 1400, cols)), 1), 2)[0]                       # (height, mirrored column)
    angles = np.arange(5 * per_turn) * (360.0 / per_turn)
    moving = np.clip(np.arange(len(angles)) - per_turn, 0, 3 * per_turn)             # a stationary turn at each end
    recorded = -moving * (80.0 / (per_turn / 2))                                     # 80 rows per half turn
    u = np.arange(cols) - axis

    def frames(true_scale):
        def line(p):
            top = 1300 + recorded[p] * true_scale - rows                             # what is at row v: height top + v
            v = top + np.arange(rows)
            i = np.floor(v).astype(int)
            img = tex[i] * (1 - (v - i))[:, None] + tex[i + 1] * (v - i)[:, None]
            if int(round(angles[p] / 180.0)) % 2:                                    # the opposite view: mirrored
                x = axis - u
                j = np.clip(np.floor(x).astype(int), 0, cols - 2)
                img = img[:, j] * (1 - (x - j)) + img[:, j + 1] * (x - j)
            return np.ascontiguousarray(img, np.float32)
        return line

    shift, corr, edge = _core.mirror_row_shift(frames(1.004)(300), frames(1.004)(400), axis, 100, 80.0, 8)
    assert not edge and corr > 0.8 and abs(shift - 80.32) < 0.05
    for true in (1.004, 1.0):
        res = calibration.pitch_scale_from_pairs(frames(true), angles, recorded, axis, (rows, cols), n_pairs=12,
                                                 log=lambda *a: None)
        assert res["n_pairs"] >= 8 and abs(res["measured"] - true) < 3e-4
        assert res["significant"] is (true != 1.0) and res["scale"] == (res["measured"] if true != 1.0 else 1.0)
    # nothing to measure: a scan that does not move, or an axis at the detector's edge
    still = calibration.pitch_scale_from_pairs(frames(1.0), angles, 0 * recorded, axis, (rows, cols), log=lambda *a: None)
    edge = calibration.pitch_scale_from_pairs(frames(1.0), angles, recorded, 290.0, (rows, cols), log=lambda *a: None)
    assert still["scale"] == 1.0 and still["measured"] is None and "helical" in still["reason"]
    assert edge["scale"] == 1.0 and edge["measured"] is None and "mirrored" in edge["reason"]


def test_z_translation_scale_parameter():
    r = bare_recon()
    assert DEFAULTS["z_translation_scale"] == "auto" and LEGACY_REFERENCE["z_translation_scale"] == 1.0
    r.cal = SimpleNamespace(z_translation_scale=1.003)
    assert r.z_translation_scale() == 1.003
    r.cal = SimpleNamespace(z_translation_scale=None)                               # (not measured)
    assert r.z_translation_scale() == 1.0
    r.params["z_translation_scale"] = "1.002"
    assert r.z_translation_scale() == 1.002
    r.params["z_translation_scale"] = 0
    with pytest.raises(ValueError):
        r.z_translation_scale()


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
    assert r.params["filters"] == [] and r.params["mode"] == "corrected"
    with pytest.raises(ValueError):
        r._update({"mode": "nabu"})
    r._update({"mode": "legacy_reference"})
    r.cal = SimpleNamespace(source="override")
    with pytest.raises(ValueError, match="measured"):
        r._check_calibration()
    cfg = legacy_reference_config()
    assert cfg["legacy_reference"] is True and cfg["backprojector"] == "ghbp" and len(cfg["filters"]) == 2


def test_default_params_are_corrected():
    assert DEFAULTS["mode"] == "corrected" and DEFAULTS["filters"] == []
    assert os.path.basename(_core.__file__).startswith("_core")
