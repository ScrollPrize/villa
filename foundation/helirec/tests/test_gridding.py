"""Gridding corrections of the default mode: the kernel of fractional column offsets (x_interp), the chunk-independent
luminosity window, and the calibration guards (rotation-axis drift, sub-pixel overlap cross-check)."""
import h5py
import numpy as np
import pytest

import helirec
from helirec import _core, calibration
from test_sirt_pipeline import helical_scan, recon

NU = 0.25   # cycles / pixel of the column pattern


def pattern_scan(tmp_path, offset, first=0, nd=24, nu=NU):
    """every frame: 1 + 5 % cos(2 pi nu column); the scan sits `offset` columns right of the merged origin and
    starts `first` angular steps after 0"""
    nr, ngrid = 12, 64
    p = np.arange(193)
    angles, z = (p + first) * (360 / ngrid), p * .06
    row = np.rint(40000 * (1 + .05 * np.cos(2 * np.pi * nu * np.arange(nd)))).astype(np.uint16)
    raw = np.broadcast_to(row, (len(p), nr, nd)).copy()
    path = tmp_path / f"pattern_{offset}_{first}_{nd}_{nu}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=raw, chunks=(1, nr, nd))
    reader = _core.RawReader([str(path)], "data", nr, nd)
    return _core.make_scan_data(angles.astype(np.float64), z.astype(np.float64) * .01, .01, nr, nd, 0, -offset, 0.,
                                np.full(len(z), .2), np.array([0], np.float32), np.zeros((nr, nd), np.float32),
                                np.full((1, nr, nd), 50000, np.float32), np.ones((nr, nd), np.float32), reader,
                                np.zeros(len(z), np.int32), np.arange(len(z), dtype=np.int32))


def amplitude(sino, nu=NU, lo=4, hi=20):
    """amplitude of the nu component over the interior columns (whole periods, away from the detector edges)"""
    x = np.arange(lo, hi)
    s = sino[:, 0, lo:hi].mean(0)
    return 2 * abs(((s - s.mean()) * np.exp(-2j * np.pi * nu * x)).mean())


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("backprojector", ["fbp", "sirt"])   # padded / measured-only gridding
def test_fractional_column_offset_kernel(tmp_path, device, backprojector):
    kw = dict(backprojector=backprojector, tomo_params={"num_iter": "1"} if backprojector == "sirt" else {})
    sino = lambda offset, **k: recon(pattern_scan(tmp_path, offset), device, **kw, **k).reconstruct(7, 8, debug=True)["sino"]
    ref = amplitude(sino(0.0))
    assert ref > 0.04
    # half a pixel: the reference splat averages two columns, cos(pi nu); the kernels keep the pattern
    expect = {"linear": np.cos(np.pi * NU), "cubic": 1.125 * np.cos(np.pi * NU) - 0.125 * np.cos(3 * np.pi * NU),
              "lanczos3": 1.027}
    for kernel, e in expect.items():
        assert abs(amplitude(sino(0.5, x_interp=kernel)) / ref - e) < 0.01, kernel
    # the default, sinc: the rows delayed exactly (see test_fractional_column_offset_exact; this detector is 24
    # columns wide, all of them near an end); measured-only gridding keeps the lanczos3 kernel
    assert abs(amplitude(sino(0.5)) / ref - (expect["lanczos3"] if backprojector == "sirt" else 1.0)) < 0.03
    assert abs(amplitude(sino(0.5, x_interp="sinc")) / amplitude(sino(0.5)) - 1) < 1e-6
    # the legacy reference keeps the splat whatever is asked
    assert abs(amplitude(sino(0.5, x_interp="lanczos3", legacy_reference=True)) / ref - expect["linear"]) < 0.01
    # the accumulated weights are the splat's
    w = lambda k: recon(pattern_scan(tmp_path, 0.5), device, x_interp=k, **kw).reconstruct(7, 8, debug=True)["weights"]
    np.testing.assert_allclose(w("lanczos3"), w("linear"), rtol=1e-5)
    np.testing.assert_allclose(w("sinc"), w("linear"), rtol=1e-5)
    with pytest.raises(Exception):
        recon(pattern_scan(tmp_path, 0.5), device, x_interp="bspline3", **kw)


@pytest.mark.parametrize("device", helirec.devices())
def test_fractional_column_offset_exact(tmp_path, device):
    """x_interp=sinc keeps the whole band of a scan placed at a fractional column offset: at 0.45 cycles / pixel and
    half a pixel the lanczos3 kernel passes 0.35, the two-column splat 0.16."""
    nu, nd, win = 0.45, 160, dict(lo=40, hi=120)    # 80 columns = 36 periods, 40 from the detector's ends
    sino = lambda offset, **k: recon(pattern_scan(tmp_path, offset, nd=nd, nu=nu), device, backprojector="fbp",
                                     **k).reconstruct(7, 8, debug=True)["sino"]
    ref = amplitude(sino(0.0), nu, **win)
    assert ref > 0.04
    lanczos = 2 * (0.6079 * np.cos(np.pi * nu) - 0.1351 * np.cos(3 * np.pi * nu) + 0.0243 * np.cos(5 * np.pi * nu)) / 0.9942
    assert abs(amplitude(sino(0.5, x_interp="sinc"), nu, **win) / ref - 1) < 0.01
    assert abs(amplitude(sino(0.5, x_interp="lanczos3"), nu, **win) / ref - lanczos) < 0.01
    assert abs(amplitude(sino(0.5, x_interp="linear"), nu, **win) / ref - np.cos(np.pi * nu)) < 0.01
    # the pattern lands where it should: the scan at 0.5 is the scan at 0 delayed by half a pixel
    a, b = sino(0.0)[:, 0, 40:120].mean(0), sino(0.5, x_interp="sinc")[:, 0, 40:120].mean(0)
    x = np.arange(40, 120)
    phase = lambda s: np.angle(((s - s.mean()) * np.exp(-2j * np.pi * nu * x)).mean())
    assert abs(np.angle(np.exp(1j * (phase(b) - phase(a) + 2 * np.pi * nu * 0.5)))) < 0.02
    # an integer offset needs no shift: identical with every setting
    np.testing.assert_array_equal(sino(1.0, x_interp="sinc"), sino(1.0, x_interp="lanczos3"))


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("backprojector", ["fbp", "ghbp"])
def test_luminosity_window_is_chunk_independent(tmp_path, device, backprojector):
    scan, *_ = helical_scan(tmp_path)
    companion, *_ = helical_scan(tmp_path, gain=.9)
    r = recon([scan, companion], device, backprojector=backprojector, luminosity_repartition=100.)
    assert r.info["margin"] == 12                      # the window's context rows
    chunk = r.reconstruct(5, 10, debug=True)
    single = [r.reconstruct(h, h + 1, debug=True) for h in range(5, 10)]
    np.testing.assert_allclose(chunk["sino"], np.concatenate([s["sino"] for s in single], axis=1), atol=2e-6)
    np.testing.assert_allclose(chunk["slices"], np.concatenate([s["slices"] for s in single]), atol=2e-6, rtol=2e-5)
    # the legacy reference clips the window to the chunk, as the reference does
    assert recon([scan, companion], device, backprojector=backprojector, luminosity_repartition=100.,
                 legacy_reference=True).info["margin"] == 0


@pytest.mark.parametrize("device", helirec.devices())
def test_angle_subsets(tmp_path, device):
    """the two halves of a scan: half the projections each, on half the angles, the same object"""
    scan, *_ = helical_scan(tmp_path)
    kw = dict(backprojector="fbp", tomo_params={})
    whole = recon(scan, device, **kw)
    halves = [recon(scan, device, n_grid=0, angle_subsets=2, angle_subset=i, **kw) for i in (0, 1)]
    assert whole.info["projections"] == [193] and [h.info["projections"] for h in halves] == [[97], [96]]
    assert all(h.info["n_grid"] == 32 and h.info["doable"] == whole.info["doable"] for h in halves)
    a = whole.reconstruct(6, 9)
    b = [h.reconstruct(6, 9) for h in halves]
    for x in b:
        assert np.abs(x - a).max() < 0.03 * np.abs(a).max()
    assert np.abs(b[0] - b[1]).max() > 0                       # (not the same projections)
    # subsets go by angle: a companion that starts one step later keeps the main scan's angles in each half
    pair = [pattern_scan(tmp_path, 0.0), pattern_scan(tmp_path, 0.5, first=1)]
    counts = [recon(pair, device, n_grid=0, angle_subsets=2, angle_subset=i, **kw).info["projections"] for i in (0, 1)]
    assert counts == [[97, 96], [96, 97]]
    with pytest.raises(Exception):
        recon(scan, device, angle_subsets=2, angle_subset=2, **kw)
    with pytest.raises(Exception):
        recon(scan, device, angle_subsets=2, angle_subset=0, legacy_reference=True, **kw)


def test_rotation_axis_drift_is_an_error():
    msgs = []
    calibration.check_cor_drift((1684.75, 1684.75), log=msgs.append)
    assert not msgs
    calibration.check_cor_drift((1684.75, 1684.875), log=msgs.append)     # one search step: noted, accepted
    assert len(msgs) == 1
    with pytest.raises(NotImplementedError, match="drifts by \\+0.500 px"):
        calibration.check_cor_drift((1684.75, 1685.25), log=msgs.append)


def test_overlap_cross_check_flags_noise_locked_offsets():
    """noisy stripes at a whole-pixel offset: the L1 search against the bilinearly interpolated companion locks to
    half pixels; the smoothed search does not, and the guard reports the disagreement"""
    rng = np.random.default_rng(0)
    g = lambda a, s: _core.gaussian_filter1d(_core.gaussian_filter1d(a, 1, s, "reflect"), 2, s, "reflect")
    texture = g(rng.standard_normal((6, 81, 400)), 1.2)
    texture /= texture.std()
    w = np.ones_like(texture)
    for noise, flagged in ((0.05, False), (1.0, True)):
        a = texture + noise * rng.standard_normal(texture.shape)
        b = texture + noise * rng.standard_normal(texture.shape)
        res = _core.find_offset(a, w, b, w, 0, 0, 3, 3)
        msgs = []
        check = calibration.overlap_cross_check(a, w, b, w, res, log=msgs.append)
        assert check["sub"] == (0.0, 0.0)
        assert check["differs"] is flagged and len(msgs) == int(flagged)
        assert (tuple(res["best_sub"]) != (0.0, 0.0)) is flagged


def rolled_scan(tmp_path, offset, slope, nu=0.2, nr=40, nd=48, c0=20.25):
    """a stationary scan whose frames show 1 + 5 % cos(2 pi nu row), displaced along the rows by `offset` and sheared
    by `slope` rows per column about column c0; with its registration (row_offset, row_slope) when `register`"""
    p = np.arange(129)
    r, c = np.mgrid[:nr, :nd].astype(np.float64)

    def make(register):
        frame = np.rint(40000 * (1 + .05 * np.cos(2 * np.pi * nu * (r - offset - slope * (c - c0))))).astype(np.uint16)
        path = tmp_path / f"rolled_{offset}_{slope}.h5"
        if not path.exists():
            with h5py.File(path, "w") as f:
                f.create_dataset("data", data=np.broadcast_to(frame, (len(p), nr, nd)).copy(), chunks=(1, nr, nd))
        reader = _core.RawReader([str(path)], "data", nr, nd)
        reg = dict(row_offset=[-offset] * len(p), row_slope=[slope] * len(p), slope_col=c0) if register else {}
        return _core.make_scan_data(p * (360 / 64), np.zeros(len(p)), .01, nr, nd, 0, 0., 0., np.full(len(p), .2),
                                    np.array([0], np.float32), np.zeros((nr, nd), np.float32),
                                    np.full((1, nr, nd), 50000, np.float32), np.ones((nr, nd), np.float32), reader,
                                    np.zeros(len(p), np.int32), np.arange(len(p), dtype=np.int32), **reg)
    return make


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("backprojector", ["fbp", "sirt"])   # padded / measured-only gridding
def test_frame_registration_undoes_offset_and_roll(tmp_path, device, backprojector):
    """frames displaced by 0.3 rows and sheared by 0.08 rows per column (up to 2.2 rows) grid like the plain ones when
    their registration is given (ScanTrajectory::row_offset, row_slope)"""
    kw = dict(backprojector=backprojector, tomo_params={"num_iter": "1"} if backprojector == "sirt" else {},
              rot_center=23.5, grid_size=33)
    sino = lambda scan, **k: recon(scan, device, **kw, **k).reconstruct(12, 28, debug=True)["sino"][:, :, 6:42]
    plain = sino(rolled_scan(tmp_path, 0.0, 0.0)(False))
    assert np.ptp(plain.mean((0, 2))) > 0.08                 # the row pattern is there
    registered = sino(rolled_scan(tmp_path, 0.3, 0.08)(True))
    left = sino(rolled_scan(tmp_path, 0.3, 0.08)(False))
    assert np.abs(registered - plain).max() < 4e-3           # (lanczos3 at 0.2 cycles / row: 0.3 % ripple)
    assert np.abs(left - plain).max() > 0.05
    # the linear vertical kernel takes the same positions
    lin = lambda scan: sino(scan, z_interp="linear")
    assert np.abs(lin(rolled_scan(tmp_path, 0.3, 0.08)(True)) - plain).max() < 0.02
    with pytest.raises(Exception):                           # one value per projection
        _core.make_scan_data(np.arange(4.) * 5, np.zeros(4), .01, 8, 8, 0, 0., 0., np.full(4, .2), np.array([0], np.float32),
                             np.zeros((8, 8), np.float32), np.ones((1, 8, 8), np.float32), np.ones((8, 8), np.float32),
                             _core.RawReader([str(tmp_path / "rolled_0.0_0.0.h5")], "data", 40, 48), np.zeros(4, np.int32),
                             np.arange(4, dtype=np.int32), row_slope=[0.1, 0.1]) and recon(None, device)


@pytest.mark.parametrize("device", helirec.devices())
def test_band_weights_without_smear_change_nothing(tmp_path, device):
    """detector_smear: a zero field gives both weight sets the same weights, and the result of the usual ones bit for
    bit; a field leaves every ray an average of its recordings (weights that sum to one)"""
    run = lambda **k: recon(pattern_scan(tmp_path, 0.5, nd=160, nu=0.45), device, backprojector="fbp",
                            **k).reconstruct(7, 8, debug=True)["sino"]
    usual = run()
    np.testing.assert_array_equal(run(detector_smear="0:0:0:0:0:0"), usual)
    smeared = run(detector_smear="20:0:0:0:1:0")             # (3.2 px across this small detector)
    assert np.isfinite(smeared).all() and abs(float(smeared.mean() - usual.mean())) < 1e-3
    for bad in dict(detector_smear="1:2:3"), dict(detector_smear="0:0:0:0:0:0", sharp_band="0.3:0.2"), dict(detector_smear="a:0:0:0:0:0"):
        with pytest.raises(Exception):
            run(**bad)
