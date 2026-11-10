"""Gridding corrections of the default mode: the kernel of fractional column offsets (x_interp), the chunk-independent
luminosity window, and the calibration guards (rotation-axis drift, sub-pixel overlap cross-check)."""
import h5py
import numpy as np
import pytest

import helirec
from helirec import _core, calibration
from test_sirt_pipeline import helical_scan, recon

NU = 0.25   # cycles / pixel of the column pattern


def pattern_scan(tmp_path, offset):
    """every frame: 1 + 5 % cos(2 pi NU column); the scan sits `offset` columns right of the merged origin"""
    nr, nd, ngrid = 12, 24, 64
    p = np.arange(193)
    angles, z = p * (360 / ngrid), p * .06
    row = np.rint(40000 * (1 + .05 * np.cos(2 * np.pi * NU * np.arange(nd)))).astype(np.uint16)
    raw = np.broadcast_to(row, (len(p), nr, nd)).copy()
    path = tmp_path / f"pattern_{offset}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=raw, chunks=(1, nr, nd))
    reader = _core.RawReader([str(path)], "data", nr, nd)
    return _core.make_scan_data(angles.astype(np.float64), z.astype(np.float64) * .01, .01, nr, nd, 0, -offset, 0.,
                                np.full(len(z), .2), np.array([0], np.float32), np.zeros((nr, nd), np.float32),
                                np.full((1, nr, nd), 50000, np.float32), np.ones((nr, nd), np.float32), reader,
                                np.zeros(len(z), np.int32), np.arange(len(z), dtype=np.int32))


def amplitude(sino):
    """amplitude of the NU component over the interior columns (whole periods, away from the detector edges)"""
    x = np.arange(4, 20)
    s = sino[:, 0, 4:20].mean(0)
    return 2 * abs(((s - s.mean()) * np.exp(-2j * np.pi * NU * x)).mean())


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
    assert abs(amplitude(sino(0.5)) / ref - expect["lanczos3"]) < 0.01                      # the default
    # the legacy reference keeps the splat whatever is asked
    assert abs(amplitude(sino(0.5, x_interp="lanczos3", legacy_reference=True)) / ref - expect["linear"]) < 0.01
    # the accumulated weights are the splat's
    w = lambda k: recon(pattern_scan(tmp_path, 0.5), device, x_interp=k, **kw).reconstruct(7, 8, debug=True)["weights"]
    np.testing.assert_allclose(w("lanczos3"), w("linear"), rtol=1e-5)
    with pytest.raises(Exception):
        recon(pattern_scan(tmp_path, 0.5), device, x_interp="bspline3", **kw)


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
