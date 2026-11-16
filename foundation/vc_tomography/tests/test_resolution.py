"""Resolution corrections: the fused MTF / Paganin Wiener filter, the vertical kernel (z_interp), nearest-bin angular
gridding at the measured angles (angle_interp) and the exposure-arc deconvolution."""
import json

import h5py
import numpy as np
import pytest

import helirec
from helirec import _core
from test_sirt_pipeline import helical_scan, recon

PH = dict(energy_kev=116.0, distance_m=1.19, pixel_size_m=8.64e-6, det_rows=1200, det_cols=2528)
MTF = {"type": "mtf_paganin", "delta_beta": 1000, "sigma": 1.06, "nsr": 0.01}


# ------------------------------------------------------------------ mtf_paganin
def response(chain, shape=(151, 192), device="cpu"):
    """frequency response of a chain, from its impulse response"""
    a = np.zeros((1, *shape), np.float32)
    a[0, shape[0] // 2, shape[1] // 2] = 1
    out = helirec.apply_filters(a, chain, helirec.filter_margin(chain, **PH), device=device, **PH)[0]
    return np.abs(np.fft.fft2(np.fft.ifftshift(out)))


def test_mtf_paganin_response_and_fusion():
    assert helirec.filter_describe([MTF], **PH).startswith("fft[mtf_paganin]@intensity m")
    m_pag = helirec.filter_margin([{"type": "paganin", "delta_beta": 1000}], **PH)
    assert m_pag < helirec.filter_margin([MTF], **PH) <= m_pag + 64
    # without phase retrieval: the Wiener inverse of the Gaussian MTF, (1 + nsr) M / (M^2 + nsr)
    sigma, nsr = 1.06, 0.01
    R = response([dict(MTF, delta_beta=0)])
    f0, f1 = np.meshgrid(np.fft.fftfreq(R.shape[0]), np.fft.fftfreq(R.shape[1]), indexing="ij")
    M = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (f0 ** 2 + f1 ** 2))
    D = (1 + nsr) * M / (M ** 2 + nsr)
    np.testing.assert_allclose(R, D, atol=0.035)           # (the kernel is truncated to its margin rows)
    assert abs(D.max() - (1 + nsr) / (2 * np.sqrt(nsr))) < 0.05 and abs(R[0, 0] - 1) < 1e-5
    # fused with Paganin: the product of the two responses
    P = response([{"type": "paganin", "delta_beta": 1000}])
    np.testing.assert_allclose(response([MTF]), P * D, atol=0.035)
    # a constant stays a constant
    flat = helirec.apply_filters(np.full((2, 151, 192), 3.0, np.float32), [MTF], helirec.filter_margin([MTF], **PH), **PH)
    np.testing.assert_allclose(flat, 3.0, rtol=2e-6)
    # a tabulated MTF: the Gaussian sampled
    nu = np.linspace(0, 0.75, 76)
    table = "/".join(f"{v:.4f}:{np.exp(-2 * np.pi ** 2 * sigma ** 2 * v ** 2):.8f}" for v in nu)
    tab = {"type": "mtf_paganin", "delta_beta": 1000, "mtf": table, "nsr": nsr}
    np.testing.assert_allclose(response([tab]), response([MTF]), atol=5e-3)
    for bad in (dict(MTF, sigma=0), dict(MTF, nsr=0), dict(MTF, mtf="0.1:0.5"), dict(tab, mtf="0.2:0.5/0.1:0.6"),
                dict(tab, mtf="0.1:1.5")):
        with pytest.raises(Exception, match="mtf_paganin"):
            helirec.filter_margin([bad], **PH)


def test_mtf_paganin_target_form():
    # the default (no nsr): the blur replaced by a Gaussian of `target` pixels, the gain capped at ~gain
    sigma, target, gain = 1.06, 0.5, 12.0
    base = {"type": "mtf_paganin", "delta_beta": 0, "sigma": sigma}
    R = response([base])
    f0, f1 = np.meshgrid(np.fft.fftfreq(R.shape[0]), np.fft.fftfreq(R.shape[1]), indexing="ij")
    nu2 = f0 ** 2 + f1 ** 2
    M, e = np.exp(-2 * np.pi ** 2 * sigma ** 2 * nu2), 1 / gain

    def D(target, gain):
        return np.exp(-2 * np.pi ** 2 * target ** 2 * nu2) * np.sqrt((1 + 1 / gain ** 2) / (M ** 2 + 1 / gain ** 2))
    np.testing.assert_allclose(R, D(target, gain), atol=0.035)
    np.testing.assert_allclose(response([dict(base, target=0.5, gain=12)]), R, atol=1e-6)      # the defaults
    np.testing.assert_allclose(response([dict(base, target=0.8, gain=6)]), D(0.8, 6), atol=0.035)
    np.testing.assert_allclose(response([dict(base, target=0, gain=6)]), D(0, 6), atol=0.035)   # the gain cap alone
    assert abs(R[0, 0] - 1) < 1e-5 and R.max() < gain
    # the system response M x D falls monotonically along a frequency axis: no pass-band hump (the Wiener form's
    # gain peaks at M = sqrt(nsr), which rings at edges and colours the noise)
    axis = (M * D(target, gain))[0, : R.shape[1] // 2]
    assert np.all(np.diff(axis) < 0) and abs(axis[0] - 1) < 1e-12
    k = int(np.argmin(np.abs(M[0, : R.shape[1] // 2] - 0.5)))                                   # where M >> 1 / gain
    assert abs(axis[k] - np.exp(-2 * np.pi ** 2 * target ** 2 * nu2[0, k])) < 0.02              # ... it is the target
    # fused with Paganin, constants, and the parameter checks
    full = dict(base, delta_beta=1000)
    np.testing.assert_allclose(response([full]), response([{"type": "paganin", "delta_beta": 1000}]) * D(target, gain), atol=0.035)
    flat = helirec.apply_filters(np.full((2, 151, 192), 3.0, np.float32), [full], helirec.filter_margin([full], **PH), **PH)
    np.testing.assert_allclose(flat, 3.0, rtol=2e-6)
    for bad in (dict(full, target=-1), dict(full, gain=1), dict(full, nsr=0.01, target=0.5), dict(full, nsr=0.01, gain=8)):
        with pytest.raises(Exception, match="mtf_paganin"):
            helirec.filter_margin([bad], **PH)


@pytest.mark.skipif(not helirec.cuda_available(), reason="no CUDA")
@pytest.mark.parametrize("chain", [MTF, {"type": "mtf_paganin", "delta_beta": 1000, "sigma": 1.06}], ids=["wiener", "target"])
def test_mtf_paganin_cpu_cuda_agree(chain):
    a = np.random.default_rng(0).uniform(0.5, 1.5, (3, 151, 200)).astype(np.float32)
    m = helirec.filter_margin([chain], **PH)
    np.testing.assert_allclose(helirec.apply_filters(a, [chain], m, device="cuda", **PH),
                               helirec.apply_filters(a, [chain], m, device="cpu", **PH), rtol=0, atol=5e-5)


# ------------------------------------------------------------------ wiener (measured tables)
def wiener_table(path, nu, transfer, theta=None, **snr):
    path.write_text(json.dumps(dict(nu=list(map(float, nu)), transfer=np.asarray(transfer).tolist(),
                                    **({"theta": list(map(float, theta))} if theta is not None else {}),
                                    **{k: np.asarray(v).tolist() for k, v in snr.items()})))
    return {"type": "wiener", "table": str(path)}


def test_wiener_table_filter(tmp_path):
    sigma, nsr = 1.06, 0.01
    nu = np.arange(1, 161) / 200.0
    M = np.exp(-2 * np.pi ** 2 * sigma ** 2 * nu ** 2)
    # a Gaussian MTF and white noise as tables: mtf_paganin's Wiener form without phase retrieval
    w = wiener_table(tmp_path / "radial.json", nu, M, snr=M ** 2 / nsr, half=M ** 2 / (2 * nsr))
    assert helirec.filter_describe([w], **PH).startswith("fft[wiener]@intensity m")
    R = response([w])
    f0, f1 = np.meshgrid(np.fft.fftfreq(R.shape[0]), np.fft.fftfreq(R.shape[1]), indexing="ij")
    Mg = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (f0 ** 2 + f1 ** 2))
    D = (1 + nsr) * Mg / (Mg ** 2 + nsr)
    tol = 0.03 * D.max()                                   # (the kernel is truncated to its margin rows)
    np.testing.assert_allclose(R, D, atol=tol)
    np.testing.assert_allclose(R, response([dict(MTF, delta_beta=0)]), atol=tol)
    # (unit DC; this table's gain is 0.99 from the first frequency on, a step the impulse's small frame cuts short)
    assert abs(R[0, 0] - 1) < 2e-4 and 1 <= helirec.filter_margin([w], **PH) <= 64
    # weight scales the noise; snr=NAME picks another table of the file; target is a Gaussian on top
    np.testing.assert_allclose(response([dict(w, weight=2)]), (1 + 2 * nsr) * Mg / (Mg ** 2 + 2 * nsr), atol=tol)
    np.testing.assert_allclose(response([dict(w, snr="half")]), response([dict(w, weight=2)]), atol=1e-6)
    T = np.exp(-2 * np.pi ** 2 * 0.7 ** 2 * (f0 ** 2 + f1 ** 2))
    np.testing.assert_allclose(response([dict(w, target=0.7)]), T * D, atol=tol)
    # weight 0: the plain inverse of the transfer (a mild one: the inverse of a strong blur has no usable kernel)
    Ms = np.exp(-2 * np.pi ** 2 * 0.4 ** 2 * nu ** 2)
    w0 = wiener_table(tmp_path / "mild.json", nu, Ms, snr=np.full(len(nu), 3.0))
    np.testing.assert_allclose(response([dict(w0, weight=0)]), np.exp(2 * np.pi ** 2 * 0.4 ** 2 * (f0 ** 2 + f1 ** 2)), rtol=0.03)
    # a table over the angle too: the rows' own transfer and noise along the row axis, the columns' along theirs
    # (as measured tables are: the signal-to-noise ratio large at low frequencies, falling smoothly)
    def mz(v):
        return np.exp(-2 * np.pi ** 2 * 0.6 ** 2 * v ** 2)

    def mx(v):
        return np.exp(-2 * np.pi ** 2 * sigma ** 2 * v ** 2)

    def snr_of(m, v):
        return m ** 2 / 0.2 * (1 + (0.15 / np.maximum(v, 1e-9)) ** 4)
    w2 = wiener_table(tmp_path / "polar.json", nu, np.stack([mz(nu), mx(nu)], 1), theta=[0, 90],
                      snr=np.stack([snr_of(mz(nu), nu), snr_of(mx(nu), nu)], 1))
    R2 = response([w2])
    v0, v1 = np.abs(f0[1:, 0]), np.abs(f1[0, 1:])
    along_rows = snr_of(mz(v0), v0) / (snr_of(mz(v0), v0) + 1) / mz(v0)
    along_cols = snr_of(mx(v1), v1) / (snr_of(mx(v1), v1) + 1) / mx(v1)
    np.testing.assert_allclose(R2[1:, 0], along_rows, atol=0.05)
    np.testing.assert_allclose(R2[0, 1:], along_cols, atol=0.08)
    assert abs(along_rows.max() - along_cols.max()) > 0.1 and abs(R2[0, 0] - 1) < 1e-5    # (the two axes do differ)
    # a constant stays a constant
    flat = helirec.apply_filters(np.full((2, 151, 192), 3.0, np.float32), [w2], helirec.filter_margin([w2], **PH), **PH)
    np.testing.assert_allclose(flat, 3.0, rtol=2e-6)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(nu=[0.2, 0.1], transfer=[1, 1], snr=[1, 1])))
    for chain in (dict(w, weight=-1), dict(w, target=-1), dict(w, snr="none"), {"type": "wiener"},
                  {"type": "wiener", "table": str(tmp_path / "missing.json")}, {"type": "wiener", "table": str(bad)}):
        with pytest.raises(Exception, match="wiener"):
            helirec.filter_margin([chain], **PH)
    if helirec.cuda_available():
        a = np.random.default_rng(0).uniform(0.5, 1.5, (3, 151, 200)).astype(np.float32)
        m = helirec.filter_margin([w2], **PH)
        np.testing.assert_allclose(helirec.apply_filters(a, [w2], m, device="cuda", **PH),
                                   helirec.apply_filters(a, [w2], m, device="cpu", **PH), rtol=0, atol=5e-5)


# ------------------------------------------------------------------ z_interp
NU = 0.25   # cycles / row of the row pattern


def row_scan(tmp_path, nr=32, values=None, angles=None, name="rows", nu=NU):
    """frames constant along the columns: 1 + 5 % cos(2 pi nu row), or `values` per projection"""
    nd, ngrid = 24, 64
    p = np.arange(193)
    angles = p * (360 / ngrid) if angles is None else np.asarray(angles, np.float64)
    z = np.arange(len(angles)) * .06
    if values is None:
        raw = np.broadcast_to(np.rint(40000 * (1 + .05 * np.cos(2 * np.pi * nu * np.arange(nr))))[None, :, None],
                              (len(z), nr, nd)).astype(np.uint16)
    else:
        raw = np.broadcast_to(np.asarray(values)[:, None, None], (len(z), nr, nd)).astype(np.uint16)
    path = tmp_path / f"{name}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=raw, chunks=(1, nr, nd))
    reader = _core.RawReader([str(path)], "data", nr, nd)
    scan = _core.make_scan_data(angles, z.astype(np.float64) * .01, .01, nr, nd, 0, 0., 0., np.full(len(z), .2),
                                np.array([0], np.float32), np.zeros((nr, nd), np.float32),
                                np.full((1, nr, nd), 50000, np.float32), np.ones((nr, nd), np.float32), reader,
                                np.zeros(len(z), np.int32), np.arange(len(z), dtype=np.int32))
    return scan, angles, z, raw


Z_TAPS = {"linear": 2, "cubic": 4, "lanczos3": 6, "lanczos8": 16}


def kernel_weights(f, kind):
    T = Z_TAPS[kind]
    d = np.abs((np.arange(T) - (T // 2 - 1) - np.float32(f)).astype(np.float32))
    if kind == "cubic":
        w = np.where(d < 1, (1.5 * d - 2.5) * d * d + 1, np.where(d < 2, ((-.5 * d + 2.5) * d - 4) * d + 2, 0))
    else:
        a = T // 2   # lanczos3, lanczos8
        w = np.where(d < 1e-6, 1, np.where(d >= a, 0, np.sinc(d) * np.sinc(d / a)))
    return w / w.sum()


def vertical_reference(raw, z, h, kind, log_first, nr):
    """sinogram column of slice h: per grid angle, the mean over the projections seeing it of the frame value at row
    h - z, by the kernel where its rows are on the detector, else linearly"""
    out = []
    for a in range(64):
        values = []
        for i in np.flatnonzero(np.arange(len(z)) % 64 == a):
            pos = h - np.float32(z[i])
            lo = int(np.floor(pos))
            f = pos - lo
            if lo < 0 or lo >= nr or (f > 0 and lo + 1 >= nr):
                continue
            col = raw[i, :, 10] / 50000.
            col = -np.log(col) if log_first else col
            T = Z_TAPS[kind]
            first = lo - (T // 2 - 1)
            if kind != "linear" and f > 0 and first >= 0 and first + T <= nr:
                values.append(float(kernel_weights(f, kind) @ col[first:first + T]))
            else:
                values.append((1 - f) * col[lo] + f * col[min(lo + 1, nr - 1)])
        out.append(np.mean(values))
    return np.array(out) if log_first else -np.log(out)


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("backprojector", ["fbp", "sirt"])   # padded intensity / measured-only line integrals
def test_vertical_kernel(tmp_path, device, backprojector):
    kw = dict(backprojector=backprojector, tomo_params={"num_iter": "1"} if backprojector == "sirt" else {})
    # a short detector: slices whose rows reach its edges fall back to the linear interpolation there
    scan, _, z, raw = row_scan(tmp_path, nr=12, name="short")
    for kind in ("linear", "cubic", "lanczos3", "lanczos8"):   # (lanczos8's 16 rows never fit: linear throughout)
        got = recon(scan, device, z_interp=kind, **kw).reconstruct(7, 8, debug=True)
        ref = vertical_reference(raw, z, 7, kind, backprojector == "sirt", 12)
        np.testing.assert_allclose(got["sino"][:, 0, 10], ref, atol=2e-5, err_msg=kind)
    # every row on the detector: the kernels follow the pattern, the linear interpolation flattens it
    scan, _, z, raw = row_scan(tmp_path, nr=32)
    h = 20
    ideal = np.array([np.mean([-np.log(.8 * (1 + .05 * np.cos(2 * np.pi * NU * (h - np.float32(z[i])))))
                               for i in np.flatnonzero(np.arange(len(z)) % 64 == a)]) for a in range(64)])
    err, weights = {}, {}
    for kind in ("linear", "cubic", "lanczos3"):
        got = recon(scan, device, z_interp=kind, **kw).reconstruct(h, h + 1, debug=True)
        err[kind] = np.abs(got["sino"][:, 0, 10] - ideal).max()
        weights[kind] = got["weights"]
    assert err["linear"] > 5e-3 and err["cubic"] < 0.5 * err["linear"] and err["lanczos3"] < 0.2 * err["linear"], err
    np.testing.assert_allclose(weights["lanczos3"], weights["linear"], rtol=1e-6)        # the weights stay linear
    # the legacy reference interpolates linearly whatever is asked
    legacy = recon(scan, device, z_interp="lanczos3", legacy_reference=True, **kw).reconstruct(h, h + 1, debug=True)
    assert np.abs(legacy["sino"][:, 0, 10] - ideal).max() > 5e-3
    with pytest.raises(Exception, match="z_interp"):
        recon(scan, device, z_interp="bspline3", **kw)


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("backprojector", ["fbp", "sirt"])
def test_vertical_kernel_lanczos8(tmp_path, device, backprojector):
    """z_interp=lanczos8: 16 rows, the given weights; at 0.4 cycles per row, where lanczos3 loses a fifth of the
    pattern over the fractions of a helix, it keeps it"""
    kw = dict(backprojector=backprojector, tomo_params={"num_iter": "1"} if backprojector == "sirt" else {})
    nu, nr, h = 0.4, 64, 40           # rows h - z = 28.5 .. 40: every kernel's rows are on the detector
    scan, _, z, raw = row_scan(tmp_path, nr=nr, name="fine", nu=nu)
    ideal = np.array([np.mean([-np.log(.8 * (1 + .05 * np.cos(2 * np.pi * nu * (h - np.float32(z[i])))))
                               for i in np.flatnonzero(np.arange(len(z)) % 64 == a)]) for a in range(64)])
    err = {}
    for kind in ("lanczos3", "lanczos8"):
        r = recon(scan, device, z_interp=kind, **kw)
        assert r.info["z_interp"] == kind
        got = r.reconstruct(h, h + 1, debug=True)
        ref = vertical_reference(raw, z, h, kind, backprojector == "sirt", nr)
        np.testing.assert_allclose(got["sino"][:, 0, 10], ref, atol=2e-5, err_msg=kind)
        err[kind] = np.abs(got["sino"][:, 0, 10] - ideal).max()
    assert err["lanczos3"] > 3e-3 and err["lanczos8"] < 0.4 * err["lanczos3"], err   # (measured: 4.95e-3, 1.25e-3)


@pytest.mark.parametrize("device", helirec.devices())
def test_vertical_kernel_chunks_and_read_ahead(tmp_path, device):
    scan, *_ = helical_scan(tmp_path)
    kw = dict(backprojector="fbp", z_interp="lanczos3", filters=[{"type": "paganin", "delta_beta": 1000.}], pixel_size_m=1e-5)
    r = recon(scan, device, **kw)
    plain = [r.reconstruct(h, h + 1) for h in (7, 8, 9)]
    chunk = r.reconstruct(7, 10)
    np.testing.assert_allclose(chunk * .001, np.concatenate(plain) * .001, atol=2e-6, rtol=2e-5)
    r = recon(scan, device, **kw)
    ahead = [r.reconstruct(h, h + 1, next=(h + 1, h + 2) if h < 9 else None) for h in (7, 8, 9)]
    np.testing.assert_array_equal(np.concatenate(ahead), np.concatenate(plain))
    assert np.abs(plain[1] - recon(scan, device, **dict(kw, z_interp="linear")).reconstruct(8, 9)).max() > 0
    if device == "cuda":
        np.testing.assert_allclose(chunk * .001, recon(scan, "cpu", **kw).reconstruct(7, 10) * .001, atol=2e-6, rtol=2e-4)
        del r
        helirec.release_pinned_memory()


# ------------------------------------------------------------------ angle_interp
@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("backprojector", ["fbp", "sirt"])
def test_nearest_angle_bins(tmp_path, device, backprojector):
    """angles off the grid and slowly drifting, one projection missing: each projection lands whole in its bin, and
    the slice is backprojected at the bins' measured mean angles"""
    ngrid, nr, h = 64, 32, 20
    p = np.delete(np.arange(193), 70)
    angles = (p * .9985 + .37) * (360 / ngrid)
    values = 30000 + 50 * np.arange(len(p))
    scan, angles, z, raw = row_scan(tmp_path, nr=nr, values=values, angles=angles, name="drift")
    kw = dict(backprojector=backprojector, tomo_params={"num_iter": "1"} if backprojector == "sirt" else {})
    got = recon(scan, device, angle_interp="nearest", **kw).reconstruct(h, h + 1, debug=True)
    f = angles / (360 / ngrid)
    bins = np.concatenate([[0], np.cumsum(np.rint(np.diff(f)).astype(int))])     # consecutive projections, consecutive bins
    assert bins[70] - bins[69] == 2                                               # the missing projection's bin is skipped
    sino, angle, filled = np.zeros(ngrid), np.zeros(ngrid), np.zeros(ngrid, bool)
    for b in range(ngrid):
        pos = h - z.astype(np.float32)
        sel = np.flatnonzero((bins % ngrid == b) & (pos >= 0) & (pos < nr - 1))
        if len(sel):
            v = raw[sel, 0, 10] / 50000.
            sino[b] = np.mean(-np.log(v)) if backprojector == "sirt" else -np.log(np.mean(v))
            angle[b] = (b + np.mean(f[sel] - bins[sel])) * 2 * np.pi / ngrid
            filled[b] = True
    assert filled.sum() >= ngrid - 1
    np.testing.assert_allclose(got["sino"][filled, 0, 10], sino[filled], atol=3e-6)
    np.testing.assert_allclose(got["angles"][0, filled], angle[filled], atol=2e-6)
    # the splat averages neighbouring projections instead, and the nominal grid angles are used
    lin = recon(scan, device, angle_interp="linear", **kw).reconstruct(h, h + 1, debug=True)
    assert np.abs(lin["sino"][filled, 0, 10] - sino[filled]).max() > 1e-3
    np.testing.assert_allclose(lin["angles"][0], np.arange(ngrid) * 2 * np.pi / ngrid, atol=1e-6)
    with pytest.raises(Exception, match="angle_interp"):
        recon(scan, device, angle_interp="cubic", **kw)


@pytest.mark.parametrize("device", helirec.devices())
def test_nearest_angles_reconstruct_the_phantom(tmp_path, device):
    scan, *_ = helical_scan(tmp_path)      # angles 0.17 of a step off the grid
    yy, xx = np.mgrid[:17, :17] - 8
    mask = xx ** 2 + yy ** 2 < 6 ** 2
    err = {}
    for kind in ("linear", "nearest"):
        r = recon(scan, device, backprojector="fbp", angle_interp=kind)
        chunk = r.reconstruct(7, 10)
        single = np.concatenate([r.reconstruct(h, h + 1) for h in range(7, 10)])
        np.testing.assert_allclose(chunk * .01, single * .01, atol=2e-6, rtol=2e-5)     # no chunk dependence
        truth = np.stack([.1 * (1 + .03 * h) * np.exp(-.5 * ((xx - 2) ** 2 + (yy + 1) ** 2) / 4) for h in range(7, 10)])
        err[kind] = np.linalg.norm((chunk - truth)[:, mask]) / np.linalg.norm(truth[:, mask])
    assert err["nearest"] < err["linear"], err
    # the default is nearest; ghbp (its plan fixes the angles) and the legacy reference keep the splat
    assert recon(scan, device, backprojector="fbp").info["angle_interp"] == "nearest"
    assert recon(scan, device, backprojector="ghbp").info["angle_interp"] == "linear"
    assert recon(scan, device, backprojector="fbp", legacy_reference=True).info["angle_interp"] == "linear"
    assert recon(scan, device, backprojector="fbp").info["z_interp"] == "lanczos8"
    assert recon(scan, device, backprojector="fbp", legacy_reference=True).info["z_interp"] == "linear"
    if device == "cuda":
        cpu = recon(scan, "cpu", backprojector="fbp", angle_interp="nearest").reconstruct(7, 10)
        np.testing.assert_allclose(chunk * .01, cpu * .01, atol=2e-6, rtol=2e-4)


# ------------------------------------------------------------------ exposure arc
def test_arc_deblur_taps():
    for arc in (0.3, 0.91, 1.2):
        g = np.asarray(_core.arc_deblur_taps(arc))
        assert len(g) == 5 and abs(g[0] + 2 * g[1:].sum() - 1) < 1e-6
        f = np.linspace(0, 0.45, 46)
        H = g[0] + 2 * sum(g[k] * np.cos(2 * np.pi * f * k) for k in range(1, 5))
        np.testing.assert_allclose(H * np.sinc(arc * f), 1, atol=0.04)     # box blur x its inverse
    with pytest.raises(Exception):
        _core.arc_deblur_taps(2.0)


@pytest.mark.parametrize("device", helirec.devices())
def test_angular_fir(device):
    rng = np.random.default_rng(5)
    s = rng.standard_normal((40, 9)).astype(np.float32)
    g = np.asarray(_core.arc_deblur_taps(0.91), np.float32)
    ref = sum(g[abs(n)] * np.roll(s, -n, axis=0) for n in range(-4, 5))
    np.testing.assert_allclose(_core.angular_fir(s, g, device=device), ref, atol=2e-6)
    support = np.ones_like(s)
    support[7, 3] = 0
    out = _core.angular_fir(s, g, support, device=device)
    touched = np.zeros(s.shape, bool)
    touched[np.arange(3, 12), 3] = True                     # every output with the unsupported sample among its taps
    np.testing.assert_array_equal(out[touched], s[touched])
    np.testing.assert_allclose(out[~touched], ref[~touched], atol=2e-6)


@pytest.mark.parametrize("device", helirec.devices())
def test_exposure_arc_in_the_pipeline(tmp_path, device):
    scan, *_ = helical_scan(tmp_path, holes=True)
    g = np.asarray(_core.arc_deblur_taps(0.9), np.float32)
    # measured-only rows (sirt): deconvolved where every tap is measured
    d = recon(scan, device, exposure_arc=0.9, tomo_params={"num_iter": "1"}).reconstruct(7, 8, debug=True)
    assert abs(recon(scan, device, exposure_arc=0.9).info["exposure_arc_bins"] - 0.9) < 1e-6
    ref = _core.angular_fir(np.ascontiguousarray(d["sino"][:, 0]), g, np.ascontiguousarray(d["weights"][:, 0]))
    np.testing.assert_allclose(d["filtered"][0], ref, atol=2e-6)
    assert np.abs(d["filtered"][0] - d["sino"][:, 0]).max() > 1e-4
    # fbp: every chunking gives the same slices, and they differ from the uncorrected ones
    r = recon(scan, device, backprojector="fbp", exposure_arc=0.9)
    chunk = r.reconstruct(7, 10)
    np.testing.assert_allclose(chunk * .01, np.concatenate([r.reconstruct(h, h + 1) for h in range(7, 10)]) * .01, atol=2e-6, rtol=2e-5)
    assert np.abs(chunk - recon(scan, device, backprojector="fbp").reconstruct(7, 10)).max() > 1e-6   # (a smooth phantom)
    for bad in (dict(exposure_arc=1.5), dict(exposure_arc=0.5, legacy_reference=True)):
        with pytest.raises(Exception, match="exposure"):
            recon(scan, device, backprojector="fbp", **bad)


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("backprojector", ["fbp", "ghbp"])
def test_pipeline_ramp_filter(tmp_path, device, backprojector):
    # the reconstructor's ramp is ramp_filter with the configured window; the default is the Ram-Lak ramp
    scan, *_ = helical_scan(tmp_path)
    run = lambda **kw: recon(scan, device, backprojector=backprojector, tomo_params={}, **kw).reconstruct(7, 8, debug=True)
    base = run()
    np.testing.assert_array_equal(base["slices"], run(ramp_filter_name="ramlak")["slices"])
    for name, par in (("hann", "1"), ("butterworth", "0.3:4"), ("shepp", None)):
        d = run(ramp_filter_name=name, ramp_filter_par=par)
        rows = _core.ramp_filter(np.ascontiguousarray(d["sino"][:, 0]), 11.5, device=device, filter_name=name, filter_par=par or "")
        np.testing.assert_allclose(d["filtered"][0], rows * d["wbal"][:, 0], atol=1e-6, err_msg=name)
        assert np.abs(d["slices"] - base["slices"]).max() > 1e-3 * np.abs(base["slices"]).max(), name
    with pytest.raises(Exception, match="unknown ramp filter"):
        run(ramp_filter_name="lanczos")

