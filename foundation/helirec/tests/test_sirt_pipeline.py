"""Measured helical sinograms, separate initialization and normalized SIRT updates."""
import h5py
import numpy as np
import pytest

import helirec
from helirec import _core


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("projector", ["sf", "chord"])
@pytest.mark.parametrize("weighting", ["mask", "confidence"])
def test_sirt_explicit_matrix(device, projector, weighting):
    n, nd, rot = 9, 14, 6.8
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False).astype(np.float32)
    basis = np.eye(n * n, dtype=np.float32).reshape(-1, n, n)
    A = np.stack([_core.chord_project(x, angles, nd, rot, projector=projector).ravel()
                  for x in basis], axis=1).astype(np.float64)
    rng = np.random.default_rng(7)
    b = rng.random((len(angles), nd)).astype(np.float32)
    weights = rng.random(b.shape).astype(np.float32)
    weights[weights < .3] = 0
    q = weights.ravel() if weighting == "confidence" else (weights.ravel() > 0).astype(float)
    rows, cols = A.sum(1), A.T @ q
    x = np.full(n * n, .02)
    for _ in range(3):
        residual = np.divide(q * (b.ravel() - A @ x), rows, out=np.zeros_like(rows), where=rows > 0)
        x += .8 * np.divide(A.T @ residual, cols, out=np.zeros_like(cols), where=cols > 0)
    actual = _core.tomo_reconstruct(b, weights, angles, n, rot, "sirt",
                                   {"num_iter": "3", "init_value": ".02", "relax": ".8",
                                    "projector": projector, "weighting": weighting}, device=device)
    np.testing.assert_allclose(actual.ravel(), x, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("init", ["constant", "gridrec"])
def test_unmeasured_values_never_enter_solver_or_initialization(device, init):
    n, nd = 17, 24
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False).astype(np.float32)
    rows = np.random.default_rng(2).random((len(angles), nd)).astype(np.float32)
    weights = np.ones_like(rows)
    weights[3:6] = 0
    weights[:, :3] = 0
    corrupt = rows.copy()
    corrupt[weights == 0] = np.nan
    run = lambda b: _core.tomo_reconstruct(b, weights, angles, n, 11.5, "sirt",
                                          {"num_iter": "5", "init": init}, device=device)
    # CUDA gridrec already uses atomic Fourier gridding: repeated identical inputs
    # differ at float roundoff. Check that scale before testing excluded values.
    expected = run(rows)
    np.testing.assert_allclose(expected, run(rows), atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(expected, run(corrupt), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("params", [{"relax": v} for v in ("0", "-1", "2", "3", "nan", "inf")] +
                         [{"num_iter": v} for v in ("-1", "1.5", "nan", "inf", "1e30")] +
                         [{"init_value": "nan"}, {"norm": "len2"}])
def test_invalid_sirt_parameters(params):
    with pytest.raises(Exception):
        _core.tomo_reconstruct(np.ones((4, 8), np.float32), None, np.zeros(4, np.float32), 7, 3.5, "sirt", params)


def helical_scan(tmp_path, *, corrupt=False, holes=False, row_pattern=False, gain=1.):
    """Analytic 3D Gaussian with linearly varying z amplitude, fractional z/x/angle positions.

    Raw HDF5 is written in the detector format consumed by the existing RawReader.
    """
    nr, nd, ngrid = 12, 24, 64
    p = np.arange(193)
    angles = p * (360 / ngrid) + (0 if row_pattern else .17 * 360 / ngrid)
    z = p * .06
    if holes:
        keep = ~np.isin(p % ngrid, [13, 14, 15, 16])
        angles, z = angles[keep], z[keep]
    theta = -np.deg2rad(angles)
    horizontal = 0 if row_pattern else .3
    coord = np.arange(nd) + horizontal - 11.5
    offset = 2 * np.cos(theta) - np.sin(theta)
    line = .1 * np.sqrt(2 * np.pi) * 2 * np.exp(-.5 * ((coord[None] - offset[:, None]) / 2) ** 2)
    line = line[:, None, :] * (1 + .03 * (z[:, None, None] + np.arange(nr)[None, :, None]))
    raw = np.rint(50000 * gain * np.exp(-line)).astype(np.uint16)
    weight = np.ones((nr, nd), np.float32)
    if row_pattern:
        raw[:] = np.where(np.arange(nr)[None, :, None] % 2, 5000, 40000)
    else:
        weight[:, 5] = 0
        if corrupt:
            raw[:, :, 5] = 0
    path = tmp_path / f"scan_{corrupt}_{holes}_{row_pattern}_{gain}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=raw, chunks=(1, nr, nd))
    reader = _core.RawReader([str(path)], "data", nr, nd)
    scan = _core.make_scan_data(angles.astype(np.float64), z.astype(np.float64) * .01, .01, nr, nd,
                               0, -horizontal, 0., np.full(len(z), .2), np.array([0], np.float32),
                               np.zeros((nr, nd), np.float32), np.full((1, nr, nd), 50000, np.float32),
                               weight, reader, np.zeros(len(z), np.int32), np.arange(len(z), dtype=np.int32))
    return scan, angles, z, raw


def recon(scan, device, **kw):
    cfg = dict(device=device, pixel_size_m=.01, rot_center=11.5, grid_size=17, n_grid=64,
               backprojector="sirt", tomo_params={"num_iter": "10", "init": "gridrec"},
               luminosity_repartition=0., filters=[], block_angles=16, energy_kev=100., distance_m=.1)
    cfg.update(kw)
    return _core.SliceReconstructor(scan if isinstance(scan, list) else [scan], cfg)


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("filters", [[], [{"type": "paganin", "delta_beta": 1000.}]])
def test_helical_mask_and_chunk_invariance(tmp_path, device, filters):
    scan, *_ = helical_scan(tmp_path, holes=True)
    altered, *_ = helical_scan(tmp_path, holes=True, corrupt=True)
    # Micrometre pixels make this a nontrivial Paganin kernel (six context rows).
    kw = dict(filters=filters, pixel_size_m=1e-5 if filters else .01)
    units = kw["pixel_size_m"] * 100
    r = recon(scan, device, **kw)
    chunk = r.reconstruct(7, 10, debug=True)
    individual = [r.reconstruct(h, h + 1, debug=True) for h in range(7, 10)]
    np.testing.assert_allclose(chunk["slices"] * units, np.concatenate([s["slices"] for s in individual]) * units, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(chunk["sino"], np.concatenate([s["sino"] for s in individual], axis=1), atol=2e-6)
    np.testing.assert_allclose(chunk["slices"] * units, recon(altered, device, **kw).reconstruct(7, 10) * units, atol=1e-7)
    # Gridding cannot bridge these missing bins. Gridrec fills its own copy only.
    assert np.all(chunk["weights"][14:17] == 0)
    np.testing.assert_array_equal(chunk["wbal"], chunk["weights"])
    np.testing.assert_allclose(chunk["filtered"], chunk["sino"].transpose(1, 0, 2), atol=0)
    if device == "cuda":
        cpu = recon(scan, "cpu", **kw).reconstruct(7, 10, debug=True)
        np.testing.assert_allclose(chunk["slices"] * units, cpu["slices"] * units, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("device", helirec.devices())
def test_next_chunk_read_ahead(tmp_path, device):
    # reading the next chunk in the background (reusing the rows it shares with this one) changes nothing
    scan, *_ = helical_scan(tmp_path)
    kw = dict(backprojector="fbp", filters=[{"type": "paganin", "delta_beta": 1000.}], pixel_size_m=1e-5)
    r = recon(scan, device, **kw)
    plain = [r.reconstruct(h, h + 1) for h in (7, 8, 9)]
    r = recon(scan, device, **kw)
    ahead = [r.reconstruct(h, h + 1, next=(h + 1, h + 2) if h < 9 else None) for h in (7, 8, 9)]
    np.testing.assert_array_equal(np.concatenate(ahead), np.concatenate(plain))
    assert np.abs(plain[1]).max() > 0
    del r
    helirec.release_pinned_memory()


@pytest.mark.parametrize("device", helirec.devices())
def test_log_before_fractional_vertical_rebinning(tmp_path, device):
    scan, angles, z, raw = helical_scan(tmp_path, row_pattern=True)
    h = 7
    actual = recon(scan, device, z_interp="linear").reconstruct(h, h + 1, debug=True)   # (kernels: test_resolution)
    expected = []
    for a in range(64):
        values = []
        for i in np.flatnonzero(np.arange(len(z)) % 64 == a):
            pos = h - np.float32(z[i])
            lo = int(np.floor(pos))
            f = pos - lo
            if lo < 0 or lo >= 12 or (f > 0 and lo + 1 >= 12):
                continue
            p0 = -np.log(raw[i, lo, 10] / 50000.)
            p1 = -np.log(raw[i, min(lo + 1, 11), 10] / 50000.)
            values.append((1 - f) * p0 + f * p1)
        expected.append(np.mean(values))
    np.testing.assert_allclose(actual["sino"][:, 0, 10], expected, atol=3e-6)


@pytest.mark.parametrize("device", helirec.devices())
def test_helical_analytic_phantom(tmp_path, device):
    scan, *_ = helical_scan(tmp_path)
    h = 7
    result = recon(scan, device, tomo_params={"num_iter": "60"}).reconstruct(h, h + 1)[0]
    yy, xx = np.mgrid[:17, :17] - 8
    truth = .1 * (1 + .03 * h) * np.exp(-.5 * ((xx - 2) ** 2 + (yy + 1) ** 2) / 4)
    mask = xx ** 2 + yy ** 2 < 6 ** 2
    relative_error = np.linalg.norm((result - truth)[mask]) / np.linalg.norm(truth[mask])
    assert relative_error < .12, relative_error


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("filters", [[], [{"type": "paganin", "delta_beta": 1000.}]])
def test_iterative_luminosity_has_full_context(tmp_path, device, filters):
    scan, *_ = helical_scan(tmp_path)
    companion, *_ = helical_scan(tmp_path, gain=.9)
    r = recon([scan, companion], device, luminosity_repartition=100., filters=filters, pixel_size_m=1e-5)
    assert r.info["margin"] >= 12
    chunk = r.reconstruct(7, 10, debug=True)
    single = [r.reconstruct(h, h + 1, debug=True) for h in range(7, 10)]
    np.testing.assert_allclose(chunk["slices"] * .001, np.concatenate([s["slices"] for s in single]) * .001, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(chunk["sino"], np.concatenate([s["sino"] for s in single], axis=1), atol=2e-6)
