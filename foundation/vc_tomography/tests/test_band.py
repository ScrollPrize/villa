"""Band reconstruction: CUDA rebalance vs CPU; band backprojection + trilinear sampling vs full-slice
backprojection + trilinear interpolation; CPU vs CUDA band sampler."""
import numpy as np
import pytest

import helirec
from helirec import _core

cuda_only = pytest.mark.skipif(not helirec.cuda_available(), reason="no CUDA")


@cuda_only
def test_rebalance_cuda_matches_cpu():
    rng = np.random.default_rng(3)
    n_proj, rows, cols = 360, 5, 211
    R = rng.standard_normal((n_proj, rows, cols)).astype(np.float32)
    W = rng.uniform(0.1, 1, (n_proj, rows, cols)).astype(np.float32)
    W[:, :, :20] = 0                       # zero weights (left as 0)
    W[10:13] = 0                           # an empty-angle hole
    W[-2:] = 0; W[:1] = 0                  # a hole wrapping through angle 0
    rc = 140.3
    for legacy in (False, True):
        r0, w0 = _core.rebalance(R, W, rc, legacy, device="cpu")
        r1, w1 = _core.rebalance(R, W, rc, legacy, device="cuda")
        assert np.array_equal(r0, r1)
        np.testing.assert_allclose(w1, w0, rtol=2e-6, atol=1e-7)


def trilinear(vol, z, y, x):
    """vol (Z, Y, X); same lower-index rule as the band sampler (min(floor, n - 2))."""
    i = [np.minimum(np.floor(a).astype(int), n - 2) for a, n in zip((z, y, x), vol.shape)]
    f = [a - b for a, b in zip((z, y, x), i)]
    out = 0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (f[0] if dz else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * (f[2] if dx else 1 - f[2])
                out = out + w * vol[i[0] + dz, i[1] + dy, i[2] + dx]
    return out


@pytest.mark.parametrize("device", helirec.devices())
def test_band_matches_full_backprojection(device):
    rng = np.random.default_rng(7)
    N, n_angles, n_det, nz = 96, 180, 140, 6
    h0 = 40                                    # view height of the first slice
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False).astype(np.float32)
    sinos = rng.standard_normal((nz, n_angles, n_det)).astype(np.float32)
    kw = dict(rot_center=70.4, scale=0.01, center=(N - 1) / 2, clip_outer_circle=True, clip_radius=N // 2)
    vol = np.stack([helirec.backproject(s, angles, N, N, device="cpu", **kw) for s in sinos])
    # points along a tilted band, including the last slice / last column edge cases
    n = 4000
    z = rng.uniform(0, nz - 1, n); y = rng.uniform(20, 70, n); x = rng.uniform(5, N - 1, n)
    z[:3] = nz - 1; x[3] = N - 1
    zlo = np.minimum(np.floor(z), nz - 2).astype(np.int32)
    order = np.argsort(zlo, kind="stable")
    zlo, fz = zlo[order], (z - zlo)[order].astype(np.float32)
    y, x, z = y[order].astype(np.float32), x[order].astype(np.float32), z[order]
    b = _core.BandSampler(zlo + h0, fz, y, x, N, device=device)
    for k in range(nz):
        b.feed(h0 + k, sinos[k], angles, **kw)
    v = b.values()
    ref = trilinear(vol.astype(np.float64), zlo + fz.astype(np.float64), y.astype(np.float64), x.astype(np.float64))
    assert np.isfinite(v).all()
    np.testing.assert_allclose(v, ref, rtol=0, atol=2e-5 * np.abs(vol).max())
    assert b.points_sampled == n
    assert b.pixels_backprojected < nz * N * N / 2      # only the band


@cuda_only
def test_band_sampler_cpu_cuda_agree():
    rng = np.random.default_rng(11)
    N, n_angles, n_det = 64, 120, 90
    angles = rng.uniform(0, 6.3, n_angles).astype(np.float32)
    sinos = rng.standard_normal((3, n_angles, n_det)).astype(np.float32)
    zlo = np.sort(rng.integers(0, 2, 500)).astype(np.int32)
    fz, y, x = (rng.uniform(0, 1, 500).astype(np.float32), rng.uniform(0, N - 1, 500).astype(np.float32),
                rng.uniform(0, N - 1, 500).astype(np.float32))
    out = []
    for dev in ("cpu", "cuda"):
        b = _core.BandSampler(zlo, fz, y, x, N, device=dev)
        for k in range(3):
            b.feed(k, sinos[k], angles, rot_center=44.2, scale=0.3)
        out.append(b.values())
    np.testing.assert_allclose(out[1], out[0], rtol=1e-5, atol=1e-5)
