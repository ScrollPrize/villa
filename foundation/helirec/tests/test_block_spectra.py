import numpy as np
import pytest

import helirec


def field(n, rng, band=0.5, slope=0.0):
    """periodic random field (n, n, n) with frequencies below `band` cycles per voxel, power ~ |k|^-slope"""
    f = np.fft.fftfreq(n)
    k = np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + np.fft.rfftfreq(n)[None, None, :] ** 2)
    spec = (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) * (k < band)
    spec[k > 0] *= k[k > 0] ** (-slope / 2)
    spec[0, 0, 0] = 0
    v = np.fft.irfftn(spec, s=(n, n, n), axes=(0, 1, 2))
    return v / v.std(), np.fft.rfftn(v / v.std())


def shifted(spec, n, d):
    """the field of `spec` displaced by d (z, y, x) voxels"""
    f = np.fft.fftfreq(n)
    ph = np.exp(-2j * np.pi * (f[:, None, None] * d[0] + f[None, :, None] * d[1] + np.fft.rfftfreq(n)[None, None, :] * d[2]))
    return np.fft.irfftn(spec * ph, s=(n, n, n), axes=(0, 1, 2)).astype(np.float32)


def upsampled(spec, n, f):
    """the same field sampled f times finer (zero-padded transform)"""
    m = n * f
    big = np.zeros((m, m, m // 2 + 1), complex)
    h = n // 2
    for sz in (slice(0, h), slice(-h, None)):
        for sy in (slice(0, h), slice(-h, None)):
            big[sz, sy, :h] = spec[sz, sy, :h]
    return (np.fft.irfftn(big, s=(m, m, m), axes=(0, 1, 2)) * f ** 3).astype(np.float32)


def numpy_tables(c, a, b):
    """the tables of BlockCorrelator.compute (no alignment), in NumPy"""
    n, T = c.n, c.thetas
    w = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(n) + 0.5) / n))
    w3 = w[:, None, None] * w[None, :, None] * w[None, None, :]
    A = np.fft.rfftn((a - (a * w3).sum() / w3.sum()) * w3)
    B = np.fft.rfftn((b - (b * w3).sum() / w3.sum()) * w3)
    k = np.fft.fftfreq(n, 1 / n).round().astype(int)
    kz, ky, kx = np.meshgrid(k, k, np.arange(n // 2 + 1), indexing="ij")
    weight = np.where(kx == 0, 1.0, 2.0) * (kz != -n // 2) * (ky != -n // 2) * (kx != n // 2)
    shell = (np.sqrt(kz ** 2 + ky ** 2 + kx ** 2) * (2 * c.shells / n)).astype(int)
    theta = np.degrees(np.arctan2(np.hypot(ky, kx), np.abs(kz)))
    tb = np.minimum((theta / (90 / T) + 1e-9).astype(int), T - 1)
    tb[0, 0, 0] = T - 1   # (the zero frequency: no direction)
    idx = (shell * T + tb).ravel()
    norm = 1 / (n ** 3 * (w ** 2).sum() ** 3)
    size = c.n_shells * T
    out = [np.bincount(idx, (weight * v).ravel(), size).reshape(c.n_shells, T) * norm
           for v in (np.abs(A) ** 2, np.abs(B) ** 2, (A * np.conj(B)).real)]
    return out + [np.bincount(idx, weight.ravel(), size).reshape(c.n_shells, T)]


@pytest.mark.parametrize("device", helirec.devices())
def test_tables_match_numpy(device):
    rng = np.random.default_rng(1)
    n = 32
    c = helirec.BlockCorrelator(n, shells=8, thetas=6, align=False, device=device)
    a = rng.standard_normal((n, n, n)).astype(np.float32) + 3
    b = (0.5 * a + rng.standard_normal((n, n, n))).astype(np.float32)
    r = c.compute(a, b)
    aa, bb, ab, counts = numpy_tables(c, a.astype(np.float64), b.astype(np.float64))
    np.testing.assert_array_equal(c.counts, counts)
    for got, ref in ((r["aa"], aa), (r["bb"], bb), (r["ab"], ab)):
        np.testing.assert_allclose(got, ref, rtol=2e-4, atol=1e-6 * ref.max())
    assert r["shift"] == (0, 0, 0) and abs(r["mean_a"] - 3) < 0.05


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("oversample", [1, 2])
def test_registration_and_oversampling(device, oversample):
    """b: the same band-limited field displaced by a sub-voxel shift, sampled `oversample` times finer"""
    rng = np.random.default_rng(2)
    n, d = 48, (0.37, -0.62, 0.81)
    v, spec = field(n, rng, band=0.4, slope=2.0)
    a = v.astype(np.float32)
    b = shifted(spec, n, d) if oversample == 1 else upsampled(spec * np.fft.rfftn(shifted(spec, n, d)) / np.where(spec == 0, 1, spec), n, oversample)
    c = helirec.BlockCorrelator(n, oversample=oversample, shells=12, device=device)
    r = c.compute(a, b)
    # (the window, common to both, pulls the shift towards 0 by about (1 / (n align_band))^2 of it)
    np.testing.assert_allclose(r["shift"], d, atol=0.02)
    cur = c.curves(r["aa"], r["bb"], r["ab"])
    inside = (cur["frequency"] > 0.05) & (cur["frequency"] < 0.36)
    assert cur["correlation"][inside].min() > 0.995 and r["peak"] > 0.99
    np.testing.assert_allclose(cur["filter"][inside], 1, atol=0.02)
    # the same shift given: the same tables; none: the correlation is lost at high frequency
    r2 = c.compute(a, b, shift=r["shift"])
    np.testing.assert_allclose(r2["ab"], r["ab"], rtol=1e-6, atol=1e-9)
    r0 = c.compute(a, b, shift=(0, 0, 0))
    assert c.curves(r0["aa"], r0["bb"], r0["ab"])["correlation"][inside][-1] < 0.5


@pytest.mark.parametrize("device", helirec.devices())
def test_correlation_measures_snr_and_direction(device):
    rng = np.random.default_rng(3)
    n = 64
    v, _ = field(n, rng, band=0.6)
    c = helirec.BlockCorrelator(n, shells=8, thetas=3, align=False, device=device)
    tot = None
    for _ in range(4):   # two noisy copies of one signal, noise power = signal power: correlation 1 / 2
        v, _ = field(n, rng, band=1.0)
        a = (v + rng.standard_normal(v.shape)).astype(np.float32)
        b = (v + rng.standard_normal(v.shape)).astype(np.float32)
        r = c.compute(a, b)
        tot = r if tot is None else {k: tot[k] + r[k] if k in ("aa", "bb", "ab") else r[k] for k in r}
    cur = c.curves(tot["aa"], tot["bb"], tot["ab"])
    use = cur["frequency"] < 0.5
    np.testing.assert_allclose(cur["correlation"][use][2:], 0.5, atol=0.03)   # (the first shells: few frequencies)
    np.testing.assert_allclose(cur["snr_split"][use][2:], 1.0, atol=0.12)
    # a filter along z (per-bin gains) changes the plain correlation of a shell where the noise is not isotropic,
    # not the matched one
    f = np.fft.fftfreq(n)
    gain = (1 / (1 + 30 * f ** 2))[:, None, None]
    zn = np.fft.irfft(np.fft.rfft(rng.standard_normal(v.shape), axis=0) * (3 * np.abs(f[: n // 2 + 1]))[:, None, None], n, axis=0)
    a64 = v + zn
    blurred = np.fft.ifftn(np.fft.fftn(a64) * gain).real.astype(np.float32)
    r1, r2 = c.compute(a64.astype(np.float32), v.astype(np.float32)), c.compute(blurred, v.astype(np.float32))
    c1, c2 = (c.curves(r["aa"], r["bb"], r["ab"]) for r in (r1, r2))
    assert (c2["correlation"][use][2:] - c1["correlation"][use][2:]).max() > 0.01
    np.testing.assert_allclose(c2["matched"][use][2:], c1["matched"][use][2:], atol=0.012)
    assert np.all(c1["matched"][use][2:] >= c1["correlation"][use][2:] - 1e-9)
    # a wave along z has its power in the axial bins, one along x in the in-plane bins
    z, x = np.arange(n)[:, None, None], np.arange(n)[None, None, :]
    for wave, lo, hi in ((np.cos(2 * np.pi * 0.2 * z) + 0 * x, 0, 30), (np.cos(2 * np.pi * 0.2 * x) + 0 * z, 60, 90)):
        w = np.broadcast_to(wave, (n, n, n)).astype(np.float32).copy()
        r = c.compute(w, w)
        part = c.curves(r["aa"], r["bb"], r["ab"], lo, hi)
        assert np.nansum(part["power_a"] * part["count"]) > 0.99 * r["aa"].sum()
        assert np.nanargmax(part["power_a"]) == 3   # 0.2 cycles per voxel: shell 3 of width 1/16


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("interp", ["linear", "cubic", "lanczos3"])
def test_resample_affine(device, interp):
    z, y, x = np.meshgrid(np.arange(20.0), np.arange(24.0), np.arange(28.0), indexing="ij")
    fn = lambda z, y, x: np.sin(0.21 * z + 0.4) * np.cos(0.17 * y) + 0.5 * np.sin(0.13 * x + 0.1 * z)
    src = fn(z, y, x).astype(np.float32)
    m = np.array([[0.5, 0.02, 0.0, 5.3], [0.0, 0.45, 0.03, 6.1], [-0.02, 0.0, 0.55, 7.7]])
    shape = (13, 17, 19)
    out = helirec.resample_affine(src, m, shape, interp, device)
    k, j, i = np.meshgrid(*[np.arange(float(s)) for s in shape], indexing="ij")
    p = np.einsum("rc,c...->r...", m, np.stack([k, j, i, np.ones_like(k)]))
    tol = {"linear": 2e-2, "cubic": 2e-3, "lanczos3": 1e-2}[interp]   # (lanczos3: ~0.5 % passband ripple per axis)
    np.testing.assert_allclose(out, fn(*p), atol=tol)
    # whole voxels of a translation are the voxels themselves; outside the volume: 0
    t = np.array([[1.0, 0, 0, 2], [0, 1.0, 0, 3], [0, 0, 1.0, -4]])
    out = helirec.resample_affine(src, t, (4, 5, 8), interp, device)
    np.testing.assert_allclose(out[:, :, 4:], src[2:6, 3:8, :4], atol=1e-6)
    if interp == "linear":
        assert np.all(out[:, :, :3] == 0)
    if device != "cpu":
        np.testing.assert_allclose(helirec.resample_affine(src, m, shape, interp, device),
                                   helirec.resample_affine(src, m, shape, interp, "cpu"), atol=1e-5)


@pytest.mark.parametrize("device", helirec.devices())
def test_registered_volume(tmp_path, device):
    """a pyramid twice finer than the grid, offset, read from two of its levels"""
    from helirec.omezarr import OmeZarrWriter
    z, y, x = np.meshgrid(np.arange(64.0), np.arange(72.0), np.arange(80.0), indexing="ij")
    fn = lambda z, y, x: 120 + 60 * np.sin(0.11 * z + 0.3) * np.cos(0.09 * y) + 40 * np.sin(0.07 * x)
    vol = fn(z, y, x).astype(np.float32)
    w = OmeZarrWriter(str(tmp_path / "ref.zarr"), vol.shape, np.float32, chunks=(16, 16, 16), levels=2)
    w.write(0, vol)
    w.close()
    affine = np.array([[2.0, 0, 0, 3.0], [0, 2.0, 0.01, 4.5], [0, -0.01, 2.0, 6.25]])   # grid (z, y, x, 1) -> level 0
    k, j, i = np.meshgrid(np.arange(8.0), np.arange(9.0), np.arange(10.0), indexing="ij")
    grid = np.stack([5 + k / 2, 6 + j / 2, 7 + i / 2, np.ones_like(k)])
    truth = fn(*np.einsum("rc,c...->r...", affine, grid))
    for level, tol in ((0, 0.05), (1, 1.0)):
        ref = helirec.RegisteredVolume(str(tmp_path / "ref.zarr"), level, affine)
        out, coverage = ref.read((5, 6, 7), 0.5, (8, 9, 10), "lanczos3", device)
        np.testing.assert_allclose(out, truth, atol=tol)
        assert coverage == 1.0
    out, coverage = ref.read((500, 6, 7), 0.5, (4, 4, 4), "cubic", device)
    assert coverage == 0.0 and not out.any()
