"""Scintillator light diffusion correction: frames against the exact point spread, and through the reconstructor."""
import h5py
import numpy as np
import pytest

import helirec
from helirec import _core
from helirec.recon import sld_config


def psf_fft(shape, length, fraction):
    """K = (delta + f t) / (1 + f), t ~ exp(-r / length) / r (r > 0, unit sum), on a periodic grid"""
    r = np.hypot(*np.meshgrid(np.fft.fftfreq(shape[0]) * shape[0], np.fft.fftfreq(shape[1]) * shape[1], indexing="ij"))
    t = np.where(r > 0, np.exp(-r / length) / np.maximum(r, 1e-9), 0.0)
    k = fraction * t / t.sum()
    k[0, 0] += 1.0
    return np.fft.rfft2(k / (1 + fraction))


def blur(frames, length, fraction):
    """frames (..., rows, cols) seen through the scintillator (dark outside the detector)"""
    pad = int(4 * length)
    width = [(0, 0)] * (frames.ndim - 2) + [(pad, pad)] * 2
    p = np.pad(frames, width)
    out = np.fft.irfft2(np.fft.rfft2(p) * psf_fft(p.shape[-2:], length, fraction), p.shape[-2:])
    return out[..., pad:-pad, pad:-pad]


@pytest.mark.parametrize("binning", [4, 8])
def test_frames_against_exact_psf(binning):
    rng = np.random.default_rng(0)
    H, W, f, L = 320, 480, 0.1, 40.0
    yy, xx = np.mgrid[:H, :W]
    true = 20000.0 * np.exp(-0.5 * ((yy - H / 2) / 140.0) ** 2)          # beam profile
    for _ in range(25):                                                   # absorbing discs
        cy, cx, r = rng.uniform(0, H), rng.uniform(60, W - 60), rng.uniform(4, 30)
        true = true * (1 - rng.uniform(0.05, 0.4) * 0.5 * (1 - np.tanh((np.hypot(yy - cy, xx - cx) - r) / 1.5)))
    true[:, :50] *= 0.25                                                  # a dense holder; open beam on the right
    seen = blur(true, L, f)
    rel = lambda a: float(np.sqrt(np.mean((a - true) ** 2)) / np.sqrt(np.mean(true ** 2)))
    out = _core.sld_correct(seen[None].astype(np.float32), f, L, binning)[0] * (1 + f)
    assert rel(seen) > 0.015                                              # the glare: 1.5 % of the signal
    assert rel(out) < 0.06 * rel(seen)
    with pytest.raises(Exception, match="fraction"):
        _core.sld_correct(seen[None].astype(np.float32), 1.5, L, binning)


def test_lit_outside():
    """a beam wider than the detector: the scene goes on beyond the field. outside=lit (the frame continued by its
    edge values) corrects the frame's margins; outside=dark takes the light from beyond for the image's own"""
    H, W, f, L, P = 240, 320, 0.3, 40.0, 160
    yy, xx = np.mgrid[-P:H + P, -P:W + P]
    big = 20000.0 * np.exp(-0.5 * ((yy - H / 2) / 400.0) ** 2) * (1 - 0.3 * (xx > 0.6 * W))   # a shadow that runs off the field
    seen = np.fft.irfft2(np.fft.rfft2(np.pad(big, 4 * int(L))) * psf_fft((big.shape[0] + 8 * int(L), big.shape[1] + 8 * int(L)), L, f),
                         (big.shape[0] + 8 * int(L), big.shape[1] + 8 * int(L)))[4 * int(L):-4 * int(L), 4 * int(L):-4 * int(L)]
    true, seen = big[P:-P, P:-P], seen[P:-P, P:-P].astype(np.float32)
    rel = lambda a: float(np.sqrt(np.mean((a - true) ** 2)) / np.sqrt(np.mean(true ** 2)))
    dark = _core.sld_correct_config(seen[None], sld_config(f"fraction={f},length={L}"))[0] * (1 + f)
    lit = _core.sld_correct_config(seen[None], sld_config(f"fraction={f},length={L},outside=lit"))[0] * (1 + f)
    assert rel(dark) > 0.02 and rel(lit) < 0.25 * rel(dark), (rel(seen), rel(dark), rel(lit))   # (measured: 0.010 as seen, 0.056 dark, 0.0015 lit)
    np.testing.assert_array_equal(dark, _core.sld_correct(seen[None], f, L, 8)[0] * (1 + f))   # the default is dark, as before


def test_spec():
    assert sld_config("fraction=0.1,outside=dark") == sld_config("fraction=0.1")
    assert sld_config("fraction=0.1,outside=lit") == dict(sld_fraction=0.1, sld_length=100.0, sld_bin=8, sld_stride=10, sld_lit_outside=True)
    for bad in ("fraction=0.1,outside=bright", "profile=bm18,fraction=0.1,length=40,outside=lit"):
        with pytest.raises(ValueError):
            sld_config(bad)
    assert sld_config("") == {} and sld_config(None) == {} and sld_config("fraction=0") == {}
    assert sld_config("fraction=0.1, length=80") == dict(sld_fraction=0.1, sld_length=80.0, sld_bin=8, sld_stride=10)
    assert sld_config(dict(fraction=0.2, bin=4, stride=5)) == dict(sld_fraction=0.2, sld_length=100.0, sld_bin=4, sld_stride=5)
    with pytest.raises(ValueError):
        sld_config("length=80")


def scan(tmp_path, name, raw, flat, dark, angles, z, current):
    nr, nd = raw.shape[1:]
    path = tmp_path / f"{name}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=raw, chunks=(1, nr, nd))
    reader = _core.RawReader([str(path)], "data", nr, nd)
    n = len(angles)
    return _core.make_scan_data(angles.astype(np.float64), z.astype(np.float64) * .01, .01, nr, nd, 0, 0., 0.,
                               current.astype(np.float64), np.array([0], np.float32), dark,
                               (flat - dark)[None].astype(np.float32), np.ones((nr, nd), np.float32), reader,
                               np.zeros(n, np.int32), np.arange(n, dtype=np.int32))


@pytest.mark.parametrize("device", helirec.devices())
def test_reconstructor_removes_the_glare(tmp_path, device):
    """A helical scan recorded through the scintillator, corrected, grids to the sinogram of the same scan recorded
    without it (varying beam current, non-uniform flat, glare maps every third projection)."""
    nr, nd, ngrid, f, L = 48, 96, 64, 0.2, 12.0
    p = np.arange(257)
    angles, z = p * (360 / ngrid) + 1.0, p * .12
    current = .2 * (1 + .05 * np.sin(p / 9.0))
    theta = -np.deg2rad(angles)
    coord = np.arange(nd) - (nd - 1) / 2
    line = np.zeros((len(p), nr, nd))
    for amp, sig, ox, oy in ((.5, 6., 14., -8.), (.35, 3., -20., 5.), (.25, 10., 0., 18.)):
        off = ox * np.cos(theta) - oy * np.sin(theta)
        line += (amp * np.exp(-.5 * ((coord[None] - off[:, None]) / sig) ** 2))[:, None, :] * \
            (1 + .01 * (z[:, None, None] + np.arange(nr)[None, :, None]))
    dark = np.full((nr, nd), 100.0, np.float32)
    beam = 40000.0 * np.exp(-.5 * ((np.arange(nr)[:, None] - nr / 2) / 30.0) ** 2) * np.ones((1, nd))
    signal = (current / .2)[:, None, None] * beam[None] * np.exp(-line)
    as_raw = lambda s: np.rint(s + dark[None]).astype(np.uint16)
    clean = scan(tmp_path, "clean", as_raw(signal), beam + dark, dark, angles, z, current)
    seen = scan(tmp_path, "seen", as_raw(blur(signal, L, f)), blur(beam, L, f) + dark, dark, angles, z, current)
    cfg = dict(device=device, pixel_size_m=.01, rot_center=(nd - 1) / 2, grid_size=nd + 1, n_grid=ngrid,
               backprojector="fbp", luminosity_repartition=0., filters=[], block_angles=16, energy_kev=100., distance_m=.1)
    sld = dict(sld_fraction=f, sld_length=L, sld_bin=2, sld_stride=3)
    ref = _core.SliceReconstructor([clean], cfg)
    lo, hi = ref.info["doable"]
    h = (lo + hi) // 2
    sino = lambda r, a=h, b=h + 2: r.reconstruct(a, b, debug=True)["sino"]
    want = sino(ref)
    glare = np.abs(sino(_core.SliceReconstructor([seen], cfg)) - want).max()
    assert glare > 0.1                                                    # line integrals up to ~1.2
    exact = _core.SliceReconstructor([seen], dict(cfg, **dict(sld, sld_bin=1, sld_stride=1)))
    assert np.abs(sino(exact) - want).max() < 0.01 * glare
    # binned maps, one per 3 projections (17 degrees apart here; 10 projections of a real scan are a fraction of one)
    fixed = _core.SliceReconstructor([seen], dict(cfg, **sld))
    got = sino(fixed)
    assert np.abs(got - want).max() < 0.08 * glare
    np.testing.assert_array_equal(sino(fixed), got)                      # cached maps: the same result
    np.testing.assert_allclose(sino(fixed, h + 1, h + 2)[:, :1], got[:, 1:], rtol=0, atol=1e-6)   # chunk independent
    with pytest.raises(Exception, match="legacy reference"):
        _core.SliceReconstructor([seen], dict(cfg, legacy_reference=True, **sld))
