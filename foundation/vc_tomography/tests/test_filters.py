"""Filter chains: compilation (fusion, margins, stages), CPU vs CUDA, median vs NumPy, Python filters, errors."""
import numpy as np
import pytest

import vc_tomography
from vc_tomography import filters

from helpers import cuda_only

PH = dict(energy_kev=116.0, distance_m=1.19, pixel_size_m=8.64e-6, det_rows=1200, det_cols=2528)


def test_compile_fusion_and_margins():
    pu = [{"type": "paganin", "delta_beta": 1000}, {"type": "unsharp", "sigma": 1.2, "coeff": 4}]
    assert vc_tomography.filter_describe(pu, **PH) == "fft[paganin*unsharp]@intensity m31"   # supports add up
    assert vc_tomography.filter_margin(pu, **PH) == 20 + 11
    assert vc_tomography.filter_describe(pu, legacy_reference=True, **PH) == "fft[paganin*unsharp]@intensity m20"
    assert vc_tomography.filter_margin(pu, legacy_reference=True, **PH) == 20   # the reference: max of the members
    chain = pu[:1] + [{"type": "median", "radius": 2}] + pu[1:] + [{"type": "gaussian", "sigma": 1, "stage": "log"}]
    d = vc_tomography.filter_describe(chain, **PH)
    assert d == "fft[paganin]@intensity m20 -> [median]@intensity m2 -> fft[unsharp]@intensity m11 -> fft[gaussian]@log m9", d
    assert vc_tomography.filter_margin(chain, **PH) == 20 + 2 + 11 + 9
    assert vc_tomography.filter_margin([], **PH) == 0
    assert "median" in vc_tomography.filter_types() and "python" in vc_tomography.filter_types()


def test_errors():
    with pytest.raises(Exception, match="unknown filter type"):
        vc_tomography.filter_margin([{"type": "nope"}], **PH)
    with pytest.raises(Exception, match="unknown parameter 'sigmaa'"):
        vc_tomography.filter_margin([{"type": "unsharp", "sigmaa": 1, "coeff": 1}], **PH)
    with pytest.raises(Exception, match="not registered"):
        vc_tomography.filter_margin([{"type": "python", "fn": "missing"}], **PH)


def median_ref(a, r):
    n, H, W = a.shape
    yy = np.clip(np.arange(H)[:, None] + np.arange(-r, r + 1)[None], 0, H - 1)
    xx = np.clip(np.arange(W)[:, None] + np.arange(-r, r + 1)[None], 0, W - 1)
    win = a[:, yy[:, None, :, None], xx[None, :, None, :]]          # (n, H, W, 2r+1, 2r+1)
    return np.median(win.reshape(n, H, W, -1), -1)


@pytest.mark.parametrize("r", [1, 2, 3])
def test_median(device, r):
    a = np.random.default_rng(r).standard_normal((3, 17, 23)).astype(np.float32)
    out = vc_tomography.apply_filters(a, [{"type": "median", "radius": r}], 0, device=device)
    np.testing.assert_array_equal(out, median_ref(a, r).astype(np.float32))


@cuda_only
@pytest.mark.parametrize("chain", [
    [{"type": "gaussian", "sigma": 2.0}],
    [{"type": "unsharp", "sigma": 1.2, "coeff": 4}],
    [{"type": "paganin", "delta_beta": 500}, {"type": "unsharp", "sigma": 1.2, "coeff": 4}],
    [{"type": "paganin", "delta_beta": 500}, {"type": "median", "radius": 1}, {"type": "gaussian", "sigma": 1}],
])
def test_cpu_cuda_agree(chain):
    a = np.random.default_rng(0).uniform(0.5, 1.5, (4, 61, 200)).astype(np.float32)
    m = vc_tomography.filter_margin(chain, **PH)
    c = vc_tomography.apply_filters(a, chain, m, device="cpu", **PH)
    g = vc_tomography.apply_filters(a, chain, m, device="cuda", **PH)
    np.testing.assert_allclose(g, c, rtol=0, atol=2e-5)


def test_butterworth_low_pass_of_the_projection():
    """1 / (1 + (f / cutoff)^(2 order)) of the frequency's length, along the rows and the columns alike: a wave keeps
    that share of its amplitude whichever way it runs, a constant stays, and the chain is one Fourier step."""
    H, W = 96, 128
    y, x = np.mgrid[:H, :W]
    chain = [{"type": "butterworth", "cutoff": 0.25, "order": 4}]
    assert vc_tomography.filter_margin(chain, **PH) == 16 and vc_tomography.filter_margin([{"type": "butterworth", "cutoff": 0.05, "order": 8}], **PH) == 64
    np.testing.assert_allclose(vc_tomography.apply_filters(np.full((1, H, W), 3.0, np.float32), chain, 0), 3.0, rtol=1e-5)
    for f0, f1 in ((0.0, 0.125), (0.125, 0.0), (0.0, 0.25), (0.25, 0.0), (0.1875, 0.1875), (0.0, 0.375), (0.375, 0.0)):   # cycles per pixel along the rows, the columns
        wave = (1.0 + 0.5 * np.cos(2 * np.pi * (f0 * y + f1 * x)))[None].astype(np.float32)
        out = vc_tomography.apply_filters(wave, chain, 0)[0]
        expect = 1.0 / (1.0 + (np.hypot(f0, f1) / 0.25) ** 8)
        # (away from the frame's edges: the frame is padded, and the wave does not close on the padded grid)
        np.testing.assert_allclose(out[28:-28, 28:-28], (1.0 + expect * (wave[0] - 1.0))[28:-28, 28:-28], atol=2e-3, err_msg=str((f0, f1)))
    described = vc_tomography.filter_describe([{"type": "paganin", "delta_beta": 120}, {"type": "unsharp", "sigma": 1.2, "coeff": 4}] + chain, **PH)
    assert described.startswith("fft[paganin*unsharp*butterworth]@intensity"), described
    with pytest.raises(ValueError, match="cutoff"):
        vc_tomography.apply_filters(np.ones((1, 8, 8), np.float32), [{"type": "butterworth", "cutoff": 0}], 0)


def test_gaussian_preserves_constant():
    a = np.full((2, 40, 60), 3.0, np.float32)
    out = vc_tomography.apply_filters(a, [{"type": "gaussian", "sigma": 3}], 0)
    np.testing.assert_allclose(out, 3.0, rtol=1e-5)


def test_python_filter(device):
    seen = {}

    def scale(f, k=1.0):
        seen["shape"], seen["k"] = f.shape, k
        f *= k                                              # in place

    def clip(f, lo=0.0):
        return np.maximum(f, lo)                            # returned copy

    filters.register("scale_t", scale)
    filters.register("clip_t", clip)
    a = np.random.default_rng(2).standard_normal((2, 9, 11)).astype(np.float32)
    chain = [{"type": "python", "fn": "scale_t", "k": 2.5}, {"type": "python", "fn": "clip_t", "lo": 0.1}]
    out = vc_tomography.apply_filters(a, chain, 0, device=device)
    np.testing.assert_allclose(out, np.maximum(a * 2.5, 0.1), rtol=1e-6)
    assert seen["shape"] == (2, 9, 11) and seen["k"] == 2.5


def test_overrides_and_text():
    c = filters.parse_chain("paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)+unsharp(sigma=3,coeff=1)")
    assert c[0] == {"type": "paganin", "delta_beta": 1000} and c[2]["sigma"] == 3
    o = filters.override(c, {"paganin.delta_beta": 500, "unsharp[1].coeff": 0})
    assert o[0]["delta_beta"] == 500 and o[2]["coeff"] == 0 and c[0]["delta_beta"] == 1000
    assert filters.parse_chain(filters.chain_text(c)) == c
    s = filters.sweep_sets(["paganin.delta_beta=250,500", "unsharp.coeff=0"])
    assert s == [{"paganin.delta_beta": 250, "unsharp.coeff": 0}, {"paganin.delta_beta": 500, "unsharp.coeff": 0}]
    with pytest.raises(KeyError):
        filters.override(c, {"median.radius": 1})
