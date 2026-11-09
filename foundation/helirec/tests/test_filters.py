"""Filter chains: compilation (fusion, margins, stages), CPU vs CUDA, median vs NumPy, Python filters, errors."""
import numpy as np
import pytest

import helirec
from helirec import filters

PH = dict(energy_kev=116.0, distance_m=1.19, pixel_size_m=8.64e-6, det_rows=1200, det_cols=2528)
cuda_only = pytest.mark.skipif(not helirec.cuda_available(), reason="no CUDA")


def test_compile_fusion_and_margins():
    pu = [{"type": "paganin", "delta_beta": 1000}, {"type": "unsharp", "sigma": 1.2, "coeff": 4}]
    assert helirec.filter_describe(pu, **PH) == "fft[paganin*unsharp]@intensity m31"   # supports add up
    assert helirec.filter_margin(pu, **PH) == 20 + 11
    assert helirec.filter_describe(pu, legacy_reference=True, **PH) == "fft[paganin*unsharp]@intensity m20"
    assert helirec.filter_margin(pu, legacy_reference=True, **PH) == 20   # the reference: max of the members
    chain = pu[:1] + [{"type": "median", "radius": 2}] + pu[1:] + [{"type": "gaussian", "sigma": 1, "stage": "log"}]
    d = helirec.filter_describe(chain, **PH)
    assert d == "fft[paganin]@intensity m20 -> [median]@intensity m2 -> fft[unsharp]@intensity m11 -> fft[gaussian]@log m9", d
    assert helirec.filter_margin(chain, **PH) == 20 + 2 + 11 + 9
    assert helirec.filter_margin([], **PH) == 0
    assert "median" in helirec.filter_types() and "python" in helirec.filter_types()


def test_errors():
    with pytest.raises(Exception, match="unknown filter type"):
        helirec.filter_margin([{"type": "nope"}], **PH)
    with pytest.raises(Exception, match="unknown parameter 'sigmaa'"):
        helirec.filter_margin([{"type": "unsharp", "sigmaa": 1, "coeff": 1}], **PH)
    with pytest.raises(Exception, match="not registered"):
        helirec.filter_margin([{"type": "python", "fn": "missing"}], **PH)


def median_ref(a, r):
    n, H, W = a.shape
    yy = np.clip(np.arange(H)[:, None] + np.arange(-r, r + 1)[None], 0, H - 1)
    xx = np.clip(np.arange(W)[:, None] + np.arange(-r, r + 1)[None], 0, W - 1)
    win = a[:, yy[:, None, :, None], xx[None, :, None, :]]          # (n, H, W, 2r+1, 2r+1)
    return np.median(win.reshape(n, H, W, -1), -1)


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("r", [1, 2, 3])
def test_median(device, r):
    a = np.random.default_rng(r).standard_normal((3, 17, 23)).astype(np.float32)
    out = helirec.apply_filters(a, [{"type": "median", "radius": r}], 0, device=device)
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
    m = helirec.filter_margin(chain, **PH)
    c = helirec.apply_filters(a, chain, m, device="cpu", **PH)
    g = helirec.apply_filters(a, chain, m, device="cuda", **PH)
    np.testing.assert_allclose(g, c, rtol=0, atol=2e-5)


def test_gaussian_preserves_constant():
    a = np.full((2, 40, 60), 3.0, np.float32)
    out = helirec.apply_filters(a, [{"type": "gaussian", "sigma": 3}], 0)
    np.testing.assert_allclose(out, 3.0, rtol=1e-5)


@pytest.mark.parametrize("device", helirec.devices())
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
    out = helirec.apply_filters(a, chain, 0, device=device)
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
