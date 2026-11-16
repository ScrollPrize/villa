"""Eikonal phase retrieval filter (epr): Paganin limit, accuracy against Fresnel propagation, backends, chunks."""
import numpy as np
import pytest

import helirec

PH = dict(energy_kev=116.0, distance_m=1.19, pixel_size_m=8.64e-6, det_rows=1200, det_cols=2528)
DB = 1000.0
PAGANIN = [{"type": "paganin", "delta_beta": DB}]
cuda_only = pytest.mark.skipif(not helirec.cuda_available(), reason="no CUDA")


def epr(**kw):
    return [{"type": "epr", "delta_beta": DB, **kw}]


def propagated(strength, n=128, os=16, seed=0):
    """Detector image of a homogeneous object (I0 = exp(-tau), phase = -(delta/beta) tau / 2) after exact Fresnel
    propagation on a grid `os` times finer than the detector, and I0, both box-binned to (n, n) detector pixels."""
    rng = np.random.default_rng(seed)
    N = n * os
    y, x = np.meshgrid(np.arange(N) / os, np.arange(N) / os, indexing="ij")
    tau = np.zeros((N, N))
    for _ in range(12):   # discs with soft edges (0.8 .. 2 pixels)
        cy, cx = rng.uniform(0.25 * n, 0.75 * n, 2)
        r, w = rng.uniform(3, 8), rng.uniform(0.8, 2.0)
        tau += rng.uniform(0.3, 1.0) * 0.5 * (1 - np.tanh((np.hypot(y - cy, x - cx) - r) / w))
    tau *= strength
    lam = 1.23984199e-9 / PH["energy_kev"]
    f = np.fft.fftfreq(N, PH["pixel_size_m"] / os)
    psi = np.exp(-0.5 * tau - 0.5j * DB * tau)
    psi = np.fft.ifft2(np.fft.fft2(psi) * np.exp(-1j * np.pi * lam * PH["distance_m"] * (f[:, None] ** 2 + f[None] ** 2)))
    binned = lambda a: a.reshape(n, os, n, os).mean((1, 3)).astype(np.float32)
    return binned(np.abs(psi) ** 2), binned(np.exp(-tau))


def test_compile_margin_and_errors():
    assert helirec.filter_margin(PAGANIN, **PH) == 20
    assert helirec.filter_describe(epr(), **PH) == "[epr]@intensity m44"            # 2 x Paganin's + 4
    assert helirec.filter_describe(epr(iters=0), **PH) == "[epr]@intensity m20"
    assert helirec.filter_describe(epr() + [{"type": "unsharp", "sigma": 1.2, "coeff": 4}], **PH) == \
        "[epr]@intensity m44 -> fft[unsharp]@intensity m11"
    with pytest.raises(Exception, match="needs the energy"):
        helirec.filter_margin(epr(), det_rows=1200, det_cols=2528)
    with pytest.raises(Exception, match="intensity stage"):
        helirec.filter_margin(epr(stage="log"), **PH)
    with pytest.raises(Exception, match="relax"):
        helirec.filter_margin(epr(relax=1.5), **PH)


def test_no_iterations_is_paganin():
    det, _ = propagated(0.1)
    np.testing.assert_array_equal(helirec.apply_filters(det[None], epr(iters=0), 20, **PH),
                                  helirec.apply_filters(det[None], PAGANIN, 20, **PH))


@pytest.mark.parametrize("device", helirec.devices())
def test_closer_to_the_object_than_paganin(device):
    """Ray shifts of a fraction of a pixel (Paganin's estimate: up to ~0.8): the second-order model removes about
    half of Paganin's error; with shifts of a few hundredths of a pixel it changes next to nothing."""
    inner = (0, slice(24, -24), slice(24, -24))
    rms = lambda a, b: float(np.sqrt(np.mean((a[inner] - b[inner[1:]]) ** 2)))
    det, obj = propagated(0.1)
    pag = helirec.apply_filters(det[None], PAGANIN, 20, **PH)
    one = helirec.apply_filters(det[None], epr(iters=1), 44, device=device, **PH)
    out = helirec.apply_filters(det[None], epr(), 44, device=device, **PH)
    many = helirec.apply_filters(det[None], epr(iters=12), 44, device=device, **PH)
    assert rms(out, obj) < 0.6 * rms(pag, obj)
    assert rms(out, obj) < rms(one, obj) < rms(pag, obj)
    np.testing.assert_allclose(many[inner], out[inner], atol=4e-4)       # converged (correction: ~1e-2), stable
    det, obj = propagated(0.005)
    pag = helirec.apply_filters(det[None], PAGANIN, 20, **PH)
    out = helirec.apply_filters(det[None], epr(), 44, device=device, **PH)
    np.testing.assert_allclose(out[inner], pag[inner], atol=6e-5)        # object contrast: 5e-3
    assert rms(out, obj) < 1.02 * rms(pag, obj)


@cuda_only
def test_cpu_cuda_agree():
    det, _ = propagated(0.1)
    noise = np.random.default_rng(0).uniform(0.5, 1.5, (3, 120, 200)).astype(np.float32)   # shift limit active
    for a in (det[None], noise):
        c = helirec.apply_filters(a, epr(), 44, device="cpu", **PH)
        g = helirec.apply_filters(a, epr(), 44, device="cuda", **PH)
        assert np.isfinite(c).all()
        np.testing.assert_allclose(g, c, rtol=0, atol=2e-5)


def test_chunk_independent():
    """Rows at least the margin away from a slab's edges do not depend on the slab."""
    det, _ = propagated(0.1)
    full = helirec.apply_filters(det[None], epr(), 44, **PH)
    slab = helirec.apply_filters(det[None, 16:112], epr(), 44, **PH)
    np.testing.assert_allclose(slab[0, 44:-44], full[0, 60:68], rtol=0, atol=5e-6)
