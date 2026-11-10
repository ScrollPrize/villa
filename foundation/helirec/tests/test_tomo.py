"""Whole-slice methods (tomo.hpp): projector, gridrec, SIRT, MLEM; exact-backprojector interpolation and
oversampling; backprojector specs."""
import numpy as np
import pytest

import helirec
from helirec import _core
from helirec.recon import backprojector_config

cuda_only = pytest.mark.skipif(not helirec.cuda_available(), reason="no CUDA")
N, ND = 65, 76
ROT = (ND - 1) / 2 + 0.3
ANG = np.linspace(0, 2 * np.pi, 160, endpoint=False).astype(np.float32)
C = (N - 1) / 2
YY, XX = np.mgrid[:N, :N] - C
DISK = (XX ** 2 + YY ** 2 <= 20 ** 2).astype(np.float32)
PH = DISK * 0.3
PH[12:18, 40:47] += 1.0
MASK = XX ** 2 + YY ** 2 < 29 ** 2


def sino_of(img):
    return _core.chord_project(img, ANG, ND, ROT, device="cpu")


def err(x):
    return np.sqrt(np.mean((x[MASK] - PH[MASK]) ** 2)) / np.sqrt(np.mean(PH[MASK] ** 2))


def test_projector_line_integrals():
    s = sino_of(np.ones((N, N), np.float32))
    # a ray through the middle crosses the whole N-pixel square: length N / max(|cos|, |sin|)
    d = int(round(ROT))
    expect = N / np.maximum(np.abs(np.cos(ANG)), np.abs(np.sin(ANG)))
    t = d - ROT
    np.testing.assert_allclose(s[:, d], expect, rtol=0.03 + abs(t) * 0)


@pytest.mark.parametrize("device", helirec.devices())
def test_projector_adjoint(device):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((N, N)).astype(np.float32)
    y = rng.standard_normal((len(ANG), ND)).astype(np.float32)
    ax = _core.chord_project(x, ANG, ND, ROT, device=device)
    aty = _core.chord_backproject(y, ANG, N, ROT, device=device)
    lhs, rhs = float((ax.astype(np.float64) * y).sum()), float((x.astype(np.float64) * aty).sum())
    assert abs(lhs - rhs) < 1e-4 * abs(lhs) + 1e-2


@cuda_only
def test_projector_cpu_cuda():
    x = np.random.default_rng(2).random((N, N)).astype(np.float32)
    np.testing.assert_allclose(_core.chord_project(x, ANG, ND, ROT, device="cuda"), sino_of(x), atol=2e-4)


@pytest.mark.parametrize("device", helirec.devices())
def test_gridrec_position_and_scale(device):
    s = sino_of(PH)
    w = np.full_like(s, 0.5)      # 360 deg: every ray measured twice
    g = _core.tomo_reconstruct(s, w, ANG, N, ROT, "gridrec", {"filter_name": "ramlak"}, device=device)
    assert np.corrcoef(g[MASK], PH[MASK])[0, 1] > 0.99
    assert abs(g[25:40, 25:40].mean() - 0.3) < 0.03              # absolute attenuation (disk 0.3; small grid)
    best = max((np.corrcoef(np.roll(np.roll(g, -dy, 0), -dx, 1)[8:-8, 8:-8].ravel(), PH[8:-8, 8:-8].ravel())[0, 1], dy, dx)
               for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    assert best[1:] == (0, 0)


ITER = [("sirt", {"num_iter": 60}, 0.2), ("mlem", {"num_iter": 40}, 0.12), ("sirt", {"num_iter": 10, "init": "gridrec"}, 0.15),
        ("mlem", {"num_iter": 10, "init": "gridrec"}, 0.12)]


@pytest.mark.parametrize("name,params,tol", ITER)
def test_iterative_converges(name, params, tol):
    s = sino_of(PH)
    w = np.full_like(s, 0.5)
    x = _core.tomo_reconstruct(s, w, ANG, N, ROT, name, {k: str(v) for k, v in params.items()}, device="cpu")
    assert err(x) < tol, err(x)
    if helirec.cuda_available():
        xg = _core.tomo_reconstruct(s, w, ANG, N, ROT, name, {k: str(v) for k, v in params.items()}, device="cuda")
        np.testing.assert_allclose(xg, x, atol=1e-2 * np.abs(x).max())


def test_tomo_errors():
    s = sino_of(PH)
    for gone in ("nope", "art", "osem", "tv", "fbp"):
        with pytest.raises(Exception, match="unknown method"):
            _core.tomo_reconstruct(s, None, ANG, N, ROT, gone)
    with pytest.raises(Exception, match="unknown parameter"):
        _core.tomo_reconstruct(s, None, ANG, N, ROT, "sirt", {"num_iterations": "3"})
    with pytest.raises(Exception, match="unknown parameter"):
        _core.tomo_reconstruct(s, None, ANG, N, ROT, "mlem", {"relax": "0.5"})
    with pytest.raises(Exception, match="unknown parameter"):
        _core.tomo_reconstruct(s, None, ANG, N, ROT, "gridrec", {"num_iter": "3"})


@pytest.mark.parametrize("device", helirec.devices())
@pytest.mark.parametrize("interp", [0, 1, 2])
def test_backproject_interp_constant(device, interp):
    # a constant row backprojects to n_angles inside the detector reach, whatever the kernel
    sino = np.ones((40, 200), np.float32)
    ang = np.linspace(0, np.pi, 40, endpoint=False).astype(np.float32)
    img = helirec.backproject(sino, ang, 64, 64, rot_center=99.5, scale=1.0, device=device, interp=interp)
    np.testing.assert_allclose(img, 40.0, rtol=1e-4)


@pytest.mark.parametrize("device", helirec.devices())
def test_upsample_rows_band_limited(device):
    n, os_ = 128, 4
    x = np.arange(n)
    rows = np.stack([np.cos(2 * np.pi * 5 * x / n + 0.3), np.sin(2 * np.pi * 9 * x / n)]).astype(np.float32)
    up = _core.upsample_rows(rows, os_, device=device)
    xu = np.arange(n * os_) / os_
    exact = np.stack([np.cos(2 * np.pi * 5 * xu / n + 0.3), np.sin(2 * np.pi * 9 * xu / n)])
    mid = slice(os_ * 16, os_ * (n - 16))   # away from the zero-padded ends
    np.testing.assert_allclose(up[:, mid], exact[:, mid], atol=0.05)


def test_backprojector_specs():
    assert backprojector_config("ghbp") == {"backprojector": "ghbp"}
    assert backprojector_config("fbp(interp=cubic,oversample=2)") == {"backprojector": "fbp", "bp_interp": "cubic", "bp_oversample": 2}
    assert backprojector_config("fbp") == {"backprojector": "fbp", "bp_interp": "bspline3", "bp_oversample": 2}
    c = backprojector_config("sirt(num_iter=5,init=gridrec)")
    assert c == {"backprojector": "sirt", "tomo_params": {"num_iter": "5", "init": "gridrec"}}
    assert backprojector_config("mlem")["backprojector"] == "mlem"
    assert backprojector_config("gridrec(filter_name=ramlak)")["tomo_params"] == {"filter_name": "ramlak"}
    with pytest.raises(ValueError):
        backprojector_config("tomopy(algorithm=sirt)")
    with pytest.raises(ValueError):
        backprojector_config("fbp(interpolation=cubic)")
    with pytest.raises(ValueError):
        backprojector_config("mbir3d")
    assert set(_core.tomo_algorithms()) == {"gridrec", "mlem", "sirt"}


def test_disk_domain():
    # pixels outside the reconstruction disk are not unknowns: they stay at 0 in the backprojection
    s = sino_of(PH)
    x = _core.tomo_reconstruct(s, None, ANG, N, ROT, "sirt", {"num_iter": "5"}, device="cpu", R=20)
    out = XX ** 2 + YY ** 2 > 20 ** 2
    assert np.all(x[out] == np.float32(1e-6)) or np.all(np.abs(x[out]) < 1e-5)
    assert np.abs(x[~out]).max() > 0.1


@pytest.mark.parametrize("device", helirec.devices())
def test_bspline_interpolation(device):
    # B-spline coefficients interpolate the samples exactly, and smooth signals between them far better than
    # linear / Keys cubic
    n = 256
    x = np.arange(n)
    f = lambda t: np.cos(2 * np.pi * 7 * t / n + 0.4) + 0.5 * np.sin(2 * np.pi * 23 * t / n)
    rows = f(x)[None].astype(np.float32)
    ang = np.zeros(1, np.float32)
    mid = slice(12, n - 12)
    err = {}
    for name, interp, deg in (("linear", 0, 0), ("cubic", 1, 0), ("bspline3", 3, 3), ("bspline5", 4, 5)):
        r = _core.bspline_prefilter(rows, deg, device=device) if deg else rows
        at = lambda s: helirec.backproject(r, ang, 1, n, rot_center=s, scale=1.0, center=0.0, device=device, interp=interp)[0]
        if deg:
            np.testing.assert_allclose(at(0.0)[mid], rows[0][mid], atol=1e-5)
        err[name] = np.abs(at(0.37)[mid] - f(x + 0.37)[mid]).max()
    assert err["bspline3"] < err["cubic"] / 5 < err["linear"] / 50
    assert err["bspline5"] < err["bspline3"] / 5


@pytest.mark.parametrize("device", helirec.devices())
def test_sf_projector(device):
    # separable footprint: exact adjoint pair, each pixel deposits exactly its area (1) per angle
    rng = np.random.default_rng(3)
    x = rng.standard_normal((N, N)).astype(np.float32)
    y = rng.standard_normal((len(ANG), ND)).astype(np.float32)
    ax = _core.chord_project(x, ANG, ND, ROT, device=device, projector="sf")
    aty = _core.chord_backproject(y, ANG, N, ROT, device=device, projector="sf")
    lhs, rhs = float((ax.astype(np.float64) * y).sum()), float((x.astype(np.float64) * aty).sum())
    assert abs(lhs - rhs) < 1e-4 * abs(lhs)
    # positive (totals do not cancel), inside the detector's reach at every angle
    xp = (rng.random((N, N)) * (XX ** 2 + YY ** 2 < 28 ** 2)).astype(np.float32)
    np.testing.assert_allclose(_core.chord_project(xp, ANG, ND, ROT, device=device, projector="sf").sum(1), xp.sum(), rtol=1e-4)
    ones = _core.chord_backproject(np.ones((len(ANG), ND), np.float32), ANG, N, ROT, device=device, projector="sf")
    np.testing.assert_allclose(ones[10:55, 10:55], len(ANG), rtol=1e-5)


@pytest.mark.parametrize("device", helirec.devices())
def test_sf_cells_off_the_image(device):
    # a detector much wider than the image: cells whose boundaries miss it take no step and are exactly 0
    n, nd, rot = 33, 160, 79.5
    s = _core.chord_project(np.ones((n, n), np.float32), ANG, nd, rot, device=device, projector="sf")
    far = np.abs(np.arange(nd) - rot) > n / np.sqrt(2) + 2
    assert far.sum() > 100 and np.all(s[:, far] == 0)
    np.testing.assert_allclose(s.sum(1), n * n, rtol=1e-5)
    np.testing.assert_allclose(s, _core.chord_project(np.ones((n, n), np.float32), ANG, nd, rot, projector="sf"), atol=1e-4)


@pytest.mark.parametrize("device", helirec.devices())
def test_sirt_noisy(device):
    # row-sum normalisation keeps SIRT stable with the sf projector on noisy data (|a_i|^2 does not)
    s0 = _core.chord_project(PH, ANG, ND, ROT, projector="sf")
    s = s0 + np.random.default_rng(0).normal(0, 0.05 * s0.std(), s0.shape).astype(np.float32)
    run = lambda alg, **p: err(_core.tomo_reconstruct(s, None, ANG, N, ROT, alg, {k: str(v) for k, v in p.items()},
                                                      R=31, device=device))
    assert run("sirt", num_iter=200) < 0.13
    with pytest.raises(Exception, match="norm must be rowsum"):
        run("sirt", num_iter=200, norm="len2")
