"""Parity and smoke tests for the CuPy/NumPy dual-backend in ``tools.py``.

Functions in ``tools.py`` operate through a module-level ``xp``/``xndimage``
pair that resolves to either NumPy/SciPy or CuPy/cupyx at import time. These
tests verify:

1. The NumPy path produces sensible output on its own (always runs).
2. The CuPy path produces output numerically equivalent to the NumPy path on
   the same input (runs only when ``cupy`` is importable).

The CuPy path is exercised by monkeypatching ``tools.xp`` and
``tools.xndimage`` rather than re-importing, so a single test process can
compare both backends back-to-back.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as scipy_ndimage

import tools  # noqa: E402  (sys.path injection happens in conftest)

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage

    _CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _CUPY_AVAILABLE = False

requires_cupy = pytest.mark.skipif(
    not _CUPY_AVAILABLE, reason="cupy is not installed in this environment"
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_volume(rng):
    return rng.random((6, 12, 12)).astype(np.float32)


@pytest.fixture
def force_numpy_backend(monkeypatch):
    """Force tools.xp / tools.xndimage to NumPy/SciPy for the duration of a test."""
    monkeypatch.setattr(tools, "xp", np)
    monkeypatch.setattr(tools, "xndimage", scipy_ndimage)


@pytest.fixture
def force_cupy_backend(monkeypatch):
    """Force tools.xp / tools.xndimage to CuPy. Skips if CuPy is unavailable."""
    if not _CUPY_AVAILABLE:
        pytest.skip("cupy is not installed in this environment")
    monkeypatch.setattr(tools, "xp", cp)
    monkeypatch.setattr(tools, "xndimage", cupy_ndimage)


# ---------------------------------------------------------------------------
# Always-on smoke tests (NumPy path)
# ---------------------------------------------------------------------------


def test_module_imports_cleanly():
    """tools.py imports without raising; xp falls back to NumPy when CuPy is absent."""
    assert tools.xp is not None
    assert tools.xndimage is not None


def test_normalize_numpy_path(force_numpy_backend, small_volume):
    out = tools.normalize(small_volume.copy())
    assert out.shape == small_volume.shape
    assert np.isfinite(out).all()
    assert float(out.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-6)


def test_divide_nonzero_numpy_path():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 0.0, 0.5, 0.0])
    out = tools.divide_nonzero(a, b, eps=1e-10)
    assert out.shape == a.shape
    assert np.isfinite(out).all()
    # Where b != 0, behaves like standard division
    assert out[0] == pytest.approx(0.5)
    assert out[2] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# CuPy parity tests (skipped when CuPy isn't installed)
# ---------------------------------------------------------------------------


@requires_cupy
def test_normalize_parity(small_volume, monkeypatch):
    monkeypatch.setattr(tools, "xp", np)
    monkeypatch.setattr(tools, "xndimage", scipy_ndimage)
    np_out = tools.normalize(small_volume.copy())

    monkeypatch.setattr(tools, "xp", cp)
    monkeypatch.setattr(tools, "xndimage", cupy_ndimage)
    cp_out = cp.asnumpy(tools.normalize(cp.asarray(small_volume.copy())))

    np.testing.assert_allclose(np_out, cp_out, rtol=1e-5, atol=1e-6)


@requires_cupy
def test_divide_nonzero_parity():
    """divide_nonzero branches on isinstance(_, cp.ndarray), so no monkeypatch needed."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 0.0, 0.5, 0.0])
    np_out = tools.divide_nonzero(a, b, eps=1e-10)
    cp_out = cp.asnumpy(tools.divide_nonzero(cp.asarray(a), cp.asarray(b), eps=1e-10))
    np.testing.assert_allclose(np_out, cp_out, rtol=1e-6, atol=1e-9)


@requires_cupy
def test_nms_3d_parity(rng, monkeypatch):
    magnitude = rng.random((6, 12, 12)).astype(np.float32)
    grad = rng.standard_normal((3, 6, 12, 12)).astype(np.float32) * 0.3

    monkeypatch.setattr(tools, "xp", np)
    monkeypatch.setattr(tools, "xndimage", scipy_ndimage)
    np_out = tools.nms_3d(magnitude.copy(), grad.copy(), precision=np.float32)

    monkeypatch.setattr(tools, "xp", cp)
    monkeypatch.setattr(tools, "xndimage", cupy_ndimage)
    cp_out = cp.asnumpy(
        tools.nms_3d(cp.asarray(magnitude.copy()), cp.asarray(grad.copy()), precision=np.float32)
    )

    # NMS makes a binary keep/discard decision per voxel based on >=/> comparisons against
    # interpolated forward/backward magnitudes. scipy.ndimage.map_coordinates and
    # cupyx.scipy.ndimage.map_coordinates produce slightly different interpolated values
    # at near-tie voxels, which can flip the decision for a small fraction of voxels.
    # Assert that the disagreement rate stays below 1% rather than asserting pointwise
    # equality.
    disagreement = np.abs(np_out - cp_out) > 1e-5
    disagreement_rate = float(disagreement.sum()) / float(disagreement.size)
    assert disagreement_rate < 0.01, (
        f"nms_3d outputs disagree at {disagreement_rate * 100:.3f}% of voxels (limit 1%)"
    )


def test_eigvalsh_sym3x3_matches_numpy(rng, force_numpy_backend):
    """The closed-form 3x3 symmetric eigvals helper agrees with numpy.linalg.eigvalsh."""
    matrices_raw = rng.standard_normal((50, 3, 3))
    matrices_sym = 0.5 * (matrices_raw + matrices_raw.transpose(0, 2, 1))

    closed_form = tools._eigvalsh_sym3x3(matrices_sym)
    lapack = np.linalg.eigvalsh(matrices_sym)  # numpy returns ascending order

    np.testing.assert_allclose(closed_form, lapack, rtol=1e-6, atol=1e-7)


def test_eigvalsh_sym3x3_handles_diagonal_and_zero(force_numpy_backend):
    """Diagonal and all-zero matrices are not handled by the closed-form division."""
    matrices = np.stack(
        [
            np.diag([1.0, 2.0, 3.0]),
            np.diag([-5.0, 0.0, 5.0]),
            np.zeros((3, 3)),
            np.eye(3) * 7.0,
        ]
    )
    out = tools._eigvalsh_sym3x3(matrices)
    expected = np.linalg.eigvalsh(matrices)
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)


@requires_cupy
def test_eigvalsh_sym3x3_parity(rng, monkeypatch):
    matrices_raw = rng.standard_normal((50, 3, 3)).astype(np.float64)
    matrices_sym = 0.5 * (matrices_raw + matrices_raw.transpose(0, 2, 1))

    monkeypatch.setattr(tools, "xp", np)
    monkeypatch.setattr(tools, "xndimage", scipy_ndimage)
    np_out = tools._eigvalsh_sym3x3(matrices_sym)

    monkeypatch.setattr(tools, "xp", cp)
    monkeypatch.setattr(tools, "xndimage", cupy_ndimage)
    cp_out = cp.asnumpy(tools._eigvalsh_sym3x3(cp.asarray(matrices_sym)))

    np.testing.assert_allclose(np_out, cp_out, rtol=1e-6, atol=1e-7)


@requires_cupy
def test_hessian_parity(small_volume, monkeypatch):
    monkeypatch.setattr(tools, "xp", np)
    monkeypatch.setattr(tools, "xndimage", scipy_ndimage)
    np_hess, np_zero = tools.hessian(small_volume.copy(), gauss_sigma=1, sigma=2)

    monkeypatch.setattr(tools, "xp", cp)
    monkeypatch.setattr(tools, "xndimage", cupy_ndimage)
    cp_hess_raw, cp_zero_raw = tools.hessian(cp.asarray(small_volume.copy()), gauss_sigma=1, sigma=2)
    cp_hess = cp.asnumpy(cp_hess_raw)
    cp_zero = cp.asnumpy(cp_zero_raw)

    # gaussian_filter implementations differ slightly between SciPy and cupyx,
    # so allow a relaxed tolerance on the Hessian magnitudes.
    np.testing.assert_allclose(np_hess, cp_hess, rtol=1e-3, atol=1e-4)
    np.testing.assert_array_equal(np_zero, cp_zero)
