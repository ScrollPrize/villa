"""helirec's solver of the geometry steps (helirec/calibration/solve.hpp, helirec.solve): a linear model fitted
under a Huber loss, against the same arithmetic done by numpy."""
import numpy as np
import pytest

from helirec import _core
from helirec.solve import Problem


def _lstsq(A, y):
    """numpy's least squares with the columns scaled to unit length first: as they are, unknowns nine decades apart
    cost it half its digits (which is why the solver scales them)"""
    norm = np.sqrt((A ** 2).sum(0))
    norm[norm == 0] = 1.0
    return np.linalg.lstsq(A / norm, y, rcond=None)[0] / norm


def _reference(A, y, k, iterations):
    """weighted least squares repeated with the Huber weights of the solve before (numpy)"""
    w = np.ones(len(y))
    for _ in range(iterations):
        x = _lstsq(A * w[:, None], y * w)
        res = y - A @ x
        w = np.sqrt(np.minimum(1.0, k / np.maximum(np.abs(res), 1e-12)))
    return x, res


def _problem(rng, m=3000, n=60, per_row=6):
    row = np.repeat(np.arange(m), per_row)
    col = rng.integers(0, n, m * per_row)
    scale = 10.0 ** rng.uniform(-6, 3, n)                     # unknowns of very different sizes, as a calibration's are
    value = rng.standard_normal(m * per_row) / scale[col]
    A = np.zeros((m, n))
    np.add.at(A, (row, col), value)                           # repeated entries add
    x = rng.standard_normal(n) * scale
    y = A @ x + 0.01 * rng.standard_normal(m)
    y[::13] += 3 * rng.standard_normal(len(y[::13]))          # outliers
    return row, col, value, A, y, x


def test_least_squares_and_huber_equal_numpy():
    rng = np.random.default_rng(5)
    row, col, value, A, y, truth = _problem(rng)
    s = _core.calibration_solve(row, col, value, y, A.shape[1])
    x0 = _lstsq(A, y)
    assert s["rank"] == A.shape[1] and np.allclose(s["x"], x0, rtol=1e-8, atol=0)
    assert np.allclose(s["residual"], y - A @ x0, atol=1e-9) and np.all(s["weight"] == 1)
    # the Huber loss: the outliers no longer pull the answer
    s = _core.calibration_solve(row, col, value, y, A.shape[1], huber=0.03, iterations=8)
    x, res = _reference(A, y, 0.03, 8)
    assert np.allclose(s["x"], x, rtol=1e-7, atol=0) and np.allclose(s["residual"], res, atol=1e-8)
    assert np.abs(s["x"] / truth - 1).max() < np.abs(x0 / truth - 1).max()
    assert s["weight"][::13].mean() < 0.5 < s["weight"].mean()
    # the threshold as a multiple of the residuals' own spread: the spread is the noise put in
    s = _core.calibration_solve(row, col, value, y, A.shape[1], huber=2.0, relative=True, iterations=8)
    assert s["spread"] == pytest.approx(0.01, rel=0.2) and np.abs(s["x"] / truth - 1).max() < np.abs(x0 / truth - 1).max()


def test_what_the_equations_do_not_determine_is_zero():
    rng = np.random.default_rng(6)
    row, col, value, A, y, _ = _problem(rng, m=400, n=12)
    # an unknown no equation holds, and one that is another's double: both are left out of the rank
    A2 = np.concatenate([A, np.zeros((len(y), 1)), 2 * A[:, :1]], 1)
    r, c = np.nonzero(A2)
    s = _core.calibration_solve(r, c, A2[r, c], y, A2.shape[1])
    assert s["rank"] == 12 and s["x"][12] == 0 and (s["x"][0] == 0 or s["x"][13] == 0)
    assert np.allclose(s["residual"], y - A @ _lstsq(A, y), atol=1e-9)       # the fit is the same
    with pytest.raises(Exception):
        _core.calibration_solve(np.array([0, 400]), np.array([0, 0]), np.ones(2), y, 12)       # an entry outside the equations
    with pytest.raises(Exception):
        _core.calibration_solve(r, c, A2[r, c], y, 3)                                           # ... or outside the unknowns


def test_problem_by_name():
    """A model stated as equations over named unknowns: a line through points, some held out and predicted."""
    rng = np.random.default_rng(8)
    t = np.linspace(0, 10, 200)
    which = rng.integers(0, 3, 200)                           # three lines of one slope, each with its own offset
    y = 0.25 * t + np.array([1.0, -2.0, 5.0])[which] + 0.01 * rng.standard_normal(200)
    y[5] += 7.0
    held = np.arange(200) % 4 == 0
    p = Problem()
    p.unknowns("slope", 1)
    p.unknowns("offset", 3)
    e = p.equations(y, fit=~held)
    e.add("slope", 0, t)
    e.add("offset", which, 1.0)
    pin = p.equations([0.0])                                  # an equation of the model's own: nothing measured
    pin.add("offset", 0, 1e-6)
    s = p.solve(huber=0.05)
    assert s["slope"][0] == pytest.approx(0.25, abs=2e-3) and np.allclose(s["offset"], [1.0, -2.0, 5.0], atol=0.02)
    left = s.left(e)
    assert left.shape == (200,) and np.sqrt(np.mean(left[held] ** 2)) < 0.02 and abs(left[5] - 7.0) < 0.05
    assert s.rank == 4 and np.all(s.weight[e.rows][held] == 0) and s.weight[e.rows][5] < 0.2
    with pytest.raises(ValueError):
        e.add("offset", 3, 1.0)
    with pytest.raises(ValueError):
        p.unknowns("slope", 2)
