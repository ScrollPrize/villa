"""Known answers and independent objectives for the Ceres sparse least-squares solver (_core.calibration_solve)."""
import numpy as np
import pytest
from vc_tomography import _core


def _lstsq(A, y):
    norm = np.linalg.norm(A, axis=0)
    norm[norm == 0] = 1
    return np.linalg.lstsq(A / norm, y, rcond=None)[0] / norm


def _problem(seed=5, m=1200, n=60):
    rng = np.random.default_rng(seed)
    row = np.repeat(np.arange(m), 6)
    col = rng.integers(0, n, len(row))
    scale = 10. ** rng.uniform(-6, 3, n)
    value = rng.standard_normal(len(row)) / scale[col]
    A = np.zeros((m, n))
    np.add.at(A, (row, col), value)  # duplicate coefficients accumulate
    truth = rng.standard_normal(n) * scale
    y = A @ truth + .01 * rng.standard_normal(m)
    y[::13] += 3 * rng.standard_normal(len(y[::13]))
    return row, col, value, A, y, truth


@pytest.mark.parametrize("n", [8, 60, 90])
def test_scaled_known_answers_and_robust_objective(n):
    row, col, value, A, y, truth = _problem(n=n)
    s = _core.calibration_solve(row, col, value, y, n)
    expected = _lstsq(A, y)
    assert s["usable"] and s["rank"] == n and s["solver_version"]
    np.testing.assert_allclose(s["x"], expected, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(s["residual"], y - A @ expected, atol=1e-9)
    assert np.all(s["weight"] == 1)
    threshold = .03
    s = _core.calibration_solve(row, col, value, y, n, huber=threshold, function_tolerance=1e-12)
    # Independent IRLS reference converged to the robust objective, not a fixed
    # number of iterations of the retired backend.
    x = expected
    for _ in range(100):
        r = y - A @ x
        w = np.sqrt(np.minimum(1., threshold / np.maximum(abs(r), 1e-30)))
        new = _lstsq(A * w[:, None], y * w)
        if np.max(abs(A @ (new-x))) < 1e-11:
            break
        x = new
    objective = lambda r: np.where(abs(r) <= threshold, .5*r*r, threshold*(abs(r)-.5*threshold)).sum()
    assert s["usable"] and s["cost"] == pytest.approx(objective(y-A@x), rel=1e-8)
    assert s["cost"] == pytest.approx(objective(s["residual"]), rel=1e-12)
    np.testing.assert_allclose(s["weight"], np.sqrt(np.minimum(1., threshold/np.maximum(abs(s["residual"]), 1e-30))))
    assert np.std(A @ (s["x"]-truth)) < np.std(A @ (expected-truth))
    assert s["spread"] == pytest.approx(.01, rel=.25)


def test_explicit_fixed_parameters_and_rank_failure():
    row, col, value, A, y, _ = _problem(n=12)
    A = np.column_stack([A, np.zeros(len(y)), 2*A[:, 0]])
    r, c = np.nonzero(A)
    args = (r, c, A[r,c], y, A.shape[1])
    failed = _core.calibration_solve(*args)
    assert not failed["usable"] and failed["termination"] == "RANK_DEFICIENT"
    assert failed["rank"] == 12 and failed["active"] == 13 and failed["unused"] == 1
    s = _core.calibration_solve(*args, fixed=[13], fixed_value=[3.])
    assert s["usable"] and s["x"][13] == 3. and s["x"][12] == 0
    np.testing.assert_allclose(s["residual"], y-A[:, :12]@_lstsq(A[:, :12], y), atol=1e-8)
    # two equations, two dependent unknowns: rank deficient until one is fixed
    r, c, v, y = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), np.array([1., 1., 2., 2.]), np.array([1., 2.])
    assert _core.calibration_solve(r, c, v, y, 2)["termination"] == "RANK_DEFICIENT"
    s = _core.calibration_solve(r, c, v, y, 2, fixed=[1], fixed_value=[2.])
    assert s["usable"] and s["x"] == pytest.approx([-2.5, 2.])


def test_cancelled_duplicates_leave_an_unused_unknown():
    t = np.linspace(0, 10, 200)
    rows = np.arange(200)
    row = np.concatenate([rows, rows, rows, rows])
    col = np.repeat([0, 1, 2, 2], 200)
    value = np.concatenate([t, np.ones(200), np.full(200, 5.), np.full(200, -5.)])
    s = _core.calibration_solve(row, col, value, .25*t+2, 3, fixed=[1], fixed_value=[2.])
    assert s["usable"] and s["x"][0] == pytest.approx(.25, abs=1e-10)
    assert s["rank"] == 1 and s["unused"] == 1
    assert abs(np.asarray(s["residual"])).max() < 1e-9


def test_input_errors_and_iteration_limit():
    r,c,v,A,y,_ = _problem(n=8)
    for kwargs in [dict(huber=-1), dict(huber=np.inf), dict(rcond=0), dict(max_iterations=0),
                   dict(fixed=[8], fixed_value=[0.]), dict(fixed=[0,0], fixed_value=[1.,2.]),
                   dict(function_tolerance=np.nan)]:
        with pytest.raises(Exception):
            _core.calibration_solve(r,c,v,y,8,**kwargs)
    for values, measured in [(v*np.nan,y),(v,y*np.inf)]:
        with pytest.raises(Exception): _core.calibration_solve(r,c,values,measured,8)
    with pytest.raises(Exception): _core.calibration_solve(r+len(y),c,v,y,8)
    s = _core.calibration_solve(r,c,v,y,8,huber=.03,max_iterations=1)
    assert s["termination"] == "NO_CONVERGENCE" and s["iterations"] == 1 and s["message"]


def test_large_encoder_line_and_unavailable_fits():
    x = 1e12 + np.arange(1000)
    fit = _core.calibration_fit_line(x.tolist(), (3.25*(x-1e12)+7).tolist())
    assert fit["usable"]
    assert fit["slope"] == pytest.approx(3.25, abs=1e-10)
    assert fit["intercept"] == pytest.approx(7-3.25e12, abs=.002)
    for x,y in [([1],[2]), ([1,1,1],[2,3,4]), ([],[])]:
        assert not _core.calibration_fit_line(x,y)["usable"]


def test_all_fixed_and_constant_equations():
    # equations 0 and 1 hold the fixed unknown; equation 2 holds no unknown at all
    s = _core.calibration_solve(np.array([0, 1]), np.array([0, 0]), np.array([1., 1.]), np.array([3., 4., 5.]), 1,
                                fixed=[0], fixed_value=[2.], huber=1.)
    assert s["rank"] == 0 and s["x"][0] == 2.
    assert s["residual"] == pytest.approx([1., 2., 5.])
    assert s["cost"] == pytest.approx(.5+1.5+4.5)
