"""Known answers and independent objectives for the Ceres named-equation interface."""
import numpy as np
import pytest
from helirec import _core
from helirec.solve import Problem, fit_line


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
    p = Problem(); p.unknowns("b", 2)
    e = p.equations([1.,2.]); e.add("b", 0, 1); e.add("b", 1, 2)
    with pytest.raises(ValueError, match="RANK_DEFICIENT"):
        p.solve()
    p.fix("b", 1, 2.)
    assert p.solve()["b"] == pytest.approx([-2.5,2.])


def test_named_heldout_and_cancelled_duplicates():
    p = Problem(); p.unknowns("slope", 1); p.unknowns("offset", 1); p.unknowns("unused", 1)
    t = np.linspace(0,10,200); held = np.arange(200)%4 == 0
    y = .25*t+2
    e = p.equations(y, fit=~held)
    e.add("slope", 0, t).add("offset", 0, 1)
    e.add("unused", 0, 5).add("unused", 0, -5)
    p.fix("offset", 0, 2.)
    s = p.solve()
    assert s["slope"] == pytest.approx([.25], abs=1e-10)
    assert s.rank == 1 and s.diagnostics["unused"] == 1
    assert abs(s.left(e)).max() < 1e-9 and np.all(s.weight[held] == 0)
    # Held-out equations do not establish rank or support.
    p = Problem(); p.unknowns("a", 1); p.equations([1.], fit=False).add("a", 0, 1.)
    with pytest.raises(ValueError, match="INSUFFICIENT_SUPPORT"):
        p.solve()


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
    slope, intercept = fit_line(x, 3.25*(x-1e12)+7)
    assert slope == pytest.approx(3.25, abs=1e-10)
    assert intercept == pytest.approx(7-3.25e12, abs=.002)
    for x,y in [([1],[2]), ([1,1,1],[2,3,4]), ([],[])]:
        assert not _core.calibration_fit_line(x,y)["usable"]
        with pytest.raises(ValueError): fit_line(x,y)


def test_all_fixed_and_constant_equations():
    p = Problem(); p.unknowns("a", 1)
    p.fix("a", 0, 2.)
    p.equations([3., 4.]).add("a", 0, 1.)
    p.equations([5.])
    s = p.solve(huber=1.)
    assert s.rank == 0 and s["a"][0] == 2.
    assert s.residual == pytest.approx([1.,2.,5.])
    assert s.cost == pytest.approx(.5+1.5+4.5)
