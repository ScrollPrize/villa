import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import solve
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_threads import benchmark, log_rates


@pytest.mark.parametrize('counts', [[1], [2], [4], [16], [1, 2, 4]])
def test_threaded_backend_preserves_original_certificate(tmp_path, counts):
    rng = np.random.default_rng(42)
    a = rng.integers(0, 35, 300)
    b = (a+rng.integers(1, 35, 300)) % 35
    g = dict(a=a, b=b, target=rng.integers(-4, 5, 300), weight=rng.uniform(.01, 2, 300))
    baseline = solve(35, **g)
    result = benchmark(35, g, tmp_path, counts, 20)['result']
    assert result['certificate']['optimal']
    np.testing.assert_allclose(result['certificate']['primal'], baseline['certificate']['primal'], rtol=1e-12)


def test_native_log_throughput_excludes_startup(tmp_path):
    log = tmp_path/'solver.log'
    log.write_text(' 0 0.0 Ph1: 0(0) 40s\n 100 -900 Pr: 10(4); Du: 0(0) 50s\n'
                   ' 200 -800 Pr: 8(3); Du: 0(0) 55s\n 400 -700 Pr: 5(2); Du: 0(0) 65s\n')
    result = log_rates(log)
    assert result['logged_iterations_per_second'] == 20
    np.testing.assert_allclose(result['objective_gain_per_second'], 200/15)
