import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_irls_benchmark import round_components, metrics, load_solver


def test_component_phase_rounding():
    np.testing.assert_array_equal(round_components(np.array([.2,1.2,2.2,-.3,.7]),
        np.array([0,0,0,1,1])), [0,1,2,0,1])


def test_sign_uses_order_not_measured_magnitude():
    events=[dict(parallel=False,family_delta=1,signed_distance=.1,normal_cosine=1),
            dict(parallel=False,family_delta=1,signed_distance=2.,normal_cosine=1)]
    result=metrics(np.zeros(2),np.zeros(2),events,np.ones(2,bool))
    assert result['signs']['perpendicular']['violations']==0


def test_hash_guard(tmp_path):
    p=tmp_path/'upstream.py';p.write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError,match='changed'):
        load_solver(p,'wrong')
