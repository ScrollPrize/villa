import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import aggregate_l1_edges, solve
from vesuvius.neural_tracing.fiber_trace_3d.fiber_lemon import quantize, native_solve


@pytest.fixture(scope='module')
def library():
    """The solver is built into vc_object_store; skip when that build lacks LEMON."""
    from vc_object_store import winding_min_cost_flow_available
    if not winding_min_cost_flow_available():
        pytest.skip('vc_object_store built without LEMON')
    return None


@pytest.mark.parametrize('backend',['network_simplex','cost_scaling'])
@pytest.mark.parametrize('scale',[1,1000])
def test_integer_objective_and_certificate_match_lp(library,backend,scale):
    rng=np.random.default_rng(123)
    a=rng.integers(0,30,300);b=(a+rng.integers(1,30,300))%30
    g=aggregate_l1_edges(a,b,rng.integers(-3,4,300),rng.uniform(.01,2,300))
    q,_=quantize(g,scale)
    result=native_solve(31,q,backend)  # includes an isolated node
    reference=solve(31,q['a'],q['b'],q['target'],q['capacity'])
    assert result['certificate']['primal']==reference['certificate']['primal']
    assert result['certificate']['gap']==0
    assert result['certificate']['conservation_max']==0


@pytest.mark.parametrize('backend',['network_simplex','cost_scaling'])
def test_tree_recovers_winding_and_negative_lower_bound(library,backend):
    q=dict(a=np.array([0,1],np.int64),b=np.array([1,2],np.int64),
           target=np.array([3,-2],np.int64),capacity=np.array([1,2],np.int64))
    result=native_solve(3,q,backend)
    np.testing.assert_array_equal(result['winding'],[0,3,1])
    assert result['certificate']['primal']==0


def test_quantization_is_explicit_and_does_not_clamp_weak_arcs():
    g=dict(a=np.array([0,0,1]),b=np.array([1,2,2]),target=np.array([0,1,2]),weight=np.array([.49,.5,.51]))
    q,stats=quantize(g,1)
    assert q['capacity'].tolist()==[1,1]
    assert stats['dropped_arcs']==1
    assert stats['dropped_original_weight']==.49
    q,stats=quantize(g,1000)
    assert q['capacity'].tolist()==[490,500,510]
    assert stats['capacity_absolute_error']==0
    with pytest.raises(ValueError,match='Integer arc targets'):
        quantize({**g,'target':np.array([0.,.5,2.])},1)
    with pytest.raises(ValueError,match='matching'):
        quantize({**g,'target':np.array([0,1])},1)


@pytest.mark.parametrize('backend',['network_simplex','cost_scaling'])
def test_all_arcs_quantized_to_zero(library,backend):
    g=dict(a=np.array([0]),b=np.array([1]),target=np.array([1]),weight=np.array([.1]))
    q,_=quantize(g,1)
    result=native_solve(2,q,backend)
    assert result['certificate']['primal']==0
    assert result['certificate']['components']==2


def test_isolated_nodes_and_components_are_reported(library):
    q=dict(a=np.array([0],np.int64),b=np.array([1],np.int64),target=np.array([2],np.int64),capacity=np.array([3],np.int64))
    result=native_solve(4,q,'network_simplex')
    assert result['winding'][1]-result['winding'][0]==2
    assert result['certificate']['components']==3
    assert result['native_timing']['total']>=result['native_timing']['solve']
