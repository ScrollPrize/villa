import os
from pathlib import Path
import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.fiber_pns import shifted_problem,write_dimacs,build_native,native_solve
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import aggregate_l1_edges,solve
from vesuvius.neural_tracing.fiber_trace_3d.fiber_lemon import quantize


def test_shifted_bounds_supplies_and_cost_constant():
    q=dict(a=np.array([0,1]),b=np.array([1,2]),target=np.array([3,-2]),capacity=np.array([1,2]))
    p=shifted_problem(3,q)
    np.testing.assert_array_equal(p['upper'],[2,4])
    np.testing.assert_array_equal(p['supply'],[1,1,-2])
    x=q['capacity'];y=x-q['capacity']
    assert p['cost']@x+p['objective_offset']==-q['target']@y
    assert p['objective_offset']==-1
    with pytest.raises(ValueError,match='flow-range'):
        shifted_problem(3,{**q,'capacity':np.array([2**30,2])})


@pytest.fixture(scope='module')
def libraries(tmp_path_factory):
    source=os.environ.get('FIBER_PNS_SOURCE')
    if not source or not Path(source).is_file():pytest.skip('Set FIBER_PNS_SOURCE to an explicitly supplied PNS source')
    root=tmp_path_factory.mktemp('pns')
    return {parallel:build_native(source,root/f'flow_{parallel}.so',parallel=parallel,avx2=False) for parallel in (False,True)}


@pytest.mark.parametrize('threads',[1,2,4])
@pytest.mark.parametrize('scale',[1,1000])
def test_pns_matches_lp_with_signed_costs(libraries,tmp_path,threads,scale):
    rng=np.random.default_rng(8)
    a=rng.integers(0,20,150);b=(a+rng.integers(1,20,150))%20
    g=aggregate_l1_edges(a,b,rng.integers(-3,4,150),rng.uniform(.01,1,150))
    q,_=quantize(g,scale);path=tmp_path/'input.min';write_dimacs(path,21,q)
    expected=solve(21,q['a'],q['b'],q['target'],q['capacity'])
    output=native_solve(libraries[threads>1],path,21,q,threads)
    assert not output['timeout']
    assert output['result']['certificate']['primal']==expected['certificate']['primal']
    assert output['result']['certificate']['gap']==0


def test_pns_timeout_does_not_claim_a_solution(libraries,tmp_path):
    rng=np.random.default_rng(9);a=rng.integers(0,100,1000);b=(a+1+rng.integers(0,99,1000))%100
    q,_=quantize(aggregate_l1_edges(a,b,rng.integers(-5,6,1000),np.ones(1000)),1)
    path=tmp_path/'input.min';write_dimacs(path,100,q)
    result=native_solve(libraries[False],path,100,q,1,time_limit=1e-12)
    assert result['timeout'] and 'result' not in result
