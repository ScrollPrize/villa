import shutil
import json
import subprocess
import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import aggregate_l1_edges, solve
from vesuvius.neural_tracing.fiber_trace_3d.fiber_lemon import build_native, quantize, native_solve


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    if not shutil.which('c++') or not shutil.which('pkg-config'):
        pytest.skip('Optional native build tools unavailable')
    if subprocess.run(['pkg-config','--exists','lemon']).returncode:
        pytest.skip('Optional LEMON development dependency unavailable')
    return build_native(tmp_path_factory.mktemp('lemon')/'flow.so')


@pytest.mark.parametrize('backend',['network_simplex','cost_scaling'])
@pytest.mark.parametrize('scale',[1,1000])
def test_integer_objective_and_certificate_match_lp(library,backend,scale):
    rng=np.random.default_rng(123)
    a=rng.integers(0,30,300);b=(a+rng.integers(1,30,300))%30
    g=aggregate_l1_edges(a,b,rng.integers(-3,4,300),rng.uniform(.01,2,300))
    q,_=quantize(g,scale)
    result=native_solve(library,31,q,backend)  # includes an isolated node
    reference=solve(31,q['a'],q['b'],q['target'],q['capacity'])
    assert result['certificate']['primal']==reference['certificate']['primal']
    assert result['certificate']['gap']==0
    assert result['certificate']['conservation_max']==0


@pytest.mark.parametrize('backend',['network_simplex','cost_scaling'])
def test_tree_recovers_winding_and_negative_lower_bound(library,backend):
    q=dict(a=np.array([0,1],np.int64),b=np.array([1,2],np.int64),
           target=np.array([3,-2],np.int64),capacity=np.array([1,2],np.int64))
    result=native_solve(library,3,q,backend)
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
    result=native_solve(library,2,q,backend)
    assert result['certificate']['primal']==0
    assert result['certificate']['components']==2


def test_progress_does_not_change_solution(library,tmp_path):
    instrumented=build_native(tmp_path/'progress.so',progress=True)
    rng=np.random.default_rng(27)
    a=rng.integers(0,30,300);b=(a+rng.integers(1,30,300))%30
    g=aggregate_l1_edges(a,b,rng.integers(-4,5,300),rng.uniform(.01,2,300))
    q,_=quantize(g,1000)
    baseline=native_solve(library,31,q,'cost_scaling')
    path=tmp_path/'progress.json'
    measured=native_solve(instrumented,31,q,'cost_scaling',progress_path=path,
        original_graph=g,reference_mean=.3,progress_interval=0.)
    np.testing.assert_array_equal(measured['winding'],baseline['winding'])
    np.testing.assert_array_equal(measured['flow'],baseline['flow'])
    progress=json.loads(path.read_text())
    assert progress['factor_vs_small']==pytest.approx(progress['normalized_objective']/.3)
    assert progress['objective_original_weights']>=0
    assert progress['optimal'] is False
