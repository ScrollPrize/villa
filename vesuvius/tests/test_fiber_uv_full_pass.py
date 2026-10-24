import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import pass_partition
from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import winding_error_metrics


@pytest.mark.parametrize('count,updates',[(19,4),(489663,1613),(6606100,1613)])
def test_full_pass_coverage(count,updates):
    rng=np.random.default_rng(17)
    order=rng.permutation(count)
    parts=[order[pass_partition(count,updates,i)] for i in range(updates)]
    np.testing.assert_array_equal(np.concatenate(parts),order)
    assert max(map(len,parts))-min(map(len,parts))<=1
    assert sum(len(p)*updates/count for p in parts)==pytest.approx(updates)


def test_global_offset_and_outlier():
    result=winding_error_metrics([1,2,3,10],[11,12,13,14])
    assert result['global_offset']==10
    assert result['mean_absolute_winding_error']==1.5
    assert result['within_two_windings_percent']==75
    np.testing.assert_allclose(result['fitted_winding_q10_q90'],[11.3,17.9])
    np.testing.assert_allclose(result['reference_winding_q10_q90'],[11.3,13.7])
    assert winding_error_metrics([101,102,103,110],[11,12,13,14])['mean_absolute_winding_error']==1.5


def test_inclusive_two_and_missing():
    result=winding_error_metrics([0,0,0,np.nan],[0,2,-2,3])
    assert result['matched_fibers']==3
    assert result['within_two_windings_percent']==100
    assert winding_error_metrics([np.nan],[1])['mean_absolute_winding_error'] is None


def test_visualization_offset_override_preserves_default():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import visualization_offset
    a=np.array([5.,7.]);b=np.array([1.,2.]);w=np.array([1.,2.])
    assert visualization_offset({},a,b,w)==5
    assert visualization_offset({'global_winding_offset':11},a,b,w)==11
    with pytest.raises(ValueError):visualization_offset({'global_winding_offset':float('nan')},a,b,w)


def test_source_aggregation_uses_curves_and_excludes_rejection(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_reference_comparison as ref
    d=dict(source_id=np.array([1,1,2,3]),start=np.zeros(4),lengths=np.ones(4),
           node_ds=np.tile([0,.5,1],(4,1)),family=np.zeros(4,dtype=int))
    monkeypatch.setattr(ref.uv,'hard_geometry',lambda *args:
        (None,None,np.array([True,True,True,False]),None,np.array([0,0,10,20]),None,None))
    projection=dict(owner=np.array([0,1,2,3]),arc=np.array([1,1,1,1]),
        weight=np.ones(4),offset=np.arange(5),chosen_winding=np.array([11,11,21,31]))
    result=ref.reference_winding_metrics(d,None,np.ones(40),{},projection)
    assert result['total_source_fibers']==3
    assert result['retained_source_fibers']==2
    assert result['matched_fibers']==2
    assert result['global_offset']==10  # actual curve U=1/11, not anchor U=0/10
    assert result['mean_absolute_winding_error']==0
