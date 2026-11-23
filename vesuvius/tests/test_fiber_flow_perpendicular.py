import numpy as np
import pytest
import json
from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_perpendicular as module
from vesuvius.neural_tracing.fiber_trace_3d.fiber_flow_perpendicular import event_core


def test_repeated_events_not_target_arcs_and_recursive_pruning():
    # Two events 0--1 survive; single-event tail1--2--3 peels recursively.
    a=np.array([0,0,1,2]);b=np.array([1,1,2,3]);w=np.array([.7,1.3,1.,1.])
    keep,initial,degree=event_core(5,a,b,w,2)
    np.testing.assert_array_equal(keep,[True,True,False,False,False])
    np.testing.assert_array_equal(initial,[2,3,2,1,0])
    np.testing.assert_array_equal(degree,[2,2,0,0,0])


def test_fractional_split_single_event_does_not_count_twice():
    keep,_,_=event_core(2,np.array([0,0]),np.array([1,1]),np.array([.2,.8]),2)
    assert not keep.any()


def test_rounded_capacities_are_not_used_for_event_count():
    with pytest.raises(ValueError,match='integral'):
        event_core(2,np.array([0]),np.array([1]),np.array([.8]),2)


def test_length_threshold_is_strict_and_precedes_core(tmp_path,monkeypatch):
    source=tmp_path/'source';geo=source/'geometry';geo.mkdir(parents=True)
    (source/'request.json').write_text(json.dumps({'config':{'perpendicular_relative_weight':1}}))
    np.savez(source/'graph.npz',a=np.array([0,1,2]),b=np.array([1,2,3]),
             target=np.zeros(3,int),weight=np.full(3,2.))
    np.savez(geo/'catalog.npz',source_id=np.array([10,20,30,40]),family=np.array([0,1,0,1]),
             point_offset=np.arange(0,9,2))
    keys=np.zeros(8,dtype=[('arc','f8')]);keys['arc'][1::2]=[999,1000,2000,3000]
    np.save(geo/'keys.npy',keys)
    monkeypatch.setattr(module,'run_full',lambda config: {'mock_solver':True})
    output=tmp_path/'result'
    result=module.run(dict(source_root=str(source),output_root=str(output),minimum_length_base=1000,reference_mean=1))
    assert result['selection']['length_eligible']==2
    with np.load(output/'geometry/catalog.npz') as d:
        np.testing.assert_array_equal(d['source_id'],[30,40])
