"""Compact persisted measurements retain only inputs used by graph solvers."""
from types import SimpleNamespace

import numpy as np
import pytest

from .fiber_graph_fixture import measured_store,uid
from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph


@pytest.mark.parametrize('workers,packed',[(1,False),(2,True)])
def test_compact_graph_canonical_endpoints_and_cache(tmp_path,workers,packed):
    curves={i:np.array([[0.,0.,0.],[100.,0.,0.]]) for i in (10,20)}
    path=measured_store(tmp_path,[(99,20,10,False,.27),(3,10,20,True,1.03)],
                        workers=workers,packed=packed,curves=curves,
                        field_overrides={99:dict(arc_a=12.345678,arc_b=2.25)})
    graph=FiberGraph(path)
    np.testing.assert_array_equal(graph.event_ids,[0,1])
    np.testing.assert_array_equal(graph.edge_uuid,[uid(1003),uid(1099)])
    np.testing.assert_array_equal(graph.fiber_a,[uid(10),uid(10)])
    np.testing.assert_array_equal(graph.fiber_b,[uid(20),uid(20)])
    np.testing.assert_array_equal(graph.parallel,[True,False])
    np.testing.assert_array_equal(graph.distance,[1.,.25])
    assert graph.arc_a[1]==2.25 and graph.arc_b[1]==float(np.float32(12.345678))
    assert graph.valid_connector.all() and graph.valid_a.all() and graph.valid_b.all()
    with pytest.raises(ValueError,match='do not store endpoint XYZ'):graph.point_a
    with pytest.raises(ValueError,match='do not store endpoint XYZ'):graph.point_b
    cache=tmp_path/'cache';graph.save_cache(cache)
    saved=FiberGraph.cached(path,cache)
    for field in ('event_ids','edge_uuid','arc_a','arc_b','normal_a','normal_b','connector','distance','parallel'):
        np.testing.assert_array_equal(getattr(saved,field),getattr(graph,field))
    assert not (cache/'point_a.npy').exists() and not (cache/'evidence_weight.npy').exists()


@pytest.mark.parametrize('fiber_sign,expected',[([1,1],-1),([-1,-1],1)])
def test_winding_event_preparation_uses_connector_not_xyz(tmp_path,fiber_sign,expected):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_full import event_batches
    path=measured_store(tmp_path,[(120,10,20,False,.5)])
    graph=FiberGraph(path)
    paths=SimpleNamespace(ids=graph.ids,catalog=dict(family=np.array([0,1]),normal_sign=np.array(fiber_sign)))
    batches=list(event_batches({},paths,graph,0,len(graph.event_ids)))
    assert len(batches)==1
    count,data=batches[0]
    assert count==1
    np.testing.assert_array_equal(data['sign'],[expected])
    np.testing.assert_array_equal(data['distance'],[.5])


def test_distance_limit_skips_instead_of_clamping(tmp_path):
    path=measured_store(tmp_path,[(1,10,20,False,15.875),(2,10,20,True,15.87501)])
    graph=FiberGraph(path)
    np.testing.assert_array_equal(graph.edge_uuid,[uid(1001)])
    np.testing.assert_array_equal(graph.distance,[15.875])


def test_direction_blob_encodes_signed_axis_and_undefined():
    from vc_object_store import encode_fiber_direction,decode_fiber_direction
    for normal in ([0,0,1],[0,0,-1],[1,0,0],[-.3,.4,-.5]):
        blob=encode_fiber_direction(normal)
        assert isinstance(blob,bytes) and len(blob)==3
        decoded=np.asarray(decode_fiber_direction(blob))
        assert decoded@normal>0
        np.testing.assert_allclose(np.linalg.norm(decoded),1.,rtol=0,atol=1e-12)
    assert encode_fiber_direction([0,0,0])==bytes(3)


def test_oriented_normal_validation_reads_compact_fields(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_oriented_normals_validation import validate
    path=measured_store(tmp_path,[(1,10,20,False,.5)])
    result=validate(path,path)
    assert result['integrity']=='ok'
    assert result['exact_constraint_fields']
    assert result['flipped_endpoints']==0
