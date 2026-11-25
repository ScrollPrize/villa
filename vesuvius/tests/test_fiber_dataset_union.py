import numpy as np
import json
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_dataset_union import combine_graphs
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers, read_polylines, fiber_volume
from .fiber_graph_fixture import uid, measured_store


def test_union_preserves_global_keys_and_remaps_edges_after_uuid_sort():
    d=dict(source_id=np.array([uid(7),uid(9)]),family=np.array([0,1]),start=np.zeros(2),end=np.array([10.,20.]),
        node_ds=np.array([[0.,10.],[0.,20.]]),links=np.empty((0,4)),
        i=np.array([0]),j=np.array([1]),ds_a=np.array([3.]),ds_b=np.array([4.]),
        distance=np.array([.3]),sign=np.ones(1),parallel=np.zeros(1,bool))
    other=dict(d,source_id=np.array([uid(6),uid(8)]))
    result,groups,original=combine_graphs([d,other])
    np.testing.assert_array_equal(result['source_id'],list(map(uid,[6,7,8,9])))
    np.testing.assert_array_equal(original,result['source_id'])
    np.testing.assert_array_equal(groups,[1,0,1,0])
    np.testing.assert_array_equal(result['i'],[1,0]);np.testing.assert_array_equal(result['j'],[3,2])
    np.testing.assert_array_equal(result['distance'],[.3,.3])
    np.testing.assert_array_equal(result['ds_a'],[3,3])
    assert result['parallel'].dtype==bool
    with pytest.raises(ValueError,match='Duplicate object UUID'):
        combine_graphs([d,d])


def test_edge_store_group_filter_skips_within_source_before_sign_evaluation(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_min_flow_selection as selection
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
    database=measured_store(tmp_path,[(1,1,2,False,.5)])
    def forbidden(*args):raise AssertionError('Within-source events must not be evaluated')
    monkeypatch.setattr(uv,'signed_order',forbidden)
    a,b=uid(1),uid(2)
    store=selection.EdgeStore(dict(database=str(database)),[a,b],{a:0,b:1},{a:1,b:1},{a:0,b:0})
    store.load([a,b])
    assert store.edges().shape==(0,7)


def test_prepare_writes_valid_metadata_and_remaps_gt(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_dataset_union as union
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_normal_convention as normal
    d=dict(family=np.array([0]),start=np.zeros(1),end=np.array([10.]),
        node_ds=np.array([[0.,10.]]),links=np.empty((0,4)),i=np.array([],int),j=np.array([],int))
    for k in ('ds_a','ds_b','distance','sign'):d[k]=np.array([],float)
    d['parallel']=np.array([],bool)
    monkeypatch.setattr(normal,'inward_solution',lambda x:x)
    sources=[]
    for fid,sign in [(9,1.),(7,-1.)]:
        source=tmp_path/f'source_{fid}.vcobj'
        row=dict(id=uid(fid),name='fiber',family='H')
        write_initial_fibers(source,[(row,np.array([[0.,0,0],[10,0,0]]))],volume=uid(999),workers=1,packed=True)
        tensor=tmp_path/f'tensor_{fid}.npz';np.savez(tensor,source_id=[uid(fid)],**d)
        solution=tmp_path/f'solution_{fid}.npz';np.savez(solution,source_id=[uid(fid)],component=[0],winding=[123])
        signs=tmp_path/f'normal_{fid}.npz';np.savez(signs,ids=[uid(fid)],normal_sign=[sign])
        events=tmp_path/f'events_{fid}.json'
        events.write_text(json.dumps([dict(test_source=uid(fid),test_piece=0,test_arc=3,gt_source=uid(99))]))
        sources.append(dict(name=str(fid),fibers=str(source),tensor_cache=str(tensor),solution=str(solution),
                            normal_solution=str(signs),events=str(events)))
    root=tmp_path/'out'
    union.prepare(dict(output_root=str(root),sources=sources))
    assert fiber_volume(root/'selected.vcobj')==uid(999)
    assert [row['id'] for row,_ in read_polylines(root/'selected.vcobj')]==[uid(7),uid(9)]
    e=json.loads((root/'cross/cross_events.json').read_text())['events']
    assert [r['test_source'] for r in e]==[uid(9),uid(7)]
    assert [r['test_piece'] for r in e]==[1,0]
    assert [r['gt_source'] for r in e]==[uid(99),uid(99)]
    mapping=np.load(root/'source_mapping.npz')
    np.testing.assert_array_equal(mapping['normal_sign'],[-1.,1.])
    assert 'winding' not in np.load(root/'within_tensors.npz').files
