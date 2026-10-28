import numpy as np
import json
import sqlite3
from vesuvius.neural_tracing.fiber_trace_3d.fiber_dataset_union import combine_graphs


def test_union_keeps_source_collisions_separate_and_preserves_edges():
    d=dict(source_id=np.array([7,9]),family=np.array([0,1]),start=np.zeros(2),end=np.array([10.,20.]),
        node_ds=np.array([[0.,10.],[0.,20.]]),links=np.empty((0,4)),
        i=np.array([0]),j=np.array([1]),ds_a=np.array([3.]),ds_b=np.array([4.]),
        distance=np.array([.3]),sign=np.ones(1),parallel=np.zeros(1,bool))
    result,groups,original=combine_graphs([d,d])
    np.testing.assert_array_equal(result['source_id'],[0,1,2,3])
    np.testing.assert_array_equal(original,[7,9,7,9])
    np.testing.assert_array_equal(groups,[0,0,1,1])
    np.testing.assert_array_equal(result['i'],[0,2]);np.testing.assert_array_equal(result['j'],[1,3])
    np.testing.assert_array_equal(result['distance'],[.3,.3])
    np.testing.assert_array_equal(result['ds_a'],[3,3])
    assert result['parallel'].dtype==bool


def test_edge_store_group_filter_skips_within_source_before_parsing(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_min_flow_selection as selection
    class DB:
        def execute(self,*args):
            # Deliberately invalid normal strings: rejected within-source rows
            # must never reach normal decoding or sign evaluation.
            return [(1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,'bad','bad')]
    monkeypatch.setattr(selection,'ro',lambda _:DB())
    store=selection.EdgeStore(dict(database='unused'),[0,1],{0:0,1:1},{0:1,1:1},{0:0,1:0})
    store.load([0,1])
    assert store.edges().shape==(0,7)


def test_prepare_writes_valid_metadata_and_remaps_gt(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_dataset_union as union
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_normal_convention as normal
    from vesuvius.neural_tracing.tracers.shared.annotations.afv import write_afv
    source=tmp_path/'source.afv'
    row=dict(id=7,name='fiber',family='H',annotation='{}')
    write_afv(None,source,[(row,np.array([[0.,0,0],[10,0,0]]))],dict(frame={},root={}))
    d=dict(source_id=np.array([7]),family=np.array([0]),start=np.zeros(1),end=np.array([10.]),
        node_ds=np.array([[0.,10.]]),links=np.empty((0,4)),i=np.array([],int),j=np.array([],int))
    for k in ('ds_a','ds_b','distance','sign'):d[k]=np.array([],float)
    d['parallel']=np.array([],bool)
    tensor=tmp_path/'tensor.npz';np.savez(tensor,**d)
    solution=tmp_path/'solution.npz';np.savez(solution,source_id=[7],component=[0],winding=[123])
    signs=tmp_path/'normal.npz';np.savez(signs,ids=[7],normal_sign=[1.])
    monkeypatch.setattr(normal,'inward_solution',lambda x:x)
    events=tmp_path/'events.json';events.write_text(json.dumps([dict(test_source=7,test_piece=0,test_arc=3,gt_source=99)]))
    spec=dict(afv=str(source),tensor_cache=str(tensor),solution=str(solution),
        normal_solution=str(signs),events=str(events))
    root=tmp_path/'out'
    union.prepare(dict(output_root=str(root),sources=[dict(spec,name='a'),dict(spec,name='b')]))
    with sqlite3.connect(root/'selected.afv') as db:
        meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
        assert meta['uuid'] and meta['complete'] and meta['fiber_count']==2
    e=json.loads((root/'cross/cross_events.json').read_text())['events']
    assert [r['test_source'] for r in e]==[0,1]
    assert [r['gt_source'] for r in e]==[99,99]
    assert 'winding' not in np.load(root/'within_tensors.npz').files
