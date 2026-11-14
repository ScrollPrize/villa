import numpy as np
import pytest
import json
from types import SimpleNamespace
from .fiber_graph_fixture import uid
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import Seam,solve,aggregate_l1_edges
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_benchmark import Paths,graph
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_full import PackedPaths,KEY_DTYPE,PACKED_GEOMETRY_FORMAT


def test_sparse_source_ids_partition_rows_exactly_once():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_full import constraint_jobs
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
    maximum=np.iinfo(np.int64).max
    ids=np.array([0,2**32+3,2**40,2**48,maximum-1,maximum],np.int64)
    jobs=constraint_jobs(ids,2)
    assert len(jobs)==3
    assert jobs[-1][1]==int(maximum)+1
    graph=object.__new__(FiberGraph);graph.event_ids=ids
    read=[]
    for first,last in jobs:
        for batch in graph.batches(1,first,last):read.extend(range(batch.start,batch.stop))
    assert read==list(range(len(ids)))
    assert list(graph.batches(1,maximum,maximum+1))==[slice(5,6)]


def test_constraint_job_rejects_stale_interval_marker(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_min_flow_full as full
    parts=tmp_path/'graph_parts';parts.mkdir()
    (parts/'0000000000.json').write_text(json.dumps(dict(first=0,last=500000,read=0)))
    np.savez(parts/'0000000000.npz',a=[],b=[],target=[],weight=[])
    calls=[]
    def events(config,paths,db,first,last):
        calls.append((first,last))
        yield 2,dict(i=np.array([],np.int64))
    monkeypatch.setattr(full,'event_batches',events)
    monkeypatch.setattr(full,'_worker',(dict(output_root=str(tmp_path)),None,None))
    result=full.constraint_job((0,2**32+4))
    assert calls==[(0,2**32+4)] and result['read']==2
    assert result['last']==2**32+4
    calls.clear()
    assert full.constraint_job((0,2**32+4))==result
    assert calls==[]


def test_legacy_sgd_preparation_uses_sparse_row_jobs(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_flow_sgd_full as sgd
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_full import constraint_jobs
    source=tmp_path/'source';source.mkdir();(source/'geometry').mkdir()
    (source/'request.json').write_text(json.dumps(dict(config=dict(database='graph',constraint_batch=2))))
    ids=np.array([0,2**32+1,2**40,2**48,np.iinfo(np.int64).max],np.int64)
    monkeypatch.setattr(sgd,'PackedPaths',lambda _:SimpleNamespace(ids=np.array(['fiber']),
        bind_graph=lambda _:None,save_graph_index=lambda:None,graph_index=np.array([0])))
    monkeypatch.setattr(sgd.FiberGraph,'cached',lambda *args:SimpleNamespace(event_ids=ids))
    def scheduled(config,stage,jobs,worker):
        assert stage=='events' and jobs==constraint_jobs(ids,2) and len(jobs)==3
        raise RuntimeError('stop after scheduling')
    monkeypatch.setattr(sgd,'parallel_stage',scheduled)
    with pytest.raises(RuntimeError,match='stop after scheduling'):
        sgd.prepare(dict(output_root=str(tmp_path/'output'),source_root=str(source)),{})


def test_packed_matches_scalar_paths_with_zero_segments_and_axis_knots(tmp_path):
    axis=np.array([[0,0,-10],[.2,.1,0],[0,0,10]])
    seam=Seam(axis)
    curves=[np.array([[-1,-10,-5],[-1,-10,-5],[1,-10,5],[2,-10,8.]]),
            np.array([[10,1,-3],[10,-1,3],[10,-2,5.]])]
    ids=np.array([uid(13),uid(79)]);offset=np.r_[0,np.cumsum([len(x) for x in curves])]
    keys=np.empty(offset[-1],KEY_DTYPE);turns=np.empty(offset[-1],np.int64)
    ordinary=object.__new__(Paths);ordinary.seam=seam;ordinary.paths={}
    for i,(sid,xyz) in enumerate(zip(ids,curves)):
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        t=seam.polyline(xyz);ordinary.paths[str(sid)]=(arc,xyz,t)
        sl=slice(offset[i],offset[i+1]);keys['source'][sl]=i;keys['arc'][sl]=arc;turns[sl]=t
    np.savez(tmp_path/'catalog.npz',format_version=PACKED_GEOMETRY_FORMAT,volume=uid(999),
             source_id=ids,point_offset=offset,axis=axis)
    np.save(tmp_path/'keys.npy',keys);np.save(tmp_path/'xyz.npy',np.concatenate(curves));np.save(tmp_path/'turns.npy',turns)
    packed=PackedPaths(tmp_path)
    sources=np.repeat(ids,100);arcs=np.concatenate([np.linspace(0,ordinary.paths[str(s)][0][-1],100) for s in ids])
    pa,ta=ordinary.at_many(sources,arcs);pb,tb=packed.at_many(sources,arcs)
    np.testing.assert_array_equal(pa,pb);np.testing.assert_array_equal(ta,tb)
    pc,tc=packed.at_indices(np.repeat(np.arange(len(ids)),100),arcs)
    np.testing.assert_array_equal(pa,pc);np.testing.assert_array_equal(ta,tc)


def test_event_batches_use_dense_graph_indices_and_reject_missing_nodes():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_full import event_batches
    graph=object.__new__(FiberGraph)
    graph.ids=np.array([uid(1),uid(2),uid(3)])
    graph.event_ids=np.arange(3);graph.i=np.array([0,0,2]);graph.j=np.array([2,1,2])
    graph.parallel=np.array([False,False,True]);graph.arc_a=np.arange(3.);graph.arc_b=np.arange(3.)+1
    graph.distance=np.ones(3);graph.valid_a=graph.valid_b=graph.valid_connector=np.ones(3,bool)
    graph.normal_a=graph.normal_b=graph.connector=np.tile([0.,0.,1.],(3,1))
    # No UUID endpoint columns are present: hot-path string lookups would fail.
    paths=object.__new__(PackedPaths);paths.ids=graph.ids[[0,2]];paths.graph_index=None
    paths.catalog=dict(family=np.array([0,1]),normal_sign=np.ones(2))
    paths.bind_graph(graph)
    assert paths.graph_index.tolist()==[0,-1,1]
    rows=list(event_batches({},paths,graph,0,3))
    assert len(rows)==1
    _,d=rows[0]
    assert d['i'].tolist()==[0,1] and d['j'].tolist()==[1,1]
    assert d['event_ids'].tolist()==[0,2]


@pytest.mark.parametrize('supplied_curves',[False,True])
def test_paths_use_working_frame_for_curves_and_axis(monkeypatch,supplied_curves):
    from vesuvius.neural_tracing.tracers.shared.annotations import fiber_store
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_normal_sign_visualization as normals
    volume=uid(901);sid=uid(1)
    axis=np.array([[10.,0,-10],[10.,0,10]])
    xyz=np.array([[9.,-1,0],[11.,-1,0]])
    calls=[]
    def load_axis(path,target):
        assert (path,target)==('axis',volume)
        return axis
    def load_curves(path,ids,*,volume):
        calls.append((path,ids,volume))
        return [(dict(id=sid),xyz)]
    monkeypatch.setattr(normals,'load_umbilicus_frame',load_axis)
    monkeypatch.setattr(fiber_store,'read_polylines',load_curves)
    d=dict(source_id=np.array([sid]),start=np.zeros(1),links=np.empty((0,4)))
    paths=Paths(dict(fibers='source',umbilicus='axis',working_volume=volume),d,
                curves=[(dict(id=sid),xyz)] if supplied_curves else None)
    assert paths.volume==volume
    assert calls==([] if supplied_curves else [('source',[sid],volume)])
    np.testing.assert_array_equal(paths.paths[sid][1],xyz)
    np.testing.assert_array_equal(paths.paths[sid][2],[0,1])


@pytest.mark.parametrize('metadata',[dict(format_version=2),dict(format_version=PACKED_GEOMETRY_FORMAT),
                                    dict(format_version=PACKED_GEOMETRY_FORMAT,volume='')])
def test_packed_rejects_unverified_frame_cache(tmp_path,metadata):
    np.savez(tmp_path/'catalog.npz',**metadata)
    with pytest.raises(ValueError,match='working (frame|volume)'):
        PackedPaths(tmp_path)


def test_geometry_worker_uses_recorded_target_frame(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_min_flow_full as mod
    root=tmp_path/'geometry';root.mkdir()
    sid=uid(1);volume=uid(901);xyz=np.array([[9.,-1,0],[13.,-1,0]])
    np.savez(root/'catalog.npz',format_version=PACKED_GEOMETRY_FORMAT,volume=volume,
             source_id=np.array([sid]),point_offset=np.array([0,2]),axis=np.array([[10.,0,-10],[10.,0,10]]))
    np.save(root/'keys.npy',np.zeros(2,KEY_DTYPE))
    np.save(root/'xyz.npy',np.zeros((2,3)))
    np.save(root/'turns.npy',np.zeros(2,np.int64))
    paths=PackedPaths(root,mode='r+')
    def read(path,ids,*,volume):
        assert path=='source' and ids.tolist()==[sid] and volume==uid(901)
        return np.array([sid]),SimpleNamespace(values=xyz,offsets=np.array([0,2]))
    monkeypatch.setattr(mod,'read_polyline_arrays',read)
    monkeypatch.setattr(mod,'_worker',(dict(fibers='source',output_root=str(tmp_path)),paths,None))
    result=mod.geometry_job((0,1))
    assert result['fibers']==1 and result['points']==2
    np.testing.assert_array_equal(paths.xyz,xyz)
    np.testing.assert_array_equal(paths.keys['arc'],[0,4])
    np.testing.assert_array_equal(paths.turns,[0,1])


def test_full_prepare_records_working_frame_but_counts_source_points(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_min_flow_full as mod
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_normal_sign_visualization as normals
    sid=uid(1);source,target=uid(900),uid(901)
    comparison=tmp_path/'comparison.npz';normal=tmp_path/'normal.npz'
    np.savez(comparison,ids=np.array([sid]),labels=np.array([0]))
    np.savez(normal,ids=np.array([sid]),normal_sign=np.array([1]),normal_convention='inward')
    monkeypatch.setattr(mod,'workflow_volume',lambda _:target)
    def catalog(path):
        assert path=='source'
        return np.array([sid]),np.array([2])
    def axis(path,volume):
        assert path=='axis' and volume==target
        return np.array([[10.,0,-10],[10.,0,10]])
    monkeypatch.setattr(mod,'fiber_point_counts',catalog)
    monkeypatch.setattr(normals,'load_umbilicus_frame',axis)
    monkeypatch.setattr(mod,'parallel_stage',lambda *args:[])
    def cached(database,root):
        assert database=='graph' and root==tmp_path/'shared-columns'
        return SimpleNamespace(event_ids=np.array([0]),ids=np.array([sid]))
    monkeypatch.setattr(mod.FiberGraph,'cached',cached)
    config=dict(fibers='source',database='graph',umbilicus='axis',comparison=str(comparison),
                normal_solution=str(normal),output_root=str(tmp_path/'out'),graph_cache=str(tmp_path/'shared-columns'))
    result=mod.prepare(config)
    assert result['volume'].item()==target
    assert result['point_offset'].tolist()==[0,2]
    # The caller has verified the immutable source/run pins. Reuse must not
    # rescan descriptors or start workers just to return completed geometry.
    (tmp_path/'out/geometry/block_00000000.json').write_text(json.dumps(dict(first=0,last=1,points=2)))
    def forbidden_counts(_):raise AssertionError('verified geometry cache must not reread source catalog')
    monkeypatch.setattr(mod,'fiber_point_counts',forbidden_counts)
    def completed(config,stage,jobs,worker):
        assert stage=='geometry' and jobs==[]
        return []
    monkeypatch.setattr(mod,'parallel_stage',completed)
    reused=mod.prepare(config,build_graph=False,reuse_catalog=True)
    np.testing.assert_array_equal(reused['point_offset'],[0,2])
    # A cache still has to match the selected frame, axis and fiber labels.
    np.savez(comparison,ids=np.array([sid]),labels=np.array([1]))
    with pytest.raises(ValueError,match='Geometry catalog changed'):
        mod.prepare(config,build_graph=False,reuse_catalog=True)
    np.savez(comparison,ids=np.array([sid]),labels=np.array([0]))
    monkeypatch.setattr(mod,'fiber_point_counts',catalog)
    # A prior cache may contain the same fibers but no trustworthy frame.
    np.savez(tmp_path/'out/geometry/catalog.npz',**{k:v for k,v in result.items() if k!='volume'})
    with pytest.raises(ValueError,match='Geometry catalog changed'):
        mod.prepare(config)


def test_geometry_worker_rejects_stale_interval_marker(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_full import completed_geometry_block
    marker=tmp_path/'block_00000000.json'
    marker.write_text(json.dumps(dict(first=0,last=4096)))
    assert completed_geometry_block(tmp_path,0,4096)==dict(first=0,last=4096)
    assert completed_geometry_block(tmp_path,0,8192) is None


def test_aggregation_preserves_objective_and_certificate():
    a=np.array([0,0,1,0,1,2,2]);b=np.array([1,1,0,2,2,1,0]);t=np.array([1,1,-1,2,1,-1,-2]);w=np.array([.25,.5,.7,1,.3,.6,.2])
    g=aggregate_l1_edges(a,b,t,w)
    assert len(g['a'])==3
    for k in ([0,0,0],[0,1,2],[0,2,1]):
        k=np.array(k);expected=np.sum(w*abs(k[b]-k[a]-t));actual=np.sum(g['weight']*abs(k[g['b']]-k[g['a']]-g['target']))
        np.testing.assert_allclose(actual,expected,rtol=1e-15)
    result=solve(4,**g);original=solve(4,a,b,t,w)
    np.testing.assert_array_equal(result['winding'],original['winding'])
    assert result['certificate']['optimal']


def test_graph_accepts_uint8_hv_labels_without_unsigned_subtraction():
    class Geometry:
        seam=Seam([[0,0,-10],[0,0,10]])
        def ends(self,side):return np.array([[10.,0,0]]),np.array([0])
    d=dict(i=np.array([0]),j=np.array([1]),family=np.array([1,0],np.uint8),
           distance=np.array([.5]),sign=np.array([1]),parallel=np.array([False]))
    g,_=graph({},d,Geometry())
    assert g['target'].tolist()==[1]


def test_prepare_only_never_launches_solver(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_min_flow_full as mod
    inputs={}
    for key in ('fibers','database','comparison','normal_solution','umbilicus'):
        p=tmp_path/key;p.write_text('input');inputs[key]=str(p)
    root=tmp_path/'output';root.mkdir()
    np.savez(root/'graph.npz',a=np.array([0]),b=np.array([1]),target=np.array([1]),weight=np.array([1.]))
    monkeypatch.setattr(mod,'store_digest',lambda _: 'test-input')
    monkeypatch.setattr(mod,'prepare',lambda _:dict(source_id=np.array([1,2])))
    def fail(*args,**kwargs):raise AssertionError('prepare_only invoked solver')
    monkeypatch.setattr(mod,'solve',fail)
    result=mod.run(dict(inputs,output_root=str(root),prepare_only=True))
    assert result['prepared_only'] and result['fibers']==2 and result['arcs']==1
    assert not (root/'solution.npz').exists()


def test_run_rejects_pre_frame_completed_cache(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_min_flow_full as mod
    inputs={}
    for key in ('fibers','database','comparison','normal_solution','umbilicus'):
        path=tmp_path/key;path.write_text('input');inputs[key]=str(path)
    root=tmp_path/'output';root.mkdir()
    config=dict(inputs,output_root=str(root))
    old=dict(config=config,inputs={key:dict(path=str((tmp_path/key).resolve()),size=5,
        mtime_ns=(tmp_path/key).stat().st_mtime_ns) for key in ('comparison','normal_solution','umbilicus')},
        stores=dict(fibers='snapshot',database='snapshot'))
    (root/'request.json').write_text(json.dumps(old))
    (root/'summary.json').write_text('{}')
    monkeypatch.setattr(mod,'store_digest',lambda _:'snapshot')
    with pytest.raises(ValueError,match='inputs/configuration changed'):
        mod.run(config)
