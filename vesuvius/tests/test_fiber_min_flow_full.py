import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import Seam,solve,aggregate_l1_edges
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_benchmark import Paths,graph
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_full import PackedPaths,KEY_DTYPE


def test_packed_matches_scalar_paths_with_zero_segments_and_axis_knots(tmp_path):
    axis=np.array([[0,0,-10],[.2,.1,0],[0,0,10]])
    seam=Seam(axis)
    curves=[np.array([[-1,-10,-5],[-1,-10,-5],[1,-10,5],[2,-10,8.]]),
            np.array([[10,1,-3],[10,-1,3],[10,-2,5.]])]
    ids=np.array([13,79]);offset=np.r_[0,np.cumsum([len(x) for x in curves])]
    keys=np.empty(offset[-1],KEY_DTYPE);turns=np.empty(offset[-1],np.int64)
    ordinary=object.__new__(Paths);ordinary.seam=seam;ordinary.paths={}
    for i,(sid,xyz) in enumerate(zip(ids,curves)):
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        t=seam.polyline(xyz);ordinary.paths[int(sid)]=(arc,xyz,t)
        sl=slice(offset[i],offset[i+1]);keys['source'][sl]=sid;keys['arc'][sl]=arc;turns[sl]=t
    np.savez(tmp_path/'catalog.npz',source_id=ids,point_offset=offset,axis=axis)
    np.save(tmp_path/'keys.npy',keys);np.save(tmp_path/'xyz.npy',np.concatenate(curves));np.save(tmp_path/'turns.npy',turns)
    packed=PackedPaths(tmp_path)
    sources=np.repeat(ids,100);arcs=np.concatenate([np.linspace(0,ordinary.paths[int(s)][0][-1],100) for s in ids])
    pa,ta=ordinary.at_many(sources,arcs);pb,tb=packed.at_many(sources,arcs)
    np.testing.assert_array_equal(pa,pb);np.testing.assert_array_equal(ta,tb)


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
    for key in ('afv','database','comparison','normal_solution','umbilicus'):
        p=tmp_path/key;p.write_text('input');inputs[key]=str(p)
    root=tmp_path/'output';root.mkdir()
    np.savez(root/'graph.npz',a=np.array([0]),b=np.array([1]),target=np.array([1]),weight=np.array([1.]))
    monkeypatch.setattr(mod,'prepare',lambda _:dict(source_id=np.array([1,2])))
    def fail(*args,**kwargs):raise AssertionError('prepare_only invoked solver')
    monkeypatch.setattr(mod,'solve',fail)
    result=mod.run(dict(inputs,output_root=str(root),prepare_only=True))
    assert result['prepared_only'] and result['fibers']==2 and result['arcs']==1
    assert not (root/'solution.npz').exists()
