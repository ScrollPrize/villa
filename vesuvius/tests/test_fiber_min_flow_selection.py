import numpy as np
from .fiber_graph_fixture import measured_store, uid

from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import Seam
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_benchmark import Paths
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_selection import EdgeStore, slab_winding_range, select_range


def paths(curves):
    p = object.__new__(Paths)
    p.seam = Seam([[0,0,-100],[0,0,100]])
    p.paths = {}
    for sid,points in curves.items():
        x=np.asarray(points,float)
        p.paths[str(sid)]=(np.r_[0,np.cumsum(np.linalg.norm(np.diff(x,axis=0),axis=1))],x,p.seam.polyline(x))
    return p


def test_slab_includes_crossing_with_both_vertices_outside():
    p=paths({1:[[1,0,-10],[1,0,10]]})
    assert slab_winding_range(p,1,3,[-1,1]) == (3.25,3.25)
    assert slab_winding_range(p,1,3,[20,30]) == (np.inf,-np.inf)


def test_winding_range_continuous_across_seam():
    p=paths({1:[[-1,-10,0],[1,-10,0]]})
    lo,hi=slab_winding_range(p,1,0,[-1,1])
    assert .9<lo<1<hi<1.1


def test_selection_rejects_disconnected_and_outside_but_keeps_gt():
    p=paths({k:[[1,0,-10],[1,0,10]] for k in range(1,6)})
    d=dict(source_id=np.asarray([str(i) for i in range(1,6)]),family=np.array([0,1,1,1,1]))
    sol=dict(winding=np.array([0,0,1,0,8]))
    keep,_,_=select_range(d,sol,p,['1','5'],[-1,1],dict(a=np.array([0,1]),b=np.array([1,2])))
    assert keep.tolist()==[True,True,False,False,True]


def test_indexed_edges_deduplicate_filter_and_apply_inward_sign(tmp_path):
    filename=measured_store(tmp_path,[(1,1,2,False,.5),(2,1,3,False,.5),(3,1,4,False,.5)])
    store=EdgeStore(dict(database=filename),[uid(1),uid(2),uid(3)],
                    {uid(1):0,uid(2):1,uid(3):0},{uid(1):-1,uid(2):-1,uid(3):-1})
    store.load([uid(1)]);store.load([uid(2)]);store.load([uid(1)])
    assert store.edges().tolist()==[[uid(1),uid(2),0,0,.5,1,False]]
