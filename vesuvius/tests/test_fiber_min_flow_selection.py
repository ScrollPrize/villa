import numpy as np
import json
import sqlite3

from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import Seam
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_benchmark import Paths
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_selection import EdgeStore, slab_winding_range, select_range


def paths(curves):
    p = object.__new__(Paths)
    p.seam = Seam([[0,0,-100],[0,0,100]])
    p.paths = {}
    for sid,points in curves.items():
        x=np.asarray(points,float)
        p.paths[sid]=(np.r_[0,np.cumsum(np.linalg.norm(np.diff(x,axis=0),axis=1))],x,p.seam.polyline(x))
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
    d=dict(source_id=np.arange(1,6),family=np.array([0,1,1,1,1]))
    sol=dict(winding=np.array([0,0,1,0,8]))
    keep,_,_=select_range(d,sol,p,[1,5],[-1,1],dict(a=np.array([0,1]),b=np.array([1,2])))
    assert keep.tolist()==[True,True,False,False,True]


def test_indexed_edges_deduplicate_filter_and_apply_inward_sign(tmp_path):
    filename=tmp_path/'events.sqlite'
    with sqlite3.connect(filename) as db:
        db.execute('CREATE TABLE constraints(id,fiber_a,fiber_b,arc_a,arc_b,ax,ay,az,bx,by,bz,parallel_score,perpendicular_score,winding_distance,parallel_winding_distance,normal_a,normal_b)')
        normal=json.dumps(dict(valid=True,xyz=[1,0,0]))
        for eid,a,b in [(1,1,2),(2,1,3),(3,1,4)]:
            db.execute('INSERT INTO constraints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (eid,a,b,0,0,0,0,0,1,0,0,0,1,.5,.5,normal,normal))
    store=EdgeStore(dict(database=filename),[1,2,3],{1:0,2:1,3:0},{1:-1,2:-1,3:-1})
    store.load([1]); store.load([2]); store.load([1])
    assert store.edges().tolist()==[[1,2,0,0,.5,1,0]]
    store.db.close()
