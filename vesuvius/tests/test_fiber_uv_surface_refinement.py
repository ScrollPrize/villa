import numpy as np
import pytest
from scipy import sparse
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_surface_refinement as ref
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_event_offsets as eo


def graph(family,buckets,source,i,j,shift,sign):
    n=len(family);k=3;s=np.zeros((n,3*k+3));s[:,:k+1]=-20;s[np.arange(n),buckets]=20
    d=dict(family=np.array(family),source_id=np.array(source),phase=np.full(n,.2),z=np.zeros(n),
        i=np.array(i),j=np.array(j),bucket_shift=np.array(shift),sign=np.array(sign),
        phase_delta=np.zeros(len(i)),ds_a=np.zeros(len(i)),ds_b=np.zeros(len(i)),
        parallel=np.array(family)[i]==np.array(family)[j],links=np.empty((0,4)))
    return d,s,np.full(k,1000.),dict(bucket_position_split=True,bucket_shared_phase=True,layout_mode='shared_anchor')


def test_unconditional_probability_and_whole_source_removal():
    d,s,w,c=graph([0,0,0,1],[1,0,1,1],[11,22,11,33],[0],[1],[0],[1])
    result=ref.select(d,s,w,c)
    assert result['removed_sources'].tolist()==[11,22]
    assert result['retained'].tolist()==[False,False,False,True]
    s[3,:4]=np.log([.05,.49,.06,.40])
    assert not ref.select(d,s,w,c)['confident'][3]
    with pytest.raises(ValueError):ref.select(d,s,w,c,.49)


def test_same_sheet_layer_order_seam_and_adjacent_winding():
    d,s,w,c=graph([0,1,0,1],[1,0,0,1],[1,2,3,4],[0,2],[1,3],[1,0],[1,1])
    result=ref.select(d,s,w,c)
    assert result['same_sheet'].tolist()==[True,False]
    assert result['retained'].all()
    # A classifier saying perpendicular cannot turn H/H into H/V coincidence.
    d['family'][1]=0;d['parallel'][0]=False
    assert not ref.select(d,s,w,c)['same_sheet'][0]


def test_sign_filter_simultaneous_not_order_dependent():
    d,s,w,c=graph([0,0,0],[2,1,0],[1,2,3],[0,1],[1,2],[0,0],[1,1])
    result=ref.select(d,s,w,c)
    assert result['removed_sources'].tolist()==[1,2,3]


def test_quadratic_matches_dense_and_reduces_uv():
    meta=eo.adjacency([1,2],[0.,0.],[0,1],4.)
    meta['piece']=np.array([0,1])
    model=dict(meta=meta,piece=np.array([0,1]),edges=np.array([0]),links=np.array([],dtype=int),
        selection=dict(same_sheet=np.array([True])),base=np.array([[0.,10.],[20.,0.]]),
        seed=np.zeros((2,2)),grouped=np.zeros((2,2)))
    a,b,initial,_=ref.build_system(model,{})
    expected=np.linalg.lstsq(a.toarray(),b,rcond=None)[0]
    actual,stats=ref.pcg(a,b,initial,rtol=1e-10,max_iterations=200)
    assert stats['converged']
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
    x=actual.reshape(2,4).T
    points=model['base']+x[:2]+x[2:]
    assert np.linalg.norm(points[1]-points[0])<.001


def test_sparse_duplicate_columns_and_disconnected_anchor():
    # First row cancels its duplicate columns; third variable is isolated/anchored.
    a=sparse.coo_matrix(([1.,-1.,1.,1.,1.],([0,0,1,2,3],[0,0,0,1,2])),shape=(4,3)).tocsr()
    b=np.array([7.,2.,3.,4.])
    x,stats=ref.pcg(a,b,np.zeros(3))
    assert stats['converged'];np.testing.assert_allclose(x,[2,3,4])


def test_refined_curve_interpolation_and_source_boundaries():
    s=dict(source_id=np.array([11,22,33]),family=np.array([0,1,0]),piece_uv_base=np.zeros((3,2)),
        endpoint_source=np.array([11,11,22,22]),endpoint_arc=np.array([0.,10.,0.,10.]),
        endpoint_uv_offset_base=np.array([[0.,0.],[10.,20.],[100.,100.],[200.,200.]]))
    p=ref.evaluate_points(s,[0,1,2],[5.,20.,5.],[7.,8.,9.])
    np.testing.assert_allclose(p,[[12,10],[200,208],[9,0]])


def test_initialize_filtered_links_and_nonzero_parent_offsets(tmp_path):
    import sqlite3
    d,s,w,c=graph([0,0,0,1],[0,0,0,0],[1,1,1,2],[0,2],[3,3],[0,0],[1,1])
    d.update(start=np.array([0.,10.,20.,0.]),end=np.array([10.,20.,30.,30.]),
        event_ids=np.array([100,101]),ds_a=np.array([5.,5.]),ds_b=np.array([5.,25.]),
        links=np.array([[0,1,10,0],[1,2,10,0]],dtype=float))
    db=tmp_path/'constraints.sqlite'
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE constraints(id INTEGER,fiber_a INTEGER,fiber_b INTEGER,arc_a REAL,arc_b REAL)')
        con.executemany('INSERT INTO constraints VALUES (?,?,?,?,?)',[(100,1,2,5,5),(101,1,2,25,25)])
    c.update(database=str(db),event_uv_offsets=True,event_offset_tie_distance_base=4.)
    meta=eo.prepare(d,c)
    parent=dict(state=s,widths=w,event_ids=d['event_ids'],
        event_uv_offsets_base=np.column_stack([meta['arc']*.01,meta['arc']*.02]))
    parent.update({'event_endpoint_'+key:meta[key] for key in ('source','arc','piece','kind','constraint_id','side')})
    s[0,:4]=np.log([.49,.005,.005,.5])
    result=ref.initialize(d,parent,c,{})
    assert result['links'].tolist()==[1] and result['edges'].tolist()==[1]
    assert result['meta']['constraint_id'].tolist()==[101,101,1,1]
    np.testing.assert_allclose(result['seed'],[[.25,.5],[.25,.5],[.2,.4],[.2,.4]])
    assert result['meta']['piece'].tolist()==[1,2,0,1]


def test_optional_postpass_dispatch_preserves_main_config(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
    seen=[]
    monkeypatch.setattr(uv,'prepare',lambda cfg:None)
    monkeypatch.setattr(uv,'launch_tensor_solve',lambda cfg,*args:dict(main=True))
    monkeypatch.setattr(ref,'run',lambda cfg:seen.append(cfg) or dict(refined=True))
    cfg=dict(output_root=str(tmp_path),devices='0,1',surface_refinement=dict(minimum_bucket_probability=.75))
    result=uv.run(cfg)
    assert result['surface_refinement']['refined'] and seen[0]['devices']=='0'
    assert seen[0]['parent_solution']==str(tmp_path/'solution.npz')
    assert seen[0]['minimum_bucket_probability']==.75
    assert 'surface_refinement' in cfg
    result=uv.run(dict(cfg,surface_refinement=dict(enabled=False)))
    assert 'surface_refinement' not in result and len(seen)==1
