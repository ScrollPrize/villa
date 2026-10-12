import numpy as np
import json
from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import (
    landmarks,winding_at,palette,weighted_median,triangle_projection,assign_sources,inverse_u,
    surface_edges,discrete_winding)


def test_reference_width_units_offsets_and_coverage(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import reference_interval_widths,reference_width_metrics
    meta=tmp_path/'meta.json'
    meta.write_text(json.dumps(dict(output_step_vx=20,winding_bounds=[
        dict(winding=i,column_begin=a,column_end=b)
        for i,a,b in ((0,0,12),(1,8,22),(2,18,42),(3,38,62))])))
    cfg=dict(reference_widths=dict(meta=str(meta),xyz_scale=4))
    # Landmarks .5,1.5,2.5 correspond to columns 10,20,40.
    np.testing.assert_allclose(reference_interval_widths(cfg,[0,.5,1,1.5,2]),
                               [np.nan,800,1200,1600,np.nan])
    m=reference_width_metrics([800,1600,999],cfg,.5)
    assert m['compared_windings']==2 and m['mean_normalized_factor']==1
    assert m['mean_absolute_relative_error_percent']==0
    assert reference_width_metrics([800],cfg,None)['compared_windings']==0


def test_reference_width_initialization_preserves_radial_prior(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_reference_comparison as ref
    d=dict(radius=np.array([200.,300.]),family=np.array([0,1]),phase=np.zeros(2),node_ds=np.zeros((2,3)))
    cfg=dict(winding_count=4,layout_mode='shared_anchor',reference_initialization='dummy',reference_width_initialization=True)
    monkeypatch.setattr(ref,'reference_interval_widths',lambda c,k:np.array([np.nan,1200.,1600.,np.nan]))
    monkeypatch.setattr(ref,'load_metric_projection',lambda *a:dict(chosen_winding=np.array([1.,2.])))
    radial=uv.radial_initial_geometry(d,cfg)[1].copy()
    state,widths=uv.fresh_initial_state(d,cfg)
    np.testing.assert_allclose(widths[1:3],[1200,1600])
    np.testing.assert_array_equal(widths[[0,3]],radial[[0,3]])
    np.testing.assert_array_equal(uv.radial_initial_geometry(d,cfg)[1],radial)
    np.testing.assert_array_equal(state[:,:4].argmax(1),[1,2])


def test_landmarks_and_colors():
    bounds=[dict(winding=i,column_begin=a,column_end=b) for i,a,b in ((10,0,12),(11,8,22),(12,18,32))]
    x,q=landmarks(bounds)
    np.testing.assert_allclose(x,[10,20])
    np.testing.assert_allclose(winding_at([5,10,15,20,25],x,q),[10,10.5,11,11.5,12])
    assert np.linalg.norm(palette([10])[0].astype(float)-palette([11])[0]) > 150
    np.testing.assert_allclose(palette([10+1e-9])[0],palette([10])[0],atol=1)
    np.testing.assert_array_equal(palette([np.nan])[0],[180]*3)
    np.testing.assert_array_equal(palette([-1,10]),palette([-1,10]))


def test_projection_plane_edges_degenerate():
    vertices=np.array([[[[0,0,0],[1,0,0],[0,1,0]]]],float)
    coords=vertices[...,:2]
    dist,rc=triangle_projection(np.array([[.2,.3,2]]),vertices,coords)
    np.testing.assert_allclose(dist,[[4]])
    np.testing.assert_allclose(rc,[[[.2,.3]]])
    dist,rc=triangle_projection(np.array([[2,0,0]]),vertices,coords)
    np.testing.assert_allclose(dist,[[1]])
    np.testing.assert_allclose(rc,[[[1,0]]])
    dist,_=triangle_projection(np.zeros((1,3)),vertices*0,coords)
    assert np.isinf(dist).all()


def test_opaque_two_pixel_overdraw(tmp_path):
    import tifffile
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import render_curves
    xy=np.array([[1,5],[9,5],[1,5],[9,5]],float)
    render_curves(tmp_path/'map',xy,np.array([[255,0,0]]*2+[[0,0,255]]*2),
                  np.array([0,0,1,1]),np.ones(4,bool),np.zeros(4),height=1000,bounds=[0,0,999,999])
    img=tifffile.imread(tmp_path/'map.tif')
    assert img.shape[0]==1000
    assert ((img[:,5,:3]==[0,0,255]).all(1)).sum()==2
    svg=(tmp_path/'map.svg').read_text()
    assert 'stroke-width="2"' in svg and 'stroke-opacity="1"' in svg


def test_source_median_shared_pieces_and_missing():
    d={'source_id':np.array([1,1,2])}
    s={'offset':np.array([0,2,4,6]),'weight':np.array([1,1,3,3,1,1])}
    choice=assign_sources(d,s,np.array([10,10,11,11,12,12]),np.array([0,0,0,0,999,999]),100)
    np.testing.assert_allclose(choice[:2],[11,11]);assert np.isnan(choice[2])
    assert weighted_median([1,2,3],[0,2,1])==2
    assert np.isnan(weighted_median([np.nan],[1]))


def test_inverse_actual_u_not_bucket():
    np.testing.assert_allclose(inverse_u(np.array([-5,0,10,20,30,40]),np.array([10,20])),[-.5,0,1,1.5,2,2.5])


def test_grouped_median_exact_empty_invalid_ties():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import grouped_weighted_median
    rng=np.random.default_rng(13)
    lengths=rng.integers(0,40,1000);offsets=np.r_[0,lengths.cumsum()]
    values=rng.integers(-4,8,int(offsets[-1])).astype(float)
    weights=rng.random(len(values));weights[::7]=0;weights[::9]=-1
    weights[::11]=np.nan;values[::13]=np.nan;values[::17]=np.inf
    expected=[weighted_median(values[a:b],weights[a:b]) for a,b in zip(offsets[:-1],offsets[1:])]
    np.testing.assert_array_equal(grouped_weighted_median(values,weights,offsets),expected)
    np.testing.assert_array_equal(grouped_weighted_median([4,4,1,2,9,7],[1,3,1,1,0,np.inf],
        [0,0,2,4,4,6,6]),[np.nan,4,1,np.nan,np.nan,np.nan])


def test_reference_metric_fast_and_shared_geometry():
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_reference_comparison as module
    d=dict(source_id=np.array([1,1,2,3]),start=np.zeros(4),family=np.array([0,0,1,1]),
           node_ds=np.array([[0,1,2]]*4),lengths=np.full(4,2.),phase=np.full(4,.1),z=np.zeros(4))
    state=np.zeros((4,12));state[:,:4]=[4,0,0,-2];state[1,:4]=[0,0,0,5]
    widths=np.full(3,10.)
    projection=dict(owner=np.repeat(np.arange(4),2),arc=np.tile([0,2],4),weight=np.ones(8),
                    offset=np.arange(0,9,2),chosen_winding=np.array([0,0,0,np.nan]))
    cfg=dict(layout_mode='shared_anchor')
    geo=module.uv.hard_geometry(d,state,widths,cfg)
    old=module.reference_winding_metrics(d,state,widths,dict(cfg,diagnostic_backend='reference'),projection)
    new=module.reference_winding_metrics(d,state,widths,cfg,projection,geometry=geo)
    assert old==new


def test_surface_gap_and_endpoint_windings():
    from types import SimpleNamespace
    surface=SimpleNamespace(stride=1,valid_cells=np.ones((4,6),bool))
    surface.valid_cells[1,2]=False
    rc=np.array([[1.2,.2],[1.2,3.2],[2.2,3.2],[2.2,4.2]])
    result=surface_edges(surface,rc,np.ones(4,bool),np.zeros(4,int),np.arange(4)*200,80)
    assert result.tolist()==[False,True,True]
    np.testing.assert_allclose(discrete_winding([9.2,10.5,135.8],(10,135)),[10,11,135])


def test_surface_reader_and_restricted_projection(tmp_path):
    import json
    import tifffile
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import Surface
    r,c=np.indices((7,16))
    for name,array in zip('xyz',(c*10,r*10,np.full(r.shape,100))):
        tifffile.imwrite(tmp_path/(name+'.tif'),array.astype(np.float32))
    bounds=[dict(winding=i,column_begin=a,column_end=b) for i,a,b in ((10,0,6),(11,4,11),(12,9,16))]
    (tmp_path/'meta.json').write_text(json.dumps(dict(winding_bounds=bounds,output_step_vx=10)))
    surface=Surface(tmp_path,scale=1,stride=1,workers=1,candidates=4)
    points=np.array([[52,22,105]],float)
    rc,dist=surface.project(points)
    np.testing.assert_allclose(rc,[[2.2,5.2]])
    np.testing.assert_allclose(dist,[5])
    limited,other=surface.project(points,np.flatnonzero(surface.labels==12))
    assert other[0]>dist[0]
    assert limited[0,1]>=10


def test_end_to_end_and_projection_cache(tmp_path,monkeypatch):
    import json
    import tifffile
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_reference_comparison as module
    from vesuvius.neural_tracing.fiber_follow.data import afv
    root=tmp_path/'reference';root.mkdir()
    r,c=np.indices((7,16))
    for name,array in zip('xyz',(c*10,r*10,np.full(r.shape,100))):
        tifffile.imwrite(root/(name+'.tif'),array.astype(np.float32))
    bounds=[dict(winding=i,column_begin=a,column_end=b) for i,a,b in ((10,0,6),(11,4,11),(12,9,16))]
    (root/'meta.json').write_text(json.dumps(dict(winding_bounds=bounds,output_step_vx=10)))
    source=tmp_path/'solve';source.mkdir();original=tmp_path/'source.afv';original.write_text('fixture')
    tensors=tmp_path/'tensors.npz'
    d=dict(source_id=np.array([1,1]),start=np.array([0.,30.]),end=np.array([30.,60.]),family=np.zeros(2,int),
           node_ds=np.array([[0.,15.,30.]]*2),lengths=np.full(2,30.),phase=np.array([.1,.7]),z=np.full(2,20.))
    np.savez(tensors,**d)
    cfg=dict(afv=str(original),tensor_cache=str(tensors),layout_mode='shared_anchor',broken_threshold=.5)
    (source/'summary.json').write_text(json.dumps(dict(config=cfg)))
    state=np.zeros((2,12));state[:,:4]=[5,-5,-5,-10]
    np.savez(source/'solution.npz',state=state,widths=np.full(3,50.),**{k:d[k] for k in ('source_id','start','end','family')})
    monkeypatch.setattr(afv,'read_afv_polylines',lambda *args:[({'id':1},np.array([[15.,20.,100.],[75.,20.,100.]]))])
    out=tmp_path/'out'
    settings=dict(reference=str(root),output_root=str(out),xyz_scale=1,surface_stride=1,workers=1,
                  render_height=40,solutions=[dict(run_root=str(source),label='toy')])
    result=module.run(settings)
    assert result[0]['retained_pieces']==2
    with np.load(out/'projection.npz') as projection:
        assert projection['chosen_winding'][0]==projection['chosen_winding'][1]
    assert (out/'toy'/'reference_fit.svg').exists()
    assert tifffile.imread(out/'toy'/'reference_fit.tif').shape[0]==40
    with tifffile.TiffFile(out/'toy'/'reference_fit_detail.tif') as detail:
        assert detail.pages[0].compression.name=='LZW'
    assert not list(out.rglob('*.png'))
    monkeypatch.setattr(module,'samples',lambda *args: (_ for _ in ()).throw(AssertionError('cache not used')))
    module.run(settings)
    settings['projection_cache_root']=str(out)
    settings['output_root']=str(tmp_path/'second')
    module.run(settings)
    assert not (tmp_path/'second'/'projection.npz').exists()
    import pytest
    settings['max_distance_base']=123
    with pytest.raises(ValueError,match='External projection cache'):
        module.run(settings)


def test_detail_migration(tmp_path):
    import tifffile
    from PIL import Image
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_comparison import convert_detail_pngs
    pixels=np.arange(60,dtype=np.uint8).reshape(4,5,3)
    source=tmp_path/'reference_own_detail.png'
    Image.fromarray(pixels).save(source)
    Image.fromarray(pixels).save(tmp_path/'unrelated.png')
    result=convert_detail_pngs(tmp_path)
    assert result['count']==1
    assert not source.exists()
    assert (tmp_path/'unrelated.png').exists()
    with tifffile.TiffFile(source.with_suffix('.tif')) as image:
        assert image.pages[0].compression.name=='LZW'
        np.testing.assert_array_equal(image.asarray(),pixels)
