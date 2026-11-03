import numpy as np
import json
from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_convention import inward_solution


def test_inward_convention_and_outward_display():
    original=dict(normal_sign=np.array([1,-1]),sign_logit=np.array([2.,-3.]),
                  sign_probability=np.array([.8,.1]),broken=np.array([False,True]))
    inward=inward_solution(original)
    for key in ('normal_sign','sign_logit'):np.testing.assert_array_equal(inward[key],-original[key])
    np.testing.assert_allclose(inward['sign_probability'],1-original['sign_probability'])
    np.testing.assert_array_equal(inward['broken'],original['broken'])
    for key,value in inward.items():np.testing.assert_array_equal(inward_solution(inward)[key],value)
    p=np.array([10.,20.,30.]);outward=np.array([1.,0.,0.])
    old,_=arrow_mesh(p,outward,128,4);new,_=arrow_mesh(p,outward,640,20)
    np.testing.assert_allclose(new-p,5*(old-p),atol=1e-12)
from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_sign_visualization import obj_edges, repair_obj_lines
from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_sign_visualization import spatial_subset
from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_sign_visualization import arrow_mesh, write_ply
from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_sign_visualization import tube_mesh, export_umbilicus, umbilicus_frame
from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_sign_visualization import catalog_rows
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_fibers
from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_sign_visualization import subset_mesh, export_classified_layers


def test_classified_layers_have_independent_indices(tmp_path):
    import pytest
    raw=[];records=[];vertices=[];faces=[];colors=[]
    for label in range(3):
        p=np.array([label*100.,0,0]);vv,ff=arrow_mesh(p,[0,0,1])
        offset=len(vv)*label;vertices.append(vv);faces.append(ff+offset);colors.append(np.tile([label*100,0,0],(len(vv),1)))
        fid=f'00000000-0000-4000-8000-00000000000{label}'
        raw.append((dict(id=fid),np.array([p,p+[10,0,0]])))
        records.append(dict(fiber_id=fid,hv_label=label,broken=label==2,rejection_stage=2 if label==2 else 0,
                            arrows=[dict(ply_first_vertex=offset,ply_vertex_count=len(vv))]))
    vv,ff,cc=map(np.concatenate,(vertices,faces,colors))
    layers=export_classified_layers(tmp_path,records,raw,vv,ff,cc)
    for name in ('h','v','broken'):
        assert layers[name]['fibers']==1 and layers[name]['obj_edges']==1
        assert (tmp_path/f'{name}_fibers.obj').read_text().endswith('l 1 2\n')
    assert layers['broken']['stage2_rejected']==1
    assert all(r['arrows'][0]['layer_ply_first_vertex']==0 for r in records)
    with pytest.raises(ValueError,match='crosses label'):subset_mesh(vv,ff,cc,[0])


def test_umbilicus_tube_and_frame(tmp_path):
    import pytest
    points=np.array([[10,10,10],[10,10,50],[10,10,50],[20,10,80]],dtype=float)
    v,f=tube_mesh(points,3)
    assert v.shape==(52,3) and f.shape==(96,3)
    assert f.min()==0 and f.max()==len(v)-1
    np.testing.assert_allclose(np.linalg.norm(v[:12]-points[0],axis=1),3)
    path=tmp_path/'axis.json'
    path.write_text(json.dumps(dict(metadata=dict(volume_slices=100,volume_height=100,volume_width=100,volume='123-scan.zarr',voxelsize_um=2.4),
                                   control_points=[dict(zip('xyz',p)) for p in points])))
    assert export_umbilicus(tmp_path,path,'123',3)['points']==4
    np.testing.assert_allclose(umbilicus_frame(path,'123')[1],.0024)
    with pytest.raises(ValueError,match='frame mismatch'):export_umbilicus(tmp_path,path,'456',3)
    path.write_text(path.read_text().replace('"volume_width": 100','"volume_width": 15'))
    with pytest.raises(ValueError,match='outside native volume'):export_umbilicus(tmp_path,path,'123',3)


def test_arrow_tip_and_faces(tmp_path):
    p=np.array([100.,200.,300.]);n=np.array([0.,-1.,0.])
    vertices,faces=arrow_mesh(p,n,64.,2.)
    np.testing.assert_allclose(vertices[-1],p+64*n)
    assert faces.min()==0 and faces.max()<len(vertices)
    assert np.isfinite(vertices).all()
    path=tmp_path/'a.ply'
    write_ply(path,vertices,faces,np.tile([255,0,0],(len(vertices),1)))
    header,payload=path.read_bytes().split(b'end_header\n',1)
    assert f'element vertex {len(vertices)}'.encode() in header
    assert len(payload)==len(vertices)*15+len(faces)*13


def test_spatial_subset_spans_axes_deterministically():
    points=np.array([[x,y,z*100] for x in (0,1) for y in (0,1) for z in (0,1)]+[[.5,.5,50]])
    ids=np.array([f'00000000-0000-4000-8000-000000000{k}' for k in range(100,100+len(points))])
    chosen=spatial_subset(ids,points,8)
    assert set(chosen)==set(ids[:8])
    assert spatial_subset(ids,points,8)==chosen
    assert len(spatial_subset(ids,points,100))==9


def test_repair_reads_store_geometry(tmp_path):
    path=tmp_path/'source.vcobj';ids=['00000000-0000-4000-8000-000000000001','00000000-0000-4000-8000-000000000007']
    write_fibers(path,[(dict(id=fid),np.array([[k,0,0],[k,1,0],[k,2,0]],dtype=float)) for k,fid in zip((1,7),ids)],volume='123')
    assert catalog_rows(path)==[(ids[0],1.,1.,0.,2.),(ids[1],7.,1.,0.,2.)]
    # Repair a previously truncated preview directly from source geometry.
    (tmp_path/'index.json').write_text(json.dumps(dict(config=dict(fibers=str(path)),records=[
        dict(fiber_id=ids[1],obj_first_vertex=1,obj_vertex_count=2,broken=True,arrows=[])])))
    for name in ('fibers.obj','broken_fibers.obj'):
        (tmp_path/name).write_text('v 7 1 0\nv 7 2 0\nl 1 2\n')
    (tmp_path/'signed_normals.ply').write_bytes(b'unchanged')
    repaired=repair_obj_lines(dict(output_root=str(tmp_path)))
    assert repaired['fibers.obj']['vertices']==3
    assert repaired['fibers.obj']['segments']==2
    assert 'v 7.000000000 0.000000000 0.000000000' in (tmp_path/'fibers.obj').read_text()
    assert (tmp_path/'signed_normals.ply').read_bytes()==b'unchanged'


def test_explicit_obj_edges_and_existing_repair(tmp_path):
    assert obj_edges(4,3)=='l 4 5\nl 5 6\n'
    records=[dict(obj_vertex_count=3,broken=False),dict(obj_vertex_count=2,broken=True)]
    (tmp_path/'index.json').write_text(json.dumps(dict(records=records)))
    original='o first\nv 0 0 0\nv 1 0 0\nv 2 0 0\nl 1 2 3\no second\nv 0 1 0\nv 1 1 0\nl 4 5\n'
    (tmp_path/'fibers.obj').write_text(original)
    (tmp_path/'broken_fibers.obj').write_text('v 0 1 0\nv 1 1 0\nl 1 2\n')
    (tmp_path/'signed_normals.ply').write_bytes(b'untouched')
    result=repair_obj_lines(dict(output_root=str(tmp_path)))
    assert result['fibers.obj']['segments']==3
    assert (tmp_path/'fibers.obj').read_text()==original.replace('l 1 2 3\n','l 1 2\nl 2 3\n')
    assert (tmp_path/'signed_normals.ply').read_bytes()==b'untouched'
    assert repair_obj_lines(dict(output_root=str(tmp_path)))==result
