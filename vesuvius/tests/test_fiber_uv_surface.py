import json
import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_surface import brackets,blend_surface,construct


def test_spanning_brackets_skip_shorter_candidate_and_limit():
    candidates=np.arange(4);positions=np.array([-20.,-10.,10.,20.])
    lo=np.array([0.,30.,30.,0.]);hi=np.array([100.,70.,70.,100.])
    args=(candidates,positions,np.array([0.]),25.,lo,hi,np.array([0.]),np.array([100.]))
    np.testing.assert_array_equal(brackets(*args)[:,0],[0,3])
    np.testing.assert_array_equal(brackets(*args[:3],15.,*args[4:])[:,0],[-1,-1])


class SyntheticCurves:
    ids=np.arange(4)
    def sample(self,source,arc):
        xyz=np.zeros((len(source),3));normal=np.tile([0.,0.,1.],(len(source),1))
        h=source<2
        xyz[:,0]=np.where(h,arc,np.where(source==2,0,100))
        xyz[:,1]=np.where(h,source*100,arc)
        xyz[:,2]=np.where(h,10,30+.2*arc+10*np.sin(np.pi*arc/100))
        return xyz,normal,np.ones(len(source),bool)


def fixture():
    d=dict(phase=np.array([0.,0.,0.,.1]),z=np.array([0.,100.,0.,0.]),family=np.array([0,0,1,1]),
        node_ds=np.tile([0.,50.,100.],(4,1)),source_id=np.arange(4),start=np.zeros(4),end=np.full(4,100.))
    state=np.zeros((4,6));state[:,:2]=np.log([.9,.1])
    return d,state,np.array([1000.]),dict(layout_mode='shared_anchor',broken_threshold=.5)


def test_curvature_transfer_exact_h_and_tifxyz(tmp_path):
    from vesuvius.tifxyz.types import Tifxyz
    from vesuvius.tifxyz.writer import write_tifxyz
    import tifffile
    xyz,valid,stats=construct(*fixture(),SyntheticCurves(),dict(grid_step=25,max_search=150))
    assert valid.all();assert stats['valid_quads']==16
    np.testing.assert_allclose(xyz[[0,-1],:,2],10)
    np.testing.assert_allclose(xyz[2,:,2],20)
    np.testing.assert_allclose(xyz[1,:,2],10+10*np.sin(np.pi/4),atol=1e-5)
    assert stats['twisted_or_degenerate_quads']==0
    valid[2,2]=False;xyz[2,2]=-1
    path=tmp_path/'surface';write_tifxyz(path,Tifxyz(*[xyz[:,:,a] for a in range(3)],_scale=(.04,.04),_mask=valid))
    assert json.loads((path/'meta.json').read_text())['scale']==[.04,.04]
    assert tifffile.imread(path/'mask.tif')[2,2]==0
    assert tifffile.imread(path/'z.tif')[2,2]==-1
    assert tifffile.imread(path/'x.tif').dtype==np.float32


def test_distance_holes_and_joint_threshold():
    d,s,w,c=fixture();xyz,valid,stats=construct(d,s,w,c,SyntheticCurves(),dict(grid_step=25,max_search=50))
    assert valid[2,2];assert not valid[1,2];assert not valid[2,1]
    s[2,:2]=np.log([.5,.5])
    _,valid,stats=construct(d,s,w,c,SyntheticCurves(),dict(grid_step=25,max_search=150))
    assert stats['selected_v']==1
    assert not valid[2,2]  # no left V; exactly50% is excluded


def test_inconsistent_normals_do_not_construct_surface():
    a=np.array([[0.,0.,10.]]);b=a+np.array([0.,100.,0.]);n=np.array([[0.,0.,1.]])
    v=((a+b)/2,a,b,n)
    _,valid=blend_surface(a,b,n,-n,v,v,np.array([.5]),np.array([.5]))
    assert not valid[0]


def test_reversed_source_arc_maps_same_surface():
    class Reversed(SyntheticCurves):
        def sample(self,source,arc):return super().sample(source,100-arc)
    d,s,w,c=fixture()
    expected,_,_=construct(d,s,w,c,SyntheticCurves(),dict(grid_step=25,max_search=150))
    d['phase'][:2]=.1;d['z'][2:]=100;d['node_ds']=-d['node_ds']
    actual,_,_=construct(d,s,w,c,Reversed(),dict(grid_step=25,max_search=150))
    np.testing.assert_allclose(actual,expected,atol=1e-5)


def test_curve_bank_actual_polyline_and_normal_invalidity(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_surface as m
    xyz=np.array([[0.,0.,10.],[10.,0.,10.],[10.,10.,10.]])
    monkeypatch.setattr(m,'read_afv_polylines',lambda *_:[({'id':7},xyz)])
    normals=np.array([[0,0,0,1,1],[16,0,0,1,1],[20,0,0,1,1]],dtype='<f8')
    normals.tofile(tmp_path/'normals.bin')
    (tmp_path/'index.json').write_text(json.dumps(dict(format='fiber-normal-curves',columns=['arc','nx','ny','nz','valid'],
        index=[[7,0,3]],statistics=dict(samples=3,sample_spacing_base=16))))
    sign=dict(ids=np.array([7]),broken=np.array([False]),normal_sign=np.array([-1]))
    bank=m.CurveBank('unused',[7],tmp_path,sign)
    point,n,ok=bank.sample(np.array([0,0,0]),np.array([5.,15.,20.]))
    np.testing.assert_allclose(point,[[5,0,10],[10,5,10],[10,10,10]])
    np.testing.assert_allclose(n,np.tile([0,0,-1],(3,1)));assert ok.all()
    with pytest.raises(ValueError,match='outside'):bank.sample(np.array([0]),np.array([21.]))
    del bank
    normals[1,4]=0;normals.tofile(tmp_path/'normals.bin')
    bank=m.CurveBank('unused',[7],tmp_path,sign)
    assert not bank.sample(np.array([0]),np.array([5.]))[2][0]


def test_source_hash_validation(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_surface as m
    cfg={key:str(tmp_path/key) for key in ('database','afv','stage1','tensor_cache')}
    for key in ('database','afv','stage1'): (tmp_path/key).write_text(key)
    expected={key:m.uv.sha256_file(cfg[key]) for key in ('database','afv','stage1')}
    (tmp_path/'tensor_cache.json').write_text(json.dumps(dict(signature=expected)))
    doc=dict(provenance=dict(config=dict(orient_database=cfg['database'],afv=cfg['afv'])))
    assert m.validate_sources(cfg,doc)==expected
    (tmp_path/'database').write_text('replaced same pathname')
    with pytest.raises(ValueError,match='hash mismatch'):m.validate_sources(cfg,doc)
