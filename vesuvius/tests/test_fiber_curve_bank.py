import json
import numpy as np
import pytest


def test_curve_bank_actual_polyline_and_normal_invalidity(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_curve_bank as m
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_fibers, fiber_source_version
    from vc_object_store import ObjectStore
    xyz=np.array([[0.,0.,10.],[10.,0.,10.],[10.,10.,10.]])
    fid='00000000-0000-4000-8000-000000000007'
    other='00000000-0000-4000-8000-000000000008'
    source=tmp_path/'fibers.vcobj'
    write_fibers(source,[(dict(id=i),xyz) for i in (fid,other)],volume='vol')
    normals=np.array([[0,0,0,1,1],[16,0,0,1,1],[20,0,0,1,1]],dtype='<f8')
    normals.tofile(tmp_path/'normals.bin')
    (tmp_path/'index.json').write_text(json.dumps(dict(format='fiber-normal-curves',columns=['arc','nx','ny','nz','valid'],
        index=[[fid,0,3]],statistics=dict(samples=3,sample_spacing_base=16),
        provenance=dict(fiber_source=fiber_source_version(source)))))
    sign=dict(ids=np.array([fid]),broken=np.array([False]),normal_sign=np.array([-1]))
    bank=m.CurveBank(source,[fid],tmp_path,sign)
    point,n,ok=bank.sample(np.array([0,0,0]),np.array([5.,15.,20.]))
    np.testing.assert_allclose(point,[[5,0,10],[10,5,10],[10,10,10]])
    np.testing.assert_allclose(n,np.tile([0,0,-1],(3,1)));assert ok.all()
    def no_geometry(*_):
        raise AssertionError('Normal-only sampling must not load fiber geometry')
    monkeypatch.setattr(m,'read_polylines',no_geometry)
    normal_bank=m.CurveBank.from_normals([fid],tmp_path,sign)
    sampled,valid=normal_bank.sample_normals(np.array([0,0,0]),np.array([5.,15.,20.]))
    np.testing.assert_array_equal(sampled,n)
    np.testing.assert_array_equal(valid,ok)
    monkeypatch.setattr(m,'read_polylines',lambda _,ids:[({'id':str(i)},xyz) for i in ids])
    with pytest.raises(ValueError,match='outside'):bank.sample(np.array([0]),np.array([21.]))
    del bank
    normals[1,4]=0;normals.tofile(tmp_path/'normals.bin')
    bank=m.CurveBank(source,[fid],tmp_path,sign)
    assert not bank.sample(np.array([0]),np.array([5.]))[2][0]
    del bank
    with pytest.raises(ValueError,match='Missing normal'):m.CurveBank(source,[other],tmp_path,sign)
    with ObjectStore.open(str(source),author='edit') as store:
        store.set_meta(fid,'coordinates',dict(volume_uuid='vol',scale_factor=2))
    with pytest.raises(ValueError,match='source changed'):m.CurveBank(source,[fid],tmp_path,sign)
