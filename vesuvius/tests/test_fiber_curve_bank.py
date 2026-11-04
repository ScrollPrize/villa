import json
import numpy as np
import pytest


def test_curve_bank_actual_polyline_and_normal_invalidity(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_curve_bank as m
    xyz=np.array([[0.,0.,10.],[10.,0.,10.],[10.,10.,10.]])
    fid='00000000-0000-4000-8000-000000000007'
    monkeypatch.setattr(m,'read_polylines',lambda _,ids:[({'id':str(i)},xyz) for i in ids])
    normals=np.array([[0,0,0,1,1],[16,0,0,1,1],[20,0,0,1,1]],dtype='<f8')
    normals.tofile(tmp_path/'normals.bin')
    (tmp_path/'index.json').write_text(json.dumps(dict(format='fiber-normal-curves',columns=['arc','nx','ny','nz','valid'],
        index=[[fid,0,3]],statistics=dict(samples=3,sample_spacing_base=16))))
    sign=dict(ids=np.array([fid]),broken=np.array([False]),normal_sign=np.array([-1]))
    bank=m.CurveBank('unused',[fid],tmp_path,sign)
    point,n,ok=bank.sample(np.array([0,0,0]),np.array([5.,15.,20.]))
    np.testing.assert_allclose(point,[[5,0,10],[10,5,10],[10,10,10]])
    np.testing.assert_allclose(n,np.tile([0,0,-1],(3,1)));assert ok.all()
    with pytest.raises(ValueError,match='outside'):bank.sample(np.array([0]),np.array([21.]))
    del bank
    normals[1,4]=0;normals.tofile(tmp_path/'normals.bin')
    bank=m.CurveBank('unused',[fid],tmp_path,sign)
    assert not bank.sample(np.array([0]),np.array([5.]))[2][0]
    del bank
    with pytest.raises(ValueError,match='Missing normal'):m.CurveBank('unused',['00000000-0000-4000-8000-000000000008'],tmp_path,sign)
