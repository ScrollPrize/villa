import json
import numpy as np
import pytest


def test_curve_bank_actual_polyline_and_normal_invalidity(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_curve_bank as m
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
