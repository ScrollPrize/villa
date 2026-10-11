import json
import numpy as np
import pytest
import tifffile
from vesuvius.neural_tracing.fiber_trace_3d import fiber_reference_comparison as ref


def test_gray_coverage_distinguishes_outside_z_and_restricted_projection(tmp_path,monkeypatch):
    identity=dict(source_id=np.arange(4),start=np.zeros(4),end=np.ones(4))
    cache=tmp_path/'tensor.npz';np.savez(cache,**identity)
    (tmp_path/'summary.json').write_text(json.dumps({'config':{'tensor_cache':str(cache)}}))
    np.savez(tmp_path/'solution.npz',state=np.zeros((4,1)),widths=np.ones(1))
    projection=tmp_path/'projection.npz'
    np.savez(projection,**identity,xyz=np.c_[np.zeros((4,2)),[0,100,200,300]],
        weight=np.ones(4),owner=np.arange(4),placed_valid=np.array([False,True,False,False]),
        nearest_distance=np.array([300,0,0,300]),chosen_winding=np.ones(4))
    monkeypatch.setattr(tifffile,'imread',lambda p:np.array([[100.,200.]]))
    monkeypatch.setattr(ref.uv,'hard_geometry',lambda *a:(None,None,np.ones(4,dtype=bool)))
    result=ref.projection_coverage_report(dict(run_root=str(tmp_path),reference=str(tmp_path),
        projection_cache=str(projection),output_root=str(tmp_path),xyz_scale=1))
    assert result['reference_physical_z']==[100,200]
    assert result['gray_length_percent']==75
    assert result['gray_outside_reference_global_z_percent']==pytest.approx(200/3)
    assert result['gray_nearest_surface_within_threshold_percent']==pytest.approx(100/3)
