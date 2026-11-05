import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_inlier_metrics import measure


def fixture():
    d=dict(i=np.array([0,0]),j=np.array([1,2]),family=np.array([0,0,1]),
        phase=np.full(3,.2),z=np.zeros(3),ds_a=np.zeros(2),ds_b=np.zeros(2),
        phase_delta=np.zeros(2),bucket_shift=np.zeros(2),sign=np.ones(2),
        distance=np.array([0.,.5]),parallel=np.array([True,False]))
    s=np.zeros((3,12));s[:,:4]=-10;s[:,0]=10
    cfg=dict(bucket_shared_phase=True,bucket_position_split=True,layout_mode='shared_anchor',
        distance_weight=.25,position_distance_weight=2.,metric_weight=.02,perpendicular_relative_weight=2.5)
    return d,s,np.full(3,1000.),cfg


def test_native_geometry_offsets_and_gate():
    d,s,w,c=fixture()
    baseline,cohort=measure(d,s,w,c)
    assert cohort.all() and baseline['fixed']['balanced_fine_loss']==0
    offsets=np.array([[0,0],[0,0],[100,128],[200,256.]])
    result,mask=measure(d,s,w,c,offsets,cohort)
    assert np.array_equal(mask,cohort)  # fine offsets never change gate
    p=result['fixed']['parallel'];v=result['fixed']['perpendicular']
    assert p['u_loss']==pytest.approx(.02) and p['v_loss']==1
    assert p['fine_loss']==pytest.approx(.03)
    assert v['u_loss']==pytest.approx(.08) and v['v_loss']==4
    assert v['fine_loss']==pytest.approx(.12)
    assert v['v_rms_base']==256


def test_unified_ignores_angles_and_scales_fine_loss():
    d,s,w,c=fixture();c.update(unified_winding_loss=True,loss_reference_width=500.)
    s[1,4]=.1;s[2,4]=.1
    d['phase_delta'][:]=1234.
    result,mask=measure(d,s,w,c)
    assert mask.all()
    for kind in ('parallel','perpendicular'):
        assert result['current'][kind]['u_loss']==pytest.approx(.08)
        assert result['current'][kind]['fine_loss']==pytest.approx(.04)

def test_integer_alignment_diagnostic_uses_fine_minus_integer():
    d,s,w,c=fixture();c.update(unified_winding_loss=True,integer_alignment_loss=True,loss_reference_width=1000.)
    s[1:,4]=.2  # shared phase: coarse delta .2
    s[1:,6]=.15  # centered across 3 buckets => bucket0 offset .1
    result,mask=measure(d,s,w,c)
    assert mask.all()
    for kind in ('parallel','perpendicular'):
        assert result['current'][kind]['u_error_l1']==pytest.approx(.3)
        assert result['current'][kind]['u_loss']==pytest.approx(.18)


def test_frozen_cohort_rejection_uses_best_nonbroken_not_last_bucket():
    d,s,w,c=fixture();_,cohort=measure(d,s,w,c)
    s[1,3]=20 # rejected; best nonbroken stays bucket0
    out,mask=measure(d,s,w,c,cohort=cohort)
    assert mask.tolist()==[False,True]
    assert out['fixed']['parallel']['events']==1
    assert out['fixed']['parallel']['retained_percent']==0
    assert out['fixed']['parallel']['fine_loss']==0
    assert out['current']['parallel']['fine_loss'] is None


def test_sign_and_distance_gate_and_unclipped_v_rms():
    d,s,w,c=fixture();_,cohort=measure(d,s,w,c)
    s[1,4]=-.01 # slight wrong sign, even though distance error<.5
    s[2,4]=.6 # right sign but distance error>.5
    offsets=np.zeros((4,2));offsets[2:,1]=2560
    out,mask=measure(d,s,w,c,offsets,cohort)
    assert not mask.any()
    assert out['fixed']['parallel']['v_loss']==100
    assert out['fixed']['parallel']['v_rms_base']==2560
    assert out['fixed']['parallel']['u_error_l1']==pytest.approx(.01)


@pytest.mark.parametrize('enabled,saved',[(True,False),(False,True)])
def test_saved_fields_fail_closed(tmp_path,enabled,saved):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_inlier_metrics import score_saved
    np.savez(tmp_path/'data.npz',dummy=np.zeros(1))
    np.savez(tmp_path/'solution.npz',**({'event_uv_offsets_base':np.zeros((2,2))} if saved else {}))
    with pytest.raises(ValueError,match='offset fields'):
        score_saved(dict(tensor_cache=str(tmp_path/'data.npz'),output_root=str(tmp_path),event_uv_offsets=enabled))


def test_standalone_cohort_identity_guard(tmp_path):
    import json
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_inlier_metrics import run
    paths=[]
    for index in range(2):
        data=tmp_path/f'data{index}.npz';np.savez(data,event_ids=np.array([index]))
        cfg=tmp_path/f'cfg{index}.json';cfg.write_text(json.dumps(dict(fiber_uv_layout=dict(tensor_cache=str(data)))))
        paths.append(str(cfg))
    with pytest.raises(ValueError,match='identical tensor'):
        run(dict(configs=paths))
