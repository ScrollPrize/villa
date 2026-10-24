import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_density_sign_report import ordering
from vesuvius.neural_tracing.fiber_trace_3d.fiber_density_sign_report import gt_ordering


def test_gt_residual_inversion_order_and_ties():
    events=[]
    for displacement,family,parallel,sign in [(0,1,False,1),(0,1,False,-1),(0,0,True,1),(-.6,0,True,1),(.6,0,True,1)]:
        measured=sign*.73
        events.append(dict(residual=displacement+.5*family-measured,
            signed_distance=measured,family_delta=family,parallel=parallel,normal_cosine=sign*.8))
    np.testing.assert_allclose(gt_ordering(events,{}),[.5,-.5,0,-.6,.6])
    np.testing.assert_allclose(gt_ordering(events,{'perpendicular_offset':1.7}),gt_ordering(events,{}))


def test_perp_calibration_and_parallel_zero_ties():
    s=torch.tensor([[2.,.2,0.,4.],[2.,.2,0.,4.],[1.,.2,0.,4.]])
    d=dict(i=np.array([0,0,0,0]),j=np.array([1,1,2,2]),family=np.array([0,1,0]),
           ds_a=np.zeros(4,dtype=np.float32),ds_b=np.zeros(4,dtype=np.float32),
           parallel=np.array([False,False,True,True]),sign=np.array([1,-1,1,-1]))
    assert torch.allclose(ordering(s,torch.ones(10)*100,d,{}),torch.tensor([.5,-.5,-1.,1.]))
    s[2,0]=2
    assert ordering(s,torch.ones(10)*100,d,{})[2]==0


def test_continuous_phase_is_included_without_rounding():
    s=torch.tensor([[2.,.9,0.,4.],[2.,.1,0.,4.]])
    d=dict(i=np.array([0]),j=np.array([1]),family=np.array([0,0]),ds_a=np.zeros(1),ds_b=np.zeros(1),parallel=np.array([True]),sign=np.array([1]))
    np.testing.assert_allclose(ordering(s,torch.ones(10)*100,d,{}).item(),-.8,atol=1e-7)


def test_saved_cavity_sigma_changes_tails_and_confidence(tmp_path):
    import json
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_density_sign_report as audit
    state=np.array([[2.,.2,0.,4.],[2.,.2,0.,4.]],dtype=np.float32)
    d=dict(i=np.array([0]),j=np.array([1]),family=np.array([0,1]),ds_a=np.zeros(1,dtype=np.float32),
        ds_b=np.zeros(1,dtype=np.float32),parallel=np.array([False]),sign=np.array([1.],dtype=np.float32),
        source_id=np.array([10,11]),start=np.zeros(2),end=np.ones(2))
    cache=tmp_path/'tensors.npz';np.savez(cache,**d)
    root=tmp_path/'solve';root.mkdir()
    np.savez(root/'solution.npz',state=state,widths=np.ones(10,dtype=np.float32)*100,
        winding_sigma=np.array([2.,2.],dtype=np.float32),**{k:d[k] for k in ('source_id','start','end','family')})
    cfg=dict(tensor_cache=str(cache),winding_sigma=.5,sigma_max=2.,cavity_winding_sigma=True)
    (root/'config.json').write_text(json.dumps(cfg))
    wide=audit.run(dict(solve_root=str(root),output_root=str(tmp_path/'wide')))
    cfg.pop('cavity_winding_sigma');(root/'config.json').write_text(json.dumps(cfg))
    fixed=audit.run(dict(solve_root=str(root),output_root=str(tmp_path/'fixed')))
    assert wide['sigma_mode']=='cavity' and wide['sigma_support_max']==2
    assert wide['rows'][0]['mean_wrong_probability']>fixed['rows'][0]['mean_wrong_probability']
    assert wide['rows'][0]['violations']==fixed['rows'][0]['violations']==0
    assert wide['deformation_meetings']==0 and fixed['deformation_meetings']==1
    assert next(r for r in wide['rows'] if r['cohort']=='both_joint_above_half')['signed_events']==0
    assert next(r for r in fixed['rows'] if r['cohort']=='both_joint_above_half')['signed_events']==1
