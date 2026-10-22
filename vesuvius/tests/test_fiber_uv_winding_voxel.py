"""Unified phase/winding geometry, using the incumbent quadratic objective."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_arc_phase as geometry

CFG=dict(bucket_position_split=True,bucket_shared_phase=True,layout_mode='shared_anchor',
         distance_objective='squared',distance_sigma=.5,distance_weight=.25,
         position_distance_weight=2.,sign_weight=20.,metric_weight=.02)

def test_quadratic_baseline_value_gradient_and_no_local_angle():
    torch.manual_seed(8)
    k=3
    sa=torch.tensor([[.2],[.4],[.9]],dtype=torch.float64,requires_grad=True)
    sb=torch.tensor([[.8],[-.1],[.1]],dtype=torch.float64,requires_grad=True)
    a=torch.arange(k,dtype=torch.float64)[None,:]+sa
    b=torch.arange(k,dtype=torch.float64)[None,:]+sb
    pa=torch.full_like(a,.2,requires_grad=True);pb=torch.full_like(b,.25,requires_grad=True)
    va=torch.randn_like(a,requires_grad=True);vb=torch.randn_like(b,requires_grad=True)
    fd=torch.tensor([1.,-1.,0.],dtype=torch.float64)
    sign=torch.tensor([1.,-1.,0.],dtype=torch.float64)
    mu=torch.tensor([.3,1.2,.8],dtype=torch.float64)
    widths=torch.full((k,),8000.,dtype=torch.float64,requires_grad=True)
    charge=torch.tensor([100.,80.,50.],dtype=torch.float64)
    weights=torch.tensor([1.,2.,.5],dtype=torch.float64)
    args=(pa,pb,a,b,va,vb,torch.zeros(3,dtype=torch.float64),fd,sign,mu,None)
    cfg=dict(CFG,unified_winding_loss=True,loss_reference_width=8000.)
    got=uv.pair_objective(*args,cfg,charge=charge,event_weights=weights,coarse=(a,b,widths))
    ref=uv.pair_objective(*args,CFG,charge=charge,event_weights=weights,bucket_shift=b[:,0]-a[:,0])
    torch.testing.assert_close(got,ref,atol=1e-12,rtol=1e-12)
    variables=(sa,sb,pa,pb,va,vb,widths)
    gg=torch.autograd.grad(got,variables,allow_unused=True,retain_graph=True)
    gr=torch.autograd.grad(ref,variables,allow_unused=True,retain_graph=True)
    for x,y in zip(gg,gr):
        if y is None:assert x is None
        else:torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
    changed=list(args);changed[6]=torch.full((3,),1234.,dtype=torch.float64)
    torch.testing.assert_close(got,uv.pair_objective(*changed,cfg,charge=charge,event_weights=weights,coarse=(a,b,widths)))
    compiled=torch.compile(uv.pair_objective,backend='aot_eager',fullgraph=True)
    torch.testing.assert_close(compiled(*args,cfg,charge=charge,event_weights=weights,coarse=(a,b,widths)),got)

def test_width_lookup_and_live_arc_gradient():
    w=torch.tensor([100.,200.,400.],dtype=torch.float64,requires_grad=True)
    a=torch.tensor([[-1.,1.2]],dtype=torch.float64);b=torch.tensor([[.5,4.]],dtype=torch.float64)
    scale=geometry.local_width(a,b,w)
    torch.testing.assert_close(scale,torch.tensor([[[100.,250.],[150.,300.]]],dtype=torch.float64))
    assert not scale.requires_grad
    bounds=torch.cat((w.new_zeros(1),w.cumsum(0)))
    q=geometry.coarse_coordinates(torch.zeros(1,12,dtype=torch.float64),torch.tensor([.5]),torch.tensor([0]),torch.tensor([100.]),w,bounds)
    q.sum().backward();assert w.grad.abs().sum()>0

def test_continuation_uniform_width_equivalence():
    q=torch.tensor([[.2,1.2]],dtype=torch.float64);p=torch.tensor([[.3,.6]],dtype=torch.float64)
    va=torch.zeros_like(q);vb=va+20.;w=torch.full((2,),8000.,dtype=torch.float64)
    torch.testing.assert_close(geometry.continuation(p,p,q,q+.01,va,vb,w,128.),
                              uv.continuation_likelihood(p,p,q*8000,(q+.01)*8000,va,vb,128.))

@pytest.mark.parametrize('flag',['arc_phase_constraints','integer_sheet_constraints','winding_voxel_loss','coarse_phase_sigma'])
def test_superseded_optimization_rejected(flag):
    with pytest.raises(ValueError,match='Superseded'):uv.reject_superseded_objectives({flag:True})

def test_configuration_preserves_tuned_settings():
    baseline=dict(distance_sigma=.5,position_distance_weight=2,lr=.07348469228349533,width_lr=2)
    cfg=dict(baseline,unified_winding_loss=True,loss_reference_width=10000)
    allowed={'code_provenance','output_root','gpu_manager','log_every','benchmark_log_every','gt_metrics','unified_winding_loss','loss_reference_width'}
    assert {k:v for k,v in cfg.items() if k not in allowed}==baseline
    assert cfg['distance_sigma']==.5
    assert cfg['position_distance_weight']==2
    assert cfg['lr']==.07348469228349533 and cfg['width_lr']==2
    for key in ('arc_phase_constraints','integer_sheet_constraints','winding_voxel_loss','coarse_phase_sigma'):
        assert key not in cfg
    uv.reject_superseded_objectives(cfg)

def test_unified_gt_continuous_measurement():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_constraint_score import event_residuals
    d=dict(phase=np.array([.25]),family=np.array([0]),start=np.array([10.]),end=np.array([110.]),node_ds=np.array([[0.,50.,100.]]))
    state=np.zeros((1,12));w=np.array([100.,200.,400.]);pick=np.array([0])
    e=dict(test_piece=0,gt_piece=0,test_arc=60.,gt_arc=60.,signed_distance=.3,family_delta=1.,parallel=False)
    value=event_residuals([e],d,state,pick,3,d,dict(state=state,widths=w),pick,dict(unified_winding_loss=True),{},w)
    np.testing.assert_allclose(value,.2)
