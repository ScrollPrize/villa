import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_bucket_uv import dynamic_same_sheet_pairs
from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_confidence import confidence_gate


def test_dynamic_geometry_and_order():
    a=torch.tensor([[0.,1.,2.]])
    b=torch.tensor([[.1,1.1,2.1]])
    sign=torch.ones(1);family=torch.ones(1)
    _,ka,kb=dynamic_same_sheet_pairs(a,b,sign,family)
    assert ka.tolist()==kb.tolist()==[0,1,2]
    # A whole-winding movement changes correspondence immediately.
    _,ka,kb=dynamic_same_sheet_pairs(a,b+1,sign,family)
    assert ka.tolist()==[1,2] and kb.tolist()==[0,1]
    # V inside H is NOT the same sheet, even at nearly matching coordinates.
    assert not len(dynamic_same_sheet_pairs(a,b,-sign,family)[0])
    # Common translation leaves the relationship invariant.
    old=dynamic_same_sheet_pairs(a,b,sign,family)
    new=dynamic_same_sheet_pairs(a+10,b+10,sign,family)
    for x,y in zip(old,new):assert torch.equal(x,y)


def test_joint_gate_rejects_competing_alternatives():
    cfg=dict(geometry_confidence_low=.5,geometry_confidence_high=.6)
    assert confidence_gate(torch.tensor(.7*.7),cfg)==0
    assert confidence_gate(torch.tensor(.9*.8),cfg)==1
    # Disjoint hypothesis pairs cannot both exceed half the joint mass.
    p=torch.tensor([.8,.2]);q=torch.tensor([.7,.3])
    assert (confidence_gate(p[:,None]*q[None,:],cfg)>0).sum()==1


def test_dynamic_loss_roundtrip_and_geometry_response():
    import numpy as np
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_bucket_uv import BucketUV
    d=dict(source_id=np.array(['00000000-0000-4000-8000-000000000011','00000000-0000-4000-8000-000000000012']),family=np.array([0,1]),start=np.zeros(2),end=np.full(2,300.),
           node_ds=np.array([[0,150,300],[0,150,300.]]),i=np.zeros(3,dtype=np.int64),
           j=np.ones(3,dtype=np.int64),ds_a=np.array([10.,100.,250.]),ds_b=np.array([20.,110.,260.]),
           parallel=np.zeros(3,dtype=bool),sign=np.ones(3),distance=np.full(3,.5))
    cfg=dict(sigma_max=1.,winding_count=10,bucket_uv_match_weight=10.,
             bucket_uv_attach_weight=.1,bucket_uv_bend_weight=1.,bucket_uv_lr=2.)
    td={k:torch.as_tensor(v) for k,v in d.items() if k!='source_id'};td['class_weight']=torch.ones(3)
    s=torch.tensor([[2.2,.1,0.,4.],[3.,-.85,0.,4.]],requires_grad=True)
    w=torch.full((10,),1000.,requires_grad=True)
    cfg.update(bucket_uv_dynamic_same_sheet=True,cavity_winding_sigma=False,winding_sigma=.2)
    ext=BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    assert ext.carry is None
    loss=sum(ext.terms(s,w,td,torch.arange(3)).values())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(s.grad).all()
    saved=ext.arrays()
    assert 'bucket_uv_carry' not in saved
    restored=BucketUV(d,cfg,'cpu');restored.restore(saved)
    with torch.no_grad():
        a=ext.terms(s,w,td,torch.arange(3))
        b=restored.terms(s,w,td,torch.arange(3))
    for key in a:torch.testing.assert_close(a[key],b[key])
