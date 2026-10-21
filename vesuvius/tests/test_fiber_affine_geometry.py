import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_affine_geometry import AffineOffsets, direction_cone
from vesuvius.neural_tracing.fiber_trace_3d import fiber_bucket_uv as b
from vesuvius.neural_tracing.fiber_trace_3d import fiber_intermediate_uv as mid
from .test_fiber_intermediate_uv import setup as baseline_setup, costs


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_packed_width_mapping_extrapolation_and_gradients(dtype):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_cache import geometry_evaluation,width_boundaries
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_grouped_gather import width_gather_mode
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import q_to_u,u_to_q
    w=torch.tensor([3.,5.,8.],dtype=dtype,requires_grad=True)
    q=torch.tensor([[-2.3,.2,1.6],[2.4,3.2,7.1]],dtype=dtype,requires_grad=True)
    def run(mode):
        with geometry_evaluation(),width_gather_mode(mode):
            bounds=width_boundaries(w)
            u=q_to_u(q,w,bounds);back=u_to_q(u+1,w,bounds)
            loss=(u.square()+back.square()).sum()
            return u,back,torch.autograd.grad(loss,(q,w))
    a,b,ga=run('select');c,d,gb=run('packed')
    torch.testing.assert_close(a,c);torch.testing.assert_close(b,d)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)


def test_bucket_local_multiplier_does_not_change_other_groups():
    _,_,_,_,_,e,_=setup()
    before=e.optimizer_groups()
    e.config['bucket_local_lr_multiplier']=10
    after=e.optimizer_groups()
    assert after[0]['params'][0] is e.values
    assert after[0]['lr']==10*before[0]['lr']
    assert [g['lr'] for g in after[1:]]==[g['lr'] for g in before[1:]]


def setup():
    d,cfg,td,s,w,old=baseline_setup()
    cfg=dict(cfg,fiber_affine_geometry=True,bucket_uv_bend_geometry='absolute',
             intermediate_uv_bend_geometry='absolute',shared_direction_weight=10.,
             intermediate_direction_weight=10.,bucket_direction_weight=10.)
    e=b.BucketUV(d,cfg,'cpu');e.prepare_fields(d,{})
    e.update(s,w,td,0)
    return d,cfg,td,s,w,e,old


@pytest.mark.parametrize('interval',[1,5,10,20])
def test_periodic_batched_window_refresh(interval):
    d,cfg,td,s,w,e,_=setup()
    e.config['bucket_uv_refresh_interval']=interval
    e.config['affine_reuse_remap']=True
    e.intermediate.config['affine_reuse_remap']=True
    before=e.labels.clone()
    with torch.no_grad():s[:,0].add_(3)
    if interval>1:
        e.update(s,w,td,interval-1)
        torch.testing.assert_close(e.labels,before,rtol=0,atol=0)
        torch.testing.assert_close(e.intermediate.labels,before,rtol=0,atol=0)
    e.update(s,w,td,interval)
    assert not torch.equal(e.labels,before)
    torch.testing.assert_close(e.intermediate.labels,e.labels,rtol=0,atol=0)


def test_initial_composition_and_plain_local_coordinates():
    d,cfg,td,s,w,e,old=setup()
    torch.testing.assert_close(e.curve.coordinates(),old.curve.coordinates(),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(e.positions(),old.positions(),atol=0,rtol=0)
    torch.testing.assert_close(e.intermediate.scaled_coordinates(),old.intermediate.scaled_coordinates(),atol=0,rtol=0)
    before=e.curve.coordinates().detach().clone()
    with torch.no_grad():e.curve.u[1]-=100.
    after=e.curve.coordinates().detach()
    # Local residual moves precisely one point; no cumsum/softplus/projection.
    expected=before.clone();expected[1,0]-=100
    torch.testing.assert_close(after,expected)
    assert after[1,0]<after[0,0]


@pytest.mark.parametrize('mapping_reuse',[False,True])
@pytest.mark.parametrize('batch_sides',[False,True])
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('fixed',[False,True])
@pytest.mark.parametrize('regularizers',[False,True])
def test_objective_geometry_cache_values_gradients_and_lifetime(mapping_reuse,batch_sides,packed,fixed,regularizers):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_cache import geometry_evaluation
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_grouped_gather import width_gather_mode
    d,cfg,td,s,w,e,_=setup()
    with torch.no_grad():
        e.affine.u[...,0].add_(.3)
        e.intermediate.affine.v[...,2].add_(.2)
    s=s.detach().requires_grad_();w=w.detach().requires_grad_()
    params=[s,w,*e.parameters()]
    def evaluate(enabled):
        cfg['batch_geometry_sides']=batch_sides and enabled
        e.curve.config['reuse_mapping_context']=mapping_reuse and enabled
        e.curve.config['fixed_shared_queries']=fixed and enabled
        e.curve.config['packed_endpoint_gathers']=packed and enabled
        e.config['packed_endpoint_gathers']=packed and enabled
        e.config['batch_layer_regularizers']=regularizers and enabled
        e.config['reuse_prior_conversion']=regularizers and enabled
        cfg['reuse_objective_geometry']=enabled
        e.intermediate.config['reuse_objective_geometry']=enabled
        with geometry_evaluation(enabled),width_gather_mode('packed' if packed and enabled else 'select'):
            terms=e.terms(s,w,td,torch.arange(3))
            pair=costs(cfg,td,s,w,e)
            loss=sum(terms.values())+sum(x.sum() for x in pair)
            if enabled:
                assert e.positions() is e.positions()
            return terms,torch.autograd.grad(loss,params,allow_unused=True)
    a,ga=evaluate(False);b,gb=evaluate(True)
    for key in a:torch.testing.assert_close(a[key],b[key],atol=1e-7,rtol=1e-5)
    for x,y in zip(ga,gb):
        if x is None:assert y is None
        else:torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    with geometry_evaluation():before=e.positions().detach().clone()
    with torch.no_grad():e.affine.u[...,0].add_(1)
    with geometry_evaluation():after=e.positions()
    torch.testing.assert_close(after[...,0],before[...,0]+1)


def test_fused_affine_algebra_and_gradients():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_kernels import affine_coordinates
    _,_,_,_,_,e,_=setup()
    a=e.affine
    with torch.no_grad():
        a.u.normal_();a.v.normal_();e.values.normal_()
    old=a.compose(e.values)
    new=affine_coordinates(a.base,e.values,a.u,a.v,a.owner,a.alpha)
    torch.testing.assert_close(old,new,rtol=1e-5,atol=1e-5)
    params=(e.values,a.u,a.v)
    go=torch.autograd.grad(old.square().mean(),params)
    gn=torch.autograd.grad(new.square().mean(),params)
    for x,y in zip(go,gn):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)


@pytest.mark.parametrize('reuse_hypotheses',[False,True])
@pytest.mark.parametrize('full_hypotheses',[False,True])
@pytest.mark.parametrize('group_results',[False,True])
def test_ragged_sigma_geometry_values_and_gradients(reuse_hypotheses,full_hypotheses,group_results):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_cache import geometry_evaluation
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as density
    d,cfg,td,s,w,e,_=setup()
    cfg.update(reuse_objective_geometry=True,batch_geometry_sides=True,pair_sigma_groups=True,
               soft_pair_weight=1.,probability_pair_weight=.3)
    e.intermediate.config['reuse_objective_geometry']=True
    s=s.detach().requires_grad_();w=w.detach().requires_grad_()
    sigma=torch.tensor([.2,.6,1.],dtype=s.dtype)
    def run(packed):
        with geometry_evaluation():
            edges=torch.tensor([2,0,1,0])
            a,b,da,db=e.pair_states(s,w,td,edges)
            geo=e.pair_geometry(s,w,edges)
            i,j=td['i'][edges],td['j'][edges]
            result=density.pair_cost(a,b,td['family'][i],td['family'][j],td['ds_a'][edges],td['ds_b'][edges],w,
                td['sign'][edges],td['distance'][edges],td['parallel'][edges],dict(cfg,batch_sigma_geometry=packed,reuse_prior_conversion=packed,reuse_packed_hypotheses=reuse_hypotheses,
                    batch_full_hypotheses=packed and full_hypotheses,batch_group_results=packed and group_results,batch_sigma_partition=packed),
                sigma_a=sigma[i],sigma_b=sigma[j],geometry=geo)
            grad=torch.autograd.grad(sum(x.sum() for x in result),[s,w,*e.parameters()],allow_unused=True)
            return result,grad
    old,go=run(False);new,gn=run(True)
    for x,y in zip(old,new):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    for x,y in zip(go,gn):
        if x is None:assert y is None
        else:torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)


def test_fiber_global_end_and_local_gradients_and_bucket_independence():
    d,cfg,td,s,w,e,_=setup();a=e.affine
    before=e.positions().detach().clone()
    col=2
    with torch.no_grad():
        a.u[0,col,0]=10.;a.v[0,col,1]=3.;a.v[0,col,2]=9.
        e.values[0,col,0]=2.
    delta=e.positions()-before
    owner=e.owner==0
    torch.testing.assert_close(delta[owner,col,1],3+6*a.alpha[owner],atol=2e-5,rtol=1e-5)
    assert delta[0,col,0]==12
    assert torch.all(delta[~owner]==0) and torch.all(delta[:,col+1]==0)
    e.positions()[owner,col].sum().backward()
    assert a.u.grad[0,col,0]==int(owner.sum())
    torch.testing.assert_close(a.v.grad[0,col,1],(1-a.alpha[owner]).sum())
    torch.testing.assert_close(a.v.grad[0,col,2],a.alpha[owner].sum())


@pytest.mark.parametrize('family',[0,1])
@pytest.mark.parametrize('sign',[-1.,1.])
def test_direction_cone_signed_axis_and_thresholds(family,sign):
    owner=torch.zeros(2,dtype=torch.long);arc=torch.tensor([0.,10.]);fam=torch.tensor([family]);direction=torch.tensor([sign])
    losses=[]
    for angle in [0.,5.,10.,20.,45.,90.,180.]:
        radians=np.deg2rad(angle)
        d=torch.tensor([10*np.cos(radians),10*np.sin(radians)],dtype=torch.float64)
        if family==1:d=d.flip(0)
        d[family]*=sign
        points=torch.stack((torch.zeros_like(d),d)).requires_grad_()
        loss,_=direction_cone(points,owner,arc,fam,direction,{})
        loss.backward();assert torch.isfinite(points.grad).all()
        losses.append(float(loss.detach()))
    assert max(losses[:3])<1e-25
    assert losses[4]==pytest.approx(1.)
    assert losses[3]<losses[4]<losses[5]<losses[6]


def test_direction_translation_invariance_and_active_mask():
    points=torch.tensor([[[0.,0.],[0.,0.]],[[10.,10.],[-10.,0.]]],dtype=torch.float64)
    args=(torch.zeros(2,dtype=torch.long),torch.tensor([0.,10.]),torch.tensor([0]),torch.tensor([1.]),{})
    mask=torch.tensor([[True,False]])
    a,_=direction_cone(points,*args,mask)
    z,_=direction_cone(points+torch.tensor([30000.,10000.]),*args,mask)
    torch.testing.assert_close(a,z);assert a==pytest.approx(1.)
    with torch.no_grad():
        _,stats=direction_cone(points,*args)
    assert stats['reversals']==1 and stats['segments']==2


def test_all_existing_losses_read_composed_positions():
    d,cfg,td,s,w,e,_=setup()
    with torch.no_grad():
        e.curve.affine.u[...,0]+=0.03;e.curve.affine.v[...,2]+=3.
        e.intermediate.affine.v[...,2]+=7.;e.affine.u[...,0]+=4.
    # Same composed geometry, represented without global/endpoint parameters.
    flat=b.BucketUV(d,cfg,'cpu');flat.prepare_fields(d,{})
    flat.update(s,w,td,0)
    flat.curve.set_coordinates(e.curve.coordinates().detach())
    flat.affine.initialize(e.positions().detach());flat.intermediate.affine.initialize(e.intermediate.scaled_coordinates().detach())
    a=e.terms(s,w,td,torch.arange(3));z=flat.terms(s,w,td,torch.arange(3))
    for k in a:torch.testing.assert_close(a[k],z[k],atol=1e-7,rtol=1e-5)
    for x,y in zip(costs(cfg,td,s,w,e),costs(cfg,td,s,w,flat)):
        torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    sum(a.values()).backward()
    for layer in [e.curve.affine,e.affine,e.intermediate.affine]:
        for p in (layer.u,layer.v):assert p.grad is not None and torch.isfinite(p.grad).all()


@pytest.mark.parametrize('reuse_window',[False,True])
def test_optimizer_initialization_remap_and_checkpoint(tmp_path,reuse_window):
    d,cfg,td,s,w,e,_=setup()
    cfg['reuse_window_geometry']=reuse_window
    # Real solve constructs optimizer BEFORE initial update; parameter replacement must work.
    e=b.BucketUV(d,cfg,'cpu');e.prepare_fields(d,{})
    opt=torch.optim.Adam(e.optimizer_groups());e.bind_optimizer(opt);e.update(s,w,td,0)
    sum(e.terms(s,w,td,torch.arange(3)).values()).backward();opt.step()
    saved_positions=e.positions().detach().clone();saved_mid=e.intermediate.scaled_coordinates().detach().clone()
    labels=e.labels.clone();base=e.affine.base.clone()
    coarse=e.affine.u.detach().clone();moment=opt.state[e.affine.u]['exp_avg'].clone()
    with torch.no_grad():s[:,0].add_(1.)
    e.update(s,w,td,1)
    for node in range(len(e.owner)):
        f=e.owner[node];label=labels[f,labels.shape[1]//2]
        oi=torch.where(labels[f]==label)[0];ni=torch.where(e.labels[f]==label)[0]
        torch.testing.assert_close(e.positions()[node,ni],saved_positions[node,oi])
        torch.testing.assert_close(e.intermediate.scaled_coordinates()[node,ni],saved_mid[node,oi])
        torch.testing.assert_close(e.affine.base[node,ni],base[node,oi])
        torch.testing.assert_close(e.affine.u[f,ni],coarse[f,oi])
        torch.testing.assert_close(opt.state[e.affine.u]['exp_avg'][f,ni],moment[f,oi])
    params=[p for g in opt.param_groups for p in g['params']]
    for layer in [e.curve.affine,e.affine,e.intermediate.affine]:
        assert any(p is layer.u for p in params) and any(p is layer.v for p in params)
    new=e.labels[:,-1];shared=e.shared_positions(s,w)
    for node in range(len(e.owner)):
        f=e.owner[node]
        if new[f]>labels[f,-1] and e.active[f,-1]:
            torch.testing.assert_close(e.positions()[node,-1],shared[node,-1])
            assert torch.all(e.affine.u[f,-1]==0)
    saved=dict(e.arrays(),state=s.detach().numpy(),widths=w.detach().numpy(),winding_sigma=e.cavity.values.numpy())
    np.savez(tmp_path/'affine.npz',**saved)
    with np.load(tmp_path/'affine.npz') as src:
        restored=mid.restore_model(d,src,cfg)
        torch.testing.assert_close(restored.positions(),e.positions())
        torch.testing.assert_close(restored.curve.coordinates(),e.curve.coordinates())
        torch.testing.assert_close(restored.intermediate.scaled_coordinates(),e.intermediate.scaled_coordinates())
    expected=e.intermediate.points(e,s,w,'a',s[td['i'],None,:])
    actual=mid.saved_points(d,saved,s.detach().numpy(),w.detach().numpy(),d['i'],d['ds_a'])
    for x,y in zip(actual,expected):np.testing.assert_allclose(x,y.detach()[:,0],atol=1e-5)


def test_no_smoothing_cost_for_physical_translation():
    d,cfg,td,s,w,e,_=setup();edges=torch.arange(3)
    before=e.terms(s,w,td,edges)
    with torch.no_grad():e.affine.u[...,0]+=100.;e.affine.v[...,0]+=200.
    after=e.terms(s,w,td,edges)
    torch.testing.assert_close(before['bucket_uv_bending'],after['bucket_uv_bending'],atol=1e-8,rtol=1e-5)
    torch.testing.assert_close(before['bucket_uv_direction'],after['bucket_uv_direction'],atol=1e-8,rtol=1e-5)


@pytest.mark.parametrize('fused',[False,True])
def test_inplace_remap_matches_replacement_and_preserves_adam_identity(fused):
    import copy
    owner=torch.tensor([0,0,1,1]);arc=torch.tensor([0.,1.,0.,1.])
    a=AffineOffsets(owner,arc,[0.,0.],[1.,1.],'cpu',torch.float32)
    initial=torch.arange(24,dtype=torch.float32).reshape(4,3,2)
    a.initialize(initial);b=copy.deepcopy(a)
    x=torch.nn.Parameter(torch.ones_like(initial));y=torch.nn.Parameter(x.detach().clone())
    oa=torch.optim.Adam([a.u,a.v,x],fused=fused);ob=torch.optim.Adam([b.u,b.v,y],fused=fused)
    for obj,opt,local in [(a,oa,x),(b,ob,y)]:
        obj.compose(local).square().sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    labels=torch.tensor([[0,1,2],[0,1,2]]);active=torch.ones_like(labels,dtype=torch.bool)
    for new_labels,new_active in [
            (torch.tensor([[1,2,3],[0,1,2]]),torch.tensor([[True,True,False],[True,True,True]])),
            (torch.tensor([[1,2,3],[-1,0,1]]),torch.tensor([[True,True,False],[True,True,True]]))]:
        identities=[id(p) for p in (b.u,b.v,y)]
        moment_ids=[id(ob.state[p]['exp_avg']) for p in (b.u,b.v,y)]
        initial=initial+10
        x=a.remap(x,initial,labels,active,new_labels,new_active,oa)
        y=b.remap(y,initial,labels,active,new_labels,new_active,ob,reuse=True)
        assert identities==[id(p) for p in (b.u,b.v,y)]
        assert moment_ids==[id(ob.state[p]['exp_avg']) for p in (b.u,b.v,y)]
        torch.testing.assert_close(a.base,b.base,atol=0,rtol=0)
        for p,q in zip((a.u,a.v,x),(b.u,b.v,y)):
            torch.testing.assert_close(p,q,atol=0,rtol=0)
            for key in oa.state[p]:torch.testing.assert_close(oa.state[p][key],ob.state[q][key],atol=0,rtol=0)
        labels,active=new_labels,new_active
    # Resizing still uses the original exact replacement path.
    wide=torch.tensor([[0,1,2,3,4],[-2,-1,0,1,2]])
    initial=torch.ones(4,5,2)
    x=a.remap(x,initial,labels,active,wide,torch.ones_like(wide,dtype=torch.bool),oa)
    y=b.remap(y,initial,labels,active,wide,torch.ones_like(wide,dtype=torch.bool),ob,reuse=True)
    torch.testing.assert_close(a.compose(x),b.compose(y),atol=0,rtol=0)
