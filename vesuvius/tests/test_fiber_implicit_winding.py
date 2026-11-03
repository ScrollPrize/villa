import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as imp
from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft


def test_order_magnitude_is_continuous_at_boundary():
    margin=torch.tensor([[[-1e-6,0.,1e-6]]],dtype=torch.float64,requires_grad=True)
    la=torch.zeros((1,1),dtype=torch.float64)
    lb=torch.full((1,3),-__import__('math').log(3),dtype=torch.float64)
    cfg=dict(sign_violation_weight=30.,unified_order_loss='margin_probability')
    q,p=imp.order_pair_terms(margin,la,lb,torch.ones(1),cfg)
    torch.testing.assert_close(q,p)
    assert q.item()==pytest.approx(1e-11)
    g,=torch.autograd.grad(q.sum(),margin)
    assert g[0,0,1:].abs().sum()==0
    assert abs(g[0,0,0])<1e-4
    assert not hasattr(imp,'sign_probability_cost')
    assert not hasattr(imp,'wrong_order_mass')


def test_gt_sign_marginalizes_integer_hypotheses():
    a=torch.tensor([[5.4,.3,0.,100.]],requires_grad=True)
    b=torch.tensor([[4.6,.3,0.,100.]],requires_grad=True)
    cfg=dict(gt_pair_mode='sign_only',winding_sigma=.5,sign_violation_weight=30.,
             soft_pair_weight=1.,probability_pair_weight=0.)
    cost=imp.pair_cost(a,b,torch.tensor([0]),torch.tensor([1]),torch.zeros(1),
        torch.zeros(1),torch.full((20,),1000.),torch.ones(1),torch.zeros(1),
        torch.tensor([False]),cfg)[0]
    ha,la=imp.hypotheses(a,.5);hb,lb=imp.hypotheses(b,.5)
    delta=hb[:,None,:,0]-ha[:,:,None,0]
    expected=30*((la.exp()[:,:,None]*lb.exp()[:,None,:])*(-delta-.5).clamp_min(0).square()).sum((1,2))
    torch.testing.assert_close(cost,expected,atol=1e-5,rtol=1e-5)
    ga,gb=torch.autograd.grad(cost.sum(),(a,b))
    assert ga[0,0]>0 and gb[0,0]<0


def test_geometry_performance_defaults_preserve_scientific_config():
    original=dict(fiber_affine_geometry=True,shared_curve_geometry=True,
                  intermediate_uv_geometry=True,lr=.17,steps=1000,batch_size=32768,
                  pair_sigma_groups=False,bucket_uv_refresh_interval=5)
    result=imp.performance_defaults(original)
    assert all(result[k]==v for k,v in original.items())
    assert result['fused_adam']=='auto'
    assert result['batch_full_hypotheses'] and result['batch_group_results']
    assert result['fixed_shared_queries'] and result['reuse_window_geometry']
    assert not result['batch_sigma_partition']
    assert not result['batched_affine_backward']
    assert not result['batched_interpolation_backward']
    assert 'fused_adam' not in original
    override=dict(original,fused_adam=False,batch_group_results=False)
    resolved=imp.performance_defaults(override)
    assert resolved['fused_adam'] is False and resolved['batch_group_results'] is False
    disabled=dict(original,geometry_performance_defaults=False)
    assert imp.performance_defaults(disabled)==disabled
    historical=dict(original,intermediate_uv_geometry=False)
    assert imp.performance_defaults(historical)==historical


def inputs():
    a=torch.tensor([[2.2,.8,20.,4.],[3.5,-.2,30.,4.]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[3.1,.9,24.,4.],[2.4,.1,36.,4.]],dtype=torch.float64,requires_grad=True)
    fa=torch.tensor([0,1]);fb=1-fa
    return a,b,fa,fb,torch.tensor([13.,-5.]),torch.tensor([-8.,17.]),torch.tensor([80.,100.,120.,130.,150.,160.]),torch.tensor([1.,-1.]),torch.tensor([1.5,1.]),torch.tensor([False,False])


def test_lasagna_scale_changes_measurement_once_not_sign_geometry():
    args=list(inputs())
    cfg=dict(winding_sigma=.5,distance_sigma=.5,soft_pair_weight=1.,probability_pair_weight=.3,
             sign_violation_weight=30.,unified_order_loss='margin_probability')
    scaled=imp.pair_cost(*args,dict(cfg,lasagna_winding_scale=1.7))
    measured=list(args);measured[8]=measured[8]*1.7
    expected=imp.pair_cost(*measured,cfg)
    for a,b in zip(scaled,expected):torch.testing.assert_close(a,b)
    sign=imp.pair_cost(*args,dict(cfg,gt_pair_mode='sign_only'))
    changed=imp.pair_cost(*args,dict(cfg,gt_pair_mode='sign_only',lasagna_winding_scale=3.,perpendicular_offset=.1))
    for a,b in zip(sign,changed):torch.testing.assert_close(a,b)


@pytest.mark.parametrize('mode',['margin_probability','one_sided_distance'])
@pytest.mark.parametrize('parallel',[False,True])
def test_unified_full_is_order_plus_additional_losses_and_gradients(mode,parallel):
    args=list(inputs());args[6]=args[6].double().requires_grad_()
    if parallel:args[3]=args[2];args[9]=torch.ones(2,dtype=torch.bool)
    cfg=dict(winding_sigma=.5,distance_sigma=.5,soft_pair_weight=1.,probability_pair_weight=.3,
             sign_violation_weight=30.,sign_probability_threshold=0.,unified_order_loss=mode)
    full=imp.pair_cost(*args,cfg)
    order=imp.pair_cost(*args,dict(cfg,gt_pair_mode='sign_only'))
    extra=imp.pair_cost(*args,dict(cfg,unified_order_loss=None,sign_violation_weight=0.))
    for f,o,e in zip(full,order,extra):torch.testing.assert_close(f,o+e)
    params=(args[0],args[1],args[6])
    gf=torch.autograd.grad(sum(x.sum() for x in full),params)
    go=torch.autograd.grad(sum((o+e).sum() for o,e in zip(order,extra)),params)
    for a,b in zip(gf,go):torch.testing.assert_close(a,b)
    changed=list(args);changed[8]=args[8]*100
    invariant=imp.pair_cost(*changed,dict(cfg,gt_pair_mode='sign_only',perpendicular_offset=123.))
    for a,b in zip(order,invariant):torch.testing.assert_close(a,b)


def test_unified_margin_preserves_successful_sign_objective():
    args=inputs();cfg=dict(gt_pair_mode='sign_only',winding_sigma=.5,
        sign_violation_weight=30.,sign_probability_threshold=0.,soft_pair_weight=1.,probability_pair_weight=.3)
    old=imp.pair_cost(*args,cfg)
    new=imp.pair_cost(*args,dict(cfg,unified_order_loss='margin_probability'))
    for a,b in zip(old,new):torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('mode',['margin_probability','one_sided_distance'])
def test_unified_order_is_one_sided_and_moves_wrong_position(mode):
    margin=torch.tensor([[[-2.]],[[1.]]],dtype=torch.float64,requires_grad=True)
    logp=torch.zeros(2,1,dtype=torch.float64)
    soft_cost,prob=imp.order_pair_terms(margin,logp,logp,torch.ones(2),
        dict(unified_order_loss=mode,sign_violation_weight=30.,distance_sigma=.5))
    assert soft_cost[1]==0 and prob[1]==0
    g=torch.autograd.grad((soft_cost+prob).sum(),margin)[0]
    assert g[0]<0 and g[1]==0
    unknown=imp.order_pair_terms(margin,logp,logp,torch.zeros(2),
        dict(unified_order_loss=mode,sign_violation_weight=30.))
    assert all(torch.equal(x,torch.zeros_like(x)) for x in unknown)


@pytest.mark.parametrize('shift',[0.,20.,-20.])
@pytest.mark.parametrize('unsigned',[False,True])
def test_grouped_support_matches_full_and_gradients(shift,unsigned):
    args=list(inputs());args[6]=args[6].double().requires_grad_()
    with torch.no_grad():args[1][:,0]+=shift
    if unsigned:args[7]=torch.zeros(2)
    cfg=dict(winding_sigma=.5,sigma_max=8.,soft_pair_weight=1.,probability_pair_weight=.3,
             sign_violation_weight=30.,same_sheet_match_multiplier=2.5)
    sigma=torch.tensor([.2,1.1],dtype=torch.float64)
    full=imp.pair_cost(*args,cfg,sigma_a=sigma,sigma_b=sigma)
    small=imp.pair_cost(*args,dict(cfg,pair_sigma_groups=True,pair_residual_full_support=True),sigma_a=sigma,sigma_b=sigma)
    for a,b in zip(full,small):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    ga=torch.autograd.grad(sum(x.sum() for x in full),(args[0],args[1],args[6]))
    gb=torch.autograd.grad(sum(x.sum() for x in small),(args[0],args[1],args[6]))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)


def test_default_grouping_never_checks_residual(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Residual-dependent support must be explicitly enabled')
    monkeypatch.setattr(imp,'residual_support_radius',forbidden)
    args=list(inputs())
    cfg=dict(winding_sigma=.5,sigma_max=8.,pair_sigma_groups=True)
    sigma=torch.full((2,),.5,dtype=args[0].dtype)
    result=imp.pair_cost(*args,cfg,sigma_a=sigma,sigma_b=sigma)
    assert all(torch.isfinite(x).all() for x in result)
    with pytest.raises(AssertionError,match='explicitly enabled'):
        imp.pair_cost(*args,dict(cfg,pair_residual_full_support=True),sigma_a=sigma,sigma_b=sigma)


def test_reductions_compiled_autograd_matches_eager():
    torch.manual_seed(0)
    pred=torch.randn(3,5,5,requires_grad=True)
    la=torch.randn(3,5,requires_grad=True);lb=torch.randn(3,5,requires_grad=True)
    phase=torch.randn(3,5,5,requires_grad=True)
    va=torch.randn(3,5,requires_grad=True);vb=torch.randn(3,5,requires_grad=True)
    inputs=(pred,torch.ones(3),la,lb,torch.tensor([1.,0.,-1.]),pred,
            torch.tensor([True,False,True]),phase,va,vb,.5,1000.,2.5,pred.detach())
    eager=imp.pair_reductions(*inputs)
    compiled=torch.compile(imp.pair_reductions,backend='aot_eager',fullgraph=True)(*inputs)
    for a,b in zip(eager,compiled):torch.testing.assert_close(a,b)
    params=(pred,la,lb,phase,va,vb)
    a=torch.autograd.grad(sum(t.sum() for t in eager),params)
    b=torch.autograd.grad(sum(t.sum() for t in compiled),params)
    for x,y in zip(a,b):torch.testing.assert_close(x,y)


def test_integer_mixture_has_no_phase_carry_jump_in_either_branch():
    def cost(phase,soft_weight):
        sa=torch.tensor([[0.,0.,0.,4.]],dtype=torch.float64)
        sb=torch.stack((phase*0,phase,phase*0,phase*0+4))[None,:]
        cfg=dict(winding_sigma=.5,soft_pair_weight=soft_weight,probability_pair_weight=1-soft_weight,
                 distance_sigma=.5,same_sheet_match_multiplier=2.5)
        return imp.pair_cost(sa,sb,torch.tensor([0]),torch.tensor([1]),
            torch.zeros(1,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),
            torch.full((20,),1000.,dtype=torch.float64),torch.ones(1),torch.full((1,),.5),
            torch.tensor([False]),cfg)
    phi=torch.tensor(.5,dtype=torch.float64,requires_grad=True)
    for weight in (0.,1.):
        assert torch.autograd.gradcheck(lambda p:cost(p,weight)[0],(phi,),eps=1e-6,atol=1e-5)
        lo=cost(torch.tensor(.5-1e-7,dtype=torch.float64),weight)[0]
        hi=cost(torch.tensor(.5+1e-7,dtype=torch.float64),weight)[0]
        assert abs(float(hi-lo))<1e-5


def test_integer_evidence_targets_not_fractional_measurement():
    delta=torch.tensor([1.,1.5,2.],dtype=torch.float64,requires_grad=True)
    energy,_=soft.integer_separation_energy(delta,torch.ones(3),torch.full((3,),1.5),torch.zeros(3),.25)
    torch.testing.assert_close(energy[0],energy[2])
    assert energy[1]>energy[0]
    assert torch.autograd.gradcheck(lambda d:soft.integer_separation_energy(d,torch.ones(3),torch.full((3,),1.5),torch.zeros(3),.25)[0],(delta,))


def test_full_pair_combination():
    args=inputs();cfg=dict(winding_sigma=.5,distance_sigma=.25,v_pair_scale=100.,
        same_sheet_match_multiplier=2.5,sign_violation_weight=10.,sign_probability_threshold=.1)
    density=imp.pair_cost(*args,cfg)
    errors=soft.pair_errors(*args)
    qa,_,_=soft.points(args[0],args[2],args[4],args[6]);qb,_,_=soft.points(args[1],args[3],args[5],args[6])
    energy,_=soft.integer_separation_energy(qb-qa,args[7],args[8],(~args[9])*(args[3]-args[2])*.5,.25)
    continuous=(energy,errors[1].square(),(errors[2]/100.).square())
    pure=imp.pair_cost(*args,dict(cfg,soft_pair_weight=1,probability_pair_weight=0))
    mixed=imp.pair_cost(*args,dict(cfg,soft_pair_weight=1,probability_pair_weight=.3))
    default=imp.pair_cost(*args,dict(cfg,soft_pair_weight=0,probability_pair_weight=1))
    for c,p,m,d,b in zip(continuous,pure,mixed,default,density):
        torch.testing.assert_close(c,p)
        torch.testing.assert_close(m,c+.3*b)
        assert torch.equal(d,b)
    sum(x.sum() for x in mixed).backward()
    assert torch.isfinite(args[0].grad).all() and torch.isfinite(args[1].grad).all()


def test_fixed_reference_combined_pair_and_cavity_sigma():
    args=list(inputs());args[1]=args[1].detach();args[5]=torch.zeros(2)
    cfg=dict(winding_sigma=.5,distance_sigma=.25,v_pair_scale=100.,sigma_max=2.)
    sigma=torch.tensor([.2,1.2],dtype=torch.float64)
    base=imp.pair_cost(*args,cfg,fixed_b=True,sigma_a=sigma)
    errors=soft.pair_errors(*args)
    mixed=imp.pair_cost(*args,dict(cfg,soft_pair_weight=1,probability_pair_weight=.3),fixed_b=True,sigma_a=sigma)
    for m,b,c in zip(mixed,base,(soft.pair_integer_energy(*args,sigma=.25),errors[1]**2,(errors[2]/100)**2)):
        torch.testing.assert_close(m,c+.3*b)
    sum(x.sum() for x in mixed).backward()
    assert args[1].grad is None
    assert torch.isfinite(args[0].grad).all() and args[0].grad.abs().sum()>0


def test_full_pair_learned_sigma_gradients():
    args=inputs();cfg=dict(winding_sigma=.5,soft_pair_weight=1,probability_pair_weight=.3,sigma_max=1.)
    sa=torch.full((2,),.5,dtype=torch.float64,requires_grad=True)
    sb=torch.full((2,),.5,dtype=torch.float64,requires_grad=True)
    costs=imp.pair_cost(*args,cfg,sigma_a=sa,sigma_b=sb)
    sum(x.sum() for x in costs).backward()
    for g in (sa.grad,sb.grad,args[0].grad,args[1].grad):
        assert torch.isfinite(g).all() and g.abs().sum()>0


def test_two_stage_pair_schedule():
    schedule=dict(soft_start=1,soft_end=.1,probability_start=0,probability_end=.4,probability_ramp_fraction=.5)
    cfg=dict(steps=1000,pair_weight_schedule=schedule)
    for iteration,expected in [(0,(1,0)),(250,(1,.2)),(500,(1,.4)),(750,(.55,.4)),(1000,(.1,.4))]:
        assert imp.full_pair_weights(dict(cfg,_optimization_iteration=iteration))==pytest.approx(expected)
    for fraction in (0,1,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):imp.full_pair_weights(dict(cfg,pair_weight_schedule=dict(schedule,probability_ramp_fraction=fraction)))


def test_full_pair_schedule():
    cfg=dict(steps=1000,pair_weight_schedule=dict(soft_start=1,soft_end=0,probability_start=0,probability_end=1))
    assert imp.full_pair_weights(cfg)==(1,0)
    assert imp.full_pair_weights(dict(cfg,_optimization_iteration=500))==(.5,.5)
    assert imp.full_pair_weights(dict(cfg,_optimization_iteration=1000))==(0,1)
    for bad in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):imp.full_pair_weights(dict(soft_pair_weight=bad))
    with pytest.raises(ValueError):imp.full_pair_weights(dict(soft_pair_weight=0,probability_pair_weight=0))
    with pytest.raises(ValueError):imp.full_pair_weights(dict(cfg,soft_pair_weight=1))
    with pytest.raises(ValueError):imp.full_pair_weights(dict(soft_pair_weight=1,additive_quadratic_weight=.1))


def test_additive_quadratic_guidance():
    a,b,*rest=inputs()
    cfg=dict(winding_sigma=.5,distance_sigma=.25,sign_violation_weight=10.,sign_probability_threshold=.1)
    base=imp.pair_cost(a,b,*rest,cfg)
    zero=imp.pair_cost(a,b,*rest,dict(cfg,additive_quadratic_weight=0))
    for x,y in zip(base,zero):assert torch.equal(x,y)
    mixed=imp.pair_cost(a,b,*rest,dict(cfg,additive_quadratic_weight=.3))
    error,_,_=soft.pair_errors(a,b,*rest)
    torch.testing.assert_close(mixed[0],base[0]+.3*soft.pair_integer_energy(a,b,*rest,sigma=.25))
    for x,y in zip(mixed[1:],base[1:]):assert torch.equal(x,y)
    mixed[0].sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad[:,0].abs().sum()>0
    for bad in [-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):imp.additive_quadratic_weight({'additive_quadratic_weight':bad})
    with pytest.raises(ValueError):
        imp.additive_quadratic_weight(dict(adaptive_quadratic_blend=True,additive_quadratic_weight=.1))
    with pytest.raises(ValueError):
        imp.pair_cost(a,b,*rest,dict(cfg,additive_quadratic_weight=.1),fixed_b=True)


def test_adaptive_blend_controller():
    s=torch.tensor([.1,.25,.625,1.,4.],requires_grad=True)
    alpha=imp.quadratic_blend_alpha(s,s,{})
    torch.testing.assert_close(alpha,torch.tensor([0.,0.,.45,.9,.9]))
    assert not alpha.requires_grad
    for c in [{'quadratic_blend_sigma_high':.25},{'quadratic_blend_max':1},
              {'quadratic_blend_sigma_low':float('nan')}]:
        with pytest.raises(ValueError):imp.quadratic_blend_alpha(s,s,c)


def test_adaptive_blend_loss_and_sign_unchanged():
    a,b,*rest=inputs()
    sigma=torch.tensor([.4,.8],dtype=a.dtype,requires_grad=True)
    cfg=dict(winding_sigma=.5,sigma_max=1.,sign_violation_weight=10.,sign_probability_threshold=.1)
    kw=dict(sigma_a=sigma,sigma_b=sigma)
    plain=imp.pair_cost(a,b,*rest,cfg,**kw)
    unsigned=imp.pair_cost(a,b,*rest,dict(cfg,sign_violation_weight=0),**kw)
    diag={}
    mixed=imp.pair_cost(a,b,*rest,dict(cfg,adaptive_quadratic_blend=True),diagnostics=diag,**kw)
    error,_,_=soft.pair_errors(a,b,*rest)
    alpha=imp.quadratic_blend_alpha(sigma,sigma,cfg)
    expected=alpha*soft.pair_integer_energy(a,b,*rest)+(1-alpha)*unsigned[0]+plain[0]-unsigned[0]
    torch.testing.assert_close(mixed[0],expected)
    for x,y in zip(mixed[1:],plain[1:]):assert torch.equal(x,y)
    zero=imp.pair_cost(a,b,*rest,dict(cfg,adaptive_quadratic_blend=True,quadratic_blend_max=0),**kw)
    for x,y in zip(zero,plain):assert torch.equal(x,y)
    sum(x.sum() for x in mixed).backward()
    assert torch.isfinite(a.grad).all() and a.grad[:,0].abs().sum()>0
    assert torch.isfinite(sigma.grad).all() and sigma.grad.abs().sum()>0
    with pytest.raises(ValueError):
        imp.pair_cost(a,b,*rest,dict(cfg,adaptive_quadratic_blend=True),fixed_b=True)
    fixedcfg=dict(cfg,adaptive_quadratic_blend=True,sigma_support_max=1.)
    fixed=imp.pair_cost(a,b,*rest,fixedcfg)
    constant=torch.full((len(a),),.5,dtype=a.dtype)
    variable=imp.pair_cost(a,b,*rest,fixedcfg,sigma_a=constant,sigma_b=constant)
    for x,y in zip(fixed,variable):torch.testing.assert_close(x,y)


def test_hard_wrong_order_mass_is_diagnostic_only():
    a,b,*rest=inputs()
    cfg={'winding_sigma':.5}
    base=imp.pair_cost(a,b,*rest,cfg)
    diag={}
    # Historical charge configuration cannot re-enable a removed loss.
    observed=imp.pair_cost(a,b,*rest,dict(cfg,sign_violation_weight=10,
        sign_probability_threshold=.1,sign_transition_width=0.),diagnostics=diag)
    for x,y in zip(base,observed):torch.testing.assert_close(x,y)
    assert not diag['wrong_probability'].requires_grad
    assert ((diag['wrong_probability']>=0)&(diag['wrong_probability']<=1)).all()
    with pytest.raises(ValueError):imp.sign_weight({'sign_violation_weight':-1})


def test_normalization_and_gauge():
    a,b,*rest=inputs()
    h,p=imp.hypotheses(a,.5)
    torch.testing.assert_close(p.exp().sum(1),torch.ones(2,dtype=a.dtype))
    c={'winding_sigma':.5}
    before=imp.pair_cost(a,b,*rest,c)
    aa=a.detach().clone();bb=b.detach().clone()
    aa[:,0]+=2;aa[:,1]-=2;bb[:,0]-=1;bb[:,1]+=1
    after=imp.pair_cost(aa,bb,*rest,c)
    for x,y in zip(before,after):torch.testing.assert_close(x,y)
    sum(x.sum() for x in before).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_learned_sigma_normalized_gradients_and_control():
    a,b,*rest=inputs()
    sigma=torch.tensor([.5,.5],dtype=torch.float64,requires_grad=True)
    _,p=imp.hypotheses(a,sigma,1.)
    _,fixed=imp.hypotheses(a,.5,1.)
    torch.testing.assert_close(p,fixed)
    torch.testing.assert_close(p.exp().sum(1),torch.ones(2,dtype=a.dtype))
    cfg={'winding_sigma':.5,'sigma_support_max':1.}
    varied=imp.pair_cost(a,b,*rest,cfg,sigma_a=sigma,sigma_b=sigma)
    control=imp.pair_cost(a,b,*rest,cfg)
    for x,y in zip(varied,control):torch.testing.assert_close(x,y)
    sum(x.sum() for x in varied).backward()
    assert torch.isfinite(sigma.grad).all() and sigma.grad.abs().sum()>0


def test_learned_sigma_bounds_prior_save():
    import numpy as np
    ext=imp.LearnedSigma({'family':[0,1]}, {'winding_sigma':.5},'cpu')
    torch.testing.assert_close(ext.sigma(),torch.full((2,),.5))
    assert ext.terms(None,None,None,None)['sigma_prior']==0
    with torch.no_grad():ext.logits.copy_(torch.tensor([-10.,10.]))
    assert (ext.sigma()>=.15).all() and (ext.sigma()<=1.).all()
    assert ext.terms(None,None,None,None)['sigma_prior']>0
    saved=ext.arrays()
    other=imp.LearnedSigma({'family':[0,1]}, {'winding_sigma':.5},'cpu')
    other.restore(saved)
    np.testing.assert_array_equal(other.arrays()['winding_sigma'],saved['winding_sigma'])


@pytest.mark.parametrize('multiplier', [0., .25, .75, 1., 10.])
def test_same_sheet_multiplier_selectivity(multiplier):
    a,b,*rest=inputs()
    a=a.detach().clone();b=b.detach().clone()
    a[:,0]=2.;b[:,0]=2.;a[:,1]=.1;b[:,1]=.2
    # Remove arc progression to keep both hypotheses on the same sheet.
    rest[2]=torch.zeros(2);rest[3]=torch.zeros(2)
    cfg={'winding_sigma':.01}
    base=imp.pair_cost(a,b,*rest,cfg)
    changed=imp.pair_cost(a,b,*rest,dict(cfg,same_sheet_match_multiplier=multiplier))
    torch.testing.assert_close(changed[0],base[0])
    delta=b[:,0]+b[:,1]-a[:,0]-a[:,1]
    _,same=soft.integer_separation_energy(delta,rest[5],rest[6],(~rest[7])*(rest[1]-rest[0])*.5,.5)
    for n in (1,2):torch.testing.assert_close(changed[n],base[n]*(1+(multiplier-1)*same))
    # Parallel constraints are unaffected; other separations use their smooth
    # posterior same-sheet mass rather than an exact integer equality mask.
    rest[-1]=torch.ones(2,dtype=torch.bool)
    for x,y in zip(imp.pair_cost(a,b,*rest,cfg),imp.pair_cost(a,b,*rest,dict(cfg,same_sheet_match_multiplier=multiplier))):
        torch.testing.assert_close(x,y)
    rest[-1]=torch.zeros(2,dtype=torch.bool);b[:,0]+=1
    base=imp.pair_cost(a,b,*rest,cfg)
    changed=imp.pair_cost(a,b,*rest,dict(cfg,same_sheet_match_multiplier=multiplier))
    _,same=soft.integer_separation_energy(b[:,0]+b[:,1]-a[:,0]-a[:,1],rest[5],rest[6],(~rest[7])*(rest[1]-rest[0])*.5,.5)
    torch.testing.assert_close(base[0],changed[0])
    for n in (1,2):torch.testing.assert_close(changed[n],base[n]*(1+(multiplier-1)*same))


@pytest.mark.parametrize('multiplier', [-.1, float('nan'), float('inf'), -float('inf')])
def test_same_sheet_multiplier_rejects_invalid_weights(multiplier):
    with pytest.raises(ValueError, match='same_sheet_match_multiplier must be finite and >=0'):
        imp.pair_cost(*inputs(), {'winding_sigma': .01, 'same_sheet_match_multiplier': multiplier})


def test_explicit_enumeration():
    a,b,*rest=inputs();c={'winding_sigma':.5}
    aa,pa=imp.hypotheses(a,.5);bb,pb=imp.hypotheses(b,.5)
    mass=torch.zeros(2,dtype=a.dtype)
    for k in range(aa.shape[1]):
        for l in range(bb.shape[1]):
            energy=soft.pair_integer_energy(aa[:,k],bb[:,l],*rest)
            mass+=torch.exp(pa[:,k]+pb[:,l]-energy)
    torch.testing.assert_close(imp.pair_cost(a,b,*rest,c)[0],-mass.log())


def test_parallel_transverse_only():
    a,b,*rest=inputs()
    rest[0]=torch.tensor([0,1]);rest[1]=rest[0].clone()
    rest[-1]=torch.ones(2,dtype=torch.bool)
    cfg={'winding_sigma':.5}
    base=imp.pair_cost(a,b,*rest,cfg)
    new=imp.pair_cost(a,b,*rest,dict(cfg,disable_parallel_transverse_match=True))
    torch.testing.assert_close(new[0],base[0])
    torch.testing.assert_close(new[1][0],base[1][0])
    torch.testing.assert_close(new[2][1],base[2][1])
    assert new[1][1]==0 and new[2][0]==0
    new[2][0].backward(retain_graph=True)
    assert a.grad.abs().sum()==0 and b.grad.abs().sum()==0
    rest[-1]=torch.zeros(2,dtype=torch.bool)
    for x,y in zip(imp.pair_cost(a,b,*rest,cfg),imp.pair_cost(a,b,*rest,dict(cfg,disable_parallel_transverse_match=True))):
        torch.testing.assert_close(x,y)


@pytest.mark.parametrize('weight', [0., .1, 1., 2.])
def test_parallel_transverse_weight(weight):
    a,b,*rest=inputs()
    rest[0]=torch.tensor([0,1]);rest[1]=rest[0].clone()
    rest[-1]=torch.ones(2,dtype=torch.bool)
    cfg={'winding_sigma':.5}
    base=imp.pair_cost(a,b,*rest,cfg)
    actual=imp.pair_cost(a,b,*rest,dict(cfg,parallel_transverse_weight=weight))
    torch.testing.assert_close(actual[0],base[0])
    torch.testing.assert_close(actual[1],base[1]*torch.tensor([1.,weight]))
    torch.testing.assert_close(actual[2],base[2]*torch.tensor([weight,1.]))
    for term,row in [(1,1),(2,0)]:
        g=torch.autograd.grad(actual[term][row],a,retain_graph=True)[0]
        expected=torch.autograd.grad(base[term][row],a,retain_graph=True)[0]*weight
        torch.testing.assert_close(g,expected)
    rest[-1]=torch.zeros(2,dtype=torch.bool)
    for x,y in zip(imp.pair_cost(a,b,*rest,cfg),imp.pair_cost(a,b,*rest,dict(cfg,parallel_transverse_weight=weight))):
        torch.testing.assert_close(x,y)


@pytest.mark.parametrize('weight', [-1., float('inf'), float('nan')])
def test_invalid_transverse_weight(weight):
    with pytest.raises(ValueError):imp.transverse_weight({'parallel_transverse_weight':weight})


def test_transverse_alias_conflict():
    assert imp.transverse_weight({'disable_parallel_transverse_match':True})==0
    with pytest.raises(ValueError):
        imp.transverse_weight({'parallel_transverse_weight':.1,'disable_parallel_transverse_match':True})


def test_narrow_is_integer_hypothesis():
    a,b,*rest=inputs();a=a.detach().clone();b=b.detach().clone()
    a[:,0]=2.;b[:,0]=3.
    result=imp.pair_cost(a,b,*rest,{'winding_sigma':.01})
    w,u,v=soft.pair_errors(a,b,*rest)
    qa,_,_=soft.points(a,rest[0],rest[2],rest[4]);qb,_,_=soft.points(b,rest[1],rest[3],rest[4])
    energy,_=soft.integer_separation_energy(qb-qa,rest[5],rest[6],(~rest[7])*(rest[1]-rest[0])*.5,.5)
    for actual,expected in zip(result,[energy,u*u,(v/1000)**2]):
        torch.testing.assert_close(actual,expected)


def test_far_evidence_gradient_and_invalid_sigma():
    a,b,*rest=inputs();rest[-2]=torch.tensor([30.,30.])
    loss=imp.pair_cost(a,b,*rest,{'winding_sigma':.5})[0].sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
    assert a.grad[:,0].abs().sum()>0
    with pytest.raises(ValueError):imp.hypotheses(a,0)


def test_initialized_solver(tmp_path):
    import json
    import numpy as np
    from .test_fiber_soft_winding import fixture
    d, nodes = fixture()
    np.savez(tmp_path/'data.npz', **d)
    np.savez(tmp_path/'samples.npz', **nodes)
    cfg=dict(tensor_cache=str(tmp_path/'data.npz'),soft_samples=str(tmp_path/'samples.npz'),
        output_root=str(tmp_path/'base'),winding_count=3,steps=2,log_every=1,
        batch_size=1,lr=.01,width_lr=.01)
    soft.solve(cfg,device='cpu')
    initial=tmp_path/'base/solution.npz'
    cfg.update(initial_solution=str(initial),output_root=str(tmp_path/'trial'),winding_sigma=.5)
    result=soft.solve(cfg,device='cpu',pair_cost=imp.pair_cost,
        diagnostics=imp.diagnostics,model='implicit_discrete_winding_v1')
    assert result['model']=='implicit_discrete_winding_v1'
    assert 0 < result['final']['mean_MAP_probability'] <= 1
    history=json.loads((tmp_path/'trial/history.json').read_text())
    data,samples=soft.tensor_inputs(d,nodes,cfg,'cpu')
    with np.load(initial) as saved:
        w=torch.tensor(saved['widths'])
        raw=w-1e-4+torch.log(-torch.expm1(-(w-1e-4)))
        _, initial_widths=soft.uv.radial_initial_geometry(d,cfg)
        expected,_=soft.objective(torch.tensor(saved['state']),raw,
            torch.tensor(initial_widths),data,samples,cfg,pair_cost=imp.pair_cost)
    assert history[0]['loss']==pytest.approx(float(expected))
    cfg.update(output_root=str(tmp_path/'learned'),learned_winding_sigma=True)
    learned=soft.solve(cfg,device='cpu',pair_cost=imp.pair_cost,
        diagnostics=imp.diagnostics,extension_factory=imp.LearnedSigma)
    assert .15<=learned['final']['sigma_mean']<=1.
    with np.load(tmp_path/'learned/solution.npz') as saved:
        assert saved['state'].shape[1]==4
        assert saved['winding_sigma'].shape==(len(d['family']),)
