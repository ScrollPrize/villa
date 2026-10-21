import numpy as np
import torch
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_scaled_lbfgs import scaled_joint


def test_fixed_length_vram_estimate():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_scaled_lbfgs import estimated_fixed_length_vram
    gib=1024**3
    assert estimated_fixed_length_vram(14142,32768)==13*gib
    assert estimated_fixed_length_vram(18945,32768)==17*gib
    assert estimated_fixed_length_vram(100000,32768)==28*gib
    assert estimated_fixed_length_vram(18945,2048)==3*gib
    assert estimated_fixed_length_vram(18945,32768,1.)>estimated_fixed_length_vram(18945,32768,.5)
    with pytest.raises(ValueError):estimated_fixed_length_vram(10,0)


def test_lbfgs_step_size_is_forwarded(monkeypatch):
    original=torch.optim.LBFGS
    seen=[]
    def optimizer(*args,**kwargs):
        seen.append(kwargs['lr'])
        return original(*args,**kwargs)
    monkeypatch.setattr(torch.optim,'LBFGS',optimizer)
    state=torch.tensor([[2.,0.,0.,100.]],dtype=torch.float64,requires_grad=True)
    raw=torch.tensor([1500.,.1],dtype=torch.float64,requires_grad=True)
    scaled_joint(lambda:(state[:,0].square().sum()+raw.sum()*0,{}),state,raw,torch.nn.Module(),2,
        optimizer_kind='torch_lbfgs',lbfgs_lr=.5,winding_coordinate_scale=2.,monotone_widths=True,relative_widths=True)
    assert seen==[.5]


def test_soft_minimum_positive_roundtrip_and_escape():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_soft_winding import encode_widths,decode_widths
    cfg=dict(width_parameterization='monotone_relative',soft_minimum_width=True)
    for v in (.00001,.00099,.001,1.,99.,100.,1000.):
        widths=torch.tensor([v,2*v],dtype=torch.float64)
        torch.testing.assert_close(decode_widths(encode_widths(widths,cfg),cfg),widths)
    raw=torch.tensor([-100.,.1],dtype=torch.float64,requires_grad=True)
    state=torch.tensor([[0.,0.,0.,100.]],dtype=torch.float64,requires_grad=True)
    def objective():
        w=decode_widths(raw,cfg)
        assert bool((w>0).all()) and bool((w.diff()>=0).all())
        return ((w[0]-2000)/1000).square()+10*torch.relu(1-raw[0]/1000).square()+state[:,0].square().sum(),{}
    result=scaled_joint(objective,state,raw,torch.nn.Module(),300,
        monotone_widths=True,relative_widths=True,optimizer_kind='torch_lbfgs',
        width_coordinate_scale=100,soft_minimum_width=True,square_match_initial_scale=True)
    assert abs(float(decode_widths(raw,cfg)[0].detach())-2000)<.1


@pytest.mark.parametrize('optimizer',['lbfgs','adam','torch_lbfgs','scipy_squared'])
def test_relative_widths_and_packed_optimizers(optimizer):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_soft_winding import encode_widths,decode_widths
    cfg=dict(width_parameterization='monotone_relative',minimum_winding_width=1000.)
    w=torch.tensor([1500.,1800.,2400.],dtype=torch.float64)
    raw=encode_widths(w,cfg).requires_grad_()
    torch.testing.assert_close(decode_widths(raw,cfg),w)
    assert torch.autograd.gradcheck(lambda r:decode_widths(r,cfg),(raw,))
    scaled=raw.detach().clone();scaled[0]*=2
    torch.testing.assert_close(decode_widths(scaled,cfg),2*w)
    state=torch.tensor([[1.,2.,3.,100.]],dtype=torch.float64,requires_grad=True)
    target=torch.tensor([1000.,2000.,3000.],dtype=torch.float64)
    def objective():
        actual=decode_widths(raw,cfg)
        assert actual[0]>=1000.-1e-9 and bool((torch.diff(actual)>=0).all())
        return ((actual-target)/1000).square().sum()+state[:,0].square().sum(),{}
    scaled_joint(objective,state,raw,torch.nn.Module(),2000 if optimizer=='adam' else 300,
                 monotone_widths=True,relative_widths=True,width_coordinate_scale=100,
                 relative_width_coordinate_scale=.1,optimizer_kind=optimizer,adam_lr=.1)
    torch.testing.assert_close(decode_widths(raw,cfg),target,rtol=1e-4,atol=.1)
    torch.testing.assert_close(state.detach()[0,1:],torch.tensor([2.,3.,100.],dtype=torch.float64))


def test_torch_plateau_stops_without_claiming_small_gradient(monkeypatch):
    monkeypatch.setattr(torch.optim.LBFGS,'step',lambda self,closure:closure())
    state=torch.tensor([[2.,0.,0.,100.]],dtype=torch.float64,requires_grad=True)
    raw=torch.tensor([1500.,.1],dtype=torch.float64,requires_grad=True)
    def objective():
        return state[:,0].square().sum()+raw.sum()*0,{}
    result=scaled_joint(objective,state,raw,torch.nn.Module(),100,
        monotone_widths=True,relative_widths=True,optimizer_kind='torch_lbfgs',
        plateau_window=3,plateau_atol=100.,plateau_rtol=0.)
    assert result['success'] and result['iterations']==3
    assert result['scaled_gradient_max']>1e-8
    assert 'plateau' in result['message']


@pytest.mark.parametrize('matched',[False,True])
def test_squared_cache_preserves_trajectory(matched):
    def run(cache):
        state=torch.tensor([[2.,0.,0.,100.]],dtype=torch.float64,requires_grad=True)
        raw=torch.tensor([1500.,.1,.2],dtype=torch.float64,requires_grad=True)
        history=[]
        def objective():
            return ((raw[0]-2000)/100).square()+((raw[1:]-.3)/.1).square().sum()+state[:,0].square().sum(),{}
        result=scaled_joint(objective,state,raw,torch.nn.Module(),50,
            callback=lambda i:history.append(torch.cat((state.detach().flatten(),raw.detach())).clone()),
            monotone_widths=True,relative_widths=True,optimizer_kind='torch_lbfgs',
            square_match_initial_scale=matched,closure_cache=cache)
        return torch.stack(history),result
    uncached,a=run(False);cached,b=run(True)
    torch.testing.assert_close(cached,uncached,rtol=0,atol=0)
    assert b['evaluations']<a['evaluations'] and b['cache_hits']>0


@pytest.mark.parametrize('name',['width_coordinate_scale','phase_coordinate_scale'])
@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_invalid_coordinate_scale(name,value):
    with pytest.raises(ValueError,match='coordinate scales'):
        scaled_joint(None,None,None,None,1,**{name:value})


def test_direct_widths_escape_positive_bound_and_remain_positive():
    ext=torch.nn.Module()
    state=torch.tensor([[0.,0.,0.,100.]],dtype=torch.float64,requires_grad=True)
    widths=torch.tensor([1e-4,2000.],dtype=torch.float64,requires_grad=True)
    def objective():
        assert bool((widths>=1e-4-1e-15).all())
        return ((widths[0]-1000)/1000).square()+((widths[1]+1000)/1000).square()+state[:,0].square().sum(),{}
    scaled_joint(objective,state,widths,ext,100,direct_widths=True)
    torch.testing.assert_close(widths.detach(),torch.tensor([1000.,1e-4],dtype=torch.float64),atol=1e-6,rtol=1e-6)


def test_width_encodings_preserve_objective_and_direct_gradients():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_soft_winding import encode_widths,decode_widths
    w=torch.tensor([1e-4,3.,10000.],dtype=torch.float64)
    for mode in ('softplus','direct'):
        cfg=dict(width_parameterization=mode)
        raw=encode_widths(w,cfg).requires_grad_()
        actual=decode_widths(raw,cfg)
        torch.testing.assert_close(actual,w)
        gradient=torch.autograd.grad(actual.sum(),raw)[0]
        if mode=='direct':torch.testing.assert_close(gradient,torch.ones_like(w))
        else:assert gradient[0]==0


def test_monotone_widths_roundtrip_and_recover_zero_increments():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_soft_winding import encode_widths,decode_widths
    cfg=dict(width_parameterization='monotone_additive',minimum_winding_width=1000.)
    initial=torch.tensor([1000.,1000.,1000.],dtype=torch.float64)
    raw=encode_widths(initial,cfg).requires_grad_()
    torch.testing.assert_close(decode_widths(raw,cfg),initial)
    state=torch.tensor([[0.,0.,0.,100.]],dtype=torch.float64,requires_grad=True)
    target=torch.tensor([2000.,2300.,3000.],dtype=torch.float64)
    def objective():
        w=decode_widths(raw,cfg)
        assert w[0]>=1000.-1e-10 and bool((torch.diff(w)>=-1e-10).all())
        return ((w-target)/1000).square().sum()+state[:,0].square().sum(),{}
    assert torch.autograd.grad(objective()[0],raw)[0][1]<0
    scaled_joint(objective,state,raw,torch.nn.Module(),100,monotone_widths=True)
    torch.testing.assert_close(decode_widths(raw,cfg),target,rtol=1e-6,atol=1e-4)
    for invalid in ([999.,1500.],[2000.,1500.]):
        with pytest.raises(ValueError,match='no implicit projection'):
            encode_widths(torch.tensor(invalid),cfg)


def test_monotone_checkpoint_allows_only_ulp_roundoff():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_soft_winding import encode_widths,decode_widths
    cfg=dict(width_parameterization='monotone_relative',soft_minimum_width=True)
    w=torch.tensor([1000.,40000.,40000.-7.275957614183426e-12,50000.],dtype=torch.float64)
    restored=decode_widths(encode_widths(w,cfg),cfg)
    torch.testing.assert_close(restored,w,rtol=1e-14,atol=0)
    w[2]=39999.999
    with pytest.raises(ValueError,match='no implicit projection'):encode_widths(w,cfg)


def test_dynamic_uncertainty_updates_only_between_line_searches(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_scaled_lbfgs as solver
    events=[]
    class Extension:
        def update(self,state,widths,data,iteration):events.append(('refresh',iteration))
    def fake(objective,state,raw,extension,steps,callback,**kwargs):
        events.append(('solve',steps));objective();objective()
        if callback:callback(steps)
        return dict(iterations=steps,evaluations=2)
    monkeypatch.setattr(solver,'scaled_joint',fake)
    result=solver.uncertainty_joint(lambda:events.append(('objective',None)),None,
        torch.tensor([1.]),Extension(),25,{},dict(cavity_winding_sigma=True,cavity_update_every=10))
    assert [e for e in events if e[0]=='refresh']==[('refresh',0),('refresh',10),('refresh',20)]
    assert [e[1] for e in events if e[0]=='solve']==[10,10,5]
    assert result['accepted_iterations']==25 and result['evaluations']==6


def test_fixed_uncertainty_does_not_restart_optimizer(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_scaled_lbfgs as solver
    calls=[]
    monkeypatch.setattr(solver,'scaled_joint',lambda *args,**kwargs:calls.append(args[4]) or {})
    solver.uncertainty_joint(None,None,None,None,123,{}, {})
    assert calls==[123]


def test_joint_masks_scales_and_rejection():
    for rejection in (False,True):
        ext=torch.nn.Module()
        ext.register_parameter('u',torch.nn.Parameter(torch.tensor([.04],dtype=torch.float64)))
        ext.register_parameter('v',torch.nn.Parameter(torch.tensor([400.],dtype=torch.float64)))
        state=torch.tensor([[4.,3.,2.,1.]],dtype=torch.float64,requires_grad=True)
        raw=torch.tensor([4000.],dtype=torch.float64,requires_grad=True)
        def objective():
            loss=state.square().sum()+(raw/1000).square().sum()+(ext.u/.01).square().sum()+(ext.v/100).square().sum()
            return loss,{}
        result=scaled_joint(objective,state,raw,ext,100,optimize_rejection=rejection)
        assert result['success']
        assert abs(float(state[0,0].detach()))<1e-7
        np.testing.assert_array_equal(state.detach()[0,1:3], [3.,2.])
        assert abs(float(state[0,3].detach())-(0 if rejection else 1))<1e-7
        assert abs(float(raw.detach()[0]))<1e-6


def test_width_only_does_not_change_state_or_geometry():
    ext=torch.nn.Linear(1,1,dtype=torch.float64)
    state=torch.ones((2,4),dtype=torch.float64,requires_grad=True)
    raw=torch.tensor([2000.],dtype=torch.float64,requires_grad=True)
    before=[p.detach().clone() for p in ext.parameters()]
    def objective():return state.square().sum()+(raw/1000).square().sum()+sum(p.square().sum() for p in ext.parameters()),{}
    scaled_joint(objective,state,raw,ext,100,optimize_width_only=True)
    assert torch.equal(state,torch.ones_like(state))
    assert all(torch.equal(p,b) for p,b in zip(ext.parameters(),before))
    assert abs(float(raw.detach()[0]))<1e-6


@pytest.mark.parametrize('direct',[False,True])
def test_fixed_enabled_subset_keeps_geometry_free(direct):
    state=torch.tensor([[2.,0.,0.,100.],[3.,0.,0.,1.]],dtype=torch.float64,requires_grad=True)
    raw=torch.tensor([1000.],dtype=torch.float64,requires_grad=True)
    def objective():
        return state[:,0].square().sum()+(state[1,3].sigmoid()-.7).square()+(raw/1000-1).square().sum(),{}
    scaled_joint(objective,state,raw,torch.nn.Module(),100,optimize_rejection=True,
                 direct_rejection=direct,fixed_enabled_mask=[True,False])
    assert float(state[0,3].detach())==100.
    assert float(state[:,0].abs().max().detach())<1e-5
    assert abs(float(state[1,3].sigmoid().detach())-.7)<1e-5


def test_accumulated_parts_match_full_value_and_gradient(monkeypatch):
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_scaled_lbfgs as solver
    values=[]
    def inspect(fun,x,**kwargs):
        value,gradient=fun(x);values.append((value,gradient.copy()))
        return SimpleNamespace(x=x,jac=gradient,success=True,message='gradient check',nit=0,nfev=1)
    monkeypatch.setattr(solver,'minimize',inspect)
    for chunked in (False,True):
        state=torch.tensor([[1.,0.,0.,2.],[3.,0.,0.,4.]],dtype=torch.float64,requires_grad=True)
        raw=torch.tensor([2000.],dtype=torch.float64,requires_grad=True)
        def parts():
            for i in range(2):
                yield state[i].square().sum()/2+(raw/1000).square().sum()/2,{}
        def full():return state.square().sum()/2+(raw/1000).square().sum(),{}
        scaled_joint(full,state,raw,torch.nn.Module(),1,optimize_rejection=True,
                     objective_parts=parts if chunked else None)
    np.testing.assert_allclose(values[0][0],values[1][0],rtol=1e-14)
    np.testing.assert_allclose(values[0][1],values[1][1],rtol=1e-14)


def test_direct_rejection_can_move_away_from_either_saturated_logit():
    for initial in (-18.,18.):
        state=torch.tensor([[0.,0.,0.,initial]],dtype=torch.float64,requires_grad=True)
        raw=torch.tensor([2000.],dtype=torch.float64,requires_grad=True)
        def objective():
            return (state[:,3].sigmoid()-.7).square().sum()+(raw/1000-2).square().sum(),{}
        fit=scaled_joint(objective,state,raw,torch.nn.Module(),100,
                         optimize_rejection=True,direct_rejection=True)
        assert fit['success']
        assert abs(float(state[0,3].sigmoid().detach())-.7)<1e-7


def test_direct_rejection_leaves_forced_active_state_unchanged():
    state=torch.tensor([[0.,0.,0.,4.]],dtype=torch.float64,requires_grad=True)
    raw=torch.tensor([2000.],dtype=torch.float64,requires_grad=True)
    def objective():return (state[:,3].sigmoid()-.7).square().sum()+(raw/1000-1).square().sum(),{}
    scaled_joint(objective,state,raw,torch.nn.Module(),100,direct_rejection=True)
    assert float(state[0,3].detach())==4.


def test_no_knot_fibers_are_rejected_not_optimized():
    ext=torch.nn.Module();ext.curve=torch.nn.Module()
    ext.curve.register_buffer('owner',torch.tensor([0,0]))
    state=torch.tensor([[2.,.4,300.,4.],[2.,.4,300.,4.]],dtype=torch.float64,requires_grad=True)
    raw=torch.tensor([2000.],dtype=torch.float64,requires_grad=True)
    def objective():
        return state[:,0].square().sum()+(state[1,1]/.01).square()+(state[1,2]/100).square()+(raw/1000).square().sum(),{}
    with pytest.raises(ValueError,match='without perpendicular nodes'):
        scaled_joint(objective,state,raw,ext,100)


@pytest.mark.parametrize('mode',['softplus','monotone_additive','monotone_relative'])
def test_managed_adapter_initializes_curve_and_publishes_early_completion(tmp_path,monkeypatch,mode):
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_scaled_lbfgs import managed_solve
    class Extension(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.curve=SimpleNamespace(ready=False,initialize=self.initialize)
        def initialize(self,*args):self.curve.ready=True
        def arrays(self):return {}
        def diagnostics(self):return {}
    ext=Extension()
    state=torch.tensor([[2.,0.,0.,4.]],dtype=torch.float64,requires_grad=True)
    config=dict(output_root=str(tmp_path),predicted_only_geometry=True,steps=100,
                width_parameterization=mode,lbfgs_width_coordinate_scale=100,
                relative_width_coordinate_scale=.1)
    raw=soft.encode_widths(torch.tensor([2000.],dtype=torch.float64),config).requires_grad_()
    def objective(state,raw,*args,**kwargs):
        assert ext.curve.ready
        val=(state[:,0]-1).square().sum()+((soft.decode_widths(raw,config)-1000)/1000).square().sum()
        return val,dict(test=val)
    monkeypatch.setattr(soft,'objective',objective)
    d={k:np.array([0]) for k in ('source_id','start','end','family')}
    result=managed_solve(config,
        state,raw,torch.tensor([2000.]),{}, {},d,ext,None,None)
    assert result['completed'] and result['final']['iteration']<100
    assert result['final']['loss']<1e-12
    assert (tmp_path/'solution.npz').exists()
    with np.load(tmp_path/'solution.npz') as saved:
        np.testing.assert_allclose(saved['widths'],[1000.],rtol=1e-6)
    assert abs(result['final']['width_factor']-.5)<1e-6


def test_gap_scaling_is_identity_on_gt_sized_spans():
    results=[]
    for enabled in (False,True):
        ext=torch.nn.Module();ext.curve=torch.nn.Module()
        ext.curve.register_buffer('owner',torch.tensor([0,0,0]))
        ext.curve.register_buffer('arc',torch.tensor([0.,111.,250.],dtype=torch.float64))
        ext.curve.u=torch.nn.Parameter(torch.tensor([.1,.2,.3],dtype=torch.float64))
        ext.curve.v=torch.nn.Parameter(torch.tensor([10.,20.,30.],dtype=torch.float64))
        state=torch.ones((1,4),dtype=torch.float64,requires_grad=True)
        raw=torch.tensor([2000.],dtype=torch.float64,requires_grad=True)
        def objective():return state.square().sum()+(raw/1000).square().sum()+(ext.curve.u/.01).square().sum()+(ext.curve.v/100).square().sum(),{}
        steps=[]
        result=scaled_joint(objective,state,raw,ext,100,lambda i:steps.append(float(objective()[0].detach())),local_gap_scaling=enabled)
        results.append((result,steps))
    assert results[0]==results[1]
