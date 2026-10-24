import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_bucket_uv as b
from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as density
from vesuvius.neural_tracing.fiber_trace_3d import fiber_intermediate_uv as mid
from .test_fiber_shared_curve import setup as shared_setup


def setup():
    d,cfg,td,s,w,_=shared_setup()
    cfg.update(intermediate_uv_geometry=True,bucket_uv_active_window=True,
               bucket_uv_fast_gathers=True,bucket_uv_attachment_probability_weighted=False,
               soft_pair_weight=1.,probability_pair_weight=.3,
               shared_curve_hypothesis_mapping=True)
    td={k:(v.float() if v.dtype==torch.float64 else v) for k,v in td.items()}
    ext=b.BucketUV(d,cfg,'cpu');ext.prepare_fields(d,{})
    ext.update(s,w,td,0)
    return d,cfg,td,s,w,ext


def costs(cfg,td,s,w,ext,geometry=True):
    edge=torch.arange(len(td['i']));sa,sb,da,db=ext.pair_states(s,w,td,edge)
    return density.pair_cost(sa,sb,td['family'][td['i']],td['family'][td['j']],da,db,w,
        td['sign'],td['distance'],td['parallel'],cfg,
        sigma_a=ext.cavity.values[td['i']],sigma_b=ext.cavity.values[td['j']],
        geometry=ext.pair_geometry(s,w,edge) if geometry else None)


def test_predicted_winding_keeps_vertical_and_ignores_local_u():
    for cached in (False,True):
        d,cfg,td,s,w,ext=setup()
        cfg.update(pair_predicted_winding=True,reuse_objective_geometry=cached)
        with torch.no_grad():
            ext.intermediate.values[...,0].add_(17000.)
            ext.intermediate.values[...,1].add_(17.)
        edges=torch.arange(len(td['i']))
        geom=ext.pair_geometry(s,w,edges)
        sa,sb,da,db=ext.pair_states(s,w,td,edges)
        for side,h,ds,ids in [('a',sa[:,None],da,td['i']),('b',sb[:,None],db,td['j'])]:
            actual=geom.points(side,h)
            expected=density.soft.points(h,td['family'][ids,None],ds[:,None],w,physical_u=True)
            for k in (0,1):torch.testing.assert_close(actual[k],expected[k])
            cfg['pair_predicted_winding']=False
            old=geom.points(side,h)
            torch.testing.assert_close(actual[2],old[2])
            cfg['pair_predicted_winding']=True
            grad=torch.autograd.grad(actual[0].sum()+actual[1].sum(),ext.intermediate.values,allow_unused=True,retain_graph=True)[0]
            assert grad is None or not grad.any()
        both=geom.points_both(sa[:,None],sb[:,None])
        for side,h,result in [('a',sa[:,None],both[0]),('b',sb[:,None],both[1])]:
            for x,y in zip(result,geom.points(side,h)):torch.testing.assert_close(x,y)


def test_confidence_gate_and_peak_force():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_confidence import confidence_gate
    p=torch.tensor([0.,.5,.7,.9,1.],requires_grad=True)
    g=confidence_gate(p,{})
    torch.testing.assert_close(g,torch.tensor([0.,0.,.5,1.,1.]))
    assert not g.requires_grad
    assert (g[1]*g[4]).item()==0
    e=torch.tensor([100.],requires_grad=True)
    large=torch.autograd.grad(b.meeting_cost(e,50.).sum(),e)[0]
    small=torch.autograd.grad(b.meeting_cost(e,5.).sum(),e)[0]
    torch.testing.assert_close(large,small*10)


def test_final_meeting_gate_does_not_change_intermediate_pair_losses():
    d,cfg,td,s,w,e=setup()
    cfg.update(geometry_confidence_gated=True)
    with torch.no_grad():
        s[:,3].fill_(-10.)
        e.intermediate.values.add_(torch.randn_like(e.intermediate.values)*20)
    gated=costs(cfg,td,s,w,e)
    ungated=costs(dict(cfg,geometry_confidence_gated=False),td,s,w,e)
    for a,z in zip(gated,ungated):torch.testing.assert_close(a,z,rtol=0,atol=0)
    ga=torch.autograd.grad(sum(x.sum() for x in gated),e.intermediate.values,retain_graph=True)[0]
    gz=torch.autograd.grad(sum(x.sum() for x in ungated),e.intermediate.values)[0]
    torch.testing.assert_close(ga,gz,rtol=0,atol=0)


def test_final_gate_leaves_all_regularizers_unchanged():
    d,cfg,td,s,w,e=setup()
    with torch.no_grad():
        s[:,3].fill_(-10.)
        e.values.add_(10.)
        e.intermediate.values.add_(3.)
    edges=torch.arange(len(td['i']))
    plain=e.terms(s,w,td,edges)
    e.config['geometry_confidence_gated']=True
    gated=e.terms(s,w,td,edges)
    assert plain['bucket_uv_meeting']>0
    assert gated['bucket_uv_meeting']==0
    for key in plain:
        if key!='bucket_uv_meeting':
            torch.testing.assert_close(plain[key],gated[key],rtol=0,atol=0)


def test_absolute_bending_uses_each_physical_curve_only():
    d,cfg,td,s,w,e=setup()
    with torch.no_grad():
        e.values.add_(torch.randn_like(e.values)*3)
        e.intermediate.values.add_(torch.randn_like(e.intermediate.values)*3)
    edges=torch.arange(len(td['i']))
    old=e.terms(s,w,td,edges)
    e.config.update(bucket_uv_bend_geometry='absolute',intermediate_uv_bend_geometry='absolute')
    actual=e.terms(s,w,td,edges)
    for k in old:
        if k not in ('bucket_uv_bending','intermediate_uv_bending'):
            torch.testing.assert_close(old[k],actual[k],rtol=0,atol=0)
    support=(e.v_offset_arc[e.triples[:,2]]-e.v_offset_arc[e.triples[:,0]])*.5
    for positions,key,weight in [(e.values,'bucket_uv_bending',cfg['bucket_uv_bend_weight']),
            (e.intermediate.physical(w),'intermediate_uv_bending',cfg.get('intermediate_uv_bend_weight',.0001))]:
        curve=b.bending(positions*e.active[e.owner,:,None],e.v_offset_arc,e.triples)
        expected=weight*(b.bending_energy(curve,e.v_offset_family[e.triples[:,1]],e.longitudinal_ratio)
            /(2*e.full_count)*support).sum()/support.sum()
        torch.testing.assert_close(actual[key],expected)
    # Final straightness has no upstream geometry dependence.
    upstream=torch.autograd.grad(actual['bucket_uv_bending'],e.intermediate.values,
        allow_unused=True,retain_graph=True)[0]
    assert upstream is None


def test_actual_straight_line_vs_constant_offset_on_bent_line():
    arc=torch.tensor([0.,2.,5.]);triples=torch.tensor([[0,1,2]])
    straight=torch.stack((arc,2*arc+1),-1)[:,None,:]
    bent=straight.clone();bent[1,0,1]+=3
    for fn in (b.bending,b.contiguous_bending):
        torch.testing.assert_close(fn(straight,arc,triples),torch.zeros((1,1,2)))
        assert fn(bent,arc,triples).abs().max()>0
        # A constant deformation of a bent upstream curve is still bent.
        torch.testing.assert_close(fn(bent+10,arc,triples),fn(bent,arc,triples))
        torch.testing.assert_close(fn((bent+10)-bent,arc,triples),torch.zeros((1,1,2)))


def test_zero_deviation_pair_equivalence_and_gradients():
    d,cfg,td,s,w,e=setup()
    actual=costs(cfg,td,s,w,e);expected=costs(cfg,td,s,w,e,False)
    for a,z in zip(actual,expected):torch.testing.assert_close(a,z,atol=3e-5,rtol=3e-5)
    sum(x.sum() for x in actual).backward()
    assert torch.isfinite(e.intermediate.values.grad).all()
    assert e.intermediate.values.grad.abs().max()>0
    assert s.grad[:,0].abs().max()>0


def test_hypothesis_independence_and_interpolation():
    d,cfg,td,s,w,e=setup()
    h,_=e.local_hypotheses(s);ids=td['i']
    before=e.intermediate.points(e,s,w,'a',h[ids])
    col=int(torch.where(e.labels[0]==2)[0][0]);node=int(e.v_offset_a[0])
    with torch.no_grad():e.intermediate.values[node,col]+=torch.tensor([1000.,30.])
    after=e.intermediate.points(e,s,w,'a',h[ids])
    torch.testing.assert_close(after[0][0,col]-before[0][0,col],torch.tensor(.1),atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(after[2][0,col]-before[2][0,col],torch.tensor(30.))
    torch.testing.assert_close(after[0][0,col+1],before[0][0,col+1])
    fractional=h[ids,:1].clone();fractional[:,:,0]=2.5
    e.curve.add_plan('between',np.array([0]),np.array([55.]))
    result=e.intermediate.points(e,s,w,'between',fractional[:1])
    assert torch.isfinite(result[0]).all()


def test_grouped_pair_equivalence():
    d,cfg,td,s,w,e=setup()
    with torch.no_grad():e.intermediate.values.add_(torch.randn_like(e.intermediate.values)*3)
    a=costs(dict(cfg,pair_sigma_groups=True),td,s,w,e)
    z=costs(dict(cfg,pair_sigma_groups=False),td,s,w,e)
    for x,y in zip(a,z):torch.testing.assert_close(x,y,atol=1e-4,rtol=1e-4)


def test_regularizers_reach_all_layers_without_probability_gating():
    d,cfg,td,s,w,e=setup()
    with torch.no_grad():
        e.intermediate.values[:,2,1]+=7
        e.values[:,2,1]-=3
    terms=e.terms(s,w,td,torch.arange(3))
    sum(terms[k] for k in ('intermediate_uv_attachment','bucket_uv_attachment')).backward()
    assert e.intermediate.values.grad[:,2,1].abs().max()>0
    assert e.values.grad[:,2,1].abs().max()>0
    assert e.curve.v.grad.abs().max()>0


def test_window_remaps_labels_and_new_labels_reset():
    d,cfg,td,s,w,e=setup()
    optimizer=torch.optim.Adam(e.optimizer_groups());e.bind_optimizer(optimizer)
    sum(e.terms(s,w,td,torch.arange(3)).values()).backward();optimizer.step()
    old=e.intermediate.values.detach().clone();labels=e.labels.clone()
    with torch.no_grad():s[:,0].add_(1)
    e.update(s,w,td,1)
    for node in range(len(e.owner)):
        f=e.owner[node];label=labels[f,labels.shape[1]//2]
        oi=torch.where(labels[f]==label)[0];ni=torch.where(e.labels[f]==label)[0]
        torch.testing.assert_close(e.intermediate.values[node,ni],old[node,oi])
    assert any(p is e.intermediate.values for group in optimizer.param_groups for p in group['params'])


def test_checkpoint_and_saved_gt_points(tmp_path):
    d,cfg,td,s,w,e=setup()
    with torch.no_grad():e.intermediate.values[...,1].add_(17)
    saved=dict(e.arrays(),state=s.detach().numpy(),widths=w.detach().numpy(),winding_sigma=e.cavity.values.numpy())
    np.savez(tmp_path/'snapshot.npz',**saved)
    with np.load(tmp_path/'snapshot.npz') as snapshot:
        restored=mid.restore_model(d,snapshot,cfg)
        torch.testing.assert_close(restored.intermediate.values,e.intermediate.values)
    expected=e.intermediate.points(e,s,w,'a',s[td['i'],None,:])
    actual=mid.saved_points(d,saved,s.detach().numpy(),w.detach().numpy(),d['i'],d['ds_a'])
    for x,y in zip(actual,expected):np.testing.assert_allclose(x,y.detach()[:,0],atol=1e-5)


def test_three_layer_visualization(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_bucket_uv_visualization import geometry,intermediate_geometry
    d,cfg,td,s,w,e=setup()
    with torch.no_grad():
        s[:,0].copy_(torch.tensor([2.,3.]));e.cavity.values.fill_(.15)
        e.intermediate.values[...,1].add_(17)
    saved=dict(e.arrays(),state=s.detach().numpy(),widths=w.detach().numpy(),winding_sigma=e.cavity.values.numpy())
    np.savez(tmp_path/'snapshot.npz',**saved)
    with np.load(tmp_path/'snapshot.npz') as src:
        before,after,maximum,predicted,actual,ext,labels=geometry(d,src,cfg)
    middle,nodes=intermediate_geometry(d,saved,ext,labels,before)
    np.testing.assert_array_equal(middle[0],before[0])
    np.testing.assert_allclose(nodes[:,1]-predicted[:,1],17,atol=1e-4)
    for i in range(len(before[0])):
        np.testing.assert_allclose(middle[2][i]-before[2][i],17,atol=1e-4)
