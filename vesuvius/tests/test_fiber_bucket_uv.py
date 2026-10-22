import numpy as np
import torch
import pytest
from vesuvius.neural_tracing.fiber_trace_3d import fiber_bucket_uv as b


def test_huber_meeting_cost_and_force():
    x=torch.tensor([[3.,-4.],[100.,-200.]],requires_grad=True)
    cost=b.meeting_cost(x,50.)
    torch.testing.assert_close(cost,torch.tensor([25.,25000.])/1e6)
    cost.sum().backward()
    torch.testing.assert_close(x.grad,torch.tensor([[6.,-8.],[100.,-100.]])/1e6)
    torch.testing.assert_close(b.meeting_cost(x),x.square().sum(-1)/1e6)
    for invalid in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError):b.meeting_cost(x,invalid)


def fixture():
    d=dict(source_id=np.array([11,12]),family=np.array([0,1]),start=np.zeros(2),end=np.ones(2)*300,
           node_ds=np.array([[0,150,300],[0,150,300.]]),i=np.zeros(3,dtype=np.int64),
           j=np.ones(3,dtype=np.int64),ds_a=np.array([10,100,250.]),ds_b=np.array([20,110,260.]),
           parallel=np.zeros(3,dtype=bool),sign=np.ones(3),distance=np.ones(3)*.5)
    cfg=dict(cavity_winding_sigma=True,sigma_max=1.,winding_sigma=.5,winding_count=10,
             bucket_uv_match_weight=10.,bucket_uv_attach_weight=.1,bucket_uv_bend_weight=1.,bucket_uv_lr=2.)
    td={k:torch.as_tensor(v) for k,v in d.items()};td['class_weight']=torch.ones(3)
    s=torch.tensor([[2.2,.1,0.,4.],[3.,-.85,0.,4.]],requires_grad=True)
    w=torch.full((10,),1000.,requires_grad=True)
    return d,cfg,td,s,w


def test_diagonal_and_probability_gradients():
    d,cfg,td,s,w=fixture();ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    assert ext.carry.tolist()==[-1,-1,-1]
    old=ext.carry.clone()
    loss=sum(ext.terms(s,w,td,torch.arange(3)).values());loss.backward()
    assert torch.isfinite(s.grad).all() and s.grad[:,0].abs().max()>0
    assert ext.values.grad.abs().max()>0
    with torch.no_grad():s[:,0]+=.7;s[:,1]+=.8
    ext.update(s,w,td,1)
    assert torch.equal(old,ext.carry) # no dynamic phase rounding or rematching
    a=torch.tensor([[0.,1.,2.]]);bb=torch.tensor([[0.,1.,2.,3.]])
    _,ia,ib=b.same_sheet_pairs(a,bb,torch.tensor([-1]))
    assert ia.tolist()==[0,1,2] and ib.tolist()==[1,2,3]


def test_attachment_diagnostics():
    d,cfg,td,s,w=fixture();ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    with torch.no_grad():
        ext.values.add_(torch.tensor([3.,4.]))
        ext.terms(s,w,td,torch.arange(3))
    stats=ext.diagnostics()
    for axis,offset in [('u',3.),('v',4.)]:
        assert stats[f'bucket_uv_attachment_{axis}_p0.1_mean']==pytest.approx(offset)
        assert stats[f'bucket_uv_attachment_sum_{axis}_p0.1_mean']==pytest.approx(2*offset)
        assert stats[f'bucket_uv_combined_{axis}_p0.1_mean']==pytest.approx(
            stats[f'bucket_uv_local_{axis}_p0.1_mean']+2*offset)
    with torch.no_grad():
        ext.values[ext.owner==0,:,0]+=2
        ext.terms(s,w,td,torch.arange(3))
    stats=ext.diagnostics()
    assert stats['bucket_uv_attachment_u_p0.1_mean']==pytest.approx(4.)
    assert stats['bucket_uv_attachment_u_p0.1_max']==pytest.approx(5.)


def test_affine_and_zigzag():
    arc=torch.tensor([0.,10.,100.,140.]);triples=torch.tensor([[0,1,2],[1,2,3]])
    offsets=(arc[:,None,None]*torch.tensor([[[.2,.5]]])+3).expand(-1,5,-1).clone()
    assert b.bending(offsets,arc,triples).abs().max()<1e-8
    offsets[2]+=5
    assert b.bending(offsets,arc,triples).square().sum()>0


def test_anisotropic_bending_axes_and_bucket_independence():
    curve=torch.tensor([[[1.,2.],[3.,4.]],[[1.,2.],[3.,4.]]],
                       dtype=torch.float64,requires_grad=True)
    family=torch.tensor([0,1])
    torch.testing.assert_close(b.bending_energy(curve,family),
                               (1000*curve).square().sum((1,2)),rtol=0,atol=0)
    energy=b.bending_energy(curve,family,10.)
    torch.testing.assert_close(energy,torch.tensor([120.,210.],dtype=torch.float64)*1e6)
    energy.sum().backward()
    torch.testing.assert_close(curve.grad,2e6*curve.detach()*torch.tensor(
        [[[10.,1.]],[[1.,10.]]],dtype=torch.float64))
    isolated=curve.detach().clone();isolated[:,1]=0
    torch.testing.assert_close(b.bending_energy(curve.detach(),family,10.),
        b.bending_energy(isolated,family,10.)+b.bending_energy(curve.detach()-isolated,family,10.))


@pytest.mark.parametrize('ratio',[0.,-1.,float('nan'),float('inf')])
def test_invalid_bending_ratio(ratio):
    d,cfg,_,_,_=fixture();cfg['bucket_uv_bend_longitudinal_ratio']=ratio
    with pytest.raises(ValueError):b.BucketUV(d,cfg,'cpu')


def test_bucket_visualization_uses_saved_absolute_positions(monkeypatch,tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_bucket_uv_visualization as v
    d,cfg,td,s,w=fixture()
    cfg['bucket_uv_active_window']=True
    ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    ext.cavity.values=torch.full((2,),.15)
    with torch.no_grad():ext.values.add_(torch.tensor([3.,4.]))
    saved=dict(ext.arrays(),state=s.detach().numpy(),widths=w.detach().numpy())
    monkeypatch.setattr(v.soft,'tensor_inputs',lambda *args:(td,None))
    path=tmp_path/'solution.npz';np.savez(path,**saved)
    with np.load(path) as checkpoint:
        before,after,maximum,predicted,actual,_,_=v.geometry(d,checkpoint,cfg)
    np.testing.assert_allclose(actual-predicted,np.broadcast_to([3.,4.],actual.shape))
    np.testing.assert_allclose(maximum,4.)
    for x,y in zip(before[1],after[1]):np.testing.assert_allclose(y-x,3.)
    for x,y in zip(before[2],after[2]):np.testing.assert_allclose(y-x,4.)


def test_visualization_metric_gate():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_bucket_uv_visualization import eligible_edges
    d=dict(i=np.array([0,0,0,0]),j=np.array([1,2,1,3]),
           family=np.array([0,1,1,0]),parallel=np.array([False,False,True,False]))
    labels=np.array([2,3,3,3]);carry=np.full(4,-1)
    np.testing.assert_array_equal(eligible_edges(d,labels,np.array([.5,.5,.49,.9]),carry),[0])
    assert not len(eligible_edges(d,labels,np.array([.499,.9,.9,.9]),carry))


def test_contiguous_bending_boundaries_and_gradients():
    arc=torch.tensor([0.,10.,100.,0.,0.,5.,20.,21.],dtype=torch.float64)
    triples=torch.tensor([[0,1,2],[4,5,6],[5,6,7]])
    values=torch.randn(8,7,2,dtype=torch.float64,requires_grad=True)
    a=b.bending(values,arc,triples)
    c=b.contiguous_bending(values,arc,triples)
    torch.testing.assert_close(a,c,atol=0,rtol=0)
    ga=torch.autograd.grad(a.square().sum(),values)[0]
    gc=torch.autograd.grad(c.square().sum(),values)[0]
    torch.testing.assert_close(ga,gc)


@pytest.mark.parametrize('flag',['reuse_constraint_work','bucket_uv_direct_meetings','bucket_uv_contiguous'])
def test_eager_speed_values_and_gradients(flag):
    d,cfg,td,s,w=fixture()
    cfg.update(bucket_uv_active_window=True,bucket_uv_fast_gathers=True,
               bucket_uv_attachment_probability_weighted=False)
    ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    with torch.no_grad():ext.values.add_(torch.randn_like(ext.values)*.1)
    edges=torch.tensor([2,0,2,1])
    baseline=ext.terms(s,w,td,edges)
    grads=torch.autograd.grad(sum(baseline.values()),(s,w,ext.values))
    cfg[flag]=True
    changed=ext.terms(s,w,td,edges)
    other=torch.autograd.grad(sum(changed.values()),(s,w,ext.values))
    for key in baseline:torch.testing.assert_close(baseline[key],changed[key])
    for a,c in zip(grads,other):torch.testing.assert_close(a,c,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('empty',[False,True])
def test_direct_meetings_unequal_windows(empty):
    d,cfg,td,s,w=fixture()
    cfg.update(bucket_uv_active_window=True,bucket_uv_fast_gathers=True,
               bucket_uv_attachment_probability_weighted=False)
    ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    ext.cavity.values=torch.tensor([.2,.8]);ext.refresh_window(s,w)
    if empty:ext.carry.fill_(100)
    edges=torch.tensor([2,0,2])
    a=ext.terms(s,w,td,edges)
    ga=torch.autograd.grad(sum(a.values()),(s,w,ext.values))
    cfg['bucket_uv_direct_meetings']=True
    c=ext.terms(s,w,td,edges)
    gc=torch.autograd.grad(sum(c.values()),(s,w,ext.values))
    for key in a:torch.testing.assert_close(a[key],c[key])
    for x,y in zip(ga,gc):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)


def test_gauge_relabels_pairs():
    d,cfg,td,s,w=fixture();c=b.fixed_carry(s,w,td)
    shifted=s.detach().clone();shifted[0,0]+=2;shifted[0,1]-=2
    cc=b.fixed_carry(shifted,w,td)
    assert torch.equal(cc,c+2)
    # A label moves by+2, B is unchanged; same physical diagonal remains.
    ka=torch.arange(-2.,6.)[None,:].expand(3,-1);kb=ka.clone()
    r,a,bb=b.same_sheet_pairs(ka,kb,c)
    rr,aa,bbb=b.same_sheet_pairs(ka+2,kb,cc)
    assert torch.equal(r,rr) and torch.equal(a,aa) and torch.equal(bb,bbb)


def test_restore_fixed_mapping_and_storage(tmp_path):
    d,cfg,td,s,w=fixture();ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    with torch.no_grad():ext.values.uniform_(-1,1)
    np.savez(tmp_path/'s.npz',**ext.arrays())
    restored=b.BucketUV(d,cfg,'cpu')
    with np.load(tmp_path/'s.npz') as saved:restored.restore(saved)
    with torch.no_grad():s[:,1]+=3
    restored.update(s,w,td,1)
    assert torch.equal(ext.carry,restored.carry)
    torch.testing.assert_close(ext.values,restored.values)
    with pytest.raises(ValueError,match='exhausted'):
        ext.lookup(torch.tensor([0]),torch.tensor([[10000.]]))


def test_short_joint_optimization_reduces_added_objective():
    d,cfg,td,s,w=fixture();ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    opt=torch.optim.Adam([s,ext.values],lr=.05);values=[]
    for _ in range(15):
        opt.zero_grad();loss=sum(ext.terms(s,w,td,torch.arange(3)).values())
        values.append(float(loss.detach()));loss.backward();opt.step()
    assert values[-1]<values[0]


def test_filtering_preserves_endpoint_identity():
    d,cfg,td,s,w=fixture();d['parallel'][0]=True;cfg['filter_constraint_families']=True
    ext=b.BucketUV(d,cfg,'cpu')
    assert len(ext.v_offset_a)==2
    assert ext.v_offset_arc[ext.v_offset_a[0]]==100
    filtered,_=b.soft.constraint_family_filter(d,True)
    td={k:torch.as_tensor(v) for k,v in filtered.items()};td['class_weight']=torch.ones(2)
    ext.update(s,w,td,0)
    assert torch.isfinite(sum(ext.terms(s,w,td,torch.arange(2)).values()))


def test_parallel_events_create_no_bucket_nodes_or_regularization():
    d,cfg,td,s,w=fixture();cfg['filter_constraint_families']=False
    d['parallel'][1]=True;td['parallel']=torch.as_tensor(d['parallel'])
    ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    assert ext.v_offset_a.tolist()==[0,-1,1]
    assert ext.v_offset_b.tolist()==[2,-1,3]
    assert ext.v_offset_arc.tolist()==[10.,250.,20.,260.]
    assert ext.triples.numel()==0  # parallel middle point is NOT a smoothing knot
    with torch.no_grad():ext.values.add_(3.)
    terms=ext.terms(s,w,td,torch.arange(3))
    assert terms['bucket_uv_bending']==0
    # Moving a parallel event's base point cannot affect any geometry or loss.
    moved={k:np.copy(v) for k,v in d.items()}
    moved['ds_a'][1]=149.;moved['ds_b'][1]=191.
    other=b.BucketUV(moved,cfg,'cpu');other.update(s,w,td,0)
    with torch.no_grad():other.values.copy_(ext.values)
    for key,value in other.terms(s,w,td,torch.arange(3)).items():
        torch.testing.assert_close(value,terms[key])


def test_no_perpendicular_events_have_no_bucket_geometry():
    d,cfg,td,s,w=fixture();d['parallel'][:]=True;cfg['filter_constraint_families']=False
    td['parallel']=torch.as_tensor(d['parallel'])
    ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    assert len(ext.owner)==0
    terms=ext.terms(s,w,td,torch.arange(3))
    for value in terms.values():assert value==0
    with torch.no_grad():ext.terms(s,w,td,torch.arange(3))
    assert ext.last['bucket_uv_nodes']==0


def test_fresh_init_has_no_bucket_topology_but_resume_checks_it(tmp_path):
    d,cfg,td,s,w=fixture();ext=b.BucketUV(d,cfg,'cpu')
    np.savez(tmp_path/'initial.npz',state=s.detach().numpy(),widths=w.detach().numpy())
    with np.load(tmp_path/'initial.npz') as saved:ext.restore(saved)
    ext.update(s,w,td,0)
    np.savez(tmp_path/'bucket.npz',**ext.arrays())
    legacy=b.BucketUV(d,dict(cfg,bucket_uv_perpendicular_nodes_only=False),'cpu')
    with np.load(tmp_path/'bucket.npz') as saved:
        with pytest.raises(ValueError,match='topology changed'):legacy.restore(saved)


def test_report_uses_saved_probabilities_and_fixed_mapping(tmp_path):
    import json
    d,cfg,td,s,w=fixture();d['links']=np.empty((0,4))
    ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    np.savez(tmp_path/'tensors.npz',**d)
    identity={k:d[k] for k in ('source_id','start','end','family')}
    np.savez(tmp_path/'initial.npz',state=s.detach().numpy(),widths=w.detach().numpy(),**identity)
    root=tmp_path/'solve';root.mkdir()
    cfg.update(tensor_cache=str(tmp_path/'tensors.npz'),initial_solution=str(tmp_path/'initial.npz'))
    (root/'config.json').write_text(json.dumps(cfg))
    np.savez(root/'solution.npz',state=s.detach().numpy(),widths=w.detach().numpy(),**identity,**ext.arrays())
    r=b.report(dict(solve_roots=[str(root)],output_root=str(tmp_path/'report')))
    m=r['rows'][0]['metrics']
    assert m['bucket_uv_same_sheet_mass']>0
    assert m['bucket_uv_max_offset_vx']==0
    assert m['bucket_uv_phase_incompatible_mass_fraction']==0


def test_unweighted_attachment_pulls_unused_disabled_buckets_to_zero():
    d,cfg,td,s,w=fixture();cfg['bucket_uv_attachment_probability_weighted']=False
    ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    with torch.no_grad():
        ext.values[0,0,0]+=100 # far outside current probability support
        ext.values[0,66,0]+=100
        s[:,3]=-30
    term=ext.terms(s,w,td,torch.arange(3))['bucket_uv_attachment']
    term.backward()
    assert ext.values.grad[0,0,0]>0
    torch.testing.assert_close(ext.values.grad[0,0,0],ext.values.grad[0,66,0])
    assert term.item()==pytest.approx(cfg['bucket_uv_attach_weight']*2*.01/len(ext.owner))


def test_absolute_positions_independent_of_shared_geometry():
    d,cfg,td,s,w=fixture();cfg['bucket_uv_attachment_probability_weighted']=False
    ext=b.BucketUV(d,cfg,'cpu');ext.update(s,w,td,0)
    before=ext.values.detach().clone()
    terms=ext.terms(s,w,td,torch.arange(3))
    width_grad=torch.autograd.grad(terms['bucket_uv_meeting'],w,allow_unused=True,retain_graph=True)[0]
    assert width_grad is None or torch.count_nonzero(width_grad)==0
    phase_grad=torch.autograd.grad(terms['bucket_uv_meeting'],s,retain_graph=True)[0]
    assert torch.count_nonzero(phase_grad[:,1:3])==0
    with torch.no_grad():w.mul_(1.1);s[:,1]+=.1;s[:,2]+=25
    ext.update(s,w,td,1)
    torch.testing.assert_close(ext.values,before)
    after=ext.terms(s,w,td,torch.arange(3))
    torch.testing.assert_close(after['bucket_uv_meeting'],terms['bucket_uv_meeting'])
    assert after['bucket_uv_attachment']>0
    after['bucket_uv_attachment'].backward()
    assert w.grad.abs().max()>0

def setup():
    d,cfg,td,s,w=fixture()
    cfg.update(bucket_uv_active_window=True,bucket_uv_attachment_probability_weighted=False)
    ext=b.BucketUV(d,cfg,'cpu')
    opt=torch.optim.Adam([s,ext.values],lr=.01);ext.bind_optimizer(opt)
    ext.update(s,w,td,0)
    return ext,opt,td,s,w


@pytest.mark.parametrize('reuse',[False,True])
def test_remap_preserves_overlap_resets_reentry_and_moments(reuse):
    ext,opt,td,s,w=setup()
    ext.config['bucket_uv_reuse_storage']=reuse
    original=ext.values
    sum(ext.terms(s,w,td,torch.arange(3)).values()).backward();opt.step()
    old_labels=ext.labels.clone();old=ext.values.detach().clone()
    moment=opt.state[ext.values]['exp_avg'].clone()
    with torch.no_grad():s[:,0]+=1
    ext.update(s,w,td,1)
    assert torch.equal(ext.labels,old_labels+1)
    if reuse:assert ext.values is original
    torch.testing.assert_close(ext.values[:,:-1],old[:,1:])
    torch.testing.assert_close(opt.state[ext.values]['exp_avg'][:,:-1],moment[:,1:])
    assert opt.state[ext.values]['exp_avg'][:,-1].count_nonzero()==0
    torch.testing.assert_close(ext.values[:,-1],ext.shared_positions(s,w)[:,-1])
    with torch.no_grad():s[:,0]-=1;w.mul_(1.1)
    ext.update(s,w,td,2)
    torch.testing.assert_close(ext.values[:,0],ext.shared_positions(s,w)[:,0])
    assert opt.state[ext.values]['exp_avg'][:,0].count_nonzero()==0
    # Repeated backward works after parameter shape/storage remapping.
    opt.zero_grad();sum(ext.terms(s,w,td,torch.arange(3)).values()).backward();opt.step()


def test_inactive_attachment_and_bending_match_masked_full_ladder():
    ext,opt,td,s,w=setup()
    # Force differing per-fiber radii without another cavity update.
    ext.cavity.values=torch.tensor([.2,1.]);ext.refresh_window(s,w)
    with torch.no_grad():ext.values.add_(3)
    terms=ext.terms(s,w,td,torch.arange(3))
    terms['bucket_uv_attachment'].backward(retain_graph=True)
    assert ext.values.grad[~ext.active[ext.owner]].count_nonzero()==0
    d,cfg,_,_,_=fixture();cfg['bucket_uv_attachment_probability_weighted']=False
    full=b.BucketUV(d,cfg,'cpu');full.update(s,w,td,0)
    full.cavity.values=ext.cavity.values
    with torch.no_grad():
        nodes,slots=torch.where(ext.active[ext.owner])
        labels=ext.labels[ext.owner[nodes],slots]-full.low
        full.values[nodes,labels]=ext.values[nodes,slots]
    oracle=full.terms(s,w,td,torch.arange(3))
    for key in ('bucket_uv_attachment','bucket_uv_bending'):
        torch.testing.assert_close(terms[key],oracle[key])
    local_grads=torch.autograd.grad(terms['bucket_uv_attachment']+terms['bucket_uv_bending'],
                                   (ext.values,s,w),retain_graph=True)
    full_grads=torch.autograd.grad(oracle['bucket_uv_attachment']+oracle['bucket_uv_bending'],
                                  (full.values,s,w))
    torch.testing.assert_close(local_grads[0][nodes,slots],full_grads[0][nodes,labels])
    for left,right in zip(local_grads[1:],full_grads[1:]):
        torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-4)
    assert ext.count<full.count


def test_compact_checkpoint_roundtrip(tmp_path):
    ext,opt,td,s,w=setup()
    with torch.no_grad():ext.values.add_(2)
    np.savez(tmp_path/'s.npz',**ext.arrays())
    d,cfg,_,_,_=fixture();cfg['bucket_uv_active_window']=True
    other=b.BucketUV(d,cfg,'cpu')
    with np.load(tmp_path/'s.npz') as saved:other.restore(saved)
    torch.testing.assert_close(other.values,ext.values)
    assert torch.equal(other.labels,ext.labels)
    assert torch.equal(other.active,ext.active)


def test_compact_report_and_carry_audit(tmp_path):
    import json
    import pytest
    ext,opt,td,s,w=setup()
    d,cfg,_,_,_=fixture();d['links']=np.empty((0,4));cfg.update(ext.config)
    np.savez(tmp_path/'tensors.npz',**d)
    identity={k:d[k] for k in ('source_id','start','end','family')}
    np.savez(tmp_path/'initial.npz',state=s.detach().numpy(),widths=w.detach().numpy(),**identity)
    root=tmp_path/'solve';root.mkdir()
    cfg.update(tensor_cache=str(tmp_path/'tensors.npz'),initial_solution=str(tmp_path/'initial.npz'))
    (root/'config.json').write_text(json.dumps(cfg))
    saved=dict(state=s.detach().numpy(),widths=w.detach().numpy(),**identity,**ext.arrays())
    np.savez(root/'solution.npz',**saved)
    settings=dict(solve_roots=[str(root)],output_root=str(tmp_path/'report'))
    assert b.report(settings)['rows'][0]['metrics']['bucket_uv_max_offset_vx']==0
    saved['bucket_uv_carry']=saved['bucket_uv_carry']+1
    np.savez(root/'solution.npz',**saved)
    with pytest.raises(ValueError,match='original fixed matching'):b.report(settings)
