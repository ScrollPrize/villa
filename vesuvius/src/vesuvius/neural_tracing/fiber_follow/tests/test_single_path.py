"""Single-path architecture, supervision, tracing, replay and learning regressions."""
import copy
from dataclasses import replace,asdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_follow import data as D
from vesuvius.neural_tracing.fiber_follow.model import FollowNet,FollowNetConfig,ARCHITECTURE,flow_targets
from vesuvius.neural_tracing.fiber_follow.policy import commit_prefix,recovery_allowed
from vesuvius.neural_tracing.fiber_follow.supervision import prefix_labels,loss_fn,masked_bce
from vesuvius.neural_tracing.fiber_follow.train import load_checkpoint,save_checkpoint,optimizer_update
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec,arclength,frame_from_heading
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer,TraceParams
from vesuvius.neural_tracing.fiber_follow.collect import DecisionCollector
from vesuvius.neural_tracing.fiber_follow.replay import import_states
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec


def config(**kw):
    args=dict(in_channels=3,depth=20,width=12,behind=10,widths=(4,8),hidden=16,n_future=4,
              hist_points=8,hist_stride=1,recent_history_points=8,flow_layers=2,flow_heads=2,
              flow_draws=2,flow_steps=4,flow_sigma=((1.,1.),)*4)
    args.update(kw)
    return FollowNetConfig(**args)


def batch(cfg,B=2):
    hist=torch.zeros(B,cfg.hist_points*cfg.hist_stride,3)
    hist[...,2]=-torch.arange(1,hist.shape[1]+1)
    q=4*(cfg.n_future-1)+1
    return dict(x=torch.randn(B,3,cfg.depth,cfg.width,cfg.width),hist=hist,hmask=torch.ones(B,hist.shape[1]),
        plane_ab=torch.zeros(B,cfg.n_future,2),plane_mask=torch.ones(B,cfg.n_future),
        dense_ab=torch.zeros(B,q,2),dense_mask=torch.ones(B,q),offtrack=torch.zeros(B),
        endpoint_known=torch.zeros(B),end_local=torch.zeros(B,3),source=torch.zeros(B),stratum=torch.full((B,),-1),
        gt_history=torch.zeros(B,cfg.recent_history_points+1,3),gt_history_mask=torch.ones(B,cfg.recent_history_points+1))


def run(model,b,**kwargs): return model(b['x'],b['hist'],b['hmask'],**kwargs)


def test_one_deterministic_curve_nine_evaluations_and_shared_gradient_routes():
    torch.manual_seed(3);cfg=config();m=FollowNet(cfg);b=batch(cfg)
    calls=[];hook=m.flow.register_forward_hook(lambda *a:calls.append(1))
    out=run(m,b,return_steps=True);hook.remove()
    assert len(calls)==9
    assert out['points'].shape==(2,4,3) and out['confidence'].shape==(2,4)
    assert out['denoising_steps'].shape==(2,5,4,3)
    assert not {'candidates','ranks','samples','support'} & out.keys()
    torch.testing.assert_close(out['points'][...,2],torch.arange(1,5).float().expand(2,-1))
    torch.testing.assert_close(out['denoising_steps'][:,0,:,:2],torch.zeros(2,4,2))
    torch.manual_seed(999);other=run(m,b)
    torch.testing.assert_close(out['points'],other['points'],rtol=0,atol=0)
    assert (out['confidence'][:,1:]<=out['confidence'][:,:-1]).all()
    assert not out['points'].requires_grad
    out['confidence_logits'].sum().backward()
    for module in (m.encoders[0][0],m.history[0],m.flow.blocks[0].query,m.flow.confidence_head):
        assert module.weight.grad is not None and module.weight.grad.abs().sum()>0
    assert m.flow.velocity.weight.grad is None


def test_joint_attention_full_history_and_fixed_masked_coordinates():
    torch.manual_seed(4);cfg=config(hist_points=128,recent_history_points=128);m=FollowNet(cfg);b=batch(cfg,1)
    original=b['hist'].clone()
    feat,ctx,deep=m.encode(b['x'],b['hist'],b['hmask'],return_deep=True)
    fixed=m.flow.conditioning(feat,deep,b['hist'],b['hmask'],m.sampling_grid)
    y=torch.zeros(1,1,4,2,requires_grad=True)
    v=m.flow(feat,ctx,y,torch.ones(1,1),fixed,m.sampling_grid)
    grad=torch.autograd.grad(v[0,0,0,0],y,retain_graph=True)[0]
    assert grad[0,0,-1].abs().sum()>0
    tokens=fixed['history'];tokens.retain_grad()
    v[0,0,0,0].backward()
    assert tokens.grad[0,-1].abs().sum()>0
    torch.testing.assert_close(b['hist'],original,rtol=0,atol=0)
    assert fixed['supported'][0,:10].all() and not fixed['supported'][0,-1]
    b['hmask'][:,4:]=0
    out=run(m,b)
    b['hist'][:,4:]=float('nan')
    altered=run(m,b)
    torch.testing.assert_close(out['points'],altered['points'],rtol=0,atol=0)
    torch.testing.assert_close(out['confidence'],altered['confidence'],rtol=0,atol=0)


@pytest.mark.parametrize('case',['no_history','truncated','unknown','departed','crop_censored'])
def test_masked_flow_and_gt_history_never_conditions_inputs(case):
    torch.manual_seed(12);cfg=config();m=FollowNet(cfg);b=batch(cfg)
    if case=='no_history': b['hmask'].zero_()
    if case=='truncated': b['hmask'][:,3:]=0
    if case=='unknown': b['plane_mask'].zero_();b['dense_mask'].zero_()
    if case=='departed': b['offtrack'].fill_(1)
    if case=='crop_censored': b['plane_ab'].fill_(100)
    g=lambda:torch.Generator().manual_seed(17)
    out=run(m,b,targets=b,generator=g())
    b['gt_history'].fill_(float('nan'))
    other=run(m,b,targets=b,generator=g())
    torch.testing.assert_close(out['flow_loss'],other['flow_loss'],rtol=0,atol=0)
    torch.testing.assert_close(out['points'],other['points'],rtol=0,atol=0)
    if case in ('unknown','departed','crop_censored'): assert out['flow_loss']==0
    out['flow_loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_labels_describe_final_detached_curve_balanced_bce_and_ramp():
    cfg=config(n_future=8,flow_sigma=((1.,1.),)*8);b=batch(cfg)
    p=torch.zeros(2,8,3,requires_grad=True)
    with torch.no_grad(): p[...,2]=torch.arange(1,9);p[1,:,0]=3
    logits=torch.zeros(2,8,requires_grad=True)
    out=dict(points=p,confidence_logits=logits,confidence=logits.sigmoid(),flow_loss=logits.sum()*0,
             flow_known_fraction=torch.ones(()),flow_censored_fraction=torch.zeros(()))
    labels,mask,_=prefix_labels(p,b)
    assert labels[0].all() and not labels[1].any() and mask.all()
    loss,_=loss_fn(out,b,cfg,update=1000)
    expected=.5*(.5*masked_bce(logits[:,:4],labels[:,:4],mask[:,:4])+.5*masked_bce(logits,labels,mask))
    torch.testing.assert_close(loss,expected);loss.backward()
    assert p.grad is None and logits.grad is not None
    b['dense_mask'][:,5:]=0
    target,mask,_=prefix_labels(p,b)
    assert not mask[0,2:].any() and mask[1].all()
    b['endpoint_known'].fill_(1);b['end_local'][:,2]=2
    target,mask,_=prefix_labels(p,b)
    assert mask.all() and not target[:,2:].any()
    b['offtrack'].fill_(1)
    target,mask,_=prefix_labels(p,b)
    assert mask.all() and not target.any()


@pytest.mark.parametrize('distance,allowed',[(3.,True),(7.,False),(float('nan'),False)])
def test_first_connection_and_unknown_end(distance,allowed):
    cfg=config();b=batch(cfg,1);p=torch.zeros(1,4,3);p[...,2]=torch.arange(1,5);p[:,0,0]=distance
    b['dense_mask'].zero_()
    assert recovery_allowed(p).item()==allowed
    target,mask,_=prefix_labels(p,b)
    if not allowed: assert mask.all() and not target.any()
    count,ok=commit_prefix(p,torch.tensor([[.9,.8,.85,.2]]),.7)
    assert count.item()==(3 if allowed else 0)
    with pytest.raises(ValueError): commit_prefix(p,torch.ones(1,4),n_commit=5)


def test_checkpoint_roundtrip_rejects_legacy_and_beam_parameter_names(tmp_path):
    cfg=config();m=FollowNet(cfg).eval();b=batch(cfg,1);expected=run(m,b)
    crop=CropSpec(depth=20,width=12,behind=10)
    sample=D.SampleConfig(crop=crop,n_history=8,recent_history_points=8,n_future=4)
    spec=FiberVolumeSpec('unused',ct_zarr='unused',ct_level=1,ct_grid_scale=8,inputs='ct+presence')
    path=tmp_path/'new.pt';save_checkpoint(path,m,m,spec,sample)
    loaded,*_=load_checkpoint(path,'cpu');actual=run(loaded,b)
    torch.testing.assert_close(expected['points'],actual['points'],rtol=0,atol=0)
    ck=torch.load(path,weights_only=False);ck['architecture']='future_flow_v10';torch.save(ck,path)
    with pytest.raises(ValueError,match='single_path_flow_v11'): load_checkpoint(path,'cpu')
    from vesuvius.neural_tracing.fiber_follow.beam.model import BeamRankNet,BeamNetConfig
    old=Path(__file__).parents[1]/'output/single_path_v11_baseline/beam_compatibility.pt'
    if old.exists():
        fixture=torch.load(old,weights_only=False)
        beam=BeamRankNet(BeamNetConfig(**fixture['cfg'])).eval();beam.load_state_dict(fixture['weights'])
        for key,value in beam(*fixture['inputs']).items():
            torch.testing.assert_close(value,fixture['output'][key],rtol=0,atol=0)


def fiber():
    p=np.c_[np.ones(240)*50,np.ones(240)*50,np.arange(240)+100.]
    return D.TracedFiber('original',p,arclength(p),'',source_hash='s')


def states(f,n=10):
    t=np.arange(n)+80.;pos=f.points[t.astype(int)].copy();pos[:,0]+=np.resize([.5,1.25,1.75,2.5,np.nan],n)
    pos[~np.isfinite(pos)]=50
    return dict(fiber_idx=np.zeros(n,int),t=t,reverse=np.zeros(n,bool),pos=pos,
        frame=np.tile(np.eye(3),(n,1,1)),hist=np.tile(f.points[60:68][::-1],(n,1,1)),hmask=np.ones((n,8)),
        offtrack=np.arange(n)%5==4,hard=np.ones(n,bool),exploratory=np.zeros(n,bool))


def test_replay_import_discards_predictions_deduplicates_checks_holdout_and_provenance(tmp_path):
    f=fiber();arrays=states(f);old=tmp_path/'old.npz'
    np.savez(old,__metadata__=json.dumps(dict(version=4,fibers=D.fiber_manifest([f]),provenance={})),
             candidates=np.zeros((10,7,64,3)),**arrays)
    with pytest.raises(ValueError,match='import'): D.OnPolicyStates.load(old)
    spec=FiberVolumeSpec('unused');cfg=D.SampleConfig(crop=CropSpec(depth=20,width=12,behind=10),n_history=8,recent_history_points=8,n_future=4)
    bank=import_states([old,old],[f],cfg,None,spec,limit=20000)
    assert len(bank)==10 and bank.provenance['counts']['duplicate']==10
    assert not hasattr(bank,'candidates') and np.isfinite(bank.drift[~bank.offtrack]).all()
    np.testing.assert_array_equal(np.sort(bank.source_row),np.arange(10))
    path=tmp_path/'bank.npz';bank.save(path);bank=D.OnPolicyStates.load(path)
    bank.validate_fibers([f]);assert bank.hist.shape==(10,8,3)
    with pytest.raises(ValueError,match='No unique eligible'):
        import_states([old],[f],cfg,D.ZBand(0,1000),spec)
    pools=D.replay_pools([bank]);assert all(pools)
    ds=D.FollowDataset([f],spec,cfg,None,fixed=[bank],onpolicy=[bank])
    rng=np.random.default_rng(3);counts=np.zeros(3,int);departed=np.zeros(3,int)
    for _ in range(20000):
        draw=ds.draw_replay(rng);source=0 if draw is None else draw[0];counts[source]+=1
        if draw is not None: departed[source]+=draw[1]==4
    np.testing.assert_allclose(counts/counts.sum(),[.5,.25,.25],atol=.015)
    np.testing.assert_allclose(departed[1:]/counts[1:],[.1,.1],atol=.02)
    ds._set_replay([]);assert all(ds.fixed_pools) and not any(ds.recent_pools)


def test_accumulation_steps_once_and_ema_matches_effective_batch(monkeypatch):
    cfg=config();a=FollowNet(cfg);b=copy.deepcopy(a);ea=copy.deepcopy(a);eb=copy.deepcopy(b)
    data=batch(cfg,8)
    data['plane_mask'][0].zero_();data['dense_mask'][1,4:]=0
    data['offtrack'][2]=1
    # Remove flow's stochastic draws to isolate accumulation arithmetic.
    def forward(self,x,hist,hmask,**kw):
        logits=self.flow.confidence_head.weight.mean()*x.mean((1,2,3,4))[:,None].expand(-1,4)
        points=x.new_zeros(len(x),4,3);points[...,2]=torch.arange(1,5)
        return dict(points=points,confidence_logits=logits,confidence=logits.sigmoid(),
                    flow_loss=logits.square().mean(),flow_known_fraction=x.new_ones(()),flow_censored_fraction=x.new_zeros(()))
    def training_forward(self,x,hist,hmask,targets,points):
        out=forward(self,x,hist,hmask)
        mask=flow_targets(targets,self.cfg)[1]
        out['flow_loss']=(out['confidence_logits'].square()*mask).sum()/mask.sum().clamp_min(1)
        out['flow_known_count']=mask.sum()
        return out
    def generate(self,x,hist,hmask,*,return_steps=False):
        points=forward(self,x,hist,hmask)['points']
        return (points,points[:,None].expand(-1,cfg.flow_steps+1,-1,-1)) if return_steps else points
    monkeypatch.setattr(FollowNet,'generate_training_curve',generate)
    monkeypatch.setattr(FollowNet,'training_forward',training_forward)
    oa=torch.optim.SGD(a.parameters(),lr=.01);ob=torch.optim.SGD(b.parameters(),lr=.01)
    _,stats_a,_=optimizer_update(a,ea,oa,[data],2000,.01,device='cpu')
    micro=[{k:v[i:i+2] for k,v in data.items()} for i in range(0,8,2)]
    _,stats_b,_=optimizer_update(b,eb,ob,micro,2000,.01,device='cpu')
    assert stats_a['refinement']==stats_b['refinement']
    for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y)
    for x,y in zip(ea.parameters(),eb.parameters()): torch.testing.assert_close(x,y)


def test_end_to_end_trace_collection_and_state_roundtrip(tmp_path,monkeypatch):
    from test_history_confidence import volume
    vol=volume(tmp_path)
    cfg=config();m=FollowNet(cfg)
    crop=CropSpec(depth=20,width=12,behind=10)
    sample=D.SampleConfig(crop=crop,n_history=8,recent_history_points=8,n_future=4)
    points=np.c_[np.ones(16)*8,np.ones(16)*8,np.arange(16)]
    f=D.TracedFiber('test',points,arclength(points),'')
    def forward(x,hist,hmask,**kw):
        p=x.new_zeros(len(x),4,3);p[...,2]=torch.arange(1,5)
        return dict(points=p,confidence=x.new_full((len(x),4),.9))
    monkeypatch.setattr(m,'forward',forward)
    tracer=ModelTracer(m,vol,crop,8,TraceParams(max_len=5),device='cpu')
    collector=DecisionCollector(f,0,5.,1,sample)
    try:
        paths,reasons=tracer.trace(np.array([[8.,8.,5.]]),np.array([[0.,0.,1.]]),on_decision=lambda i,s:collector(s))
    finally:tracer.close()
    assert reasons==['max_len'] and paths[0][-1,2]==10
    rows=collector.finish();assert rows and 'candidates' not in rows[0]
    bank=D.OnPolicyStates(manifest=D.fiber_manifest([f]),**{k:np.asarray([r[k] for r in rows]) for k in D.OnPolicyStates.FIELDS+tuple(D.OnPolicyStates.OPTIONAL)})
    path=tmp_path/'collected.npz';bank.save(path);loaded=D.OnPolicyStates.load(path);loaded.validate_fibers([f])
    assert len(loaded)==len(rows)


def test_synthetic_parallel_fibers_learn_identity_recovery():
    torch.manual_seed(91)
    cfg=config(depth=44,behind=32,hist_points=32,recent_history_points=32,flow_draws=4,flow_layers=1)
    m=FollowNet(cfg);b=batch(cfg)
    # Both samples have identical image evidence: two parallel fibers. Only
    # older observed history identifies the correct one; recent history drifts
    # toward the ambiguous midpoint. No GT history goes into the model.
    axis=torch.arange(cfg.width)-(cfg.width-1)/2
    image=(torch.exp(-((axis-2)/.6)**2)+torch.exp(-((axis+2)/.6)**2))
    b['x']=image[None,None,None,None].expand(2,3,cfg.depth,cfg.width,-1).clone()
    sign=torch.tensor([-1.,1.])
    b['hist'][...,0]=sign[:,None]*2*torch.linspace(0,1,32)[None]
    b['plane_ab'][...,0]=sign[:,None]*2;b['dense_ab'][...,0]=sign[:,None]*2
    target=b['plane_ab']
    with torch.no_grad():before=(run(m,b)['points'][...,:2]-target).square().mean().item()
    opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=1e-4)
    for _ in range(160):
        opt.zero_grad(set_to_none=True)
        f,c,d=m.encode(b['x'],b['hist'],b['hmask'],return_deep=True)
        fixed=m.flow.conditioning(f,d,b['hist'],b['hmask'],m.sampling_grid)
        loss=m.flow_loss(f,c,b['hist'],b['hmask'],b,fixed=fixed)['flow_loss']
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
    with torch.no_grad():after=(run(m,b)['points'][...,:2]-target).square().mean().item()
    assert after<before*.15 and after<.25,(before,after)


@pytest.mark.parametrize('explore,patience',[(0,1),(8,1),(0,3)])
def test_recovery_limit_cannot_be_overridden(tmp_path,monkeypatch,explore,patience):
    from test_history_confidence import volume
    vol=volume(tmp_path);m=FollowNet(config());seen=[]
    def forward(x,hist,hmask,**kw):
        p=x.new_zeros(len(x),4,3);p[...,2]=torch.arange(1,5);p[...,0]=7
        return dict(points=p,confidence=x.new_zeros(len(x),4))
    monkeypatch.setattr(m,'forward',forward)
    tracer=ModelTracer(m,vol,CropSpec(depth=20,width=12,behind=10),8,
                       TraceParams(explore_calls=explore,stop_patience=patience,max_len=10),device='cpu')
    try:
        paths,reasons=tracer.trace(np.array([[8.,8.,5.]]),np.array([[0.,0.,1.]]),on_decision=lambda i,s:seen.append(s))
    finally:tracer.close()
    assert reasons==['recovery_limit'] and len(paths[0])==1 and len(seen)==1


def test_unknown_future_tokens_do_not_affect_known_velocities_and_cpu_bf16():
    cfg=config();m=FollowNet(cfg);b=batch(cfg,1);b['hmask'].zero_()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        f,c,d=m.encode(b['x'],b['hist'],b['hmask'],return_deep=True)
        fixed=m.flow.conditioning(f,d,b['hist'],b['hmask'],m.sampling_grid)
        y=torch.randn(1,1,4,2);mask=torch.tensor([[1.,1.,0.,0.]])
        a=m.flow(f,c,y,torch.ones(1,1),fixed,m.sampling_grid,future_mask=mask)
        y[:,:,2:]=float('nan')
        z=m.flow(f,c,y,torch.ones(1,1),fixed,m.sampling_grid,future_mask=mask)
        torch.testing.assert_close(a[:,:,:2],z[:,:,:2],rtol=0,atol=0)
        assert torch.isfinite(a).all()


def test_midpoint_convergence_and_refresh_queries(monkeypatch):
    cfg=config();m=FollowNet(cfg);b=batch(cfg,1)
    f,c,d=m.encode(b['x'],b['hist'],b['hmask'],return_deep=True)
    fixed=m.flow.conditioning(f,d,b['hist'],b['hmask'],m.sampling_grid)
    queries=[]
    def field(features,context,y,t,fixed,sampling_grid,**kwargs):
        queries.append(y.detach().clone());return y+1
    monkeypatch.setattr(m.flow,'forward',field)
    errors=[]
    for steps in (2,4,8):
        m.cfg.flow_steps=steps
        y,_=m.refine(f,c,fixed)
        errors.append(abs(y[0,0,0,0].item()-(np.e-1)))
    assert errors[0]/errors[1]>3 and errors[1]/errors[2]>3
    assert not torch.equal(queries[0],queries[1])


def test_initial_recovery_state_preserves_exact_frame_and_history(tmp_path,monkeypatch):
    from test_history_confidence import volume
    vol=volume(tmp_path);m=FollowNet(config());seen=[]
    def forward(x,hist,hmask,**kw):
        p=x.new_zeros(len(x),4,3);p[...,2]=torch.arange(1,5)
        return dict(points=p,confidence=x.new_zeros(len(x),4))
    monkeypatch.setattr(m,'forward',forward)
    pos=np.array([[8.,8.,8.]]);frame=frame_from_heading([0,0,1],[0,1,0])
    h=np.tile([8.,8.,7.],(8,1));h[:,0]+=np.linspace(0,1,8)
    state=dict(hist=h,hmask=np.ones(8),frame=frame)
    tracer=ModelTracer(m,vol,CropSpec(depth=20,width=12,behind=10),8,TraceParams(),device='cpu')
    try:tracer.trace(pos,np.array([[0,0,1.]]),initial_states=[state],on_decision=lambda i,s:seen.append(s))
    finally:tracer.close()
    np.testing.assert_array_equal(seen[0]['hist'],h)
    np.testing.assert_array_equal(seen[0]['frame'],frame)


def test_count_aggregation_and_paired_bootstrap():
    import importlib.util
    path=Path(__file__).parents[1]/'scripts/evaluate_single_path.py'
    spec=importlib.util.spec_from_file_location('eval_single',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rows=[dict(drift=1.2,departed=False,four_known=True,four_correct=True,first_known=True,
               first_correct=True,would_stop=True)]
    table=mod.recovery_counts(rows)
    assert table['1-1.5']['four_known']==1 and table['1-1.5']['false_stops']==1
    assert table['2-3.5']['states']==0 and table['2-3.5']['four_known']==0
    from vesuvius.neural_tracing.fiber_follow.evaluate import score_trace
    from vesuvius.neural_tracing.fiber_follow.experiment import paired_bootstrap
    f=fiber();row=score_trace(f.points[30:60],f,30,1)
    row.update(fiber=0,t0=30,sign=1)
    result=paired_bootstrap([row],[row],repeats=20)
    assert all(v==[0.,0.,0.] for v in result.values())


def test_ema_ramp_and_resumable_checkpoint(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.train import (
        update_ema,resume_training,training_rng_state,read_checkpoint)
    from vesuvius.neural_tracing.fiber_follow.runloop import prepare_run_dir
    cfg=config();m=FollowNet(cfg);e=copy.deepcopy(m)
    with torch.no_grad():
        for p in m.parameters(): p.add_(1.)
    update_ema(e,m,0,.999)  # ramped decay .1 at the first update, not .999
    for a,b in zip(e.parameters(),m.parameters()): torch.testing.assert_close(a,b-.1)
    update_ema(e,m,10**6,.999)
    for a,b in zip(e.parameters(),m.parameters()): torch.testing.assert_close(a,b-.1*.999)
    b=batch(cfg,1);opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    run(m,b,targets=b)['flow_loss'].backward();opt.step()
    crop=CropSpec(depth=20,width=12,behind=10)
    sample=D.SampleConfig(crop=crop,n_history=8,recent_history_points=8,n_future=4)
    spec=FiberVolumeSpec('unused',ct_zarr='unused',ct_level=1,ct_grid_scale=8,inputs='ct+presence')
    torch.manual_seed(5);expected=torch.rand(3)
    torch.manual_seed(5)
    save_checkpoint(tmp_path/'last.pt',m,e,spec,sample,dict(step=7,replay_seen=3,optimizer=opt.state_dict(),rng=training_rng_state()))
    save_checkpoint(tmp_path/'source.pt',m,e,spec,sample,dict(step=7))
    m2=FollowNet(cfg);e2=copy.deepcopy(m2);opt2=torch.optim.AdamW(m2.parameters(),lr=1e-3)
    with pytest.raises(ValueError,match='optimizer'): resume_training(read_checkpoint(tmp_path/'source.pt','cpu'),m2,e2,opt2)
    assert resume_training(read_checkpoint(tmp_path/'last.pt','cpu'),m2,e2,opt2)==(7,3)
    torch.testing.assert_close(torch.rand(3),expected)
    for a,c in zip(m.parameters(),m2.parameters()): torch.testing.assert_close(a,c,rtol=0,atol=0)
    for a,c in zip(e.parameters(),e2.parameters()): torch.testing.assert_close(a,c,rtol=0,atol=0)
    for s,t in zip(opt.state_dict()['state'].values(),opt2.state_dict()['state'].values()):
        torch.testing.assert_close(s['exp_avg'],t['exp_avg'],rtol=0,atol=0)
    with pytest.raises(FileNotFoundError): prepare_run_dir(tmp_path,'missing',resume=True)
    (tmp_path/'run').mkdir();(tmp_path/'run'/'config.json').write_text('{}')
    with pytest.raises(FileExistsError): prepare_run_dir(tmp_path,'run')
    assert prepare_run_dir(tmp_path,'run',resume=True)==tmp_path/'run'
