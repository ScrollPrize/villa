"""Stop fallback, gate diagnostics and replay collection configuration."""
from types import SimpleNamespace

import numpy as np
import torch

from test_single_path import batch, config
from vesuvius.neural_tracing.fiber_follow.flow_matching.model import FollowNet
from vesuvius.neural_tracing.fiber_follow.flow_matching.supervision import loss_fn
from vesuvius.neural_tracing.fiber_follow.flow_matching.train import resolve_candidate_selection
from vesuvius.neural_tracing.fiber_follow.shared.online import OnlineCollector
from vesuvius.neural_tracing.fiber_follow.shared.policy import commit_prefix, select_candidate
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer, TraceParams


def proposals():
    points=torch.zeros(1,3,4,3)
    points[...,2]=torch.arange(1,5)
    confidence=torch.tensor([[[.49,.49,.49,.49],[.9,.9,.48,.48],[.95,.95,.95,.95]]])
    points[:,2,:,0]=10  # Ineligible first connection, despite high confidence.
    return points,confidence


def test_fallback_uses_an_acceptable_prefix_without_relaxing_limits():
    points,confidence=proposals()
    assert select_candidate(points,confidence,4).item()==0
    selected=select_candidate(points,confidence,4,stop_threshold=.5)
    assert selected.item()==1
    counts,_=commit_prefix(points[:,1],confidence[:,1],.5,4)
    assert counts.item()==2
    # No candidate qualifies: retain the old winner and let the gate stop.
    assert select_candidate(points,confidence,4,stop_threshold=.96).item()==0
    # An already acceptable original winner is unchanged.
    assert select_candidate(points,confidence,4,stop_threshold=.4).item()==0
    # At a shorter commit horizon, the original ranking already picks B.
    assert select_candidate(points,confidence,2,stop_threshold=.5).item()==1


def test_fallback_ties_prefer_longer_prefix_then_confidence_then_candidate_order():
    points=torch.zeros(1,4,4,3);points[...,2]=torch.arange(1,5)
    confidence=torch.tensor([[[.49]*4,[.9,.8,.1,.1],[.95,.7,.1,.1],[.8,.8,.8,.1]]])
    assert select_candidate(points,confidence,4,stop_threshold=.5).item()==3
    confidence[:,3]=confidence[:,1]
    assert select_candidate(points,confidence,4,stop_threshold=.5).item()==1


def test_model_selects_threshold_specific_curve_and_logs_its_labels(monkeypatch):
    cfg=config(scorer='passage',gaussian_candidates=2,selection_horizon=4,candidate_selection='stop_fallback')
    model=FollowNet(cfg);data=batch(cfg,1);points,confidence=proposals()
    monkeypatch.setattr(model,'score_candidates',lambda *args:torch.logit(confidence))
    # Use exact proposals to isolate policy from the learned generator.
    out=model._forward(data['x'],data['hist'],data['hmask'],targets=data,training_points=points,
                       confidence_threshold=.5,n_commit=4)
    assert out['selected_candidate'].item()==1
    torch.testing.assert_close(out['points'],points[:,1])
    neutral=model.training_forward(data['x'],data['hist'],data['hmask'],data,points)
    assert neutral['selected_candidate'].item()==0
    _,metrics=loss_fn(neutral,data,cfg,update=2000,n_commit=4)
    assert metrics['candidate_fallback_count_0.5']==1
    assert metrics['candidate_fallback_correct_0.5']==1
    assert metrics['false_stop_count_0.5']==0
    # Legacy policy stays unchanged even when a caller supplies a threshold.
    cfg.candidate_selection='prefix'
    assert model._forward(data['x'],data['hist'],data['hmask'],training_points=points,
                          confidence_threshold=.5)['selected_candidate'].item()==0
    assert resolve_candidate_selection(None,{'model_cfg':{}})=='prefix'
    assert resolve_candidate_selection('stop_fallback',{'model_cfg':{}})=='stop_fallback'
    assert resolve_candidate_selection(None,{'model_cfg':{'candidate_selection':'stop_fallback'}})=='stop_fallback'


def test_tracer_passes_actual_threshold_and_commit_limit(tmp_path,monkeypatch):
    from test_history_confidence import volume
    vol=volume(tmp_path)
    model=FollowNet(config(candidate_selection='stop_fallback'))
    seen=[]
    def forward(x,hist,hmask,**kwargs):
        seen.append(kwargs)
        points=x.new_zeros(len(x),4,3);points[...,2]=torch.arange(1,5)
        return dict(points=points,confidence=x.new_zeros(len(x),4))
    monkeypatch.setattr(model,'forward',forward)
    tracer=ModelTracer(model,vol,model.crop,8,TraceParams(confidence=.83,n_commit=2),device='cpu')
    try:
        _,reasons=tracer.trace(np.array([[8.,8.,8.]]),np.array([[0.,0.,1.]]))
    finally:
        tracer.close()
    assert reasons==['confidence']
    assert seen[0]['confidence_threshold']==.83 and seen[0]['n_commit']==2


def test_collector_forwards_fiber_diversity_options(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.shared import online
    commands=[]
    def start(command,**kwargs):
        commands.append(command)
        return SimpleNamespace(poll=lambda:None)
    monkeypatch.setattr(online.subprocess,'Popen',start)
    collector=OnlineCollector(tmp_path,'fibers',(1,2),'cpu',max_seeds=128,seeds_per_fiber=1)
    assert collector.launch(1000,lambda path:None)
    collector.log.close()
    command=commands[0]
    assert command[command.index('--max-seeds')+1]=='128'
    assert command[command.index('--seeds-per-fiber')+1]=='1'


def test_recovery_rows_use_the_candidate_selected_at_each_threshold():
    from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig, TracedFiber
    from vesuvius.neural_tracing.fiber_follow.shared.recovery import evaluate_recovery_states
    cfg=config(scorer='passage',gaussian_candidates=2,selection_horizon=4,candidate_selection='stop_fallback')
    sample=SampleConfig(crop=FollowNet(cfg).crop,n_history=8,recent_history_points=8,n_future=4)
    arc=np.arange(300,dtype=float)
    fiber=TracedFiber('line',np.c_[arc*0,arc*0,arc],arc,'')
    states=SimpleNamespace(validate_fibers=lambda fibers:None,fiber_idx=np.array([0]),
        pos=np.array([[0.,0.,150.]]),frame=np.eye(3)[None],
        hist=np.array([[[0.,0.,150.-k] for k in range(1,9)]]),hmask=np.ones((1,8)),
        t=np.array([150.]),reverse=np.array([False]),offtrack=np.array([False]),drift=np.array([0.]))
    class States:
        def __len__(self): return 1
        def __getattr__(self,name): return getattr(states,name)
    points,confidence=proposals();points[:,0,:,0]=3.
    class Model:
        def __init__(self): self.cfg=cfg
        def __call__(self,x,hist,hmask,**kwargs):
            assert kwargs['initial_noise'].shape==(1,3,4,2)
            return dict(points=points[:,0],confidence=confidence[:,0],candidate_points=points,
                        candidate_confidence=confidence)
    class Tracer:
        def __init__(self,*args,**kwargs): self.threshold=args[4].confidence
        def trace(self,pos,heading,initial_states):
            if self.threshold>.9: return [pos],['confidence']
            return [np.stack([pos[0],pos[0]+[0,0,1]])],['max_len']
        def close(self): pass
    rows,_=evaluate_recovery_states(Model(),None,States(),[fiber],sample,thresholds=(.5,.96),
        n_commit=4,tracer_class=Tracer,batch_builder=lambda items,vol:batch(cfg,1))
    assert rows[0]['first_correct'] and not rows[0]['would_stop']
    assert not rows[1]['first_correct'] and rows[1]['would_stop']
