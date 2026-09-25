"""One Gaussian sampler across learning, tracing, diagnostics and evaluation."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from test_single_path import batch, config, run
from vesuvius.neural_tracing.fiber_follow.model import FollowNet, initial_residuals
from vesuvius.neural_tracing.fiber_follow.sampling import trace_generator, trace_noise
from vesuvius.neural_tracing.fiber_follow.train import (
    load_checkpoint, save_checkpoint, resolve_sampler_mode, initialized_config, optimizer_update,
    compile_training_model, training_rng_state, restore_training_rng, resume_training,
)
from vesuvius.neural_tracing.fiber_follow.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer, TraceParams


def test_prior_and_shared_training_inference_curve():
    cfg=config(sampler_mode='gaussian')
    m=FollowNet(cfg); b=batch(cfg)
    generator=lambda:torch.Generator().manual_seed(14)
    noise=initial_residuals(cfg,2,'cpu',generator())
    expected=torch.randn(2,1,4,2,generator=generator())
    torch.testing.assert_close(noise,expected,rtol=0,atol=0)
    training=m.generate_training_curve(b['x'],b['hist'],b['hmask'],initial_noise=noise)
    actual=run(m,b,initial_noise=noise,return_steps=True)
    torch.testing.assert_close(training,actual['points'],rtol=0,atol=0)
    torch.testing.assert_close(actual['denoising_steps'][:,0,:,:2],noise[:,0],rtol=0,atol=0)
    torch.testing.assert_close(actual['points'],run(m,b,generator=generator())['points'],rtol=0,atol=0)
    other=run(m,b,generator=torch.Generator().manual_seed(15))
    assert not torch.equal(actual['points'],other['points'])
    confidence=m.training_forward(b['x'],b['hist'],b['hmask'],b,training)
    torch.testing.assert_close(confidence['points'],actual['points'],rtol=0,atol=0)
    torch.testing.assert_close(confidence['confidence'],actual['confidence'],rtol=0,atol=0)
    with pytest.raises(ValueError,match='initial_noise'):
        run(m,b,initial_noise=noise[:,0])


def test_sampler_mode_legacy_loading_new_default_and_resume_guard(tmp_path):
    assert resolve_sampler_mode()=='gaussian'
    cfg=config();m=FollowNet(cfg).eval();b=batch(cfg)
    sample=SampleConfig(crop=CropSpec(depth=20,width=12,behind=10),n_history=8)
    spec=FiberVolumeSpec('unused',ct_zarr='unused',inputs='ct+presence')
    path=tmp_path/'legacy.pt'
    save_checkpoint(path,m,m,spec,sample)
    ck=torch.load(path,weights_only=False);ck['model_cfg'].pop('sampler_mode');torch.save(ck,path)
    old,*_=load_checkpoint(path,'cpu')
    assert old.cfg.sampler_mode=='zero'
    torch.testing.assert_close(run(m,b)['points'],run(old,b)['points'],rtol=0,atol=0)
    assert resolve_sampler_mode(resume=ck)=='zero'
    with pytest.raises(ValueError,match='--init-from'):
        resolve_sampler_mode('gaussian',ck)
    new_cfg=initialized_config(ck,config(sampler_mode='gaussian'))
    new=FollowNet(new_cfg).eval();new.load_state_dict(ck['ema'])
    assert new.cfg.flow_sigma==cfg.flow_sigma and new.cfg.sampler_mode=='gaussian'
    save_checkpoint(path,new,new,spec,sample)
    loaded,*_=load_checkpoint(path,'cpu')
    assert loaded.cfg.sampler_mode=='gaussian'
    noise=initial_residuals(new_cfg,2,'cpu')
    torch.testing.assert_close(run(new,b,initial_noise=noise)['points'],run(loaded,b,initial_noise=noise)['points'])


def test_trace_streams_independent_of_batching_other_traces_and_global_rng():
    cfg=config(sampler_mode='gaussian')
    seeds=np.array([[1.,2.,3.],[4.,5.,6.]]);heads=np.array([[0.,0.,1.],[1.,0.,0.]])
    make=lambda i:trace_generator(52,seeds[i],heads[i])
    a,b=make(0),make(1)
    grouped=trace_noise(cfg,[a,b],'cpu')
    torch.randn(200)
    torch.testing.assert_close(grouped[:1],trace_noise(cfg,[make(0)],'cpu'),rtol=0,atol=0)
    # One trace can consume arbitrary extra draws without changing the other.
    solo=make(1);trace_noise(cfg,[solo],'cpu')
    for _ in range(7): trace_noise(cfg,[a],'cpu')
    torch.testing.assert_close(trace_noise(cfg,[b],'cpu'),trace_noise(cfg,[solo],'cpu'),rtol=0,atol=0)
    assert not torch.equal(grouped[:1],trace_noise(cfg,[trace_generator(53,seeds[0],heads[0])],'cpu'))


def test_tracer_delivers_same_noise_when_batch_regrouped_or_another_trace_stops(tmp_path,monkeypatch):
    from test_history_confidence import volume
    vol=volume(tmp_path);m=FollowNet(config(sampler_mode='gaussian'))
    seen=[]
    def forward(x,hist,hmask,*,initial_noise):
        seen.append(initial_noise.clone())
        points=torch.zeros(len(hist),4,3);points[...,2]=torch.arange(1,5)
        return dict(points=points,confidence=torch.ones(len(hist),4))
    monkeypatch.setattr(m,'forward',forward)
    tracer=ModelTracer(m,vol,CropSpec(depth=20,width=12,behind=10),8,
                       TraceParams(n_commit=1,max_len=3,seed=14),device='cpu')
    seeds=np.array([[8.,8.,5.],[9.,8.,5.]]);heads=np.array([[0.,0.,1.],[0.,0.,1.]])
    try:
        tracer.trace(seeds,heads,on_decision=lambda i,state:i!=0)
        batched=[seen[0][1:].clone(),*seen[1:]];seen.clear()
        tracer.trace(seeds[1:],heads[1:])
        assert len(batched)==len(seen)==3
        for a,b in zip(batched,seen):torch.testing.assert_close(a,b,rtol=0,atol=0)
    finally:tracer.close()


def test_diagnostics_and_logging_do_not_change_stochastic_updates():
    torch.manual_seed(15);cfg=config(sampler_mode='gaussian')
    first=FollowNet(cfg);second=copy.deepcopy(first);b=batch(cfg)
    ema1=copy.deepcopy(first);ema2=copy.deepcopy(second)
    opt1=torch.optim.SGD(first.parameters(),lr=.001);opt2=torch.optim.SGD(second.parameters(),lr=.001)
    state=training_rng_state()
    loss1,_,_=optimizer_update(first,ema1,opt1,[b],2000,.001,device='cpu',compute_metrics=False)
    next_rng=torch.get_rng_state()
    restore_training_rng(state)
    run(ema2,b,return_steps=True,generator=torch.Generator().manual_seed(2000))
    loss2,_,_=optimizer_update(second,ema2,opt2,[b],2000,.001,device='cpu',compute_metrics=True)
    assert loss1==loss2
    assert torch.equal(torch.get_rng_state(),next_rng)
    for p,q in zip(first.parameters(),second.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_sampler_rng_checkpoint_restores_next_update(tmp_path):
    torch.manual_seed(11);cfg=config(sampler_mode='gaussian');m=FollowNet(cfg);ema=copy.deepcopy(m)
    opt=torch.optim.AdamW(m.parameters(),lr=.001);data=batch(cfg)
    optimizer_update(m,ema,opt,[data],1,.001,device='cpu')
    sample=SampleConfig(crop=CropSpec(depth=20,width=12,behind=10),n_history=8)
    path=tmp_path/'resume.pt';spec=FiberVolumeSpec('unused',ct_zarr='unused',inputs='ct+presence')
    save_checkpoint(path,m,ema,spec,sample,dict(step=1,optimizer=opt.state_dict(),rng=training_rng_state()))
    loss1,_,_=optimizer_update(m,ema,opt,[data],2,.001,device='cpu')
    restored=FollowNet(cfg);average=copy.deepcopy(restored);optimizer=torch.optim.AdamW(restored.parameters(),lr=.001)
    ck=torch.load(path,weights_only=False);resume_training(ck,restored,average,optimizer)
    loss2,_,_=optimizer_update(restored,average,optimizer,[data],2,.001,device='cpu')
    assert loss1==loss2
    for p,q in zip(m.parameters(),restored.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_calibration_locks_sampling_repeats_and_bootstrap_retains_them():
    path=Path(__file__).parents[1]/'scripts/evaluate_single_path.py'
    spec=importlib.util.spec_from_file_location('eval_stochastic',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    choose=module.evaluation_sampling_seeds
    assert choose('gaussian')==[0,1,2] and choose('zero')==[0]
    assert choose('gaussian',selection={'sampling_seeds':[8,9]})==[8,9]
    with pytest.raises(ValueError,match='locked'):choose('gaussian',[0],{'sampling_seeds':[8,9]})
    with pytest.raises(ValueError,match='distinct'):choose('gaussian',[1,1])
    from vesuvius.neural_tracing.fiber_follow.experiment import paired_bootstrap
    with pytest.raises(ValueError,match='Duplicate'):
        paired_bootstrap([dict(fiber=0,t0=0,sign=1)]*2,[dict(fiber=0,t0=0,sign=1)]*2)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gaussian_compiled_training_and_explicit_noise_agree():
    torch.manual_seed(2);cfg=config(sampler_mode='gaussian');m=FollowNet(cfg).cuda();ema=copy.deepcopy(m)
    b=batch(cfg);opt=torch.optim.AdamW(m.parameters(),lr=.001)
    gpu={k:v.cuda() for k,v in b.items()}
    noise=initial_residuals(cfg,2,'cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        eager=m.generate_training_curve(gpu['x'],gpu['hist'],gpu['hmask'],initial_noise=noise)
    compile_training_model(m)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        compiled=m.generate_training_curve(gpu['x'],gpu['hist'],gpu['hmask'],initial_noise=noise)
    torch.testing.assert_close(eager,compiled,atol=.05,rtol=.05)
    for step in (1,2):
        loss,_,_=optimizer_update(m,ema,opt,[b],step,.001,device='cuda')
        assert np.isfinite(loss)
    assert m.flow.velocity.weight.grad.abs().sum()>0
