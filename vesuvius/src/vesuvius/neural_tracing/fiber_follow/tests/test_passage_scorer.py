"""General scorer contract, mixed flow proposals, supervision and checkpointing."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_single_path import config, batch, run
from vesuvius.neural_tracing.fiber_follow.model import FollowNet, initial_residuals, ARCHITECTURE
from vesuvius.neural_tracing.fiber_follow.passage_scorer import PassageScorer, sample_path_features
from vesuvius.neural_tracing.fiber_follow.supervision import candidate_prefix_labels, prefix_labels, loss_fn
from vesuvius.neural_tracing.fiber_follow.train import (
    optimizer_update, save_checkpoint, load_checkpoint, resolve_scorer_options,
    compile_training_model, main, training_rng_state, restore_training_rng, resume_training,
)
from vesuvius.neural_tracing.fiber_follow.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.sampling import trace_generator, trace_noise


def mixed_config(**kwargs):
    return config(scorer='passage', gaussian_candidates=4, selection_horizon=2, **kwargs)


@pytest.mark.parametrize('n', [4, 16, 48])
def test_general_scorer_has_no_model_or_horizon_dependency(n):
    torch.manual_seed(1)
    head = PassageScorer(7, 5, 16, 2).eval()
    evidence = torch.randn(2, 3, n, 7, requires_grad=True)
    context = torch.randn(2, 1, n, 5, requires_grad=True)
    paths = torch.randn(2, 3, n, 3, requires_grad=True)
    frontier = torch.zeros(2, 3)
    scores = head(evidence, context, paths, frontier)
    assert scores.shape == (2, 3, n) and torch.isfinite(scores).all()
    order = [2, 0, 1]
    torch.testing.assert_close(head(evidence[:, order], context, paths[:, order], frontier), scores[:, order])
    scores.sum().backward()
    assert evidence.grad.abs().sum() > 0 and context.grad.abs().sum() > 0
    assert paths.grad is None


def test_deep_evidence_samples_actual_stride_centers_and_support():
    crop = CropSpec(depth=20, width=12, behind=10, spacing=1.)
    z, y, x = torch.meshgrid(torch.arange(5.), torch.arange(3.), torch.arange(3.), indexing='ij')
    features = (x+10*y+100*z)[None, None]
    # At encoder stride 4, these coordinates hit lattice index [z=2,y=1,x=1].
    points = torch.tensor([[[-1.5, -1.5, -2.], [8., 0., 0.]]])
    sampled = sample_path_features(features, points, crop, stride=4)
    torch.testing.assert_close(sampled[0,0], torch.tensor([211.,1.]))
    assert sampled[0,1,-1] == 0


def test_mixed_proposals_keep_zero_and_train_all_heads_without_coordinate_gradients():
    torch.manual_seed(3)
    cfg = mixed_config(); model = FollowNet(cfg); data = batch(cfg)
    noise = initial_residuals(cfg, 2, 'cpu', torch.Generator().manual_seed(14))
    assert noise.shape == (2, 5, 4, 2) and noise[:,0].eq(0).all()
    assert noise[:,1:].abs().sum() > 0
    paths = model.generate_training_curve(data['x'], data['hist'], data['hmask'], initial_noise=noise)
    out = run(model, data, initial_noise=noise, return_steps=True)
    torch.testing.assert_close(out['candidate_points'], paths, rtol=0, atol=0)
    assert out['points'].shape == (2,4,3) and out['denoising_steps'].shape == (2,5,4,3)
    chosen = out['selected_candidate']
    torch.testing.assert_close(out['points'], paths[torch.arange(2),chosen])
    torch.testing.assert_close(out['denoising_steps'][:,-1], out['points'])
    assert (out['candidate_confidence'][...,1:] <= out['candidate_confidence'][...,:-1]).all()
    training = model.training_forward(data['x'], data['hist'], data['hmask'], data, paths)
    torch.testing.assert_close(training['candidate_logits'], out['candidate_logits'], rtol=0, atol=0)
    out['candidate_logits'].sum().backward()
    for weight in (model.encoders[0][0].weight, model.history[0].weight,
                   model.flow.blocks[0].query.weight, model.flow.confidence_head.passage_score[-1].weight):
        assert weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    assert model.flow.velocity.weight.grad is None and not paths.requires_grad
    solo = FollowNet(config(scorer='passage', selection_horizon=2))
    solo.load_state_dict(model.state_dict())
    torch.testing.assert_close(run(solo,data)['points'], paths[:,0], atol=2e-6, rtol=2e-6)


def test_selection_uses_commit_prefix_recovery_eligibility_and_zero_tie_break(monkeypatch):
    cfg = mixed_config(); model = FollowNet(cfg); data = batch(cfg, 1)
    paths = torch.zeros(1,5,4,3); paths[...,2] = torch.arange(1,5)
    paths[:,1,:,0] = .5; paths[:,2,:,0] = 10
    logits = torch.tensor([[[2., 2., 2., 6.], [3.,3.,-5.,-5.], [8.,8.,8.,8.],
                            [0.,0.,0.,0.], [0.,0.,0.,0.]]])
    monkeypatch.setattr(model,'score_candidates',lambda *args: logits)
    out = model.training_forward(data['x'],data['hist'],data['hmask'],data,paths)
    assert out['selected_candidate'].item() == 1  # 2 is geometrically ineligible; far failure doesn't veto prefix.
    assert out['confidence'][0,0] > .9 and out['confidence'][0,-1] < .01
    logits.fill_(0)
    assert model.training_forward(data['x'],data['hist'],data['hmask'],data,paths)['selected_candidate'].item() == 0


def test_all_candidates_receive_censored_prefix_supervision():
    cfg = mixed_config(); data = batch(cfg, 3)
    paths = torch.zeros(3,5,4,3); paths[...,2] = torch.arange(1,5)
    paths[:,1,:,0] = 3.; paths[:,2,1,0] = 3.
    data['dense_mask'][1].zero_(); data['offtrack'][2] = 1
    y, known, _ = candidate_prefix_labels(paths, data)
    for i in range(5):
        a, b, _ = prefix_labels(paths[:,i],data)
        torch.testing.assert_close(y[:,i],a); torch.testing.assert_close(known[:,i],b)
    assert not known[1].any() and not y[2].any() and known[2].all()
    assert y[0,2,-1] == 0  # returning to the original fiber cannot erase a failure.
    logits = torch.zeros(3,5,4,requires_grad=True)
    out = dict(points=paths[:,0], confidence_logits=logits[:,0], confidence=logits[:,0].sigmoid(),
               candidate_points=paths, candidate_logits=logits, flow_loss=logits.sum()*0,
               flow_known_fraction=torch.ones(()), flow_censored_fraction=torch.zeros(()))
    loss, metrics = loss_fn(out,data,cfg,update=2000,n_commit=2)
    loss.backward()
    assert logits.grad[0,0,0] < 0 and logits.grad[0,1,0] > 0
    assert logits.grad[1].eq(0).all() and logits.grad[2].gt(0).all()
    assert metrics['candidate_states'] == 2


def test_mixed_training_cache_equivalence_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(7)
    cfg=mixed_config(); a=FollowNet(cfg); b=copy.deepcopy(a)
    averages=[copy.deepcopy(m).requires_grad_(False).eval() for m in (a,b)]
    optimizers=[torch.optim.AdamW(m.parameters(),lr=.001) for m in (a,b)]
    data=batch(cfg,4); data['offtrack'][1]=1; data['dense_mask'][2,5:]=0
    micro=[{k:v[i:i+2] for k,v in data.items()} for i in (0,2)]
    for step in (2000,2001):
        results=[]
        for i,m in enumerate((a,b)):
            torch.manual_seed(step)
            results.append(optimizer_update(m,averages[i],optimizers[i],micro,step,.001,
                           device='cpu',cache_training_encoding=bool(i),n_commit=2))
        assert results[0][0] == pytest.approx(results[1][0],rel=1e-6)
        for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q)
        assert a.flow.velocity.weight.grad.abs().sum() > 0
    sample=SampleConfig(crop=CropSpec(depth=20,width=12,behind=10),n_history=8)
    volume=FiberVolumeSpec('unused',ct_zarr='unused',inputs='ct+presence')
    path=tmp_path/'last.pt'
    save_checkpoint(path,b,averages[1],volume,sample,dict(step=2001,optimizer=optimizers[1].state_dict(),rng=training_rng_state()))
    loaded,*_=load_checkpoint(path,'cpu')
    loaded.requires_grad_(False)  # Match EMA inference kernel dispatch as well as its weights.
    assert loaded.cfg.scorer=='passage' and loaded.cfg.gaussian_candidates==4
    noise=initial_residuals(cfg,4,'cpu')
    with torch.no_grad():
        torch.testing.assert_close(run(loaded,data,initial_noise=noise)['candidate_logits'],
                                   run(averages[1],data,initial_noise=noise)['candidate_logits'],rtol=0,atol=0)
    ck=torch.load(path,weights_only=False)
    fresh=FollowNet(cfg); ema=copy.deepcopy(fresh); opt=torch.optim.AdamW(fresh.parameters(),lr=.001)
    rng=training_rng_state()
    resume_training(ck,fresh,ema,opt)
    expected=optimizer_update(fresh,ema,opt,micro,2002,.001,device='cpu',n_commit=2)[0]
    restore_training_rng(ck['rng'])
    actual=optimizer_update(b,averages[1],optimizers[1],micro,2002,.001,device='cpu',n_commit=2)[0]
    assert expected==actual
    restore_training_rng(rng)


def test_legacy_checkpoint_defaults_and_new_resume_guard(tmp_path):
    cfg=config(); model=FollowNet(cfg).eval(); data=batch(cfg)
    sample=SampleConfig(crop=CropSpec(depth=20,width=12,behind=10),n_history=8)
    path=tmp_path/'legacy.pt'
    save_checkpoint(path,model,model,FiberVolumeSpec('unused'),sample)
    ck=torch.load(path,weights_only=False)
    for key in ('scorer','gaussian_candidates','selection_horizon'): ck['model_cfg'].pop(key)
    torch.save(ck,path)
    loaded,*_=load_checkpoint(path,'cpu')
    torch.testing.assert_close(run(model,data)['confidence'],run(loaded,data)['confidence'],rtol=0,atol=0)
    assert resolve_scorer_options(resume=ck)==dict(scorer='legacy',gaussian_candidates=0)
    with pytest.raises(ValueError,match='resume'): resolve_scorer_options('passage',resume=ck)
    with pytest.raises(ValueError,match='resume'): resolve_scorer_options(gaussian_candidates=4,resume=ck)
    with pytest.raises(ValueError,match='Gaussian alternatives'): config(gaussian_candidates=4)
    with pytest.raises(ValueError,match='Gaussian alternatives'): mixed_config(sampler_mode='gaussian')


def test_mixed_sampling_streams_and_evaluation_repeats():
    cfg=mixed_config()
    make=lambda:trace_generator(2,[1,2,3],[0,0,1])
    together=trace_noise(cfg,[make(),make()],'cpu')
    assert together[:,0].eq(0).all()
    torch.testing.assert_close(together[:1],trace_noise(cfg,[make()],'cpu'),rtol=0,atol=0)
    assert not torch.equal(together[:1],trace_noise(cfg,[trace_generator(3,[1,2,3],[0,0,1])],'cpu'))
    path=Path(__file__).parents[1]/'scripts/evaluate_single_path.py'
    spec=importlib.util.spec_from_file_location('evaluate_mixed',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.evaluation_sampling_seeds('zero',gaussian_candidates=4)==[0,1,2]


@pytest.mark.parametrize('field,value',[('scorer','legacy'),('gaussian_candidates',0),('selection_horizon',4)])
def test_preflight_rejects_different_scorer_configuration(tmp_path,field,value):
    bench=dict(architecture=ARCHITECTURE,passed=True,microbatch=2,flow_draws=64,crop=[176,96,96],
               cache_training_encoding=True,compile_model=False,sampler_mode='zero',scorer='passage',
               gaussian_candidates=4,selection_horizon=8)
    bench[field]=value; path=tmp_path/'bench.json'; path.write_text(json.dumps(bench))
    with pytest.raises(ValueError,match='matching full-crop benchmark'):
        main(['--device','cpu','--fiber-zarrs','unused','--fibers','unused','--ct','unused',
              '--name','unused','--fixed-bank','unused','--manifest','unused','--benchmark',str(path),
              '--sampler-mode','zero','--scorer','passage','--gaussian-candidates','4'])


@pytest.mark.parametrize('joined',[False,True])
def test_launcher_forwards_scorer_settings_to_preflight_without_starting_training(tmp_path,joined):
    capture=tmp_path/'arguments.json'; wrapper=tmp_path/'python'
    wrapper.write_text(f'#!{sys.executable}\nimport json, subprocess, sys\n'
        f'if sys.argv[1]=="-c": sys.exit(subprocess.call([{sys.executable!r}, *sys.argv[1:]]))\n'
        f'open({str(capture)!r},"w").write(json.dumps(sys.argv[1:]))\nsys.exit(42)\n')
    wrapper.chmod(0o755)
    script=Path(__file__).parents[1]/'scripts/launch_single_path.sh'
    options={'--scorer':'passage','--gaussian-candidates':'4','--sampler-mode':'zero','--n-commit':'4'}
    args=[f'{key}={value}' for key,value in options.items()] if joined else [v for pair in options.items() for v in pair]
    result=subprocess.run(['bash',str(script),'test_scorer_launcher',*args],env=dict(os.environ,PYTHON=str(wrapper)),capture_output=True,text=True)
    assert result.returncode==42, result.stderr
    forwarded=json.loads(capture.read_text())
    assert forwarded[0].endswith('benchmark_single_path.py')
    for key,value in options.items(): assert forwarded[forwarded.index(key)+1]==value


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_compiled_mixed_training_and_inference():
    torch.manual_seed(5)
    model=FollowNet(mixed_config()).cuda(); ema=copy.deepcopy(model)
    data=batch(model.cfg); gpu={k:v.cuda() for k,v in data.items()}
    noise=initial_residuals(model.cfg,2,'cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        eager=model.generate_training_curve(gpu['x'],gpu['hist'],gpu['hmask'],initial_noise=noise)
    compile_training_model(model)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        compiled=model.generate_training_curve(gpu['x'],gpu['hist'],gpu['hmask'],initial_noise=noise)
    torch.testing.assert_close(eager,compiled,atol=.05,rtol=.05)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    for step in (2000,2001):
        loss,metrics,_=optimizer_update(model,ema,optimizer,[data],step,.001,device='cuda',n_commit=2)
        assert torch.isfinite(torch.tensor(loss)) and 'candidate_rescues' in metrics
