"""Shared evidence, causal recurrence, full-crop I/O and checkpoint contracts."""
import copy
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from test_identity import config, batch
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectFollower, ARCHITECTURE
from vesuvius.neural_tracing.fiber_follow.regression.memory_data import memory_images, memory_allowed
from vesuvius.neural_tracing.fiber_follow.regression.supervision import identity_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch, compile_training_model
from vesuvius.neural_tracing.fiber_follow.regression.model import stratified_history
from vesuvius.neural_tracing.fiber_follow.shared.data import ZBand


def scene(b=1, steps=2, grad=2):
    cfg = config(memory_slots=3,memory_steps=steps,memory_grad_steps=grad,spatial_recent=1,spatial_archive=2,spatial_retrieve=1)
    data = batch(cfg,b)
    x = data['x']
    t = steps+1
    x.update(history_crops=torch.rand(b,steps,*x['fine'].shape[1:]),seed_crop=torch.rand_like(x['fine']),
        memory_mask=torch.ones(b,t,dtype=torch.bool),memory_positions=torch.zeros(b,t,3),
        memory_frames=torch.eye(3).expand(b,t,-1,-1).clone(),memory_seed_valid=torch.ones(b,dtype=torch.bool),
        memory_seed_position=torch.zeros(b,3),memory_seed_frame=torch.eye(3).expand(b,-1,-1).clone())
    x['memory_positions'][...,2] = torch.arange(t)
    return cfg,data


def state(out,model):
    return {k:out['memory_'+k] for k in model.initial_memory(len(out['points']),'cpu')}


def test_one_encoder_and_shared_identity_reaches_geometry_confidence_and_seed():
    torch.manual_seed(4)
    cfg,data = scene()
    model = DirectFollower(cfg)
    assert model.architecture == ARCHITECTURE
    data['x']['seed_crop'].requires_grad_()
    data['x']['history_crops'].requires_grad_()
    for key in ('points','confidence_logits','memory_probe'):
        model.zero_grad(set_to_none=True)
        data['x']['seed_crop'].grad = data['x']['history_crops'].grad = None
        out = model(data['x'],data['hist'],data['hmask'])
        out[key].square().mean().backward()
        for p in (model.encoder.stem[0].weight,model.embedding.weight,model.write_gate.weight,
                  model.decoder.layers[0].multihead_attn.in_proj_weight):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        assert data['x']['seed_crop'].grad.abs().sum() > 0
        assert data['x']['history_crops'].grad.abs().sum() > 0


def test_streaming_matches_unroll_and_seed_is_immutable():
    torch.manual_seed(2)
    cfg,data = scene(b=2)
    model = DirectFollower(cfg).eval()
    x = data['x']
    with torch.no_grad():
        whole = model(x,data['hist'],data['hmask'])
        carry = None
        probes = []
        for j in range(cfg.memory_steps+1):
            step = dict(x)
            step['fine'] = x['history_crops'][:,j] if j < cfg.memory_steps else x['fine']
            step['history_crops'] = x['history_crops'][:,:0]
            for k in ('memory_mask','memory_positions','memory_frames'):
                step[k] = x[k][:,j:j+1]
            if j:
                step['seed_crop'] = torch.full_like(x['seed_crop'],torch.nan)
            out = model(step,data['hist'],data['hmask'],memory=carry)
            carry = state(out,model)
            probes.append(out['memory_probe'])
        for k,v in carry.items():
            torch.testing.assert_close(v,whole['memory_'+k],rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(torch.cat(probes,1),whole['memory_probe'],rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(out['points'],whole['points'],rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(out['confidence'],whole['confidence'],rtol=2e-5,atol=2e-6)
        single = model({k:v[:1] for k,v in x.items()},data['hist'][:1],data['hmask'][:1])
        torch.testing.assert_close(single['points'],whole['points'][:1],rtol=2e-5,atol=2e-6)


def test_candidates_read_memory_independently_without_writing_it_or_moving_coordinates():
    cfg,data = scene()
    model = DirectFollower(cfg)
    curves = torch.zeros(1,2,cfg.n_future,3)
    curves[...,2] = model.planes
    curves[:,1,:,0] = 2
    curves.requires_grad_()
    out = model(data['x'],data['hist'],data['hmask'],candidates=curves)
    out['candidate_confidence_logits'].sum().backward()
    assert curves.grad is None
    assert model.embedding.weight.grad.abs().sum() > 0
    with torch.no_grad():
        reverse = model(data['x'],data['hist'],data['hmask'],candidates=curves.flip(1))
        torch.testing.assert_close(reverse['candidate_confidence_logits'],out['candidate_confidence_logits'].flip(1))
        for k,v in state(out,model).items():
            torch.testing.assert_close(v,reverse['memory_'+k])


def test_remote_seed_receives_infonce_when_visible_history_is_absent():
    cfg,data = scene()
    model = DirectFollower(cfg)
    data['x']['seed_mask'].zero_()  # stored original seed is outside current crop
    data['hmask'].zero_()
    data['x']['seed_crop'].requires_grad_()
    out = model(data['x'],data['hist'],data['hmask'],queries=data['identity_points'])
    assert out['reference_mask'][0,-1]
    identity_terms(out,data,cfg)['identity_per_state'].sum().backward()
    assert data['x']['seed_crop'].grad.abs().sum() > 0


def test_burn_in_masks_padding_and_detaches_old_encodings():
    cfg,data = scene(steps=3,grad=1)
    model = DirectFollower(cfg)
    data['x']['history_crops'].requires_grad_()
    out = model(data['x'],data['hist'],data['hmask'])
    out['points'].square().sum().backward()
    grad = data['x']['history_crops'].grad
    assert grad[:,:2].eq(0).all() and grad[:,2].abs().sum() > 0
    padded = copy.deepcopy(data)
    padded['x']['memory_mask'][:,:2] = False
    padded['x']['history_crops'] = padded['x']['history_crops'].detach()
    padded['x']['history_crops'][:,:2] = torch.nan
    result = model(padded['x'],padded['hist'],padded['hmask'])
    assert torch.isfinite(result['points']).all()


def test_full_crop_sampling_reuses_current_crop_and_checks_holdout():
    cfg,_ = scene()
    item = dict(pos=np.array([0.,0.,0.]),frame=np.eye(3),hist_local=np.zeros((cfg.n_history,3)),
                hmask=np.zeros(cfg.n_history),seed_pos=np.array([0.,0.,-20.]),
                seed_tangent=np.array([0.,0.,1.]),seed_valid=True)
    reads = []
    def crop(items,vol,spec,pool):
        assert spec == cfg.fine
        reads.extend(items)
        return torch.zeros(len(items),2,spec.depth,spec.width,spec.width)
    x = memory_images([item],None,cfg,crop)
    assert len(reads) == 1  # seed only; current crop already exists
    assert x['history_crops'].shape[1] == cfg.memory_steps
    assert not memory_allowed(item,cfg,ZBand(-18,-17))
    item['memory_warm'] = True
    reads.clear()
    x = memory_images([item],None,cfg,crop)
    assert not reads and x['history_crops'].shape[1] == 0
    moved = move_batch(x,'cpu')
    assert moved['history_crops'] is x['history_crops']




def test_missing_references_and_partially_padded_nan_observations_are_safe():
    cfg,data = scene(b=2)
    model = DirectFollower(cfg)
    x = data['x']
    x['memory_seed_valid'].zero_()
    x['seed_crop'].fill_(torch.nan)
    x['seed'].fill_(torch.nan)
    x['seed_mask'].zero_()
    data['hist'].fill_(torch.nan)
    data['hmask'].zero_()
    x['memory_mask'][0,0] = False
    x['history_crops'][0,0] = torch.nan
    x['memory_positions'][0,0] = torch.nan
    x['memory_frames'][0,0] = torch.nan
    out = model(x,data['hist'],data['hmask'])
    assert torch.isfinite(out['points']).all() and torch.isfinite(out['confidence']).all()
    (out['points'].square().mean()+out['confidence_logits'].square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_whole_observation_checkpoint_matches_nested_checkpoint_values_and_gradients():
    from torch.utils.checkpoint import checkpoint
    cfg,data = scene(steps=1)
    cfg = replace(cfg,activation_checkpointing=True)
    model = DirectFollower(cfg)
    image = data['x']['fine'].requires_grad_()
    def nested(x):
        dense,_,tokens = model.encode(x)
        return model.observation_features(dense,tokens)
    before = checkpoint(nested,image,use_reentrant=False)
    old_grads = torch.autograd.grad(before.square().sum(),(image,model.encoder.stem[0].weight,model.encoder.blocks[0].axes[0].qkv.weight))
    after = model.encode_observation(image,'cpu',True)
    new_grads = torch.autograd.grad(after.square().sum(),(image,model.encoder.stem[0].weight,model.encoder.blocks[0].axes[0].qkv.weight))
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    for old,new in zip(old_grads,new_grads):
        torch.testing.assert_close(old,new,rtol=0,atol=0)


def test_scheduling_only_labeled_probes_preserves_task_outputs_and_gradients():
    cfg,data = scene()
    model = DirectFollower(cfg)
    out = model(data['x'],data['hist'],data['hmask'])
    loss = out['points'].square().sum()+out['memory_probe'][:,-1].square().sum()
    before = torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True)
    mask = torch.zeros_like(data['x']['memory_mask'])
    mask[:,-1] = True
    scheduled = model(data['x'],data['hist'],data['hmask'],probe_mask=mask)
    loss = scheduled['points'].square().sum()+scheduled['memory_probe'][:,-1].square().sum()
    after = torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True)
    for k in ('points','confidence_logits'):
        torch.testing.assert_close(out[k],scheduled[k],rtol=0,atol=0)
    torch.testing.assert_close(out['memory_probe'][:,-1],scheduled['memory_probe'][:,-1],rtol=0,atol=0)
    for a,b in zip(before,after):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)


def test_compiled_tensor_modules_preserve_checkpoint_names_and_training_gradients():
    cfg,data = scene(steps=5,grad=5)
    data['x']['history_crops'].requires_grad_()
    model = DirectFollower(cfg)
    keys = set(model.state_dict())
    compiled = compile_training_model(model,backend='eager')
    assert compiled is model and set(compiled.state_dict()) == keys
    out = compiled(data['x'],data['hist'],data['hmask'])
    out['points'].square().sum().backward()
    assert model.encoder.stem[0].weight.grad.abs().sum() > 0
    assert model.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0
    assert (data['x']['history_crops'].grad.flatten(2).abs().sum(-1) > 0).sum() == 4




@pytest.mark.parametrize('mask_kind',['empty','one_curve','mixed'])
def test_candidate_scheduling_preserves_losses_gradients_and_metrics(mask_kind):
    from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
    from vesuvius.neural_tracing.fiber_follow.regression.diagnostics import candidate_decisions
    torch.manual_seed(22)
    cfg,data = scene(b=2,steps=1)
    model = DirectFollower(cfg)
    curves = torch.randn(2,2,cfg.n_future,3)
    curves[...,2] = model.planes
    mask = torch.zeros(2,2,cfg.n_future,dtype=torch.bool)
    if mask_kind != 'empty':
        mask[0,1] = True
    if mask_kind == 'mixed':
        mask[1,0,-1] = True  # retain labels outside the geometry commit window too
    data.update(candidate_points=curves,candidate_mask=mask,candidate_labels=torch.ones_like(mask),
                decision_kind=torch.ones(2),decision_tail=torch.ones(2))
    probes = torch.zeros_like(data['x']['memory_mask'])
    calls = []
    hook = model.decoder.register_forward_hook(lambda *args: calls.append(1))
    results,grads,losses,metrics,counts = [],[],[],[],[]
    for scheduled in (False,True):
        calls.clear()
        kwargs = {'candidate_mask':mask} if scheduled else {}
        out = model(data['x'],data['hist'],data['hmask'],candidates=curves,probe_mask=probes,**kwargs)
        terms = loss_terms(out,data,cfg,1.5,n_commit=2)
        loss = sum(terms[k].sum() for k in ('geometry_per_state','confidence_per_state','candidate_per_state'))
        grads.append(torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True))
        results.append(out)
        losses.append(loss.detach())
        metrics.append(candidate_decisions(out,data,cfg,n_commit=2))
        counts.append(len(calls))
    hook.remove()
    assert counts[0]-counts[1] == int((~mask.any(dim=(0,2))).sum())
    assert metrics[0] == metrics[1]
    torch.testing.assert_close(losses[0],losses[1],rtol=0,atol=0)
    for key in ('points','confidence_logits','memory_slots'):
        torch.testing.assert_close(results[0][key],results[1][key],rtol=0,atol=0)
    torch.testing.assert_close(results[0]['candidate_confidence_logits'][mask],
                               results[1]['candidate_confidence_logits'][mask],rtol=0,atol=0)
    for a,b in zip(*grads):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)


def test_optimizer_schedules_candidates_from_cpu_masks(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update
    cfg,data = scene(b=2,steps=1)
    data.update(candidate_points=torch.zeros(2,2,cfg.n_future,3),
                candidate_mask=torch.zeros(2,2,cfg.n_future,dtype=torch.bool),
                candidate_labels=torch.zeros(2,2,cfg.n_future))
    model = DirectFollower(cfg)
    ema = copy.deepcopy(model).requires_grad_(False)
    original = model.forward
    received = []
    def forward(*args,**kwargs):
        received.append(kwargs['candidate_mask'])
        return original(*args,**kwargs)
    monkeypatch.setattr(model,'forward',forward)
    opt = torch.optim.AdamW(model.parameters(),lr=1e-3)
    metrics = optimizer_update(model,ema,opt,[data],1,1e-3,compute_metrics=False)
    assert len(received) == 1 and received[0] is data['candidate_mask']
    assert metrics['identity']['candidate_loss'] == 0


def test_sampled_encoder_gradients_keep_outputs_and_write_gradients():
    torch.manual_seed(5)
    cfg,data = scene(steps=4,grad=4)
    full = DirectFollower(replace(cfg,memory_encoder_grad_steps=0))
    sampled = DirectFollower(replace(cfg,memory_encoder_grad_steps=1))
    sampled.load_state_dict(full.state_dict())
    outputs, grads = [], []
    for model in (full,sampled):
        data['x']['history_crops'].requires_grad_()
        data['x']['history_crops'].grad = None
        out = model(data['x'],data['hist'],data['hmask'])
        out['points'].square().sum().backward()
        outputs.append(out['points'].detach())
        grads.append((data['x']['history_crops'].grad.clone(),model.write_gate.weight.grad.clone()))
    # Values differ only by encode-batch rounding.
    torch.testing.assert_close(outputs[0],outputs[1])
    torch.testing.assert_close(grads[0][1],grads[1][1],rtol=1e-4,atol=1e-7)  # writes keep full gradients
    # Exactly one of four history crops receives encoder gradient, scaled by 4.
    reached = grads[1][0].flatten(2).abs().sum(-1) > 0
    assert reached.sum().item() == 1
    j = reached[0].nonzero()[0,0]
    torch.testing.assert_close(grads[1][0][:,j],4*grads[0][0][:,j],rtol=1e-4,atol=1e-7)


def test_stratified_sampling_covers_time_and_weights_uneven_groups(monkeypatch):
    own = [(0,j) for j in (2,4,5,7,9)]
    counts = dict.fromkeys(own,0)
    # Enumerate every draw from strata of sizes two and three. Each observation's
    # mean weighted contribution must equal its full-gradient contribution.
    for offsets in product(range(2),range(3)):
        draws = iter(offsets)
        monkeypatch.setattr(torch,'randint',lambda *args,**kwargs: torch.tensor(next(draws)))
        selected = stratified_history(own,2)
        assert len(set(selected) & set(own[:2])) == 1
        assert len(set(selected) & set(own[2:])) == 1
        for pair,weight in selected.items():
            assert weight == (2 if pair in own[:2] else 3)
            counts[pair] += weight
    assert list(counts.values()) == [6]*5
    assert stratified_history([],4) == {}
    assert stratified_history(own,8) == dict.fromkeys(own,1)


def test_stratified_gradients_respect_burn_padding_and_seed_head():
    torch.manual_seed(9)
    cfg,data = scene(b=3,steps=7,grad=5)
    x = data['x']
    x['memory_mask'][1,3:5] = False  # three valid history entries in the gradient window
    x['memory_mask'][2,:-1] = False  # no valid history
    for key in ('history_crops','seed_crop','fine'):
        x[key].requires_grad_()
    full = DirectFollower(replace(cfg,memory_encoder_grad_steps=0))
    sampled = DirectFollower(replace(cfg,memory_encoder_grad_steps=2))
    sampled.load_state_dict(full.state_dict())
    outputs,grads = [],[]
    for model in (full,sampled):
        # Isolate gradient routing from different encode-batch kernel rounding.
        model.observation_batch = (1,1)
        for key in ('history_crops','seed_crop','fine'):
            x[key].grad = None
        out = model(x,data['hist'],data['hmask'])
        (out['points'].square().sum()+out['memory_probe'].square().sum()).backward()
        outputs.append(out['points'].detach())
        grads.append({key:x[key].grad.clone() for key in ('history_crops','seed_crop','fine')})
    torch.testing.assert_close(outputs[0],outputs[1])
    torch.testing.assert_close(full.write_gate.weight.grad,sampled.write_gate.weight.grad,rtol=1e-4,atol=1e-7)
    for key in ('seed_crop','fine'):
        assert grads[1][key].abs().sum() > 0
        torch.testing.assert_close(grads[0][key],grads[1][key],rtol=1e-4,atol=1e-7)
    reached = grads[1]['history_crops'].flatten(2).abs().sum(-1) > 0
    assert not reached[:,:2].any()  # burn-in
    assert not reached[2].any()
    for i,strata in enumerate(((range(2,4),range(4,7)),((2,),(5,6)))):
        for stratum in strata:
            selected = [j for j in stratum if reached[i,j]]
            assert len(selected) == 1
            j = selected[0]
            torch.testing.assert_close(grads[1]['history_crops'][i,j],
                                       len(stratum)*grads[0]['history_crops'][i,j],rtol=1e-4,atol=1e-7)
    assert not reached[1,3:5].any()


def test_stratified_sampler_replays_from_torch_rng_state():
    own = [(0,j) for j in range(32)]
    rng = torch.get_rng_state()
    selected = stratified_history(own,4)
    torch.set_rng_state(rng)
    assert stratified_history(own,4) == selected
    assert list(selected.values()) == [8]*4
    assert [j//8 for _,j in selected] == list(range(4))


@pytest.mark.parametrize('budget',[None,0,2])
def test_preflight_and_training_wire_encoder_gradient_budget(monkeypatch,tmp_path,budget):
    from vesuvius.neural_tracing.fiber_follow.regression import identity_preflight,train
    from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig
    class Configured(Exception):
        pass
    def capture(**kwargs):
        raise Configured(DirectConfig(**kwargs))
    capture.memory_encoder_grad_steps = DirectConfig.memory_encoder_grad_steps
    for key in ('memory_slots','spatial_recent','spatial_archive','spatial_retrieve','trajectory_window'):
        setattr(capture,key,getattr(DirectConfig,key))
    monkeypatch.setattr(identity_preflight,'DirectConfig',capture)
    monkeypatch.setattr(torch,'set_num_threads',lambda _: None)
    option = [] if budget is None else ['--memory-encoder-grad-steps',str(budget)]
    with pytest.raises(Configured) as result:
        identity_preflight.main(['--out',str(tmp_path),'--memory-slots','3',*option])
    expected = 4 if budget is None else budget
    assert result.value.args[0].memory_encoder_grad_steps == expected
    assert train.build_parser().get_default('memory_encoder_grad_steps') == 4
