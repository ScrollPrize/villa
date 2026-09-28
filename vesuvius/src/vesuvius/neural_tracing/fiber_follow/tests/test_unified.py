"""Shared evidence, causal recurrence, full-crop I/O and checkpoint contracts."""
import copy
from dataclasses import replace

import numpy as np
import torch

from test_identity import config, batch
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectFollower, UNIFIED_ARCHITECTURE
from vesuvius.neural_tracing.fiber_follow.regression.memory_data import memory_images, memory_allowed
from vesuvius.neural_tracing.fiber_follow.regression.supervision import identity_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch, checkpoint_config
from vesuvius.neural_tracing.fiber_follow.shared.data import ZBand


def scene(b=1, steps=2, grad=2):
    cfg = config(memory_slots=3,memory_steps=steps,memory_grad_steps=grad,memory_version=3)
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
    assert model.architecture == UNIFIED_ARCHITECTURE
    assert model.encoder.condition is None
    assert not hasattr(model,'recurrent_memory') and not hasattr(model,'path_evidence')
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


def test_remote_seed_receives_infonce_even_with_visible_history():
    cfg,data = scene()
    model = DirectFollower(cfg)
    data['x']['seed_mask'].zero_()  # stored original seed is outside current crop
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


def test_checkpoint_configuration_identifies_new_architecture():
    cfg,_ = scene()
    assert checkpoint_config(dict(architecture=UNIFIED_ARCHITECTURE,model_cfg=cfg.to_dict())) == cfg


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
    from vesuvius.neural_tracing.fiber_follow.regression.model import sample_features
    cfg,data = scene(steps=1)
    cfg = replace(cfg,activation_checkpointing=True)
    model = DirectFollower(cfg)
    image = data['x']['fine'].requires_grad_()
    def nested(x):
        dense,_,_ = model.encode(x)
        return sample_features(dense,model.stencil[None].expand(len(x),-1,-1),cfg.fine)[0]
    before = checkpoint(nested,image,use_reentrant=False)
    old_grads = torch.autograd.grad(before.square().sum(),(image,model.encoder.stem[0].weight,model.encoder.blocks[0].axes[0].qkv.weight))
    after = model.encode_observation(image,'cpu',True)
    new_grads = torch.autograd.grad(after.square().sum(),(image,model.encoder.stem[0].weight,model.encoder.blocks[0].axes[0].qkv.weight))
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    for old,new in zip(old_grads,new_grads):
        torch.testing.assert_close(old,new,rtol=0,atol=0)
