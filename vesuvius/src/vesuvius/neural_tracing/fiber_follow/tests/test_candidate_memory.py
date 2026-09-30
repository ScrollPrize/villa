"""Continuous proposals, exact small route search, supervision, and v4 compatibility."""
import copy
from dataclasses import replace
import itertools

import pytest
import torch
import torch.nn.functional as F

from test_detailed_memory import config as v4_config
from test_trajectory_memory import memory_batch, state_from, training_chunk
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model, CANDIDATE_MEMORY_ARCHITECTURE
from vesuvius.neural_tracing.fiber_follow.regression.candidate_model import (
    continuous_route, direction_alignment, shortlist, initialize_candidate_model,
)
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update, checkpoint_config
from vesuvius.neural_tracing.fiber_follow.regression.feature_sequences import FeatureStreamStates


def config(**kwargs):
    return replace(v4_config(), memory_version=5, recurrent_refinement_steps=2, proposal_candidates=4, **kwargs)


def test_route_matches_brute_force_with_curved_continuous_candidates():
    torch.manual_seed(410)
    positions = torch.randn(1, 4, 3, 3)
    positions[..., 2] = torch.arange(1, 5)[None, :, None]
    scores = torch.randn(1, 4, 3)
    directions = torch.zeros(1, 4, 3, 6)
    valid = torch.ones_like(scores, dtype=torch.bool)
    route, connected = continuous_route(scores, positions, valid, directions, .1, .3, 6.)
    choices = []
    for indices in itertools.product(range(3), repeat=4):
        p = positions[0, torch.arange(4), torch.tensor(indices)]
        tangent = F.normalize(torch.cat((p[:1], p[1:]-p[:-1])), dim=-1)
        cost = F.smooth_l1_loss(tangent[1:], tangent[:-1], beta=.25, reduction='sum')*.3
        choices.append((float(scores[0, torch.arange(4), torch.tensor(indices)].sum()-cost), indices))
    assert connected.all()
    assert tuple(route[0].tolist()) == max(choices)[1]
    # A curved route can move more than one old lattice cell between planes.
    p = torch.tensor([[[[0., 0., 1.]], [[3., 0., 2.]], [[7., 1., 3.]]]])
    _, connected = continuous_route(torch.zeros(1, 3, 1), p, torch.ones(1, 3, 1, dtype=torch.bool),
                                    torch.zeros(1, 3, 1, 6), .1, .05, 6.)
    assert connected.all()


def test_direction_mixtures_and_disconnected_routes():
    t = F.normalize(torch.tensor([[1., 2., 3.]]), dim=-1)
    n = F.normalize(torch.tensor([[2., 1., 1.]]), dim=-1)
    matrix = n[..., :, None]*n[..., None, :]
    components = matrix[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    torch.testing.assert_close(direction_alignment(t, components), direction_alignment(-t, components))
    assert direction_alignment(t, torch.zeros_like(components)).eq(0).all()
    isotropic = torch.tensor([[1/3, 1/3, 1/3, 0., 0., 0.]])
    assert direction_alignment(t, isotropic).max() < 1e-6
    p = torch.tensor([[[[9., 0., 1.]], [[9., 0., 2.]]]])
    _, ok = continuous_route(torch.zeros(1, 2, 1), p, torch.ones(1, 2, 1, dtype=torch.bool),
                            torch.zeros(1, 2, 1, 6), .1, .05, 6.)
    assert not ok.any()


def test_shortlist_reachability_suppression_and_padding():
    p = torch.tensor([[[[9., 0., 1.], [.1, 0., 1.], [.2, 0., 1.], [1., 0., 1.]]]])
    indices, valid = shortlist(torch.tensor([[[10., 3., 2., 1.]]]), p, 4, .5, 5.)
    assert indices[0, 0, :2].tolist() == [1, 3]
    assert valid[0, 0].tolist() == [True, True, False, False]


def test_initial_coordinates_are_continuous_and_geometry_trains_offsets():
    torch.manual_seed(411)
    model = build_model(config())
    b = memory_batch(model.cfg, 1)
    out = model(b['x'], b['hist'], b['hmask'])
    assert out['refinement_points'].shape[1] == 3
    fraction = out['initial_points'][..., :2]/model.proposal_spacing
    assert (fraction-fraction.round()).abs().max() > .001
    assert (out['points'][:, 0].norm(dim=-1) <= model.cfg.max_recovery_distance+1e-5).all()
    terms = loss_terms(out, b, model.cfg)
    terms['geometry_per_state'].sum().backward()
    assert model.proposal_head[-1].weight.grad[1:].abs().sum() > 0
    assert model.refinement_delta.weight.grad.abs().sum() > 0


def test_spatial_supervision_trains_earlier_features_even_when_departed():
    torch.manual_seed(412)
    model = build_model(config())
    old, b = memory_batch(model.cfg, 1), memory_batch(model.cfg, 1, step=1)
    old['x']['fine'].requires_grad_()
    first = model(old['x'], old['hist'], old['hmask'])
    b.update(offtrack=torch.ones(1), route_ab=torch.full((1, model.cfg.n_future, 2), .13),
             route_mask=torch.ones(1, model.cfg.n_future, dtype=torch.bool))
    out = model(b['x'], b['hist'], b['hmask'], memory=state_from(model, first))
    terms = loss_terms(out, b, model.cfg)
    assert terms['geometry_per_state'].eq(0).all()
    terms['proposal_per_state'].sum().backward()
    assert old['x']['fine'].grad.abs().sum() > 0
    assert model.proposal_head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    b['identity_observable'] = torch.zeros(1, dtype=torch.bool)
    terms = loss_terms(out, b, model.cfg)
    assert terms['proposal_per_state'].eq(0).all()


def test_migration_shared_parameters_and_checkpoint_roundtrip(tmp_path):
    old = build_model(v4_config())
    cfg = replace(old.cfg, memory_version=5, recurrent_refinement_steps=2, proposal_candidates=4)
    model = initialize_candidate_model(old, cfg)
    for name, value in old.state_dict().items():
        if name.startswith('coordinates.'):
            continue
        actual = model.state_dict()[name]
        if name == 'refinement_stage.weight':
            actual = actual[:len(value)]
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    ck = dict(model_cfg=cfg.to_dict(), architecture=model.architecture, model=model.state_dict())
    torch.save(ck, tmp_path/'v5.pt')
    ck = torch.load(tmp_path/'v5.pt', weights_only=True)
    restored = build_model(checkpoint_config(ck))
    restored.load_state_dict(ck['model'], strict=True)
    assert restored.architecture == CANDIDATE_MEMORY_ARCHITECTURE


def test_streamed_optimizer_and_endpoint_replay_include_new_losses():
    model = build_model(config())
    ema, opt, states = copy.deepcopy(model), torch.optim.AdamW(model.parameters()), FeatureStreamStates()
    chunk = training_chunk(model.cfg, end=True)
    for row in chunk['feature_sequence']:
        row['replay_select'] = torch.ones(2, dtype=torch.bool)
    metrics = optimizer_update(model, ema, opt, [chunk], 1, .0001, compute_metrics=False, stream_states=states)
    assert metrics['replay_endpoints'] == 2
    assert metrics['memory']['proposal_loss'] > 0
    assert not states.states and not states.replay.streams
    assert torch.isfinite(torch.tensor(metrics['loss']))


def test_fullgraph_forward_and_backward():
    model = build_model(config())
    b = memory_batch(model.cfg, 1)
    out = torch.compile(model, backend='eager', fullgraph=True)(b['x'], b['hist'], b['hmask'])
    terms = loss_terms(out, b, model.cfg)
    (terms['proposal_per_state']+terms['geometry_per_state']).sum().backward()
    assert model.proposal_head[-1].weight.grad.abs().sum() > 0


def test_v5_causal_builder_and_matched_endpoints(monkeypatch, tmp_path):
    import test_trajectory_memory as existing
    monkeypatch.setattr(existing, 'cfg', config)
    existing.test_builder_streams_causal_main_crops_and_keeps_paired_endpoints_identical(tmp_path, monkeypatch)


def test_v5_cold_seed_and_streaming_tracer(monkeypatch):
    import test_trajectory_memory as existing
    monkeypatch.setattr(existing, 'cfg', config)
    existing.test_cold_remote_seed_is_encoded_once_and_warm_trace_reads_only_current_crop(monkeypatch)


def test_transfer_warmup_freezes_then_unfreezes_inherited_weights():
    model = build_model(config())
    new, inherited = [], []
    for name, param in model.named_parameters():
        (new if name.startswith('proposal_') else inherited).append(param)
    opt = torch.optim.AdamW([dict(params=new), dict(params=inherited, lr_scale=.1, freeze_until=1)])
    ema = copy.deepcopy(model)
    before = model.encoder.stem[0].weight.detach().clone()
    chunk = training_chunk(model.cfg, end=True)
    optimizer_update(model, ema, opt, [chunk], 1, .001, compute_metrics=False)
    torch.testing.assert_close(model.encoder.stem[0].weight, before, rtol=0, atol=0)
    optimizer_update(model, ema, opt, [chunk], 2, .001, compute_metrics=False)
    assert not torch.equal(model.encoder.stem[0].weight, before)
    assert opt.param_groups[1]['lr'] == .0001


def test_fresh_optimizer_resume_preserves_weights_and_future_resume_schedule():
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_follow.regression.train import initialize_training_optimizer
    from vesuvius.neural_tracing.fiber_follow.shared.runloop import training_rng_state, lr_at
    args = SimpleNamespace(reset_optimizer=False, init_tracer='v4.pt', lr=.0003,
                           proposal_warmup_steps=500, proposal_inherited_lr_scale=.1)
    model = build_model(config())
    ema = copy.deepcopy(model)
    opt, _, _ = initialize_training_optimizer(model, ema, args)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    ck = dict(model=copy.deepcopy(model.state_dict()), ema=copy.deepcopy(ema.state_dict()),
              optimizer=opt.state_dict(), rng=training_rng_state(), step=3000)
    assert len(ck['optimizer']['param_groups']) == 2 and ck['optimizer']['state']
    expected_rng = torch.rand(3)
    reset_model = build_model(config()).requires_grad_(False)
    reset_ema = copy.deepcopy(reset_model)
    args.init_tracer, args.reset_optimizer = None, True
    fresh, done, origin = initialize_training_optimizer(reset_model, reset_ema, args, ck)
    assert done == origin == 3000
    assert len(fresh.param_groups) == 1 and not fresh.state
    assert all(p.requires_grad for p in reset_model.parameters())
    assert 'lr_scale' not in fresh.param_groups[0] and 'freeze_until' not in fresh.param_groups[0]
    torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
    for actual, saved in ((reset_model, ck['model']), (reset_ema, ck['ema'])):
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    fresh.param_groups[0]['lr'] = lr_at(done+1-origin, args.lr, 500, 100000-origin)
    assert fresh.param_groups[0]['lr'] == pytest.approx(.0003/500)
    for p in reset_model.parameters():
        p.grad = torch.ones_like(p)
    fresh.step()
    after = dict(model=reset_model.state_dict(), ema=reset_ema.state_dict(),
                 optimizer=fresh.state_dict(), rng=training_rng_state(), step=3001,
                 lr_restart_step=origin)
    args.reset_optimizer = False
    restored = build_model(config())
    restored_opt, done, origin = initialize_training_optimizer(restored, copy.deepcopy(restored), args, after)
    assert done == 3001 and origin == 3000 and len(restored_opt.param_groups) == 1
    assert len(restored_opt.state) == len(fresh.state)
    assert lr_at(done+1-origin, args.lr, 500, 100000-origin) == pytest.approx(.0003*2/500)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compiled_bf16_proposal_and_memory_gradients():
    from vesuvius.neural_tracing.fiber_follow.regression.train import compile_training_model, move_batch
    torch.manual_seed(414)
    eager = build_model(config()).cuda()
    compiled = copy.deepcopy(eager)
    b = move_batch(memory_batch(eager.cfg, 2), 'cuda')
    b.update(route_ab=torch.full((2, eager.cfg.n_future, 2), .13, device='cuda'),
             route_mask=torch.ones(2, eager.cfg.n_future, device='cuda', dtype=torch.bool))
    outputs = []
    for model in (eager, compile_training_model(compiled)):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            first = model(b['x'], b['hist'], b['hmask'])
            out = model(b['x'], b['hist'], b['hmask'], memory=state_from(eager, first))
            terms = loss_terms(out, b, eager.cfg)
            loss = terms['proposal_per_state'].mean()+terms['geometry_per_state'].mean()
        outputs.append((loss.detach(), out['proposal_route'].detach()))
        loss.backward()
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=.03, rtol=.02)
    torch.testing.assert_close(outputs[0][1], outputs[1][1])
    for prefix in ('encoder.', 'recurrent_memory.', 'proposal_head.'):
        grads = [torch.cat([p.grad.flatten() for name, p in model.named_parameters()
                           if name.startswith(prefix) and p.grad is not None]) for model in (eager, compiled)]
        assert all(torch.isfinite(g).all() for g in grads)
        assert F.cosine_similarity(*grads, dim=0) > .99
        assert (grads[1].norm()/grads[0].norm()).item() == pytest.approx(1., rel=.05)



@pytest.mark.skipif(not torch.cuda.is_available(), reason='Inductor replay transition needs CUDA')
def test_compiled_replay_transition_matches_eager_gradients():
    """A chain of FP32 writer transitions, as endpoint replay runs it: stored
    detached features plus one differentiable re-encoded crop."""
    from vesuvius.neural_tracing.fiber_follow.regression.stratified_replay import StratifiedReplay
    torch.manual_seed(412)
    model = build_model(config()).cuda()
    b = memory_batch(model.cfg, 1)
    b = {k: ({n: t.cuda() for n, t in v.items()} if isinstance(v, dict) else v.cuda()) for k, v in b.items()}
    live = model.observation_features(b['x'], b['hist'], b['hmask'])
    stored = [tuple(t.detach()+.1*torch.randn_like(t) if t.is_floating_point() else t for t in live) for _ in range(5)]
    poses = [dict(query_position=torch.tensor([[0., 0., 8.*t]], device='cuda'),
                  query_frame=torch.eye(3, device='cuda')[None], feature_seed_here=torch.tensor([t == 0], device='cuda'))
             for t in range(6)]
    grads = []
    for compiled in (False, True):
        replay = StratifiedReplay()
        transition = replay.transition(model, compiled)
        model.zero_grad(set_to_none=True)
        live = model.observation_features(b['x'], b['hist'], b['hmask'])
        state, total = None, 0.
        for t, features in enumerate(stored[:3]+[live]+stored[3:]):
            state, retrieved = transition(*features, poses[t], state)
            total = total+state['probe'].square().sum()+retrieved.square().mean()
            state = {k: v for k, v in state.items() if k != 'probe'}
        (total+state['slots'].square().sum()).backward()
        assert (replay._compiled is not None) == compiled
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys() and any(n.startswith('encoder.') for n in grads[0])
    # Eager cuDNN convolution backward alone varies by ~1e-4 between repeats.
    # Inductor's historical gradient corruption was 30-10,000x in magnitude.
    for name, eager in grads[0].items():
        error = (grads[1][name]-eager).norm()/eager.norm().clamp_min(1e-12)
        assert error < 1e-3, (name, float(error))
