"""Continuous coordinates, direct identity gradients, and v4 train/trace parity."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_learned_memory import memory_config, memory_batch
from test_neighbor_bank import make_bank, add_shard, publish
from test_neighbor_following import clean_sample
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    build_model, DirectConfig, TRAJECTORY_MEMORY_ARCHITECTURE,
)
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, DirectTracer
from vesuvius.neural_tracing.fiber_follow.regression.identity_decisions import decision_pair
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    initialize_trajectory_model, add_direction_inputs, optimizer_update, save_checkpoint,
    load_checkpoint, checkpoint_config, build_parser,
)
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams
from vesuvius.neural_tracing.fiber_follow.shared.runloop import resume_training, training_rng_state


def cfg(**kwargs):
    return memory_config(memory_version=4, correction=False, **kwargs)


def test_single_pass_continuous_curve_and_connection_bound():
    torch.manual_seed(21)
    model = build_model(cfg(max_recovery_distance=2.)).eval()
    b = memory_batch(model.cfg)
    calls = []
    handle = model.decoder.register_forward_hook(lambda *args: calls.append(1))
    out = model(b['x'], b['hist'], b['hmask'])
    handle.remove()
    assert calls == [1]
    assert not hasattr(model, 'correction_head') and not hasattr(model, 'route_score')
    assert 'route_logits' not in out
    assert out['points'].shape == (2, model.cfg.n_future, 3)
    assert out['refinement_points'].shape == (2, 1, model.cfg.n_future, 3)
    torch.testing.assert_close(out['initial_points'], out['points'], rtol=0, atol=0)
    torch.testing.assert_close(out['points'][..., 2], model.planes.expand(2, -1))
    assert (out['points'][..., :2]-out['points'][..., :2].round()).abs().max() > 1e-3
    assert (out['confidence'].diff(dim=-1) <= 0).all()
    with torch.no_grad():
        model.coordinates.weight.zero_()
        model.coordinates.bias.fill_(100.)
        out = model(b['x'], b['hist'], b['hmask'])
    assert (out['points'][:, 0].norm(dim=-1) <= model.cfg.max_recovery_distance+1e-5).all()
    assert out['points'][..., :2].abs().max() <= model.cfg.lateral_limit


def test_geometry_alone_trains_seed_writer_and_earlier_observations():
    torch.manual_seed(19)
    model = build_model(cfg())
    b = memory_batch(model.cfg)
    # No visible seed/history can supply an alternative identity gradient path.
    b['hmask'].zero_()
    b['x']['seed_mask'].zero_()
    b['x']['memory_patches'].requires_grad_()
    b['x']['memory_seed_patch'].requires_grad_()
    b['dense_ab'].fill_(.8)  # recovery offset, not a straight-head target
    out = model(b['x'], b['hist'], b['hmask'])
    terms = loss_terms(out, b, model.cfg)
    assert 'route_per_state' not in terms
    terms['geometry_per_state'].mean().backward()
    assert b['x']['memory_patches'].grad[:, 0].abs().sum() > 0
    assert b['x']['memory_seed_patch'].grad.abs().sum() > 0
    for param in (model.coordinates.weight, model.decoder.layers[0].self_attn.in_proj_weight,
                  model.recurrent_memory.gate.weight, model.recurrent_memory.proposal.weight,
                  model.recurrent_memory.probe_head[-1].weight):
        assert param.grad is not None and param.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_future_points_influence_each_other_in_both_directions():
    torch.manual_seed(9)
    c = cfg(n_future=8, fine=CropSpec(depth=32, width=17, behind=8))
    model = build_model(c).eval()
    b = memory_batch(c, 1)
    queries = []
    handle = model.query.register_forward_hook(lambda module, inputs, output: queries.append(output))
    out = model(b['x'], b['hist'], b['hmask'])
    handle.remove()
    # Point 2 depends on query 8, and point 8 depends on query 2. This fails
    # for independent point heads or a causal future-point attention mask.
    for predicted, other in ((1, 7), (7, 1)):
        gradient, = torch.autograd.grad(out['points'][0, predicted, 0], queries[0], retain_graph=True)
        assert gradient[0, other].abs().sum() > 1e-8


def test_same_crop_reads_persistent_slots_and_seed_in_crop_frame():
    torch.manual_seed(23)
    model = build_model(cfg()).eval()
    b = memory_batch(model.cfg, 1)
    frame = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    b['x']['query_frame'] = frame
    ctx = model.context(b['x'], b['hist'], b['hmask'])
    state = model.recurrent_memory.observe(b['x'])
    # Hold the local crop and latest observation fixed; vary only old memory.
    state = {k: v.detach() for k, v in state.items()}
    ctx['recurrent'] = state
    baseline = model.predict(ctx, b['hist'])['points']
    for key in ('slots', 'anchor'):
        changed = dict(state)
        changed[key] = state[key]+torch.randn_like(state[key])*2
        ctx['recurrent'] = changed
        actual = model.predict(ctx, b['hist'])['points']
        assert (actual-baseline).abs().max() > 1e-5, key
    seen = []
    original = model.recurrent_memory.read_tokens
    def capture(query_state):
        seen.append(query_state['frame'].clone())
        return original(query_state)
    model.recurrent_memory.read_tokens = capture
    ctx['recurrent'] = state
    model.predict(ctx, b['hist'])
    torch.testing.assert_close(seen[0], frame)
    torch.testing.assert_close(state['frame'], torch.eye(3)[None])


def test_censored_and_departed_states_do_not_train_coordinates():
    model = build_model(cfg())
    b = memory_batch(model.cfg)
    b['offtrack'][0] = 1
    b['dense_mask'][1].zero_()
    out = model(b['x'], b['hist'], b['hmask'])
    geometry = loss_terms(out, b, model.cfg)['geometry_per_state']
    assert geometry.eq(0).all()
    geometry.sum().backward()
    assert model.coordinates.weight.grad.eq(0).all()


@pytest.mark.parametrize('version', [2, 3])
def test_migration_reuses_shared_weights_and_expands_directions(version):
    old = build_model(memory_config(memory_version=version))
    c = replace(old.cfg, memory_version=4, correction=False, route_refinement_radius=None)
    model = initialize_trajectory_model(old, c)
    assert model.architecture == TRAJECTORY_MEMORY_ARCHITECTURE
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, old.state_dict()[key], rtol=0, atol=0)
    assert old.cfg.memory_version == version and old.cfg.correction
    expanded = add_direction_inputs(model)
    assert expanded.cfg.direction_inputs and expanded.cfg.memory_version == 4
    assert expanded.encoder.stem[0].weight[:, 2:].eq(0).all()
    assert expanded.recurrent_memory.patch_encoder[0].weight[:, 2:].eq(0).all()


def test_training_sequence_checkpoint_and_optimizer_resume(tmp_path):
    torch.manual_seed(32)
    model = build_model(cfg())
    b = memory_batch(model.cfg)
    b['trajectory_sequence'] = memory_batch(model.cfg, 1)
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    before = model.coordinates.weight.detach().clone()
    metrics = optimizer_update(model, ema, opt, [b], 1, .001, device='cpu', compute_metrics=False)
    assert np.isfinite(metrics['loss'])
    assert not torch.equal(before, model.coordinates.weight)
    assert metrics['memory']['sequence_states'] == 1 and metrics['memory']['sequence_loss'] > 0
    assert 'route_loss' not in metrics['memory']
    path = tmp_path/'v4.pt'
    spec = FiberVolumeSpec('/tmp/presence', ct_zarr='/tmp/ct', inputs='ct+presence')
    sample = SampleConfig(crop=model.cfg.fine, n_history=model.cfg.n_history, n_future=model.cfg.n_future)
    save_checkpoint(path, model, ema, spec, sample,
                    dict(optimizer=opt.state_dict(), rng=training_rng_state(), step=1))
    restored, *_, ck = load_checkpoint(path, 'cpu')
    assert restored.architecture == TRAJECTORY_MEMORY_ARCHITECTURE
    with torch.no_grad():
        expected = ema.eval()(b['x'], b['hist'], b['hmask'])
        actual = restored(b['x'], b['hist'], b['hmask'])
    for key in ('points', 'confidence', 'memory_slots'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    restored_ema = copy.deepcopy(restored)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=.001)
    assert resume_training(ck, restored, restored_ema, restored_opt)[0] == 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    for key, state in opt.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(restored_opt.state_dict()['state'][key][name], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match='architecture'):
        checkpoint_config(dict(ck, architecture='axial_fiber_memory_v3'))


def test_builder_keeps_paired_local_inputs_identical_and_includes_causal_replay(tmp_path, monkeypatch):
    bank, parent = make_bank(tmp_path)
    publish(tmp_path, [add_shard(tmp_path, 0, x=4., z_range=(20., 180.))])
    c = cfg(fine=CropSpec(depth=40, width=25, behind=16, spacing=1.), memory_steps=16)
    builder = IdentityObservationBuilder(c, [parent], negative_bank=bank, augment=True)
    rng = np.random.default_rng(11)
    rows = decision_pair(bank, clean_sample(c), c, rng)
    assert rows is not None
    for row in rows:
        builder.prepare(row, row.get('supervision_fiber', parent), rng)
        assert builder.footprint_allowed(row, None)
    def images(items, vol, crop, pool=None, **kwargs):
        return torch.ones(len(items), 2, crop.depth, crop.width, crop.width)*.25
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.image_crop', images)
    b = builder(rows, None)
    assert 'trajectory_sequence' in b and 'route_mask' not in b
    assert 'trajectory_sequence' not in b['trajectory_sequence']
    assert {'route_ab', 'plane_ab', 'offtrack', 'memory_target_identity'}.isdisjoint(b['x'])
    assert b['x']['query_frame'].shape == (2, 3, 3)
    assert not b['x']['seed_mask'].any()
    torch.testing.assert_close(b['x']['fine'][0], b['x']['fine'][1], rtol=0, atol=0)
    model = build_model(c)
    metrics = optimizer_update(model, copy.deepcopy(model), torch.optim.AdamW(model.parameters()),
                               [b], 1, .001, device='cpu', compute_metrics=False)
    assert np.isfinite(metrics['loss']) and metrics['memory']['sequence_states'] > 0


def test_streaming_memory_matches_reconstruction_and_respects_burn_in():
    torch.manual_seed(12)
    model = build_model(cfg(memory_steps=6, memory_grad_steps=2)).eval()
    b = memory_batch(model.cfg, 1)
    x = b['x']
    sequence_keys = ('memory_patches', 'memory_mask', 'memory_positions', 'memory_frames')
    with torch.no_grad():
        expected = model(x, b['hist'], b['hmask'])
        state = model.initial_memory(1, 'cpu')
        for index in range(x['memory_mask'].shape[1]):
            current = {k: v[:, index:index+1] if k in sequence_keys else v for k, v in x.items()}
            current['memory_seed_valid'] = x['memory_seed_valid'] & (index == 0)
            actual = model(current, b['hist'], b['hmask'], memory=state)
            state = {k: actual['memory_'+k] for k in state}
    for key in ('points', 'confidence', 'memory_slots', 'memory_anchor', 'memory_recent'):
        torch.testing.assert_close(actual[key], expected[key])
    x['memory_patches'].requires_grad_()
    out = model(x, b['hist'], b['hmask'])
    loss_terms(out, b, model.cfg)['geometry_per_state'].sum().backward()
    assert x['memory_patches'].grad[:, :-3].eq(0).all()
    assert x['memory_patches'].grad[:, -3:].abs().sum() > 0


def test_tracer_retains_memory_across_commits(monkeypatch):
    model = build_model(cfg())
    def images(items, vol, crop, pool=None, **kwargs):
        return torch.ones(len(items), 2, crop.depth, crop.width, crop.width)*.25
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.image_crop', images)
    seen = []
    observe = model.recurrent_memory.observe
    def capture(x, state=None):
        seen.append((x['memory_seed_valid'].clone(), state['slots'].clone(), x['query_frame'].clone()))
        return observe(x, state)
    monkeypatch.setattr(model.recurrent_memory, 'observe', capture)
    tracer = DirectTracer(model, SimpleNamespace(shape=(1000, 1000, 1000)), model.cfg.fine, model.cfg.n_history,
                          TraceParams(n_commit=1, max_len=8, confidence=0.), device='cpu')
    try:
        paths, _ = tracer.trace(np.array([[100., 100., 400.]]), np.array([[0., 0., 1.]]))
    finally:
        tracer.close()
    assert len(seen) >= 2 and len(paths[0]) > 2
    assert seen[0][0].all() and all(not row[0].any() for row in seen[1:])
    assert not torch.equal(seen[0][1], seen[1][1])


def test_fullgraph_capture_and_backward():
    model = build_model(cfg())
    b = memory_batch(model.cfg, 1)
    compiled = torch.compile(model, backend='eager', fullgraph=True)
    out = compiled(b['x'], b['hist'], b['hmask'])
    loss_terms(out, b, model.cfg)['geometry_per_state'].mean().backward()
    assert model.recurrent_memory.gate.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('kwargs', [dict(memory_slots=0), dict(correction=True),
                                    dict(route_refinement_radius=1.), dict(trajectory_sequence_weight=-1.),
                                    dict(trajectory_sequence_weight=float('nan'))])
def test_v4_rejects_incompatible_configuration(kwargs):
    options = dict(memory_version=4, memory_slots=4, correction=False)
    options.update(kwargs)
    with pytest.raises(ValueError):
        DirectConfig(**options)


def test_parser_exposes_single_pass_v4():
    args = build_parser().parse_args(['--name', 'test', '--memory-version', '4', '--no-correction',
                                     '--trajectory-sequence-weight', '0', '--fiber-zarrs', '/tmp/presence',
                                     '--fibers', '/tmp/fibers', '--ct', '/tmp/ct', '--manifest', '/tmp/seeds.json'])
    assert args.memory_version == 4 and not args.correction and args.trajectory_sequence_weight == 0
