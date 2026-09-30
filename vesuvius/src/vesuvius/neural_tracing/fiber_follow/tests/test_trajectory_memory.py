"""Continuous coordinates, direct identity gradients, and v4 train/trace parity."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_identity import config as memory_config
from test_identity import batch as base_batch
from test_neighbor_bank import make_bank, add_shard, publish
from test_neighbor_following import clean_sample
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    build_model, DirectConfig, ARCHITECTURE,
)
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, DirectTracer
from vesuvius.neural_tracing.fiber_follow.regression.identity_decisions import decision_pair
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    optimizer_update, save_checkpoint, prepare_training,
    load_checkpoint, checkpoint_config, build_parser,
)
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams
from vesuvius.neural_tracing.fiber_follow.shared.runloop import resume_training, training_rng_state


def cfg(**kwargs):
    return memory_config(**kwargs)


def memory_batch(c, b=2, step=0):
    data = base_batch(c, b)
    data['x'].update(query_position=torch.tensor([0., 0., float(step)*30]).expand(b, -1).clone(),
        query_frame=torch.eye(3).expand(b, -1, -1).clone(), feature_seed_here=torch.full((b,), step == 0),
        memory_mask=torch.ones(b, 1, dtype=torch.bool), memory_seed_valid=torch.ones(b, dtype=torch.bool))
    return data


def state_from(model, output):
    return {k: output['memory_'+k] for k in model.initial_memory(0, 'cpu')}


def training_chunk(c, start=0, end=False):
    steps = [memory_batch(c, 2, step=t) for t in range(start, start+2)]
    for t, b in enumerate(steps, start):
        b.update(stream_id=torch.tensor([10, 20]), stream_reset=torch.full((2,), t == 0),
                 stream_end=torch.full((2,), end and t == start+1),
                 stream_index=torch.full((2,), t), decision_mask=torch.ones(2, dtype=torch.bool),
                 loss_weight=torch.ones(2), retain_until=torch.full((2,), 256),
                 encoder_indices=torch.tensor([[0 if t else -1, -1, -1]]).expand(2, -1).clone())
    return dict(feature_sequence=steps)


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


def test_geometry_alone_trains_seed_writer_and_earlier_main_encoder_features():
    torch.manual_seed(19)
    model = build_model(cfg())
    old = memory_batch(model.cfg, step=0)
    b = memory_batch(model.cfg, step=1)
    for row in (old, b):
        row['hmask'].zero_()
        row['x']['seed_mask'].zero_()
        row['x']['fine'].requires_grad_()
    b['dense_ab'].fill_(.8)
    first = model(old['x'], old['hist'], old['hmask'])
    out = model(b['x'], b['hist'], b['hmask'], memory=state_from(model, first))
    terms = loss_terms(out, b, model.cfg)
    assert 'route_per_state' not in terms
    terms['geometry_per_state'].mean().backward()
    assert old['x']['fine'].grad.abs().sum() > 0
    assert b['x']['fine'].grad.abs().sum() > 0
    for param in (model.coordinates.weight, model.decoder.layers[0].self_attn.in_proj_weight,
                  model.encoder.stem[0].weight, model.recurrent_memory.gate.weight,
                  model.recurrent_memory.proposal.weight, model.recurrent_memory.feature_projection[-1].weight):
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
    state, _ = model.observe_context(ctx, b['x'], b['hist'], b['hmask'], None)
    # Hold the local crop and latest observation fixed; vary only old memory.
    state = {k: v.detach() for k, v in state.items()}
    ctx['recurrent'] = state
    baseline = model.select_prediction(model.predict(ctx, b['hist']))['points']
    for key in ('slots', 'anchor'):
        changed = dict(state)
        changed[key] = state[key]+torch.randn_like(state[key])*2
        ctx['recurrent'] = changed
        actual = model.select_prediction(model.predict(ctx, b['hist']))['points']
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
    torch.testing.assert_close(state['frame'], frame)


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




@pytest.mark.parametrize('encoder', ['conv', 'patch4'])
def test_training_sequence_checkpoint_and_optimizer_resume(tmp_path, encoder):
    torch.manual_seed(32)
    model = build_model(cfg(encoder=encoder))
    b = memory_batch(model.cfg)
    chunk = training_chunk(model.cfg)
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    before = model.coordinates.weight.detach().clone()
    prepare_training(model, backend='eager')
    metrics = optimizer_update(model, ema, opt, [chunk], 1, .001, device='cpu', compute_metrics=False)
    assert np.isfinite(metrics['loss'])
    assert not torch.equal(before, model.coordinates.weight)
    assert metrics['observed_states'] == 4
    assert 'route_loss' not in metrics.get('memory', {})
    path = tmp_path/'current.pt'
    spec = FiberVolumeSpec('/tmp/presence', ct_zarr='/tmp/ct', inputs='ct+presence')
    sample = SampleConfig(crop=model.cfg.fine, n_history=model.cfg.n_history, n_future=model.cfg.n_future)
    save_checkpoint(path, model, ema, spec, sample,
                    dict(optimizer=opt.state_dict(), rng=training_rng_state(), step=1))
    if encoder == 'conv':
        legacy = torch.load(path, weights_only=False)
        del legacy['model_cfg']['encoder']
        torch.save(legacy, path)
    restored, *_, ck = load_checkpoint(path, 'cpu')
    assert restored.architecture == model.cfg.architecture
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
    # Matching tensor shapes do not make an older architecture acceptable.
    for version in range(1, 9):
        torch.save(dict(ck, architecture=f'axial_fiber_memory_v{version}'), path)
        with pytest.raises(ValueError, match='Checkpoint'):
            load_checkpoint(path, 'cpu')


def test_builder_streams_causal_main_crops_and_keeps_paired_endpoints_identical(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.regression.feature_sequences import sequence_batches
    from vesuvius.neural_tracing.fiber_follow.regression.stratified_replay import ObservationHistory
    bank, parent = make_bank(tmp_path)
    publish(tmp_path, [add_shard(tmp_path, 0, x=4., z_range=(20., 180.))])
    c = cfg(fine=CropSpec(depth=40, width=25, behind=16, spacing=1.), memory_steps=4)
    builder = IdentityObservationBuilder(c, [parent], negative_bank=bank, augment=True)
    rng = np.random.default_rng(11)
    rows = decision_pair(bank, clean_sample(c), c, rng)
    assert rows is not None
    for row in rows:
        builder.prepare(row, row.get('supervision_fiber', parent), rng)
    reads = []
    def images(items, vol, crop, pool=None, **kwargs):
        reads.extend((np.asarray(i['pos']).copy(), crop) for i in items)
        return torch.ones(len(items), 2, crop.depth, crop.width, crop.width)*.25
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.image_crop', images)
    chunks = list(sequence_batches(builder, rows, None))
    assert len(chunks) >= 2
    steps = [b for chunk in chunks for b in chunk['feature_sequence']]
    assert len(reads) == sum(len(b['hist']) for b in steps)
    assert all(crop == c.fine for _, crop in reads)
    endpoints = []
    for b in steps:
        assert 'memory_patches' not in b['x'] and 'feature_seed_x' not in b['x']
        assert not any(k.startswith('memory_target_') for k in b)
        endpoints.extend(b['x']['fine'][b['stream_end']].unbind())
    torch.testing.assert_close(endpoints[0], endpoints[1], rtol=0, atol=0)
    model = build_model(c)
    ema, opt, states = copy.deepcopy(model), torch.optim.AdamW(model.parameters()), ObservationHistory()
    prepare_training(model, backend='eager')
    for j, chunk in enumerate(chunks):
        metrics = optimizer_update(model, ema, opt, [chunk], j+1, .001, device='cpu',
                                   compute_metrics=False, stream_states=states)
        assert np.isfinite(metrics['loss'])
        assert all(not v.requires_grad for rows in states.streams.values() for row in rows for v in row['features'])
    assert not states.streams


def test_streaming_keeps_remote_seed_and_cache_until_explicit_eviction():
    model = build_model(cfg(memory_steps=2)).eval()
    state = None
    original_anchor = None
    for t in range(5):
        b = memory_batch(model.cfg, 1, t)
        with torch.no_grad():
            out = model(b['x'], b['hist'], b['hmask'], memory=state)
        state = state_from(model, out)
        if original_anchor is None:
            original_anchor = state['anchor'].clone()
        torch.testing.assert_close(state['anchor'], original_anchor, rtol=0, atol=0)
    assert state['cache_valid'].all()
    # The previous crop is 30 voxels away, well beyond the current crop width.
    assert (state['cache_xyz'][:, 0, :, 2]-state['position'][:, None, 2]).abs().min() > 10
    _, padding = model.recurrent_memory.read_tokens(state)
    assert not padding.any()
    assert state['cache'].shape[1] == 2
    assert state['cache_age'].tolist() == [[1., 0.]]
    ctx = model.context(b['x'], b['hist'], b['hmask'])
    model.observe_context(ctx, b['x'], b['hist'], b['hmask'], state)
    ctx['recurrent'] = state
    baseline = model.select_prediction(model.predict(ctx, b['hist']))['points']
    changed = dict(state, cache=state['cache'].clone())
    changed['cache'][:, 0] += torch.randn_like(changed['cache'][:, 0])*2
    ctx['recurrent'] = changed
    assert (model.select_prediction(model.predict(ctx, b['hist']))['points']-baseline).abs().max() > 1e-5
    # Detaching preserves the exact information and predictions.
    current = memory_batch(model.cfg, 1, 6)
    a = model(current['x'], current['hist'], current['hmask'], memory=state)
    z = model(current['x'], current['hist'], current['hmask'], memory={k:v.detach() for k,v in state.items()})
    torch.testing.assert_close(a['points'], z['points'], rtol=0, atol=0)


def test_tracer_retains_memory_across_commits(monkeypatch):
    model = build_model(cfg())
    def images(items, vol, crop, pool=None, **kwargs):
        return torch.ones(len(items), 2, crop.depth, crop.width, crop.width)*.25
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.image_crop', images)
    seen = []
    observe = model.recurrent_memory.observe_tokens
    def capture(tokens, xyz, valid, x, state=None):
        seen.append((x['feature_seed_here'].clone(), state['slots'].clone(), x['query_frame'].clone()))
        return observe(tokens, xyz, valid, x, state)
    monkeypatch.setattr(model.recurrent_memory, 'observe_tokens', capture)
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






def test_cold_remote_seed_is_encoded_once_and_warm_trace_reads_only_current_crop(monkeypatch):
    model = build_model(cfg()).eval()
    reads = []
    def images(items, vol, crop, pool=None, **kwargs):
        reads.extend(np.asarray(i['pos']).copy() for i in items)
        return torch.ones(len(items), 2, crop.depth, crop.width, crop.width)*.25
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.image_crop', images)
    tracer = DirectTracer(model, SimpleNamespace(shape=(1000, 1000, 1000)), model.cfg.fine,
                          model.cfg.n_history, TraceParams(n_commit=1), device='cpu')
    b = memory_batch(model.cfg, 1)
    pos = np.array([[100., 100., 400.]])
    paths = [dict(seed_pos=np.array([100., 100., 300.]), seed_tangent=np.array([0., 0., 1.]),
                  seed_valid=True, seed_age=100., memory_warm=False)]
    try:
        x = tracer.build_inputs(pos, np.eye(3)[None], b['hist'].numpy(), b['hmask'].numpy(), paths)
        assert 'feature_seed_x' in x and len(reads) == 2
        with torch.no_grad():
            first = model(x, b['hist'], b['hmask'])
        state = state_from(model, first)
        assert state['anchor_valid'].all()
        torch.testing.assert_close(state['anchor_position'], torch.tensor([[100., 100., 300.]]))
        paths[0]['memory_warm'] = True
        x = tracer.build_inputs(pos+np.array([0., 0., 8.]), np.eye(3)[None], b['hist'].numpy(), b['hmask'].numpy(), paths)
        assert 'feature_seed_x' not in x and len(reads) == 3
        with torch.no_grad():
            second = model(x, b['hist'], b['hmask'], memory=state)
        torch.testing.assert_close(second['memory_anchor'], first['memory_anchor'], rtol=0, atol=0)
    finally:
        tracer.close()


def test_stream_ownership_reset_detach_and_eviction_across_optimizer_updates():
    from vesuvius.neural_tracing.fiber_follow.regression.stratified_replay import ObservationHistory
    model = build_model(cfg())
    ema, opt, states = copy.deepcopy(model), torch.optim.AdamW(model.parameters()), ObservationHistory()
    prepare_training(model, backend='eager')
    with pytest.raises(ValueError, match='Missing or out-of-order'):
        states.start(training_chunk(model.cfg, start=2)['feature_sequence'][0])
    first = training_chunk(model.cfg)
    optimizer_update(model, ema, opt, [first], 1, .001, compute_metrics=False, stream_states=states)
    assert set(states.streams) == {10, 20}
    assert all(not v.requires_grad for rows in states.streams.values() for row in rows for v in row['features'])
    later = training_chunk(model.cfg, start=2, end=True)
    # Keep associations when a loader batch reorders independent traces.
    later['feature_sequence'][0]['stream_id'] = torch.tensor([20, 10])
    diagnostic = {}
    result = optimizer_update(model, ema, opt, [later], 2, .001, compute_metrics=False,
                              stream_states=states, diagnostic=diagnostic)
    assert np.isfinite(result['loss']) and result['observed_states'] == 4
    assert not states.streams
    assert diagnostic['memory']['anchor_valid'].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_compiled_feature_sequence_bf16_gradients_match_eager():
    from vesuvius.neural_tracing.fiber_follow.regression.train import prepare_training, conv_memory_format
    torch.manual_seed(77)
    eager = build_model(cfg()).to('cuda', memory_format=conv_memory_format('cuda'))
    compiled_base = copy.deepcopy(eager)
    chunk = training_chunk(eager.cfg)
    results = []
    # Capture the same training graph without Inductor as the numerical oracle.
    for model in (prepare_training(eager, backend='eager'), prepare_training(compiled_base)):
        results.append(optimizer_update(model, copy.deepcopy(eager), torch.optim.AdamW(model.parameters()),
            [chunk], 1, 0., device='cuda', compute_metrics=False, memory_grad_clip=0., rest_grad_clip=0.))
    assert results[0]['loss'] == pytest.approx(results[1]['loss'], rel=.02, abs=.002)
    for prefix in ('encoder.', 'decoder.', 'recurrent_memory.'):
        gradients = [torch.cat([p.grad.flatten() for name,p in m.named_parameters()
                               if name.startswith(prefix) and p.grad is not None]).float()
                     for m in (eager, compiled_base)]
        assert all(torch.isfinite(g).all() for g in gradients)
        assert torch.nn.functional.cosine_similarity(*gradients, dim=0) > .99
        assert (gradients[1].norm()/gradients[0].norm()).item() == pytest.approx(1., rel=.05)
