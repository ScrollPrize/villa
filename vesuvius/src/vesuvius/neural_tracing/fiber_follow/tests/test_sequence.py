"""Sequence follower (model 'sequence'): one transformer reads the trace's history and predicts the path."""
from types import SimpleNamespace

import numpy as np
import torch

from model_fixtures import coordinate_batch, ct_volume
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.models.sequence import SequenceConfig
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.train.sequence import episode_forward


def small_config(**options):
    return SequenceConfig(**dict(dict(fine=CropSpec(depth=32, width=16, behind=8, spacing=.5), n_future=8, gate_plane=4,
                                      hidden=32, layers=3, heads=2, ffn=64, cnn_channels=(4, 8, 8, 16), cnn_blocks=1,
                                      n_history=32), **options))


def small_model(seed=0):
    torch.manual_seed(seed)
    return build_model(small_config())


def episode_batch(model, episodes=2, steps=5, supervised=2):
    n = episodes*steps
    batch = coordinate_batch(model.cfg, n)
    step = torch.arange(steps).repeat(episodes)
    pos = torch.zeros(n, 3, dtype=torch.float64)
    pos[:, 2] = step.double()*4.
    segment = pos[:, None].float()+torch.tensor([0., 0., 1.])*torch.arange(5.)[None, :, None]
    batch.update(episode_index=torch.arange(episodes).repeat_interleave(steps), episode_step=step,
                 episode_supervised=step >= steps-supervised, episode_segment=segment,
                 episode_segment_mask=torch.ones(n, 5, dtype=torch.bool), crop_pos=pos,
                 crop_frame=torch.eye(3, dtype=torch.float64).expand(n, 3, 3).clone(), travelled=step.float()*4.)
    return batch


def test_history_stream_is_causal_and_extends_incrementally():
    model = small_model().eval()
    tokens = torch.randn(2, 6, model.cfg.hidden)
    with torch.no_grad():
        full = model.history_states(tokens)
        changed = tokens.clone()
        changed[:, 4:] += 1.
        for a, b in zip(model.history_states(changed), full):
            torch.testing.assert_close(a[:, :4], b[:, :4])  # no layer input of a step reads a later step
        layers = [[] for _ in range(model.cfg.layers)]
        for j in range(6):
            past = [torch.stack(layer, 1) if layer else tokens.new_zeros(2, 0, model.cfg.hidden) for layer in layers]
            for layer, value in zip(layers, model.extend_history(tokens[:, j], past)):
                layer.append(value)
    for layer, expected in zip(layers, full):
        torch.testing.assert_close(torch.stack(layer, 1), expected, rtol=1e-4, atol=1e-5)


def test_a_decision_reads_only_earlier_steps_of_its_own_episode():
    model = small_model().eval()
    batch = episode_batch(model)
    with torch.no_grad():
        out, rows = episode_forward(model, batch, .4, 4)
        assert len(rows['hist']) == 4 and out['points'].shape == (4, 8, 3)
        changed = dict(batch, x=dict(batch['x'], fine=batch['x']['fine'].clone()))
        changed['x']['fine'][4] += 1.  # episode 0, step 4: the step-3 decision must not read it
        changed['x']['fine'][9] += 1.  # episode 1, step 4: episode 0 never reads episode 1
        out2, _ = episode_forward(model, changed, .4, 4)
        changed['x']['fine'][2] += 1.  # episode 0, step 2: read by the step-3 decision
        out3, _ = episode_forward(model, changed, .4, 4)
    torch.testing.assert_close(out2['points'][0], out['points'][0])
    torch.testing.assert_close(out2['points'][2], out['points'][2])
    assert not torch.allclose(out3['points'][0], out['points'][0])


def test_training_step_on_an_episode_batch():
    from vesuvius.neural_tracing.fiber_follow.train.train import (
        initialize_model_weights, initialize_training_optimizer, optimizer_update, prepare_training)
    torch.manual_seed(1)
    model, ema = initialize_model_weights(small_config(), 'cpu')
    opt, _, _ = initialize_training_optimizer(model, ema, SimpleNamespace(lr=.001, reset_optimizer=False))
    prepare_training(model, 4)
    batch = episode_batch(model)
    metrics = optimizer_update(model, ema, opt, [batch], 1, .001, tolerance=2., confidence_threshold=.4, gate='full')
    assert np.isfinite(metrics['loss']) and metrics['supervised_states'] == 4 and metrics['observed_states'] == 10
    assert model.history_token[1].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.cnn.parameters())


def test_tracing_builds_whole_trace_history_one_step_per_commit(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.data.observations import FiberTracer
    from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams

    class Recorder(FiberTracer):
        def sequence_attach(self, x, idx, sequence, pos, frames, length):
            super().sequence_attach(x, idx, sequence, pos, frames, length)
            self.read.append((idx.copy(), (~x['sequence_padding']).sum(1).tolist()))
            self.sequence = sequence

    model = small_model().eval()
    vol = ct_volume(tmp_path)
    tracer = Recorder(model, vol, model.cfg.fine, model.cfg.n_history,
                      TraceParams(n_commit=1, max_len=3., confidence=0.), device='cpu')
    tracer.read = []
    try:
        paths, reasons = tracer.trace(np.array([[24., 24., 24.], [24.3, 23.8, 24.1]]), np.array([[.3, .4, .8660254]]*2))
    finally:
        tracer.close()
    assert all(r == 'max_len' for r in reasons)
    assert tracer.read[0][1] == [0, 0]  # the first decision reads no history
    for decision, (idx, counts) in enumerate(tracer.read):
        assert counts == [decision]*len(idx)  # one history token per earlier commit of the same trace
    assert all(len(s['heads']) == len(tracer.read) and all(len(l) == len(tracer.read) for l in s['layers'])
               for s in tracer.sequence)


def test_single_episode_loader_items_merge_into_one_microbatch():
    from vesuvius.neural_tracing.fiber_follow.train.sequence import merge_episodes

    def item(steps, segment, counter=None):
        out = dict(hist=torch.zeros(steps, 4, 3), x=dict(fine=torch.full((steps, 1, 2, 2, 2), float(steps))),
                   episode_index=torch.zeros(steps, dtype=torch.long), episode_step=torch.arange(steps),
                   episode_segment=torch.ones(steps, segment, 3), episode_segment_mask=torch.ones(steps, segment, dtype=torch.bool),
                   _live_states=[None]*steps)
        if counter is not None:
            out['ct_seed_rejections'] = torch.zeros(steps, dtype=torch.long)
            out['ct_seed_rejections'][0] = counter
        return out
    batch = merge_episodes([item(3, 13), item(2, 17, counter=5)])
    assert batch['episode_index'].tolist() == [0, 0, 0, 1, 1] and batch['x']['fine'][:, 0, 0, 0, 0].tolist() == [3, 3, 3, 2, 2]
    assert batch['episode_segment'].shape == (5, 17, 3) and batch['episode_segment_mask'].sum(1).tolist() == [13]*3+[17]*2
    assert batch['ct_seed_rejections'].sum() == 5 and len(batch['_live_states']) == 5
