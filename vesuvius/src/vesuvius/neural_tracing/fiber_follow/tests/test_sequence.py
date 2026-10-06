"""Sequence follower (model 'sequence'): one transformer reads the trace's history and predicts the path."""
from types import SimpleNamespace

import numpy as np
import torch

from model_fixtures import coordinate_batch, line_fiber
from vesuvius.neural_tracing.fiber_follow.data import data as D
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.models.sequence import SequenceConfig, relative_pose
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength
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
    batch['x'] = {k: v for k, v in batch['x'].items() if not k.startswith('history_')}
    step = torch.arange(steps).repeat(episodes)
    pos = torch.zeros(n, 3, dtype=torch.float64)
    pos[:, 2] = step.double()*4.
    segment = pos[:, None].float()+torch.tensor([0., 0., 1.])*torch.arange(5.)[None, :, None]
    batch.update(episode_index=torch.arange(episodes).repeat_interleave(steps), episode_step=step,
                 episode_supervised=step >= steps-supervised, episode_segment=segment,
                 episode_segment_mask=torch.ones(n, 5, dtype=torch.bool), crop_pos=pos,
                 crop_frame=torch.eye(3, dtype=torch.float64).expand(n, 3, 3).clone(), travelled=step.float()*4.)
    return batch


def test_a_decision_without_history_predicts_every_plane():
    model = small_model().eval()
    batch = coordinate_batch(model.cfg, 2)
    with torch.no_grad():
        out = model(batch['x'], batch['hist'], batch['hmask'])
    assert out['points'].shape == (2, 8, 3) and out['confidence'].shape == (2, 8)
    assert out['sequence_cells'].shape == (2, 16, 4, 2, 2)
    assert torch.all(out['confidence'][:, 1:] <= out['confidence'][:, :-1])


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


def test_relative_pose_is_the_past_head_in_the_decision_frame():
    frame = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    local, age = relative_pose(torch.tensor([[[1., 2., 3.]]]), torch.tensor([[2.]]), torch.tensor([[0., 0., 3.]]),
                               frame[None], torch.tensor([10.]))
    torch.testing.assert_close(local, torch.tensor([[[2., 1., 0.]]]))
    torch.testing.assert_close(age, torch.tensor([[8.]]))


def test_episode_decisions_are_consecutive_heads_on_one_trace():
    fiber = line_fiber(600.)
    cfg = D.SampleConfig(n_future=8, n_history=32, recent_history_points=32)
    out = D.episode_decisions(fiber, 100., False, steps=6, commit=12, cfg=cfg, rng=np.random.default_rng(0))
    assert len(out) == 6
    for j, (item, segment) in enumerate(out):
        np.testing.assert_allclose(segment[0], item['pos'])
        assert len(segment) == 13
        if j:
            previous, _ = out[j-1]
            np.testing.assert_allclose(item['observed_path'][:len(previous['observed_path'])], previous['observed_path'])
            np.testing.assert_allclose(out[j-1][1][-1], item['pos'])  # a step commits exactly up to the next head
            assert abs(arclength(item['observed_path'])[-1]-arclength(previous['observed_path'])[-1]-12) < 1.5


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
    from test_rollout_threading import ct_volume
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


def recorded_trace(decisions=30, commit=12, terminal=20):
    """A replay cache holding every decision of one collected trace along z (heads ``commit`` voxels apart)."""
    from replay_fixtures import replay_states
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import REPLAY_CLASS, TERMINAL
    fiber = line_fiber(800.)
    track = np.c_[np.full(decisions*commit+1, 100.), np.full(decisions*commit+1, 100.), 200.+np.arange(decisions*commit+1.)]
    rows = [dict(pos=track[k*commit], t=200.+k*commit, travelled=float(k*commit), seq_start=0, seq_end=k*commit+1,
                 episode=0, source_row=k, hist=np.zeros((4, 3)), hmask=np.zeros(4, np.float32),
                 **(dict(replay_class=REPLAY_CLASS['terminal'], supervision=TERMINAL, event_id=0, hard=True)
                    if k == terminal else {}))
            for k in range(decisions)]
    return replay_states([fiber], rows, track=track), track


def episode_dataset(cache, steps=8, supervised=3):
    from vesuvius.neural_tracing.fiber_follow.data.data import FollowDataset, ReplayIndex
    ds = FollowDataset.__new__(FollowDataset)
    ds.episodes = D.EpisodeSpec(steps=steps, supervised=supervised, commit=12)
    ds.onpolicy, ds.index = [cache], ReplayIndex([cache])
    ds.eligible_caches, ds.claim = lambda: {0}, lambda key: True
    ds.replay_item = lambda op, j, rng: dict(row=j, pos=np.asarray(op.pos[j]))  # labels/crops are tested elsewhere
    return ds


def test_replay_episodes_are_consecutive_recorded_decisions_ending_near_the_event():
    cache, track = recorded_trace()
    ds = episode_dataset(cache)
    for seed in range(12):
        items = ds.replay_episode_items('terminal', np.random.default_rng(seed), index=3)
        rows = [it['row'] for it in items]
        assert len(items) == 8 and rows == list(range(rows[0], rows[0]+8))  # consecutive decisions of the trace
        supervised = [it['row'] for it in items if it['episode_supervised']]
        assert len(supervised) == 3 and 20 in supervised  # the terminal decision is supervised
        assert [it['episode_step'] for it in items] == list(range(8)) and {it['episode_index'] for it in items} == {3}
        for step, item in enumerate(items[:-1]):
            # A step commits the recorded trace from its head to the next head.
            np.testing.assert_allclose(item['episode_segment'], track[rows[step]*12:rows[step+1]*12+1])
        np.testing.assert_allclose(items[-1]['episode_segment'], track[rows[-1]*12][None])
    assert ds.replay_episode_items('recoverable', np.random.default_rng(0), index=0) is None  # no such event


def test_replay_episodes_need_every_decision_of_a_trace():
    import pytest
    cache, _ = recorded_trace()
    cache.source_row[10:] += 1  # one decision thinned away
    with pytest.raises(ValueError, match='every decision'):
        episode_dataset(cache).replay_episode_items('terminal', np.random.default_rng(0), index=0)


def test_episode_segments_pad_to_the_longest_recorded_commit():
    items = [dict(episode_index=0, episode_step=k, episode_supervised=k == 1, episode_segment=np.zeros((n, 3)))
             for k, n in enumerate((13, 17))]
    out = D.episode_tensors(items, 12)
    assert out['episode_segment'].shape == (2, 17, 3) and out['episode_segment_mask'].sum(1).tolist() == [13, 17]
