"""Live states use inference commits, original-fiber labels and bounded feedback."""
from dataclasses import replace
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.regression.live_continuation import LiveContinuation, LiveContinuationSource
from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch
from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset, SampleConfig, TracedFiber, ZBand, collate_targets, make_sample
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength
from vesuvius.neural_tracing.fiber_follow.shared.heading import orient_item
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec


@pytest.fixture
def setup(monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.regression.live_continuation as module
    # Deterministic in-process queues for numeric tests; real IPC is tested below.
    monkeypatch.setattr(module.mp, 'get_context', lambda: SimpleNamespace(
        Queue=Queue, Value=lambda *a: SimpleNamespace(value=0)))
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.ct_tensor',
                        lambda vol, pos: np.diag([3., 1., .1]))
    z = np.arange(500.)
    p = np.c_[100.+.4*np.sin(z/20), np.full(len(z), 100.), z+100.]
    fiber = TracedFiber('live', p, arclength(p), 'V', endpoint_stop=(True, True))
    cfg = SampleConfig(crop=CropSpec(depth=24, width=16, behind=8), n_history=32,
                       n_future=16, full_observed_history=True)
    ds = FollowDataset([fiber], FiberVolumeSpec('unused'), cfg, None, chunk=2,
                       clean_fraction=.7, replay_continuation_fraction=.8,
                       gt_perturb_probability=.25, prefer_replay_for_light_gt=True)
    live = LiveContinuationSource(steps=(4, 4), n_commit=16)
    ds.live_continuation = live
    vol = SimpleNamespace(shape=(1000,)*3)
    rng = np.random.default_rng(5)
    item = make_sample(fiber, 100., False, cfg, rng, perturb=False)
    item.update(fiber_ref=(0, 100., False), gt_unperturbed=True, source=0)
    orient_item(item, vol)
    return live, ds, vol, rng, item


def proposal(live, item, step=100, confidence=None):
    meta = live.metadata(item)
    assert meta is not None
    points = np.c_[np.full(16, .2), np.zeros(16), np.arange(1, 17)].astype(np.float32)
    wrapper = LiveContinuation.__new__(LiveContinuation)
    wrapper.sources = [live]
    wrapper.feedback(dict(_live_states=[meta]),
        dict(points=torch.tensor(points[None], requires_grad=True),
             confidence=torch.tensor([confidence or [.9]*16])), step)
    queue = live.chains if meta['depth'] else live.seeds
    return queue.get_nowait()


@pytest.mark.parametrize('reverse', [False, True])
def test_live_state_matches_next_inference_observation(setup, reverse):
    live, ds, vol, rng, item = setup
    if reverse:
        item = make_sample(ds.fibers[0], 100., True, ds.cfg, rng, perturb=False)
        item.update(fiber_ref=(0, 100., True), gt_unperturbed=True, source=0)
        orient_item(item, vol)
    state = proposal(live, item, confidence=[.9]*7+[.1]*9)
    assert state['commit'] == 7
    result = live.advance(state, ds, vol, rng)
    assert result['live_correct_continuation'] and result['live_depth'] == 1

    class Policy(torch.nn.Module):
        cfg = SimpleNamespace(n_future=16, max_recovery_distance=6.)
        def forward(self, x, hist, mask):
            return dict(points=torch.tensor(state['points'][None]),
                        confidence=torch.tensor([[.9]*7+[.1]*9]))

    class Tracer(ModelTracer):
        def build_inputs(self, *args):
            return None

    tracer = Tracer(Policy(), vol, ds.cfg.crop, ds.cfg.n_history, TraceParams(n_commit=16), device='cpu')
    initial = dict(item, hist=item['hist_local'] @ item['frame'].T+item['pos'], heading_start=0)
    observations = []
    def observe(i, decision):
        observations.append(decision)
        return len(observations) < 2
    try:
        tracer.trace(np.array([item['pos']]), np.array([item['frame'][:, 2]]),
                     initial_states=[initial], on_decision=observe)
    finally:
        tracer.close()
    expected = observations[1]
    np.testing.assert_allclose(result['observed_path'], expected['observed_path'], atol=1e-12)
    np.testing.assert_allclose(result['frame'], expected['frame'], atol=1e-12)
    np.testing.assert_allclose(result['hist_local'], (expected['hist']-expected['pos']) @ expected['frame'], atol=1e-12)
    np.testing.assert_array_equal(result['hmask'], expected['hmask'])
    assert result['seed_age'] == pytest.approx(expected['seed_age'])
    assert result['heading_start'] == expected['heading_start']
    # Original labels lie on the annotation, not the displaced prediction.
    world_targets = result['fut_local'] @ result['frame'].T+result['pos']
    assert np.max(np.abs(world_targets[:, 1]-100.)) < 1e-9
    assert result['fiber_ref'][2] == reverse
    assert result['fiber_ref'][1] > item['fiber_ref'][1]


def test_chain_resets_at_limit_and_preserves_seed(setup):
    live, ds, vol, rng, item = setup
    seed = item['seed_pos'].copy()
    for depth in range(1, 5):
        state = proposal(live, item)
        item = live.advance(state, ds, vol, rng)
        assert item is not None and item['live_depth'] == depth
        np.testing.assert_array_equal(item['seed_pos'], seed)
    assert live.metadata(item) is None


def test_low_confidence_and_recovery_limit_do_not_create_states(setup):
    live, ds, vol, rng, item = setup
    wrapper = LiveContinuation.__new__(LiveContinuation)
    wrapper.sources = [live]
    for confidence, first in [(.1, 1.), (.9, 7.)]:
        points = torch.zeros(1, 16, 3)
        points[..., 2] = torch.arange(1, 17)
        points[0, 0, 2] = first
        wrapper.feedback(dict(_live_states=[live.metadata(item)]),
            dict(points=points, confidence=torch.full((1, 16), confidence)), 10)
        assert live.seeds.empty() and live.chains.empty()


def test_wrong_commit_gets_failure_labels_and_ends_chain(setup):
    live, ds, vol, rng, item = setup
    state = proposal(live, item)
    state['points'][:, 0] = np.arange(1, 17)*.6
    result = live.advance(state, ds, vol, rng)
    assert result['offtrack'] and result['live_failure']
    assert not result['live_correct_continuation']
    assert live.metadata(result) is None
    assert not result['plane_mask'].any()


def test_bank_switch_is_failure_even_within_geometric_departure_radius(setup):
    live, ds, vol, rng, item = setup
    state = proposal(live, item)
    live.detector = SimpleNamespace(first_contact=lambda *a: dict(
        distance=.5, pos=item['pos'], bank_path='test', bank_run='test'))
    result = live.advance(state, ds, vol, rng)
    assert result['offtrack'] and result['failure_kind'] == 1
    assert live.metadata(result) is None


def test_holdout_rejected_before_ct_read(setup, monkeypatch):
    live, ds, vol, rng, item = setup
    state = proposal(live, item)
    ds.exclude = ZBand(210., 230.)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.live_continuation.ct_frame',
                        lambda *a, **kw: pytest.fail('Held-out CT must not be read'))
    assert live.advance(state, ds, vol, rng) is None


@pytest.mark.parametrize('annotated', [True, False])
def test_annotation_end_stop_vs_unannotated_boundary(setup, annotated):
    live, ds, vol, rng, item = setup
    fiber = ds.fibers[0]
    ds.fibers[0] = replace(fiber, endpoint_stop=(False, annotated))
    item = make_sample(ds.fibers[0], fiber.length-4, False, ds.cfg, rng, perturb=False)
    item.update(fiber_ref=(0, fiber.length-4, False), source=0, gt_unperturbed=True)
    orient_item(item, vol)
    result = live.advance(proposal(live, item), ds, vol, rng)
    if annotated:
        assert result['offtrack'] and result['failure_kind'] == 3
    else:
        assert result is None


def test_live_slots_never_read_correct_replay_and_bootstrap_without_cache(setup):
    live, ds, vol, rng, item = setup
    # No saved replay at all: correct allocation still reserves live slots.
    ds.replay_continuation_fraction = 1.
    draw = ds.draw_replay(rng, force=True)
    fallback = ds.replay_item(draw, rng)
    assert fallback['live_requested'] and fallback['live_fallback']
    assert fallback['gt_unperturbed']
    light = ds.correct_continuation_item(rng)
    assert light['live_light_slot'] and light['live_requested']
    live.step.value = 200
    stale = proposal(live, item, step=100)
    live.publish(stale)
    assert live.resolve(fallback, ds, vol, rng) is fallback
    state = proposal(live, item, step=199)
    live.publish(state)
    result = live.resolve(light, ds, vol, rng)
    assert result['live_continuation'] and result['live_light_slot']
    assert not result.get('replay_correct_continuation', False)
    assert not result.get('live_fallback', False)


def test_metadata_excludes_old_replay_and_synthetic_rows(setup):
    live, ds, vol, rng, item = setup
    for source in (2, 3, 4, 5):
        other = dict(item, source=source, gt_unperturbed=False)
        assert live.metadata(other) is None
    cpu = dict(x={'a': torch.ones(1)}, _live_states=[live.metadata(item)])
    assert set(move_batch(cpu, 'cpu')) == {'x'}


def _publish_from_child(source, state):
    source.publish(state)
    source.seeds.close()
    source.seeds.join_thread()


def test_feedback_queue_crosses_spawn_boundary():
    import multiprocessing as mp
    source = LiveContinuationSource(capacity=2)
    state = dict(depth=0, observed_path=np.array([[1., 2., 3.]]), source_step=123)
    process = mp.get_context().Process(target=_publish_from_child, args=(source, state))
    try:
        process.start()
        received = source.seeds.get(timeout=30)
        process.join(timeout=30)
        assert process.exitcode == 0
        np.testing.assert_array_equal(received['observed_path'], state['observed_path'])
        assert received['source_step'] == 123
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        source.close()


def test_loader_keeps_feedback_geometry_on_cpu_as_numpy(setup):
    from vesuvius.neural_tracing.fiber_follow.regression.live_continuation import preserve_live_metadata
    live, ds, vol, rng, item = setup
    cpu = dict(hist=torch.ones(1, 3, 3), _live_states=[live.metadata(item)])
    loader = torch.utils.data.DataLoader([cpu], batch_size=None, collate_fn=preserve_live_metadata)
    batch = next(iter(loader))
    assert isinstance(batch['_live_states'][0]['observed_path'], np.ndarray)
    assert batch['hist'].device.type == 'cpu'


def test_live_feedback_is_consumed_by_loader_next_batch(setup, monkeypatch):
    live, ds, vol, rng, item = setup
    import vesuvius.neural_tracing.fiber_follow.shared.data as module
    monkeypatch.setattr(module, 'FiberVolume', lambda *a, **kw: vol)
    class Builder:
        def __call__(self, items, volume):
            for row in items:
                orient_item(row, volume)
            return dict(collate_targets(items), hist=torch.tensor(np.stack([i['hist_local'] for i in items])),
                        hmask=torch.tensor(np.stack([i['hmask'] for i in items])))
    ds.batch_builder = Builder()
    ds.clean_fraction = 0.
    ds.replay_continuation_fraction = 1.
    # Force precisely two continuation requests without independent source noise.
    monkeypatch.setattr(ds, 'clean_requests', lambda rng: dict(decision=0, bank_following=0, memory_switch=0, recent=2))
    live.step.value = 100
    stream = iter(ds)
    first = next(stream)
    assert first['live_fallback'].all() and not first['live_continuation'].any()
    wrapper = LiveContinuation.__new__(LiveContinuation)
    wrapper.sources = [live]
    points = torch.zeros(2, 16, 3)
    points[..., 2] = torch.arange(1, 17)
    wrapper.feedback(first, dict(points=points, confidence=torch.ones(2, 16)), 100)
    second = next(stream)
    assert second['live_continuation'].all() and not second['live_fallback'].any()
    assert second['live_depth'].tolist() == [1, 1]
    assert not second['replay_correct_continuation'].any()


def test_mixed_source_feedback_never_crosses_volumes(setup):
    live, ds, vol, rng, item = setup
    second = LiveContinuationSource(n_commit=16)
    wrapper = LiveContinuation.__new__(LiveContinuation)
    wrapper.sources = [live, second]
    points = torch.zeros(1, 16, 3)
    points[..., 2] = torch.arange(1, 17)
    wrapper.feedback(dict(_live_states=[live.metadata(item)], dataset_id=torch.ones(1, dtype=torch.long)),
        dict(points=points, confidence=torch.ones(1, 16)), 321)
    assert live.seeds.empty()
    assert second.seeds.get_nowait()['source_step'] == 321


def test_failure_replay_allocation_remains_replay(setup, monkeypatch):
    live, ds, vol, rng, item = setup
    ds.replay_continuation_fraction = 0.
    marker = object()
    ds.recent_pools[5] = {0: marker}
    monkeypatch.setattr(ds, 'draw_pool', lambda pool, source, band, rng: (source, band, pool[0], 0))
    draw = ds.draw_replay(rng, force=True)
    assert draw[1] == 5 and draw[2] is marker


def test_live_depth_and_age_aggregate_in_training_log():
    from vesuvius.neural_tracing.fiber_follow.shared.training_log import DirectTrainingInterval
    log = DirectTrainingInterval()
    log.add(dict(observed_states=10, supervised_states=10, live_continuation_fraction=.3,
                 live_correct_continuation_fraction=.2, live_failure_fraction=.1,
                 live_depth_sum=9., live_policy_age_sum=30.))
    log.add(dict(observed_states=10, supervised_states=10, live_continuation_fraction=.1,
                 live_correct_continuation_fraction=.1, live_depth_sum=4., live_policy_age_sum=20.))
    row = log.summary()
    assert row['live_continuation_fraction'] == pytest.approx(.2)
    assert row['live_correct_continuation_fraction'] == pytest.approx(.15)
    assert row['live_depth_sum'] == 13 and row['live_policy_age_sum'] == 50


def test_invalid_ct_context_falls_back_but_io_errors_propagate(setup, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.shared.heading import SeedHeadingError
    live, ds, vol, rng, item = setup
    live.step.value = 100
    fallback = live.placeholder(ds, rng)
    state = proposal(live, item)
    live.publish(state)
    def invalid(*args):
        raise SeedHeadingError('Invalid CT orientation')
    monkeypatch.setattr(live, 'advance', invalid)
    assert live.resolve(fallback, ds, vol, rng) is fallback
    live.publish(state)
    def io_failure(*args):
        raise OSError('CT store unavailable')
    monkeypatch.setattr(live, 'advance', io_failure)
    with pytest.raises(OSError, match='unavailable'):
        live.resolve(fallback, ds, vol, rng)
