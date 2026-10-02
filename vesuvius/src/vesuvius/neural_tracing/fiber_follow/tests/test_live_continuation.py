"""Live states use the operating policy's commits, the shared state contract and bounded feedback."""
import copy
from dataclasses import replace
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.train.live_continuation import LiveContinuation, LiveContinuationSource
from vesuvius.neural_tracing.fiber_follow.train.train import move_batch
from vesuvius.neural_tracing.fiber_follow.data.data import SOURCE, STARTUP_CATEGORIES, TASK, FollowDataset, SampleConfig, TaskBudget, TracedFiber, ZBand, collate_targets, make_sample
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength
from vesuvius.neural_tracing.fiber_follow.tracing.heading import orient_item
from vesuvius.neural_tracing.fiber_follow.tracing.policy import OperatingPolicy
from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING, REASON, RECOVERABLE, REPLAY_CLASS, TERMINAL
from vesuvius.neural_tracing.fiber_follow.tracing.trace import ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec

POLICY = OperatingPolicy(n_commit=16)


class Values:
    """In-process stand-in for a shared multiprocessing value."""
    def __init__(self, *args):
        self.value = 0

    def get_lock(self):
        from contextlib import nullcontext
        return nullcontext()


@pytest.fixture
def setup(monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.train.live_continuation as module
    # Deterministic in-process queues for numeric tests; real IPC is tested below.
    monkeypatch.setattr(module.mp, 'get_context', lambda: SimpleNamespace(Queue=Queue, Value=Values))
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.ct_tensor',
                        lambda vol, pos: np.diag([3., 1., .1]))
    z = np.arange(500.)
    p = np.c_[100.+.4*np.sin(z/20), np.full(len(z), 100.), z+100.]
    fiber = TracedFiber('live', p, arclength(p), 'V', endpoint_stop=(True, True))
    # No simulated trace error: label-boundary windows below do not move with noise refits.
    cfg = SampleConfig(crop=CropSpec(depth=24, width=16, behind=8), n_history=32, n_future=16,
                       startup_shares=(0., 0., 0., 1.), excursion_probability=0., trace_noise_sigma=(0., 0.),
                       trace_noise_smoothing=0.)
    ds = FollowDataset([fiber], FiberVolumeSpec('unused'), cfg, None, chunk=2)
    live = LiveContinuationSource(policy=POLICY, steps=(4, 4), step=ds.step)
    ds.live_continuation = live
    vol = SimpleNamespace(shape=(1000,)*3)
    rng = np.random.default_rng(5)
    item = start(ds, fiber, 100., False, rng)
    orient_item(item, vol)
    return live, ds, vol, rng, item


def start(ds, fiber, t, reverse, rng):
    item = make_sample(fiber, t, reverse, ds.cfg, rng)
    item.update(fiber_ref=(0, t, reverse), source=SOURCE['fresh'], live_start='seed',
                live_loop_start=len(item['observed_path'])-1)
    return item


def proposal(live, item, step=100, confidence=None, lateral=.2, drift=0.):
    meta = live.metadata(item)
    assert meta is not None
    points = np.c_[lateral+drift*np.arange(1, 17), np.zeros(16), np.arange(1, 17)].astype(np.float32)
    wrapper = LiveContinuation.__new__(LiveContinuation)
    wrapper.sources, wrapper.policy = [live], POLICY
    wrapper.feedback(dict(_live_states=[meta]),
        dict(points=torch.tensor(points[None], requires_grad=True),
             confidence=torch.tensor([confidence or [.9]*16])), step)
    queue = live.chains if meta['depth'] else live.seeds
    return queue.get_nowait()


def test_live_state_matches_next_inference_observation(setup):
    live, ds, vol, rng, item = setup
    reverse = True
    item = start(ds, ds.fibers[0], 100., True, rng)
    orient_item(item, vol)
    state = proposal(live, item, confidence=[.9]*7+[.1]*9)
    assert state['commit'] == 7
    result = live.advance(state, ds, vol, rng)
    assert result['live_continuation'] and result['live_depth'] == 1
    assert result['supervision'] == FOLLOWING and not result['live_terminal']

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


def test_rejected_decisions_and_recovery_limit_never_advance(setup):
    live, ds, vol, rng, item = setup
    wrapper = LiveContinuation.__new__(LiveContinuation)
    wrapper.sources, wrapper.policy = [live], POLICY
    for confidence, first in [(.1, 1.), (.9, 7.)]:
        points = torch.zeros(1, 16, 3)
        points[..., 2] = torch.arange(1, 17)
        points[0, 0, 2] = first
        wrapper.feedback(dict(_live_states=[live.metadata(item)]),
            dict(points=points, confidence=torch.full((1, 16), confidence)), 10)
        assert live.seeds.empty() and live.chains.empty()


def test_displaced_commit_continues_as_recoverable(setup):
    live, ds, vol, rng, item = setup
    state = proposal(live, item, lateral=0., drift=.28)
    result = live.advance(state, ds, vol, rng)
    assert result['supervision'] == RECOVERABLE and result['geometry_valid']
    assert result['match_distance'] > 3 and not result['live_terminal']
    # The chain keeps going from a safe recoverable state.
    assert live.metadata(result) is not None
    # A connection beyond the commit limit is delivered with proposal labels only and ends the chain.
    state = proposal(live, item, lateral=0., drift=.39)
    uncertain = live.advance(state, ds, vol, rng)
    assert uncertain['supervision_reason'] == REASON['unsupported_connection']
    assert not uncertain['geometry_valid'] and uncertain['confidence_valid'] and live.metadata(uncertain) is None


def test_terminal_live_outcomes_end_the_chain(setup):
    live, ds, vol, rng, item = setup
    def check(result, reason):
        assert result['terminal'] and result['live_terminal'] and result['supervision'] == TERMINAL
        assert result['supervision_reason'] == REASON[reason] and live.metadata(result) is None
        assert not result['geometry_valid'] and result['confidence_valid']
        return result
    state = proposal(live, item)
    state['points'][:, 0] = np.arange(1, 17)*.6
    check(live.advance(state, ds, vol, rng), 'unreachable')
    # A certified bank contact is terminal even within the geometric departure radius.
    state = proposal(live, item)
    live.detector = SimpleNamespace(first_contact=lambda *a: dict(
        distance=.5, pos=item['pos'], bank_path='test', bank_run='test'))
    assert check(live.advance(state, ds, vol, rng), 'switch')['match_distance'] <= 3
    live.detector = None
    # Past the annotation end: a physical endpoint is terminal, an unannotated end is censored.
    fiber = ds.fibers[0]
    for annotated in (True, False):
        ds.fibers[0] = replace(fiber, endpoint_stop=(False, annotated))
        end = start(ds, ds.fibers[0], fiber.length-4, False, rng)
        orient_item(end, vol)
        result = live.advance(proposal(live, end), ds, vol, rng)
        if annotated:
            check(result, 'endpoint')
        else:
            assert result is None


def test_holdout_rejected_before_ct_read(setup, monkeypatch):
    live, ds, vol, rng, item = setup
    state = proposal(live, item)
    ds.exclude = ZBand(210., 230.)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.train.live_continuation.ct_frame',
                        lambda *a, **kw: pytest.fail('Held-out CT must not be read'))
    assert live.advance(state, ds, vol, rng) is None


def replay_start(ds, item, travelled=40.):
    """A recorded valid pre-excursion row whose prefix is the item's observed path."""
    from replay_fixtures import replay_states
    path = item['observed_path']
    row = dict(t=item['trace_facts']['t'], pos=item['pos'], frame=item['frame'],
               hist=item['hist_local'] @ item['frame'].T+item['pos'], hmask=item['hmask'],
               seed_pos=path[0], seed_tangent=item['seed_tangent'], seed_age=item['seed_age'], seed_valid=True,
               travelled=travelled, seq_start=0, seq_end=len(path), departure_distance=12., bad_run=1,
               bad_run_start=travelled, replay_class=REPLAY_CLASS['pre_excursion'], event_id=0)
    return replay_states(ds.fibers, [row], track=path, n_history=ds.cfg.n_history)


def test_real_prefix_restart_preserves_seed_history_heading_and_events(setup, monkeypatch):
    live, ds, vol, rng, item = setup
    ds._set_replay([replay_start(ds, item)])
    monkeypatch.setattr(ds, 'prepare', lambda value, rng: value)
    class ReplayHalf:  # take the recorded-prefix half of live starts
        def __init__(self, rng): self.rng = rng
        def random(self): return .1
        def __getattr__(self, name): return getattr(self.rng, name)
    restart = ds.live_start(ReplayHalf(rng), [])
    assert restart['live_start'] == 'replay' and restart['source'] == SOURCE['replay']
    meta = live.metadata(restart)
    np.testing.assert_array_equal(meta['observed_path'], item['observed_path'])
    np.testing.assert_array_equal(meta['seed_pos'], item['observed_path'][0])
    assert meta['loop_start'] == 0 and meta['travelled'] == 40.
    assert meta['labeler']['departure_distance'] == 12. and meta['labeler']['bad_run'] == 1
    assert meta['labeler']['t'] == pytest.approx(item['trace_facts']['t'])


def test_live_slots_never_take_replay_task_slots(setup, monkeypatch):
    live, ds, vol, rng, item = setup
    ds._set_replay([replay_start(ds, item)])
    monkeypatch.setattr(ds, 'prepare', lambda value, rng: value)
    # A replay task slot is replay even when live states are queued.
    live.publish(proposal(live, item, step=99))
    replay = ds.task_item(TASK['dagger_pre_excursion'], rng, [])
    assert replay['source'] == SOURCE['replay'] and not replay.get('live_requested')
    assert not live.seeds.empty()
    # A live slot resolves to a chain state, with the requested task preserved.
    placeholder = ds.task_item(TASK['live'], rng, [])
    assert placeholder['live_requested'] and placeholder['task_requested'] == TASK['live']
    live.step.value = 100
    result = live.resolve(placeholder, ds, vol, rng)
    assert result['live_continuation'] and result['task_requested'] == TASK['live']
    # Stale states are discarded and the start itself is delivered.
    live.publish(proposal(live, item, step=10))
    live.step.value = 200
    assert live.resolve(placeholder, ds, vol, rng) is placeholder
    assert live.take_outcomes()['stale'] == 1


def test_metadata_excludes_replay_slots_synthetic_and_terminal_rows(setup):
    live, ds, vol, rng, item = setup
    for source in (SOURCE['replay'], SOURCE['synthetic']):
        other = {k: v for k, v in item.items() if k != 'live_start'}
        assert live.metadata(dict(other, source=source)) is None
    assert live.metadata(dict(item, supervision=TERMINAL)) is None
    cpu = dict(x={'a': torch.ones(1)}, _live_states=[live.metadata(item)])
    assert set(move_batch(cpu, 'cpu')) == {'x'}
    # Loader workers hand feedback geometry over on the CPU as numpy.
    from vesuvius.neural_tracing.fiber_follow.train.live_continuation import preserve_live_metadata
    cpu = dict(hist=torch.ones(1, 3, 3), _live_states=[live.metadata(item)])
    loader = torch.utils.data.DataLoader([cpu], batch_size=None, collate_fn=preserve_live_metadata)
    batch = next(iter(loader))
    assert isinstance(batch['_live_states'][0]['observed_path'], np.ndarray)
    assert batch['hist'].device.type == 'cpu'


def _publish_from_child(source, state):
    source.publish(state)
    source.seeds.close()
    source.seeds.join_thread()


def test_feedback_queue_crosses_spawn_boundary():
    import multiprocessing as mp
    source = LiveContinuationSource(policy=POLICY, capacity=2)
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


def test_live_feedback_is_consumed_by_loader_next_batch(setup, monkeypatch):
    live, ds, vol, rng, item = setup
    import vesuvius.neural_tracing.fiber_follow.data.data as module
    monkeypatch.setattr(module, 'FiberVolume', lambda *a, **kw: vol)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.oriented_seed_heading',
                        lambda vol, pos, family, direction: np.asarray(direction))
    class Builder:
        def __call__(self, items, volume):
            for row in items:
                orient_item(row, volume)
            return dict(collate_targets(items), hist=torch.tensor(np.stack([i['hist_local'] for i in items])),
                        hmask=torch.tensor(np.stack([i['hmask'] for i in items])))
    ds.batch_builder = Builder()
    ds.budget = TaskBudget.parse(['live=1', 'fresh=0', 'dagger_pre_excursion=0', 'dagger_recoverable=0',
                                  'dagger_terminal=0', 'dagger_premature_stop=0', 'dagger_ordinary=0',
                                  'synthetic_terminal=0'])
    live.step.value = 100
    stream = iter(ds)
    first = next(stream)
    assert first['live_requested'].all() and not first['live_continuation'].any()
    assert (first['startup'] == STARTUP_CATEGORIES.index('seed_only')).all()  # no replay yet: seed-only starts
    wrapper = LiveContinuation.__new__(LiveContinuation)
    wrapper.sources, wrapper.policy = [live], POLICY
    points = torch.zeros(2, 16, 3)
    points[..., 2] = torch.arange(1, 17)
    wrapper.feedback(first, dict(points=points, confidence=torch.ones(2, 16)), 100)
    second = next(stream)
    assert second['live_continuation'].all()
    assert second['live_depth'].tolist() == [1, 1]
    assert (second['source'] == SOURCE['live']).all() and (second['task_requested'] == TASK['live']).all()


def test_mixed_source_feedback_never_crosses_volumes(setup):
    live, ds, vol, rng, item = setup
    second = LiveContinuationSource(policy=POLICY)
    wrapper = LiveContinuation.__new__(LiveContinuation)
    wrapper.sources, wrapper.policy = [live, second], POLICY
    points = torch.zeros(1, 16, 3)
    points[..., 2] = torch.arange(1, 17)
    wrapper.feedback(dict(_live_states=[live.metadata(item)], dataset_id=torch.ones(1, dtype=torch.long)),
        dict(points=points, confidence=torch.ones(1, 16)), 321)
    assert live.seeds.empty()
    assert second.seeds.get_nowait()['source_step'] == 321


def test_invalid_ct_context_falls_back_but_io_errors_propagate(setup, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import SeedHeadingError
    live, ds, vol, rng, item = setup
    live.step.value = 100
    fallback = dict(item, live_requested=True, task_requested=TASK['live'])
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


def test_stratified_limits_balance_across_worker_copies():
    source = LiveContinuationSource(policy=OperatingPolicy(), steps=(12, 32))
    worker = copy.copy(source)
    rng = np.random.default_rng(45)
    try:
        limits = [s.draw_limit(rng) for _ in range(300) for s in (source, worker)]
        bands = np.array([(v-12)//7 for v in limits])
        assert np.bincount(bands).tolist() == [200, 200, 200]
        assert min(limits) == 12 and max(limits) == 32
    finally:
        source.close()
