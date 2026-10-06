"""Live continuation states are the observation the tracer would make next."""
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.train.live_continuation import LiveContinuation, LiveContinuationSource
from vesuvius.neural_tracing.fiber_follow.data.data import SOURCE, FollowDataset, SampleConfig, TracedFiber, make_sample
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength
from vesuvius.neural_tracing.fiber_follow.tracing.heading import orient_item
from vesuvius.neural_tracing.fiber_follow.tracing.policy import OperatingPolicy
from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING
from vesuvius.neural_tracing.fiber_follow.tracing.trace import ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec

POLICY = OperatingPolicy(n_commit=16, gate='prefix')  # these tests exercise prefix-length commits


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
    # Deterministic in-process queues.
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

    tracer = Tracer(Policy(), vol, ds.cfg.crop, ds.cfg.n_history, TraceParams(n_commit=16, gate='prefix'), device='cpu')
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
