"""Fresh training decisions are built like tracing decisions: the simulated trace is the tracer's state of its path."""
from dataclasses import replace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.data import data as D
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength, interp_at
from vesuvius.neural_tracing.fiber_follow.tracing.heading import trace_heading
from vesuvius.neural_tracing.fiber_follow.tracing.trace import trace_history


def curved_fiber():
    u = np.linspace(0, 100, 401)
    points = np.c_[u, 4*np.sin(u/12), 2*np.cos(u/9)]
    return D.TracedFiber('curved', points, arclength(points), 'H')


def sample_config(**kwargs):
    return replace(D.SampleConfig(crop=CropSpec(depth=12, width=9, behind=2), n_history=32, n_future=4), **kwargs)


def lateral_tangents(p, s, arcs):
    tangent = interp_at(p, s, np.clip(arcs+3., 0., s[-1]))-interp_at(p, s, np.clip(arcs-3., 0., s[-1]))
    return tangent/np.linalg.norm(tangent, axis=1, keepdims=True)


def test_simulated_trace_is_the_tracer_state_of_its_observed_path():
    f, cfg = curved_fiber(), sample_config(startup_shares=(.2, .15, .15, .5), excursion_probability=0.)
    for t, reverse, seed in ((t, r, k) for t, r in ((0., False), (2.5, True), (60., False)) for k in range(12)):
        p, s = D.traversal_curve(f, reverse)
        item = D.make_sample(f, t, reverse, cfg, np.random.default_rng(seed))
        path = item['observed_path']
        np.testing.assert_array_equal(item['pos'], path[-1])
        np.testing.assert_array_equal(item['seed_pos'], path[0])
        assert item['seed_valid'] and item['seed_age'] == pytest.approx(arclength(path)[-1])
        # The trace starts on the annotation; its commit error is lateral to GT, plus the unfollowed wiggle.
        arcs = t-np.arange(len(path)-1, -1, -1)*cfg.history_step
        residual = path-interp_at(p, s, arcs)
        np.testing.assert_allclose(residual[0], 0., atol=1e-12)
        commit_error = residual-D.annotation_smoothing(arcs, p, s, cfg.trace_noise_smoothing)
        np.testing.assert_allclose((commit_error*lateral_tangents(p, s, arcs)).sum(-1), 0., atol=1e-9)
        # Crop heading and history are the tracer's own functions of this path.
        np.testing.assert_allclose(item['frame'][:, 2], trace_heading(path, 0, item['seed_tangent']), atol=1e-12)
        hist, mask = trace_history(list(path), cfg.n_history)
        np.testing.assert_array_equal(item['hmask'], mask)
        valid = mask > 0
        np.testing.assert_allclose(item['hist_local'][valid] @ item['frame'].T+item['pos'], hist[valid], atol=1e-9)
        assert ('_pending_seed_heading' in item) == (arclength(path)[-1] < 12-1e-8)
        assert item['seed_heading_family'] == 'H'
        # Labels remain the GT continuation, now from the offset head.
        future = interp_at(p, s, np.clip(t+cfg.future_s, 0, s[-1]))
        np.testing.assert_allclose(item['fut_local'] @ item['frame'].T+item['pos'], future, atol=1e-9)
