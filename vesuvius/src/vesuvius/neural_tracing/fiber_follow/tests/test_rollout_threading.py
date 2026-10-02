"""Rollout CPU work in a thread pool changes scheduling only, never values."""
import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.regression.data import ObservationBuilder, DirectTracer
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectFollower
from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams

from test_direction_inputs import config, item, volume


def shifted_items(cfg, count=4):
    base = item(cfg)
    items = []
    for k in range(count):
        shift = np.array([.3, -.2, .1])*k
        state = copy.deepcopy(base)
        for key in ('pos', 'seed_pos', 'observed_path'):
            state[key] = base[key]+shift
        items.append(state)
    return items


def test_pooled_observations_match_serial_including_inference_mode(tmp_path):
    cfg = config()
    vol = volume(tmp_path)
    items = shifted_items(cfg)
    builder = ObservationBuilder(cfg)
    serial = builder.images(copy.deepcopy(items), vol)
    with ThreadPoolExecutor(3) as pool:
        pooled = builder.images(copy.deepcopy(items), vol, pool)
        # Pool threads must honor the caller's inference mode for in-place row writes.
        with torch.inference_mode():
            inference = builder.images(copy.deepcopy(items), vol, pool)
    for key in serial:
        if key != 'history_load_seconds':
            torch.testing.assert_close(pooled[key], serial[key], rtol=0, atol=0)
            torch.testing.assert_close(inference[key], serial[key], rtol=0, atol=0)


def test_tracer_pool_matches_serial_paths_and_reasons(tmp_path):
    cfg = config()
    vol = volume(tmp_path)
    torch.manual_seed(0)
    model = DirectFollower(cfg).eval()
    seeds = np.array([[20., 20., 20.], [20.3, 19.8, 20.1], [19.7, 20.2, 19.9]])
    headings = np.array([[.3, .4, .8660254]]*3)
    results = []
    for pooled in (True, False):
        # Several one-point steps that stay inside the small synthetic CT context.
        tracer = DirectTracer(model, vol, cfg.fine, cfg.n_history,
                              TraceParams(n_commit=1, max_len=3., confidence=0.), device='cpu')
        if not pooled:
            tracer.pool.shutdown(wait=True)
            tracer.pool = None
        try:
            results.append(tracer.trace(seeds, headings))
        finally:
            if pooled:
                tracer.close()
    (paths_a, reasons_a), (paths_b, reasons_b) = results
    assert reasons_a == reasons_b
    assert all(len(p) > 2 for p in paths_a)
    for a, b in zip(paths_a, paths_b):
        np.testing.assert_array_equal(a, b)
