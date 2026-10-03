"""Rollout CPU work in a thread pool changes scheduling only, never values."""
import copy

import numpy as np
import pytest
import torch

from model_fixtures import coordinate_config, array_at
from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionFollower
from vesuvius.neural_tracing.fiber_follow.data.observations import FiberTracer
from vesuvius.neural_tracing.fiber_follow.data import ct_normalization as norm
from vesuvius.neural_tracing.fiber_follow.shared.geometry import frame_from_heading
from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume, FiberVolumeSpec


def ct_volume(root):
    z, y, x = np.indices((80, 80, 80))
    array_at(root/'ct'/'0', (x+2*y+z).clip(0, 255))
    spec = FiberVolumeSpec('', ct_zarr=str(root/'ct'), ct_level=0, ct_grid_scale=4., inputs='ct', load_presence=False)
    norm.prepare_normalization(root/'run', [spec], known=dict(method=norm.ZSCORE_METHOD, volumes={}))
    return FiberVolume(spec, cache_bytes=1 << 20)


@pytest.fixture
def one_intraop_thread():
    # Values do not depend on it; an oversubscribed loaded host makes serial forwards ~30x slower.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_pooled_rollout_matches_serial_including_inference_mode(tmp_path, one_intraop_thread):
    cfg = coordinate_config()
    vol = ct_volume(tmp_path)
    torch.manual_seed(0)
    model = CoordinateRegressionFollower(cfg).eval()
    seeds = np.array([[20., 20., 20.], [20.3, 19.8, 20.1], [19.7, 20.2, 19.9]])
    headings = np.array([[.3, .4, .8660254]]*3)
    results = []
    for pooled in (True, False):
        tracer = FiberTracer(model, vol, cfg.fine, cfg.n_history,
                              TraceParams(n_commit=1, max_len=2., confidence=0.), device='cpu')
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
    # Pool threads honor the caller's inference mode for in-place row writes.
    frame = frame_from_heading(headings[0])
    items = [dict(pos=s, frame=frame, hist_local=np.zeros((cfg.n_history, 3)), hmask=np.zeros(cfg.n_history),
                  seed_valid=True, seed_pos=s-[0., 0., 1.], seed_tangent=np.array([0., 0., 1.]), seed_age=1.,
                  observed_path=np.stack([s-[0., 0., 1.], s])) for s in seeds]
    tracer = FiberTracer(model, vol, cfg.fine, cfg.n_history, TraceParams(n_commit=1), device="cpu")
    try:
        serial = tracer.observations.images(copy.deepcopy(items), vol)
        with torch.inference_mode():
            pooled = tracer.observations.images(copy.deepcopy(items), vol, tracer.pool)
    finally:
        tracer.close()
    for key in serial:
        if key != 'history_load_seconds':
            torch.testing.assert_close(pooled[key], serial[key], rtol=0, atol=0)
