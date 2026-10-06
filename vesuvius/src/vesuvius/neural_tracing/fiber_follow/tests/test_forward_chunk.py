"""Chunked tracer forwards (TraceParams.forward_chunk) when chunks stop refining after different attempt counts."""
import contextlib
from types import SimpleNamespace

import torch

from model_fixtures import coordinate_batch, coordinate_config
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.tracing.trace import ModelTracer


def test_chunks_with_different_refinement_attempts_match_one_forward():
    torch.manual_seed(115)
    model = build_model(coordinate_config(recurrent_refinement_steps=2)).eval()
    batch = coordinate_batch(model.cfg, 2)
    x, hist, hmask = batch['x'], batch['hist'], batch['hmask']
    with torch.no_grad():
        confidence = model(x, hist, hmask)['refinement_confidence'][:, 0, -1]
        threshold = float(confidence.mean())  # one row accepts its first proposal, the other retries
        widths = {model({k: v[i:i+1] for k, v in x.items()}, hist[i:i+1], hmask[i:i+1],
                        confidence_threshold=threshold)['refinement_mask'].shape[1] for i in range(2)}
        assert len(widths) == 2  # the per-row chunks really return different attempt counts
        tracer = lambda chunk: SimpleNamespace(model=model, p=SimpleNamespace(forward_chunk=chunk),
                                               autocast=contextlib.nullcontext)
        sampling = dict(confidence_threshold=threshold)
        whole = ModelTracer.forward(tracer(None), x, hist, hmask, sampling)
        chunked = ModelTracer.forward(tracer(1), x, hist, hmask, sampling)
    for key, value in chunked.items():
        torch.testing.assert_close(value, whole[key], rtol=1e-5, atol=1e-6, msg=key)
