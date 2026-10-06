"""Legacy pre-cleanup flow follower (legacy/flow_v5.py): checkpoint gating and tracing with the current tracer."""
import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.legacy.flow_v5 import LegacyFlowConfig, LegacyFlowFollower, config_from_checkpoint
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec

TINY = dict(fine=dict(depth=32, width=16, behind=8, spacing=.5), hidden=32, heads=2, layers=1, decoder_layers=1,
            scorer_layers=1, encoder_ffn=32, decoder_ffn=64, stem_channels=4, stem_blocks=1, n_future=4, n_history=16,
            flow_steps=2, flow_sigma=((1., 1.),)*4, flow_samples=2, flow_time_conditioning='adaln')


def recorded(**changes):
    """A pre-cleanup model_cfg: the tiny legacy fields plus fields the legacy model ignores."""
    return dict(TINY, model_type='flow_matching', memory='none', stem='stride2', path_planes='future',
                flow_loss='pseudo_huber', flow_draws=64, flow_unknown_planes='own_path') | changes


def test_checkpoint_gating_keeps_the_legacy_fields_and_refuses_unsupported_models():
    cfg = config_from_checkpoint(dict(model_cfg=recorded()))
    assert type(cfg) is LegacyFlowConfig and cfg.fine == CropSpec(depth=32, width=16, behind=8, spacing=.5)
    assert cfg.flow_samples == 2 and not hasattr(cfg, 'flow_loss')  # training-only fields are dropped
    for changes in (dict(memory='decisions'), dict(identity_objective='verify'), dict(model_type='unified_flow'),
                    dict(stem='residual'), dict(path_planes='crop')):
        with pytest.raises(ValueError, match='Not a supported legacy flow checkpoint'):
            config_from_checkpoint(dict(model_cfg=recorded(**changes)))


def test_the_current_tracer_runs_the_legacy_model(tmp_path):
    from test_rollout_threading import ct_volume
    from vesuvius.neural_tracing.fiber_follow.data.observations import FiberTracer
    from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams
    torch.manual_seed(0)
    model = LegacyFlowFollower(config_from_checkpoint(dict(model_cfg=recorded()))).eval()
    forward, calls = model.forward, []
    model.forward = lambda *args, **kwargs: calls.append(kwargs) or forward(*args, **kwargs)
    tracer = FiberTracer(model, ct_volume(tmp_path), model.cfg.fine, model.cfg.n_history,
                         TraceParams(n_commit=2, max_len=4., confidence=0.), device='cpu')
    try:
        paths, reasons = tracer.trace(np.array([[24., 24., 24.]]), np.array([[.3, .4, .8660254]]))
    finally:
        tracer.close()
    assert reasons == ['max_len'] and len(paths[0]) > 1
    # The tracer hands the model its operating threshold and commit window, as for the original model.
    assert calls and all(call.keys() == {'confidence_threshold', 'n_commit'} for call in calls)
