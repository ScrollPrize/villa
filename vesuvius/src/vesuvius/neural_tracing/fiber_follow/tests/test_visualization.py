"""Atlas diagnostics must preserve inference and reject invented legacy history."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from test_identity import config, batch
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.visualization.capture import attention, capture, validate_attention, array
from vesuvius.neural_tracing.fiber_follow.visualization.interpret import replay_item, analyze, annotation_item
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.shared.reference import observed_path


@pytest.mark.parametrize('version', [5, 6, 7])
def test_replay_rejects_pre_ct_normal_versions(tmp_path,version):
    import json
    (tmp_path/'metadata.json').write_text(json.dumps(dict(version=version)))
    with pytest.raises(ValueError,match='Incompatible replay version'):
        replay_item(tmp_path,0,[],None)


@pytest.mark.parametrize('empty', [False, True])
def test_capture_preserves_refinement_outputs_and_masked_attention(empty):
    torch.manual_seed(13)
    model = build_model(config(encoder='patch4', token_only=True, recurrent_refinement_steps=1)).eval()
    inputs = batch(model.cfg, 1)
    if empty:
        inputs['x']['history_valid'].zero_()
    with torch.inference_mode():
        with capture(model) as captured:
            out = model(inputs['x'], inputs['hist'], inputs['hmask'], confidence_threshold=1.)
        plain = model(inputs['x'], inputs['hist'], inputs['hmask'], confidence_threshold=1.)
    for key in out:
        torch.testing.assert_close(out[key], plain[key], rtol=0, atol=0)
        captured[key] = array(out[key][0])
    validate_attention(captured)
    assert len(out['refinement_points'][0]) == 2
    assert captured['patch'].shape == (*model.cfg.token_shape, model.cfg.hidden)
    assert not model.encoder.patch_projection._forward_hooks
    assert 'attend_memory' not in model.decoder.layers[0].__dict__
    if not empty:
        arrays, report = analyze(model, inputs['x'], inputs['hist'], inputs['hmask'], captured, 1., model.cfg.n_future)
        assert report['metrics']['baseline']['max_path_shift'] == 0
        # Removing generator history cannot change scores of an identical fixed curve.
        np.testing.assert_array_equal(arrays['baseline_candidate_confidence'], arrays['no_generator_history_candidate_confidence'])


def test_capture_restores_methods_after_exception():
    model = build_model(config(encoder='patch4', token_only=True)).eval()
    with pytest.raises(RuntimeError, match='interrupted'):
        with capture(model):
            raise RuntimeError('interrupted')
    assert 'forward_cached' not in model.history_attention.__dict__
    assert 'attend_memory' not in model.decoder.layers[0].__dict__
    assert not model.encoder.patch_projection._forward_hooks


@pytest.mark.parametrize('tail', [-5., -10., -15., -20.])
def test_long_attention_bank_normalizes_without_losing_mask(tail):
    # One dominant key and thousands of small contributions expose the FP32
    # softmax reduction error that stopped the step-17000 atlas extraction.
    q = torch.ones(1, 4, 16, 1)
    k = torch.full((1, 4, 20409, 1), tail)
    k[:, :, 0] = 0
    padding = torch.zeros(20409, dtype=torch.bool)
    padding[-32:] = True
    weights = attention(q, k, ~padding[None, None, None])
    assert weights.dtype == torch.float32
    expected = torch.full((20409,), np.exp(tail), dtype=torch.float64)
    expected[0] = 1
    expected[padding] = 0
    expected /= expected.sum()
    torch.testing.assert_close(weights, expected.float().expand_as(weights), rtol=1e-6, atol=0)
    arrays = dict(generator_attention_0_0=array(weights[0]), context_padding=array(padding))
    validate_attention(arrays)
    # The validation still rejects real normalization errors.
    arrays['generator_attention_0_0'] *= 1.001
    with pytest.raises(AssertionError, match='generator_attention_0_0'):
        validate_attention(arrays)


@pytest.mark.parametrize('reverse', [False, True])
def test_annotation_selection_preserves_complete_prefix(reverse):
    points = np.zeros((81, 3))
    points[:, 2] = np.arange(81)
    fiber = SimpleNamespace(points=points, s=np.arange(81, dtype=float), length=80.,
                            endpoint_stop=(False, False))
    cfg = config(encoder='patch4', token_only=True)
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history,
                          n_future=cfg.n_future, future_step=cfg.future_step)
    item = annotation_item(fiber, 60., reverse, sample)
    prefix = observed_path(item)
    np.testing.assert_array_equal(prefix[0], points[-1] if reverse else points[0])
    np.testing.assert_array_equal(prefix[-1], [0., 0., 20. if reverse else 60.])
    assert len(prefix) == 61
    assert item['seed_age'] == 60.
