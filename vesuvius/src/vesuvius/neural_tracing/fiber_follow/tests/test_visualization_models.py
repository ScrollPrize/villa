"""Interpretation uses component capabilities and preserves either generator's outputs."""
import numpy as np
import pytest
import torch

from model_fixtures import coordinate_config, coordinate_batch
from test_flow_model import config as flow_config, take
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.visualization.capture import capture, array, validate_attention, capabilities
from vesuvius.neural_tracing.fiber_follow.visualization.interpret import analyze


@pytest.mark.parametrize('kind', ['coordinate_regression', 'flow_matching'])
def test_shared_capture_and_interventions_preserve_both_models(kind):
    cfg = coordinate_config() if kind == 'coordinate_regression' else flow_config()
    model = build_model(cfg).eval()
    b = take(coordinate_batch(cfg), slice(0, 1))
    args = b['x'], b['hist'], b['hmask']
    with torch.inference_mode():
        baseline = model(*args)
        with capture(model) as captured:
            measured = model(*args)
        for key in baseline:
            torch.testing.assert_close(measured[key], baseline[key], atol=0, rtol=0)
            captured[key] = array(measured[key][0])
    validate_attention(captured)
    saved, report = analyze(model, *args, captured, .5, cfg.n_future)
    np.testing.assert_array_equal(saved['baseline_points'], captured['points'])
    assert report['metrics']['baseline']['max_path_shift'] == 0.
    assert capabilities(model)['solver'] == (kind == 'flow_matching')


def test_common_report_works_without_optional_components(tmp_path):
    import json
    from vesuvius.neural_tracing.fiber_follow.visualization.render import render
    from vesuvius.neural_tracing.fiber_follow.visualization.validate import validate
    class Generic(torch.nn.Module):
        def forward(self, x, hist, hmask, **kwargs):
            points = hist.new_zeros(1, 4, 3); points[..., 2] = torch.arange(1, 5)
            confidence = hist.new_full((1, 4), .8)
            return dict(points=points, confidence=confidence, refinement_points=points[:, None],
                        refinement_confidence=confidence[:, None])
    model = Generic()
    b = coordinate_batch(coordinate_config(), 1)
    with capture(model) as a:
        out = model(b['x'], b['hist'], b['hmask'])
    assert a == {}
    a.update({key: array(value[0]) for key, value in out.items()})
    a.update(input=array(b['x']['fine'][0]), hist=array(b['hist'][0]), hmask=array(b['hmask'][0]))
    saved, report = analyze(model, b['x'], b['hist'], b['hmask'], a, .5, 4)
    assert report['unavailable'] and report['interventions'] == {}
    meta = dict(model_cfg=coordinate_config().to_dict(), capabilities=capabilities(model), threads=2,
                shapes={key: list(value.shape) for key, value in a.items()})
    np.savez_compressed(tmp_path/'sample_activations.npz', **a)
    np.savez_compressed(tmp_path/'analysis_arrays.npz', **saved)
    (tmp_path/'sample_provenance.json').write_text(json.dumps(meta))
    (tmp_path/'analysis_summary.json').write_text(json.dumps(report))
    render(tmp_path)
    assert validate(tmp_path)['pages'] == 1
