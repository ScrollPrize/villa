"""gate_plane: predict more planes than the gate reads; acceptance, retries and commits follow the gate plane."""
import pytest
import torch

from model_fixtures import coordinate_batch, coordinate_config
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.tracing.policy import commit_count, gate_horizon


def line(n=16):
    points = torch.zeros(1, n, 3)
    points[..., 2] = torch.arange(1., n+1.)
    return points


def test_commit_count_decides_on_the_gate_plane():
    confidence = torch.ones(1, 16)
    confidence[0, 10:] = .2  # confident through plane 10 only
    assert commit_count(line(), confidence, .4, 8, 6., 'full')[0].item() == 0  # the last plane decides by default
    assert commit_count(line(), confidence, .4, 8, 6., 'full', horizon=10)[0].item() == 8
    assert commit_count(line(), confidence, .4, 8, 6., 'full', horizon=11)[0].item() == 0


def test_gate_plane_and_query_scale_config():
    assert coordinate_config().gate_horizon == gate_horizon(coordinate_config()) == 4  # default: the last plane
    assert coordinate_config(gate_plane=2).gate_horizon == 2
    for bad in (dict(gate_plane=5), dict(gate_plane=0), dict(query_scale=0.)):
        with pytest.raises(ValueError):
            coordinate_config(**bad)
    assert build_model(coordinate_config()).query_scale == 4.
    assert build_model(coordinate_config(query_scale=16.)).query_scale == 16.


def test_retries_end_at_the_gate_plane_not_the_last_plane():
    torch.manual_seed(3)
    last = build_model(coordinate_config(recurrent_refinement_steps=2)).eval()
    gated = build_model(coordinate_config(recurrent_refinement_steps=2, gate_plane=2)).eval()
    gated.load_state_dict(last.state_dict())  # same weights: only the acceptance plane differs
    batch = coordinate_batch(last.cfg, 1)
    args = (batch['x'], batch['hist'], batch['hmask'])
    with torch.no_grad():
        confidence = last(*args)['refinement_confidence'][0, 0]
        assert confidence[1] > confidence[3]
        threshold = float(confidence[1]+confidence[3])/2  # plane 2 passes, plane 4 fails
        assert last(*args, confidence_threshold=threshold)['refinement_mask'][0, 1:].any()  # retried
        assert not gated(*args, confidence_threshold=threshold)['refinement_mask'][0, 1:].any()  # accepted at plane 2
        t = torch.tensor(threshold)
        assert last.training_forward(*args, t)['refinement_mask'][0, 1:].any()
        assert not gated.training_forward(*args, t)['refinement_mask'][0, 1:].any()
