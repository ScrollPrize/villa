"""Operating policy: proposal selection, commit gates and the gate plane."""
import torch

from model_fixtures import coordinate_batch, coordinate_config, proposal_output
from vesuvius.neural_tracing.fiber_follow.evaluation.diagnostics import decision_rows
from vesuvius.neural_tracing.fiber_follow.models.model import build_model, select_refinement
from vesuvius.neural_tracing.fiber_follow.tracing.policy import commit_count, commit_prefix, selection_window


def proposal(first=(0., 0., 1.)):
    points = torch.zeros(1, 16, 3)
    points[..., 2] = torch.arange(1., 17.)
    points[:, 0] = torch.tensor(first)
    return points


def test_full_gate_commits_n_points_only_when_the_last_plane_passes():
    falling = torch.linspace(.9, .3, 16)[None]  # passes 0.4 for the first planes only
    assert commit_count(proposal(), falling, .4, 8, 6., 'full')[0].item() == 0
    assert commit_count(proposal(), falling, .4, 8, 6., 'prefix')[0].item() == 8  # prefix commits the confident part
    assert commit_count(proposal(), falling, .25, 8, 6., 'full')[0].item() == 8
    # The recovery limit still blocks a far first connection.
    counts, allowed = commit_count(proposal((9., 0., 1.)), torch.ones(1, 16), .4, 8, 6., 'full')
    assert counts.item() == 0 and not allowed.item()
    torch.testing.assert_close(commit_count(proposal(), falling, .5, 16, 6., 'prefix')[0],
                               commit_prefix(proposal(), falling, .5, 16, 6.)[0].to(torch.int64), check_dtype=False)
    assert selection_window(8, 16, 'full') == 16 and selection_window(8, 16, 'prefix') == 8


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


def policy_output():
    curves = torch.zeros(1, 3, 4, 3)
    curves[..., 2] = torch.arange(1, 5)
    curves[0, :, :, 0] = torch.tensor([0., 2., 4.])[:, None]
    confidence = torch.tensor([[[.99, .8, .7, .1], [.95, .9, .4, .3], [.9, .85, .75, .2]]])
    previous = torch.cat((torch.ones_like(confidence[..., :1]), confidence[..., :-1]), -1)
    return proposal_output(curves, torch.logit(1-confidence/previous))


def test_selection_uses_actual_threshold_horizon_and_matching_scores():
    output, c = policy_output(), coordinate_config()
    for threshold, window, expected in ((.5, 4, 2), (.5, 1, 0), (.79, 4, 1), (1., 4, 0)):
        selected = select_refinement(output, c, threshold, window)
        assert selected['selected_refinement'].item() == expected
        for name in ('points', 'hazard_logits', 'confidence_logits', 'confidence'):
            torch.testing.assert_close(selected[name], output['refinement_'+name][:, expected])
    stopped = select_refinement(output, c, 1., 4)
    assert commit_prefix(stopped['points'], stopped['confidence'], 1., 4)[0].item() == 0
    # Diagnostics label the proposal the policy chose at each threshold.
    rows = decision_rows(output, coordinate_batch(c, 1), c, n_commit=1, thresholds=(.5, 1.))
    assert rows[0]['gate_0.5']['accepted_wrong'] == 0
    assert rows[0]['gate_1.0']['false_stops'] == 1
    # An invalid connection cannot win even with the longest confident prefix.
    output['refinement_points'][:, 2, :, 0] = 20.
    assert select_refinement(output, c)['selected_refinement'].item() == 0
    # Equal prefix length and confidence retain the earlier proposal.
    output = policy_output()
    output['refinement_confidence'][:, 2] = output['refinement_confidence'][:, 0]
    assert select_refinement(output, c)['selected_refinement'].item() == 0
