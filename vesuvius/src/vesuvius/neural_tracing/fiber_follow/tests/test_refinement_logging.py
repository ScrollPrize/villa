"""Refinement diagnostics must preserve training and remain aggregatable."""
import copy
import json

import torch

from test_single_path import batch, config
from vesuvius.neural_tracing.fiber_follow.model import FollowNet
from vesuvius.neural_tracing.fiber_follow.runloop import RunLog
from vesuvius.neural_tracing.fiber_follow.supervision import refinement_metrics
from vesuvius.neural_tracing.fiber_follow.train import optimizer_update
from vesuvius.neural_tracing.fiber_follow.training_log import format_training_log


def test_refinement_masks_drift_boundaries_empty_bins_and_nonfinite_predictions():
    cfg = config()
    b = batch(cfg, 7)
    b['gt_history'][:, 0, 0] = torch.tensor([.5, 1.25, 1.75, 2.5, 3.5, float('nan'), 0.])
    b['gt_history_mask'][5] = 0
    b['offtrack'][6] = 1
    b['plane_mask'][0, 1:] = 0
    b['plane_ab'][0, 1:] = float('nan')  # Arbitrary missing-label padding.
    b['plane_ab'][1, 2:] = 100  # Crop-censored despite being annotated.
    b['plane_mask'][2] = 0
    steps = torch.zeros(7, 3, 4, 3)
    steps[..., 0] = torch.tensor([2., 1., 3.])[None, :, None]
    steps[3, 1, 0, 0] = float('nan')
    result = refinement_metrics(steps, b, cfg)
    bands = result['by_drift']
    assert result['departed_count'] == 1 and result['first_n'] == 4
    all_states = bands['all']
    assert all_states['state_count'] == 6
    assert all_states['known_point_count'] == 15
    assert all_states['point_count'] == [15, 14, 15]
    assert all_states['error_sum'] == [30., 14., 45.]
    assert all_states['error_mean'] == [2., 1., 3.]
    assert all_states['nonfinite_point_count'] == [0, 1, 0]
    assert all_states['comparison_count'] == [4, 4]
    assert all_states['improved_count'] == [4, 0]
    assert all_states['worsened_count'] == [0, 4]
    assert bands['1.5-2']['error_mean'] == [None]*3
    assert bands['1.5-2']['comparison_count'] == [0, 0]
    assert bands['>=3.5']['state_count'] == 1
    assert bands['unknown']['state_count'] == 1
    assert bands['2-3.5']['nonfinite_point_count'] == [0, 1, 0]
    json.dumps(result, allow_nan=False)
    # Refinement beyond the commit window cannot affect these measurements.
    long_cfg = config(n_future=8, flow_sigma=((1., 1.),)*8)
    long_steps = torch.zeros(1, 3, 8, 3)
    long_steps[:, :, 4:] = 100
    assert refinement_metrics(long_steps, batch(long_cfg, 1), long_cfg, n_commit=4)['by_drift']['all']['error_mean'] == [0.]*3


def test_logging_preserves_rng_rollout_cost_loss_and_optimizer_update(monkeypatch):
    torch.manual_seed(17)
    cfg = config()
    model = FollowNet(cfg)
    initial = copy.deepcopy(model.state_dict())
    b = batch(cfg)
    outputs = []
    retained = []
    generate = FollowNet.generate_training_curve

    def track(self, *args, **kwargs):
        retained.append(kwargs.get('return_steps', False))
        return generate(self, *args, **kwargs)

    monkeypatch.setattr(FollowNet, 'generate_training_curve', track)
    for logged in (False, True):
        model.load_state_dict(initial)
        ema = copy.deepcopy(model)
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        calls = []
        hook = model.flow.register_forward_hook(lambda *args: calls.append(1))
        torch.manual_seed(91)
        loss, metrics, _ = optimizer_update(model, ema, opt, [b], 2000, .01,
                                            device='cpu', compute_metrics=logged)
        hook.remove()
        outputs.append((loss, metrics, copy.deepcopy(model.state_dict()),
                        copy.deepcopy(ema.state_dict()), torch.get_rng_state(), len(calls)))
    off,on = outputs
    assert retained == [False, True]
    assert off[0] == on[0] and off[5] == on[5] == 2*cfg.flow_steps+2
    assert 'refinement' not in off[1]
    assert len(on[1]['refinement']['by_drift']['all']['error_mean']) == cfg.flow_steps+1
    for index in (2, 3):
        for name in off[index]:
            torch.testing.assert_close(off[index][name], on[index][name], rtol=0, atol=0)
    assert torch.equal(off[4], on[4])


def test_terminal_blocks_preserve_json_and_format_events(tmp_path, capsys):
    cfg = config()
    metrics = refinement_metrics(torch.zeros(2, 3, 4, 3), batch(cfg), cfg)
    row = dict(step=1250, loss=1.23456, lr=.001, samples_per_second=8.765,
               flow=1.1, confidence_loss=.269, confidence_coefficient=.5,
               fresh_fraction=.5, fixed_fraction=.25, recent_fraction=.25,
               replay_samples_seen=5000, commit_correct_count=3, commit_known_count=4, commit_window=8,
               refinement=metrics)
    for threshold in (.5, .85):
        row.update({f'false_stop_count_{threshold}': 1, f'correct_first_count_{threshold}': 4,
                    f'departed_continue_count_{threshold}': 0, f'departed_count_{threshold}': 0})
    event = dict(step=1250, dagger_launched=True)
    rollout = dict(step=1250, threshold=.85, roll_coverage=.5, roll_precision=.95, roll_diverged=.1)
    path = tmp_path/'log.jsonl'
    log = RunLog(path, formatter=format_training_log)
    try:
        for item in (row, event, rollout):
            log.record(item)
    finally:
        log.close()
    assert [json.loads(line) for line in path.read_text().splitlines()] == [row, event, rollout]
    printed = capsys.readouterr().out
    assert 'Step 1,250 | loss 1.2346 | lr 1.00e-03 | 8.77 samples/s' in printed
    assert 'initial' in printed and 'step 2' in printed and 'refinement' in printed
    assert '1/4 (25.0%)' in printed and 'n/a (0 known)' in printed
    assert 'dagger_launched: true' in printed and 'precision 95.0%' in printed
    assert '"refinement"' not in printed
    assert format_training_log(dict(dagger_discarded=True)) == 'Training | dagger_discarded: true'
