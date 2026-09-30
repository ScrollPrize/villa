"""Individual points and interval logging must not inherit prefix semantics."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_regression import batch, config
from vesuvius.neural_tracing.fiber_follow.regression.supervision import point_correctness, loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectFollower
from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update
from vesuvius.neural_tracing.fiber_follow.shared.training_log import DirectTrainingInterval, format_training_log
from vesuvius.neural_tracing.fiber_follow.shared.diag import plot_curves
from vesuvius.neural_tracing.fiber_follow.shared.runloop import RunLog


def test_individual_points_recover_after_wrong_point_and_ignore_confidence():
    cfg = replace(config(), fine=replace(config().fine, depth=40), n_future=16, recurrent_refinement_steps=0)
    data = batch(cfg, 1)
    data['dense_ab'].zero_()
    points = torch.zeros(1, 16, 3)
    points[..., 2] = torch.arange(1, 17)
    points[0, 2, 0] = 2.
    output = dict(points=points, confidence_logits=torch.full((1,16), -10.), hazard_logits=torch.zeros(1,16))
    terms = loss_terms(output, data, cfg, n_commit=16)
    assert terms['point_correct_count'] == 15
    assert terms['point_wrong_count'] == 1
    assert terms['point_unknown_count'] == 0
    assert terms['correct_count'] < terms['point_correct_count']  # cumulative labels still train confidence
    output['confidence_logits'].fill_(10.)
    other = loss_terms(output, data, cfg, n_commit=8)
    for key in ('point_correct_count', 'point_wrong_count', 'point_unknown_count'):
        assert terms[key] == other[key]  # all predicted points, independent of commit policy


def test_point_masks_neighbors_departures_endpoints_and_nonfinite():
    cfg = config()
    data = batch(cfg, 7)
    data['dense_ab'].zero_()
    points = torch.zeros(7, 4, 3)
    points[..., 2] = torch.arange(1,5)
    data['plane_ab'] = torch.zeros(7,4,2)
    data['plane_mask'] = torch.ones(7,4)
    data['plane_mask'][0,1] = 0
    data['plane_ab'][0,1] = float('nan')
    data['identity_observable'] = torch.tensor([True,False,True,True,True,True,True])
    data['offtrack'][2] = 1
    data['plane_mask'][3,2:] = 0
    data['endpoint_known'][3] = 1
    data['end_local'][3,2] = 2.
    points[4,1,0] = float('nan')
    data['plane_ab'][5,1,0] = 100.  # censored point doesn't censor subsequent points
    foreign = torch.zeros(7,13,dtype=torch.bool)
    foreign[6,4] = True  # wrong original-fiber identity, even at zero geometric error
    result = point_correctness(points,data,cfg,1.5,foreign)
    assert {k:int(v) for k,v in result.items()} == dict(
        point_correct_count=14, point_wrong_count=8, point_unknown_count=6)


def interval_row(crops, right, wrong, loss):
    return dict(observed_states=crops, point_correct_count=right, point_wrong_count=wrong,
                point_unknown_count=0, geometry=loss, loss=loss, confidence_loss=.2,
                geometry_count=right+wrong, error_sum=right+wrong,
                replay_endpoints=1, replay_encoder_crops=4,
                memory_grad_norm=3., memory_grad_clip_scale=.5)


def test_interval_pools_counts_weights_crop_means_and_preserves_json(tmp_path,capsys):
    interval = DirectTrainingInterval()
    interval.add(interval_row(1,16,0,2.))
    interval.add(interval_row(3,0,48,4.))
    summary = interval.summary()
    assert 'fixed_fraction' not in summary
    assert summary['loss'] == 3.5
    assert summary['crops'] == 4 and summary['updates'] == 2
    assert summary['replay_endpoints'] == 2 and summary['replay_encoder_crops'] == 8
    assert summary['memory_clipped_updates'] == 2
    row = dict(step=10050,geometry=4.,loss=4.,lr=.001,interval=summary,n_future=16,tolerance=1.5,
               interval_update_seconds=1.,interval_data_seconds=.1,interval_samples_per_second=4.)
    path = tmp_path/'log.jsonl'
    log = RunLog(path,formatter=format_training_log)
    log.record(row)
    log.close()
    printed = capsys.readouterr().out
    assert '16 right / 48 wrong | 25.0% correct | 0 unknown' in printed
    assert 'fixed' not in printed
    assert 'last 2 updates / 4 crops' in printed
    assert '500 ms/update' in printed and '50 ms/update' in printed
    assert '16-point correctness' not in printed and len(printed.splitlines()) <= 10
    assert json.loads(path.read_text()) == row
    assert DirectTrainingInterval().summary()['crops'] == 0
    row['interval']['point_correct_count'] = row['interval']['point_wrong_count'] = 0
    assert 'n/a correct' in format_training_log(row)


def test_every_optimizer_update_reports_point_counts_without_detailed_metrics():
    torch.manual_seed(12)
    model = DirectFollower(config())
    data = batch(model.cfg)
    data['source'] = torch.tensor([0, 2])
    ema = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=.001)
    metrics = optimizer_update(model,ema,opt,[data],1,.001,compute_metrics=False)
    assert 'fixed_fraction' not in metrics
    assert metrics['fresh_fraction'] == metrics['recent_fraction'] == .5
    assert 'decisions' not in metrics
    assert sum(metrics[k] for k in ('point_correct_count','point_wrong_count','point_unknown_count')) == 8


def test_curves_show_interval_point_accuracy_and_drop_rolled_back_steps(tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    from PIL import Image
    saved = {}
    original = Figure.savefig
    def capture(self,*args,**kwargs):
        for ax in self.axes:
            for line in ax.lines:
                saved[line.get_label()] = (list(line.get_xdata()), list(line.get_ydata()))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Figure,'savefig',capture)
    def row(step, correct, wrong):
        return dict(step=step,geometry=100.,confidence_loss=100.,error_mean=100.,
                    interval=dict(geometry=.2,confidence_loss=.3,error_mean=.4,
                                  point_correct_count=correct,point_wrong_count=wrong))
    rows = [row(10050,1,0),dict(event='resume_configuration',step=10000),row(10050,1,3),row(10100,0,0)]
    log = tmp_path/'log.jsonl'
    log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    plot_curves(log,tmp_path/'curves.png',loss_key='geometry')
    steps,values = saved['individual points correct (interval)']
    assert steps == [10050,10100] and values[0] == .25 and np.isnan(values[1])
    assert saved['geometry loss'][1] == [.2,.2]
    with Image.open(tmp_path/'curves.png') as im:
        im.verify()
    assert not (tmp_path/'curves.tmp.png').exists()
