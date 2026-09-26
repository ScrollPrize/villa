"""Beam diagnostics preserve training state and plot the paths they score."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from test_diagnostic_images import monitor
from vesuvius.neural_tracing.fiber_follow.beam.diag import rollout_images
from vesuvius.neural_tracing.fiber_follow.beam.native import NativeBeam, BeamSpec
from vesuvius.neural_tracing.fiber_follow.runloop import RunLog
from vesuvius.neural_tracing.fiber_follow.training_log import format_training_log


class Beam:
    def __init__(self):
        self.calls = []

    def trace_open(self, pos, heading, distance, hook=None):
        self.calls.append((pos.copy(), heading.copy(), distance, hook))
        if hook:
            hook.calls += 1
            hook.last = 'diagnostic'
        return np.stack([pos, pos+heading*distance]), 'trace_distance', True


def test_rollouts_are_paired_and_preserve_hook_model_rng(tmp_path, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow import diag as shared
    beam = Beam()
    hook = SimpleNamespace(model=torch.nn.Linear(3, 1), stop_threshold=.8, calls=4, last={'old':True})
    vol = SimpleNamespace(sample_image_nearest=lambda q: np.full(q.shape[:-1], 128, np.uint8))
    fibers, seeds = monitor()
    weights = copy.deepcopy(hook.model.state_dict())
    torch_rng, np_rng = torch.get_rng_state(), np.random.get_state()
    rendered = []
    original = shared.plot_rollouts
    def capture(vol, fibers, seeds, paths, reasons, path, **kwargs):
        rendered.append((paths, kwargs['rows']))
        return original(vol, fibers, seeds, paths, reasons, path, **kwargs)
    monkeypatch.setattr(shared, 'plot_rollouts', capture)
    reports = rollout_images(beam, hook, vol, fibers, seeds, tmp_path, 500, max_len=8., confidence=.5)
    assert len(beam.calls) == 4  # two seeds each, no additional traces for plotting
    assert [c[-1] for c in beam.calls] == [None, None, hook, hook]
    for a,b in zip(beam.calls[:2],beam.calls[2:]):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    assert [r['beam_rollout'] for r in reports] == ['hand','model']
    assert all(r['coverage_mean'] == 1 and r['coverage_max_len'] == 8 for r in reports)
    assert len(rendered) == 2 and all(len(paths) == len(rows) == 2 for paths,rows in rendered)
    for report in reports:
        with Image.open(report['image']) as im:
            assert im.format == 'PNG' and min(im.size) > 100
            im.verify()
    assert hook.model.training and hook.stop_threshold == .8 and hook.calls == 4 and hook.last == {'old':True}
    for name,value in hook.model.state_dict().items(): torch.testing.assert_close(value,weights[name])
    torch.testing.assert_close(torch.get_rng_state(),torch_rng)
    np.testing.assert_array_equal(np.random.get_state()[1],np_rng[1])
    assert rollout_images(beam,hook,vol,[],[],tmp_path,501) == []


def test_rollout_error_restores_model_and_threshold(tmp_path):
    hook = SimpleNamespace(model=torch.nn.Linear(3,1),stop_threshold=None,calls=0,last=None)
    def fail(*a, **kw): raise RuntimeError('trace failed')
    fibers,seeds=monitor()
    with pytest.raises(RuntimeError,match='trace failed'):
        rollout_images(SimpleNamespace(trace_open=fail),hook,None,fibers,seeds,tmp_path,1)
    assert hook.model.training and hook.stop_threshold is None


def test_open_trace_enables_same_scoring_mode_as_span_trace():
    received=[]
    class Adapter(NativeBeam):
        @property
        def grid_to_trace(self): return 1.
        def config(self,hook=None):
            received.append(hook)
            return {'learned_scoring':hook is not None}
        def _ensure(self):
            return SimpleNamespace(trace_extrapolation=lambda *a,**kw:
                SimpleNamespace(points=np.array([[0.,0.,0.],[1.,0.,0.]]),reason='trace_distance',reached_trace_length=True))
    beam=Adapter(BeamSpec('unused'),1.)
    hook=lambda pool:None
    beam.trace_open([0,0,0],[1,0,0],1.,hook=hook)
    beam.trace_open([0,0,0],[1,0,0],1.)
    assert received == [hook,None]


def test_shared_beam_formatting_keeps_json_records(tmp_path,capsys):
    rows=[dict(step=500,loss=1.2,lr=.001,samples_per_second=5.,ranking=.8,onfiber=.4,
               model_top1_onfiber=.9,oracle_onfiber=.95,labeled_candidates=128.,hard_fraction=.5,mined_fraction=.75),
          dict(step=500,span_fibers=2,span_segments=10,span_length_grid=1000.,
               span_success_hand=.8,span_success_model=.9,span_restarts_kvx_hand=2.,span_restarts_kvx_model=1.),
          dict(step=500,beam_rollout='model',threshold=.5,n=4,coverage_mean=.7,length_precision=.9,
               diverged=.25,image='rollout_000500_model.png')]
    log=RunLog(tmp_path/'log.jsonl',formatter=format_training_log)
    try:
        for row in rows: log.record(row)
    finally: log.close()
    assert [json.loads(s) for s in (tmp_path/'log.jsonl').read_text().splitlines()]==rows
    text=capsys.readouterr().out
    for label in ('ranking 0.8000','model top-1 90.0%','model: success 90.0%',
                  'beam rollout model @ 0.50','image: rollout_000500_model.png'):
        assert label in text
    assert 'no held-out fibers' in format_training_log(dict(step=500,span_fibers=0))
