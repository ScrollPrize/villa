"""Image diagnostics reuse scored traces and leave training state untouched."""
import copy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from test_direct import batch, config
from vesuvius.neural_tracing.fiber_follow.data import TracedFiber
from vesuvius.neural_tracing.fiber_follow.diag import plot_denoising, rollout_diag
from vesuvius.neural_tracing.fiber_follow.direct.model import DirectFollower
from vesuvius.neural_tracing.fiber_follow.direct.train import training_diagnostics
from vesuvius.neural_tracing.fiber_follow.evaluate import evaluate, monitor_coverage
from vesuvius.neural_tracing.fiber_follow.experiment import rollout_summary
from vesuvius.neural_tracing.fiber_follow.runloop import RunLog


class Tracer:
    def __init__(self):
        self.p = SimpleNamespace(confidence=.7, max_len=8.)
        self.vol = SimpleNamespace(sample_image_nearest=lambda q: np.full(q.shape[:-1], 128, np.uint8))
        self.calls = 0

    def trace(self, positions, headings):
        self.calls += 1
        return [np.stack((p, p+h*self.p.max_len)) for p, h in zip(positions, headings)], ['max_len']*len(positions)


def monitor():
    arc = np.arange(100, dtype=float)
    fiber = TracedFiber('line', np.c_[arc*0, arc*0, arc], arc, '')
    seeds = [dict(fiber=0, t=t, sign=1., pos=[0., 0., t], heading=[0., 0., 1.]) for t in (10., 90.)]
    return [fiber], seeds


@pytest.mark.parametrize('correction', [False, True])
def test_direct_images_preserve_scores_rng_and_parameters(tmp_path, monkeypatch, correction):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    cfg = config()
    cfg.correction = correction
    model = DirectFollower(cfg)
    data = batch(cfg)
    data.update(plane_ab=torch.ones(2, cfg.n_future, 2), plane_mask=torch.ones(2, cfg.n_future),
                gt_history=data['hist'].clone(), gt_history_mask=data['hmask'].clone())
    # Unknown futures and missing history should also render safely.
    data['plane_mask'][1] = 0
    data['plane_ab'][1] = float('nan')
    data['hmask'][1] = 0
    for x in data['x'].values():
        x[:, 0] = .2
        x[:, 1] = 1.  # Presence must not be mistaken for rendered history.
    rgb = []
    labels = {}
    original_imshow, original_save = Axes.imshow, Figure.savefig

    def capture_image(self, image, *args, **kwargs):
        if np.ndim(image) == 3:
            rgb.append(np.asarray(image).copy())
        return original_imshow(self, image, *args, **kwargs)

    def capture_labels(self, path, *args, **kwargs):
        labels[str(path)] = [line.get_label() for ax in self.axes for line in ax.lines]
        return original_save(self, path, *args, **kwargs)

    monkeypatch.setattr(Axes, 'imshow', capture_image)
    monkeypatch.setattr(Figure, 'savefig', capture_labels)
    fibers, seeds = monitor()
    reference, _ = evaluate(Tracer(), fibers, seeds, batch=1, coverage_max_len=8.)
    expected = rollout_summary(reference)
    tracer = Tracer()
    weights = copy.deepcopy(model.state_dict())
    rng, numpy_rng = torch.get_rng_state(), np.random.get_state()
    log = RunLog(tmp_path/'log.jsonl')
    try:
        log.record(dict(step=1000, geometry=.3, confidence_loss=.5, error_mean=.7))
        training_diagnostics(model, data, tracer, fibers, seeds, tmp_path, 1000, log, device='cpu')
    finally:
        log.close()
    assert tracer.calls == 4  # Two seeds at two thresholds, no repeat for images.
    assert tracer.p.confidence == .7 and tracer.p.max_len == 8.
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
    records = [json.loads(line) for line in (tmp_path/'log.jsonl').read_text().splitlines()]
    for record in records[1:]:
        assert {k: record[k] for k in expected} == expected
    assert expected['coverage_mean'] == 1.
    assert all(record['coverage_max_len'] == 8. for record in records[1:])
    assert len(rgb) == 8
    for pixels in rgb:
        np.testing.assert_allclose(pixels, .2, atol=1e-7)
    expected_images = ['batch_001000.png', 'batch_coarse_001000.png', 'correction_001000.png',
                       'rollout_001000_c0.5.png', 'rollout_001000_c0.85.png']
    for path in [tmp_path/'images'/name for name in expected_images]+[tmp_path/'curves.png']:
        with Image.open(path) as image:
            assert image.format == 'PNG' and min(image.size) > 100
            image.verify()
    correction_labels = labels[str(tmp_path/'images/correction_001000.png')]
    assert 'GT' in correction_labels
    assert ('corrected proposal' in correction_labels) == correction
    assert 'dense curve error (voxels)' in labels[str(tmp_path/'curves.png')]


def test_flow_rollout_and_denoising_plot_contract_is_preserved(tmp_path):
    fibers, seeds = monitor()
    tracer = Tracer()
    summary = rollout_diag(tracer, fibers, seeds, tmp_path/'rollout.png', max_len=4., batch=2)
    assert tracer.calls == 1 and tracer.p.max_len == 8.
    assert summary['coverage_mean'] == 1. and summary['length_precision'] == 1.
    direct_tracer = Tracer()
    direct_tracer.p.max_len = 4.
    _, direct_summary = evaluate(direct_tracer, fibers, seeds, batch=2, coverage_max_len=4.)
    assert direct_summary == summary
    _, final_summary = evaluate(direct_tracer, fibers, seeds, batch=2)
    assert final_summary['coverage_mean'] == pytest.approx((4/89+4/9)/2)
    curves = torch.zeros(1, 5, 4, 3)
    curves[..., 2] = torch.arange(1, 5)
    plot_denoising(curves, torch.zeros(1, 3, 3), torch.zeros(1, 3), tmp_path/'denoising.png')
    with Image.open(tmp_path/'denoising.png') as image:
        image.verify()


@pytest.mark.parametrize('available,followed,cap,expected', [
    (1000., 100., 400., .25), (100., 25., 400., .25),
    (1000., 600., 400., 1.), (100., 100., 400., 1.), (0., 0., 400., 0.),
])
def test_shared_monitor_denominator_keeps_precision_fields(available, followed, cap, expected):
    row = dict(avail=available, followed=followed, coverage=followed/max(available, 1e-6),
               avail_nb=available, coverage_nb=followed/max(available, 1e-6),
               correct=80., offtrack=20., length=100., diverged=True)
    before = dict(row)
    result = monitor_coverage(row, cap)
    assert row == before
    assert result['coverage'] == result['coverage_nb'] == expected
    assert all(result[k] == row[k] for k in ('correct', 'offtrack', 'length', 'diverged'))
