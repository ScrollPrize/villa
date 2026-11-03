"""Raw leaf reporting must not change weighted objectives or gradients."""
from dataclasses import replace

import pytest
import torch

from test_normal_tube import tube_batch
from test_fiber_annotations import model
from test_paired_embeddings import paired_batch, split_batch
from vesuvius.neural_tracing.fiber_trace_3d.embedding import embedding_statistics
from vesuvius.neural_tracing.fiber_trace_3d.train import _forward_loss_statistics, _finalize_loss_statistics
from vesuvius.neural_tracing.fiber_trace_3d.refinement import RefinementConfig, refinement_report


def test_annotation_logs_raw_and_total_weighted():
    batch, cfg = tube_batch(sideband=True)
    net = model()
    net.annotations = replace(cfg.annotations, normal_weight=3., width_weight=.02, hv_weight=.01)
    stats = _forward_loss_statistics(net, batch)
    losses = _finalize_loss_statistics(stats, direction_weight=7., presence_weight=2., sideband_weight=.4)
    for name in ('normal', 'width', 'hv'):
        expected = getattr(stats, name + '_unweighted_sum') / getattr(stats, name + '_count').clamp_min(1)
        torch.testing.assert_close(losses[name], expected)
    expected_total = (7*losses['direction'] + 2*losses['presence'] +
                      3*losses['normal'] + .02*losses['width'] + .01*losses['hv'])
    torch.testing.assert_close(losses['total'], expected_total)
    torch.testing.assert_close(losses['presence'], losses['presence_ridge'] +
        .5*losses['presence_distribution'] + .5*losses['presence_background'] +
        .2*(losses['presence_sideband_left'] + losses['presence_sideband_right']))
    params = list(net.parameters())
    actual = torch.autograd.grad(losses['total'], params, retain_graph=True)
    expected = torch.autograd.grad(expected_total, params)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


def test_embedding_combination_uses_raw_leaves_but_total_keeps_weight():
    batch, _ = tube_batch()
    stats = _forward_loss_statistics(model(), batch)
    v = stats.direction_sum.new_tensor
    stats = replace(stats, positive_embedding_sum=v(.4), positive_embedding_count=v(2.),
        positive_embedding_unweighted_sum=v(4.), negative_embedding_sum=v(1.2),
        negative_embedding_count=v(2.), negative_embedding_unweighted_sum=v(12.),
        contrastive_embedding_sum=v(.6), contrastive_embedding_count=v(2.),
        contrastive_embedding_unweighted_sum=v(6.))
    loss = _finalize_loss_statistics(stats, direction_weight=0., presence_weight=0.)
    assert loss['embedding_positive'] == 2.
    assert loss['embedding_negative'] == 6.
    assert loss['embedding_contrastive'] == 3.
    assert loss['embedding'] == 7.  # .5*(2+6) + 3; no outer .1 factor
    annotation_total = sum(getattr(stats, k+'_sum') / getattr(stats, k+'_count').clamp_min(1)
                           for k in ('normal', 'width', 'hv'))
    torch.testing.assert_close(loss['total'] - annotation_total, v(.7))


@pytest.mark.parametrize('mode', ['pairwise', 'local_contrastive', 'split'])
@pytest.mark.parametrize('weight', [0., .03, 5.])
def test_embedding_raw_sums_ignore_outer_weight(mode, weight):
    batch = split_batch() if mode == 'split' else paired_batch()
    batch = replace(batch, embedding_geometry=tuple(replace(g, weight=weight,
        objective='pairwise' if mode == 'pairwise' else 'local_contrastive',
        pair_id=None if mode == 'pairwise' else g.pair_id) for g in batch.embedding_geometry))
    unit_batch = replace(batch, embedding_geometry=tuple(replace(g, weight=1.) for g in batch.embedding_geometry))
    torch.manual_seed(14)
    e = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    p = torch.rand(2, 1, 8, 8, 8)
    raw = {}
    values = embedding_statistics(e, p, batch, raw_sums=raw)
    reference = embedding_statistics(e, p, unit_batch)
    names = ('positive', 'negative', 'contrastive', 'short', 'long')
    for i, name in enumerate(names[:len(values)//2]):
        torch.testing.assert_close(values[2*i], reference[2*i]*weight)
        torch.testing.assert_close(raw.get(name+'_embedding_unweighted_sum', e.new_zeros(())), reference[2*i])
    # Reporting is detached and does not add another autograd path.
    assert all(not value.requires_grad for value in raw.values())
    actual_grad = torch.autograd.grad(sum(values[::2]), e, retain_graph=True)[0]
    unit_grad = torch.autograd.grad(sum(reference[::2]), e)[0]
    torch.testing.assert_close(actual_grad, unit_grad*weight)


def test_refinement_raw_leaves_weighted_combined_and_disabled_terms():
    cfg = RefinementConfig(gt_weight=.2, shape_normal_weight=10., shape_tangent_weight=100., direction_weight=.1)
    result = refinement_report([2., .4, 6., 8., 10., 12.], cfg)
    assert result['presence'] == pytest.approx(.2)
    assert result['smoothing_normal'] == pytest.approx(.3)
    assert result['smoothing_tangent'] == pytest.approx(.04)
    assert result['localization'] == pytest.approx(25.)
    assert result['direction'] == pytest.approx(60.)
    assert result['smoothing'] == pytest.approx(7.)
    assert result['total'] == pytest.approx(18.2)
    disabled = refinement_report([2., .4, 0., 0., 0., 0.], replace(cfg, gt_weight=0.,
        shape_normal_weight=0., shape_tangent_weight=0., direction_weight=0.))
    assert 'direction' not in disabled and 'localization' not in disabled
    assert disabled['smoothing'] == 0.
    # New statistics retain raw values even when their weight is zero.
    fresh = refinement_report([2., .4, 0., 0., 0., 0., .4, 6., 8., 10., float('nan')],
        replace(cfg, gt_weight=0., shape_normal_weight=0., shape_tangent_weight=0., direction_weight=0.))
    assert fresh['smoothing_normal'] == 3. and fresh['smoothing_tangent'] == 4.
    assert fresh['localization'] == 5. and fresh['total'] == .2
