"""Whole-crop path head (path_planes='crop') and Gaussian tube head: targets, model outputs, losses, training step."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model_fixtures import coordinate_config, coordinate_batch
from vesuvius.neural_tracing.fiber_follow.data import data as D
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.models.whole_crop import crop_path_loss, tube_loss, tube_targets
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength, crop_path_planes
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.train.train import (
    initialize_model_weights, initialize_training_optimizer, optimizer_update, prepare_training)


def line_fiber(length=400, lateral=0., tags=(False, False)):
    z = np.arange(length+1.)
    p = np.c_[np.full_like(z, lateral), np.zeros_like(z), z]
    return D.TracedFiber('line.json', p, arclength(p), 'H', endpoint_stop=tags)


def test_crop_planes_and_proposal_slice():
    crop = CropSpec(depth=160, width=128, behind=96, spacing=1.)
    planes = crop_path_planes(crop)
    assert planes[0] == -96 and planes[-1] == 63 and len(planes) == 160
    cfg = coordinate_config(fine=crop, n_future=16, path_planes='crop')
    assert list(cfg.path_plane_values[cfg.proposal_slice]) == list(range(1, 17))
    default = coordinate_config()
    assert list(default.path_plane_values[default.proposal_slice]) == list(default.path_plane_values)


def test_crop_targets_follow_the_original_fiber_at_any_offset_behind_and_ahead():
    crop = CropSpec(depth=40, width=24, behind=24, spacing=1.)
    fiber = line_fiber(lateral=5.)  # the head sits 5 vox beside the original fiber
    p, s = D.traversal_curve(fiber, False)
    planes = crop_path_planes(crop)
    ab, mask = D.crop_plane_targets(p, s, 200., np.array([0., 0., 200.]), np.eye(3), planes, (crop.width-1)/2)
    assert mask.all() and np.allclose(ab[:, 0], 5.) and np.allclose(ab[:, 1], 0.)
    # Outside the lateral extent the crossing is unknown.
    far = line_fiber(lateral=20.)
    p, s = D.traversal_curve(far, False)
    assert not D.crop_plane_targets(p, s, 200., np.array([0., 0., 200.]), np.eye(3), planes, (crop.width-1)/2)[1].any()
    # An annotation end inside the crop leaves later planes unknown.
    short = line_fiber(length=205)
    p, s = D.traversal_curve(short, False)
    _, mask = D.crop_plane_targets(p, s, 200., np.array([0., 0., 200.]), np.eye(3), planes, (crop.width-1)/2)
    assert mask[planes <= 5].all() and not mask[planes > 5].any()


def test_continuation_targets_add_whole_crop_targets_only_when_requested():
    crop = CropSpec(depth=40, width=24, behind=24, spacing=1.)
    fiber = line_fiber()
    off = D.continuation_targets(fiber, 200., False, np.array([0., 0., 200.]), np.eye(3), D.SampleConfig(crop=crop))
    assert 'crop_ab' not in off
    on = D.continuation_targets(fiber, 200., False, np.array([0., 0., 200.]), np.eye(3), D.SampleConfig(crop=crop, crop_targets=True))
    assert on['crop_ab'].shape == (len(crop_path_planes(crop)), 2) and on['crop_mask'].all()
    assert on['crop_curve'].shape == (D.CROP_CURVE_POINTS, 3) and on['crop_curve_mask'].sum() > 100
    assert not on['crop_curve_open_ends'].any()


def test_tube_target_is_a_gaussian_of_the_distance_to_the_fiber_and_ignores_open_ends():
    crop = CropSpec(depth=24, width=20, behind=8, spacing=1.)
    z = torch.arange(-30., 60.)
    curve = torch.stack((torch.full_like(z, 2.), torch.zeros_like(z), z), -1)[None]
    mask = torch.ones(1, len(z), dtype=torch.bool)
    target, known = tube_targets(curve, mask, torch.zeros(1, 2, dtype=torch.bool), crop, 1.5)
    half = (crop.width-1)/2
    # sample (d, h, w) has local x = w - half, y = h - half; the fiber is at x = 2, y = 0
    w = torch.arange(crop.width)-half
    profile = target[0, 10, int(half-.5)]  # a row near y = 0 (y = -0.5)
    expected = torch.exp(-.5*(((w-2.)**2+.25)**.5/1.5)**2)
    torch.testing.assert_close(profile, expected, atol=2e-2, rtol=0)
    assert known.all()
    # A curve that ends inside the crop at an untagged end: samples around that end are unknown.
    ends = torch.tensor([[False, True]])
    short = curve[:, :45]  # ends at z = 14
    _, known = tube_targets(short, mask[:, :45], ends, crop, 1.5)
    assert not known[0, 8+14, int(half), int(half+2)] and known[0, 0, 0, 0]  # around the end (d, y=0, x=2) vs far away


@pytest.mark.parametrize('tube', [False, True])
def test_whole_crop_model_outputs_losses_and_gradients(tube):
    torch.manual_seed(1)
    crop = CropSpec(depth=24, width=20, behind=12, spacing=1.)
    cfg = coordinate_config(fine=crop, n_future=4, path_planes='crop', tube_head=tube, memory='none')
    model = build_model(cfg).train()
    batch = crop_batch(cfg)
    hist, hmask = batch['hist'], batch['hmask']
    out = model.select_prediction(model.training_forward(batch['x'], hist, hmask, hist.new_full((), .5)))
    planes = len(cfg.path_plane_values)
    assert out['refinement_points'].shape[-2] == cfg.n_future
    assert out['refinement_crop_points'].shape[-2:] == (planes, 3)
    torch.testing.assert_close(out['refinement_crop_points'][:, :, cfg.proposal_slice], out['refinement_points'])
    assert ('tube_logits' in out) == tube
    terms = loss_terms(out, batch, cfg)
    loss = terms['geometry_per_state'].sum()+terms['confidence_per_state'].sum()
    if tube:
        assert out['tube_logits'].shape == (2, crop.depth, crop.width, crop.width)
        loss = loss+tube_loss(out['tube_logits'], batch, crop, 1.5)[0].sum()
    loss.backward()
    assert model.coordinates.weight.grad.abs().sum() > 0
    if tube:
        assert model.tube.readout.weight.grad.abs().sum() > 0
    # Inference returns the committed proposal (planes 1..n_future).
    model.eval()
    with torch.no_grad():
        assert model(batch['x'], hist, hmask)['points'].shape == (2, cfg.n_future, 3)


def test_crop_path_loss_ignores_unknown_planes():
    batch = dict(crop_ab=torch.zeros(1, 3, 2), crop_mask=torch.tensor([[1., 0., 1.]]))
    curve = torch.tensor([[[0., 0., -1.], [9., 9., 0.], [0., 0., 1.]]])
    assert crop_path_loss(curve, batch).item() == 0.


def test_training_step_with_whole_crop_and_tube_heads():
    torch.manual_seed(2)
    crop = CropSpec(depth=24, width=20, behind=12, spacing=1.)
    model, ema = initialize_model_weights(coordinate_config(fine=crop, n_future=4, path_planes='crop', tube_head=True,
                                                            memory='none'), 'cpu')
    opt, _, _ = initialize_training_optimizer(model, ema, SimpleNamespace(lr=.001, reset_optimizer=False))
    prepare_training(model, backend='eager')
    metrics = optimizer_update(model, ema, opt, [crop_batch(model.cfg)], 1, .001, tube_weight=1., tube_sigma=1.5)
    assert metrics['prediction_loss_type'] == 'whole-crop geometry' and metrics['tube_loss'] > 0
    assert np.isfinite(metrics['loss'])


def crop_batch(cfg, b=2):
    batch = coordinate_batch(cfg, b)
    batch['x'] = {k: v for k, v in batch['x'].items() if not k.startswith('history_')}
    planes = len(cfg.path_plane_values)
    batch['crop_ab'] = torch.zeros(b, planes, 2)
    batch['crop_mask'] = torch.ones(b, planes)
    z = torch.arange(-40., 40.)
    batch['crop_curve'] = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z), -1)[None].repeat(b, 1, 1)
    batch['crop_curve_mask'] = torch.ones(b, len(z), dtype=torch.bool)
    batch['crop_curve_open_ends'] = torch.zeros(b, 2, dtype=torch.bool)
    return batch
