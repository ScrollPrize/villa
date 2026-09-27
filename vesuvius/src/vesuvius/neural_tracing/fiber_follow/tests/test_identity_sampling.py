"""Regression checks for label-independent interpolation and full history anchors."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.regression.supervision import identity_terms
from vesuvius.neural_tracing.fiber_follow.shared.components import (
    ComponentRule, crop_indices, lateral_components, sample_pairs, volume_at,
)
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid


def fixture(width=41):
    crop = CropSpec(depth=45, width=width, behind=8, spacing=.5)
    grid = crop_local_grid(crop)
    curve = np.c_[np.full(81, .13), np.full(81, -.17), np.linspace(-3.87, 12.13, 81)]
    presence = ((np.abs(grid[..., 0]-4.) <= .8) & (np.abs(grid[..., 1]) <= .8)).astype(np.float32)
    found = lateral_components(presence, crop, curve)
    return crop, curve, presence, found


@pytest.mark.parametrize('width', [40, 41])
def test_negative_interpolation_weights_match_positive_and_keep_valid_support(width):
    crop, curve, presence, found = fixture(width)
    kwargs = dict(positives=4, negatives=8, margin=2., along_margin=1.,
                  component_labels=found['labels'])
    result = sample_pairs(curve, presence, crop, found['local'], found['nearest'],
                          np.random.default_rng(7), **kwargs)
    repeated = sample_pairs(curve, presence, crop, found['local'], found['nearest'],
                            np.random.default_rng(7), **kwargs)
    for a, b in zip(result, repeated):
        np.testing.assert_array_equal(a, b)
    pos, pm, neg, nm = result
    assert pm.all() and nm.all()
    for p, points in zip(pos, neg):
        assert np.linalg.norm(curve-p, axis=1).min() < 1e-6  # positive stays on annotation
        difference = crop_indices(crop, points)-crop_indices(crop, [p])
        np.testing.assert_allclose(difference, np.rint(difference), atol=2e-6)
        assert np.all(volume_at(presence, crop, points, order=1) >= .7)
        assert np.all(volume_at(found['foreign'], crop, points, order=1) >= .7)
        assert np.all(np.linalg.norm(points[:, None]-curve[None], axis=-1).min(-1) > 1.5)
        assert np.max(np.abs(points[:, 2]-p[2])) <= ComponentRule().along_window+.11
    bounds = crop_local_grid(crop)
    assert np.all(neg[..., 2] >= bounds[..., 2].min()+1.)
    assert np.all(neg[..., 2] <= bounds[..., 2].max()-1.)


def test_reject_shifted_negative_without_snapping_when_interpolated_presence_is_low():
    crop = CropSpec(depth=25, width=25, behind=4, spacing=.5)
    grid = crop_local_grid(crop)
    curve = np.c_[np.full(30, .24), np.zeros(30), np.arange(30)*.25]
    presence = ((grid[..., 0] == 4.) & (grid[..., 1] == 0.)).astype(np.float32)
    found = lateral_components(presence, crop, curve)
    assert len(found['local']) > 0
    pos, pm, neg, nm = sample_pairs(curve, presence, crop, found['local'], found['nearest'],
                                   np.random.default_rng(2), margin=0.)
    assert pm.all() and not nm.any()  # shifted presence is at most .52, despite grid presence 1


def test_departed_head_uses_same_phase_and_checks_component_and_receptive_field():
    crop, curve, presence, found = fixture()
    kwargs = dict(positives=2, negatives=1, margin=2., along_margin=1.,
                  extra_negative=np.array([4., 0., 0.]), component_labels=found['labels'])
    pos, pm, neg, nm = sample_pairs(curve, presence, crop, np.empty((0, 3)), np.empty(0, int),
                                   np.random.default_rng(2), **kwargs)
    assert pm.all() and nm.all()
    diff = crop_indices(crop, neg[:, 0])-crop_indices(crop, pos)
    np.testing.assert_allclose(diff, np.rint(diff), atol=2e-6)
    # A head outside the appearance crop's full receptive field cannot supply a negative.
    kwargs['appearance_crop'] = CropSpec(depth=30, width=41, behind=0, spacing=.5)
    _, pm, _, nm = sample_pairs(curve, presence, crop, np.empty((0, 3)), np.empty(0, int),
                                np.random.default_rng(2), **kwargs)
    assert pm.any() and not nm.any()


def test_full_target_history_scores_all_candidates_and_padding_has_no_effect():
    cfg = SimpleNamespace(recent_patches=4)
    history = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 1.]]], requires_grad=True)
    output = dict(history_embedding=history, patch_mask=torch.ones(1, 4),
                  query_embedding=torch.tensor([[[0., 1.], [1., 0.], [-1., 0.]]]),
                  query_support=torch.ones(1, 3, dtype=torch.bool))
    batch = dict(patch_on_fiber=torch.ones(1, 4), positive_mask=torch.ones(1, 1),
                 negative_mask=torch.tensor([[[1., 0.]]]))
    full = identity_terms(output, batch, cfg)['identity_per_state']
    full.sum().backward()
    assert (history.grad.norm(dim=-1) > 0).all()  # includes all older positive-history patches
    output['query_embedding'][0, 2] = torch.tensor([1000., -1000.])
    torch.testing.assert_close(identity_terms(output, batch, cfg)['identity_per_state'], full)
    output['patch_mask'] = torch.tensor([[1., 1., 0., 0.]])
    short = identity_terms(output, batch, cfg)['identity_per_state']
    assert short.item() > full.item()+1.  # truncation really would weaken this positive; we do not do it
    batch['negative_mask'].zero_()
    terms = identity_terms(output, batch, cfg)
    assert terms['identity_count'] == 0 and terms['identity_per_state'].item() == 0
