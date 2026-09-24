"""Contracts of the direct decoder: connectivity, learning, and target censoring."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.data import SampleConfig, TracedFiber, continuation_targets
from vesuvius.neural_tracing.fiber_follow.geometry import arclength, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.model import FollowNet, FollowNetConfig
from vesuvius.neural_tracing.fiber_follow.supervision import coordinate_loss, loss_fn, teacher_candidates


def config():
    return FollowNetConfig(in_channels=3, depth=24, width=17, behind=6, spacing=.5,
                           widths=(8, 16), hidden=16, n_future=8, future_step=1.,
                           hist_points=4, hist_stride=2, clean_points=4, n_candidates=3,
                           norm='group')


def inputs(cfg):
    x = torch.randn(1, 3, cfg.depth, cfg.width, cfg.width)
    hist = torch.zeros(1, cfg.hist_points*cfg.hist_stride, 3)
    hist[..., 2] = -torch.arange(1, hist.shape[1]+1)
    return x, hist, torch.ones(hist.shape[:2])


def targets(cfg):
    q = (cfg.n_future-1)*4+1
    z = torch.linspace(1, cfg.n_future, q)
    ab = torch.stack([.04*z*z, .1*z], -1)[None]
    return dict(dense_ab=ab, dense_mask=torch.ones(1, q), offtrack=torch.zeros(1),
                plane_ab=ab[:, ::4], plane_mask=torch.ones(1, cfg.n_future),
                clean_local=torch.zeros(1, cfg.clean_points+1, 3),
                clean_mask=torch.ones(1, cfg.clean_points+1),
                hmask=torch.ones(1, cfg.hist_points*cfg.hist_stride))


def test_all_paths_are_connected_even_with_extreme_predictions():
    torch.manual_seed(1)
    cfg = config()
    model = FollowNet(cfg).eval()
    with torch.no_grad():
        model.clean_head[-1].bias.fill_(1000.)
        model.proposal_decoder.velocity_head.bias.fill_(1000.)
    x, hist, mask = inputs(cfg)
    out = model(x, hist, mask)
    assert not any('tube' in k or 'heat' in k for k in out)
    assert not any('heat' in name or 'tube' in name for name, _ in model.named_parameters())
    observed = torch.cat([torch.zeros(1, 1, 3), hist[:, :cfg.clean_points]], 1)
    assert (out['clean_history']-observed).norm(dim=-1).max() <= cfg.max_history_correction+1e-5
    anchor = out['clean_history'][:, None, :1, :2].expand(-1, cfg.n_candidates, -1, -1)
    path = torch.cat([anchor, out['candidates'][..., :2]], 2)
    assert (path[:, :, 1:]-path[:, :, :-1]).norm(dim=-1).max() <= cfg.max_lateral_slope*cfg.future_step+1e-5
    bound = ((cfg.max_history_correction+cfg.max_lateral_slope*cfg.future_step)**2+cfg.future_step**2)**.5
    assert out['candidates'][:, :, 0].norm(dim=-1).max() <= bound+1e-5
    torch.testing.assert_close(out['candidates'][0, 0, :, 2], torch.arange(1, 9).float())


def test_coordinate_loss_reaches_decoder_images_and_history():
    torch.manual_seed(5)
    cfg = config()
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    out = model(x, hist, mask)
    loss, _ = coordinate_loss(out['candidates'], out['clean_history'], targets(cfg), cfg)
    loss.backward()
    for parameter in [model.proposal_decoder.velocity_head.weight,
                      model.proposal_decoder.cell.weight_ih,
                      model.proposal_decoder.mode_tokens.weight,
                      model.history_tokens[0].weight, model.clean_head[-1].weight,
                      model.encoders[0][0].weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_direct_paths_respond_to_history_and_are_independent_of_teacher_paths():
    torch.manual_seed(11)
    cfg = config()
    model = FollowNet(cfg).eval()
    x, hist, mask = inputs(cfg)
    original = model(x, hist, mask)
    other = hist.clone()
    other[..., 0] = torch.linspace(-.1, -1., hist.shape[1])
    changed = model(x, other, mask)
    assert (original['candidates']-changed['candidates']).abs().max() > 1e-3
    extra = torch.randn(1, 5, cfg.n_future, 3)*20
    with_extra = model(x, hist, mask, extra)
    for key in ('candidates', 'ranks', 'confidence_logits'):
        torch.testing.assert_close(original[key], with_extra[key][:, :cfg.n_candidates], atol=1e-6, rtol=1e-5)


def test_direct_decoder_is_independent_across_batches():
    torch.manual_seed(12)
    cfg = config()
    model = FollowNet(cfg).eval()
    args = inputs(cfg)
    batch = [a.expand(2, *a.shape[1:]).clone() for a in args]
    batch[1][1, :, 0] = 1.
    together = model(*batch)['candidates']
    for i in range(2):
        single = model(*[a[i:i+1] for a in batch])['candidates']
        torch.testing.assert_close(together[i:i+1], single, atol=1e-5, rtol=1e-5)


def test_coordinate_optimization_learns_a_curved_continuation():
    torch.manual_seed(21)
    cfg = config()
    model = FollowNet(cfg)
    args = inputs(cfg)
    batch = targets(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        out = model(*args)
        loss, _ = coordinate_loss(out['candidates'], out['clean_history'], batch, cfg)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]*.3


@pytest.mark.parametrize('case', ['unknown', 'offtrack'])
def test_unlabeled_states_have_no_coordinate_gradient(case):
    cfg = config()
    batch = targets(cfg)
    if case == 'unknown':
        batch['dense_mask'].zero_()
        batch['plane_mask'].zero_()
    else:
        batch['offtrack'].fill_(1)
    paths = torch.randn(1, cfg.n_candidates, cfg.n_future, 3, requires_grad=True)
    clean = torch.randn(1, cfg.clean_points+1, 3, requires_grad=True)
    loss, _ = coordinate_loss(paths, clean, batch, cfg)
    loss.backward()
    assert loss.item() == 0 and not paths.grad.any() and not clean.grad.any()


def test_unknown_tail_is_censored_and_teacher_candidates_cannot_win_coordinate_assignment():
    cfg = config()
    batch = targets(cfg)
    batch['dense_mask'][:, 5:] = 0  # Only z=1..2 is annotated.
    batch['plane_mask'][:, 2:] = 0
    paths = torch.randn(1, cfg.n_candidates+5, cfg.n_future, 3, requires_grad=True)
    clean = torch.zeros(1, cfg.clean_points+1, 3)
    loss, _ = coordinate_loss(paths, clean, batch, cfg)
    loss.backward()
    assert paths.grad[:, :cfg.n_candidates, :2].abs().sum() > 0
    assert not paths.grad[:, :, 2:].any()
    assert not paths.grad[:, cfg.n_candidates:].any()


def test_whole_path_assignment_does_not_select_a_different_mode_at_each_point():
    cfg = replace(config(), n_candidates=2)
    batch = targets(cfg)
    batch['dense_ab'].zero_()
    batch['plane_ab'].zero_()
    paths = torch.zeros(1, 2, 8, 3)
    paths[:, 0, :4, 0] = 2.
    paths[:, 1, 4:, 0] = 2.
    _, metrics = coordinate_loss(paths, torch.zeros(1, 5, 3), batch, cfg)
    assert metrics['coordinate_full'] > .2


def test_follower_labels_do_not_construct_gaussian_targets():
    cfg = SampleConfig(n_future=8, future_step=1.)
    points = np.array([[float(x), 0., 0.] for x in range(30)])
    fiber = TracedFiber('line', points, arclength(points), '')
    out = continuation_targets(fiber, 10., False, np.array([10., 0., 0.]),
                               frame_from_heading(np.array([1., 0., 0.])), cfg)
    assert not any('tube' in k or 'heat' in k for k in out)
    assert out['plane_mask'].sum() == 8


def test_absent_history_is_not_a_cleaning_target():
    torch.manual_seed(22)
    cfg = config()
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    mask.zero_()
    batch = targets(cfg)
    batch.update(hist=hist, hmask=mask, endpoint_known=torch.zeros(1), end_local=torch.zeros(1, 3),
                 replay_valid=torch.zeros(1, cfg.n_candidates),
                 replay_candidates=torch.zeros(1, cfg.n_candidates, cfg.n_future, 3))
    out = model(x, hist, mask, teacher_candidates(batch, cfg))
    _, before = loss_fn(out, batch, cfg)
    batch['clean_local'][:, 1:] = 1000
    _, after = loss_fn(out, batch, cfg)
    assert before['clean_loss'] == after['clean_loss']


def test_beam_owns_its_optional_head_without_direct_decoder_parameters():
    from vesuvius.neural_tracing.fiber_follow.beam.model import BeamNetConfig, BeamRankNet
    cfg = BeamNetConfig(**config().to_dict())
    model = BeamRankNet(cfg)
    x, hist, mask = inputs(cfg)
    paths = torch.randn(1, 2, 8, 3)
    out = model(x, hist, mask, paths, torch.ones(1, 2, 8), torch.zeros(1, 2))
    assert out['ranks'].shape == (1, 2) and out['tube_logits'].shape == x.shape[:1]+x.shape[2:]
    assert not hasattr(model, 'proposal_decoder')
