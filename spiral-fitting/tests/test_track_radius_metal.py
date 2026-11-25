"""The fused track radius kernels against the eager loss.

track_loss_metal replaces the radius half of tracks.iter_track_losses on MPS.
Each case builds a sample shaped like tracks._draw_track_sample's (primary
rows, then crossing-partner rows in group order, variable row lengths, rows
that cross theta = 0 several times, target indices from the same linspace
rounding) and compares the loss, the point gradient and the dr_per_winding
gradient with the eager path on MPS, and both with a float64 CPU run of the
eager path. The fused result must be about as close to float64 as eager
float32 is: within FACTOR times the eager spread plus a small floor.

A known negative checks that the comparison can fail: dropping the partner
alignment (a plausible kernel bug) must break it.
"""
import math
import os

import pytest
import torch

import track_loss_metal
from sample_spiral import get_theta_and_radii
from tracks import (_crossing_row_alignments, _grouped_same_radius_loss,
                    _unwrap_ragged_track_radii)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='needs MPS')

CFG = {
    'track_radius_target': 'mean',
    'track_radius_loss_margin': 0.025,
    'track_radius_within_norm_p': 6.0,
}
DR = 37.5
TARGET_POINTS = 96
FACTOR = 4.0
# float32 grad_dr errors measured over 20 seeds: fused <= 3.1, eager <= 2.2 in
# units of eps32 * sum|g_shifted * theta| / 2pi
DR_ULPS = 8.0
SEEDS = range(7, 7 + int(os.environ.get('TRACK_RADIUS_SEEDS', '4')))


def _sample(seed, groups=300, max_partners=3, device='cpu'):
    """A random sample with _draw_track_sample's layout, and its points."""
    gen = torch.Generator().manual_seed(seed)
    partners = torch.randint(0, max_partners + 1, (groups,), generator=gen)
    partner_group = torch.repeat_interleave(torch.arange(groups), partners)
    partner_slot = torch.cat([torch.arange(int(n)) for n in partners]) \
        if int(partners.sum()) else torch.empty(0, dtype=torch.int64)
    rows = groups + int(partners.sum())
    lengths = torch.randint(1, 14, (rows,), generator=gen)
    row_starts = torch.zeros(rows + 1, dtype=torch.int64)
    torch.cumsum(lengths, 0, out=row_starts[1:])
    row_id = torch.repeat_interleave(torch.arange(rows), lengths)
    n = int(row_starts[-1])

    # points walk around the umbilicus, so rows cross theta = 0, at radii
    # around a few windings; a few sit on the x axis's near-zero clamp
    base_theta = torch.rand(rows, generator=gen) * 2 * math.pi
    base_radius = DR * (2 + 6 * torch.rand(rows, generator=gen))
    step = 0.7 * (torch.rand(rows, generator=gen) - 0.5)
    local = torch.arange(n) - row_starts[:-1][row_id]
    theta = base_theta[row_id] + step[row_id] * local * 3
    radius = base_radius[row_id] + theta / (2 * math.pi) * DR \
        + 2.0 * torch.randn(n, generator=gen)
    spiral = torch.stack([
        torch.rand(n, generator=gen) * 100,
        torch.sin(theta) * radius,
        torch.cos(theta) * radius,
    ], dim=-1)
    spiral[::97, 2] = 0.0

    group_id = torch.cat([torch.arange(groups), partner_group])
    row_slot = torch.cat([torch.zeros(groups, dtype=torch.int64), partner_slot + 1])
    fractions = torch.linspace(0.0, 1.0, TARGET_POINTS)
    target_local = torch.round(
        fractions[None, :] * (lengths[:, None] - 1)).to(torch.int64)
    target_flat_idx = row_starts[:-1, None] + target_local
    partner_rows = torch.arange(groups, rows, dtype=torch.int64)
    self_local = (torch.rand(len(partner_rows), generator=gen)
                  * lengths[partner_group]).long()
    partner_local = (torch.rand(len(partner_rows), generator=gen)
                     * lengths[partner_rows]).long()
    sample = {
        'row_id': row_id,
        'row_starts': row_starts,
        'group_id': group_id,
        'row_slot': row_slot,
        'num_groups': groups,
        'group_width': max_partners + 1,
        'target_flat_idx': target_flat_idx,
        'partner_rows': partner_rows,
        'primary_cross_flat': row_starts[partner_group] + self_local,
        'partner_cross_flat': row_starts[partner_rows] + partner_local,
        'track_idx': torch.arange(rows),
    }
    sample = {k: v.to(device) if torch.is_tensor(v) else v
              for k, v in sample.items()}
    return sample, spiral.to(device)


def _eager(spiral, dr, sample, align=True):
    """The radius half of tracks.iter_track_losses, verbatim."""
    theta, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
    shifted, _adjust = _unwrap_ragged_track_radii(
        theta, shifted, dr, sample['row_id'], sample['row_starts'])
    if align and sample['partner_rows'].numel() > 0:
        alignment = _crossing_row_alignments(
            shifted, sample['primary_cross_flat'],
            sample['partner_cross_flat'], sample['partner_rows'],
            len(sample['track_idx']))
        shifted = shifted + alignment[sample['row_id']]
    target_values = shifted[sample['target_flat_idx']]
    loss, _, _ = _grouped_same_radius_loss(
        shifted, target_values, sample['row_id'], sample['group_id'],
        sample['row_slot'], sample['num_groups'], sample['group_width'], dr,
        CFG)
    return loss


def _run(fn, spiral, dr_value, sample, dtype, device):
    spiral = spiral.to(device=device).to(dtype).detach().requires_grad_(True)
    dr = torch.tensor(dr_value, device=device, dtype=dtype, requires_grad=True)
    sample = {k: v.to(device) if torch.is_tensor(v) else v
              for k, v in sample.items()}
    loss = fn(spiral, dr, sample)
    loss.backward()
    return (loss.detach().cpu().double(), spiral.grad.cpu().double(),
            dr.grad.cpu().double())


def _dr_condition(spiral, sample):
    """float64 sum|d loss / d shifted * theta| / 2pi, grad_dr's rounding scale"""
    spiral = spiral.double().detach().requires_grad_(True)
    dr = torch.tensor(DR, dtype=torch.float64)
    theta, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
    shifted.retain_grad()
    unwrapped, _ = _unwrap_ragged_track_radii(
        theta, shifted, dr, sample['row_id'], sample['row_starts'])
    if sample['partner_rows'].numel() > 0:
        alignment = _crossing_row_alignments(
            unwrapped, sample['primary_cross_flat'],
            sample['partner_cross_flat'], sample['partner_rows'],
            len(sample['track_idx']))
        unwrapped = unwrapped + alignment[sample['row_id']]
    loss, _, _ = _grouped_same_radius_loss(
        unwrapped, unwrapped[sample['target_flat_idx']], sample['row_id'],
        sample['group_id'], sample['row_slot'], sample['num_groups'],
        sample['group_width'], dr, CFG)
    loss.backward()
    return (shifted.grad * theta.detach()).abs().sum().item() / (2 * math.pi)


def _fused(spiral, dr, sample):
    return track_loss_metal.track_radius_loss_metal(spiral, dr, sample, CFG)[0]


def _spread(a, b):
    return (a - b).abs().max().item()


@pytest.mark.parametrize('seed', SEEDS)
def test_fused_matches_eager(seed):
    sample, spiral = _sample(seed)
    ref = _run(_eager, spiral, DR, sample, torch.float64, 'cpu')
    eager = _run(_eager, spiral, DR, sample, torch.float32, 'mps')
    fused = _run(_fused, spiral, DR, sample, torch.float32, 'mps')
    dr_floor = DR_ULPS * torch.finfo(torch.float32).eps * _dr_condition(spiral, sample)
    for name, r, e, f in zip(('loss', 'grad_spiral', 'grad_dr'), ref, eager, fused):
        scale = r.abs().max().item()
        floor = 1e-6 * max(scale, 1e-12)
        if name == 'grad_dr':
            floor = max(floor, dr_floor)
        bound = FACTOR * _spread(e, r) + floor
        assert _spread(f, r) <= bound, (
            f'{name}: fused {_spread(f, r):.3e} vs eager {_spread(e, r):.3e} '
            f'(scale {scale:.3e})')


def test_known_negative_without_alignment_fails():
    sample, spiral = _sample(SEEDS[0])
    ref = _run(_eager, spiral, DR, sample, torch.float64, 'cpu')
    eager = _run(_eager, spiral, DR, sample, torch.float32, 'mps')
    wrong = _run(lambda s, d, smp: _eager(s, d, smp, align=False), spiral, DR,
                 sample, torch.float32, 'mps')
    bound = FACTOR * _spread(eager[0], ref[0]) + 1e-6 * ref[0].abs().item()
    assert _spread(wrong[0], ref[0]) > bound


def test_detached_outputs_match_eager_intermediates():
    sample, spiral = _sample(SEEDS[0], device='mps')
    dr = torch.tensor(DR, device='mps')
    _, unwrapped, adjust, theta, targets = track_loss_metal.track_radius_loss_metal(
        spiral, dr, sample, CFG)
    theta_e, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
    shifted, adjust_e = _unwrap_ragged_track_radii(
        theta_e, shifted, dr, sample['row_id'], sample['row_starts'])
    alignment = _crossing_row_alignments(
        shifted, sample['primary_cross_flat'], sample['partner_cross_flat'],
        sample['partner_rows'], len(sample['track_idx']))[sample['row_id']]
    assert torch.allclose(theta, theta_e, atol=1e-5)
    assert torch.allclose(unwrapped, shifted + alignment, rtol=1e-5, atol=1e-3)
    assert torch.allclose(adjust, adjust_e + alignment, rtol=1e-5, atol=1e-3)
    # the group targets feed the DT loss's snap target, so their values matter
    unwrapped_e = shifted + alignment
    _, targets_e, _ = _grouped_same_radius_loss(
        unwrapped_e, unwrapped_e[sample['target_flat_idx']], sample['row_id'],
        sample['group_id'], sample['row_slot'], sample['num_groups'],
        sample['group_width'], dr, CFG)
    assert targets.shape == (sample['num_groups'],)
    assert torch.allclose(targets, targets_e, rtol=1e-5, atol=1e-3)
