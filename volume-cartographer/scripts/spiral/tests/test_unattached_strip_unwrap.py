"""Regression tests for the unattached-strip loss's theta=0 unwrap on strips
that span multiple windings (long fibers).

The loss samples only num_points_per_pcl points per row; on a fiber wrapping
several turns, consecutive samples can sit more than pi apart in theta, where
a consecutive-diff crossing detector miscounts seam crossings and assigns
wrong winding offsets even to a perfectly-fit strip. The fix counts crossings
along the full dense walk (_dense_walk_crossing_adjustments); these tests pin
that a geometrically perfect spiral fiber yields ~zero radius and DT loss
regardless of how many windings it spans."""

import numpy as np
import torch

from config import Config
from losses import (
    PackedDenseWalks,
    _dense_walk_crossing_adjustments,
    build_pcl_sampling_strata,
    get_unattached_pcl_strip_losses,
)

DR = 12.0


class IdentityTransform:
    def __call__(self, zyxs):
        return zyxs

    def inv(self, spiral_zyxs):
        return spiral_zyxs


def _make_cfg():
    return Config().as_dict()


def _perfect_spiral_fiber(wraps, winding=66, spacing=40.0, theta0=0.0):
    # Points exactly on the spiral sheet r = DR * (winding + theta / 2pi),
    # arc-length stepped at `spacing` -- matching pcl_fiber_min_point_spacing.
    # Lying exactly on a sheet means the strip-median DT target snaps to the
    # fiber's own winding, so both losses must read ~zero.
    pts = []
    theta = theta0
    while theta < theta0 + wraps * 2 * np.pi:
        r = DR * (winding + theta / (2 * np.pi))
        pts.append([0.0, np.sin(theta) * r, np.cos(theta) * r])  # z, y, x
        theta += spacing / r
    return np.asarray(pts, dtype=np.float32)


def _flat_bundle(zyxs_list):
    lengths = np.array([len(z) for z in zyxs_list], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    return {
        'zyxs': torch.from_numpy(np.concatenate(zyxs_list, axis=0)),
        'windings': torch.zeros(int(starts[-1]), dtype=torch.float32),
        'starts_cpu': torch.from_numpy(starts),
        'total': int(starts[-1]),
    }


def _strips(zyxs_list):
    return [
        {'id': i, 'name': f'strip{i}', 'source_file': None,
         'zyxs': z, 'windings': np.zeros(len(z), dtype=np.float32),
         'link_points': {}}
        for i, z in enumerate(zyxs_list)
    ]


def _run_losses(zyxs_list, cfg, num_steps=25, compute_dt=True, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    strips = _strips(zyxs_list)
    flat = _flat_bundle(zyxs_list)
    strata = build_pcl_sampling_strata(
        ['fibers'] * len(zyxs_list), cfg,
        member_weights=[1] * len(zyxs_list))
    components = [[i] for i in range(len(zyxs_list))]
    edges = [[] for _ in zyxs_list]
    dr = torch.tensor(DR)
    radius_losses, dt_losses = [], []
    for _ in range(num_steps):
        radius_loss, dt_loss = get_unattached_pcl_strip_losses(
            IdentityTransform(), dr, strips, components, edges, strata,
            lambda _strips, _device: flat,
            len(zyxs_list), cfg['sample_count_unattached_pcl_points_per_step'],
            compute_dt=compute_dt, cfg=cfg,
        )
        radius_losses.append(float(radius_loss))
        dt_losses.append(float(dt_loss))
    return np.array(radius_losses), np.array(dt_losses)


def test_multiwrap_perfect_fiber_has_zero_radius_and_dt_loss():
    # A perfect 20-wrap fiber sampled at 32 points: consecutive picks are far
    # more than pi apart in theta, so the sparse unwrap this test regresses
    # against miscounted crossings (mean radius loss ~4.4*DR).
    cfg = _make_cfg()
    radius_losses, dt_losses = _run_losses([_perfect_spiral_fiber(20.0)], cfg)
    assert radius_losses.max() < 1e-3
    assert dt_losses.max() < 1e-3


def test_five_wrap_perfect_fiber_has_zero_radius_loss():
    cfg = _make_cfg()
    radius_losses, _ = _run_losses(
        [_perfect_spiral_fiber(5.0)], cfg, compute_dt=False)
    assert radius_losses.max() < 1e-3


def test_subwrap_strip_across_seam_still_has_zero_loss():
    # A short strip crossing theta=0 exactly once: the legacy sparse unwrap
    # handled this correctly, so the dense-walk adjustments (re-anchored at
    # each row's first pick) must reproduce zero loss here too.
    cfg = _make_cfg()
    fiber = _perfect_spiral_fiber(0.5, theta0=1.75 * np.pi)
    radius_losses, dt_losses = _run_losses([fiber], cfg)
    assert radius_losses.max() < 1e-3
    assert dt_losses.max() < 1e-3


def test_dense_walk_adjustments_are_anchored_at_first_pick():
    # The adjustment at each row's first pick must be zero (legacy frame), and
    # picks separated by full wraps must differ by whole -DR steps.
    fiber = _perfect_spiral_fiber(3.0, spacing=40.0)
    flat = _flat_bundle([fiber])
    theta = np.arctan2(fiber[:, 1], fiber[:, 2]) % (2 * np.pi)
    crossings = np.concatenate([[0], np.cumsum(np.diff(theta) < -np.pi)])
    first, mid, last = 3, len(fiber) // 2, len(fiber) - 1
    picks = np.array([[first, mid, last]], dtype=np.int64)
    packed = PackedDenseWalks(
        rows=torch.tensor([0]),
        walk_zyxs=flat['zyxs'][None],
        pick_positions=torch.from_numpy(picks),
    )
    sampled_theta = torch.from_numpy(theta[picks])
    adjustments, rows = _dense_walk_crossing_adjustments(
        IdentityTransform(), torch.tensor(DR), sampled_theta, packed)
    expected = -DR * (crossings[[first, mid, last]] - crossings[first])
    assert torch.equal(rows, torch.tensor([0]))
    assert torch.allclose(
        adjustments, torch.tensor(expected, dtype=torch.float32)[None], atol=1e-5)
