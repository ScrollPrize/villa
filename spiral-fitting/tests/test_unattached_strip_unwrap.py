"""Regression tests for the unattached-strip loss's theta=0 unwrap on strips
that span multiple windings (long fibers).

The loss samples only num_points_per_pcl points per row; on a fiber wrapping
several turns, consecutive samples can sit more than pi apart in theta, where
a consecutive-diff crossing detector miscounts seam crossings and assigns
wrong winding offsets even to a perfectly-fit strip. The crossing map caches
the full source topology; these tests pin
that a geometrically perfect spiral fiber yields ~zero radius and DT loss
regardless of how many windings it spans."""

import numpy as np
import torch

import losses
from config import Config
from losses import (
    build_pcl_sampling_strata,
    get_unattached_pcl_strip_losses,
)
from theta_crossing_map import ThetaCrossingMap

DR = 12.0


class IdentityTransform:
    def __call__(self, zyxs):
        return zyxs

    def inv(self, spiral_zyxs):
        return spiral_zyxs


class CountingIdentityTransform(IdentityTransform):
    def __init__(self):
        self.forward_counts = []
        self.inverse_counts = []

    def __call__(self, zyxs):
        self.forward_counts.append(len(zyxs))
        return zyxs

    def inv(self, spiral_zyxs):
        self.inverse_counts.append(len(spiral_zyxs))
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


def _run_losses(
        zyxs_list, cfg, num_steps=25, compute_dt=True, seed=0,
        num_points_per_pcl=None, transform=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    strips = _strips(zyxs_list)
    flat = _flat_bundle(zyxs_list)
    crossing_map = ThetaCrossingMap('cpu')
    node_start = crossing_map.register_nodes(
        flat['total'], lambda lo, hi: flat['zyxs'][lo:hi])
    starts = flat['starts_cpu'].numpy()
    for strip_idx, strip in enumerate(strips):
        ids = node_start + np.arange(starts[strip_idx], starts[strip_idx + 1])
        strip['_theta_node_ids'] = ids
        crossing_map.register_edges(np.stack([ids[:-1], ids[1:]], axis=1))
    crossing_map.force_refresh(IdentityTransform())
    strata = build_pcl_sampling_strata(
        ['fibers'] * len(zyxs_list), cfg,
        member_weights=[1] * len(zyxs_list))
    components = [[i] for i in range(len(zyxs_list))]
    edges = [[] for _ in zyxs_list]
    dr = torch.tensor(DR)
    transform = transform or IdentityTransform()
    if num_points_per_pcl is None:
        num_points_per_pcl = cfg[
            'sample_count_unattached_pcl_points_per_step']
    radius_losses, dt_losses = [], []
    for _ in range(num_steps):
        radius_loss, dt_loss = get_unattached_pcl_strip_losses(
            transform, dr, strips, components, edges, strata,
            lambda _strips, _device: flat,
            len(zyxs_list), num_points_per_pcl,
            compute_dt=compute_dt, crossing_map=crossing_map, cfg=cfg,
        )
        radius_losses.append(float(radius_loss))
        dt_losses.append(float(dt_loss))
    return np.array(radius_losses), np.array(dt_losses)


def test_short_unequal_strips_transform_each_available_point_once():
    cfg = _make_cfg()
    base = _perfect_spiral_fiber(1.0)
    transform = CountingIdentityTransform()
    radius_losses, dt_losses = _run_losses(
        [base[:3], base[:5]], cfg, num_steps=1,
        num_points_per_pcl=1024, transform=transform)

    assert transform.forward_counts == [8]
    assert transform.inverse_counts == [8]
    assert radius_losses.max() < 1e-3
    assert dt_losses.max() < 1e-3


def test_mixed_short_and_long_strips_use_independent_caps(monkeypatch):
    cfg = _make_cfg()
    short = _perfect_spiral_fiber(1.0)[:3]
    long = _perfect_spiral_fiber(10.0)
    assert len(long) > 1024
    transform = CountingIdentityTransform()
    captured_counts = []
    real_helper = losses.strip_dt_target_in_sample_frame

    def capture_mask(*args, **kwargs):
        captured_counts.append(kwargs['sample_mask'].sum(dim=-1).tolist())
        return real_helper(*args, **kwargs)

    monkeypatch.setattr(
        losses, 'strip_dt_target_in_sample_frame', capture_mask)
    radius_losses, dt_losses = _run_losses(
        [short, long], cfg, num_steps=1,
        num_points_per_pcl=1024, transform=transform)

    assert sorted(captured_counts[0]) == [3, 1024]
    assert transform.forward_counts == [1027]
    assert transform.inverse_counts == [1027]
    assert radius_losses.max() < 1e-3
    assert dt_losses.max() < 1e-3


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
