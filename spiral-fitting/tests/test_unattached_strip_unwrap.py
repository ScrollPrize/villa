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
from dt_targets import compute_strip_dt_target_cache
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


class RecordingIdentityTransform(IdentityTransform):
    def __init__(self):
        self.forward_inputs = []

    def __call__(self, zyxs):
        self.forward_inputs.append(zyxs.detach().clone())
        return zyxs


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


def _flat_bundle(zyxs_list, radial_offsets=None):
    lengths = np.array([len(z) for z in zyxs_list], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    flat = {
        'zyxs': torch.from_numpy(np.concatenate(zyxs_list, axis=0)),
        'windings': torch.zeros(int(starts[-1]), dtype=torch.float32),
        'starts_cpu': torch.from_numpy(starts),
        'total': int(starts[-1]),
    }
    if radial_offsets is not None:
        flat['radial_offsets'] = torch.from_numpy(np.concatenate([
            np.full(len(z), float(o), dtype=np.float32)
            for z, o in zip(zyxs_list, radial_offsets)]))
    return flat


def _strips(zyxs_list):
    return [
        {'id': i, 'name': f'strip{i}', 'source_file': None,
         'zyxs': z, 'windings': np.zeros(len(z), dtype=np.float32),
         'link_points': {}}
        for i, z in enumerate(zyxs_list)
    ]


def _run_losses(
        zyxs_list, cfg, num_steps=25, compute_dt=True, seed=0,
        num_points_per_pcl=None, transform=None, whole_object_cache=False,
        components=None, component_edges=None, num_pcls_per_step=None,
        radial_offsets=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    strips = _strips(zyxs_list)
    flat = _flat_bundle(zyxs_list, radial_offsets)
    crossing_map = ThetaCrossingMap('cpu')
    node_start = crossing_map.register_nodes(
        flat['total'], lambda lo, hi: flat['zyxs'][lo:hi])
    starts = flat['starts_cpu'].numpy()
    for strip_idx, strip in enumerate(strips):
        ids = node_start + np.arange(starts[strip_idx], starts[strip_idx + 1])
        strip['_theta_node_ids'] = ids
        crossing_map.register_edges(np.stack([ids[:-1], ids[1:]], axis=1))
    if components is None:
        components = [[i] for i in range(len(zyxs_list))]
    if component_edges is None:
        component_edges = [[] for _ in components]
    for edges in component_edges:
        junctions = [
            (strips[a]['_theta_node_ids'][pos_a],
             strips[b]['_theta_node_ids'][pos_b])
            for a, pos_a, b, pos_b in edges
        ]
        if junctions:
            crossing_map.register_edges(junctions)
    transform = transform or IdentityTransform()
    crossing_map.force_refresh(transform)
    strata = build_pcl_sampling_strata(
        ['fibers'] * len(components), cfg,
        member_weights=[len(members) for members in components])
    dr = torch.tensor(DR)
    dt_target_cache = None
    if whole_object_cache:
        dt_target_cache = compute_strip_dt_target_cache(
            transform, dr, flat['zyxs'], flat['starts_cpu'],
            windings=flat['windings'], num_points_per_strip=512,
            max_stride=128, radial_offsets=flat.get('radial_offsets'))
    if num_points_per_pcl is None:
        num_points_per_pcl = cfg[
            'sample_count_unattached_pcl_points_per_step']
    if num_pcls_per_step is None:
        num_pcls_per_step = len(zyxs_list)
    # Counting/recording transforms measure the loss evaluations only, not
    # the topology refresh or the whole-object cache build.
    for attribute in ('forward_counts', 'inverse_counts', 'forward_inputs'):
        if hasattr(transform, attribute):
            getattr(transform, attribute).clear()
    radius_losses, dt_losses = [], []
    for _ in range(num_steps):
        radius_loss, dt_loss = get_unattached_pcl_strip_losses(
            transform, dr, strips, components, component_edges, strata,
            lambda _strips, _device: flat,
            num_pcls_per_step, num_points_per_pcl,
            compute_dt=compute_dt, dt_target_cache=dt_target_cache,
            crossing_map=crossing_map, cfg=cfg,
        )
        radius_losses.append(float(radius_loss))
        dt_losses.append(float(dt_loss))
    return np.array(radius_losses), np.array(dt_losses)


def test_short_unequal_strips_carry_one_dt_anchor_per_row():
    cfg = _make_cfg()
    base = _perfect_spiral_fiber(1.0)
    transform = CountingIdentityTransform()
    radius_losses, dt_losses = _run_losses(
        [base[:3], base[:5]], cfg, num_steps=1,
        num_points_per_pcl=1024, transform=transform)

    assert transform.forward_counts == [10]
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
    real_helper = losses.endpoint_strip_dt_target_in_sample_frame

    def capture_mask(*args, **kwargs):
        captured_counts.append(kwargs['sample_mask'].sum(dim=-1).tolist())
        return real_helper(*args, **kwargs)

    monkeypatch.setattr(
        losses, 'endpoint_strip_dt_target_in_sample_frame', capture_mask)
    radius_losses, dt_losses = _run_losses(
        [short, long], cfg, num_steps=1,
        num_points_per_pcl=1024, transform=transform)

    assert sorted(captured_counts[0]) == [3, 1024]
    assert transform.forward_counts == [1029]
    assert transform.inverse_counts == [1027]
    assert radius_losses.max() < 1e-3
    assert dt_losses.max() < 1e-3


def test_endpoint_cache_carries_anchor_without_changing_loss_samples():
    cfg = _make_cfg()
    fiber = _perfect_spiral_fiber(10.0)
    median_transform = RecordingIdentityTransform()
    endpoint_transform = RecordingIdentityTransform()

    median_losses = _run_losses(
        [fiber], cfg, num_steps=1, seed=17, num_points_per_pcl=32,
        transform=median_transform)
    endpoint_losses = _run_losses(
        [fiber], cfg, num_steps=1, seed=17, num_points_per_pcl=32,
        transform=endpoint_transform, whole_object_cache=True)

    assert median_transform.forward_inputs[0].shape == (33, 3)
    assert endpoint_transform.forward_inputs[0].shape == (33, 3)
    # Both modes carry the same endpoint and random loss positions.
    torch.testing.assert_close(
        endpoint_transform.forward_inputs[0],
        median_transform.forward_inputs[0], rtol=0, atol=0)
    np.testing.assert_allclose(endpoint_losses[0], median_losses[0], atol=1e-6)
    np.testing.assert_allclose(endpoint_losses[1], median_losses[1], atol=1e-6)


def test_endpoint_cache_adds_one_forward_point_but_no_inverse_loss_point():
    cfg = _make_cfg()
    base = _perfect_spiral_fiber(1.0)
    transform = CountingIdentityTransform()
    radius_losses, dt_losses = _run_losses(
        [base[:3], base[:5]], cfg, num_steps=1,
        num_points_per_pcl=1024, transform=transform,
        whole_object_cache=True)

    assert transform.forward_counts == [10]
    assert transform.inverse_counts == [8]
    assert radius_losses.max() < 1e-3
    assert dt_losses.max() < 1e-3


def test_endpoint_cache_handles_both_component_walk_directions(monkeypatch):
    cfg = _make_cfg()
    cfg['loss_fiber_link_branch_probability'] = 1.0
    fiber = _perfect_spiral_fiber(8.0)
    split = len(fiber) // 2
    strips = [fiber[:split + 1], fiber[split:]]
    components = [[0, 1]]
    component_edges = [[(0, len(strips[0]) - 1, 1, 0)]]
    observed_sides = []
    real_helper = losses.endpoint_strip_dt_target_in_sample_frame

    def capture_endpoint(*args, **kwargs):
        observed_sides.extend(bool(v) for v in args[6])
        return real_helper(*args, **kwargs)

    monkeypatch.setattr(
        losses, 'endpoint_strip_dt_target_in_sample_frame', capture_endpoint)
    for seed in range(12):
        radius_losses, dt_losses = _run_losses(
            strips, cfg, num_steps=1, seed=seed, num_points_per_pcl=32,
            whole_object_cache=True,
            components=components, component_edges=component_edges,
            num_pcls_per_step=1)
        assert radius_losses.max() < 1e-3
        assert dt_losses.max() < 1e-3

    assert set(observed_sides) == {False, True}


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


def _radially_displaced(fiber, delta):
    # Move every point of a spiral fiber outward by `delta` voxels along its
    # own radial direction (the sheet normal for a near-circular winding).
    out = fiber.copy()
    r = np.linalg.norm(fiber[:, 1:], axis=1, keepdims=True)
    out[:, 1:] = fiber[:, 1:] * (1.0 + delta / r)
    return out


def test_back_face_vertical_strip_is_satisfied_with_its_radial_offset():
    # A fiber 4 voxels outside a perfect winding (on the sheet's back face) is
    # off the sheet by far more than the 2.5%-of-a-winding hinge margins
    # (0.3 vx at DR=12): without an offset the DT term pulls it in, with
    # radial_offsets=4 both terms are zero, in strip-median and whole-object
    # DT target modes alike.
    cfg = _make_cfg()
    fiber = _radially_displaced(_perfect_spiral_fiber(3.0), 4.0)
    _, dt_without = _run_losses([fiber], cfg, num_steps=5)
    assert dt_without.min() > 1.0
    for whole_object_cache in (False, True):
        radius_with, dt_with = _run_losses(
            [fiber], cfg, num_steps=5, radial_offsets=[4.0],
            whole_object_cache=whole_object_cache)
        assert radius_with.max() < 1e-3
        assert dt_with.max() < 1e-3


class RadialScaleTransform(IdentityTransform):
    """Scroll -> spiral map that scales the yx plane about the axis by `scale`
    (z untouched): a uniform radial stretch, so a scroll distance d along the
    sheet normal is d * scale in spiral radius."""

    def __init__(self, scale):
        self.scale = float(scale)

    def __call__(self, zyxs):
        out = zyxs.clone()
        out[..., 1:] = out[..., 1:] * self.scale
        return out

    def inv(self, spiral_zyxs):
        out = spiral_zyxs.clone()
        out[..., 1:] = out[..., 1:] / self.scale
        return out


def test_radial_offset_is_a_scroll_distance_under_a_stretching_transform():
    # The sheet is a perfect winding in spiral space; the transform stretches
    # the yx plane by 2, so in scroll space the fiber is half as far out and a
    # back-face fiber 4 scroll voxels outside the sheet lands 8 spiral units
    # out. A constant spiral-space offset of 4 would leave a 4-unit residual
    # (far beyond the hinge margins); the physical offset must read ~zero in
    # strip-median and whole-object DT target modes alike.
    cfg = _make_cfg()
    scale = 2.0
    transform = RadialScaleTransform(scale)
    sheet_scroll = transform.inv(torch.from_numpy(_perfect_spiral_fiber(3.0)))
    fiber = _radially_displaced(sheet_scroll.numpy(), 4.0)
    _, dt_without = _run_losses(
        [fiber], cfg, num_steps=5, transform=transform)
    assert dt_without.min() > 1.0
    for whole_object_cache in (False, True):
        radius_with, dt_with = _run_losses(
            [fiber], cfg, num_steps=5, radial_offsets=[4.0],
            transform=transform, whole_object_cache=whole_object_cache)
        assert radius_with.max() < 1e-3
        assert dt_with.max() < 1e-3
    # The same fiber, read with a 4-unit constant, sits a whole 4 spiral
    # units off: the stretch is what makes the offset land.
    _, dt_constant = _run_losses(
        [_radially_displaced(sheet_scroll.numpy(), 4.0 / scale)], cfg,
        num_steps=5, radial_offsets=[4.0], transform=transform)
    assert dt_constant.min() > 1.0


class AnisotropicScaleTransform(IdentityTransform):
    """Scroll -> spiral map scaling y by `sy` and x by `sx` about the axis, so
    the scan-space sheet normal (the pulled-back winding gradient) and the
    line to the umbilicus point different ways off the axes."""

    def __init__(self, sy, sx):
        self.sy, self.sx = float(sy), float(sx)

    def __call__(self, zyxs):
        out = zyxs.clone()
        out[..., 1] = out[..., 1] * self.sy
        out[..., 2] = out[..., 2] * self.sx
        return out

    def inv(self, spiral_zyxs):
        out = spiral_zyxs.clone()
        out[..., 1] = out[..., 1] / self.sy
        out[..., 2] = out[..., 2] / self.sx
        return out


def test_offset_direction_is_the_winding_gradient_not_the_umbilicus_line():
    # Under an anisotropic map the fitted sheet in scan space is an ellipse-
    # like curve whose normal is J^T n, the scan-space gradient of the fitted
    # winding. A fiber 4 scroll voxels off the sheet along J^T n (increasing
    # winding) is exactly what offset=4 expects; the same 4 voxels along the
    # straight line to the umbilicus is not on the expected surface.
    cfg = _make_cfg()
    sy, sx = 1.0, 3.0
    transform = AnisotropicScaleTransform(sy, sx)
    sheet_spiral = torch.from_numpy(_perfect_spiral_fiber(3.0))
    sheet_scroll = transform.inv(sheet_spiral)
    n = torch.nn.functional.normalize(sheet_spiral[:, 1:], dim=-1)  # spiral radial
    gradient = torch.stack([n[:, 0] * sy, n[:, 1] * sx], dim=-1)  # J^T n (yx)
    gradient = torch.nn.functional.normalize(gradient, dim=-1)
    radial = torch.nn.functional.normalize(sheet_scroll[:, 1:], dim=-1)
    # The two directions genuinely differ off the axes.
    assert float((gradient * radial).sum(-1).min()) < 0.9

    along_gradient = sheet_scroll.clone()
    along_gradient[:, 1:] += 4.0 * gradient
    along_umbilicus_line = sheet_scroll.clone()
    along_umbilicus_line[:, 1:] += 4.0 * radial
    for whole_object_cache in (False, True):
        radius_loss, dt_loss = _run_losses(
            [along_gradient.numpy()], cfg, num_steps=5, radial_offsets=[4.0],
            transform=transform, whole_object_cache=whole_object_cache)
        assert radius_loss.max() < 1e-3
        assert dt_loss.max() < 1e-3
    radius_loss, dt_loss = _run_losses(
        [along_umbilicus_line.numpy()], cfg, num_steps=5,
        radial_offsets=[4.0], transform=transform)
    assert radius_loss.max() > 0.1


def test_zero_offsets_add_no_transform_evaluations():
    # The stretch estimate costs six extra transform evaluations per sampled
    # point; a bundle whose offsets are all zero must not pay for it.
    cfg = _make_cfg()
    base = _perfect_spiral_fiber(1.0)
    transform = CountingIdentityTransform()
    _run_losses(
        [base[:3], base[:5]], cfg, num_steps=1, num_points_per_pcl=1024,
        transform=transform, radial_offsets=[0.0, 0.0])
    assert transform.forward_counts == [10]


def test_offset_strip_is_satisfied_under_a_stretching_transform():
    # Under a 2.5x radial stretch a back-face fiber 4 scroll voxels outside
    # the sheet is 10 spiral units out; the physical offset of 4 puts every
    # point on the sheet. A fiber only 4 *spiral* units out (what a constant
    # spiral-space offset of 4 would have expected) reads 6 units inside its
    # winding with the same offset -- the largest possible residual, outside
    # the 0.45*DR satisfaction band -- so none of it is satisfied.
    from fit_spiral import _build_strip_flat_bundle
    from satisfaction_metrics import get_unattached_pcl_satisfied_counts
    scale = 2.5
    transform = RadialScaleTransform(scale)
    sheet_scroll = transform.inv(torch.from_numpy(_perfect_spiral_fiber(2.0)))
    physical = _radially_displaced(sheet_scroll.numpy(), 4.0)
    constant = _radially_displaced(sheet_scroll.numpy(), 4.0 / scale)
    for fiber, expect_all in ((physical, True), (constant, False)):
        strips = _strips([fiber])
        flat = _build_strip_flat_bundle(
            [(fiber, np.zeros(len(fiber), np.float32),
              np.full(len(fiber), 4.0, np.float32))],
            torch.device('cpu'))
        satisfied, total, _ = get_unattached_pcl_satisfied_counts(
            transform, torch.tensor(DR), strips, lambda _s, _d: flat)
        assert int(total[0]) == len(fiber)
        assert int(satisfied[0]) == (len(fiber) if expect_all else 0)


def test_radial_offset_only_moves_the_offset_strip():
    # Mixed row of an on-sheet horizontal (offset 0) and a back-face vertical
    # (offset 4) walked through a junction: both read as the same winding.
    cfg = _make_cfg()
    horizontal = _perfect_spiral_fiber(1.0)
    vertical = _radially_displaced(_perfect_spiral_fiber(1.0, theta0=0.3), 4.0)
    radius_losses, dt_losses = _run_losses(
        [horizontal, vertical], cfg, num_steps=5,
        radial_offsets=[0.0, 4.0],
        components=[[0, 1]], component_edges=[[(0, 1, 1, 0)]])
    assert radius_losses.max() < 1e-3
    assert dt_losses.max() < 1e-3
