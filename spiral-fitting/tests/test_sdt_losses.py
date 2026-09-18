"""Tests for the surf-SDT crossing-count spacing and attachment losses
(docs/spiral_pred_dt_dense_spacing.md, tests 1-13 and 16-18 where they are
implementable without machine-local data)."""

import math
import unittest
import numpy as np
import torch
from sdt_losses import (
    _sample_spacing_pairs,
    compute_pair_counts,
    get_dense_attachment_loss,
    iter_phase_bundle_losses,
    sample_sdt_trilinear,
)
from sdt_fixtures import (
    DR_PER_WINDING, PerfectSpiralToX, make_volume, sheet_volume,
    spacing_cfg, spiral_shifted_radius,
)


def pair_counts(transform, volume, k, m=1, theta=0.0, cfg=None):
    k = torch.as_tensor(k, dtype=torch.float32).reshape(-1)
    n = k.shape[0]
    return compute_pair_counts(
        transform,
        torch.tensor(DR_PER_WINDING),
        volume,
        k,
        torch.full([n], int(m), dtype=torch.long),
        torch.full([n], float(theta)),
        torch.full([n], 1.5),
        cfg or spacing_cfg(),
    )


def run_bundle_count(transform, volume, outer_winding_idx=6, cfg=None,
                     generator=None):
    """Run the count component of the phase bundle (phase disabled via a
    missing normal store) and return (loss, metrics)."""
    cfg = cfg or spacing_cfg()
    for name, loss, metrics in iter_phase_bundle_losses(
            None, transform, torch.tensor(DR_PER_WINDING), volume, None,
            outer_winding_idx, cfg, 1, 2, generator=generator):
        if name == 'dense_spacing_count':
            return loss, metrics
    return torch.zeros([]), {}


class TrilinearSamplingTests(unittest.TestCase):
    def test_decode_values(self):
        # Test 8: values 0, 1, 128, 255 -> no-data, -127, 0, +127, honoring
        # store-attribute scale/offset.
        volume = make_volume(np.full([4, 4, 8], 128, np.uint8))
        volume['volume'][1, 1, 1] = 0
        volume['volume'][1, 1, 3] = 1
        volume['volume'][1, 1, 5] = 255
        points = torch.tensor([
            [1., 1., 1.], [1., 1., 3.], [1., 1., 5.], [2., 2., 2.],
        ])
        value, valid, _ = sample_sdt_trilinear(volume, points)
        self.assertFalse(bool(valid[0]))  # all weight on a no-data corner
        self.assertTrue(bool(valid[1]) and bool(valid[2]) and bool(valid[3]))
        self.assertAlmostEqual(float(value[1]), -127.0, places=4)
        self.assertAlmostEqual(float(value[2]), 127.0, places=4)
        self.assertAlmostEqual(float(value[3]), 0.0, places=4)

    def test_scale_and_unit_are_honoured(self):
        # Group-1 convention: 2 working voxels per stored grid voxel, and a
        # non-unit encoding must decode through the store's own attributes.
        volume = make_volume(np.full([4, 4, 4], 130, np.uint8), scale=2.0, unit=0.5)
        value, valid, _ = sample_sdt_trilinear(volume, torch.tensor([[4., 4., 4.]]))
        self.assertTrue(bool(valid[0]))  # working (4,4,4) -> grid (2,2,2)
        self.assertAlmostEqual(float(value[0]), (130 - 128) * 0.5, places=4)

    def test_trilinear_is_smooth_and_differentiable(self):
        # Test 6: values change smoothly as mapped points cross a voxel, and
        # gradient flows through the fractional weights.
        ramp = np.tile(np.arange(8, dtype=np.uint8)[None, None, :] * 4 + 100, (4, 4, 1))
        volume = make_volume(ramp)
        xs = torch.linspace(1.0, 5.0, 41, requires_grad=True)
        points = torch.stack([torch.full_like(xs, 1.5), torch.full_like(xs, 1.5), xs], -1)
        value, valid, _ = sample_sdt_trilinear(volume, points)
        self.assertTrue(bool(valid.all()))
        diffs = value.diff()
        self.assertTrue(bool((diffs > 0).all()))  # monotone ramp stays monotone
        self.assertLess(float(diffs.max() - diffs.min()), 1e-3)  # and uniform
        value.sum().backward()
        self.assertTrue(bool((xs.grad.abs() > 0).all()))


class CrossingCountTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_center_to_center_counts_m(self):
        # Tests 1 and 5: sheet center to the m-th neighbour counts m, with the
        # sampled pair anchored on integer windings of the fitted domain.
        volume = sheet_volume(96, [10, 20, 30, 40, 50, 60])
        transform = PerfectSpiralToX()
        for m in (1, 2, 3):
            result = pair_counts(transform, volume, k=[1.0, 2.0], m=m,
                                 theta=1.0)
            self.assertTrue(bool(result['seg_valid'].all()))
            # The soft indicator never quite reaches 0/1, so each crossing
            # contributes slightly under 1 (~0.96-0.98 at s_count = 0.5,
            # matching the GT-measured 0.96-0.98 * m means).
            for count in result['count'].tolist():
                self.assertAlmostEqual(count, m, delta=0.02 + 0.04 * m)

    def test_oblique_crossing_is_topological(self):
        # Test 3: count through a flat sheet is invariant to path obliquity -
        # a stretched mapping crosses the same one sheet and still counts ~1.
        volume = sheet_volume(96, [10, 20])
        straight = pair_counts(PerfectSpiralToX(), volume, k=[1.0], m=1)
        # radial_scale 1 with an offset produces an oblique/stretched path in
        # working space via a longer polyline; counts must agree.
        oblique = pair_counts(
            PerfectSpiralToX(radial_scale=1.0, x_offset=0.0), volume, k=[1.0],
            m=1, theta=2.5)
        self.assertAlmostEqual(float(straight['count'][0]),
                               float(oblique['count'][0]), delta=0.08)

    def test_thin_sheet_not_stepped_over(self):
        # Test 12: a 4-voxel sheet is caught at the ~1 working-voxel step.
        volume = sheet_volume(96, [10, 20, 30], half_thickness=2.0)
        result = pair_counts(PerfectSpiralToX(), volume, k=[1.0, 2.0], m=1)
        for count in result['count'].tolist():
            self.assertAlmostEqual(count, 1.0, delta=0.08)

    def test_step_violation_detected_when_chord_underestimates(self):
        # Test 18b: a mapping whose polyline is much longer than its endpoint
        # chord produces adjacent mapped steps above the max and is rejected.
        class ZigZag:
            def inv(self, points):
                shifted = spiral_shifted_radius(points)
                return torch.stack([
                    torch.full_like(shifted, 1.5),
                    1.5 + 20.0 * torch.sin(shifted * 8.0),
                    shifted,
                ], dim=-1)

        volume = make_volume(np.full([4, 64, 96], 130, np.uint8))
        result = pair_counts(ZigZag(), volume, k=[1.0], m=1)
        self.assertTrue(bool(result['step_violation'][0]))
        self.assertFalse(bool(result['seg_valid'][0]))

    def test_multi_winding_path_allocates_steps_per_gap(self):
        # A single endpoint chord sees only the average stretch over this
        # three-winding path. Uniform radial samples based on that chord make
        # >2-wv jumps through the 100-wv middle gap; per-gap allocation keeps
        # the path valid without oversampling both short neighbours.
        class UnevenGaps:
            def inv(self, points):
                winding = spiral_shifted_radius(points) / DR_PER_WINDING
                mapped_x = torch.where(
                    winding < 2.0,
                    10.0 + 5.0 * (winding - 1.0),
                    torch.where(
                        winding < 3.0,
                        15.0 + 100.0 * (winding - 2.0),
                        115.0 + 5.0 * (winding - 3.0),
                    ),
                )
                return torch.stack([
                    torch.full_like(mapped_x, 1.5),
                    torch.full_like(mapped_x, 1.5),
                    mapped_x,
                ], dim=-1)

        volume = make_volume(np.full([4, 4, 140], 130, np.uint8))
        result = pair_counts(
            UnevenGaps(), volume, k=[1.0], m=3,
            cfg=spacing_cfg(dense_spacing_max_steps=160))
        self.assertFalse(bool(result['too_long'][0]))
        self.assertFalse(bool(result['step_violation'][0]))
        self.assertTrue(bool(result['seg_valid'][0]))

    def test_partial_coverage_invalidates_whole_segment(self):
        # Whole-segment gating: a no-data band along the path invalidates the
        # pair instead of undercounting it.
        volume = sheet_volume(96, [10, 20])
        volume['volume'][:, :, 14:16] = 0
        result = pair_counts(PerfectSpiralToX(), volume, k=[1.0], m=1)
        self.assertFalse(bool(result['seg_valid'][0]))

    def test_support_gate_is_detached_and_cannot_be_gamed(self):
        # Test 9: support carries no gradient even when the transform does.
        volume = sheet_volume(96, [10, 20])
        offset = torch.tensor(0.0, requires_grad=True)
        result = pair_counts(PerfectSpiralToX(x_offset=offset), volume, k=[1.0])
        self.assertTrue(result['count'].requires_grad)
        self.assertFalse(result['support'].requires_grad)


class SpacingLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def loss_and_metrics(self, volume, transform=None, **cfg_overrides):
        cfg = spacing_cfg(**cfg_overrides)
        return run_bundle_count(
            transform or PerfectSpiralToX(), volume, outer_winding_idx=6,
            cfg=cfg)

    def test_perfect_fit_has_near_zero_loss(self):
        volume = sheet_volume(96, [10, 20, 30, 40, 50, 60])
        loss, metrics = self.loss_and_metrics(volume)
        self.assertLess(float(loss), 0.06)
        self.assertGreater(metrics['dense_spacing_count_valid_fraction'], 0.99)
        self.assertAlmostEqual(metrics['dense_spacing_count_mean'], 1.0, delta=0.05)
        self.assertEqual(metrics['dense_spacing_count_floor_active'], 0.0)

    def test_zero_support_batch_is_finite_and_floored(self):
        # Test 10: a batch with zero effective support has a finite, defined
        # (zero) loss, and uniformly tiny support scales the loss down via the
        # nominal-mass floor instead of collapsing to the ordinary mean.
        air = make_volume(np.full([4, 4, 96], 255, np.uint8))  # sd = +127 everywhere
        loss, metrics = self.loss_and_metrics(air)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertAlmostEqual(float(loss), 0.0, places=5)
        self.assertEqual(metrics['dense_spacing_count_floor_active'], 1.0)

        # Uniform small-but-nonzero support: encoded sd = +8 wv everywhere ->
        # support = exp(-4)^2 ~ 1.1e-7 per pair. The residual is ~1 (count 0
        # against m=1), so an unfloored weighted mean would be ~1.
        faint = make_volume(np.full([4, 4, 96], 136, np.uint8))
        loss, metrics = self.loss_and_metrics(faint)
        self.assertEqual(metrics['dense_spacing_count_floor_active'], 1.0)
        self.assertLess(float(loss), 0.01)

    def test_multi_m_pairs_stay_calibrated(self):
        # Multi-m pairs: m is drawn from the short range (clamped to the
        # winding domain), longer rays get proportionally more steps, and
        # a perfect fit still reads count ~ m for every baseline.
        volume = sheet_volume(96, [10, 20, 30, 40, 50, 60])
        loss, metrics = self.loss_and_metrics(
            volume, dense_spacing_pair_m_short=(1, 4),
            dense_spacing_max_steps=128)
        self.assertGreater(metrics['dense_spacing_count_valid_fraction'], 0.99)
        self.assertEqual(metrics['dense_spacing_count_too_long_fraction'], 0.0)
        # Soft-count conservatism scales with m (~0.96-0.98 per crossing), so
        # the residual grows with baseline but stays a few percent of m.
        self.assertLess(metrics['dense_spacing_count_residual_mean'], 0.2)
        self.assertLess(float(loss), 0.2)

    def test_shared_ray_counts_match_independent_implementation(self):
        # The bundle's shared central-ray count must equal the independent
        # compute_pair_counts implementation under identical (k, m, theta, z)
        # samples, including gradients through the transform.
        volume = sheet_volume(96, [10, 20, 30, 40, 50, 60])
        cfg = spacing_cfg(sample_count_dense_spacing_pairs=64)
        seed = 11

        offset_bundle = torch.tensor(1.5, requires_grad=True)
        loss, _ = run_bundle_count(
            PerfectSpiralToX(x_offset=offset_bundle), volume, 6, cfg,
            generator=torch.Generator().manual_seed(seed))
        loss.backward()

        offset_expected = torch.tensor(1.5, requires_grad=True)
        k, m, theta, z = _sample_spacing_pairs(
            cfg, 1, 5, 64, torch.device('cpu'), 1, 2,
            torch.Generator().manual_seed(seed))
        pair = compute_pair_counts(
            PerfectSpiralToX(x_offset=offset_expected),
            torch.tensor(DR_PER_WINDING), volume, k, m, theta, z, cfg)
        weight = (pair['support'] * pair['seg_valid'].float()).detach()
        residual = (pair['count'] - pair['target_m']).abs()
        alpha = float(cfg['dense_spacing_support_floor_alpha'])
        expected = (weight * residual).sum() / torch.maximum(
            weight.sum(), torch.tensor(alpha * 64.0))
        expected.backward()

        torch.testing.assert_close(loss.detach(), expected.detach())
        torch.testing.assert_close(offset_bundle.grad, offset_expected.grad)

    def test_gradient_recovery_from_gap(self):
        # Test 16c: with an endpoint inside a gap, the count residual supplies
        # a live gradient and optimisation reaches count ~ 1. Note it recovers
        # the *count*, not a specific winding assignment: the SDT has no
        # winding identity, so a configuration crossing a different single
        # sheet is an equally valid minimum for this term alone.
        volume = sheet_volume(96, [10, 20, 30])
        scale = torch.tensor(0.75, requires_grad=True)  # outer endpoint at x=15, mid-gap
        optimiser = torch.optim.Adam([scale], lr=0.02)
        residual_value = None
        for _ in range(200):
            optimiser.zero_grad()
            result = pair_counts(PerfectSpiralToX(radial_scale=scale), volume,
                                 k=[1.0], m=1)
            residual = (result['count'] - 1.0).abs().sum()
            residual_value = float(residual)
            if residual_value < 0.05:
                break
            residual.backward()
            optimiser.step()
        self.assertLess(residual_value, 0.1)


class AttachmentLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def run_loss(self, volume, transform, **cfg_overrides):
        return get_dense_attachment_loss(
            transform, torch.tensor(DR_PER_WINDING), volume,
            outer_winding_idx=6, cfg=spacing_cfg(**cfg_overrides),
            z_begin=1, z_end=2)

    def test_on_sheet_is_zero_and_exterior_pulls_back(self):
        # Test 11: an exterior ramp produces an attachment gradient toward the
        # sheet; on/inside the mask the penalty is zero.
        volume = sheet_volume(96, [10, 20, 30, 40, 50, 60])
        loss, _ = self.run_loss(volume, PerfectSpiralToX())
        self.assertAlmostEqual(float(loss), 0.0, places=5)

        offset = torch.tensor(4.0, requires_grad=True)
        loss, metrics = self.run_loss(volume, PerfectSpiralToX(x_offset=offset))
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        # The fit sits +4 wv outside every sheet; decreasing the offset
        # decreases the exterior distance, so the gradient is positive.
        self.assertGreater(float(offset.grad), 0.0)
        self.assertGreater(metrics['dense_attachment_live_gradient_fraction'], 0.99)

    def test_saturated_block_is_reported_gradient_free(self):
        # Test 11b: a saturated +/-127 block is recognised as gradient-free
        # and reported in diagnostics rather than mistaken for attraction.
        air = make_volume(np.full([4, 4, 96], 255, np.uint8))
        offset = torch.tensor(0.0, requires_grad=True)
        loss, metrics = self.run_loss(air, PerfectSpiralToX(x_offset=offset))
        self.assertGreater(float(loss), 0.0)  # residual exists...
        loss.backward()
        self.assertAlmostEqual(float(offset.grad), 0.0, places=6)  # ...but no gradient
        self.assertEqual(metrics['dense_attachment_live_gradient_fraction'], 0.0)
        self.assertAlmostEqual(metrics['dense_attachment_saturated_fraction'], 1.0, places=2)


SHIPPED_STORE = '/home/sean/Desktop/spiral_dataset/to_hf/lasagna_inputs/las_008_surf_sdt.ome.zarr'
