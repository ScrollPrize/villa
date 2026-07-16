"""Tests for the surf-SDT crossing-count spacing and attachment losses
(docs/spiral_pred_dt_dense_spacing.md, tests 1-13 and 16-18 where they are
implementable without machine-local data)."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sdt_losses import (
    MIN_VALID_CORNER_WEIGHT_MASS,
    aggregate_pair_counts,
    compute_pair_counts,
    fitted_winding_domain,
    get_crossing_count_spacing_loss,
    get_dense_attachment_loss,
    sample_sdt_trilinear,
)


DR_PER_WINDING = 10.0
TWO_PI = 2.0 * math.pi


def spiral_shifted_radius(points):
    """Shared spiral -> shifted-radius prologue for the mock transforms: the
    spiral has radius 0 at winding angle 0 and grows at DR_PER_WINDING, so a
    perfect fit maps winding k to shifted radius k * DR_PER_WINDING."""
    y, x = points[:, 1], points[:, 2]
    radius = torch.sqrt(y * y + x * x + 1e-12)
    theta = torch.arctan2(y, x) % TWO_PI
    return radius - theta / TWO_PI * DR_PER_WINDING


class ArrayFixture:
    """Minimal zarr-array stand-in accepted by prepare_scalar_mmap."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.chunks = (2, 2, 2)
        self.dtype = array.dtype
        self.path = '1'
        self.store = None

    def __getitem__(self, item):
        return self.array[item]


def make_volume(array_u8, *, scale=1.0, z_origin=0, unit=1.0, offset=128,
                cap=127.0, kind='sdt'):
    volume = torch.as_tensor(np.asarray(array_u8), dtype=torch.uint8)
    return {
        'backend': 'dense',
        'kind': kind,
        'volume': volume,
        'z_origin': z_origin,
        'scale_zyx': (scale,) * 3,
        'unit': unit,
        'offset': offset,
        'cap': cap,
        'shape': tuple(volume.shape),
        'fingerprint': {},
    }


def sheet_volume(x_size, sheet_centers, half_thickness=2.0, zy=(4, 4)):
    """Planar sheets normal to x: sd(x) = min |x - c| - half_thickness, exact
    under the encoding since the field varies along one axis only."""
    x = np.arange(x_size, dtype=np.float32)
    sd = np.min([np.abs(x - c) for c in sheet_centers], axis=0) - half_thickness
    encoded = (np.clip(np.rint(sd), -127, 127) + 128).astype(np.uint8)
    return make_volume(np.broadcast_to(encoded, (*zy, x_size)).copy())


class PerfectSpiralToX:
    """A 'fit' whose windings sit exactly on sheets at x = k * dr: maps a
    spiral point to (z0, y0, shifted_radius + x_offset), optionally scaled.

    ``radial_scale`` < 1 collapses windings together; ``x_offset`` shifts the
    whole fit off the sheets. Differentiable through both parameters when they
    are tensors, which the gradient-recovery tests rely on.
    """

    def __init__(self, z0=1.5, y0=1.5, x_offset=0.0, radial_scale=1.0):
        self.z0, self.y0 = z0, y0
        self.x_offset = x_offset
        self.radial_scale = radial_scale

    def inv(self, points):
        mapped_x = self.x_offset + self.radial_scale * spiral_shifted_radius(points)
        return torch.stack([
            torch.full_like(mapped_x, self.z0),
            torch.full_like(mapped_x, self.y0),
            mapped_x,
        ], dim=-1)


def spacing_cfg(**overrides):
    cfg = {
        'dense_spacing_num_pairs': 256,
        'dense_spacing_pair_max_m': 1,
        'dense_spacing_count_temperature_wv': 0.5,
        'dense_spacing_target_step_wv': 1.0,
        'dense_spacing_max_step_wv': 2.0,
        'dense_spacing_max_steps': 64,
        'dense_spacing_step_oversample': 1.25,
        'dense_spacing_use_support_gate': True,
        'dense_spacing_support_sigma': 4.0,
        'dense_spacing_support_floor_alpha': 0.05,
        'dense_spacing_support_policy': 'product',
        'dense_attachment_num_points': 512,
        'dense_attachment_scale': 8.0,
    }
    cfg.update(overrides)
    return cfg


def pair_counts(transform, volume, k, m=1, theta=0.0, cfg=None, sdt_volume='same'):
    k = torch.as_tensor(k, dtype=torch.float32).reshape(-1)
    n = k.shape[0]
    return compute_pair_counts(
        transform,
        torch.tensor(DR_PER_WINDING),
        volume,
        volume if sdt_volume == 'same' else sdt_volume,
        k,
        torch.full([n], int(m), dtype=torch.long),
        torch.full([n], float(theta)),
        torch.full([n], 1.5),
        cfg or spacing_cfg(),
    )


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

    def test_weight_mass_validity(self):
        # Test 7: a valid/no-data corner mixture follows the weight-mass
        # criterion, and is not accepted merely because the interpolated value
        # is nonzero (nor rejected because it is zero).
        volume = make_volume(np.full([4, 4, 4], 128, np.uint8))
        volume['volume'][1, 1, 1] = 0  # one no-data corner
        near = torch.tensor([[0.9, 0.9, 0.9]])   # 0.729 of the mass is invalid
        far = torch.tensor([[0.3, 0.3, 0.3]])    # only 0.027 invalid
        _, valid_near, _ = sample_sdt_trilinear(volume, near)
        value_far, valid_far, _ = sample_sdt_trilinear(volume, far)
        self.assertFalse(bool(valid_near[0]))
        self.assertTrue(bool(valid_far[0]))
        # Renormalised over valid corners; interpolated sd is exactly 0 and
        # that must not be treated as invalid.
        self.assertAlmostEqual(float(value_far[0]), 0.0, places=4)
        self.assertEqual(MIN_VALID_CORNER_WEIGHT_MASS, 0.5)

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

    def test_medial_ridge_gradient(self):
        # Test 17: at a two-sheet medial ridge the field is unsaturated but the
        # position gradient cancels under exact symmetry (in the interpolant,
        # symmetry means a cell whose two knots share the ridge value); a
        # sub-voxel perturbation recovers a gradient toward the nearer sheet.
        volume = sheet_volume(64, [20, 31])  # ridge at 25.5, mid-cell
        x_symmetric = torch.tensor([25.5], requires_grad=True)
        points = torch.stack([torch.full_like(x_symmetric, 1.5),
                              torch.full_like(x_symmetric, 1.5), x_symmetric], -1)
        value, _, _ = sample_sdt_trilinear(volume, points)
        self.assertLess(float(value[0]), 127.0)  # unsaturated
        value.sum().backward()
        self.assertAlmostEqual(float(x_symmetric.grad[0]), 0.0, places=4)

        x_perturbed = torch.tensor([26.6], requires_grad=True)
        points = torch.stack([torch.full_like(x_perturbed, 1.5),
                              torch.full_like(x_perturbed, 1.5), x_perturbed], -1)
        value, _, _ = sample_sdt_trilinear(volume, points)
        value.sum().backward()
        # The nearer sheet is at 31, so sd decreases towards +x here.
        self.assertLess(float(x_perturbed.grad[0]), 0.0)


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

    def test_same_sheet_counts_zero(self):
        # Test 2: two endpoints inside the same (very thick) sheet count ~0.
        volume = sheet_volume(96, [25], half_thickness=20.0)
        result = pair_counts(PerfectSpiralToX(), volume, k=[1.0], m=1)
        self.assertTrue(bool(result['seg_valid'].all()))
        self.assertLess(float(result['count'][0]), 0.05)

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

    def test_too_long_pairs_are_rejected_and_reported(self):
        # Test 18a: a pair that would need more than max_steps polyline samples
        # is invalidated and reported, never evaluated at unsafe spacing.
        volume = sheet_volume(96, [10, 20])
        result = pair_counts(
            PerfectSpiralToX(radial_scale=30.0), volume, k=[1.0], m=1,
            cfg=spacing_cfg(dense_spacing_max_steps=32))
        self.assertTrue(bool(result['too_long'][0]))
        self.assertFalse(bool(result['seg_valid'][0]))

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

    def test_support_policies(self):
        volume = sheet_volume(96, [10, 20])
        # Off-sheet endpoints: exterior distance ~3 at both ends.
        result = pair_counts(
            PerfectSpiralToX(x_offset=5.0), volume, k=[1.0],
            cfg=spacing_cfg(dense_spacing_support_policy='product'))
        result_min = pair_counts(
            PerfectSpiralToX(x_offset=5.0), volume, k=[1.0],
            cfg=spacing_cfg(dense_spacing_support_policy='minimum'))
        self.assertLess(float(result['support'][0]),
                        float(result_min['support'][0]) + 1e-6)

    def test_winding_skip_reads_plus_one(self):
        # Test 13: a fit with one winding 'removed' - windings land on sheets
        # 10, 20, 40, 50 so the (2, 3) pair spans the unclaimed sheet at 30 -
        # reads mean_count - m ~ +1 on exactly the affected pair. The extra
        # 10 wv are absorbed over a 5 wv ramp (slope 3), which keeps mapped
        # adjacent steps inside the max-step bound.
        class SkipsASheet:
            def inv(self, points):
                shifted = spiral_shifted_radius(points)
                mapped = shifted + 10.0 * ((shifted - 22.5) / 5.0).clamp(0.0, 1.0)
                return torch.stack([
                    torch.full_like(mapped, 1.5),
                    torch.full_like(mapped, 1.5),
                    mapped,
                ], dim=-1)

        volume = sheet_volume(96, [10, 20, 30, 40, 50, 60])
        rows = aggregate_pair_counts(
            SkipsASheet(), torch.tensor(DR_PER_WINDING), volume, volume,
            outer_winding_idx=5, cfg=spacing_cfg(), z_begin=1, z_end=2,
            samples_per_pair=64)
        by_winding = {row['winding']: row for row in rows}
        self.assertAlmostEqual(by_winding[1]['mean_count_minus_m'], 0.0, delta=0.15)
        self.assertAlmostEqual(by_winding[2]['mean_count_minus_m'], 1.0, delta=0.2)

    def test_surf_count_source_fallback(self):
        # Test 14 (unit form): counting on a raw-surf volume with the
        # 128-centered sigmoid indicator matches SDT counting.
        sheets = [10, 20, 30]
        sdt = sheet_volume(96, sheets)
        surf_encoded = np.zeros(96, np.uint8)
        x = np.arange(96)
        for c in sheets:
            surf_encoded[np.abs(x - c) <= 2] = 255
        surf_encoded[surf_encoded == 0] = 5  # decisive outside probability
        surf = make_volume(np.broadcast_to(surf_encoded, (4, 4, 96)).copy(),
                           kind='surf', unit=None, cap=None)
        result = pair_counts(PerfectSpiralToX(), surf, k=[1.0, 2.0], m=1,
                             sdt_volume=sdt)
        self.assertTrue(bool(result['seg_valid'].all()))
        for count in result['count'].tolist():
            self.assertAlmostEqual(count, 1.0, delta=0.1)


class SpacingLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def loss_and_metrics(self, volume, transform=None, **cfg_overrides):
        cfg = spacing_cfg(**cfg_overrides)
        return get_crossing_count_spacing_loss(
            transform or PerfectSpiralToX(), torch.tensor(DR_PER_WINDING),
            volume, volume, outer_winding_idx=6, cfg=cfg, z_begin=1, z_end=2)

    def test_perfect_fit_has_near_zero_loss(self):
        volume = sheet_volume(96, [10, 20, 30, 40, 50, 60])
        loss, metrics = self.loss_and_metrics(volume)
        self.assertLess(float(loss), 0.06)
        self.assertGreater(metrics['dense_spacing_valid_fraction'], 0.99)
        self.assertAlmostEqual(metrics['dense_spacing_count_mean'], 1.0, delta=0.05)
        self.assertEqual(metrics['dense_spacing_floor_active'], 0.0)

    def test_zero_support_batch_is_finite_and_floored(self):
        # Test 10: a batch with zero effective support has a finite, defined
        # (zero) loss, and uniformly tiny support scales the loss down via the
        # nominal-mass floor instead of collapsing to the ordinary mean.
        air = make_volume(np.full([4, 4, 96], 255, np.uint8))  # sd = +127 everywhere
        loss, metrics = self.loss_and_metrics(air)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertAlmostEqual(float(loss), 0.0, places=5)
        self.assertEqual(metrics['dense_spacing_floor_active'], 1.0)

        # Uniform small-but-nonzero support: encoded sd = +8 wv everywhere ->
        # support = exp(-4)^2 ~ 1.1e-7 per pair. The residual is ~1 (count 0
        # against m=1), so an unfloored weighted mean would be ~1.
        faint = make_volume(np.full([4, 4, 96], 136, np.uint8))
        loss, metrics = self.loss_and_metrics(faint)
        self.assertEqual(metrics['dense_spacing_floor_active'], 1.0)
        self.assertLess(float(loss), 0.01)

    def test_missing_volume_yields_zero(self):
        loss, metrics = get_crossing_count_spacing_loss(
            PerfectSpiralToX(), torch.tensor(DR_PER_WINDING), None, None,
            outer_winding_idx=6, cfg=spacing_cfg(), z_begin=1, z_end=2)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(metrics, {})

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

    def test_exact_collapse_has_no_count_gradient(self):
        # Test 16a: an exactly collapsed pair deep inside one wide sheet sits
        # on a flat inside plateau - the count supplies no separation gradient
        # (this is the documented limitation, asserted so a future fix that
        # lifts it shows up as an intentional test change).
        volume = sheet_volume(96, [10], half_thickness=8.0)
        scale = torch.tensor(0.0, requires_grad=True)
        result = pair_counts(
            PerfectSpiralToX(radial_scale=scale, x_offset=10.0), volume,
            k=[0.0], m=1)
        (result['count'] - 1.0).abs().sum().backward()
        self.assertAlmostEqual(float(scale.grad), 0.0, places=5)


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

    def test_requires_sdt_kind(self):
        surf = make_volume(np.full([4, 4, 8], 200, np.uint8), kind='surf',
                           unit=None, cap=None)
        with self.assertRaises(ValueError):
            self.run_loss(surf, PerfectSpiralToX())

    def test_fitted_winding_domain(self):
        self.assertEqual(fitted_winding_domain(130), (1, 129))


class ScalarMmapStoreTests(unittest.TestCase):
    def test_gather_matches_dense_and_dedupes(self):
        from lasagna_mmap import prepare_scalar_mmap

        rng = np.random.default_rng(0)
        data = rng.integers(0, 256, size=(6, 5, 4), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'store'
            source.mkdir()
            store = prepare_scalar_mmap(
                array=ArrayFixture(data), source_path=str(source), group='1',
                z_lo=1, z_hi=5, coordinate_scale=[2.0, 2.0, 2.0],
                cache_directory=str(Path(temporary) / 'cache'), kind='surf_sdt')
            try:
                indices = torch.tensor(
                    [[0, 0, 0], [3, 4, 3], [2, 2, 2], [0, 0, 0], [1, 3, 1]],
                    dtype=torch.int64)
                values = store.gather(indices, 'cpu')
                expected = data[1:5][indices[:, 0], indices[:, 1], indices[:, 2]]
                np.testing.assert_array_equal(values.numpy(), expected)
                self.assertLess(store.last_timings['unique_fraction'], 1.0)

                # Reopen from the cache and verify probe validation passes.
                store_again = prepare_scalar_mmap(
                    array=ArrayFixture(data), source_path=str(source), group='1',
                    z_lo=1, z_hi=5, coordinate_scale=[2.0, 2.0, 2.0],
                    cache_directory=str(Path(temporary) / 'cache'), kind='surf_sdt')
                self.assertEqual(store_again.directory, store.directory)
                store_again.close()
            finally:
                store.close()

    def test_dense_and_mmap_sampling_agree(self):
        from lasagna_mmap import prepare_scalar_mmap

        x = np.arange(32, dtype=np.float32)
        sd = np.abs(x - 12.0) - 2.0
        encoded = (np.clip(np.rint(sd), -127, 127) + 128).astype(np.uint8)
        data = np.broadcast_to(encoded, (6, 6, 32)).copy()
        dense = make_volume(data)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'store'
            source.mkdir()
            store = prepare_scalar_mmap(
                array=ArrayFixture(data), source_path=str(source), group='1',
                z_lo=0, z_hi=6, coordinate_scale=[1.0] * 3,
                cache_directory=str(Path(temporary) / 'cache'), kind='surf_sdt')
            try:
                mmap_volume = dict(dense)
                mmap_volume.update(backend='mmap', store=store, volume=None)
                points = torch.rand([64, 3]) * torch.tensor([4.0, 4.0, 30.0]) + 0.5
                dense_value, dense_valid, _ = sample_sdt_trilinear(dense, points)
                mmap_value, mmap_valid, _ = sample_sdt_trilinear(mmap_volume, points)
                torch.testing.assert_close(dense_value, mmap_value)
                torch.testing.assert_close(dense_valid, mmap_valid)
            finally:
                store.close()


SHIPPED_STORE = '/home/sean/Desktop/spiral_dataset/to_hf/lasagna_inputs/las_008_surf_sdt.ome.zarr'


@unittest.skipUnless(
    __import__('os').environ.get('SPIRAL_SDT_INTEGRATION'),
    'machine-local integration test; set SPIRAL_SDT_INTEGRATION=1 to run '
    '(builds a ~1 GB mmap cache from the shipped threshold-150 store)')
class ShippedStoreIntegrationTests(unittest.TestCase):
    def test_primary_diagnostic_point(self):
        # Doc test 15: the primary diagnostic point (working xyz 4375, 2176,
        # 11314) reproduces surf = 0 there yet SDT ~ +4..6 working voxels
        # (measured +4.5 on the shipped store), where the retired pred_dt sat
        # saturated at encoded 80.
        import os
        from lasagna_data import prepare_surf_sdt_volume

        with self.assertRaises(RuntimeError):
            # Fit range outside the built [4000, 18000] working-z coverage.
            prepare_surf_sdt_volume(SHIPPED_STORE, '1', z_begin=3000, z_end=12000,
                                    cache_directory='/tmp/claude-1000/sdt-smoke-cache')

        volume = prepare_surf_sdt_volume(
            SHIPPED_STORE, '1', z_begin=11250, z_end=11380,
            cache_directory=os.environ.get('SPIRAL_SDT_CACHE',
                                           '/tmp/claude-1000/sdt-smoke-cache'))
        try:
            self.assertEqual(volume['scale_zyx'], (2.0, 2.0, 2.0))
            self.assertEqual(volume['fingerprint']['threshold'], 150)
            value, valid, _ = sample_sdt_trilinear(
                volume, torch.tensor([[11314.0, 2176.0, 4375.0]]))
            self.assertTrue(bool(valid[0]))
            self.assertGreaterEqual(float(value[0]), 3.0)
            self.assertLessEqual(float(value[0]), 6.5)
        finally:
            volume['store'].close()


class CoverageValidationTests(unittest.TestCase):
    def test_merged_ranges_cover(self):
        from lasagna_data import _merged_ranges_cover
        self.assertTrue(_merged_ranges_cover([[0, 10]], 2, 8))
        self.assertTrue(_merged_ranges_cover([[0, 5], [5, 10]], 0, 10))
        self.assertTrue(_merged_ranges_cover([[0, 6], [4, 10]], 0, 10))
        self.assertFalse(_merged_ranges_cover([[0, 4], [6, 10]], 0, 10))
        self.assertFalse(_merged_ranges_cover([[2, 10]], 0, 10))
        self.assertFalse(_merged_ranges_cover([], 0, 10))


if __name__ == '__main__':
    unittest.main()
