import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from dt_targets import (
    compute_patch_dt_target_cache,
    patch_dt_target_in_sample_frame,
    strip_dt_target_in_sample_frame,
)
from fit_spiral import FitContext
from losses import _patch_radius_and_dt_losses
from sample_spiral import radius_from_unwrapped_shifted, unwrap_shifted_radii
from satisfaction_metrics import get_patch_satisfied_areas, metrics_config


class _IdentityTransform:
    def __call__(self, points):
        return points

    def inv(self, points):
        return points


def _spiral_point(theta, winding, dr):
    radius = (winding + theta / (2 * math.pi)) * dr
    return torch.tensor([
        0.0,
        math.sin(theta) * radius,
        math.cos(theta) * radius,
    ], dtype=torch.float32)


def _patch_with_quad_centers(centers):
    """Build a minimal patch whose one-column quad centers equal ``centers``."""
    vertices = [centers[0]]
    for center in centers:
        vertices.append(2 * center - vertices[-1])
    zyxs = torch.stack(vertices)[:, None, :].repeat(1, 2, 1)
    return SimpleNamespace(
        zyxs=zyxs,
        valid_quad_mask=torch.ones(len(centers), 1, dtype=torch.bool),
        area=float(len(centers)),
    )


class ThetaCrossingLossTests(unittest.TestCase):
    def test_patch_target_cache_uses_theta_potential_winding_units(self):
        dr = torch.tensor(10.0)
        patch_row = SimpleNamespace(
            _dt_target_ijs=np.array([[0.25, 0.25], [0.75, 0.75]]))

        class Atlas:
            def theta_node_ids(self, patch_indices, ijs):
                return np.array([4, 5], dtype=np.int64)

            def lookup(self, patch_indices, ijs):
                return torch.stack([
                    _spiral_point(0.0, 3, float(dr)),
                    _spiral_point(0.0, 3, float(dr)),
                ])

        crossing_map = SimpleNamespace(
            winding_potentials=lambda node_ids, theta: torch.tensor([2, 2]))
        cache = compute_patch_dt_target_cache(
            _IdentityTransform(), dr, [patch_row], Atlas(), crossing_map,
            floating_threshold=0.25)

        self.assertEqual(cache['frame'], 'theta_potential')
        # Radius 3 plus theta potential 2 targets winding 5. The selector
        # already returns winding units even though dr is non-unit.
        torch.testing.assert_close(
            cache['target_relative'].to(torch.int64), torch.tensor([5]))
        target = patch_dt_target_in_sample_frame(
            torch.tensor([[51.0]]), torch.zeros((1, 1, 2)),
            torch.zeros((1, 1)), torch.zeros((1, 1)), dr, cache,
            torch.tensor([0]))
        torch.testing.assert_close(target, torch.tensor([[50.0]]))

    def test_patch_losses_ignore_padded_samples_in_both_radius_modes(self):
        dr = torch.tensor(10.0)
        theta = torch.tensor([[0.2, 1.1, 2.0, 4.0]])
        shifted = torch.tensor([[30.0, 30.0, 130.0, -70.0]])
        radii = shifted + theta / (2 * math.pi) * dr
        spiral = torch.stack([
            torch.zeros_like(theta),
            torch.sin(theta) * radii,
            torch.cos(theta) * radii,
        ], dim=-1)
        mask = torch.tensor([[True, True, False, False]])
        with patch('losses.record_loss_samples') as record:
            for inverse in (False, True):
                radius_loss, dt_loss = _patch_radius_and_dt_losses(
                    _IdentityTransform(), dr, spiral, spiral, theta, shifted,
                    torch.zeros_like(shifted), 1, 1, True, None,
                    0.0, inverse, 3.0, 0.0, 1.0, 3.0,
                    sample_mask=mask)
                self.assertLess(float(radius_loss), 2e-5)
                self.assertLess(float(dt_loss), 2e-5)
        for call in record.call_args_list:
            torch.testing.assert_close(call.args[3], mask)

    def test_patch_dt_cache_anchor_ignores_padded_points(self):
        dr = torch.tensor(10.0)
        sample_ijs = torch.tensor([[[0.0, 0.0], [10.0, 10.0], [10.0, 10.0]]])
        sample_mask = torch.tensor([[True, False, False]])
        zeros = torch.zeros((1, 3))
        cache = {
            'ijs': torch.tensor([[[0.0, 0.0], [10.0, 10.0]]]),
            'point_valid': torch.tensor([[True, True]]),
            'target_relative': torch.tensor([3.0]),
            'theta': torch.zeros((1, 2)),
            'relative_adjustment': torch.tensor([[0.0, 5.0]]),
            'valid': torch.tensor([True]),
            'anchor_dist_sq_limit': torch.tensor([1.0]),
        }
        target = patch_dt_target_in_sample_frame(
            torch.tensor([[30.0, -20.0, -20.0]]), sample_ijs, zeros, zeros,
            dr, cache, torch.tensor([0]), sample_mask=sample_mask)
        torch.testing.assert_close(target, torch.tensor([[30.0]]))

    def test_strip_dt_median_fallback_ignores_padded_points(self):
        dr = torch.tensor(10.0)
        sample_mask = torch.tensor([[True, False, False]])
        zeros = torch.zeros((1, 3))
        cache = {
            'frame': 'strip_endpoints',
            'anchor_theta': torch.zeros((1, 2)),
            'anchor_adjustment': torch.tensor([[0, 5]]),
            'target_relative': torch.tensor([3.0]),
            'valid': torch.tensor([True]),
        }
        target = strip_dt_target_in_sample_frame(
            torch.tensor([[30.0, -20.0, -20.0]]),
            torch.tensor([[4, 10, 10]]), zeros, zeros,
            dr, cache, torch.tensor([0]), sample_mask=sample_mask)
        torch.testing.assert_close(target, torch.tensor([[30.0]]))

    def test_unwrapped_target_converts_back_to_the_same_physical_winding(self):
        dr = torch.tensor(10.0)
        theta = torch.tensor([[[2 * math.pi - 0.1, 0.1]]])
        raw_shifted = torch.tensor([[[30.0, 40.0]]])

        unwrapped, adjustments = unwrap_shifted_radii(theta, raw_shifted, dr)
        torch.testing.assert_close(unwrapped, torch.full_like(unwrapped, 30.0))
        torch.testing.assert_close(
            adjustments,
            torch.tensor([[[0.0, -10.0]]]),
        )

        target_radii = radius_from_unwrapped_shifted(
            theta, torch.full_like(unwrapped, 30.0), adjustments, dr,
        )
        expected = raw_shifted + theta / (2 * math.pi) * dr
        torch.testing.assert_close(target_radii, expected)

    def test_patch_inverse_radius_and_dt_losses_are_zero_across_theta_seam(self):
        dr = torch.tensor(10.0)
        theta = torch.tensor([[[2 * math.pi - 0.1, 0.1]]])
        raw_shifted = torch.tensor([[[30.0, 40.0]]])
        unwrapped, adjustments = unwrap_shifted_radii(theta, raw_shifted, dr)
        radii = raw_shifted + theta / (2 * math.pi) * dr
        spiral = torch.stack([
            torch.zeros_like(theta),
            torch.sin(theta) * radii,
            torch.cos(theta) * radii,
        ], dim=-1)

        radius_loss, dt_loss = _patch_radius_and_dt_losses(
            _IdentityTransform(),
            dr,
            spiral,
            spiral,
            theta,
            unwrapped,
            adjustments,
            1,
            1,
            True,
            None,
            0.0,
            True,
            1.0,
            0.0,
            1.0,
            1.0,
        )

        self.assertLess(float(radius_loss), 1e-5)
        self.assertLess(float(dt_loss), 2e-5)


class ThetaCrossingSatisfactionTests(unittest.TestCase):
    def test_center_column_uses_branch_consistent_shifted_radii(self):
        dr = torch.tensor(10.0)
        patch = _patch_with_quad_centers([
            _spiral_point(2 * math.pi - 0.10, 3, float(dr)),
            _spiral_point(0.05, 4, float(dr)),
            _spiral_point(0.10, 4, float(dr)),
        ])

        satisfied, _, _, masks, _, _ = get_patch_satisfied_areas(
            _IdentityTransform(), dr, [patch], -1, 1,
        )

        self.assertTrue(bool(satisfied[0]))
        self.assertTrue(bool(masks[0].all()))

    def test_metrics_overrides_are_call_local(self):
        dr = torch.tensor(10.0)
        theta = 0.2
        radius = (3.47 + theta / (2 * math.pi)) * float(dr)
        center = torch.tensor([
            0.0,
            math.sin(theta) * radius,
            math.cos(theta) * radius,
        ], dtype=torch.float32)
        patch = _patch_with_quad_centers([center])
        original = dict(metrics_config)

        strict, *_ = get_patch_satisfied_areas(
            _IdentityTransform(), dr, [patch], -1, 1,
        )
        loose, *_ = get_patch_satisfied_areas(
            _IdentityTransform(),
            dr,
            [patch],
            -1,
            1,
            metrics_overrides={
                'satisfaction_radius_tolerance': 0.495,
                'satisfaction_distance_tolerance': 12.0,
                'satisfied_patch_quad_fraction': 0.90,
            },
        )

        self.assertFalse(bool(strict[0]))
        self.assertTrue(bool(loose[0]))
        self.assertEqual(metrics_config, original)


class ThetaCrossingCacheCadenceTests(unittest.TestCase):
    @staticmethod
    def _context():
        context = FitContext.__new__(FitContext)
        context.config = {
            'theta_crossing_map_update_interval': 100,
            'dt_target_update_interval': 100,
            'track_max_tortuosity': 0,
            'track_exclusion_radius': 0,
        }
        context.shell_map = None
        context.shell_outer_winding_idx = None
        context.shell_valid_zyxs_gpu = None
        context.tracks = []
        context.prepared_main_tracks = None
        context.verified_patches_list = []
        context.unverified_patches = None
        context.unverified_patches_list = []
        context.unverified_patch_sampling_probabilities = None
        context.unverified_patch_atlas = None
        context.dt_target_cache_manager = SimpleNamespace(
            update_interval=100, reset=Mock())
        context.theta_crossing_map = SimpleNamespace(
            invalidate=Mock(), refresh_if_due=Mock(return_value=True))
        context._enforce_theta_liftability = Mock()
        return context

    def test_live_change_through_either_alias_updates_both(self):
        for key in ('theta_crossing_map_update_interval',
                    'dt_target_update_interval'):
            context = self._context()
            context.apply_config({key: 37}, current_iteration=0)
            self.assertEqual(
                context.config['theta_crossing_map_update_interval'], 37)
            self.assertEqual(context.config['dt_target_update_interval'], 37)
            self.assertEqual(context.dt_target_cache_manager.update_interval, 37)
            context.theta_crossing_map.invalidate.assert_called_once_with()
            self.assertGreaterEqual(
                context.dt_target_cache_manager.reset.call_count, 1)

    def test_conflicting_alias_update_rolls_back_both_values(self):
        context = self._context()
        with self.assertRaisesRegex(ValueError, 'one shared cadence'):
            context.apply_config({
                'theta_crossing_map_update_interval': 10,
                'dt_target_update_interval': 11,
            }, current_iteration=0)
        self.assertEqual(
            context.config['theta_crossing_map_update_interval'], 100)
        self.assertEqual(context.config['dt_target_update_interval'], 100)

    def test_theta_refresh_invalidates_dt_targets(self):
        context = self._context()
        refreshed = context._refresh_theta_crossing_map_for_step(
            23, object())
        self.assertTrue(refreshed)
        context._enforce_theta_liftability.assert_called_once_with()
        context.dt_target_cache_manager.reset.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
