import math
from types import SimpleNamespace
import unittest

import torch

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


if __name__ == '__main__':
    unittest.main()
