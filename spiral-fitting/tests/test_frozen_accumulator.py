"""Periodic flow reset with frozen-accumulator baking (transforms-level).

Covers the reset invariants from plans/reset_plan_a_frozen_accumulator.md:
exactness of the restructuring at reset, grid fidelity vs the exact
composition, compounding across multiple bakes, the ray-specialized chain
matcher, and the checkpoint payload round-trip.
"""

import unittest

import torch

from transforms import (
    FrozenDisplacementTransform,
    SpiralAndTransform,
    ray_specialized_spiral_to_scroll,
)


TINY_CONFIG = {
    'model_initial_dr_per_winding': 16.,
    'model_flow_voxel_resolution': 8,
    'model_flow_field_type': 'cartesian',
    'model_num_flow_timesteps': 1,
    'model_num_flow_stages': 1,
    'model_linear_z_resolution': 48,
    'model_gap_expander_logit_resolution': 24,
    'model_gap_expander_num_windings': 6,
    'model_gap_expander_lr_scale': 0.3,
    'model_bake_grid_resolution': 4,
    'output_first_winding': 1,
}


def make_tiny_model(**config_overrides):
    torch.manual_seed(0)
    min_corner = torch.tensor([0, -96, -96], dtype=torch.int64)
    max_corner = torch.tensor([192, 96, 96], dtype=torch.int64)
    umbilicus = torch.zeros([5, 3])
    umbilicus[:, 0] = torch.linspace(0., 192., 5)
    # A non-trivial umbilicus shear, so the bake has to absorb it.
    umbilicus[:, 1] = 10.
    umbilicus[:, 2] = -5.
    config = dict(TINY_CONFIG)
    config.update(config_overrides)
    return SpiralAndTransform(
        flow_integration_steps=3,
        flow_integration_solver='rk4',
        flow_min_corner_zyx=min_corner,
        flow_max_corner_zyx=max_corner,
        umbilicus_zyx=umbilicus,
        config=config,
    )


@torch.no_grad()
def randomize_live_smooth_params_(model, seed, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    for flow_field in model.flow_fields:
        flow_field.flows[0].copy_(
            torch.randn(flow_field.flows[0].shape, generator=generator) * 0.01 * scale)
        flow_field.flows[1].copy_(
            torch.randn(flow_field.flows[1].shape, generator=generator) * 0.003 * scale)
    model.linear_logits.copy_(
        torch.randn(model.linear_logits.shape, generator=generator) * 0.002 * scale)


def spiral_probe_points(n=512, seed=7):
    generator = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, generator=generator) * 2 * torch.pi
    winding = 1 + torch.rand(n, generator=generator) * 3
    radius = winding * TINY_CONFIG['model_initial_dr_per_winding']
    z = 20. + torch.rand(n, generator=generator) * 150.
    return torch.stack(
        [z, radius * torch.sin(theta), radius * torch.cos(theta)], dim=-1)


class FrozenDisplacementTransformTest(unittest.TestCase):

    def test_round_trip_on_synthetic_warp(self):
        # An affine warp is reproduced exactly by trilinear interpolation.
        offset = torch.tensor([1.5, -2.0, 0.5])
        scale = torch.tensor([1.02, 0.98, 1.01])

        def fn(points):
            return points * scale + offset

        def fn_inv(points):
            return (points - offset) / scale

        from transforms import _bake_displacement_grid
        min_corner = torch.tensor([0., -50., -50.])
        max_corner = torch.tensor([100., 50., 50.])
        grid_fwd, image_min, image_max = _bake_displacement_grid(
            fn, min_corner, max_corner, resolution=10.)
        grid_inv, _, _ = _bake_displacement_grid(
            fn_inv, image_min, image_max, resolution=10.)
        accum = FrozenDisplacementTransform(
            grid_fwd, min_corner, max_corner, grid_inv, image_min, image_max)
        points = torch.rand([256, 3]) * (max_corner - min_corner) + min_corner
        torch.testing.assert_close(accum._call(points), fn(points), atol=1e-3, rtol=0)
        torch.testing.assert_close(
            accum._inverse(accum._call(points)), points, atol=1e-3, rtol=0)

    def test_gradients_flow_through_input_only(self):
        grid = torch.randn([3, 4, 4, 4]) * 0.1
        accum = FrozenDisplacementTransform(
            grid, torch.zeros(3), torch.full([3], 10.),
            grid.clone(), torch.zeros(3), torch.full([3], 10.))
        points = (torch.rand([32, 3]) * 10.).requires_grad_(True)
        accum._call(points).sum().backward()
        self.assertIsNotNone(points.grad)
        self.assertFalse(accum.grid_fwd.requires_grad)


class BakeResetTest(unittest.TestCase):

    def _mapped(self, transform, spiral_points):
        # slice_to_spiral maps scroll->spiral; .inv is spiral->scroll.
        with torch.no_grad():
            return transform.inv(spiral_points)

    def test_bake_preserves_total_map_and_resets_live_params(self):
        model = make_tiny_model()
        randomize_live_smooth_params_(model, seed=1)
        with torch.no_grad():
            model.gap_expander_params.logits.normal_(0., 0.002)
        gap_before = model.gap_expander_params.logits.detach().clone()
        dr_before = model.dr_per_winding_logit.detach().clone()

        points = spiral_probe_points()
        before = self._mapped(model.get_slice_to_spiral_transform(), points)

        model.snapshot_live_epoch_(iteration=100)
        stats = model.rebake_accumulator()
        model.reset_live_smooth_params_()

        # Live smooth params are identity; gap and dr are untouched.
        for flow_field in model.flow_fields:
            for param in flow_field.parameters():
                self.assertEqual(float(param.detach().abs().max()), 0.)
        self.assertEqual(float(model.linear_logits.detach().abs().max()), 0.)
        torch.testing.assert_close(model.gap_expander_params.logits, gap_before)
        torch.testing.assert_close(model.dr_per_winding_logit, dr_before)

        # Exactness at reset: the exact-chain stand-in reproduces the
        # pre-bake map to numerical noise.
        exact = self._mapped(
            model.get_slice_to_spiral_transform(exact_frozen=True), points)
        torch.testing.assert_close(exact, before, atol=1e-4, rtol=0)

        # Grid fidelity: the baked accumulator matches to sub-voxel error.
        baked = self._mapped(model.get_slice_to_spiral_transform(), points)
        grid_error = (baked - before).norm(dim=-1)
        self.assertLess(float(grid_error.max()), 1.0)
        self.assertLess(float(grid_error.mean()), 0.2)
        self.assertLess(stats['bake_probe_fwd_error_max'], 1.0)
        self.assertLess(stats['bake_probe_inv_error_max'], 1.0)
        self.assertEqual(stats['bake_num_epochs'], 1)

        # Round trip through the accumulator chain stays tight.
        with torch.no_grad():
            transform = model.get_slice_to_spiral_transform()
            round_trip = transform(transform.inv(points))
        self.assertLess(float((round_trip - points).norm(dim=-1).max()), 1.0)

    def test_second_bake_composes_exactly(self):
        model = make_tiny_model()
        randomize_live_smooth_params_(model, seed=1)
        model.snapshot_live_epoch_(iteration=100)
        model.rebake_accumulator()
        model.reset_live_smooth_params_()

        randomize_live_smooth_params_(model, seed=2)
        points = spiral_probe_points()
        # The exact map before the second bake: live stages composed with the
        # exact first epoch (not the grid, whose error is not part of the map).
        before = self._mapped(
            model.get_slice_to_spiral_transform(exact_frozen=True), points)

        model.snapshot_live_epoch_(iteration=200)
        stats = model.rebake_accumulator()
        model.reset_live_smooth_params_()

        after = self._mapped(
            model.get_slice_to_spiral_transform(exact_frozen=True), points)
        torch.testing.assert_close(after, before, atol=1e-4, rtol=0)
        self.assertEqual(stats['bake_num_epochs'], 2)
        # The grid is rebaked from scratch through the exact composition, so
        # its error does not compound with the number of resets.
        baked = self._mapped(model.get_slice_to_spiral_transform(), points)
        self.assertLess(float((baked - before).norm(dim=-1).max()), 1.0)

    def test_multi_stage_bake(self):
        model = make_tiny_model(model_num_flow_stages=2)
        randomize_live_smooth_params_(model, seed=3)
        points = spiral_probe_points()
        before = self._mapped(model.get_slice_to_spiral_transform(), points)
        model.snapshot_live_epoch_()
        model.rebake_accumulator()
        model.reset_live_smooth_params_()
        exact = self._mapped(
            model.get_slice_to_spiral_transform(exact_frozen=True), points)
        torch.testing.assert_close(exact, before, atol=1e-4, rtol=0)

    def test_history_without_grids_is_refused(self):
        model = make_tiny_model()
        randomize_live_smooth_params_(model, seed=1)
        model.snapshot_live_epoch_()
        payload = model.serialize_frozen_epochs(4)
        restored = make_tiny_model()
        restored.load_frozen_epochs(payload, rebake=False)
        with self.assertRaises(AssertionError):
            restored.get_slice_to_spiral_transform()
        # The exact chain never needs the grids.
        restored.get_slice_to_spiral_transform(exact_frozen=True)


class RaySpecializedTest(unittest.TestCase):

    def test_matches_generic_chain_with_accumulator(self):
        model = make_tiny_model()
        randomize_live_smooth_params_(model, seed=1)
        model.snapshot_live_epoch_()
        model.rebake_accumulator()
        model.reset_live_smooth_params_()
        randomize_live_smooth_params_(model, seed=4, scale=0.5)

        transform = model.get_slice_to_spiral_transform()
        generator = torch.Generator().manual_seed(11)
        num_rays, samples_per_ray = 24, 8
        theta = torch.rand(num_rays, generator=generator) * 2 * torch.pi
        z = 20. + torch.rand(num_rays, generator=generator) * 150.
        pair_id = torch.arange(num_rays).repeat_interleave(samples_per_ray)
        radii = (1 + torch.rand(num_rays * samples_per_ray, generator=generator) * 3) \
            * TINY_CONFIG['model_initial_dr_per_winding']
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)

        with torch.no_grad():
            out = ray_specialized_spiral_to_scroll(
                transform, radii, theta, z, pair_id, sin_t, cos_t)
            self.assertIsNotNone(
                out, 'chain matcher rejected the accumulator chain')
            spiral = torch.stack([
                z[pair_id], sin_t[pair_id] * radii, cos_t[pair_id] * radii,
            ], dim=-1)
            expected = transform.inv(spiral)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=0)


class CheckpointPayloadTest(unittest.TestCase):

    def test_round_trip_through_two_bakes(self):
        model = make_tiny_model()
        randomize_live_smooth_params_(model, seed=1)
        model.snapshot_live_epoch_(iteration=100)
        model.rebake_accumulator()
        model.reset_live_smooth_params_()
        randomize_live_smooth_params_(model, seed=2)
        model.snapshot_live_epoch_(iteration=200)
        model.rebake_accumulator()
        model.reset_live_smooth_params_()

        payload = model.serialize_frozen_epochs(
            TINY_CONFIG['model_bake_grid_resolution'])
        self.assertEqual([e['iteration'] for e in payload['epochs']], [100, 200])

        restored = make_tiny_model()
        restored.load_state_dict(model.state_dict())
        self.assertEqual(restored.frozen_epochs_compatibility(payload), [])
        stats = restored.load_frozen_epochs(payload)
        self.assertEqual(stats['bake_num_epochs'], 2)

        points = spiral_probe_points()
        with torch.no_grad():
            for exact in (False, True):
                torch.testing.assert_close(
                    restored.get_slice_to_spiral_transform(exact_frozen=exact).inv(points),
                    model.get_slice_to_spiral_transform(exact_frozen=exact).inv(points),
                    atol=1e-4, rtol=0)

    def test_incompatible_history_is_reported(self):
        model = make_tiny_model()
        randomize_live_smooth_params_(model, seed=1)
        model.snapshot_live_epoch_()
        payload = model.serialize_frozen_epochs(4)
        other = make_tiny_model(model_num_flow_stages=2)
        self.assertTrue(other.frozen_epochs_compatibility(payload))
        self.assertEqual(other.frozen_epochs_compatibility(None),
                         ['frozen_epochs payload is not an epoch-list mapping'])

    def test_empty_history_clears_state(self):
        model = make_tiny_model()
        randomize_live_smooth_params_(model, seed=1)
        model.snapshot_live_epoch_()
        model.rebake_accumulator()
        model.load_frozen_epochs(None)
        self.assertEqual(model.frozen_epochs, [])
        self.assertIsNone(model.accum_transform)
        model.get_slice_to_spiral_transform()


if __name__ == '__main__':
    unittest.main()
