"""Constraint-bake reset machinery (plan B), CPU-only.

Validates the plan's core exactness claims on a small model:
- a snapshot materialises a transform identical to the live one;
- baked constraints under the reset (identity) transform reproduce the
  pre-reset spiral coordinates, so spiral-space losses are invariant across
  a bake (up to gap-stage fp noise) — including for the ACW (flipped) sense;
- the round-trip probe error is sub-voxel for small flows;
- the composed frozen+live map reproduces the pre-reset map, across >= 2
  bakes;
- resets preserve dr and clear exactly the reset parameters' optimiser
  state;
- the refusal gates fire for every unbakeable input class;
- snapshots survive torch.save round-trips;
- the winding-inference store gathers baked crossing points with unchanged
  winding-difference targets.
"""

import copy
import io
import unittest

import torch

import constraint_baking
from config import Config
from transforms import SpiralAndTransform
from winding_supervision import WindingInferenceStore


def make_small_spiral_model(seed, *, sense='CW', param_std=0.01):
    cfg = Config().as_dict()
    cfg['model_gap_expander_num_windings'] = 10
    z_span = 16 * 12
    flow_min = torch.tensor([0, -96, -96], dtype=torch.int64)
    flow_max = torch.tensor([z_span, 96, 96], dtype=torch.int64)
    zs = torch.arange(0, z_span + 1, dtype=torch.float32)
    umbilicus_zyx = torch.stack(
        [zs, torch.full_like(zs, 3.), torch.full_like(zs, -2.)], dim=-1)
    torch.manual_seed(seed)
    model = SpiralAndTransform(
        flow_integration_steps=3,
        flow_integration_solver='rk4',
        flow_min_corner_zyx=flow_min,
        flow_max_corner_zyx=flow_max,
        umbilicus_zyx=umbilicus_zyx,
        config=cfg,
        spiral_outward_sense=sense,
    )
    perturb_parameters(model, param_std)
    return model, umbilicus_zyx, cfg


def perturb_parameters(model, std, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel() > 1:
                parameter.normal_(std=std)


def sample_scroll_points(num_points, seed):
    generator = torch.Generator().manual_seed(seed)
    z = torch.rand(num_points, generator=generator) * 150 + 20
    theta = torch.rand(num_points, generator=generator) * 2 * torch.pi
    radius = torch.rand(num_points, generator=generator) * 60 + 10
    return torch.stack([
        z,
        radius * torch.sin(theta) + 3.,
        radius * torch.cos(theta) - 2.,
    ], dim=-1)


class SnapshotMaterializeTests(unittest.TestCase):

    def test_materialized_transform_matches_live(self):
        model, umbilicus, cfg = make_small_spiral_model(7)
        snapshot = constraint_baking.snapshot_frozen_epoch(
            model, umbilicus, cfg)
        rebuilt = constraint_baking.materialize_frozen_model(
            snapshot, cfg, torch.device('cpu'))
        points = sample_scroll_points(64, 1)
        with torch.no_grad():
            expected = model.get_slice_to_spiral_transform()(points)
            actual = rebuilt.get_slice_to_spiral_transform()(points)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertAlmostEqual(
            snapshot['dr_per_winding'],
            float(model.get_dr_per_winding().detach()), places=5)

    def test_snapshot_survives_torch_save_round_trip(self):
        model, umbilicus, cfg = make_small_spiral_model(9)
        snapshot = constraint_baking.snapshot_frozen_epoch(
            model, umbilicus, cfg)
        buffer = io.BytesIO()
        torch.save(snapshot, buffer)
        buffer.seek(0)
        reloaded = torch.load(buffer, weights_only=False)
        rebuilt = constraint_baking.materialize_frozen_model(
            reloaded, cfg, torch.device('cpu'))
        points = sample_scroll_points(32, 2)
        with torch.no_grad():
            expected = model.get_slice_to_spiral_transform()(points)
            actual = rebuilt.get_slice_to_spiral_transform()(points)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_snapshot_records_construction_config(self):
        model, umbilicus, cfg = make_small_spiral_model(3)
        snapshot = constraint_baking.snapshot_frozen_epoch(
            model, umbilicus, cfg)
        for key in constraint_baking.MODEL_CONSTRUCTION_CONFIG_KEYS:
            self.assertEqual(snapshot['model_config'][key], cfg[key])
        # A later live-config change must not leak into materialisation.
        drifted = dict(cfg)
        drifted['model_gap_expander_num_windings'] = 25
        rebuilt = constraint_baking.materialize_frozen_model(
            snapshot, drifted, torch.device('cpu'))
        self.assertEqual(
            len(rebuilt.gap_expander_params.num_by_winding),
            len(model.gap_expander_params.num_by_winding))


class BakeAndResetTests(unittest.TestCase):

    def _bake_and_reset_case(self, sense):
        model, umbilicus, cfg = make_small_spiral_model(11, sense=sense)
        points = sample_scroll_points(96, 3)
        transform = model.get_slice_to_spiral_transform()
        with torch.no_grad():
            spiral_before = transform(points)
        baked = constraint_baking.bake_points(transform, points)
        torch.testing.assert_close(baked, spiral_before, rtol=0, atol=0)

        dr_before = float(model.get_dr_per_winding())
        constraint_baking.reset_live_transform_(model)
        self.assertEqual(model.spiral_outward_sense, 'CW')
        self.assertAlmostEqual(
            float(model.get_dr_per_winding()), dr_before, places=6)
        for parameter in constraint_baking.reset_parameters(model):
            self.assertEqual(float(parameter.abs().max()), 0.0)

        with torch.no_grad():
            spiral_after = model.get_slice_to_spiral_transform()(baked)
        # The reset transform is the identity over baked inputs, so the
        # spiral coordinates every loss reads are unchanged across the bake.
        torch.testing.assert_close(
            spiral_after, spiral_before, rtol=0, atol=1e-3)

    def test_bake_then_reset_preserves_spiral_coordinates_cw(self):
        self._bake_and_reset_case('CW')

    def test_bake_then_reset_preserves_spiral_coordinates_acw(self):
        # The frozen chain's x-flip is committed into the baked inputs; the
        # reset chain must not apply it a second time.
        self._bake_and_reset_case('ACW')

    def test_reset_umbilicus_is_identity(self):
        model, _, _ = make_small_spiral_model(13)
        constraint_baking.reset_live_transform_(model)
        points = sample_scroll_points(16, 4)
        with torch.no_grad():
            moved = model.umbilicus_transform._call(points)
        torch.testing.assert_close(moved, points, rtol=0, atol=0)

    def test_bake_points_chunking_is_transparent(self):
        model, _, _ = make_small_spiral_model(17)
        transform = model.get_slice_to_spiral_transform()
        points = sample_scroll_points(50, 5)
        whole = constraint_baking.bake_points(transform, points)
        chunked = constraint_baking.bake_points(
            transform, points, chunk_size=7)
        # Different chunk sizes take different fp vectorisation paths (last
        # few ulps); rank-determinism comes from the FIXED default chunk
        # size, not from chunk-size invariance.
        torch.testing.assert_close(chunked, whole, rtol=0, atol=1e-4)
        self.assertEqual(points.dtype, chunked.dtype)

    def test_round_trip_probe_is_subvoxel_for_small_flows(self):
        model, _, _ = make_small_spiral_model(19)
        probe = constraint_baking.probe_round_trip_error(model)
        self.assertGreater(probe['num_points'], 0)
        self.assertLess(probe['max'], 0.5)
        self.assertLessEqual(probe['mean'], probe['max'])


class ComposedTransformTests(unittest.TestCase):

    def test_one_epoch_composed_matches_pre_reset_map(self):
        model, umbilicus, cfg = make_small_spiral_model(23)
        points = sample_scroll_points(64, 6)
        with torch.no_grad():
            spiral_before = model.get_slice_to_spiral_transform()(points)
        snapshot = constraint_baking.snapshot_frozen_epoch(
            model, umbilicus, cfg)
        constraint_baking.reset_live_transform_(model)
        frozen = constraint_baking.materialize_frozen_model(
            snapshot, cfg, torch.device('cpu'))
        composed = constraint_baking.composed_slice_to_spiral_transform(
            [frozen], model.get_slice_to_spiral_transform())
        with torch.no_grad():
            actual = composed(points)
        torch.testing.assert_close(actual, spiral_before, rtol=0, atol=1e-3)

    def test_two_epochs_compose_in_scroll_to_spiral_order(self):
        model, umbilicus, cfg = make_small_spiral_model(29)
        points = sample_scroll_points(48, 7)
        transform_1 = model.get_slice_to_spiral_transform()
        baked_1 = constraint_baking.bake_points(transform_1, points)
        snapshot_1 = constraint_baking.snapshot_frozen_epoch(
            model, umbilicus, cfg)
        constraint_baking.reset_live_transform_(model)

        # Simulate an epoch of training in the canonical frame, then a
        # second bake.
        perturb_parameters(model, 0.005, seed=31)
        transform_2 = model.get_slice_to_spiral_transform()
        baked_2 = constraint_baking.bake_points(transform_2, baked_1)
        snapshot_2 = constraint_baking.snapshot_frozen_epoch(
            model, constraint_baking.identity_umbilicus_like(umbilicus), cfg)
        constraint_baking.reset_live_transform_(model)
        perturb_parameters(model, 0.005, seed=37)

        frozen = [
            constraint_baking.materialize_frozen_model(
                snapshot, cfg, torch.device('cpu'))
            for snapshot in (snapshot_1, snapshot_2)
        ]
        live = model.get_slice_to_spiral_transform()
        composed = constraint_baking.composed_slice_to_spiral_transform(
            frozen, live)
        with torch.no_grad():
            expected = live(baked_2)
            actual = composed(points)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-3)


class OptimizerResetTests(unittest.TestCase):

    def test_reset_clears_exactly_the_reset_parameter_state(self):
        model, _, _ = make_small_spiral_model(41)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        points = sample_scroll_points(24, 8)
        loss = model.get_slice_to_spiral_transform()(points).square().mean()
        loss = loss + model.get_dr_per_winding().square()
        loss.backward()
        for flow_field in model.flow_fields:
            flow_field.apply_accumulated_field_grad()
        optimiser.step()

        reset = constraint_baking.reset_parameters(model)
        self.assertTrue(all(p in optimiser.state for p in reset))
        self.assertIn(model.dr_per_winding_logit, optimiser.state)
        constraint_baking.clear_optimizer_state_(optimiser, reset)
        for parameter in reset:
            self.assertNotIn(parameter, optimiser.state)
        # dr keeps both its value and its moments across a reset.
        self.assertIn(model.dr_per_winding_logit, optimiser.state)


class RefusalGateTests(unittest.TestCase):

    def test_no_reasons_when_everything_is_bakeable(self):
        self.assertEqual(constraint_baking.bake_refusal_reasons(
            interactive=False, influence_active=False, phase_mode=False,
            dense_normals_enabled=False, grad_mag_enabled=False), [])

    def test_each_unbakeable_input_is_named(self):
        flags = ('interactive', 'influence_active', 'phase_mode',
                 'dense_normals_enabled', 'grad_mag_enabled',
                 'warmup_active')
        for flag in flags:
            kwargs = {name: name == flag for name in flags}
            reasons = constraint_baking.bake_refusal_reasons(**kwargs)
            self.assertEqual(len(reasons), 1, flag)
        reasons = constraint_baking.bake_refusal_reasons(
            **{name: True for name in flags})
        self.assertEqual(len(reasons), len(flags))


def make_synthetic_winding_store():
    store = WindingInferenceStore.__new__(WindingInferenceStore)
    store.origin = torch.tensor(
        [[0., 0., 0.], [1., 2., 3.]], dtype=torch.float32)
    store.step = torch.tensor(
        [[0., 0., 1.], [0., 1., 0.]], dtype=torch.float32)
    store.offset = torch.tensor([0, 3, 5], dtype=torch.int64)
    store.crossing_t = torch.tensor(
        [1., 2., 4., 1., 3.], dtype=torch.float32)
    store.crossing_level = torch.tensor(
        [1, 2, 3, 4, 6], dtype=torch.int32)
    store.length = store.offset[1:] - store.offset[:-1]
    store._z_eligible = torch.ones(2, dtype=torch.bool)
    store.density_rays = torch.nonzero(
        (store.length >= 2) & store._z_eligible, as_tuple=False).squeeze(-1)
    store._relative_rays = {}
    store.crossing_zyx = None
    return store


class WindingStoreBakeTests(unittest.TestCase):

    def test_materialized_points_match_ray_form(self):
        store = make_synthetic_winding_store()
        table = store.materialized_crossing_zyx()
        expected_row_2 = store.origin[0] + 4. * store.step[0]
        torch.testing.assert_close(table[2], expected_row_2)
        expected_row_4 = store.origin[1] + 3. * store.step[1]
        torch.testing.assert_close(table[4], expected_row_4)

    def test_baked_store_gathers_baked_points_with_unchanged_targets(self):
        store = make_synthetic_winding_store()
        ray = torch.tensor([0, 1])
        first = torch.tensor([0, 0])
        second = torch.tensor([2, 1])
        before = store._materialize(ray, first, second)
        shift = torch.tensor([100., 0., 0.])
        store.bake_crossing_points_(lambda points: points + shift)
        after = store._materialize(ray, first, second)
        torch.testing.assert_close(
            after['points'], before['points'] + shift)
        torch.testing.assert_close(after['target'], before['target'])

    def test_unbaked_store_still_uses_ray_form(self):
        store = make_synthetic_winding_store()
        samples = store._materialize(
            torch.tensor([1]), torch.tensor([0]), torch.tensor([1]))
        self.assertIsNone(store.crossing_zyx)
        torch.testing.assert_close(samples['target'], torch.tensor([2.]))


if __name__ == '__main__':
    unittest.main()
