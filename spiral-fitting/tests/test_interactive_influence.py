import math
import unittest
import numpy as np
import torch
from influence import _gap_logit_zst, influence_weight, make_influence_state
from transforms import GapExpandingTransform, SpiralAndTransform

TINY_CONFIG = {
    'model_initial_dr_per_winding': 16.,
    'model_flow_voxel_resolution': 8,  # 24^3 high-res lattice, 4^3 low-res: both have interior points
    'model_flow_field_type': 'cartesian',
    'model_linear_z_resolution': 48,
    'model_gap_expander_logit_resolution': 24,
    'model_gap_expander_num_windings': 6,
    'model_gap_expander_capacity_windings': 6,
    'model_gap_expander_min_gap': 1.0,
    'model_gap_expander_softplus_bias': 4.0,
    'model_gap_expander_lr_scale': 0.3,
    'output_first_winding': 1,
}

INFLUENCE_CONFIG = {
    'optimizer_random_seed': 1,
    'influence_enabled': True,
    'influence_z': 48.0,
    'influence_windings': 1.5,
    'influence_theta_frac': 0.25,
    'influence_sigma': 0.3333,
    'sample_count_influence_footprint_points': 256,
    'loss_weight_anchor': 20.0,
    'sample_count_influence_anchor_lattice_points': 200,
    'sample_count_influence_anchor_geometry_points': 200,
    'sample_count_influence_anchor_samples_per_step': 64,
    'influence_anchor_ramp_power': 2.0,
    'shell_outer_winding_idx': 5,
}


def make_tiny_model(**config_overrides):
    torch.manual_seed(0)
    min_corner = torch.tensor([0, -96, -96], dtype=torch.int64)
    max_corner = torch.tensor([192, 96, 96], dtype=torch.int64)
    umbilicus = torch.zeros([5, 3])
    umbilicus[:, 0] = torch.linspace(0., 192., 5)
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


def make_optimiser(model, gap_weight_decay=1.e-2):
    flow_field_params = list(model.flow_field.parameters())
    gap_expander_params = list(model.gap_expander_params.parameters())
    linear_params = [model.linear_logits]
    grouped = {id(p) for p in flow_field_params + gap_expander_params + linear_params}
    other_params = [p for p in model.parameters() if id(p) not in grouped]
    return torch.optim.AdamW([
        {'params': other_params, 'weight_decay': 0.0},
        {'params': linear_params, 'weight_decay': 0.0},
        {'params': gap_expander_params, 'weight_decay': gap_weight_decay},
        {'params': flow_field_params, 'weight_decay': 0.0},
    ], lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8)


def zst(z, s, theta):
    return torch.tensor([[float(z), float(s), float(theta)]])


class InfluenceWeightTests(unittest.TestCase):
    LIMITS = (100., 2., 0.5 * 2. * math.pi)
    SIGMA = 1. / 3.

    def weight(self, query, footprint):
        return influence_weight(query, footprint, self.LIMITS, self.SIGMA)

    def test_zero_beyond_each_hard_extent(self):
        footprint = zst(500., 3., 1.)
        for axis, offset in ((0, self.LIMITS[0]), (1, self.LIMITS[1])):
            query = footprint.clone()
            query[0, axis] += offset * 1.01
            self.assertEqual(float(self.weight(query, footprint)), 0.)
            query[0, axis] = footprint[0, axis] - offset * 1.01
            self.assertEqual(float(self.weight(query, footprint)), 0.)
        # Interior points at less than the extent have positive weight.
        query = footprint.clone()
        query[0, 0] += self.LIMITS[0] * 0.99
        self.assertGreater(float(self.weight(query, footprint)), 0.)

    def test_theta_distance_is_circular(self):
        footprint = zst(500., 3., 0.05)
        near_wrap = zst(500., 3., 2. * math.pi - 0.05)
        far = zst(500., 3., math.pi)
        w_near = float(self.weight(near_wrap, footprint))
        w_far = float(self.weight(far, footprint))
        self.assertGreater(w_near, 0.9)
        self.assertLess(w_far, w_near)

    def test_union_is_max_of_per_input_fields(self):
        footprint_a = torch.tensor([[500., 3., 1.]])
        footprint_b = torch.tensor([[700., 5., 4.]])
        queries = torch.stack([
            torch.empty([64]).uniform_(400., 800.),
            torch.empty([64]).uniform_(2., 6.),
            torch.empty([64]).uniform_(0., 2. * math.pi),
        ], dim=-1)
        pooled = self.weight(queries, torch.cat([footprint_a, footprint_b]))
        separate = torch.maximum(self.weight(queries, footprint_a),
                                 self.weight(queries, footprint_b))
        torch.testing.assert_close(pooled, separate)


class GapCoordinateTests(unittest.TestCase):
    def test_perturbing_one_logit_changes_radii_only_at_predicted_location(self):
        model = make_tiny_model()
        gap_params = model.gap_expander_params
        min_z = float(model.flow_min_corner_zyx[0])
        max_z = float(model.flow_max_corner_zyx[0])
        z_rows, s_cols, theta_cols = _gap_logit_zst(gap_params, min_z, max_z, torch.device('cpu'))
        total = s_cols.shape[0]
        column = total // 2
        row = gap_params.num_z // 2
        bucket = int(s_cols[column] - 0.5)

        theta_probe = torch.linspace(0., 2. * math.pi * (1. - 1e-6), 64)
        z_probe = torch.full_like(theta_probe, float(z_rows[row]))

        def radii():
            transform = GapExpandingTransform(
                gap_params, torch.tensor(16.), min_z, max_z, TINY_CONFIG['model_gap_expander_lr_scale'])
            with torch.no_grad():
                return transform.get_transformed_winding_radii(theta_probe, z_probe)

        before = radii()
        with torch.no_grad():
            gap_params.logits[0, 0, row, column] += 0.05
            gap_params._triton_consts = None
        after = radii()
        diff = (after - before).abs()

        # Windings at or below the perturbed gap's inner winding are untouched
        # (the cumsum runs outwards); the winding just outside changes near the
        # logit's theta.
        self.assertEqual(float(diff[:, :bucket + 1].max()), 0.)
        theta_spacing = 2. * math.pi / float(gap_params.num_by_winding[bucket])
        near = (theta_probe - float(theta_cols[column])).abs() < theta_spacing
        self.assertGreater(float(diff[near, bucket + 1].max()), 0.)
        far = (theta_probe - float(theta_cols[column])).abs() > 2. * theta_spacing
        self.assertEqual(float(diff[far, bucket + 1].max()), 0.)


class FlowLatticeMappingTests(unittest.TestCase):
    def test_flowbox_to_spiral_round_trip_with_nonzero_flow(self):
        model = make_tiny_model()
        with torch.no_grad():
            for flow in model.flow_field.flows:
                flow.normal_(std=1e-3)
        transform = model.get_flowbox_to_spiral_transform(include_diffeomorphism=True)
        points = torch.stack([
            torch.empty([256]).uniform_(10., 180.),
            torch.empty([256]).uniform_(-80., 80.),
            torch.empty([256]).uniform_(-80., 80.),
        ], dim=-1)
        with torch.no_grad():
            round_trip = transform.inv(transform(points))
        torch.testing.assert_close(round_trip, points, atol=0.5, rtol=0.)


class FreezeInvariantTests(unittest.TestCase):
    def _prepopulated(self):
        model = make_tiny_model()
        optimiser = make_optimiser(model)
        for _ in range(3):
            for p in model.parameters():
                p.grad = torch.randn_like(p) * 1e-3
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
        return model, optimiser

    def test_fully_masked_elements_are_bitwise_frozen(self):
        model, optimiser = self._prepopulated()
        state = make_influence_state(INFLUENCE_CONFIG, torch.device('cpu'))
        state._allocate_masks(model)
        torch.manual_seed(1)
        for mask in state.masks.values():
            mask.copy_((torch.rand(mask.shape) * 1.5 - 0.5).clamp(0., 1.).to(mask.dtype))
        state.num_incorporations = 1
        state._apply_optimizer_surgery(model, optimiser)

        lr_flow, hr_flow = model.flow_field.flows
        gap_logits = model.gap_expander_params.logits
        snapshot = {name: p.detach().clone() for name, p in model.named_parameters()}

        for _ in range(3):
            for p in model.parameters():
                p.grad = torch.randn_like(p) * 1e-3
            state.apply_grad_masks_(model)
            optimiser.step()
            state.apply_masked_gap_decay_(model, optimiser)
            optimiser.zero_grad(set_to_none=True)

        for param, mask in ((lr_flow, state.masks['flow_lr']),
                            (hr_flow, state.masks['flow_hr']),
                            (gap_logits, state.masks['gap'])):
            name = next(n for n, p in model.named_parameters() if p is param)
            frozen = (mask == 0).expand_as(param)
            self.assertTrue(torch.equal(param.detach()[frozen], snapshot[name][frozen]),
                            f'{name}: fully-masked elements moved')
            free = (mask > 0.5).expand_as(param)
            if free.any():
                self.assertFalse(torch.equal(param.detach()[free], snapshot[name][free]),
                                 f'{name}: unmasked elements did not move')
        for name in ('linear_logits', 'dr_per_winding_logit'):
            param = dict(model.named_parameters())[name]
            self.assertTrue(torch.equal(param.detach(), snapshot[name]),
                            f'{name}: frozen parameter moved')

    def test_flow_masks_apply_to_every_stage(self):
        model = make_tiny_model(model_num_flow_stages=2)
        state = make_influence_state(INFLUENCE_CONFIG, torch.device('cpu'))
        state._allocate_masks(model)
        state.masks['flow_lr'].zero_()
        state.masks['flow_hr'].zero_()
        for flow in model.flow_field.flows:
            flow.grad = torch.ones_like(flow)
        state.apply_grad_masks_(model)
        for flow in model.flow_field.flows:
            self.assertEqual(flow.shape[0], 2)
            self.assertEqual(float(flow.grad.abs().sum()), 0.)

    def test_gap_weight_decay_disabled_and_reemulated_in_region(self):
        model, optimiser = self._prepopulated()
        state = make_influence_state(INFLUENCE_CONFIG, torch.device('cpu'))
        state._allocate_masks(model)
        state.masks['gap'][:, : state.masks['gap'].shape[1] // 2] = 1.
        state.num_incorporations = 1
        state._apply_optimizer_surgery(model, optimiser)
        gap_group = state._find_param_group(optimiser, model.gap_expander_params.logits)
        self.assertEqual(gap_group['weight_decay'], 0.0)
        self.assertEqual(state.saved_gap_weight_decay, 1.e-2)
        # In-region decay shrinks logits multiplicatively even with zero grads.
        gap_logits = model.gap_expander_params.logits
        with torch.no_grad():
            gap_logits.fill_(1.)
        state.apply_masked_gap_decay_(model, optimiser)
        in_region = gap_logits.detach()[0, 0, :, : gap_logits.shape[-1] // 2]
        out_region = gap_logits.detach()[0, 0, :, gap_logits.shape[-1] // 2:]
        self.assertTrue((in_region < 1.).all())
        self.assertTrue((out_region == 1.).all())

        state.deactivate_(model, optimiser)
        self.assertEqual(gap_group['weight_decay'], 1.e-2)


class ActivateAndAnchorTests(unittest.TestCase):
    def _make_collection(self, num_points=32):
        torch.manual_seed(3)
        theta = torch.rand([num_points]) * math.pi * 0.25
        radius = (2.0 + theta / (2. * math.pi)) * 16.
        zyx = torch.stack([
            torch.empty([num_points]).uniform_(80., 110.),
            torch.sin(theta) * radius,
            torch.cos(theta) * radius,
        ], dim=-1)
        points = {i: {'zyx': zyx[i].numpy().astype(np.float32)} for i in range(num_points)}
        return {'points': points}

    def _activate(self, model, optimiser):
        state = make_influence_state(INFLUENCE_CONFIG, torch.device('cpu'))
        anchor_geometry = torch.stack([
            torch.empty([512]).uniform_(10., 180.),
            torch.empty([512]).uniform_(-70., 70.),
            torch.empty([512]).uniform_(-70., 70.),
        ], dim=-1)
        state.activate_or_extend_(
            new_patches={},
            new_collections={1: self._make_collection()},
            spiral_and_transform=model,
            optimiser=optimiser,
            cfg=INFLUENCE_CONFIG,
            z_begin=0,
            z_end=192,
            anchor_geometry_zyx=anchor_geometry,
        )
        return state

    def test_anchor_loss_zero_after_refresh_then_positive_after_perturbation(self):
        model = make_tiny_model()
        optimiser = make_optimiser(model)
        state = self._activate(model, optimiser)
        transform = model.get_slice_to_spiral_transform()
        dr = model.get_dr_per_winding()
        loss = state.get_anchor_loss(transform, dr, 128)
        self.assertLess(float(loss), 1e-5)
        with torch.no_grad():
            for flow in model.flow_field.flows:
                flow.normal_(std=1e-3)
        transform = model.get_slice_to_spiral_transform()
        loss = state.get_anchor_loss(transform, dr, 128)
        self.assertGreater(float(loss), 1e-6)
        loss.backward()
        # Flow gradients accumulate in the field-grad buffer and land on the
        # parameters only via this handoff (same as the training loop).
        model.flow_field.apply_accumulated_field_grad()
        self.assertTrue(any(flow.grad is not None and flow.grad.abs().sum() > 0
                            for flow in model.flow_field.flows))

    def test_fiber_revision_replaces_its_influence_contribution(self):
        model = make_tiny_model()
        optimiser = make_optimiser(model)
        state = make_influence_state(INFLUENCE_CONFIG, torch.device('cpu'))
        anchor_geometry = torch.stack([
            torch.empty([512]).uniform_(10., 180.),
            torch.empty([512]).uniform_(-70., 70.),
            torch.empty([512]).uniform_(-70., 70.),
        ], dim=-1)
        first = self._make_collection()
        first['metadata'] = {
            'logical_input_kind': 'fiber',
            'logical_input_id': 'fiber-7',
            'logical_input_revision': 'r1',
        }
        state.activate_or_extend_(
            new_patches={}, new_collections={1: first},
            spiral_and_transform=model, optimiser=optimiser,
            cfg=INFLUENCE_CONFIG, z_begin=0, z_end=192,
            anchor_geometry_zyx=anchor_geometry)
        old_footprint = state.contributions[('fiber', 'fiber-7')]['zst'].clone()
        old_anchor_target = state.anchor_target.clone()
        with torch.no_grad():
            for flow in model.flow_field.flows:
                flow.normal_(std=1e-3)
        revision_boundary_target = model.get_slice_to_spiral_transform()(
            state.anchor_scroll).to(torch.float32)
        self.assertFalse(torch.equal(revision_boundary_target, old_anchor_target))

        revised = self._make_collection()
        for point in revised['points'].values():
            point['zyx'] = point['zyx'] + np.array([60., 0., 0.], dtype=np.float32)
        revised['metadata'] = {
            'logical_input_kind': 'fiber',
            'logical_input_id': 'fiber-7',
            'logical_input_revision': 'r2',
        }
        state.activate_or_extend_(
            new_patches={}, new_collections={2: revised},
            spiral_and_transform=model, optimiser=optimiser,
            cfg=INFLUENCE_CONFIG, z_begin=0, z_end=192,
            anchor_geometry_zyx=None)

        self.assertEqual(list(state.contributions), [('fiber', 'fiber-7')])
        contribution = state.contributions[('fiber', 'fiber-7')]
        self.assertEqual(contribution['revision'], 'r2')
        self.assertGreater(float(contribution['zst'][:, 0].mean()),
                           float(old_footprint[:, 0].mean()) + 40.)
        torch.testing.assert_close(state.anchor_target, revision_boundary_target)

    def test_deleting_logical_pcl_removes_its_influence_contribution(self):
        model = make_tiny_model()
        optimiser = make_optimiser(model)
        state = make_influence_state(INFLUENCE_CONFIG, torch.device('cpu'))
        first = self._make_collection()
        first['metadata'] = {
            'logical_input_kind': 'same_winding',
            'logical_input_id': '5',
        }
        second = self._make_collection()
        for point in second['points'].values():
            point['zyx'] = point['zyx'] + np.array(
                [60., 0., 0.], dtype=np.float32)
        second['metadata'] = {
            'logical_input_kind': 'same_winding',
            'logical_input_id': '7',
        }
        state.activate_or_extend_(
            new_patches={}, new_collections={1: first, 2: second},
            spiral_and_transform=model, optimiser=optimiser,
            cfg=INFLUENCE_CONFIG, z_begin=0, z_end=192,
            anchor_geometry_zyx=None)

        state.remove_logical_contributions_(
            [('same_winding', '5')],
            spiral_and_transform=model, optimiser=optimiser)

        self.assertEqual(list(state.contributions),
                         [('same_winding', '7')])
        self.assertEqual(state.footprints[0]['input_id'], '7')
        self.assertTrue(state.active)
