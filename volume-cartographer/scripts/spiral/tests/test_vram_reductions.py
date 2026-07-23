import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from flow_fields import CartesianFlowField, sample_field
from checkpoint_io import load_checkpoint_cpu
from tifxyz import Patch
from tracks import (
    _grouped_same_radius_loss,
    _pack_track_points,
    _sample_prepared_track_points,
    configure_prepared_track_sampling,
    filter_tracks_to_outer_shell,
    iter_track_losses,
    prepare_main_phase_tracks,
    validate_track_sampling_config,
)


class CartesianFlowGradientTests(unittest.TestCase):
    def test_accumulator_reuse_matches_dense_autograd(self):
        torch.manual_seed(4)
        resolution = torch.tensor([12, 12, 12])
        flow = CartesianFlowField(resolution, spatial_scale_factor=6, lr_scale_factor=0.2)
        with torch.no_grad():
            flow.flows[0].normal_(std=0.1)
            flow.flows[1].normal_(std=0.1)

        points = torch.rand(37, 3, requires_grad=True)
        reference_points = points.detach().clone().requires_grad_(True)
        reference_lr = flow.flows[0].detach().clone().requires_grad_(True)
        reference_hr = flow.flows[1].detach().clone().requires_grad_(True)

        reference_lr_up = F.interpolate(
            reference_lr,
            size=tuple(reference_hr.shape[2:]),
            mode='trilinear',
        )[0]
        reference_field = reference_lr_up + reference_hr[0] * 0.2
        reference_output = sample_field(reference_points, reference_field)
        reference_loss = reference_output.square().sum()
        reference_loss.backward()

        output = flow.get_sampler(0.0)(points)
        loss = output.square().sum()
        loss.backward()
        flow.apply_accumulated_field_grad()

        torch.testing.assert_close(output, reference_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(points.grad, reference_points.grad, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(flow.flows[0].grad, reference_lr.grad, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(flow.flows[1].grad, reference_hr.grad, rtol=2e-4, atol=2e-5)
        self.assertEqual(
            flow.flows[1].grad.untyped_storage().data_ptr(),
            flow._field_grad_acc.untyped_storage().data_ptr(),
        )

    def test_multiple_streamed_backwards_accumulate_before_field_backward(self):
        torch.manual_seed(9)
        resolution = torch.tensor([12, 12, 12])
        combined = CartesianFlowField(resolution, spatial_scale_factor=6, lr_scale_factor=0.2)
        streamed = CartesianFlowField(resolution, spatial_scale_factor=6, lr_scale_factor=0.2)
        with torch.no_grad():
            combined.flows[0].normal_(std=0.1)
            combined.flows[1].normal_(std=0.1)
            streamed.load_state_dict(combined.state_dict())
        # In the fitter these coordinates are outputs of earlier trainable
        # transforms, which is what connects the custom sampler to autograd.
        points_a = torch.rand(29, 3, requires_grad=True)
        points_b = torch.rand(41, 3, requires_grad=True)

        combined_sampler = combined.get_sampler(0.0)
        (combined_sampler(points_a).square().mean()
         + combined_sampler(points_b).abs().mean()).backward()
        combined.apply_accumulated_field_grad()

        streamed_sampler = streamed.get_sampler(0.0)
        streamed_sampler(points_a).square().mean().backward(retain_graph=True)
        streamed_sampler(points_b).abs().mean().backward(retain_graph=True)
        streamed.apply_accumulated_field_grad()

        torch.testing.assert_close(streamed.flows[0].grad, combined.flows[0].grad)
        torch.testing.assert_close(streamed.flows[1].grad, combined.flows[1].grad)


class CpuTrackStorageTests(unittest.TestCase):
    @staticmethod
    def _line_track(length, *, z=10, y=10, axis=2):
        points = np.zeros((int(length) + 1, 3), dtype=np.float32)
        points[:, 0] = z
        points[:, 1] = y
        points[:, axis] = np.arange(int(length) + 1, dtype=np.float32)
        return points

    def test_only_sampled_track_batch_moves_to_training_device(self):
        tracks = [
            np.arange(18, dtype=np.float32).reshape(6, 3),
            np.arange(30, 48, dtype=np.float32).reshape(6, 3),
        ]
        prepared = prepare_main_phase_tracks(
            tracks,
            anchor_scroll_zyxs=None,
            exclusion_radius=0.0,
            device='cpu',
        )

        self.assertIn('flat_zyx_cpu', prepared)
        self.assertNotIn('flat_zyx', prepared)
        self.assertEqual(prepared['flat_zyx_cpu'].device.type, 'cpu')
        sample = _sample_prepared_track_points(prepared, 2, 4)
        self.assertEqual(sample['sampled_scroll'].shape, (4, 3))
        self.assertEqual(sample['sampled_scroll'].device.type, 'cpu')

    def test_tracks_entirely_outside_outer_shell_are_removed(self):
        class RadiusFiveShell:
            @staticmethod
            def lookup(points):
                radius = torch.linalg.norm(points[:, 1:], dim=-1)
                target = torch.full_like(radius, 5.0)
                confidence = torch.ones_like(radius)
                valid = torch.ones_like(radius, dtype=torch.bool)
                return target, radius, confidence, valid

        inside = np.array([[10, 1, 1], [10, 2, 2]], dtype=np.float32)
        crossing = np.array([[10, 6, 0], [10, 4, 0]], dtype=np.float32)
        outside = np.array([[10, 6, 0], [10, 7, 0]], dtype=np.float32)

        tracks, families = filter_tracks_to_outer_shell(
            [inside, crossing, outside], RadiusFiveShell(),
            ['horizontal', 'vertical', 'vertical'], max_points_per_chunk=2)

        self.assertEqual(len(tracks), 2)
        np.testing.assert_array_equal(tracks[0], inside)
        np.testing.assert_array_equal(tracks[1], crossing)
        self.assertEqual(families, ['horizontal', 'vertical'])

    def test_zero_exclusion_fast_path_drops_short_tracks_without_reordering(self):
        tracks = [
            np.array([[1, 2, 3]], dtype=np.float32),
            np.arange(18, dtype=np.float32).reshape(6, 3),
            np.arange(30, 48, dtype=np.float32).reshape(6, 3),
        ]
        prepared = prepare_main_phase_tracks(
            tracks,
            anchor_scroll_zyxs=None,
            exclusion_radius=0.0,
            device='cpu',
        )
        np.testing.assert_array_equal(
            prepared['flat_zyx_cpu'].numpy(),
            np.concatenate(tracks[1:], axis=0),
        )
        np.testing.assert_array_equal(prepared['offsets'].numpy(), [0, 6, 12])
        np.testing.assert_array_equal(prepared['lengths'].numpy(), [6, 6])

    def test_disabled_sampling_policies_preserve_seeded_legacy_draw(self):
        tracks = [
            self._line_track(5, y=10),
            self._line_track(7, y=20),
            self._line_track(9, y=30),
        ]
        legacy = prepare_main_phase_tracks(tracks, None, 0.0, 'cpu')
        configured = prepare_main_phase_tracks(
            tracks, None, 0.0, 'cpu',
            sampling_config={
                'track_length_bin_weights': None,
                'track_max_tortuosity': None,
                'max_track_crossing_per_step': 0,
            },
        )

        torch.manual_seed(123)
        legacy_sample = _sample_prepared_track_points(legacy, 3, 4)
        torch.manual_seed(123)
        configured_sample = _sample_prepared_track_points(configured, 3, 4)
        for key in (
                'track_idx', 'sampled_scroll', 'row_id', 'row_starts',
                'row_lengths', 'group_id', 'target_flat_idx',
                'target_source_idx'):
            torch.testing.assert_close(configured_sample[key], legacy_sample[key])

    def test_short_tracks_preserve_endpoints_without_forcing_target_count(self):
        short = self._line_track(2, y=10)
        long_a = self._line_track(8, y=20)
        long_b = self._line_track(10, y=30)
        prepared = prepare_main_phase_tracks(
            [short, long_a, long_b], None, 0.0, 'cpu')
        prepared['sampling_probabilities'] = torch.tensor([1., 0., 0.])

        torch.manual_seed(7)
        sample = _sample_prepared_track_points(prepared, 1, 4)

        self.assertEqual(sample['track_idx'].tolist(), [0])
        self.assertEqual(sample['row_lengths'].tolist(), [2])
        torch.testing.assert_close(
            sample['sampled_scroll'][0], torch.from_numpy(short[0]))
        torch.testing.assert_close(
            sample['sampled_scroll'][-1], torch.from_numpy(short[-1]))

    def test_complete_track_sample_stays_between_20_and_60_voxel_spacing(self):
        track = self._line_track(125, y=10)
        prepared = prepare_main_phase_tracks([track], None, 0.0, 'cpu')

        sample = _sample_prepared_track_points(
            prepared, 1, 24,
            min_sample_spacing=20.0, max_sample_spacing=60.0)
        points = sample['sampled_scroll']
        spacing = torch.linalg.norm(torch.diff(points, dim=0), dim=-1)

        torch.testing.assert_close(points[0], torch.from_numpy(track[0]))
        torch.testing.assert_close(points[-1], torch.from_numpy(track[-1]))
        self.assertLess(len(points), 24)
        self.assertGreaterEqual(float(spacing.min()), 20.0)
        self.assertLessEqual(float(spacing.max()), 60.0)

    def test_resampled_cache_key_tracks_both_spacing_bounds(self):
        track = self._line_track(125, y=10)
        prepared = prepare_main_phase_tracks([track], None, 0.0, 'cpu')

        _sample_prepared_track_points(
            prepared, 1, 4,
            min_sample_spacing=20.0, max_sample_spacing=60.0)
        self.assertEqual(list(prepared['resampled_cache']), [(20.0, 60.0)])

        _sample_prepared_track_points(
            prepared, 1, 4,
            min_sample_spacing=10.0, max_sample_spacing=40.0)
        self.assertEqual(list(prepared['resampled_cache']), [(10.0, 40.0)])

    def test_length_bin_weights_are_distributed_within_tertiles(self):
        tracks = [self._line_track(length, y=length * 2) for length in range(1, 10)]
        prepared = prepare_main_phase_tracks(
            tracks, None, 0.0, 'cpu',
            sampling_config={
                'track_length_bin_weights': [0.15, 0.25, 0.60],
                'track_max_tortuosity': None,
                'max_track_crossing_per_step': 0,
            },
        )

        probabilities = prepared['sampling_probabilities'].numpy()
        np.testing.assert_allclose([
            probabilities[:3].sum(),
            probabilities[3:6].sum(),
            probabilities[6:].sum(),
        ], [0.15, 0.25, 0.60], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(probabilities[:3], np.full(3, 0.05))

    def test_length_bin_weights_can_change_on_prepared_tracks(self):
        tracks = [self._line_track(length, y=length * 2) for length in range(1, 10)]
        prepared = prepare_main_phase_tracks(tracks, None, 0.0, 'cpu')

        configure_prepared_track_sampling(prepared, {
            'track_length_bin_weights': [0.6, 0.3, 0.1],
        })

        probabilities = prepared['sampling_probabilities'].numpy()
        np.testing.assert_allclose([
            probabilities[:3].sum(),
            probabilities[3:6].sum(),
            probabilities[6:].sum(),
        ], [0.6, 0.3, 0.1], rtol=1e-6, atol=1e-7)

        configure_prepared_track_sampling(prepared, {
            'track_length_bin_weights': None,
        })
        self.assertNotIn('sampling_probabilities', prepared)

    def test_tortuosity_filter_is_opt_in_and_uses_arclength_over_chord(self):
        straight = np.array([
            [10, 10, 0], [10, 10, 5], [10, 10, 10],
        ], dtype=np.float32)
        tortuous = np.array([
            [10, 10, 0], [10, 13, 0], [10, 13, 4], [10, 10, 4],
        ], dtype=np.float32)

        unfiltered = prepare_main_phase_tracks(
            [straight, tortuous], None, 0.0, 'cpu',
            sampling_config={'track_max_tortuosity': None},
        )
        filtered = prepare_main_phase_tracks(
            [straight, tortuous], None, 0.0, 'cpu',
            sampling_config={'track_max_tortuosity': 2.0},
        )

        self.assertEqual(unfiltered['lengths'].numel(), 2)
        self.assertEqual(filtered['lengths'].numel(), 1)
        np.testing.assert_array_equal(filtered['flat_zyx_cpu'].numpy(), straight)

    def test_crossing_partners_are_opposite_family_angled_and_spaced(self):
        primary = self._line_track(20, z=10, y=10, axis=2)

        def vertical_at(x):
            track = np.zeros((21, 3), dtype=np.float32)
            track[:, 0] = 10
            track[:, 1] = np.arange(21, dtype=np.float32)
            track[:, 2] = x
            return track

        tracks = [
            primary,
            vertical_at(4),
            vertical_at(10),
            vertical_at(16),
            primary.copy(),  # Opposite provenance, but parallel: reject it.
        ]
        prepared = prepare_main_phase_tracks(
            tracks, None, 0.0, 'cpu',
            sampling_config={
                'track_length_bin_weights': None,
                'track_max_tortuosity': None,
                'track_crossing_precompute_max': 2,
                'max_track_crossing_per_step': 2,
            },
            track_families=['horizontal', 'vertical', 'vertical', 'vertical', 'vertical'],
        )

        np.testing.assert_array_equal(
            prepared['crossing_partners'][0].numpy(), [1, 3])
        np.testing.assert_array_equal(
            prepared['crossing_self_local'][0].numpy(), [4, 16])
        np.testing.assert_array_equal(
            prepared['crossing_partner_local'][0].numpy(), [10, 10])
        self.assertNotIn(4, prepared['crossing_partners'][0].tolist())

        configure_prepared_track_sampling(prepared, {
            'max_track_crossing_per_step': 1,
        })

        # Force primary track zero so the first draw uses the Run-scoped limit.
        prepared['sampling_probabilities'] = torch.tensor([1., 0., 0., 0., 0.])
        sample = _sample_prepared_track_points(prepared, 1, 4)
        np.testing.assert_array_equal(sample['track_idx'].numpy(), [0, 1])
        self.assertEqual(sample['row_lengths'].shape, (2,))
        self.assertEqual(sample['group_id'].tolist(), [0, 0])
        self.assertEqual(sample['group_width'], 2)

        configure_prepared_track_sampling(prepared, {
            'max_track_crossing_per_step': 2,
        })
        sample = _sample_prepared_track_points(prepared, 1, 4)
        np.testing.assert_array_equal(sample['track_idx'].numpy(), [0, 1, 3])
        self.assertEqual(sample['group_width'], 3)
        for primary_flat, partner_flat in zip(
                sample['primary_cross_flat'], sample['partner_cross_flat']):
            torch.testing.assert_close(
                sample['sampled_scroll'][primary_flat],
                sample['sampled_scroll'][partner_flat],
            )

    def test_crossing_index_uses_first_local_index_for_repeated_voxel(self):
        horizontal = np.array([
            [10, 10, 0],
            [10, 10, 1],
            [10, 10, 2],
            [10, 10, 1],
            [10, 10, 2],
            [10, 10, 3],
        ], dtype=np.float32)
        vertical = np.array([
            [10, 8, 2],
            [10, 9, 2],
            [10, 10, 2],
            [10, 11, 2],
            [10, 12, 2],
        ], dtype=np.float32)
        same_family = vertical.copy()
        prepared = prepare_main_phase_tracks(
            [horizontal, vertical, same_family], None, 0.0, 'cpu',
            sampling_config={
                'track_crossing_precompute_max': 1,
                'max_track_crossing_per_step': 1,
            },
            track_families=['horizontal', 'vertical', 'vertical'],
        )

        self.assertEqual(prepared['crossing_partners'][0, 0], 1)
        self.assertEqual(prepared['crossing_self_local'][0, 0], 2)
        self.assertEqual(prepared['crossing_partner_local'][0, 0], 2)
        self.assertNotIn(2, prepared['crossing_partners'][1].tolist())

    def test_track_point_packing_is_chunk_independent(self):
        points = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [(1 << 20) - 1, (1 << 20) - 2, (1 << 20) - 3],
        ], dtype=np.float32)
        expected = (
            points[:, 0].astype(np.uint64) << np.uint64(40)
            | points[:, 1].astype(np.uint64) << np.uint64(20)
            | points[:, 2].astype(np.uint64)
        )
        np.testing.assert_array_equal(
            _pack_track_points(points, chunk_size=1), expected)
        np.testing.assert_array_equal(
            _pack_track_points(points, chunk_size=len(points)), expected)

    def test_crossing_group_uses_one_radius_target_for_both_tracks(self):
        shifted = torch.tensor([0., 0., 10., 10.])
        target_values = shifted.reshape(2, 2)
        row_id = torch.tensor([0, 0, 1, 1])
        group_id = torch.tensor([0, 0])
        row_slot = torch.tensor([0, 1])
        cfg = {
            'track_radius_target': 'mean',
            'track_radius_loss_margin': 0.0,
            'track_radius_within_norm_p': 1.0,
        }

        loss, targets, _ = _grouped_same_radius_loss(
            shifted, target_values, row_id, group_id, row_slot,
            1, 2, torch.tensor(10.0), cfg)

        torch.testing.assert_close(targets, torch.tensor([5.0]))
        torch.testing.assert_close(loss, torch.tensor(5.0))

    def test_sampling_policy_defaults_to_20_and_60_voxel_spacing(self):
        policy = validate_track_sampling_config({})
        self.assertEqual(policy['min_sample_spacing'], 20.0)
        self.assertEqual(policy['max_sample_spacing'], 60.0)

    def test_sampling_policy_validation_rejects_malformed_values(self):
        with self.assertRaisesRegex(ValueError, 'short, medium, long'):
            validate_track_sampling_config({'track_length_bin_weights': [1, 2]})
        with self.assertRaisesRegex(ValueError, '>= 1'):
            validate_track_sampling_config({'track_max_tortuosity': 0.9})
        with self.assertRaisesRegex(ValueError, 'non-negative integer'):
            validate_track_sampling_config({'max_track_crossing_per_step': 1.5})
        with self.assertRaisesRegex(ValueError, 'non-negative integer'):
            validate_track_sampling_config({
                'track_crossing_precompute_max': -1,
            })
        with self.assertRaisesRegex(ValueError, 'finite number > 0'):
            validate_track_sampling_config({'track_max_sample_spacing': 0})
        with self.assertRaisesRegex(ValueError, 'finite number > 0'):
            validate_track_sampling_config({'track_min_sample_spacing': 0})
        with self.assertRaisesRegex(ValueError, 'must be <='):
            validate_track_sampling_config({
                'track_min_sample_spacing': 61,
                'track_max_sample_spacing': 60,
            })

    def test_staged_track_backward_matches_combined_backward(self):
        class Translation:
            def __init__(self, parameter, sign=1.0):
                self.parameter = parameter
                self.sign = sign

            def __call__(self, points):
                return points + self.parameter * self.sign

            @property
            def inv(self):
                return Translation(self.parameter, -self.sign)

        tracks = [
            np.array([[1, 5, 8], [2, 6, 9], [3, 7, 10], [4, 8, 11]], dtype=np.float32),
            np.array([[5, 9, 12], [6, 10, 13], [7, 11, 14], [8, 12, 15]], dtype=np.float32),
        ]
        prepared = prepare_main_phase_tracks(tracks, None, 0.0, 'cpu')
        config = {
            'track_num_per_step': 2,
            'track_num_points_per_step': 4,
            'track_radius_loss_margin': 0.025,
            'track_radius_target': 'mean',
            'track_radius_within_norm_p': 3.0,
            'track_dt_loss_margin': 0.025,
            'track_dt_within_track_norm_p': 3.0,
            'track_dt_norm_p': 0.5,
        }
        dr = torch.tensor(10.0)

        combined_parameter = torch.tensor(0.2, requires_grad=True)
        torch.manual_seed(12)
        combined_parts = list(iter_track_losses(
            Translation(combined_parameter), dr, prepared, config, compute_dt=True,
        ))
        sum(value for _, value in combined_parts).backward()

        staged_parameter = torch.tensor(0.2, requires_grad=True)
        torch.manual_seed(12)
        staged_parts = []
        for name, value in iter_track_losses(
            Translation(staged_parameter), dr, prepared, config, compute_dt=True,
        ):
            staged_parts.append((name, value.detach()))
            value.backward()

        self.assertEqual([name for name, _ in staged_parts], ['track_radius', 'track_dt'])
        torch.testing.assert_close(
            torch.stack([value for _, value in staged_parts]),
            torch.stack([value.detach() for _, value in combined_parts]),
        )
        torch.testing.assert_close(staged_parameter.grad, combined_parameter.grad)


class LazyPatchCacheTests(unittest.TestCase):
    def test_large_derived_tensors_are_lazy_and_releasable(self):
        zyxs = torch.zeros([5, 6, 3], dtype=torch.float32)
        patch = Patch(zyxs, torch.ones(3), None, None)
        self.assertIsNone(patch._valid_vertex_indices)
        self.assertIsNone(patch._valid_quad_indices)
        self.assertIsNone(patch._valid_zyxs)
        self.assertEqual(patch.valid_zyxs.shape, (30, 3))
        patch.release_derived_caches()
        self.assertIsNone(patch._valid_zyxs)


class CheckpointLoadingTests(unittest.TestCase):
    def test_modern_checkpoint_loads_on_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'checkpoint.pt'
            torch.save({'tensor': torch.arange(8), 'cfg': {'value': 3}}, path)
            loaded = load_checkpoint_cpu(path)
            torch.testing.assert_close(loaded['tensor'], torch.arange(8))
            self.assertEqual(loaded['cfg']['value'], 3)


if __name__ == '__main__':
    unittest.main()
