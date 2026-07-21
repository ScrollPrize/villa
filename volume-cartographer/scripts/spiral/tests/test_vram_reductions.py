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
    _sample_prepared_track_points,
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
        _, _, sampled = _sample_prepared_track_points(prepared, 2, 4)
        self.assertEqual(sampled.shape, (2, 4, 3))
        self.assertEqual(sampled.device.type, 'cpu')

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
        for actual, expected in zip(configured_sample, legacy_sample):
            torch.testing.assert_close(actual, expected)

    def test_track_point_indices_are_unique_and_short_tracks_are_ineligible(self):
        short = self._line_track(2, y=10)  # Three points: cannot supply four uniquely.
        long_a = self._line_track(8, y=20)
        long_b = self._line_track(10, y=30)
        prepared = prepare_main_phase_tracks(
            [short, long_a, long_b], None, 0.0, 'cpu')

        torch.manual_seed(7)
        track_ids, local_indices, _ = _sample_prepared_track_points(
            prepared, 3, 4)

        self.assertNotIn(0, track_ids.tolist())
        for row in local_indices:
            self.assertEqual(torch.unique(row).numel(), 4)
            self.assertTrue(torch.all(row[1:] > row[:-1]))

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
                'max_track_crossing_per_step': 2,
            },
            track_families=['horizontal', 'vertical', 'vertical', 'vertical', 'vertical'],
        )

        np.testing.assert_array_equal(
            prepared['crossing_partners'][0].numpy(), [1, 3])
        self.assertNotIn(4, prepared['crossing_partners'][0].tolist())

        # Force primary track zero so the draw deterministically appends its two
        # prepared, well-spaced crossing partners.
        prepared['sampling_probabilities'] = torch.tensor([1., 0., 0., 0., 0.])
        track_ids, _, sampled = _sample_prepared_track_points(prepared, 1, 4)
        np.testing.assert_array_equal(track_ids.numpy(), [0, 1, 3])
        self.assertEqual(sampled.shape, (3, 4, 3))

    def test_sampling_policy_validation_rejects_malformed_values(self):
        with self.assertRaisesRegex(ValueError, 'short, medium, long'):
            validate_track_sampling_config({'track_length_bin_weights': [1, 2]})
        with self.assertRaisesRegex(ValueError, '>= 1'):
            validate_track_sampling_config({'track_max_tortuosity': 0.9})
        with self.assertRaisesRegex(ValueError, 'non-negative integer'):
            validate_track_sampling_config({'max_track_crossing_per_step': 1.5})

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
