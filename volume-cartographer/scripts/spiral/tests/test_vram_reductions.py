import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from flow_fields import CartesianFlowField, sample_field
from checkpoint_io import load_checkpoint_cpu
from tifxyz import Patch
from tracks import _sample_prepared_track_points, iter_track_losses, prepare_main_phase_tracks


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
