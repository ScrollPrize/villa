import unittest

import numpy as np
import torch
import torch.nn.functional as F

from flow_fields import CartesianFlowField, sample_field
from tracks import _sample_prepared_track_points, prepare_main_phase_tracks


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


if __name__ == '__main__':
    unittest.main()
