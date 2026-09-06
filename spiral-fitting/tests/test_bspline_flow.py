import unittest

import numpy as np
import scipy.ndimage
import torch

from flow_fields import BSplineFlowField, sample_field_bspline


def _random_points(num_points, seed, lo=-0.3, hi=1.3):
    # Includes points outside [0, 1] to exercise the border clamp.
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(num_points, 3, generator=generator, dtype=torch.float64) * (hi - lo) + lo


class BSplineSamplerTests(unittest.TestCase):
    def test_matches_scipy_direct_bspline(self):
        # scipy.ndimage.map_coordinates with prefilter=False evaluates the
        # B-spline whose coefficients are the input array directly, with
        # mode='nearest' replicating edge coefficients -- an independent
        # reference for the same interpolant.
        torch.manual_seed(2)
        field = torch.randn(3, 5, 6, 7, dtype=torch.float64)
        points = _random_points(200, 3)

        output = sample_field_bspline(field, points)

        shape_m1 = np.array(field.shape[1:], dtype=np.float64) - 1
        coords = np.clip(points.numpy() * shape_m1, 0., shape_m1).T
        reference = np.stack([
            scipy.ndimage.map_coordinates(
                field[c].numpy(), coords, order=3, prefilter=False, mode='nearest')
            for c in range(3)
        ], axis=-1)
        torch.testing.assert_close(
            output, torch.from_numpy(reference), rtol=1e-10, atol=1e-10)

    def test_partition_of_unity_on_constant_field(self):
        # The cubic B-spline basis sums to one everywhere (including in the
        # replicated border region), so a constant lattice must reproduce the
        # constant exactly at every query point.
        field = torch.zeros(3, 4, 5, 6, dtype=torch.float64)
        constant = torch.tensor([0.7, -1.3, 2.1], dtype=torch.float64)
        field += constant[:, None, None, None]
        output = sample_field_bspline(field, _random_points(500, 4))
        torch.testing.assert_close(
            output, constant.expand_as(output), rtol=1e-12, atol=1e-12)

    def test_gradcheck(self):
        torch.manual_seed(5)
        field = torch.randn(3, 4, 5, 4, dtype=torch.float64, requires_grad=True)
        # Interior points: the field is C2 across knots, but the [0, 1] border
        # clamp itself is a kink that finite differences would straddle.
        points = (_random_points(11, 6, lo=0.05, hi=0.95)).requires_grad_(True)
        torch.autograd.gradcheck(sample_field_bspline, (field, points))

    def test_sampler_output_shape_preserved(self):
        field = torch.randn(3, 4, 4, 4, dtype=torch.float64)
        points = _random_points(24, 7).view(2, 3, 4, 3)
        self.assertEqual(sample_field_bspline(field, points).shape, (2, 3, 4, 3))


class BSplineFlowGradientTests(unittest.TestCase):
    def test_streamed_backwards_and_pending_field_grad_match_dense_autograd(self):
        torch.manual_seed(11)
        flow = BSplineFlowField(torch.tensor([12, 12, 12]), spatial_scale_factor=6)
        with torch.no_grad():
            flow.flows[0].normal_(std=0.1)
            flow.flows[1].normal_(std=0.1)

        points_a = torch.rand(29, 3, requires_grad=True)
        points_b = torch.rand(41, 3, requires_grad=True)
        reference_a = points_a.detach().clone().requires_grad_(True)
        reference_b = points_b.detach().clone().requires_grad_(True)
        reference_lr = flow.flows[0].detach().clone().requires_grad_(True)
        reference_hr = flow.flows[1].detach().clone().requires_grad_(True)

        def reference_sample(pts):
            return (
                sample_field_bspline(reference_lr[0], pts)
                + sample_field_bspline(reference_hr[0], pts)
            )

        reference_out_a = reference_sample(reference_a)
        reference_out_b = reference_sample(reference_b)
        (reference_out_a.square().mean() + reference_out_b.abs().mean()).backward()

        sampler = flow.get_sampler(0.0)
        out_a = sampler(points_a)
        out_b = sampler(points_b)
        # Two independent backwards through the one cached sampler, WITHOUT
        # retain_graph: the shared field graphs are cut at detached leaves.
        out_a.square().mean().backward()
        out_b.abs().mean().backward()
        flow.apply_accumulated_field_grad()

        torch.testing.assert_close(out_a, reference_out_a)
        torch.testing.assert_close(out_b, reference_out_b)
        torch.testing.assert_close(points_a.grad, reference_a.grad)
        torch.testing.assert_close(points_b.grad, reference_b.grad)
        torch.testing.assert_close(flow.flows[0].grad, reference_lr.grad)
        torch.testing.assert_close(flow.flows[1].grad, reference_hr.grad)
        self.assertIsNone(flow._pending_field_graphs)

    def test_no_grad_sampler_has_no_pending_record(self):
        flow = BSplineFlowField(torch.tensor([12, 12, 12]))
        with torch.no_grad():
            flow.get_sampler(0.0)(torch.rand(5, 3))
        self.assertIsNone(flow._pending_field_graphs)
        flow.apply_accumulated_field_grad()  # no-op


if __name__ == '__main__':
    unittest.main()
