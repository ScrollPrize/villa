import unittest

import numpy as np
import scipy.ndimage
import torch

import flow_triton
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


def _eager_rk4(low, high, pts, h, n_steps):
    y = pts
    for _ in range(n_steps):
        k1 = sample_field_bspline(low, y) + sample_field_bspline(high, y)
        k2 = (sample_field_bspline(low, y + (h / 2) * k1)
              + sample_field_bspline(high, y + (h / 2) * k1))
        k3 = (sample_field_bspline(low, y + (h / 2) * k2)
              + sample_field_bspline(high, y + (h / 2) * k2))
        k4 = (sample_field_bspline(low, y + h * k3)
              + sample_field_bspline(high, y + h * k3))
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def _channels_last(field):
    # rk4_bspline_integrate reads field values from channels-last copies.
    return field.permute(1, 2, 3, 0).contiguous()


def _mixed_points(num_random, seed, device, dtype):
    # Random interior/exterior points plus exact box corners and edges, so
    # the border clamp and its inclusive gradient mask are exercised.
    generator = torch.Generator().manual_seed(seed)
    pts = torch.rand(num_random, 3, generator=generator) * 1.6 - 0.3
    special = torch.tensor([
        [0., 0., 0.], [1., 1., 1.], [0., 0.5, 1.],
        [1e-6, 0.999999, 0.5], [-0.2, 0.5, 1.2],
    ])
    return torch.cat([pts, special]).to(device=device, dtype=dtype)


@unittest.skipUnless(
    torch.cuda.is_available() and flow_triton._HAS_TRITON,
    'needs CUDA and triton')
class BSplineTritonEquivalenceTests(unittest.TestCase):
    H = 1.0 / 3.0
    N_STEPS = 3

    def _make_fields(self, seed, device='cuda', dtype=torch.float32):
        torch.manual_seed(seed)
        # Distinct per-axis sizes catch index-stride mix-ups.
        low = (torch.randn(3, 3, 4, 5) * 0.1).to(device=device, dtype=dtype)
        high = (torch.randn(3, 13, 11, 9) * 0.1).to(device=device, dtype=dtype)
        return low, high

    def test_forward_matches_eager_and_fp64_reference(self):
        low, high = self._make_fields(7)
        # > one CUDA block plus a ragged tail.
        pts = _mixed_points(300, 8, 'cuda', torch.float32)

        with torch.no_grad():
            triton_out = flow_triton.rk4_bspline_integrate(
                pts, _channels_last(low), _channels_last(high),
                None, None, self.H, self.N_STEPS)
            eager_out = _eager_rk4(low, high, pts, self.H, self.N_STEPS)
            ref64 = _eager_rk4(
                low.double(), high.double(), pts.double(), self.H, self.N_STEPS)

        torch.testing.assert_close(triton_out, eager_out, rtol=1e-4, atol=1e-6)
        # Accuracy sandwich: the kernel's FP-association changes must not cost
        # more accuracy (vs a float64 reference) than eager fp32 noise.
        triton_err = (triton_out.double() - ref64).abs().max().item()
        eager_err = (eager_out.double() - ref64).abs().max().item()
        self.assertLess(triton_err, 3 * eager_err + 1e-6)

    def test_backward_matches_eager_autograd(self):
        low, high = self._make_fields(9)
        pts = _mixed_points(300, 10, 'cuda', torch.float32)

        reference_pts = pts.detach().clone().requires_grad_(True)
        reference_low = low.detach().clone().requires_grad_(True)
        reference_high = high.detach().clone().requires_grad_(True)
        torch.manual_seed(11)
        proj = torch.randn_like(pts)
        (_eager_rk4(reference_low, reference_high, reference_pts,
                    self.H, self.N_STEPS) * proj).sum().backward()

        triton_pts = pts.detach().clone().requires_grad_(True)
        acc_lo = torch.zeros_like(low)
        acc_hi = torch.zeros_like(high)
        out = flow_triton.rk4_bspline_integrate(
            triton_pts, _channels_last(low), _channels_last(high),
            acc_lo, acc_hi, self.H, self.N_STEPS)
        (out * proj).sum().backward()

        torch.testing.assert_close(
            triton_pts.grad, reference_pts.grad, rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(
            acc_lo, reference_low.grad, rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(
            acc_hi, reference_high.grad, rtol=1e-3, atol=1e-5)

    def test_module_integrator_accumulates_param_grads(self):
        # The BSplineFlowField wiring: detached-field integrator, per-lattice
        # accumulators, apply_accumulated_field_grad -> parameter gradients.
        torch.manual_seed(13)
        flow = BSplineFlowField(torch.tensor([12, 12, 12])).cuda()
        with torch.no_grad():
            flow.flows[0].normal_(std=0.1)
            flow.flows[1].normal_(std=0.1)
        pts = _mixed_points(200, 14, 'cuda', torch.float32).requires_grad_(True)

        reference_pts = pts.detach().clone().requires_grad_(True)
        reference_low = flow.flows[0][0].detach().clone().requires_grad_(True)
        reference_high = flow.flows[1][0].detach().clone().requires_grad_(True)
        (_eager_rk4(reference_low, reference_high, reference_pts,
                    self.H, self.N_STEPS).square().sum()).backward()

        integrate = flow.get_time_invariant_integrator()
        integrate(pts, self.H, self.N_STEPS).square().sum().backward()
        flow.apply_accumulated_field_grad()

        torch.testing.assert_close(
            pts.grad, reference_pts.grad, rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(
            flow.flows[0].grad[0], reference_low.grad, rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(
            flow.flows[1].grad[0], reference_high.grad, rtol=1e-3, atol=1e-5)
        self.assertEqual(
            flow.flows[1].grad.untyped_storage().data_ptr(),
            flow._acc_hi.untyped_storage().data_ptr())

    def test_full_model_grads_match_eager_path(self):
        import os
        from tests.test_vram_reductions import (
            _make_small_spiral_model, _sample_scroll_points)

        def run(disable_triton):
            model = _make_small_spiral_model(23, 'bspline', device='cuda')
            points = _sample_scroll_points(41, 5).cuda()
            previous = os.environ.get('FIT_SPIRAL_TRITON')
            os.environ['FIT_SPIRAL_TRITON'] = '0' if disable_triton else '1'
            try:
                transform = model.get_slice_to_spiral_transform()
                loss = (transform(points)[..., 1:].norm(dim=-1)
                        / model.get_dr_per_winding()).mean()
                loss.backward()
            finally:
                if previous is None:
                    os.environ.pop('FIT_SPIRAL_TRITON', None)
                else:
                    os.environ['FIT_SPIRAL_TRITON'] = previous
            for flow_field in model.flow_fields:
                flow_field.apply_accumulated_field_grad()
            return loss.detach(), {
                name: p.grad for name, p in model.named_parameters()}

        eager_loss, eager_grads = run(disable_triton=True)
        triton_loss, triton_grads = run(disable_triton=False)

        torch.testing.assert_close(triton_loss, eager_loss, rtol=1e-4, atol=1e-7)
        for name, eager_grad in eager_grads.items():
            triton_grad = triton_grads[name]
            if eager_grad is None and triton_grad is None:
                continue
            torch.testing.assert_close(
                triton_grad, eager_grad, rtol=2e-3, atol=1e-5,
                msg=lambda base, name=name: f'{name}: {base}')


if __name__ == '__main__':
    unittest.main()
