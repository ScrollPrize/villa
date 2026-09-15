import os
import unittest
import numpy as np
import scipy.ndimage
import torch
import flow_triton
from flow_fields import BSplineCylindricalFlowField


def _points_from_cylindrical(z_cont, r_cont, phi, nz, nr):
    # Invert the sampler's query mapping: (z_cont, r_cont, phi) lattice
    # coordinates -> normalised [0, 1] cartesian points.
    rr = r_cont / (nr - 1)
    z_n = z_cont / (nz - 1) * 2. - 1.
    y_n = rr * torch.sin(phi)
    x_n = rr * torch.cos(phi)
    return torch.stack([(z_n + 1.) / 2., (y_n + 1.) / 2., (x_n + 1.) / 2.], dim=-1)


def _to_local(output, phi):
    # Invert _cylindrical_local_to_cartesian's rotation (orthogonal).
    z_c, y_c, x_c = output[..., 0], output[..., 1], output[..., 2]
    sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
    r_c = y_c * sin_phi + x_c * cos_phi
    p_c = y_c * cos_phi - x_c * sin_phi
    return torch.stack([z_c, r_c, p_c], dim=-1)


def _ragged_lattice(num_phi_list):
    ring_num_phi = torch.tensor(num_phi_list, dtype=torch.long)
    ring_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(ring_num_phi, dim=0)])
    return ring_num_phi, ring_offsets


class BSplineCylindricalSamplerTests(unittest.TestCase):
    def test_matches_scipy_on_uniform_rings(self):
        # With every ring given the same phi count the interpolant is a true
        # tensor-product cubic B-spline on a regular (z, r, phi) grid:
        # border-replicated along z and r, periodic along phi.
        # scipy.ndimage.map_coordinates(order=3, prefilter=False,
        # mode='nearest') on a phi-tiled copy (queried in the middle copy, so
        # 'nearest' never fires along phi) is an independent reference.
        torch.manual_seed(0)
        nz, nr, m = 5, 6, 8
        field = torch.randn(3, nz, nr * m, dtype=torch.float64)
        ring_num_phi, ring_offsets = _ragged_lattice([m] * nr)

        generator = torch.Generator().manual_seed(1)
        n = 300
        z_cont = torch.rand(n, generator=generator, dtype=torch.float64) * (nz - 1)
        r_cont = 0.05 + torch.rand(n, generator=generator, dtype=torch.float64) * (nr - 1.05)
        phi = (torch.rand(n, generator=generator, dtype=torch.float64) * 2. - 1.) * np.pi * 0.999
        points = _points_from_cylindrical(z_cont, r_cont, phi, nz, nr)

        output = BSplineCylindricalFlowField._sample_lattice(
            field, ring_num_phi, ring_offsets, points)
        local = _to_local(output, phi)

        phi_cont = (phi.numpy() % (2. * np.pi)) * m / (2. * np.pi)
        coords = np.stack([z_cont.numpy(), r_cont.numpy(), phi_cont + m])
        reference = np.stack([
            scipy.ndimage.map_coordinates(
                np.concatenate([field[c].reshape(nz, nr, m).numpy()] * 3, axis=2),
                coords, order=3, prefilter=False, mode='nearest')
            for c in range(3)
        ], axis=-1)
        torch.testing.assert_close(
            local, torch.from_numpy(reference), rtol=1e-10, atol=1e-10)

    def test_gradcheck(self):
        torch.manual_seed(3)
        ring_num_phi, ring_offsets = _ragged_lattice([1, 6, 13, 19])
        nz, nr = 4, 4
        field = torch.randn(
            3, nz, int(ring_offsets[-1]), dtype=torch.float64, requires_grad=True)
        # Interior queries: away from the axis basis singularity, the z border
        # clamps and the rr=1 disk clamp (tap-index clamping and the phi wrap
        # are fine -- piecewise-constant indices, smooth weights).
        generator = torch.Generator().manual_seed(4)
        n = 9
        z_cont = 0.5 + torch.rand(n, generator=generator, dtype=torch.float64) * (nz - 2)
        r_cont = 1.2 + torch.rand(n, generator=generator, dtype=torch.float64) * 1.4
        phi = (torch.rand(n, generator=generator, dtype=torch.float64) * 2. - 1.) * np.pi * 0.999
        points = _points_from_cylindrical(z_cont, r_cont, phi, nz, nr)
        points = points.detach().requires_grad_(True)

        torch.autograd.gradcheck(
            lambda f, p: BSplineCylindricalFlowField._sample_lattice(
                f, ring_num_phi, ring_offsets, p),
            (field, points))


class BSplineCylindricalGradientTests(unittest.TestCase):
    def test_streamed_backwards_and_pending_field_grad_match_dense_autograd(self):
        torch.manual_seed(11)
        flow = BSplineCylindricalFlowField(torch.tensor([12, 12, 12]))
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
            total = 0.
            for param, num_phi, offsets in (
                    (reference_lr, flow._lr_num_phi, flow._lr_offsets),
                    (reference_hr, flow._hr_num_phi, flow._hr_offsets)):
                field = param[0]
                n0 = int(num_phi[0])
                field = torch.cat(
                    [torch.zeros_like(field[:, :, :n0]), field[:, :, n0:]], dim=2)
                total = total + BSplineCylindricalFlowField._sample_lattice(
                    field, num_phi, offsets, pts)
            return total

        reference_out_a = reference_sample(reference_a)
        reference_out_b = reference_sample(reference_b)
        (reference_out_a.square().mean() + reference_out_b.abs().mean()).backward()

        sampler = flow.get_sampler()
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


@unittest.skipUnless(
    torch.cuda.is_available() and flow_triton._HAS_TRITON,
    'needs CUDA and triton')
class BSplineCylindricalTritonTests(unittest.TestCase):
    # Kernel-level fused-vs-eager checks (edge points, adjoint, shared
    # accumulators) run in tests/test_cylindrical_triton.py for both
    # interpolants; this is the end-to-end model check.

    def test_full_model_grads_match_eager_path(self):
        from flow_fixtures import (
            _make_small_spiral_model, _sample_scroll_points)

        def run(disable_triton):
            model = _make_small_spiral_model(23, 'bspline_cylindrical', device='cuda')
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
            model.flow_field.apply_accumulated_field_grad()
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
