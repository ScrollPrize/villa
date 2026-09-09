import unittest

import numpy as np
import scipy.ndimage
import torch

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

    def test_constant_local_field_reproduced_away_from_axis(self):
        # Per-axis weights sum to 1 (periodic wrap and border replication
        # included), so a lattice constant in the LOCAL basis is reproduced
        # exactly at queries whose radial stencil misses the pinned r=0 ring
        # (r_cont >= 2). The z/radial constants also exercise the rotation.
        flow = BSplineCylindricalFlowField(torch.tensor([8, 24, 24])).double()
        nr = flow._hr_num_phi.shape[0]
        with torch.no_grad():
            flow.flows[1][:, 0] = 0.7   # local z
            flow.flows[1][:, 1] = -1.3  # local radial

        generator = torch.Generator().manual_seed(2)
        n = 400
        z_cont = torch.rand(n, generator=generator, dtype=torch.float64) * 7
        r_cont = 2.0 + torch.rand(n, generator=generator, dtype=torch.float64) * (nr - 3.05)
        phi = (torch.rand(n, generator=generator, dtype=torch.float64) * 2. - 1.) * np.pi * 0.999
        points = _points_from_cylindrical(z_cont, r_cont, phi, 8, nr)

        with torch.no_grad():
            output = flow.get_sampler(0.0)(points)

        expected = torch.stack([
            torch.full_like(phi, 0.7),
            -1.3 * torch.sin(phi),
            -1.3 * torch.cos(phi),
        ], dim=-1)
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)

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

    def test_time_invariant_integrator_matches_manual_sampler_loop(self):
        # The override must run the CUBIC sampler loop, not the inherited
        # trilinear fused/eager integrator.
        torch.manual_seed(13)
        flow = BSplineCylindricalFlowField(torch.tensor([12, 12, 12]))
        with torch.no_grad():
            flow.flows[0].normal_(std=0.1)
            flow.flows[1].normal_(std=0.1)
        points = torch.rand(37, 3)
        h, n_steps = 1.0 / 3.0, 3

        with torch.no_grad():
            integrated = flow.get_time_invariant_integrator()(points, h, n_steps)
            sampler = flow.get_sampler(0.0)
            y = points
            for _ in range(n_steps):
                k1 = sampler(y)
                k2 = sampler(y + (h / 2) * k1)
                k3 = sampler(y + (h / 2) * k2)
                k4 = sampler(y + h * k3)
                y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        torch.testing.assert_close(integrated, y)


if __name__ == '__main__':
    unittest.main()
