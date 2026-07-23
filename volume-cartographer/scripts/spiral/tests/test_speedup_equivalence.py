"""Equivalence tests for the fit-speedup changes.

Each test compares the optimised code path against an inline copy of the
pre-change implementation. Cache/batching changes require bitwise identity;
fused or algebraically equivalent kernels use tight tolerances for expected
floating-point reassociation differences.

Run from scripts/spiral with the opt versions of transforms.py / losses.py in
place:  python -m pytest <this file> -v   (or unittest main).
"""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from transforms import GapExpanderParams, GapExpandingTransform, IntegratedFlowDiffeomorphism
from flow_fields import CartesianFlowField
from geom_utils import bilinear_atlas_lookup
from losses import (
    _batched_pcl_chain_seam_adjustments,
    _masked_all_pairs_l1,
    _unwrap_track_shifted_radii,
)
from sample_spiral import get_theta_and_radii
from tracks import (
    _same_radius_loss_for_shifted_radii,
    _unwrap_track_shifted_radii as _unwrap_main_track_shifted_radii,
)


class _ReferenceGapExpandingTransform(GapExpandingTransform):
    # Pre-change get_transformed_winding_radii: rebuilds the pinned+scaled
    # logits tensor on every call.
    def get_transformed_winding_radii(self, theta, z):
        num_windings = len(self.params.num_by_winding)
        winding_first_logit_idx = self.params.winding_first_logit_idx
        theta_normalised = theta / (2 * torch.pi)
        winding_coords = torch.lerp(winding_first_logit_idx[:-1], winding_first_logit_idx[1:], theta_normalised[..., None])
        winding_coords_normalised = winding_coords / winding_first_logit_idx[-1] * 2 - 1
        z_normalised = (z - self.min_z) / (self.max_z - self.min_z) * 2 - 1
        logits = torch.cat([torch.zeros_like(self.params.logits[..., :1]), self.params.logits[..., 1:]], dim=-1)
        logits = logits * self.gap_expander_lr_scale
        logits_by_winding = F.grid_sample(
            logits,
            torch.stack([winding_coords_normalised, z_normalised[..., None].expand(*theta.shape, num_windings)], dim=-1).view(1, -1, num_windings, 2),
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        ).squeeze(1).squeeze(0).view(*theta.shape, num_windings)
        scales_by_winding = torch.exp(logits_by_winding * 2.e2)
        if self.truncate_frac is not None:
            scales_by_winding = torch.lerp(torch.ones([], device=scales_by_winding.device), scales_by_winding, self.truncate_frac)
        inter_winding_distances = self.dr_per_winding * scales_by_winding
        winding_zero_radii = self.dr_per_winding * theta_normalised
        winding_radii = winding_zero_radii[..., None] + torch.cat([torch.zeros_like(inter_winding_distances[..., :1]), torch.cumsum(inter_winding_distances, dim=-1)[..., :-1]], dim=-1)
        return winding_radii


def _make_gap_setup(seed):
    torch.manual_seed(seed)
    dr = torch.tensor(16.0)
    params = GapExpanderParams(resolution=24, min_z=0.0, max_z=500.0, num_windings=12, dr_per_winding=16.0)
    with torch.no_grad():
        params.logits.normal_(std=3e-3)
    points = torch.randn(64, 3) * torch.tensor([120., 60., 60.]) + torch.tensor([250., 0., 0.])
    return params, dr, points


class GapExpanderLogitsCacheTests(unittest.TestCase):
    def test_call_and_inverse_bitwise_match_reference(self):
        params, dr, points = _make_gap_setup(0)
        opt = GapExpandingTransform(params, dr, 0.0, 500.0, gap_expander_lr_scale=0.3)
        ref = _ReferenceGapExpandingTransform(params, dr, 0.0, 500.0, gap_expander_lr_scale=0.3)
        # Two consecutive calls: the second exercises the cached tensor.
        for _ in range(2):
            self.assertTrue(torch.equal(opt._call(points), ref._call(points)))
            self.assertTrue(torch.equal(opt._inverse(points), ref._inverse(points)))

    def test_gradient_through_shared_cached_logits_matches_reference(self):
        params_opt, dr, points = _make_gap_setup(1)
        params_ref = GapExpanderParams(resolution=24, min_z=0.0, max_z=500.0, num_windings=12, dr_per_winding=16.0)
        with torch.no_grad():
            params_ref.logits.copy_(params_opt.logits)

        opt = GapExpandingTransform(params_opt, dr, 0.0, 500.0, gap_expander_lr_scale=0.3)
        ref = _ReferenceGapExpandingTransform(params_ref, dr, 0.0, 500.0, gap_expander_lr_scale=0.3)

        # Mimic the training loop: several transform invocations inside one
        # iteration, backwarded family-by-family with retain_graph=True.
        (opt._call(points).square().mean()).backward(retain_graph=True)
        (opt._inverse(points).abs().mean()).backward(retain_graph=True)
        (ref._call(points).square().mean()).backward(retain_graph=True)
        (ref._inverse(points).abs().mean()).backward(retain_graph=True)

        torch.testing.assert_close(params_opt.logits.grad, params_ref.logits.grad, rtol=0., atol=0.)


class BatchedChainSeamAdjustmentTests(unittest.TestCase):
    def _reference(self, transform, dr, chains):
        # Pre-change per-chain loop from get_patch_rel_winding_loss.
        out = []
        for chain in chains:
            chain_t = torch.from_numpy(chain)
            chain_spiral = transform(chain_t)
            chain_theta, _, _ = get_theta_and_radii(chain_spiral[..., 1:], dr)
            zero_shifted = torch.zeros_like(chain_theta)
            adj = _unwrap_track_shifted_radii(chain_theta, zero_shifted, dr)
            out.append(adj[-1])
        return torch.stack(out)

    def test_batched_matches_per_chain_loop_bitwise(self):
        rng = np.random.default_rng(3)
        dr = torch.tensor(16.0)

        # A deterministic nonlinear per-point map that swings points across the
        # theta=0 seam (rotation by a z-dependent angle plus a radial stretch).
        def transform(zyx):
            z, y, x = zyx[..., 0], zyx[..., 1], zyx[..., 2]
            ang = z * 0.021
            c, s = torch.cos(ang), torch.sin(ang)
            y2 = (c * y - s * x) * 1.17
            x2 = (s * y + c * x) * 1.17
            return torch.stack([z, y2, x2], dim=-1)

        chains = []
        for n in (1, 2, 3, 7, 4, 2):
            base_theta = rng.uniform(0, 2 * np.pi)
            thetas = base_theta + np.cumsum(rng.uniform(0.2, 1.4, size=n))
            radius = rng.uniform(40., 220., size=n)
            z = rng.uniform(0., 400., size=n)
            chain = np.stack([z, np.sin(thetas) * radius, np.cos(thetas) * radius], axis=-1).astype(np.float32)
            chains.append(chain)

        batched = _batched_pcl_chain_seam_adjustments(transform, dr, chains)
        reference = self._reference(transform, dr, chains)
        self.assertFalse(batched.requires_grad)
        self.assertTrue(torch.equal(batched, reference))


class MaskedAllPairsL1Tests(unittest.TestCase):
    @staticmethod
    def _reference(p1, p2, mask1, mask2, expected_diff):
        diff = p2[:, :, None] - p1[:, None, :]
        pair_mask = mask2[:, :, None] & mask1[:, None, :]
        error = (diff - expected_diff[:, None, None]).abs()
        return (error * pair_mask).sum() / pair_mask.sum().clamp(min=1)

    def test_values_and_gradients_match_quadratic_reference(self):
        torch.manual_seed(31)
        p1 = torch.randn(5, 29, requires_grad=True)
        p2 = torch.randn(5, 29, requires_grad=True)
        expected = torch.randn(5, requires_grad=True)
        mask1 = torch.rand(5, 29) > 0.25
        mask2 = torch.rand(5, 29) > 0.2
        reference = self._reference(p1, p2, mask1, mask2, expected)
        reference_grads = torch.autograd.grad(reference, (p1, p2, expected))
        actual = _masked_all_pairs_l1(p1, p2, mask1, mask2, expected)
        actual_grads = torch.autograd.grad(actual, (p1, p2, expected))

        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-7)

    def test_empty_mask_returns_zero_with_zero_gradients(self):
        p1 = torch.randn(2, 7, requires_grad=True)
        p2 = torch.randn(2, 7, requires_grad=True)
        expected = torch.randn(2, requires_grad=True)
        mask1 = torch.zeros(2, 7, dtype=torch.bool)
        mask2 = torch.ones(2, 7, dtype=torch.bool)

        loss = _masked_all_pairs_l1(p1, p2, mask1, mask2, expected)
        grads = torch.autograd.grad(loss, (p1, p2, expected))
        self.assertEqual(loss.item(), 0.0)
        for grad in grads:
            self.assertEqual(torch.count_nonzero(grad).item(), 0)


class BilinearAtlasLookupTests(unittest.TestCase):
    @staticmethod
    def _reference(zyxs_flat, offsets, widths, patch_indices, ijs):
        base = offsets[patch_indices]
        width = widths[patch_indices]
        i0 = ijs[..., 0].floor().to(torch.int64)
        j0 = ijs[..., 1].floor().to(torch.int64)
        di = (ijs[..., 0] - i0.to(torch.float32)).unsqueeze(-1)
        dj = (ijs[..., 1] - j0.to(torch.float32)).unsqueeze(-1)
        flat_tl = base + i0 * width + j0
        tl = zyxs_flat[flat_tl]
        tr = zyxs_flat[flat_tl + 1]
        bl = zyxs_flat[flat_tl + width]
        br = zyxs_flat[flat_tl + width + 1]
        top = tl + (tr - tl) * dj
        bottom = bl + (br - bl) * dj
        return top + (bottom - top) * di

    def test_packed_lookup_matches_original_arithmetic(self):
        torch.manual_seed(32)
        height, width, num_patches = 7, 9, 4
        patch_size = height * width
        zyxs_flat = torch.randn(num_patches * patch_size, 3)
        offsets = torch.arange(num_patches + 1, dtype=torch.int64) * patch_size
        widths = torch.full((num_patches,), width, dtype=torch.int64)
        patch_indices = torch.randint(num_patches, (3, 5))
        ijs = torch.rand(3, 5, 2) * torch.tensor([height - 1.01, width - 1.01])

        actual = bilinear_atlas_lookup(zyxs_flat, offsets, widths, patch_indices, ijs)
        reference = self._reference(zyxs_flat, offsets, widths, patch_indices, ijs)
        self.assertTrue(torch.equal(actual, reference))


class CompiledTrackTensorHelperTests(unittest.TestCase):
    def test_main_track_unwrap_matches_shared_loss_helper(self):
        torch.manual_seed(33)
        theta = torch.rand(13, 24) * (2 * torch.pi)
        shifted = torch.randn(13, 24, requires_grad=True)
        dr = torch.tensor(16.0, requires_grad=True)
        actual = _unwrap_main_track_shifted_radii(theta, shifted, dr)
        reference = _unwrap_track_shifted_radii(theta, shifted, dr)
        self.assertTrue(torch.equal(actual, reference))

    def test_radius_reducer_matches_reference_for_mean_and_median(self):
        torch.manual_seed(34)
        shifted = torch.randn(41, 24, requires_grad=True)
        dr = torch.tensor(16.0, requires_grad=True)
        for target in ('mean', 'median'):
            cfg = {
                'track_radius_loss_margin': 0.025,
                'track_radius_target': target,
                'track_radius_within_norm_p': 6.0,
            }
            if target == 'mean':
                centre = shifted.mean(dim=-1, keepdim=True)
            else:
                centre = shifted.median(dim=-1, keepdim=True).values
            hinged = F.relu((shifted - centre).abs() - dr.detach() * 0.025)
            reference = (((hinged + 1.e-5) ** 6.0).mean(dim=-1) ** (1.0 / 6.0)).mean()
            actual = _same_radius_loss_for_shifted_radii(shifted, dr, cfg)
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)


class SparseSamplerConstCacheTests(unittest.TestCase):
    def _reference_backward(self, pts, low_field, high_field, high_scale, grad_out):
        # Inline copy of the pre-change _SparseAccumTrilinearSample backward
        # (fresh constant tensors every call).
        from flow_fields import _corner_bits
        shape = torch.tensor(low_field.shape[1:], device=pts.device, dtype=pts.dtype)
        coord_raw = pts * (shape - 1)
        coord = coord_raw.clamp(min=torch.zeros_like(shape), max=shape - 1)
        lo = torch.nan_to_num(coord, nan=0.0).floor().clamp(
            min=torch.zeros_like(shape), max=shape - 2).to(torch.int64)
        frac = coord - lo.to(coord.dtype)
        Y, X = low_field.shape[2], low_field.shape[3]
        base = (lo[:, 0] * Y + lo[:, 1]) * X + lo[:, 2]
        strides = (Y * X, X, 1)

        flat_low = low_field.reshape(3, -1)
        flat_high = high_field.reshape(3, -1)
        acc = torch.zeros_like(high_field)
        acc_flat = acc.reshape(3, -1)
        corners = _corner_bits(pts.device)
        strides_t = torch.tensor(strides, device=pts.device, dtype=torch.int64)
        idx_all = base[:, None] + (corners * strides_t).sum(-1)[None, :]
        f = torch.stack([1 - frac, frac], dim=-1)
        fz = f[:, 0, corners[:, 0]]
        fy = f[:, 1, corners[:, 1]]
        fx = f[:, 2, corners[:, 2]]
        w_all = fz * fy * fx
        vals = (grad_out[:, None, :] * w_all[..., None]).permute(2, 0, 1).reshape(3, -1)
        acc_flat.index_add_(1, idx_all.reshape(-1), vals)
        flat_indices = idx_all.reshape(-1)
        gathered = (
            flat_low[:, flat_indices]
            + flat_high[:, flat_indices] * high_scale
        ).view(3, *idx_all.shape)
        v_dot_g = (gathered * grad_out.T[:, :, None]).sum(0)
        signs = (2 * corners - 1).to(grad_out.dtype)
        grad_coord = torch.stack([
            (v_dot_g * signs[:, 0] * fy * fx).sum(1),
            (v_dot_g * signs[:, 1] * fz * fx).sum(1),
            (v_dot_g * signs[:, 2] * fz * fy).sum(1),
        ], dim=-1)
        unclipped = (coord_raw >= 0) & (coord_raw <= shape - 1)
        return grad_coord * unclipped.to(grad_coord.dtype) * (shape - 1), acc

    def _run_case(self, device):
        from flow_fields import _SparseAccumTrilinearSample
        torch.manual_seed(7)
        low = (torch.randn(3, 9, 11, 10) * 0.1).to(device)
        high = (torch.randn(3, 9, 11, 10) * 0.1).to(device)
        # Include out-of-range points so the border clamp path is exercised.
        pts = (torch.rand(97, 3) * 1.2 - 0.1).to(device)
        pts.requires_grad_(True)
        grad_out = torch.randn(97, 3).to(device)
        acc = torch.zeros_like(high)

        ref_grad_pts, ref_acc = self._reference_backward(
            pts.detach(), low, high, 0.2, grad_out)

        # Repeat past the TorchScript profiling-executor warm-up (the scripted
        # backward switches to its optimised plan after a couple of calls) so
        # the comparison also covers the steady-state execution plan.
        for _ in range(5):
            pts.grad = None
            acc.zero_()
            out = _SparseAccumTrilinearSample.apply(pts, low, high, 0.2, acc)
            out.backward(gradient=grad_out)

            # The coordinate gradient uses no atomics, so it must match bitwise.
            self.assertTrue(torch.equal(pts.grad, ref_grad_pts))
            if device == 'cpu':
                self.assertTrue(torch.equal(acc, ref_acc))
            else:
                # CUDA index_add_ is atomic (accumulation order varies run to
                # run), so the field-gradient buffer is compared with a tolerance.
                torch.testing.assert_close(acc, ref_acc, rtol=1e-5, atol=1e-7)

    def test_backward_bitwise_matches_uncached_reference(self):
        self._run_case('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
    def test_backward_bitwise_matches_uncached_reference_cuda(self):
        self._run_case('cuda')


class RK4FusedIntegratorTests(unittest.TestCase):
    # The fused single-node RK4 integrator must match the eager sampler-based
    # loop bitwise: identical forward arithmetic, and a hand-written adjoint
    # that reproduces autograd's exact op/accumulation order.

    @staticmethod
    def _make_flow(seed):
        torch.manual_seed(seed)
        flow = CartesianFlowField(torch.tensor([12, 12, 12]), spatial_scale_factor=6, lr_scale_factor=0.2)
        with torch.no_grad():
            flow.flows[0].normal_(std=0.1)
            flow.flows[1].normal_(std=0.1)
        return flow

    @staticmethod
    def _reference_integrate(flow, y, h, n_steps):
        # The pre-change eager loop from IntegratedFlowDiffeomorphism._call.
        sampler = flow.get_sampler(0.0)
        for _ in range(n_steps):
            k1 = sampler(y)
            k2 = sampler(y + (h / 2) * k1)
            k3 = sampler(y + (h / 2) * k2)
            k4 = sampler(y + h * k3)
            y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y

    def test_forward_and_grads_bitwise_match_sampler_loop(self):
        for h in (1.0 / 3, -1.0 / 3):  # forward and inverse integration directions
            ref = self._make_flow(21)
            fused = self._make_flow(21)
            with torch.no_grad():
                fused.load_state_dict(ref.state_dict())
            pts = torch.rand(63, 3) * 1.1 - 0.05

            ref_pts = pts.clone().requires_grad_(True)
            ref_out = self._reference_integrate(ref, ref_pts, h, 3)
            ref_out.square().mean().backward()
            ref.apply_accumulated_field_grad()

            fused_pts = pts.clone().requires_grad_(True)
            fused_out = fused.get_time_invariant_integrator()(fused_pts, h, 3)
            fused_out.square().mean().backward()
            fused.apply_accumulated_field_grad()

            self.assertTrue(torch.equal(fused_out, ref_out))
            self.assertTrue(torch.equal(fused_pts.grad, ref_pts.grad))
            self.assertTrue(torch.equal(fused.flows[0].grad, ref.flows[0].grad))
            self.assertTrue(torch.equal(fused.flows[1].grad, ref.flows[1].grad))

    def test_no_grad_forward_matches(self):
        ref = self._make_flow(22)
        fused = self._make_flow(22)
        with torch.no_grad():
            fused.load_state_dict(ref.state_dict())
        pts = torch.rand(41, 3)
        with torch.no_grad():
            ref_out = self._reference_integrate(ref, pts, 1.0 / 3, 3)
            fused_out = fused.get_time_invariant_integrator()(pts, 1.0 / 3, 3)
        self.assertTrue(torch.equal(fused_out, ref_out))


class CachedSamplerGradModeUpgradeTests(unittest.TestCase):
    def test_no_grad_first_call_does_not_poison_field_gradients(self):
        torch.manual_seed(11)
        flow = CartesianFlowField(torch.tensor([12, 12, 12]), spatial_scale_factor=6, lr_scale_factor=0.2)
        with torch.no_grad():
            flow.flows[0].normal_(std=0.05)
            flow.flows[1].normal_(std=0.05)
        diffeo = IntegratedFlowDiffeomorphism(
            flow,
            flow_min_corner_zyx=torch.tensor([0., 0., 0.]),
            flow_max_corner_zyx=torch.tensor([12., 12., 12.]),
            num_steps=3,
            solver='rk4',
        )
        # In the fitter the sampled coordinates are outputs of earlier trainable
        # transforms; requires_grad on the points stands in for that.
        pts = (torch.rand(17, 3) * 12.).requires_grad_(True)

        with torch.no_grad():
            out_nograd = diffeo._call(pts)

        out = diffeo._call(pts)
        self.assertTrue(torch.equal(out.detach(), out_nograd))
        out.square().sum().backward()
        flow.apply_accumulated_field_grad()
        self.assertIsNotNone(flow.flows[0].grad)
        self.assertIsNotNone(flow.flows[1].grad)
        self.assertGreater(flow.flows[1].grad.abs().max().item(), 0.)


class TritonRK4IntegrateTests(unittest.TestCase):
    # The Triton fused RK4 integrator (flow_triton) runs the same arithmetic
    # as _RK4SparseFlowIntegrate but lets the compiler contract mul+add into
    # FMAs, so equivalence is tolerance-based (last-ulp scale on smooth
    # fields), unlike the bitwise tests above. The field-gradient accumulator
    # additionally carries the usual atomic-order noise.

    @staticmethod
    def _skip_unless_available():
        import flow_triton
        if not torch.cuda.is_available():
            raise unittest.SkipTest('needs CUDA')
        if not flow_triton.rk4_triton_available(torch.zeros(1, device='cuda')):
            raise unittest.SkipTest('triton unavailable or disabled')

    def test_matches_eager_within_fma_tolerance(self):
        self._skip_unless_available()
        import flow_triton
        from flow_fields import _RK4SparseFlowIntegrate
        for seed, n_steps, h in [(0, 3, 1.0 / 3), (1, 3, -1.0 / 3), (2, 1, 0.5)]:
            torch.manual_seed(seed)
            # Smooth, small fields comparable to real trained flows; random
            # high-frequency fields amplify ulp differences chaotically
            # through the integration and test nothing useful.
            low = (torch.randn(3, 9, 11, 10) * 1e-2).cuda()
            high = (torch.randn(3, 9, 11, 10) * 1e-2).cuda()
            pts = (torch.rand(4097, 3) * 1.2 - 0.1).cuda()
            grad_out = torch.randn(4097, 3).cuda()

            def run(fn):
                p = pts.clone().requires_grad_(True)
                acc = torch.zeros_like(high)
                out = fn(p, low, high, 0.2, acc, h, n_steps)
                out.backward(gradient=grad_out)
                return out.detach(), p.grad, acc

            o_ref, g_ref, a_ref = run(_RK4SparseFlowIntegrate.apply)
            o_tri, g_tri, a_tri = run(flow_triton.rk4_integrate)
            torch.testing.assert_close(o_tri, o_ref, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(g_tri, g_ref, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(a_tri, a_ref, rtol=1e-4, atol=1e-6)

    def test_no_grad_forward_matches(self):
        self._skip_unless_available()
        import flow_triton
        from flow_fields import _RK4SparseFlowIntegrate
        torch.manual_seed(3)
        low = (torch.randn(3, 9, 11, 10) * 1e-2).cuda()
        high = (torch.randn(3, 9, 11, 10) * 1e-2).cuda()
        pts = (torch.rand(999, 3) * 1.2 - 0.1).cuda()
        with torch.no_grad():
            o_ref = _RK4SparseFlowIntegrate.apply(pts, low, high, 0.2, None, 0.25, 4)
            o_tri = flow_triton.rk4_integrate(pts, low, high, 0.2, None, 0.25, 4)
        torch.testing.assert_close(o_tri, o_ref, rtol=1e-5, atol=1e-6)


class TritonGapExpanderTests(unittest.TestCase):
    # The fused gap-expander kernels (gap_triton) match the eager pipeline up
    # to (a) sequential-vs-parallel-scan cumsum order, (b) FMA contraction,
    # and (c) gradient attribution for points within an ulp of a logit-lattice
    # cell boundary landing in the neighboring cell (mass-preserving), so the
    # logit gradient is compared after a 3x3 box filter.

    def _run_case(self, direction, tf, seed=0, n=4096):
        import gap_triton
        if not torch.cuda.is_available():
            raise unittest.SkipTest('needs CUDA')
        if not gap_triton.gap_triton_available(torch.zeros(1, device='cuda')):
            raise unittest.SkipTest('triton unavailable or disabled')
        import os
        dev = 'cuda'
        torch.manual_seed(seed)
        params = GapExpanderParams(resolution=24, min_z=0.0, max_z=500.0,
                                   num_windings=30, dr_per_winding=16.0).to(dev)
        with torch.no_grad():
            params.logits.normal_(std=3e-3)
        dr_param = torch.tensor(16.0 / 12.0, device=dev, requires_grad=True)
        pts = (torch.randn(n, 3, device=dev)
               * torch.tensor([120., 140., 140.], device=dev)
               + torch.tensor([250., 0., 0.], device=dev)).requires_grad_(True)
        g_out = torch.randn(n, 3, device=dev)

        def go(enabled):
            prev = os.environ.get('FIT_SPIRAL_TRITON')
            os.environ['FIT_SPIRAL_TRITON'] = '1' if enabled else '0'
            try:
                params.logits.grad = None
                dr_param.grad = None
                pts.grad = None
                dr = F.softplus(dr_param * 12.)
                t = GapExpandingTransform(params, dr, 0.0, 500.0,
                                          gap_expander_lr_scale=0.3, truncate_frac=tf)
                out = t._call(pts) if direction == 'call' else t._inverse(pts)
                out.backward(gradient=g_out)
                return (out.detach().clone(), pts.grad.clone(),
                        params.logits.grad.clone(), dr_param.grad.clone())
            finally:
                if prev is None:
                    os.environ.pop('FIT_SPIRAL_TRITON', None)
                else:
                    os.environ['FIT_SPIRAL_TRITON'] = prev

        o_e, gp_e, gl_e, gd_e = go(False)
        o_t, gp_t, gl_t, gd_t = go(True)
        torch.testing.assert_close(o_t, o_e, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(gp_t, gp_e, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(gd_t, gd_e, rtol=1e-3, atol=1e-4)
        k = torch.ones(1, 1, 3, 3, device=dev) / 9.
        atol = 1e-5 * gl_e.abs().max().item()
        torch.testing.assert_close(F.conv2d(gl_t, k, padding=1),
                                   F.conv2d(gl_e, k, padding=1),
                                   rtol=1e-3, atol=atol)
        self.assertLess(abs((gl_t - gl_e).sum().item()),
                        1e-6 * gl_e.abs().sum().item())

    def test_call_matches_eager(self):
        self._run_case('call', None)
        self._run_case('call', 0.37)

    def test_inverse_matches_eager(self):
        self._run_case('inverse', None)
        self._run_case('inverse', 0.37)


class BilinearAtlasLookupClampTest(unittest.TestCase):
    """Edge samples must stay inside their patch (float32-rounded jitter can
    reach exactly 1.0 and floor one cell past the last valid quad row/col)."""

    def _make_atlas(self):
        # two patches with distinct, recognisable values
        a = torch.arange(4 * 5 * 3, dtype=torch.float32).reshape(4, 5, 3)
        b = 1000. + torch.arange(3 * 4 * 3, dtype=torch.float32).reshape(3, 4, 3)
        zyxs_flat = torch.cat([a.reshape(-1, 3), b.reshape(-1, 3)], dim=0)
        offsets = torch.tensor([0, 20, 32], dtype=torch.int64)
        widths = torch.tensor([5, 4], dtype=torch.int64)
        heights = torch.tensor([4, 3], dtype=torch.int64)
        return a, b, zyxs_flat, offsets, widths, heights

    def test_edge_sample_stays_in_patch(self):
        from geom_utils import bilinear_atlas_lookup
        a, b, zyxs_flat, offsets, widths, heights = self._make_atlas()
        # i lands exactly on the last row of patch 0 (jitter rounded to 1.0)
        ij = torch.tensor([[3.0, 2.5]], dtype=torch.float32)
        idx = torch.tensor([0], dtype=torch.int64)
        out = bilinear_atlas_lookup(zyxs_flat, offsets, widths, idx, ij,
                                    heights=heights)
        # clamped to the (2, 2)-(3, 3) quad at di=1: row 3 interpolated at j=2.5
        expected = (a[3, 2] + a[3, 3]) / 2
        torch.testing.assert_close(out[0], expected)
        # without the clamp the same sample reads into patch 1's first row
        leaked = bilinear_atlas_lookup(zyxs_flat, offsets, widths, idx, ij)
        self.assertGreater(float(leaked.max()), 999.)

    def test_last_patch_edge_does_not_index_oob(self):
        from geom_utils import bilinear_atlas_lookup
        _a, _b, zyxs_flat, offsets, widths, heights = self._make_atlas()
        ij = torch.tensor([[2.0, 3.0]], dtype=torch.float32)  # last row+col
        idx = torch.tensor([1], dtype=torch.int64)
        out = bilinear_atlas_lookup(zyxs_flat, offsets, widths, idx, ij,
                                    heights=heights)
        self.assertTrue(bool(torch.isfinite(out).all()))
        with self.assertRaises(IndexError):
            bilinear_atlas_lookup(zyxs_flat, offsets, widths, idx, ij)


if __name__ == '__main__':
    unittest.main()
