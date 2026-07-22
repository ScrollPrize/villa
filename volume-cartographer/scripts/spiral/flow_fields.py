from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import flow_triton


def sample_field(normalised_zyx, field_for_grid_sample):
    # normalised_zyx :: *, zyx in [0, 1]; field_for_grid_sample :: zyx, z, y, x
    orig_shape = normalised_zyx.shape
    zyx = (normalised_zyx * 2. - 1.).view(1, -1, 1, 1, 3)
    field_samples = F.grid_sample(
        input=field_for_grid_sample[None],
        grid=zyx.flip(-1),
        align_corners=True,
        mode='bilinear',
        padding_mode='border',
    )  # 1, zyx, n, 1, 1
    return field_samples.squeeze(0).squeeze(-2).squeeze(-1).T.view(*orig_shape[:-1], 3)  # *, zyx


_CORNER_BITS_CACHE = {}


def _corner_bits(device):
    t = _CORNER_BITS_CACHE.get(device)
    if t is None:
        t = torch.tensor(
            [[dz, dy, dx] for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)],
            device=device,
            dtype=torch.int64,
        )
        _CORNER_BITS_CACHE[device] = t
    return t


_FIELD_CONSTS_CACHE = {}


def _field_consts(field, pts_dtype):
    # Per-(shape, device, dtype) constant tensors used by the sampler backward.
    # That backward runs for every RK4 stage of every transform call (hundreds
    # of times per training step), and rebuilding these tiny tensors each call
    # costs a pageable host->device copy plus a stream sync per tensor. The
    # cached values are identical constants, so results are unchanged.
    key = (tuple(field.shape), field.device, pts_dtype)
    consts = _FIELD_CONSTS_CACHE.get(key)
    if consts is None:
        device = field.device
        shape = torch.tensor(field.shape[1:], device=device, dtype=pts_dtype)
        Y, X = field.shape[2], field.shape[3]
        corners = _corner_bits(device)
        strides_t = torch.tensor((Y * X, X, 1), device=device, dtype=torch.int64)
        consts = {
            'shape': shape,
            'zeros': torch.zeros_like(shape),
            'shape_m1': shape - 1,
            'shape_m2': shape - 2,
            # (corners * strides).sum(-1) is constant integer math; precompute.
            'corner_offsets': (corners * strides_t).sum(-1),
            'signs': (2 * corners - 1).to(pts_dtype),
        }
        _FIELD_CONSTS_CACHE[key] = consts
    return consts


@torch.jit.script
def _sparse_backward_impl(
    grad_out,
    pts,
    low_field,
    high_field,
    high_scale: float,
    acc: Optional[torch.Tensor],
    shape_m1,
    zeros3,
    shape_m2,
    corner_offsets,
    signs,
    corners,
    Y: int,
    X: int,
):
    # Body of _SparseAccumTrilinearSample.backward. This runs for every RK4
    # stage of every transform call (hundreds of times per training step); the
    # ~25 aten calls are unchanged (same kernels, same order -> same values),
    # but TorchScript dispatches them from C++, removing most of the Python
    # per-op overhead that otherwise makes the backward CPU-bound.
    coord_raw = pts * shape_m1
    coord = coord_raw.clamp(min=zeros3, max=shape_m1)
    lo = torch.nan_to_num(coord, nan=0.0).floor().clamp(
        min=zeros3,
        max=shape_m2,
    ).to(torch.int64)
    frac = coord - lo.to(coord.dtype)
    base = (lo[:, 0] * Y + lo[:, 1]) * X + lo[:, 2]

    flat_low = low_field.reshape(3, -1)
    flat_high = high_field.reshape(3, -1)

    idx_all = base[:, None] + corner_offsets[None, :]

    f = torch.stack([1 - frac, frac], dim=-1)
    fz = f[:, 0, corners[:, 0]]
    fy = f[:, 1, corners[:, 1]]
    fx = f[:, 2, corners[:, 2]]
    w_all = fz * fy * fx

    vals = (grad_out[:, None, :] * w_all[..., None]).permute(2, 0, 1).reshape(3, -1)
    if acc is not None:
        acc_flat = acc.reshape(3, -1)
        acc_flat.index_add_(1, idx_all.reshape(-1), vals)

    flat_indices = idx_all.reshape(-1)
    gathered = (
        flat_low[:, flat_indices]
        + flat_high[:, flat_indices] * high_scale
    ).view(3, idx_all.shape[0], idx_all.shape[1])
    v_dot_g = (gathered * grad_out.T[:, :, None]).sum(0)
    grad_coord = torch.stack([
        (v_dot_g * signs[:, 0] * fy * fx).sum(1),
        (v_dot_g * signs[:, 1] * fz * fx).sum(1),
        (v_dot_g * signs[:, 2] * fz * fy).sum(1),
    ], dim=-1)
    unclipped = (coord_raw >= 0) & (coord_raw <= shape_m1)
    return grad_coord * unclipped.to(grad_coord.dtype) * shape_m1


class _SparseAccumTrilinearSample(torch.autograd.Function):
    # Trilinear sampling matching sample_field / grid_sample(align_corners=True,
    # padding_mode='border'), with sparse field-gradient accumulation into a
    # caller-owned dense buffer. This avoids materialising one full field-sized
    # gradient tensor per sampler call in the RK4 integration loop.

    @staticmethod
    def forward(ctx, pts, low_field, high_field, high_scale, acc):
        ctx.set_materialize_grads(False)
        # Keep the two full-resolution sources separate.  Materialising their
        # sum costs another complete Cartesian field (about 1.49 GiB with the
        # default full-scroll configuration), while sparse sampling is linear.
        out = (
            sample_field(pts, low_field)
            + sample_field(pts, high_field) * high_scale
        )
        ctx.save_for_backward(pts, low_field, high_field)
        ctx.high_scale = high_scale
        ctx.acc = acc
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None, None, None
        pts, low_field, high_field = ctx.saved_tensors
        consts = _field_consts(low_field, pts.dtype)
        if consts['signs'].dtype == grad_out.dtype:
            signs = consts['signs']
        else:
            signs = (2 * _corner_bits(pts.device) - 1).to(grad_out.dtype)
        grad_pts = _sparse_backward_impl(
            grad_out,
            pts,
            low_field,
            high_field,
            float(ctx.high_scale),
            ctx.acc,
            consts['shape_m1'],
            consts['zeros'],
            consts['shape_m2'],
            consts['corner_offsets'],
            signs,
            _corner_bits(pts.device),
            low_field.shape[2],
            low_field.shape[3],
        )
        return grad_pts, None, None, None, None


class _RK4SparseFlowIntegrate(torch.autograd.Function):
    # The whole time-invariant RK4 integration as a single autograd node.
    #
    # The eager sampler-based loop creates ~42 autograd nodes per transform
    # call (12 sampler Functions plus the stage arithmetic), and the engine's
    # per-node dispatch dominates step time. This Function runs the identical
    # forward arithmetic, saves the 12 stage points (the same tensors the
    # eager graph would keep alive via the sampler nodes' save_for_backward),
    # and hand-writes the adjoint sweep in backward.
    #
    # The backward reproduces the eager graph's exact floating-point operation
    # and gradient-accumulation order (verified bitwise in
    # tests/test_speedup_equivalence.py):
    #   k_bar4 = g*(h/6)
    #   k_bar3 = (g*(h/6))*2 + x_bar4*h
    #   k_bar2 = (g*(h/6))*2 + x_bar3*(h/2)
    #   k_bar1 =  g*(h/6)    + x_bar2*(h/2)
    #   g_prev = (((g + x_bar4) + x_bar3) + x_bar2) + x_bar1
    # where x_bar_i is the sampler backward at stage point x_i (which also
    # accumulates the field gradient into `acc`, in the same k4..k1 order the
    # autograd engine used).

    @staticmethod
    def forward(ctx, y0, low_field, high_field, high_scale, acc, h, n_steps):
        ctx.set_materialize_grads(False)
        stage_points = []
        y = y0
        for _ in range(n_steps):
            k1 = sample_field(y, low_field) + sample_field(y, high_field) * high_scale
            x2 = y + (h / 2) * k1
            k2 = sample_field(x2, low_field) + sample_field(x2, high_field) * high_scale
            x3 = y + (h / 2) * k2
            k3 = sample_field(x3, low_field) + sample_field(x3, high_field) * high_scale
            x4 = y + h * k3
            k4 = sample_field(x4, low_field) + sample_field(x4, high_field) * high_scale
            stage_points.extend([y, x2, x3, x4])
            y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        ctx.save_for_backward(low_field, high_field, *stage_points)
        ctx.high_scale = float(high_scale)
        ctx.acc = acc
        ctx.h = float(h)
        ctx.n_steps = int(n_steps)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        if grad_y is None:
            return None, None, None, None, None, None, None
        saved = ctx.saved_tensors
        low_field, high_field = saved[0], saved[1]
        stage_points = saved[2:]
        h = ctx.h
        scale = ctx.high_scale
        acc = ctx.acc
        consts = _field_consts(low_field, grad_y.dtype)
        corners = _corner_bits(grad_y.device)
        if consts['signs'].dtype == grad_y.dtype:
            signs = consts['signs']
        else:
            signs = (2 * corners - 1).to(grad_y.dtype)
        Y, X = low_field.shape[2], low_field.shape[3]

        def stage_grad(g, pts):
            return _sparse_backward_impl(
                g, pts, low_field, high_field, scale, acc,
                consts['shape_m1'], consts['zeros'], consts['shape_m2'],
                consts['corner_offsets'], signs, corners, Y, X,
            )

        for step in range(ctx.n_steps - 1, -1, -1):
            x1, x2, x3, x4 = stage_points[4 * step: 4 * step + 4]
            g6 = grad_y * (h / 6)
            xb4 = stage_grad(g6, x4)
            xb3 = stage_grad(g6 * 2 + xb4 * h, x3)
            xb2 = stage_grad(g6 * 2 + xb3 * (h / 2), x2)
            xb1 = stage_grad(g6 + xb2 * (h / 2), x1)
            grad_y = grad_y + xb4
            grad_y = grad_y + xb3
            grad_y = grad_y + xb2
            grad_y = grad_y + xb1
        return grad_y, None, None, None, None, None, None


class CartesianFlowField(nn.Module):

    def __init__(self, resolution, spatial_scale_factor=6, lr_scale_factor=1.e-1, num_flow_timesteps=1):
        super().__init__()
        self.num_flow_timesteps = num_flow_timesteps
        self.flow_scales = [1., lr_scale_factor]
        self.flows = nn.ParameterList([
            nn.Parameter(torch.zeros([num_flow_timesteps, 3, *shape]))
            for shape in [
                [resolution[0] // spatial_scale_factor, resolution[1] // spatial_scale_factor, resolution[2] // spatial_scale_factor],
                resolution,
            ]
        ])
        self._pending_lr_upsampled = None
        self._pending_hr_scale = None
        self._field_grad_acc = None
        self._lr_grad_acc = None
        self._pending_direct = False

    def _prepare_time_invariant_fields(self):
        # Shared state for get_sampler / get_time_invariant_integrator when
        # num_flow_timesteps == 1. Returns (low_field, high_field, high_scale,
        # acc); acc is None outside training (no grad, or frozen fields).
        lr_flow, hr_flow = self.flows[0], self.flows[1]
        hr_shape = tuple(hr_flow.shape[2:])
        # Time-invariant: HR flow is already at the target resolution, so skip interpolating it.
        lr_upsampled = F.interpolate(lr_flow, size=hr_shape, mode='trilinear')[0] * self.flow_scales[0]
        training_field = torch.is_grad_enabled() and (lr_flow.requires_grad or hr_flow.requires_grad)
        if not training_field:
            return lr_upsampled, hr_flow[0], float(self.flow_scales[1]), None

        # Sampling keeps the detached LR-upsampled and HR fields separate and
        # accumulates dL/d(their effective sum) in one caller-owned buffer.
        # Keep only the LR interpolation graph; after all point losses have
        # backpropagated, its gradient is propagated normally while the same
        # full-resolution accumulator becomes the HR parameter gradient.
        low_field = lr_upsampled.detach()
        high_field = hr_flow[0].detach()
        if self._field_grad_acc is None or self._field_grad_acc.shape != high_field.shape:
            self._field_grad_acc = torch.zeros_like(high_field)
        else:
            self._field_grad_acc.zero_()
        self._pending_lr_upsampled = lr_upsampled
        self._pending_hr_scale = float(self.flow_scales[1])
        return low_field, high_field, self._pending_hr_scale, self._field_grad_acc

    def _get_direct_integrator(self):
        # Direct-LR mode: sample the LR lattice at query points instead of
        # upsampling it to HR every step (FIT_SPIRAL_DIRECT_LR=1). Field
        # gradients accumulate unscaled into per-lattice buffers; the value
        # scales are applied in apply_accumulated_field_grad.
        lr_flow, hr_flow = self.flows[0], self.flows[1]
        low, high = lr_flow[0], hr_flow[0]
        training_field = torch.is_grad_enabled() and (
            lr_flow.requires_grad or hr_flow.requires_grad)
        acc_lo = acc_hi = None
        if training_field:
            low, high = low.detach(), high.detach()
            if self._lr_grad_acc is None or self._lr_grad_acc.shape != low.shape:
                self._lr_grad_acc = torch.zeros_like(low)
            else:
                self._lr_grad_acc.zero_()
            if self._field_grad_acc is None or self._field_grad_acc.shape != high.shape:
                self._field_grad_acc = torch.zeros_like(high)
            else:
                self._field_grad_acc.zero_()
            acc_lo, acc_hi = self._lr_grad_acc, self._field_grad_acc
            # Only one pending-gradient record may be armed per iteration; a
            # stale upsampled-mode record would fire on the shared
            # accumulator after the direct branch consumed it.
            assert self._pending_lr_upsampled is None
            self._pending_direct = True
        lo_scale, hi_scale = float(self.flow_scales[0]), float(self.flow_scales[1])
        low_c, high_c = low.contiguous(), high.contiguous()

        def integrate(y_flat, h, n_steps):
            return flow_triton.rk4_direct_integrate(
                y_flat, low_c, high_c, lo_scale, hi_scale, acc_lo, acc_hi, h, n_steps)

        return integrate

    def get_time_invariant_integrator(self):
        # Returns integrate(y_flat, h, n_steps) -> y_flat running the whole RK4
        # integration as one autograd node (see _RK4SparseFlowIntegrate). Only
        # valid for num_flow_timesteps == 1.
        assert self.num_flow_timesteps == 1
        if (flow_triton.direct_lr_enabled()
                and flow_triton.rk4_triton_available(self.flows[0], self.flows[1])):
            return self._get_direct_integrator()
        low_field, high_field, high_scale, acc = self._prepare_time_invariant_fields()

        if flow_triton.rk4_triton_available(low_field, high_field, acc):
            low_c, high_c = low_field.contiguous(), high_field.contiguous()

            def integrate(y_flat, h, n_steps):
                return flow_triton.rk4_integrate(
                    y_flat, low_c, high_c, high_scale, acc, h, n_steps,
                )

            return integrate

        def integrate(y_flat, h, n_steps):
            if not (torch.is_grad_enabled() and y_flat.requires_grad):
                # No adjoint sweep will run; the plain sampler loop avoids the
                # Function's transient retention of all 4*n_steps stage points
                # (matters for large no-grad evals like previews/satisfaction).
                y = y_flat
                for _ in range(n_steps):
                    k1 = sample_field(y, low_field) + sample_field(y, high_field) * high_scale
                    x2 = y + (h / 2) * k1
                    k2 = sample_field(x2, low_field) + sample_field(x2, high_field) * high_scale
                    x3 = y + (h / 2) * k2
                    k3 = sample_field(x3, low_field) + sample_field(x3, high_field) * high_scale
                    x4 = y + h * k3
                    k4 = sample_field(x4, low_field) + sample_field(x4, high_field) * high_scale
                    y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                return y
            return _RK4SparseFlowIntegrate.apply(
                y_flat, low_field, high_field, high_scale, acc, h, n_steps,
            )

        return integrate

    def get_sampler(self, t):
        # Returns a callable mapping normalised zyx points in [0, 1] to flow velocity at time t.
        # Materialises the flow as a [3, Z, Y, X] cartesian tensor of zyx vector components once
        # and reuses it across the (e.g. RK4) integrator's many sample calls.
        lr_flow, hr_flow = self.flows[0], self.flows[1]
        hr_shape = tuple(hr_flow.shape[2:])
        if self.num_flow_timesteps == 1:
            low_field, high_field, high_scale, acc = self._prepare_time_invariant_fields()
            if acc is None:
                return lambda y: (
                    sample_field(y, low_field)
                    + sample_field(y, high_field) * high_scale
                )

            def sample(normalised_zyx):
                flat = normalised_zyx.reshape(-1, 3)
                return _SparseAccumTrilinearSample.apply(
                    flat,
                    low_field,
                    high_field,
                    high_scale,
                    acc,
                ).view(*normalised_zyx.shape[:-1], 3)

            return sample
        else:
            t_scaled = (t.clamp(-1. + 1.e-4, 1. - 1.e-4) + 1) / 2 * (self.num_flow_timesteps - 1)
            t_idx_before = int(t_scaled)
            flows_interpolated = [
                F.interpolate(flow[t_idx_before : t_idx_before + 2], size=hr_shape, mode='trilinear') * flow_scale
                for flow, flow_scale in zip(self.flows, self.flow_scales)
            ]
            field = sum(
                torch.lerp(flow_interpolated[0], flow_interpolated[1], t_scaled % 1.)
                for flow_interpolated in flows_interpolated
            )
        return lambda y: sample_field(y, field)

    def apply_accumulated_field_grad(self):
        if self._pending_direct:
            self._pending_direct = False
            for param, acc, scale in (
                (self.flows[0], self._lr_grad_acc, self.flow_scales[0]),
                (self.flows[1], self._field_grad_acc, self.flow_scales[1]),
            ):
                # Reuse the accumulator storage as the parameter gradient.
                acc.mul_(scale)
                grad = acc.unsqueeze(0)
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad.add_(grad)
            return

        if self._pending_lr_upsampled is None:
            return

        # dL/dLR is the interpolation backward of dL/dfield.
        self._pending_lr_upsampled.backward(gradient=self._field_grad_acc)
        self._pending_lr_upsampled = None

        # dL/dHR = scale * dL/dfield.  Reuse the accumulator storage as the
        # parameter's gradient instead of materialising another [3,Z,Y,X] tensor.
        self._field_grad_acc.mul_(self._pending_hr_scale)
        self._pending_hr_scale = None
        hr_param = self.flows[1]
        hr_grad = self._field_grad_acc.unsqueeze(0)
        if hr_param.grad is None:
            hr_param.grad = hr_grad
        else:
            hr_param.grad.add_(hr_grad)


class CylindricalFlowField(nn.Module):

    # Flow field with parameters on a cylindrical lattice (z, r, phi). The cylinder axis lies
    # along z at the centre of the y, x box; the lattice spans z=[0,Z) and the inscribed disk in
    # y, x (radius<=1 in normalised cartesian; corners outside the disk are clamped on r). Stored
    # per-cell vectors are in the local (z, radial, tangential) basis: component 1 points outward
    # radially, component 2 in the direction of increasing phi (right-hand rule about +z). The
    # integrator samples the lattice directly at cartesian query points and rotates the (r, phi)
    # components into (y, x) on the fly using the local basis at each query point.
    #
    # Rings have *varying* numbers of angular cells: ring r holds num_phi[r] = max(1, round(2*pi*r))
    # cells (= circumference / lattice radial spacing), so inner rings are coarse and outer rings
    # fine. All rings are packed end-to-end along the last (phi) axis of the parameter tensor,
    # which is therefore "ragged"; sampling does explicit per-query gathers (one per surrounding
    # corner of the (z, r, phi) trilinear stencil).
    #
    # Note: near r=0 the cylindrical basis is degenerate; ring 0 holds a single cell that is
    # pinned to zero.

    def __init__(self, resolution, spatial_scale_factor=6, lr_scale_factor=1.e-1, num_flow_timesteps=1):
        # resolution is interpreted as the equivalent cartesian (Z, Y, X) voxel shape; the
        # cylindrical lattice sizes are derived from it.
        super().__init__()
        self.num_flow_timesteps = num_flow_timesteps
        Z, Y, X = (int(s) for s in resolution)

        nz_hr = Z
        nr_hr = max(2, min(Y, X) // 2)
        nz_lr = max(2, nz_hr // spatial_scale_factor)
        nr_lr = max(2, nr_hr // spatial_scale_factor)

        # The lr lattice has spatial_scale_factor-wider rings, so its ring r covers the same
        # circumference as the hr ring r*spatial_scale_factor; the factors cancel in
        # "cells per (sub-)ring unit", so the same 2*pi*r formula applies to both lattices.
        def compute_num_phi(nr):
            return torch.tensor(
                [1 if r == 0 else max(1, int(round(2 * np.pi * r))) for r in range(nr)],
                dtype=torch.long,
            )

        lr_num_phi = compute_num_phi(nr_lr)
        hr_num_phi = compute_num_phi(nr_hr)
        lr_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lr_num_phi, dim=0)])
        hr_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(hr_num_phi, dim=0)])
        self.register_buffer('_lr_num_phi', lr_num_phi)
        self.register_buffer('_hr_num_phi', hr_num_phi)
        self.register_buffer('_lr_offsets', lr_offsets)
        self.register_buffer('_hr_offsets', hr_offsets)

        self.flow_scales = [1., lr_scale_factor]
        self.flows = nn.ParameterList([
            nn.Parameter(torch.zeros([num_flow_timesteps, 3, nz_lr, int(lr_offsets[-1])])),
            nn.Parameter(torch.zeros([num_flow_timesteps, 3, nz_hr, int(hr_offsets[-1])])),
        ])

    @staticmethod
    def _sample_lattice(field, ring_num_phi, ring_offsets, normalised_zyx):
        # field :: 3, nz, total_phi -- rings packed end-to-end along the last axis
        # ring_num_phi :: nr (long) -- per-ring phi cell counts
        # ring_offsets :: nr+1 (long) -- cumulative ring start offsets in the flat phi axis
        # normalised_zyx :: *, 3 in [0, 1] (cartesian box-relative)
        # Returns: *, 3 with components in cartesian (z, y, x).
        nz = field.shape[1]
        nr = ring_num_phi.shape[0]
        orig_shape = normalised_zyx.shape
        pts = normalised_zyx.reshape(-1, 3) * 2. - 1.  # n, 3 in [-1, 1] cartesian
        z_n, y_n, x_n = pts[:, 0], pts[:, 1], pts[:, 2]
        # The cylindrical basis is singular exactly on the axis: sqrt(0) and atan2(0, 0)
        # have finite forward values but undefined gradients. Use a fixed +x basis there.
        axis_eps = torch.finfo(pts.dtype).eps
        on_axis = (y_n.abs() <= axis_eps) & (x_n.abs() <= axis_eps)
        safe_y_n = torch.where(on_axis, torch.zeros_like(y_n), y_n)
        safe_x_n = torch.where(on_axis, torch.ones_like(x_n), x_n)
        rr = torch.sqrt(safe_y_n ** 2 + safe_x_n ** 2).clamp(max=1.)  # inscribed-disk clamp
        rr = torch.where(on_axis, torch.zeros_like(rr), rr)
        phi = torch.atan2(safe_y_n, safe_x_n)  # in (-pi, pi]

        # Map to continuous lattice indices, align_corners=True style.
        z_cont = ((z_n + 1.) * 0.5 * (nz - 1)).clamp(0., float(nz - 1))
        r_cont = rr * (nr - 1)
        phi_in_2pi = phi % (2. * np.pi)  # in [0, 2pi)

        z_lo = torch.floor(z_cont).clamp(max=nz - 2).long()
        z_hi = z_lo + 1
        frac_z = (z_cont - z_lo.to(z_cont.dtype)).unsqueeze(0)  # 1, n

        r_lo = torch.floor(r_cont).clamp(max=nr - 2).long()
        r_hi = r_lo + 1
        frac_r = (r_cont - r_lo.to(r_cont.dtype)).unsqueeze(0)  # 1, n

        def sample_at_ring(r_idx):
            # r_idx :: n (long). Returns 3, n -- bilinear in (z, phi) at this integer ring.
            num_phi_r = ring_num_phi[r_idx]
            offset_r = ring_offsets[r_idx]
            phi_cont = phi_in_2pi * (num_phi_r.to(phi_in_2pi.dtype) / (2. * np.pi))
            phi_lo_floor = torch.floor(phi_cont)
            phi_lo = phi_lo_floor.long() % num_phi_r
            phi_hi = (phi_lo + 1) % num_phi_r  # cyclic wrap
            frac_phi = (phi_cont - phi_lo_floor).unsqueeze(0)
            flat_lo = offset_r + phi_lo
            flat_hi = offset_r + phi_hi
            v00 = field[:, z_lo, flat_lo]
            v01 = field[:, z_lo, flat_hi]
            v10 = field[:, z_hi, flat_lo]
            v11 = field[:, z_hi, flat_hi]
            v0 = v00 + (v01 - v00) * frac_phi
            v1 = v10 + (v11 - v10) * frac_phi
            return v0 + (v1 - v0) * frac_z

        v_rlo = sample_at_ring(r_lo)
        v_rhi = sample_at_ring(r_hi)
        sampled = v_rlo + (v_rhi - v_rlo) * frac_r  # 3, n in (z, r, phi) local components

        z_c, r_c, p_c = sampled[0], sampled[1], sampled[2]
        # phi = atan2(y, x), so outward-radial in (y, x) is (sin(phi), cos(phi)) and tangential
        # (d/dphi unit) is (cos(phi), -sin(phi)). Rotate local (r, phi) components into (y, x).
        sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
        y_c = r_c * sin_phi + p_c * cos_phi
        x_c = r_c * cos_phi - p_c * sin_phi
        return torch.stack([z_c, y_c, x_c], dim=-1).view(*orig_shape)

    def get_sampler(self, t):
        # Returns a callable mapping normalised zyx points in [0, 1] to flow velocity at time t,
        # by sampling the cylindrical lattice directly at each query point. The closure captures
        # the time-interpolated, scale-applied, axis-pinned LR & HR lattices so those one-time
        # costs amortise across the integrator's sample calls.
        if self.num_flow_timesteps == 1:
            lr_field = self.flows[0][0]
            hr_field = self.flows[1][0]
        else:
            t_scaled = (t.clamp(-1. + 1.e-4, 1. - 1.e-4) + 1) / 2 * (self.num_flow_timesteps - 1)
            t_idx_before = int(t_scaled)
            frac = t_scaled % 1.
            lr_field = torch.lerp(self.flows[0][t_idx_before], self.flows[0][t_idx_before + 1], frac)
            hr_field = torch.lerp(self.flows[1][t_idx_before], self.flows[1][t_idx_before + 1], frac)
        # Pin the r=0 ring (axis singularity) to zero by replacing its flat-phi slice with a
        # constant zero, so no gradient flows to those parameters; they stay zero indefinitely.
        n0_lr = int(self._lr_num_phi[0])
        n0_hr = int(self._hr_num_phi[0])
        lr_field = torch.cat([torch.zeros_like(lr_field[:, :, :n0_lr]), lr_field[:, :, n0_lr:]], dim=2) * self.flow_scales[0]
        hr_field = torch.cat([torch.zeros_like(hr_field[:, :, :n0_hr]), hr_field[:, :, n0_hr:]], dim=2) * self.flow_scales[1]

        sample_lattice = self._sample_lattice
        lr_num_phi = self._lr_num_phi
        lr_offsets = self._lr_offsets
        hr_num_phi = self._hr_num_phi
        hr_offsets = self._hr_offsets

        def sample(normalised_zyx):
            return (
                sample_lattice(lr_field, lr_num_phi, lr_offsets, normalised_zyx)
                + sample_lattice(hr_field, hr_num_phi, hr_offsets, normalised_zyx)
            )
        return sample
