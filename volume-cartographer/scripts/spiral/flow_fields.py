import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class _SparseAccumTrilinearSample(torch.autograd.Function):
    # Trilinear sampling matching sample_field / grid_sample(align_corners=True,
    # padding_mode='border'), with sparse field-gradient accumulation into a
    # caller-owned dense buffer. This avoids materialising one full field-sized
    # gradient tensor per sampler call in the RK4 integration loop.

    @staticmethod
    def _corners(pts, field):
        shape = torch.tensor(field.shape[1:], device=pts.device, dtype=pts.dtype)
        coord_raw = pts * (shape - 1)
        coord = coord_raw.clamp(min=torch.zeros_like(shape), max=shape - 1)
        lo = torch.nan_to_num(coord, nan=0.0).floor().clamp(
            min=torch.zeros_like(shape),
            max=shape - 2,
        ).to(torch.int64)
        frac = coord - lo.to(coord.dtype)
        Y, X = field.shape[2], field.shape[3]
        base = (lo[:, 0] * Y + lo[:, 1]) * X + lo[:, 2]
        return coord_raw, shape, frac, base, (Y * X, X, 1)

    @staticmethod
    def forward(ctx, pts, field, acc):
        ctx.set_materialize_grads(False)
        out = sample_field(pts, field)
        ctx.save_for_backward(pts, field)
        ctx.acc = acc
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None
        pts, field = ctx.saved_tensors
        coord_raw, shape, frac, base, strides = _SparseAccumTrilinearSample._corners(pts, field)
        flat_field = field.reshape(3, -1)
        acc_flat = ctx.acc.reshape(3, -1)

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

        gathered = flat_field[:, idx_all.reshape(-1)].view(3, *idx_all.shape)
        v_dot_g = (gathered * grad_out.T[:, :, None]).sum(0)
        signs = (2 * corners - 1).to(grad_out.dtype)
        grad_coord = torch.stack([
            (v_dot_g * signs[:, 0] * fy * fx).sum(1),
            (v_dot_g * signs[:, 1] * fz * fx).sum(1),
            (v_dot_g * signs[:, 2] * fz * fy).sum(1),
        ], dim=-1)
        unclipped = (coord_raw >= 0) & (coord_raw <= shape - 1)
        return grad_coord * unclipped.to(grad_coord.dtype) * (shape - 1), None, None


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

    def get_sampler(self, t):
        # Returns a callable mapping normalised zyx points in [0, 1] to flow velocity at time t.
        # Materialises the flow as a [3, Z, Y, X] cartesian tensor of zyx vector components once
        # and reuses it across the (e.g. RK4) integrator's many sample calls.
        lr_flow, hr_flow = self.flows[0], self.flows[1]
        hr_shape = tuple(hr_flow.shape[2:])
        if self.num_flow_timesteps == 1:
            # Time-invariant: HR flow is already at the target resolution, so skip interpolating it.
            lr_upsampled = F.interpolate(lr_flow, size=hr_shape, mode='trilinear')[0] * self.flow_scales[0]
            training_field = torch.is_grad_enabled() and (lr_flow.requires_grad or hr_flow.requires_grad)
            if not training_field:
                field = lr_upsampled + hr_flow[0] * self.flow_scales[1]
                return lambda y: sample_field(y, field)

            # Sampling uses a detached combined field and accumulates dL/dfield in
            # one caller-owned buffer.  Keep only the small LR interpolation graph;
            # after all point losses have backpropagated, its gradient is propagated
            # normally while the same full-resolution accumulator becomes the HR
            # parameter gradient.  This avoids allocating a second HR-sized gradient.
            field = lr_upsampled.detach() + hr_flow[0].detach() * self.flow_scales[1]
            if self._field_grad_acc is None or self._field_grad_acc.shape != field.shape:
                self._field_grad_acc = torch.zeros_like(field)
            else:
                self._field_grad_acc.zero_()
            self._pending_lr_upsampled = lr_upsampled
            self._pending_hr_scale = float(self.flow_scales[1])
            acc = self._field_grad_acc

            def sample(normalised_zyx):
                flat = normalised_zyx.reshape(-1, 3)
                return _SparseAccumTrilinearSample.apply(flat, field, acc).view(*normalised_zyx.shape[:-1], 3)

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
