import numpy as np
import torch
import torch.nn as nn


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
