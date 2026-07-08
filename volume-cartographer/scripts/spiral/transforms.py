import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions
from einops import rearrange
from torchdiffeq import odeint

import sample_spiral
from flow_fields import CartesianFlowField, CylindricalFlowField
from geom_utils import expm_2x2, interp1d
from sample_spiral import get_bounding_windings, get_theta_and_radii


class IntegratedFlowDiffeomorphism(pyro.distributions.transforms.Transform):

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, flow_field, flow_min_corner_zyx, flow_max_corner_zyx, num_steps, solver, truncate_at_step=None, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.flow_field = flow_field
        self.flow_min_corner_zyx = flow_min_corner_zyx
        self.flow_max_corner_zyx = flow_max_corner_zyx
        self.num_steps = num_steps
        self.solver = solver
        self.truncate_at_step = truncate_at_step
        self._event_dim = event_dim
        self._flow_range_zyx = self.flow_max_corner_zyx - self.flow_min_corner_zyx
        self.num_flow_timesteps = getattr(flow_field, 'num_flow_timesteps', 1)
        # Cached sampler closure at t=0 for the num_flow_timesteps==1 fast path. Built once
        # per diffeomorphism instance (one per training iteration), shared across forward and
        # inverse calls so per-iteration setup (e.g. trilinear LR->HR upsampling) is amortised.
        self._cached_sampler = None

    def _velocity(self, t_int, current_zyx_scaled):
        # t_int is a scalar in [0, 1]; flow_field expects t in [-1, 1]
        t_flow = t_int * 2 - 1
        return self.flow_field.get_sampler(t_flow)(current_zyx_scaled)

    def _call(self, input_zyx, inverse=False):
        # ODE integration of the temporally-varying flow to give a diffeomorphism.
        # The flow & diffeomorphism represent shifts in normalised units [0,1] over the flow region.
        y = (input_zyx - self.flow_min_corner_zyx) / self._flow_range_zyx
        n_steps = self.num_steps if self.truncate_at_step is None else self.truncate_at_step
        t_span = n_steps / self.num_steps
        h = (-t_span if inverse else t_span) / n_steps
        if self.num_flow_timesteps == 1:
            # Time-invariant flow: build the sampler once and inline a manual rk4 loop to skip
            # torchdiffeq's per-step dispatch overhead.
            assert self.solver == 'rk4'
            if self._cached_sampler is None:
                self._cached_sampler = self.flow_field.get_sampler(0.0)
            sampler = self._cached_sampler
            for _ in range(n_steps):
                k1 = sampler(y)
                k2 = sampler(y + (h / 2) * k1)
                k3 = sampler(y + (h / 2) * k2)
                k4 = sampler(y + h * k3)
                y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            t0 = 1. if inverse else 0.
            ts = torch.linspace(t0, t0 + h * n_steps, n_steps + 1, device=y.device)
            y = odeint(self._velocity, y, ts, method=self.solver)[-1]
        return y * self._flow_range_zyx + self.flow_min_corner_zyx

    def _inverse(self, input_yx):
        return self._call(input_yx, inverse=True)


class GapExpanderParams(nn.Module):

    def __init__(self, resolution, min_z, max_z, num_windings, dr_per_winding):
        super().__init__()
        self.num_by_winding = (2 * torch.pi * (torch.arange(1, num_windings) + 0.5) * dr_per_winding / resolution + 0.5).to(torch.int64)
        self.num_z = int((max_z - min_z) / resolution)
        self.logits = nn.Parameter(torch.zeros([1, 1, self.num_z, sum(self.num_by_winding)]))
        self.register_buffer('winding_first_logit_idx', torch.cat([torch.zeros([1]), torch.cumsum(self.num_by_winding, dim=0)]))


class GapExpandingTransform(pyro.distributions.transforms.Transform):

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, params, dr_per_winding, min_z, max_z, gap_expander_lr_scale, truncate_frac=None, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.params = params
        self.dr_per_winding = dr_per_winding
        self.min_z = min_z
        self.max_z = max_z
        self.gap_expander_lr_scale = gap_expander_lr_scale
        self.truncate_frac = truncate_frac

    def get_transformed_winding_radii(self, theta, z):
        # This returns the sequence of winding radii (true, not shifted) for the radials given by theta and z
        num_windings = len(self.params.num_by_winding)
        winding_first_logit_idx = self.params.winding_first_logit_idx
        theta_normalised = theta / (2 * torch.pi)
        winding_coords = torch.lerp(winding_first_logit_idx[:-1], winding_first_logit_idx[1:], theta_normalised[..., None])
        winding_coords_normalised = winding_coords / winding_first_logit_idx[-1] * 2 - 1
        z_normalised = (z - self.min_z) / (self.max_z - self.min_z) * 2 - 1
        # Pin the 0th logit (i.e. theta=0 on 1th winding) to be zero, to avoid a jump going from winding #0 to #1
        logits = torch.cat([torch.zeros_like(self.params.logits[..., :1]), self.params.logits[..., 1:]], dim=-1)
        logits = logits * self.gap_expander_lr_scale
        # Note the 0th logit/scale/distance here adjusts the gap directly outside the 0th winding (with the 0th winding being always canonical)
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

    def _call(self, input_zyx):
        theta, original_radius, inner_winding, _ = get_bounding_windings(input_zyx[..., 1:], self.dr_per_winding)
        transformed_winding_radii = self.get_transformed_winding_radii(theta, input_zyx[..., 0])
        inner_winding_clipped = inner_winding.to(torch.int64).clip(min=0, max=transformed_winding_radii.shape[-1] - 2)
        transformed_inner_radius = torch.gather(transformed_winding_radii, dim=-1, index=inner_winding_clipped[..., None]).squeeze(-1)
        transformed_outer_radius = torch.gather(transformed_winding_radii, dim=-1, index=(inner_winding_clipped + 1)[..., None]).squeeze(-1)
        original_inner_radius = (inner_winding_clipped + theta / (2 * torch.pi)) * self.dr_per_winding
        original_outer_radius = original_inner_radius + self.dr_per_winding
        frac = (original_radius - original_inner_radius) / (original_outer_radius - original_inner_radius)
        transformed_radius = torch.lerp(transformed_inner_radius, transformed_outer_radius, frac)
        delta_radius = transformed_radius - original_radius
        outward_direction = torch.cat([torch.zeros_like(input_zyx[..., :1]), F.normalize(input_zyx[..., 1:], dim=-1)], dim=-1)
        transformed_zyx = input_zyx + outward_direction * delta_radius[..., None]
        return transformed_zyx

    def _inverse(self, input_zyx):
        theta, transformed_radius, _ = get_theta_and_radii(input_zyx[..., 1:], self.dr_per_winding)
        transformed_winding_radii = self.get_transformed_winding_radii(theta, input_zyx[..., 0])
        inner_winding_indices = torch.searchsorted(transformed_winding_radii, transformed_radius[..., None]).squeeze(-1) - 1
        # If shifted_radius is exactly zero, avoid this being -1.
        inner_winding_clipped = inner_winding_indices.clip(min=0, max=transformed_winding_radii.shape[-1] - 2)

        transformed_inner_radius = torch.gather(transformed_winding_radii, dim=-1, index=inner_winding_clipped[..., None]).squeeze(-1)
        transformed_outer_radius = torch.gather(transformed_winding_radii, dim=-1, index=(inner_winding_clipped + 1)[..., None]).squeeze(-1)
        original_inner_radius = (inner_winding_clipped + theta / (2 * torch.pi)) * self.dr_per_winding
        original_outer_radius = original_inner_radius + self.dr_per_winding
        frac = (transformed_radius - transformed_inner_radius) / (transformed_outer_radius - transformed_inner_radius)
        original_radius = torch.lerp(original_inner_radius, original_outer_radius, frac)
        delta_radius = original_radius - transformed_radius
        outward_direction = torch.cat([torch.zeros_like(input_zyx[..., :1]), F.normalize(input_zyx[..., 1:], dim=-1)], dim=-1)
        transformed_zyx = input_zyx + outward_direction * delta_radius[..., None]

        return transformed_zyx


class VaryingLinearTransform(pyro.distributions.transforms.Transform):

    # This applies a z-dependent 2x2 linear transform M(z) on yx, parametrised
    # as M(z) = expm(L(z)) where L(z) is an unconstrained 2x2 matrix.
    # det(M) = exp(tr(L)) > 0, so M is always invertible and orientation-preserving.

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, logits, min_z, max_z, truncate_frac=None, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.min_z = min_z
        self.max_z = max_z
        self.logits = logits
        self.truncate_frac = truncate_frac

    def _call(self, input_zyx, inverse=False):
        zs = input_zyx[..., :1]
        normalised_zs = (zs.view(-1) - self.min_z) / (self.max_z - self.min_z) * 2 - 1
        logits = F.grid_sample(
            rearrange(self.logits, 'z r c -> 1 (r c) z 1'),
            torch.stack([torch.zeros_like(normalised_zs), normalised_zs], dim=-1)[None, None],
            padding_mode='border',
            align_corners=True
        ).squeeze(2).squeeze(0).T.view(*input_zyx.shape[:-1], 2, 2)
        if inverse:
            logits = -logits
        if self.truncate_frac is not None:
            # In log-space, scaling by truncate_frac gives a geodesic interpolation
            # towards the identity at frac=0
            logits = logits * self.truncate_frac
        M = expm_2x2(logits)
        yx_out = (M @ input_zyx[..., 1:, None]).squeeze(-1)
        return torch.cat([zs, yx_out], dim=-1)

    def _inverse(self, input_zyx):
        return self._call(input_zyx, inverse=True)


class UmbilicusTransform(pyro.distributions.transforms.Transform):

    # This translates in the yx plane by a z-dependent value (i.e. shears the volume) s.t. the origin is moved to the umbilicus

    domain = pyro.distributions.constraints.real_vector
    codomain = domain

    def __init__(self, umbilicus_zyx, event_dim=2, cache_size=0):
        super().__init__(cache_size=cache_size)
        self._event_dim = event_dim
        yx_filtered = scipy.ndimage.gaussian_filter1d(umbilicus_zyx[:, 1:].cpu().numpy(), sigma=75., axis=0, mode='nearest')
        self._yx = torch.from_numpy(yx_filtered).to(umbilicus_zyx.device).contiguous()
        self._z = umbilicus_zyx[:, :1].contiguous()

    def _call(self, input_zyx, inverse=False):
        centre_yx = interp1d(input_zyx[..., 0].contiguous(), self._z, self._yx)
        return input_zyx + torch.cat([torch.zeros_like(centre_yx[..., :1]), centre_yx], dim=-1) * (-1 if inverse else 1)

    def _inverse(self, input_zyx):
        return self._call(input_zyx, inverse=True)


class SpiralAndTransform(nn.Module):

    def __init__(self, flow_integration_steps, flow_integration_solver, flow_min_corner_zyx, flow_max_corner_zyx, umbilicus_zyx, config, spiral_outward_sense='CW'):

        super().__init__()

        self.cfg = config
        self.spiral_outward_sense = spiral_outward_sense
        self.flow_integration_steps = flow_integration_steps
        self.flow_integration_solver = flow_integration_solver
        self.flow_min_corner_zyx = flow_min_corner_zyx
        self.flow_max_corner_zyx = flow_max_corner_zyx
        self.spiral_intensity = 200 / 255
        self.dr_per_winding_scale = 12.  # larger value increases effective learning rate
        self.linear_logits_scale = 40.  # larger value increases effective learning rate

        self.umbilicus_transform = UmbilicusTransform(umbilicus_zyx)
        self.dr_per_winding_logit = nn.Parameter(torch.tensor(config['initial_dr_per_winding'] / self.dr_per_winding_scale, dtype=torch.float32))

        flow_resolution = (flow_max_corner_zyx - flow_min_corner_zyx) // config['flow_voxel_resolution']
        if config['flow_field_type'] == 'cylindrical':
            self.flow_field = CylindricalFlowField(
                flow_resolution,
                lr_scale_factor=config['flow_field_high_res_lr_scale_initial'],
                num_flow_timesteps=config['num_flow_timesteps'],
            )
        else:
            self.flow_field = CartesianFlowField(
                flow_resolution,
                lr_scale_factor=config['flow_field_high_res_lr_scale_initial'],
                num_flow_timesteps=config['num_flow_timesteps'],
            )

        self.linear_logits = nn.Parameter(torch.zeros([int(flow_max_corner_zyx[0] - flow_min_corner_zyx[0]) // config['linear_z_resolution'], 2, 2], dtype=torch.float32))

        self.gap_expander_params = GapExpanderParams(
            resolution=config['gap_expander_logit_resolution'],
            min_z=flow_min_corner_zyx[0],
            max_z=flow_max_corner_zyx[0],
            num_windings=config['gap_expander_num_windings'],
            dr_per_winding=config['initial_dr_per_winding'],  # this is a nominal (fixed) winding spacing which we only use to calculate the number of logits
        )

    @property
    def device(self):
        return self.linear_logits.device

    def get_slice_to_spiral_transform(self, truncate_at_step=None):
        truncate_frac = None if truncate_at_step is None else truncate_at_step / (self.flow_integration_steps - 1)
        diffeomorphism = IntegratedFlowDiffeomorphism(self.flow_field, self.flow_min_corner_zyx, self.flow_max_corner_zyx, num_steps=self.flow_integration_steps, solver=self.flow_integration_solver, truncate_at_step=truncate_at_step)
        gap_expander = GapExpandingTransform(
            self.gap_expander_params,
            self.get_dr_per_winding(),
            self.flow_min_corner_zyx[0],
            self.flow_max_corner_zyx[0],
            self.cfg['gap_expander_lr_scale'],
            truncate_frac,
        )
        if self.spiral_outward_sense == 'CW':
            maybe_flip = []
        else:
            assert self.spiral_outward_sense == 'ACW'
            # To make spiral go anticlockwise in slice space (going outwards from the centre), flip it horizontally
            maybe_flip = [pyro.distributions.transforms.AffineTransform(loc=0., scale=torch.tensor([1., 1., -1.], device=self.device))]
        return pyro.distributions.transforms.ComposeTransform([
            gap_expander,
            *maybe_flip,
            diffeomorphism,
            VaryingLinearTransform(self.linear_logits * self.linear_logits_scale, self.flow_min_corner_zyx[0], self.flow_max_corner_zyx[0], truncate_frac),
            self.umbilicus_transform,
        ]).inv

    def get_dr_per_winding(self):
        return F.softplus(self.dr_per_winding_logit * self.dr_per_winding_scale)

    def get_spiral_density(self, spiral_zyx, winding_range=None):
        if winding_range is None:
            winding_range = (self.cfg['output_first_winding'], float('inf'))
        return sample_spiral.get_spiral_density(
            spiral_zyx[..., 1:],
            dr_per_winding=self.get_dr_per_winding(),
            sigma=1.,
            winding_range=winding_range,
        ) * self.spiral_intensity
