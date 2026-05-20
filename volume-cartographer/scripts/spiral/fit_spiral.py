
import os
import copy
import json
import glob
import zarr
import torch
import wandb
import pickle
import kornia
import hashlib
import trimesh
import datetime
import itertools
import numpy as np
import scipy.ndimage
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
import pyro.distributions
from einops import rearrange
from torchdiffeq import odeint

from tifxyz import load_tifxyz, save_tifxyz
from geom_utils import interp1d
from point_collection import load_point_collection, _process_point_collections, normalise_pcl_winding_annotations
from umbilicus import thaumato_umbilicus_z_to_yx, json_umbilicus_z_to_yx


# PHercParis4
volpkg_path = '/home/paul/projects/vesuvius-scrolls/volpkgs/s1_ds2.volpkg'
scroll_zarr_path = None
cross_patch_pcl_json_paths = ['/home/paul/projects/vesuvius-scrolls/spiral/windings_8205.json']
unattached_pcl_json_paths = [
    '/home/paul/projects/vesuvius-scrolls/spiral/same_winding_annotations.json',
    '/home/paul/projects/vesuvius-scrolls/spiral/same_winding_annotations_2.json',
    '/home/paul/projects/vesuvius-scrolls/spiral/s1_relative_windings.json',
]
spiral_outward_sense = 'CW'  # CW | ACW
umbilicus_z_to_yx = lambda f: json_umbilicus_z_to_yx(f'{volpkg_path}/umbilicus.json', downsample_factor=f)
scroll_name = 's1'
z_begin, z_end = 6000, 16000
patches_path = '/home/paul/projects/vesuvius-scrolls/spiral/custom_patches'
voxel_size_um = 2.4 * 4  # before downsampling
seed_patch_id = 'auto_grown_20260429215626691_sel_20260512_102916_79'

# # PHerc0172
# volpkg_path = '/home/paul/projects/vesuvius-scrolls/volpkgs/PHerc0172.volpkg'
# scroll_zarr_path = f'{volpkg_path}/volumes/s5_masked_ome.zarr/2'
# cross_patch_pcl_json_paths = [f'{volpkg_path}/atlas-pcl.json']
# unattached_pcl_json_paths = []
# spiral_outward_sense = 'CW'  # CW | ACW
# umbilicus_z_to_yx = lambda f: thaumato_umbilicus_z_to_yx('/home/paul/projects/vesuvius-scrolls/data/s5-umbilicus.txt', downsample_factor=f)
# scroll_name = 's5'
# z_begin, z_end = 15400, 16600
# voxel_size_um = 8.0
# seed_patch_id = None

cache_path = '../cache'
downsample_factor = 4
z_begin //= downsample_factor
z_end //= downsample_factor

default_config = {
    'random_seed': 1,
    'learning_rate': 2.e-4,
    'exp_lr_schedule': True,
    'lr_final_factor': 0.05,
    'num_training_steps': 10_000,
    'num_flow_integration_steps': 3,
    'flow_integration_solver': 'rk4',
    'num_flow_timesteps': 1,
    'flow_bounds_z_margin': 40,
    'flow_bounds_radius': 800,
    'flow_voxel_resolution': 5,
    'flow_field_high_res_lr_scale': 1.5e-1,
    'gap_expander_logit_resolution': 6,
    'gap_expander_num_windings': 85,
    'gap_expander_lr_scale': 0.1,
    'linear_z_resolution': 12,
    'initial_dr_per_winding': 4.,
    'radius_loss_margin': 0.025,
    'patch_loss_z_margin': 0,
    'patch_dt_norm_p': 0.5,
    'patch_dt_within_patch_norm_p': 3.0,
    'num_patches_per_step': 120,
    'num_patches_per_step_for_dt': 80,
    'num_points_per_patch': 800,
    'winding_number_num_pairs': 2000,
    'winding_number_num_pcls': 1,
    'winding_number_adjacent_patches_only': True,
    'unattached_pcl_num_per_step': 16,
    'unattached_pcl_num_points_per_step': 32,
    'normals_num_points': 2000,
    'regularisation_num_points': 1500,
    'loss_weight_patch_radius': 8.e0,
    'loss_weight_uv_distance': 0.,
    'loss_weight_patch_dt': 16.e0,
    'loss_weight_winding_number': 10.,
    'loss_weight_unattached_pcl_radius': 8.e0,
    'loss_weight_unattached_pcl_dt': 16.e0,
    'loss_weight_patch_stretch': 40.0,
    'loss_weight_patch_normals': 75.0,
    'loss_weight_umbilicus': 5.,
    'loss_start_patch_dt': 5000,
    'working_set_mode': 'global',  # 'progressive' (grow from a seed when satisfied) | 'progressive_fixed' (grow from a seed on a fixed iteration schedule) | 'global' (fit all patches at once)
    'working_set_check_interval': 10,
    'progressive_fixed_add_interval': 50,
    'output_winding_range': (10, 100),
    'output_winding_margin': 4,
    'output_step_size': 20,
}


# Thresholds defining the patch-satisfaction metrics
metrics_config = {
    'satisfaction_radius_tolerance': 0.5,  # spiral-space, in units of dr_per_winding
    'satisfaction_distance_tolerance': 4.0,  # absolute scan-space distance, in voxels
    'satisfied_patch_quad_fraction': 0.95,  # min fraction of valid quads satisfied for a patch to count as satisfied
    'boundary_satisfied_patch_quad_fraction': 0.90,  # min fraction of boundary quads satisfied for the boundary metric
}


def get_env_config_overrides():
    overrides_json = os.environ.get('FIT_SPIRAL_CONFIG_OVERRIDES')
    if not overrides_json:
        return {}
    overrides = json.loads(overrides_json)
    unknown_keys = sorted(set(overrides) - set(default_config))
    if unknown_keys:
        raise KeyError(f'unknown FIT_SPIRAL_CONFIG_OVERRIDES keys: {unknown_keys}')
    return overrides


def get_spiral_yxs(num_windings, dr_per_winding, inter_point_spacing, group_by_winding=False):

    # Note this is not differentiable wrt dr_per_winding nor inter_point_spacing!

    # r = b * theta => b = drpw / 2pi
    # ...so r = dr_per_winding * theta / (2 * pi)

    # Kth winding has average radius (K + 0.5) * dr_per_winding => circumference (K + 0.5) * dr_per_winding * 2 * pi
    # ...so should have (K + 0.5) * dr_per_winding * 2 * pi / inter_point_spacing steps
    # can construct these thetas directly, then r's via formula

    thetas = [
        winding_idx * 2 * torch.pi + torch.arange(
            0, 2 * np.pi,
            step=inter_point_spacing / (winding_idx + 0.5) / float(dr_per_winding),
            device='cuda'
        )
        for winding_idx in range(num_windings)
    ]
    radii = [dr_per_winding * thetas_for_winding / (2 * torch.pi) for thetas_for_winding in thetas]

    yxs = [
        torch.stack([torch.sin(thetas_for_winding), torch.cos(thetas_for_winding)], dim=-1) * radii_for_winding[:, None]
        for thetas_for_winding, radii_for_winding in zip(thetas, radii)
    ]

    if group_by_winding:
        return yxs
    else:
        return torch.cat(yxs, dim=0)


def get_spiral_points(predictions_slice, centre_xy, dr_per_winding=10):

    inter_point_spacing = 4  # pixels; this doesn't affect the shape of the spiral, just where we sample it

    # This only affects how far 'out' we go, it doesn't affect the shape. We set it such that the spiral just
    # touches the most distant-from-umbilicus edge of the slice
    num_windings = int(1 + np.maximum(centre_xy, predictions_slice.shape[::-1] - centre_xy).max() / dr_per_winding)

    yxs = centre_xy[::-1] + get_spiral_yxs(num_windings, dr_per_winding, inter_point_spacing).cpu().numpy()

    yxs = (yxs + 0.5).astype(np.int64)
    yxs = yxs[(0 <= yxs[:, 0]) & (yxs[:, 0] < predictions_slice.shape[0])]
    yxs = yxs[(0 <= yxs[:, 1]) & (yxs[:, 1] < predictions_slice.shape[1])]

    return yxs


def get_winding_xy(winding_idx, theta, dr_per_winding):
    winding_radius = winding_idx * dr_per_winding + theta / (2 * np.pi) * dr_per_winding
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1) * winding_radius[..., None]


def get_theta(relative_yx):
    relative_yx = torch.stack([
        relative_yx[..., 0],
        torch.where(relative_yx[..., 1].abs() < 1.e-10, 1.e-10, relative_yx[..., 1]),
    ], dim=-1)  # avoid NaN gradients from atan2 / sqrt
    theta = torch.arctan2(relative_yx[..., 0], relative_yx[..., 1]) % (2 * np.pi)  # [0, 2pi]; zero along x-axis
    return theta, relative_yx


def get_theta_and_radii(relative_yx, dr_per_winding):
    theta, relative_yx = get_theta(relative_yx)
    radius = torch.linalg.norm(relative_yx, dim=-1)
    # The spiral has radius 0 at winding angle 0 then increases linearly at rate dr_per_winding
    # Note get_fibre_loss assumes this form!
    shifted_radius = radius - theta / (2 * np.pi) * dr_per_winding
    shifted_radius = shifted_radius.clamp(min=0.)
    return theta, radius, shifted_radius


def get_bounding_windings(relative_yx, dr_per_winding):
    # The spiral has radius 0 at winding angle 0 then increases linearly at rate dr_per_winding
    # Want to find the two windings that bracket yx
    # If theta=+eps, then these are given by floor/ceil of radius / dr_per_winding
    # For other theta, we shift the point radially so 'as if' it were at theta=0
    theta, radius, shifted_radius = get_theta_and_radii(relative_yx, dr_per_winding)
    inner_winding = torch.floor(shifted_radius / dr_per_winding)
    outer_winding = torch.ceil(shifted_radius / dr_per_winding)
    return theta, radius, inner_winding, outer_winding


def get_spiral_density(relative_yx, dr_per_winding=10., sigma=3.):

    min_w, max_w = cfg['output_winding_range']
    theta, radius, inner_winding, outer_winding = get_bounding_windings(relative_yx, dr_per_winding)
    def evaluate_kernel(winding_idx):
        winding_xy = get_winding_xy(winding_idx, theta, dr_per_winding)
        distance = torch.linalg.norm(winding_xy.flip(-1) - relative_yx, dim=-1)
        kernel = torch.exp(-distance ** 2 / sigma ** 2)
        kernel = torch.where((winding_idx >= min_w) & (winding_idx < max_w), kernel, torch.zeros_like(kernel))
        return kernel
    result = evaluate_kernel(inner_winding) + evaluate_kernel(outer_winding)
    return result.clip(0., 1.)


class ExplicitFlowField(nn.Module):

    def __init__(self, resolution, spatial_scale_factor=6, lr_scale_factor=1.e-1):
        super().__init__()
        self.flow_scales = [1., lr_scale_factor]
        self.flows = nn.ParameterList([
            nn.Parameter(torch.zeros([cfg['num_flow_timesteps'], 3, *shape]))
            for shape in [
                [resolution[0] // spatial_scale_factor, resolution[1] // spatial_scale_factor, resolution[2] // spatial_scale_factor],
                resolution,
            ]
        ])

    def __call__(self, t):
        lr_flow, hr_flow = self.flows[0], self.flows[1]
        hr_shape = tuple(hr_flow.shape[2:])
        if cfg['num_flow_timesteps'] == 1:
            # Time-invariant: HR flow is already at the target resolution, so skip interpolating it.
            lr_upsampled = F.interpolate(lr_flow, size=hr_shape, mode='trilinear')[0] * self.flow_scales[0]
            return lr_upsampled + hr_flow[0] * self.flow_scales[1]
        t_scaled = (t.clamp(-1. + 1.e-4, 1. - 1.e-4) + 1) / 2 * (cfg['num_flow_timesteps'] - 1)
        t_idx_before = int(t_scaled)
        flows_interpolated = [
            F.interpolate(flow[t_idx_before : t_idx_before + 2], size=hr_shape, mode='trilinear') * flow_scale
            for flow, flow_scale in zip(self.flows, self.flow_scales)
        ]
        flows_interpolated = [
            torch.lerp(flow_interpolated[0], flow_interpolated[1], t_scaled % 1.)
            for flow_interpolated in flows_interpolated
        ]
        return sum(flows_interpolated)


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

    def _velocity(self, t_int, current_zyx_scaled):
        # t_int is a scalar in [0, 1]; flow_field expects t in [-1, 1]
        t_flow = t_int * 2 - 1
        return sample_field(current_zyx_scaled, self.flow_field(t_flow))

    def _call(self, input_zyx, inverse=False):

        # ODE integration of the temporally-varying flow to give a diffeomorphism.
        # The flow & diffeomorphism represent shifts in normalised units [0,1] over the flow region.
        y = (input_zyx - self.flow_min_corner_zyx) / self._flow_range_zyx
        n_steps = self.num_steps if self.truncate_at_step is None else self.truncate_at_step
        t_span = n_steps / self.num_steps
        h = (-t_span if inverse else t_span) / n_steps
        if cfg['num_flow_timesteps'] == 1:
            # Time-invariant flow: precompute the upsampled flow once and inline a manual rk4 loop
            # to skip torchdiffeq's per-step dispatch overhead.
            assert self.solver == 'rk4'
            cached_flow = self.flow_field(0.0)
            for _ in range(n_steps):
                k1 = sample_field(y, cached_flow)
                k2 = sample_field(y + (h / 2) * k1, cached_flow)
                k3 = sample_field(y + (h / 2) * k2, cached_flow)
                k4 = sample_field(y + h * k3, cached_flow)
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

    def __init__(self, params, dr_per_winding, min_z, max_z, truncate_frac=None, event_dim=0, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.params = params
        self.dr_per_winding = dr_per_winding
        self.min_z = min_z
        self.max_z = max_z
        self.truncate_frac = truncate_frac

    def get_transformed_winding_radii(self, theta, z):
        # This returns the sequence of winding radii (true, not shifted) for the radials given by theta and z
        num_windings = len(self.params.num_by_winding)
        winding_first_logit_idx = self.params.winding_first_logit_idx
        theta_normalised = theta / (2 * torch.pi)
        winding_coords = torch.lerp(winding_first_logit_idx[:-1], winding_first_logit_idx[1:], theta_normalised[..., None])  # *, winding-idx
        winding_coords_normalised = winding_coords / winding_first_logit_idx[-1] * 2 - 1
        z_normalised = (z - self.min_z) / (self.max_z - self.min_z) * 2 - 1
        # Pin the 0th logit (i.e. theta=0 on 1th winding) to be zero, to avoid a jump going from winding #0 to #1
        logits = torch.cat([torch.zeros_like(self.params.logits[..., :1]), self.params.logits[..., 1:]], dim=-1)  # 1, 1, z, winding-angle
        logits = logits * cfg['gap_expander_lr_scale']
        # Note the 0th logit/scale/distance here adjusts the gap directly outside the 0th winding (with the 0th winding being always canonical)
        logits_by_winding = F.grid_sample(
            logits,
            torch.stack([winding_coords_normalised, z_normalised[..., None].expand(*theta.shape, num_windings)], dim=-1).view(1, -1, num_windings, 2),
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        ).squeeze(1).squeeze(0).view(*theta.shape, num_windings)  # *, winding-idx
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
        inner_winding_clipped = inner_winding.to(torch.int64).clip(max=transformed_winding_radii.shape[-1] - 2)
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
        inner_winding_clipped = inner_winding_indices.clip(min=0, max=transformed_winding_radii.shape[-1] - 2)  # if shifted_radius is exactly zero, avoid this being -1

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
        self.logits = logits  # z, 2, 2
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
        M = torch.linalg.matrix_exp(logits)
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

    def __init__(self, flow_integration_steps, flow_integration_solver, flow_min_corner_zyx, flow_max_corner_zyx, umbilicus_zyx):

        super().__init__()

        self.flow_integration_steps = flow_integration_steps
        self.flow_integration_solver = flow_integration_solver
        self.flow_min_corner_zyx = flow_min_corner_zyx
        self.flow_max_corner_zyx = flow_max_corner_zyx
        self.spiral_intensity = 200 / 255
        self.dr_per_winding_scale = 12.  # larger value increases effective learning rate
        self.linear_logits_scale = 40.  # larger value increases effective learning rate

        self.umbilicus_transform = UmbilicusTransform(umbilicus_zyx)
        self.dr_per_winding_logit = nn.Parameter(torch.tensor(cfg['initial_dr_per_winding'] / self.dr_per_winding_scale, dtype=torch.float32))

        flow_resolution = (flow_max_corner_zyx - flow_min_corner_zyx) // cfg['flow_voxel_resolution']
        self.flow_field = ExplicitFlowField(flow_resolution, lr_scale_factor=cfg['flow_field_high_res_lr_scale'])

        self.linear_logits = nn.Parameter(torch.zeros([int(flow_max_corner_zyx[0] - flow_min_corner_zyx[0]) // cfg['linear_z_resolution'], 2, 2], dtype=torch.float32))

        self.gap_expander_params = GapExpanderParams(
            resolution=cfg['gap_expander_logit_resolution'],
            min_z=flow_min_corner_zyx[0],
            max_z=flow_max_corner_zyx[0],
            num_windings=cfg['gap_expander_num_windings'],
            dr_per_winding=cfg['initial_dr_per_winding'],  # this is a nominal (fixed) winding spacing which we only use to calculate the number of logits
        )

    @property
    def device(self):
        return self.linear_logits.device

    def get_slice_to_spiral_transform(self, truncate_at_step=None):
        truncate_frac = None if truncate_at_step is None else truncate_at_step / (self.flow_integration_steps - 1)
        diffeomorphism = IntegratedFlowDiffeomorphism(self.flow_field, self.flow_min_corner_zyx, self.flow_max_corner_zyx, num_steps=self.flow_integration_steps, solver=self.flow_integration_solver, truncate_at_step=truncate_at_step)
        gap_expander = GapExpandingTransform(self.gap_expander_params, self.get_dr_per_winding(), self.flow_min_corner_zyx[0], self.flow_max_corner_zyx[0], truncate_frac)
        if spiral_outward_sense == 'CW':
            maybe_flip = []
        else:
            assert spiral_outward_sense == 'ACW'
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

    def get_spiral_density(self, spiral_zyx):
        return get_spiral_density(spiral_zyx[..., 1:], dr_per_winding=self.get_dr_per_winding(), sigma=1.) * self.spiral_intensity


def run_containing_index(mask_1d: np.ndarray, idx: int) -> tuple[int, int] | None:
    """Return (start, end) of the contiguous True run containing idx."""
    padded = np.concatenate([[False], mask_1d, [False]])
    diff = np.diff(padded.astype(int))  # diff will be +1 at start of runs, -1 at end of runs
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0] - 1
    run_idx = np.searchsorted(run_starts, idx, side='right') - 1
    return run_starts[run_idx], run_ends[run_idx] + 1


def _min_pair_distance(a, b):
    # Minimum pairwise distance between two point clouds (each [N, 3]).
    return torch.cdist(a, b).min()


def _build_progressive_patch_order(patches, seed_patch_idx, num_sample_points=500):
    # Greedy nearest-to-active ordering: starting from seed_patch_idx, repeatedly add the
    # not-yet-active patch with the smallest min-pairwise distance to its closest active patch.
    device = torch.device('cuda')
    samples = []
    for patch in patches:
        valid_zyxs = patch.valid_zyxs.to(device=device, dtype=torch.float32)
        if valid_zyxs.shape[0] > num_sample_points:
            idx = torch.randperm(valid_zyxs.shape[0], device=device)[:num_sample_points]
            valid_zyxs = valid_zyxs[idx]
        samples.append(valid_zyxs)

    n = len(patches)
    active = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf, dtype=np.float64)
    active[seed_patch_idx] = True
    min_dist[seed_patch_idx] = -np.inf

    order = [seed_patch_idx]
    last_added_pts = samples[seed_patch_idx]
    for _ in tqdm(range(n - 1), desc='ordering patches'):
        for i in range(n):
            if active[i]:
                continue
            d = float(_min_pair_distance(samples[i], last_added_pts).item())
            if d < min_dist[i]:
                min_dist[i] = d
        next_idx = int(np.argmin(np.where(active, np.inf, min_dist)))
        order.append(next_idx)
        active[next_idx] = True
        min_dist[next_idx] = -np.inf
        last_added_pts = samples[next_idx]
    return order


def _get_working_set_probabilities(full_probabilities, working_set_indices):
    masked = np.zeros_like(full_probabilities)
    masked[working_set_indices] = full_probabilities[working_set_indices]
    total = masked.sum()
    if total <= 0:
        raise ValueError('Working set has zero total sampling probability')
    return masked / total


def _prepare_patch_sampling_cache(patches, patch_loss_z_margin):
    patch_areas = np.empty(len(patches), dtype=np.float64)
    for patch_idx, patch in enumerate(patches):
        # Use the quad-valid mask so bilinear interpolation at (row_idx+di, j+dj)
        # is well-defined for di, dj in [0, 1).
        valid_quad_mask_np = patch.valid_quad_mask.cpu().numpy()
        # Restrict sampling to quads whose representative z is in [z_begin, z_end),
        # so patch-loss tracks don't waste samples outside the optimisation ROI.
        zyxs_z_np = patch.zyxs[..., 0].cpu().numpy()
        quad_zs_np = (zyxs_z_np[:-1, :-1] + zyxs_z_np[1:, :-1] + zyxs_z_np[:-1, 1:] + zyxs_z_np[1:, 1:]) / 4
        z_in_roi_np = (
            (quad_zs_np >= z_begin - patch_loss_z_margin)
            & (quad_zs_np < z_end + patch_loss_z_margin)
        )
        in_roi_quad_mask_np = valid_quad_mask_np & z_in_roi_np
        if not in_roi_quad_mask_np.any():
            # Fallback if no quad falls in the ROI; should be rare since patches
            # entirely outside the z-ROI are dropped earlier.
            in_roi_quad_mask_np = valid_quad_mask_np
        patch._sampling_valid_quad_mask_np = in_roi_quad_mask_np
        patch._sampling_valid_quad_rows = np.flatnonzero(in_roi_quad_mask_np.any(axis=1))
        patch._sampling_valid_quad_cols = np.flatnonzero(in_roi_quad_mask_np.any(axis=0))
        patch_areas[patch_idx] = float(patch.area)
    inv_weights = patch_areas ** 0.5
    return inv_weights / inv_weights.sum()


def _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding):
    if theta.shape[-1] <= 1:
        return shifted_radii

    # Use detached theta only to detect branch-cut crossings. This keeps the loss
    # differentiable through the sampled points while making the unwrap robust when
    # a sampled point lands exactly on the theta=0 seam.
    theta_diffs = theta.detach()[..., 1:] - theta.detach()[..., :-1]
    step_adjustments = (
        (theta_diffs > np.pi).to(shifted_radii.dtype)
        - (theta_diffs < -np.pi).to(shifted_radii.dtype)
    ) * dr_per_winding.detach()
    adjustments = torch.cat([
        torch.zeros([*theta.shape[:-1], 1], device=shifted_radii.device, dtype=shifted_radii.dtype),
        torch.cumsum(step_adjustments, dim=-1),
    ], dim=-1)
    return shifted_radii + adjustments


def _sample_strip_ijs(line_valid, seed, fixed_coord, axis, num_points):
    # Sample num_points fractional ijs along the contiguous True run of `line_valid`
    # containing `seed`, fixed at `fixed_coord` along `axis` (axis=0 -> fixed i, varying j;
    # axis=1 -> fixed j, varying i), with sub-pixel jitter. Caller guarantees line_valid[seed].
    # The contiguous range lets _unwrap_track_shifted_radii reliably handle theta=0 crossings.
    lo, hi = run_containing_index(line_valid, seed)
    run_len = hi - lo
    coords = np.sort(np.random.choice(run_len, num_points, replace=num_points > run_len))
    ijs = np.empty([num_points, 2], dtype=np.float32)
    var_axis = 1 - axis
    ijs[:, axis] = fixed_coord + float(np.random.uniform(0., 1.))
    ijs[:, var_axis] = lo + coords + np.random.uniform(0., 1., size=num_points)
    return ijs


def _sample_patch_tracks(slice_to_spiral_transform, dr_per_winding, patches, patch_indices, umbilicus_zyx=None):
    if len(patch_indices) == 0:
        raise ValueError('Expected at least one patch index')

    # For each patch, we take one row and one column. _sample_strip_ijs picks a contiguous
    # subrange of each so _unwrap_track_shifted_radii can reliably handle theta=0 crossings
    # between consecutive sorted samples.

    # TODO: instead of 'strict' horizontal & vertical strips, could/should take wiggly strips that take a mostly-horizontal
    #  or mostly-vertical patch between distant points, skirting around gaps/holes; important for long, ragged traces

    num_points_per_direction = cfg['num_points_per_patch'] // 2
    N = len(patch_indices)

    horizontal_ijs_by_patch = np.empty([N, num_points_per_direction, 2], dtype=np.float32)
    vertical_ijs_by_patch = np.empty([N, num_points_per_direction, 2], dtype=np.float32)
    for n, patch_idx in enumerate(patch_indices):
        patch = patches[patch_idx]
        valid_quad_mask_np = patch._sampling_valid_quad_mask_np

        # Subsample from a contiguous region of one row
        row_idx = np.random.choice(patch._sampling_valid_quad_rows)
        row_valid = valid_quad_mask_np[row_idx, :]
        seed_j = np.random.choice(np.flatnonzero(row_valid))
        horizontal_ijs_by_patch[n] = _sample_strip_ijs(row_valid, seed_j, row_idx, axis=0, num_points=num_points_per_direction)

        # Subsample from a contiguous region of one column
        col_idx = np.random.choice(patch._sampling_valid_quad_cols)
        col_valid = valid_quad_mask_np[:, col_idx]
        seed_i = np.random.choice(np.flatnonzero(col_valid))
        vertical_ijs_by_patch[n] = _sample_strip_ijs(col_valid, seed_i, col_idx, axis=1, num_points=num_points_per_direction)

    # Bilinear-interpolate all sampled (i,j) in one batched call when all patch_indices reference
    # the same patch (the common single-patch case); otherwise fall back to per-patch calls.
    unique_patch_indices = np.unique(patch_indices)
    if len(unique_patch_indices) == 1:
        patch = patches[unique_patch_indices[0]]
        combined_ijs = np.stack([horizontal_ijs_by_patch, vertical_ijs_by_patch], axis=0)
        combined_zyxs, _ = patch.ij_to_zyx(torch.from_numpy(combined_ijs))
        all_slice_zyxs = combined_zyxs.cuda()
    else:
        # TODO: merge with above -- group by unique patches, then feed as batches
        h_list = [patches[idx].ij_to_zyx(torch.from_numpy(horizontal_ijs_by_patch[n]))[0]
                  for n, idx in enumerate(patch_indices)]
        v_list = [patches[idx].ij_to_zyx(torch.from_numpy(vertical_ijs_by_patch[n]))[0]
                  for n, idx in enumerate(patch_indices)]
        all_slice_zyxs = torch.stack([torch.stack(h_list, dim=0), torch.stack(v_list, dim=0)], dim=0).cuda()

    # When the caller also needs umbilicus_spiral (radius loss path), pack it into the same
    # forward ODE call to amortise the per-call overhead
    patches_flat = all_slice_zyxs.reshape(-1, 3)
    if umbilicus_zyx is not None:
        n_patch_pts = patches_flat.shape[0]
        combined = torch.cat([patches_flat, umbilicus_zyx], dim=0)
        combined_spiral = slice_to_spiral_transform(combined)
        all_spiral_zyxs = combined_spiral[:n_patch_pts].reshape(*all_slice_zyxs.shape)
        umbilicus_spiral = combined_spiral[n_patch_pts:]
    else:
        all_spiral_zyxs = slice_to_spiral_transform(patches_flat).reshape(*all_slice_zyxs.shape)
        umbilicus_spiral = None

    all_theta, _, all_shifted_radii = get_theta_and_radii(all_spiral_zyxs[..., 1:], dr_per_winding)
    all_shifted_radii = _unwrap_track_shifted_radii(all_theta, all_shifted_radii, dr_per_winding)

    return all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii, umbilicus_spiral


def _get_patch_valid_points(patch, device, max_points=None, fixed_num_points=None):
    valid_mask = patch.valid_vertex_mask
    z_in_roi = (patch.zyxs[..., 0] >= z_begin) & (patch.zyxs[..., 0] < z_end)
    valid_indices = torch.where(valid_mask & z_in_roi)
    if len(valid_indices[0]) == 0:
        valid_indices = torch.where(valid_mask)
    n = len(valid_indices[0])
    if fixed_num_points is not None:
        sel = np.random.choice(n, fixed_num_points, replace=(n < fixed_num_points))
        valid_indices = (valid_indices[0][sel], valid_indices[1][sel])
    elif max_points is not None and n > max_points:
        sel = np.random.choice(n, max_points, replace=False)
        valid_indices = (valid_indices[0][sel], valid_indices[1][sel])
    return patch.zyxs[valid_indices[0], valid_indices[1], :].to(device=device, dtype=torch.float32)


def get_patch_and_umbilicus_losses(slice_to_spiral_transform, dr_per_winding, num_patches_for_radius, num_patches_for_dt, patches, patch_sampling_probabilities, umbilicus_zyx, compute_dt=True):

    # Sample once and share the tracks between the radius and DT losses; the loss using
    # fewer patches takes a prefix of the larger sample.
    num_patches_to_sample = max(num_patches_for_radius, num_patches_for_dt) if compute_dt else num_patches_for_radius
    patch_indices = np.random.choice(len(patches), num_patches_to_sample, p=patch_sampling_probabilities, replace=True)

    all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii, umbilicus_spiral = _sample_patch_tracks(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        patch_indices,
        umbilicus_zyx,
    )

    hinge_margin = dr_per_winding.detach() * cfg['radius_loss_margin']

    # Each patch row/col should lie at constant shifted-radius
    radius_shifted_radii = all_shifted_radii[:, :num_patches_for_radius]
    mean_radii = radius_shifted_radii.mean(dim=-1, keepdim=True)
    radius_deviations = (radius_shifted_radii - mean_radii).abs()
    radius_deviations_hinge = F.relu(radius_deviations - hinge_margin)
    mean_radius_deviation = radius_deviations_hinge.mean()

    # Umbilicus should map to the spiral origin (yx ≈ 0)
    umbilicus_loss = umbilicus_spiral[..., 1:].abs().mean()

    if compute_dt:
        dt_slice_zyxs = all_slice_zyxs[:, :num_patches_for_dt]
        dt_spiral_zyxs = all_spiral_zyxs[:, :num_patches_for_dt]
        dt_theta = all_theta[:, :num_patches_for_dt]
        dt_shifted_radii = all_shifted_radii[:, :num_patches_for_dt]

        dt_norm_p = cfg['patch_dt_norm_p']  # this is the 'across different tracks' norm; we prefer many tracks with zero norm when p -> 0
        within_patch_norm_p = cfg['patch_dt_within_patch_norm_p']  # this is the 'within a track' norm; we strongly penalise isolated badly-misaligned points when p -> inf

        # Define the DT target from the same sampled row/column tracks as the radius loss:
        # each track is snapped to the nearest integer-winding shifted-radius, then every
        # sampled point on the track is pulled towards the corresponding point on that
        # target winding.
        target_shifted_radii = torch.round(dt_shifted_radii.median(dim=-1, keepdim=True).values / dr_per_winding) * dr_per_winding
        target_radii = target_shifted_radii + dt_theta / (2 * np.pi) * dr_per_winding
        target_spiral_zyxs = torch.stack([
            dt_spiral_zyxs[..., 0],
            torch.sin(dt_theta) * target_radii,
            torch.cos(dt_theta) * target_radii,
        ], dim=-1).detach()

        target_scroll_zyxs = slice_to_spiral_transform.inv(target_spiral_zyxs.reshape(-1, 3)).reshape(*target_spiral_zyxs.shape)

        point_distances = torch.linalg.norm(dt_slice_zyxs - target_scroll_zyxs, dim=-1)
        point_distances = F.relu(point_distances - hinge_margin) + 1.e-5  # epsilon to avoid NaN in p-norm backward
        track_losses = (point_distances ** within_patch_norm_p).mean(dim=-1) ** (1 / within_patch_norm_p)
        patch_dt_loss = ((track_losses ** dt_norm_p).sum() / track_losses.numel()) ** (1 / dt_norm_p)
    else:
        patch_dt_loss = torch.zeros([], device=dr_per_winding.device)

    return mean_radius_deviation, umbilicus_loss, patch_dt_loss


def _sample_single_l_shape(valid_quad, i_q, j_q, leg1_axis, leg1_dir, leg2_dir, num_points):
    # Sample a single L-shape on `valid_quad` starting at (i_q, j_q). Leg 1 walks along
    # `leg1_axis` (0 -> varying j, 1 -> varying i) in direction `leg1_dir` (+1 or -1) to a
    # uniformly random turn point inside the contiguous valid run. Leg 2 walks from the
    # turn point along the perpendicular axis in direction `leg2_dir` (+1 or -1) to the end
    # of its valid run. Returns a float32 [num_points, 2] sampled in traversal order, with
    # subpixel jitter; the fixed-axis jitter is shared within each leg (matching the
    # _sample_strip_ijs convention), so the unwrap can stitch theta=0 crossings along the
    # full L (the only ~sqrt(2)-quad jump is across the corner, still well within the
    # |dtheta| < pi requirement). Caller guarantees valid_quad[i_q, j_q].

    if leg1_axis == 0:
        line1_valid = valid_quad[i_q, :]
        var_start1 = j_q
    else:
        line1_valid = valid_quad[:, j_q]
        var_start1 = i_q
    lo1, hi1 = run_containing_index(line1_valid, var_start1)
    var_far1 = (hi1 - 1) if leg1_dir > 0 else lo1
    leg1_max_steps = abs(var_far1 - var_start1)
    turn_step = int(np.random.randint(0, leg1_max_steps + 1))
    var_turn = var_start1 + leg1_dir * turn_step

    if leg1_axis == 0:
        i_turn, j_turn = i_q, var_turn
    else:
        i_turn, j_turn = var_turn, j_q

    leg2_axis = 1 - leg1_axis
    if leg2_axis == 0:
        line2_valid = valid_quad[i_turn, :]
        var_start2 = j_turn
    else:
        line2_valid = valid_quad[:, j_turn]
        var_start2 = i_turn
    lo2, hi2 = run_containing_index(line2_valid, var_start2)
    var_far2 = (hi2 - 1) if leg2_dir > 0 else lo2
    leg2_max_steps = abs(var_far2 - var_start2)

    total_steps = turn_step + leg2_max_steps  # leg 1 spans [0, turn_step]; leg 2 spans (turn_step, total_steps]
    num_positions = total_steps + 1
    steps = np.sort(np.random.choice(num_positions, num_points, replace=num_points > num_positions))

    ijs = np.empty([num_points, 2], dtype=np.float32)
    leg1_fixed_jitter = float(np.random.uniform(0, 1))
    leg2_fixed_jitter = float(np.random.uniform(0, 1))

    on_leg1 = steps <= turn_step
    leg1_steps = steps[on_leg1]
    leg2_steps = steps[~on_leg1] - turn_step

    leg1_var = (var_start1 + leg1_dir * leg1_steps).astype(np.float32) + np.random.uniform(0., 1., size=leg1_steps.shape).astype(np.float32)
    leg1_fixed = float(i_q if leg1_axis == 0 else j_q) + leg1_fixed_jitter
    if leg1_axis == 0:
        ijs[on_leg1, 0] = leg1_fixed
        ijs[on_leg1, 1] = leg1_var
    else:
        ijs[on_leg1, 0] = leg1_var
        ijs[on_leg1, 1] = leg1_fixed

    leg2_var = (var_start2 + leg2_dir * leg2_steps).astype(np.float32) + np.random.uniform(0., 1., size=leg2_steps.shape).astype(np.float32)
    leg2_fixed = float(i_turn if leg2_axis == 0 else j_turn) + leg2_fixed_jitter
    if leg2_axis == 0:
        ijs[~on_leg1, 0] = leg2_fixed
        ijs[~on_leg1, 1] = leg2_var
    else:
        ijs[~on_leg1, 0] = leg2_var
        ijs[~on_leg1, 1] = leg2_fixed

    return ijs


def _sample_l_shapes_at_ij(patch, i, j, num_points):
    # Sample 4 L-shapes anchored on the annotated point (i, j) of `patch`, one per cardinal
    # primary direction: right (+j), left (-j), down (+i), up (-i). For each, leg 2's
    # perpendicular direction is chosen uniformly at random. Returns a list of 4 float32
    # [num_points, 2] arrays sampled in traversal order, or None if (i, j) doesn't lie on
    # a valid quad. Each L is a single contiguous walk in patch space, so the unwrap can
    # handle theta=0 seam crossings along the bent strip just as it does along a straight
    # row/column.
    valid_quad = patch._sampling_valid_quad_mask_np
    H_q, W_q = valid_quad.shape
    i_q = min(max(int(i), 0), H_q - 1)
    j_q = min(max(int(j), 0), W_q - 1)
    if not valid_quad[i_q, j_q]:
        return None

    primary_specs = [(0, +1), (0, -1), (1, +1), (1, -1)]  # (leg1_axis, leg1_dir)
    return [
        _sample_single_l_shape(
            valid_quad, i_q, j_q, leg1_axis, leg1_dir,
            leg2_dir=int(np.random.choice([-1, +1])),
            num_points=num_points,
        )
        for leg1_axis, leg1_dir in primary_specs
    ]


def get_patch_winding_number_loss(slice_to_spiral_transform, dr_per_winding, patches_dict, point_collections):
    # For pairs of annotated PCL points on different patches, constrain the spiral
    # shifted-radius gap to match the annotated winding-number difference. Each
    # cross-patch pcl exposes its attached points grouped by patch
    # (pcl['points_by_patch']); we form the set of all pairs (p1, p2) whose patches
    # differ and sample uniformly from it. For each annotated point we build 4
    # L-shaped strips: from (i, j), walk along one of the cardinal patch directions
    # (right, left, down, up) to a uniformly-random turn point inside the contiguous
    # valid run, then 90-degree-turn into a uniformly-random perpendicular direction
    # and walk to the end of that valid run. Each L is sampled in traversal order
    # along its bent path, so _unwrap_track_shifted_radii can stitch theta=0 seam
    # crossings along the whole strip (the corner only introduces a ~sqrt(2)-quad ij
    # jump). We then pool all 4 L-strips per annotated point into one set of sample
    # points and take a single all-pairs diff between p1's and p2's pooled sets,
    # regressing it onto winding_diff * dr_per_winding.

    num_points_per_strip = cfg['num_points_per_patch'] // 2
    num_strips_per_pcl = 4
    num_strips_per_pair = 2 * num_strips_per_pcl  # 8

    # Each entry: (ls1, ls2, pid1, pid2, winding_diff) where ls* is a list of 4 L-shape ij strips
    strip_pairs = []

    num_pcls_per_step = min(cfg['winding_number_num_pcls'], len(point_collections))
    if num_pcls_per_step <= 0:
        return torch.zeros([], device='cuda')
    selected_idxs = np.random.choice(len(point_collections), num_pcls_per_step, replace=False)
    selected_pcls = [point_collections[i] for i in selected_idxs]

    for pcl in selected_pcls:
        # Pair patches either only with their immediate neighbour in the pcl's
        # patch ordering (first-seen order; built in main()),
        # or with every other patch.
        if cfg['winding_number_adjacent_patches_only']:
            cross_pairs = [(p1, p2) for p1, p2 in zip(pcl['points_by_patch'], list(pcl['points_by_patch'])[1:])]
        else:
            cross_pairs = list(itertools.combinations(pcl['points_by_patch'], r=2))
        if not cross_pairs:
            continue

        num_pairs_for_pcl = min(
            len(cross_pairs),
            cfg['winding_number_num_pairs'] // max(1, len(selected_pcls)),
        )
        if num_pairs_for_pcl <= 0:
            continue
        chosen = np.random.choice(len(cross_pairs), num_pairs_for_pcl, replace=False)
        pid_pairs = [cross_pairs[i] for i in chosen]

        for pid1, pid2 in pid_pairs:
            points1 = pcl['points_by_patch'][pid1]
            points2 = pcl['points_by_patch'][pid2]
            p1 = points1[np.random.randint(len(points1))]
            p2 = points2[np.random.randint(len(points2))]
            winding_diff = p2['winding_annotation'] - p1['winding_annotation']
            i1, j1 = int(p1['on_patch']['ij'][0]), int(p1['on_patch']['ij'][1])
            i2, j2 = int(p2['on_patch']['ij'][0]), int(p2['on_patch']['ij'][1])

            ls1 = _sample_l_shapes_at_ij(patches_dict[pid1], i1, j1, num_points_per_strip)
            ls2 = _sample_l_shapes_at_ij(patches_dict[pid2], i2, j2, num_points_per_strip)
            if ls1 is None or ls2 is None:
                continue
            strip_pairs.append((ls1, ls2, pid1, pid2, winding_diff))

    if not strip_pairs:
        return torch.zeros([], device='cuda')

    # Flatten: 8 strips per pair, ordered as p1's 4 strips followed by p2's 4 strips.
    total_strips = len(strip_pairs) * num_strips_per_pair
    flat_ijs = np.empty([total_strips, num_points_per_strip, 2], dtype=np.float32)
    flat_pids = []
    for k, (ls1, ls2, pid1, pid2, _) in enumerate(strip_pairs):
        base = k * num_strips_per_pair
        for s, strip in enumerate(ls1):
            flat_ijs[base + s] = strip
        for s, strip in enumerate(ls2):
            flat_ijs[base + num_strips_per_pcl + s] = strip
        flat_pids.extend([pid1] * num_strips_per_pcl + [pid2] * num_strips_per_pcl)

    # Group strips by patch and run ij_to_zyx in a single batched call per patch.
    flat_zyxs = torch.empty([total_strips, num_points_per_strip, 3], dtype=torch.float32)
    strips_by_patch = {}
    for k, pid in enumerate(flat_pids):
        strips_by_patch.setdefault(pid, []).append(k)
    for pid, strip_ks in strips_by_patch.items():
        idxs = np.asarray(strip_ks)
        ij_batch = torch.from_numpy(flat_ijs[idxs])
        zyx_batch, _ = patches_dict[pid].ij_to_zyx(ij_batch)
        flat_zyxs[idxs] = zyx_batch.to(dtype=torch.float32, device='cpu')

    flat_zyxs = flat_zyxs.cuda()
    flat_spiral = slice_to_spiral_transform(flat_zyxs.reshape(-1, 3)).reshape(*flat_zyxs.shape)
    theta, _, shifted_radii = get_theta_and_radii(flat_spiral[..., 1:], dr_per_winding)
    shifted_radii = _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding)

    # [num_pairs, 8, num_points_per_strip] -> pool each side's 4 strips into a single set.
    shifted_radii = shifted_radii.reshape(len(strip_pairs), num_strips_per_pair, num_points_per_strip)
    num_points_per_side = num_strips_per_pcl * num_points_per_strip
    p1_r = shifted_radii[:, :num_strips_per_pcl].reshape(len(strip_pairs), num_points_per_side)
    p2_r = shifted_radii[:, num_strips_per_pcl:].reshape(len(strip_pairs), num_points_per_side)

    winding_diffs = torch.tensor(
        [sp[4] for sp in strip_pairs],
        device='cuda',
        dtype=torch.float32,
    )
    expected_diff = (winding_diffs * dr_per_winding)[:, None, None]

    diff = p2_r[:, :, None] - p1_r[:, None, :]
    return (diff - expected_diff).abs().mean()


def _prepare_unattached_pcl_strips(pcls_dict):
    # For each unattached pcl, materialise an id-sorted strip of point zyxs and the
    # corresponding winding annotations. Strips with <2 points are dropped.
    prepared = []
    for pcl in pcls_dict.values():
        sorted_items = sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
        if len(sorted_items) < 2:
            continue
        zyxs = np.stack([p['zyx'] for _, p in sorted_items], axis=0).astype(np.float32)
        windings = np.array([p['winding_annotation'] for _, p in sorted_items], dtype=np.float32)
        prepared.append({'zyxs': zyxs, 'windings': windings})
    return prepared


def get_unattached_pcl_strip_losses(
    slice_to_spiral_transform,
    dr_per_winding,
    pcl_strips,
    num_pcls_per_step,
    num_points_per_pcl,
    compute_dt,
):
    # Unattached pcls are treated as ordered strips, indexed by int(point_id), and
    # assumed to be locally dense enough that adjacent samples have |dtheta| < pi
    # (so _unwrap_track_shifted_radii can stitch theta=0 crossings, exactly like a patch
    # row/column). Two losses are computed, analogous to the patch radius
    # and DT losses: (1) shifted-radius should be constant along the strip after
    # subtracting per-point winding-annotation offsets; (2) each point should snap to
    # its target winding, with the target taken from the snapped strip median.
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if not pcl_strips:
        return zero, zero

    num_to_sample = min(num_pcls_per_step, len(pcl_strips))
    chosen = np.random.choice(len(pcl_strips), num_to_sample, replace=False)

    sampled_zyxs = np.empty([num_to_sample, num_points_per_pcl, 3], dtype=np.float32)
    sampled_winding = np.empty([num_to_sample, num_points_per_pcl], dtype=np.float32)
    for k, pcl_idx in enumerate(chosen):
        strip = pcl_strips[pcl_idx]
        N = len(strip['zyxs'])
        coords = np.sort(np.random.choice(N, num_points_per_pcl, replace=num_points_per_pcl > N))
        sampled_zyxs[k] = strip['zyxs'][coords]
        sampled_winding[k] = strip['windings'][coords]

    zyxs_t = torch.from_numpy(sampled_zyxs).to(device=device)
    winding_t = torch.from_numpy(sampled_winding).to(device=device)

    spiral_zyxs = slice_to_spiral_transform(zyxs_t.reshape(-1, 3)).reshape(*zyxs_t.shape)
    theta, _, shifted_radii = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
    shifted_radii = _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding)

    # Normalise so a pcl with mixed annotations still reads as a single 'strip'.
    normalised_radii = shifted_radii - winding_t * dr_per_winding

    hinge_margin = dr_per_winding.detach() * cfg['radius_loss_margin']

    mean_radii = normalised_radii.mean(dim=-1, keepdim=True)
    radius_deviations = (normalised_radii - mean_radii).abs()
    radius_loss = F.relu(radius_deviations - hinge_margin).mean()

    if not compute_dt:
        return radius_loss, zero

    target_normalised = torch.round(normalised_radii.median(dim=-1, keepdim=True).values / dr_per_winding) * dr_per_winding
    target_shifted = target_normalised + winding_t * dr_per_winding
    target_radii = target_shifted + theta / (2 * np.pi) * dr_per_winding
    target_spiral_zyxs = torch.stack([
        spiral_zyxs[..., 0],
        torch.sin(theta) * target_radii,
        torch.cos(theta) * target_radii,
    ], dim=-1).detach()
    target_scroll_zyxs = slice_to_spiral_transform.inv(target_spiral_zyxs.reshape(-1, 3)).reshape(*target_spiral_zyxs.shape)

    within_p = cfg['patch_dt_within_patch_norm_p']
    across_p = cfg['patch_dt_norm_p']
    point_distances = torch.linalg.norm(zyxs_t - target_scroll_zyxs, dim=-1)
    point_distances = F.relu(point_distances - hinge_margin) + 1.e-5
    track_losses = (point_distances ** within_p).mean(dim=-1) ** (1 / within_p)
    dt_loss = ((track_losses ** across_p).sum() / track_losses.numel()) ** (1 / across_p)

    return radius_loss, dt_loss


def get_patch_stretch_and_normals_loss(slice_to_spiral_transform, num_points, patches, patch_sampling_probabilities, epsilon=1.0):
    # Sample patch points and enforce, at each point, two local rigidity properties of the scroll->spiral transform:
    #   (stretch) epsilon-steps along the discrete patch tangents (+i and +j neighbors in patch coordinates) preserve
    #             length in spiral space;
    #   (normals) an epsilon-step along the patch surface normal -- taken as the cross-product of the +i and +j patch
    #             coordinate deltas -- maps to a step in the outward radial direction in spiral space (since the
    #             spiral centre is at the origin, that's normalize(spiral_yx)). We only enforce direction match (not
    #             magnitude) since the local scale of the normal can differ from the in-surface scale.
    # The normal's sign relative to outward is ambiguous; |cosine| absorbs that.
    patch_idx = int(np.random.choice(len(patches), p=patch_sampling_probabilities))
    patch = patches[patch_idx]
    valid_indices = patch.valid_quad_indices
    sampled = np.random.randint(len(valid_indices), size=num_points)
    i_coords = valid_indices[sampled, 0]
    j_coords = valid_indices[sampled, 1]

    H, W = patch.valid_vertex_mask.shape
    in_bounds_i = i_coords + 1 < H
    in_bounds_j = j_coords + 1 < W
    safe_i_next = (i_coords + 1).clamp(max=H - 1)
    safe_j_next = (j_coords + 1).clamp(max=W - 1)
    neighbor_i_valid = patch.valid_vertex_mask[safe_i_next, j_coords] & in_bounds_i
    neighbor_j_valid = patch.valid_vertex_mask[i_coords, safe_j_next] & in_bounds_j
    # Require both neighbors so the cross-product normal is well-defined.
    mask = (neighbor_i_valid & neighbor_j_valid).float().cuda()

    scroll_zyx = patch.zyxs[i_coords, j_coords].cuda()
    scroll_neighbor_i = patch.zyxs[safe_i_next, j_coords].cuda()
    scroll_neighbor_j = patch.zyxs[i_coords, safe_j_next].cuda()
    delta_i = scroll_neighbor_i - scroll_zyx
    delta_j = scroll_neighbor_j - scroll_zyx
    tangent_i = F.normalize(delta_i, dim=-1)
    tangent_j = F.normalize(delta_j, dim=-1)
    scroll_normal = F.normalize(torch.linalg.cross(delta_i, delta_j, dim=-1), dim=-1)

    scroll_shift_i = scroll_zyx + tangent_i * epsilon
    scroll_shift_j = scroll_zyx + tangent_j * epsilon
    scroll_shift_n = scroll_zyx + scroll_normal * epsilon
    combined_scroll = torch.cat([scroll_zyx, scroll_shift_i, scroll_shift_j, scroll_shift_n], dim=0)
    combined_spiral = slice_to_spiral_transform(combined_scroll)
    spiral_zyx, spiral_shift_i, spiral_shift_j, spiral_shift_n = combined_spiral.chunk(4, dim=0)

    stretch_residual = 0.5 * (
        (torch.linalg.norm(spiral_shift_i - spiral_zyx, dim=-1) - epsilon).abs() +
        (torch.linalg.norm(spiral_shift_j - spiral_zyx, dim=-1) - epsilon).abs()
    )

    spiral_outward_yx = F.normalize(spiral_zyx[:, 1:], dim=-1)
    spiral_outward_zyx = torch.cat([torch.zeros_like(spiral_outward_yx[:, :1]), spiral_outward_yx], dim=-1)
    spiral_normal_step = F.normalize(spiral_shift_n - spiral_zyx, dim=-1)
    normals_residual = 1. - (spiral_normal_step * spiral_outward_zyx).sum(dim=-1).abs()

    denom = mask.sum().clamp(min=1)
    stretch_loss = (stretch_residual * mask).sum() / denom
    normals_loss = (normals_residual * mask).sum() / denom
    return stretch_loss, normals_loss


def get_patch_satisfied_areas(slice_to_spiral_transform, dr_per_winding, patches, verbose=False):
    """Per-patch satisfaction metrics.

    Returns ``(satisfied_patches, satisfied_areas, total_areas, satisfied_quad_masks,
    boundary_satisfied_count, target_winding_idx_per_patch)``: a bool flag per patch
    indicating whether at least ``metrics_config['satisfied_patch_quad_fraction']`` of
    its valid quads are satisfied, the satisfied/total area tensors, the per-patch
    (H-1, W-1) bool quad masks, a bool flag per patch indicating whether at least
    ``metrics_config['boundary_satisfied_patch_quad_fraction']`` of its boundary quads
    (in-ROI valid quads with at least one 4-neighbor that is out-of-bounds or not
    in-ROI-valid) are satisfied, and the per-patch (H-1, W-1) int64 winding-index
    tensors (the integer output-mesh winding each quad's snap-target sits on; -1 where
    the quad has no target set, e.g. invalid quads or quads in disconnected unwrap
    components).

    For each patch we first find valid quads whose footprint touches the z-ROI. Each
    such quad is then evaluated only at its center point, defined as the mean of its
    four scan-space corners. We (1) take a vertical column at the patch's central valid
    quad-column, (2) snap its median shifted-radius to the nearest integer-winding
    shifted-radius (the "target"), then (3) walk each quad-row outward from that center
    column, unwrapping shifted-radius across theta=0 crossings (signed, so left and
    right work alike). The satisfied area for the patch is patch.area scaled by
    satisfied-quads / valid-quads.

    A quad is satisfied when its center point passes both (a) the spiral-space
    shifted-radius tolerance of `satisfaction_radius_tolerance * dr_per_winding`, and
    (b) the absolute scan-space distance tolerance of
    `satisfaction_distance_tolerance` voxels to the corresponding point on the target
    winding.
    """
    spiral_tolerance = dr_per_winding.detach() * metrics_config['satisfaction_radius_tolerance']
    scan_tolerance = metrics_config['satisfaction_distance_tolerance']
    dr = dr_per_winding.detach()
    device = dr_per_winding.device

    satisfied_patches = torch.ones(len(patches), dtype=torch.bool)
    boundary_satisfied_patches = torch.ones(len(patches), dtype=torch.bool)
    satisfied_areas = torch.zeros(len(patches), dtype=torch.float64)
    total_areas = torch.zeros(len(patches), dtype=torch.float64)
    satisfied_quad_masks = [torch.zeros([max(p.zyxs.shape[0] - 1, 0), max(p.zyxs.shape[1] - 1, 0)], dtype=torch.bool) for p in patches]
    target_winding_idx_per_patch = [torch.full([max(p.zyxs.shape[0] - 1, 0), max(p.zyxs.shape[1] - 1, 0)], -1, dtype=torch.int64) for p in patches]

    with torch.no_grad():
        for patch_index, patch in enumerate(patches):
            patch_zyxs = patch.zyxs.to(device=device, dtype=torch.float32)
            patch_valid_quad_mask_full = patch.valid_quad_mask.to(device=device)
            quad_center_zyxs = (
                patch_zyxs[:-1, :-1]
                + patch_zyxs[1:, :-1]
                + patch_zyxs[:-1, 1:]
                + patch_zyxs[1:, 1:]
            ) / 4
            quad_zs = torch.stack([
                patch_zyxs[:-1, :-1, 0],
                patch_zyxs[1:, :-1, 0],
                patch_zyxs[:-1, 1:, 0],
                patch_zyxs[1:, 1:, 0],
            ], dim=0)
            quad_touches_roi_mask = (quad_zs.amax(dim=0) >= z_begin) & (quad_zs.amin(dim=0) < z_end)
            in_roi_valid_quad_mask = patch_valid_quad_mask_full & quad_touches_roi_mask

            total_full_valid_quads = int(patch_valid_quad_mask_full.sum().item())
            total_areas[patch_index] = float(patch.area) * int(in_roi_valid_quad_mask.sum().item()) / max(total_full_valid_quads, 1)
            if not in_roi_valid_quad_mask.any():
                continue

            Hq, Wq = quad_center_zyxs.shape[:2]
            valid_idx_i, valid_idx_j = torch.where(in_roi_valid_quad_mask)
            valid_zyxs = quad_center_zyxs[valid_idx_i, valid_idx_j]

            chunk = 65536
            spiral_pieces = []
            for start in range(0, valid_zyxs.shape[0], chunk):
                spiral_pieces.append(slice_to_spiral_transform(valid_zyxs[start : start + chunk]))
            spiral_zyxs_valid = torch.cat(spiral_pieces, dim=0) if len(spiral_pieces) > 1 else spiral_pieces[0]
            theta_v, _, shifted_radius_v = get_theta_and_radii(spiral_zyxs_valid[..., 1:], dr_per_winding)

            theta_all = torch.full([Hq, Wq], float('nan'), device=device)
            shifted_radius_all = torch.full([Hq, Wq], float('nan'), device=device)
            spiral_z_all = torch.full([Hq, Wq], float('nan'), device=device)
            theta_all[valid_idx_i, valid_idx_j] = theta_v
            shifted_radius_all[valid_idx_i, valid_idx_j] = shifted_radius_v
            spiral_z_all[valid_idx_i, valid_idx_j] = spiral_zyxs_valid[..., 0]

            cols_with_valid = torch.where(in_roi_valid_quad_mask.any(dim=0))[0]
            if len(cols_with_valid) == 0:
                continue
            center_col = int(cols_with_valid[len(cols_with_valid) // 2].item())

            satisfied_quad_mask = torch.zeros([Hq, Wq], dtype=torch.bool, device=device)
            target_raw_shifted_all = torch.full([Hq, Wq], float('nan'), device=device)
            valid_quad_mask_np = in_roi_valid_quad_mask.cpu().numpy()
            row_infos = [None] * Hq

            def seed_branch_offset(subrow, anchor_col):
                anchor_pos = min(max(anchor_col - subrow['j_min'], 0), subrow['unwrapped_shifted'].numel() - 1)
                subrow['branch_offset'] = subrow['cum_adj'][anchor_pos]

            def propagate_branch_offset(source, source_pos, target, target_pos):
                if source['branch_offset'] is None or target['branch_offset'] is not None:
                    return False
                shifted_diff = target['unwrapped_shifted'][target_pos] - source['unwrapped_shifted'][source_pos]
                winding_delta = torch.round(shifted_diff / dr) * dr
                target['branch_offset'] = source['branch_offset'] + winding_delta
                return True

            all_subrows = []

            for i in range(Hq):
                row_valid = valid_quad_mask_np[i]
                if not np.any(row_valid):
                    continue
                steps = np.nonzero(np.diff(np.concatenate([[0], row_valid, [0]])))[0]
                subrows = np.stack([steps[::2], steps[1::2]], axis=1)
                subrow_infos = []
                for j_min, j_max in subrows:
                    row_thetas = theta_all[i, j_min:j_max]
                    row_shifted = shifted_radius_all[i, j_min:j_max]
                    if row_thetas.numel() <= 1:
                        cum_adj = torch.zeros_like(row_thetas)
                    else:
                        theta_diffs = row_thetas[1:] - row_thetas[:-1]
                        # Signed: theta_diff < -pi means we wrapped 2pi->0+ (theta jumped down),
                        # so naive shifted_radius is too high by dr; subtract. Opposite for >+pi.
                        step_adj = ((theta_diffs > np.pi).to(row_thetas.dtype) - (theta_diffs < -np.pi).to(row_thetas.dtype)) * dr
                        cum_adj = torch.cat([torch.zeros([1], device=device, dtype=row_thetas.dtype), torch.cumsum(step_adj, dim=0)], dim=0)
                    subrow_infos.append({
                        'row_idx': i,
                        'j_min': int(j_min),
                        'j_max': int(j_max),
                        'cum_adj': cum_adj,
                        'unwrapped_shifted': row_shifted + cum_adj,
                        'branch_offset': None,
                        'neighbors': [],
                    })
                row_infos[i] = subrow_infos
                all_subrows.extend(subrow_infos)

            for i in range(Hq - 1):
                upper_subrows = row_infos[i]
                lower_subrows = row_infos[i + 1]
                if upper_subrows is None or lower_subrows is None:
                    continue
                upper_idx = 0
                lower_idx = 0
                while upper_idx < len(upper_subrows) and lower_idx < len(lower_subrows):
                    upper = upper_subrows[upper_idx]
                    lower = lower_subrows[lower_idx]
                    overlap_min = max(upper['j_min'], lower['j_min'])
                    overlap_max = min(upper['j_max'], lower['j_max'])
                    if overlap_max > overlap_min:
                        j_anchor = (overlap_min + overlap_max - 1) // 2
                        upper_pos = j_anchor - upper['j_min']
                        lower_pos = j_anchor - lower['j_min']
                        upper['neighbors'].append((lower, upper_pos, lower_pos))
                        lower['neighbors'].append((upper, lower_pos, upper_pos))
                    if upper['j_max'] <= lower['j_max']:
                        upper_idx += 1
                    else:
                        lower_idx += 1

            rows_with_center = torch.where(in_roi_valid_quad_mask[:, center_col])[0]
            if len(rows_with_center) == 0:
                continue
            center_row = int(rows_with_center[len(rows_with_center) // 2].item())
            center_subrows = row_infos[center_row]
            center_subrow = None
            for subrow in center_subrows:
                if subrow['j_min'] <= center_col < subrow['j_max']:
                    seed_branch_offset(subrow, center_col)
                    center_subrow = subrow
                    break
            if center_subrow is None:
                continue

            queue = [center_subrow]
            queue_pos = 0
            while queue_pos < len(queue):
                source = queue[queue_pos]
                queue_pos += 1
                for target, source_pos, target_pos in source['neighbors']:
                    if propagate_branch_offset(source, source_pos, target, target_pos):
                        queue.append(target)

            component_center_rows = [
                subrow['row_idx']
                for subrow in all_subrows
                if subrow['branch_offset'] is not None and subrow['j_min'] <= center_col < subrow['j_max']
            ]
            if len(component_center_rows) == 0:
                continue
            component_center_rows_t = torch.tensor(component_center_rows, device=device, dtype=torch.long)
            component_col_shifted = shifted_radius_all[component_center_rows_t, center_col]
            median_shifted_radius = torch.median(component_col_shifted)
            modulus = median_shifted_radius % dr
            target_shifted_radius = torch.where(
                modulus < dr / 2,
                median_shifted_radius - modulus,
                median_shifted_radius + dr - modulus,
            )

            if verbose and any(subrow['branch_offset'] is None for subrow in all_subrows):
                print(f'Warning: patch {patch_index} has multiple disconnected subrow components; using only the component containing the center column')

            for subrow in all_subrows:
                branch_offset = subrow['branch_offset']
                if branch_offset is None:
                    continue
                i = subrow['row_idx']
                j_min = subrow['j_min']
                j_max = subrow['j_max']
                cum_adj = subrow['cum_adj']
                adjusted_shifted = subrow['unwrapped_shifted'] - branch_offset

                in_band = (adjusted_shifted - target_shifted_radius).abs() <= spiral_tolerance
                satisfied_quad_mask[i, j_min:j_max] = in_band

                # Per-quad raw target shifted-radius (consistent with the unwrap, so the
                # target sits on the same physical winding across theta=0 crossings).
                target_raw_shifted_all[i, j_min:j_max] = target_shifted_radius - cum_adj + branch_offset

            # Scan-space distance check: for every quad-center with a per-row target set,
            # build the corresponding spiral-space point on the target winding (same
            # theta, same z, target shifted-radius), invert to scan space, and require
            # the scan-voxel distance to the original quad-center be within tolerance.
            target_set_mask = (~torch.isnan(target_raw_shifted_all)) & in_roi_valid_quad_mask
            scan_in_band = torch.zeros([Hq, Wq], dtype=torch.bool, device=device)
            if target_set_mask.any():
                sel_i, sel_j = torch.where(target_set_mask)
                theta_sel = theta_all[sel_i, sel_j]
                target_raw_sel = target_raw_shifted_all[sel_i, sel_j]
                target_radius_sel = target_raw_sel + theta_sel / (2 * np.pi) * dr
                target_spiral_zyx_sel = torch.stack([
                    spiral_z_all[sel_i, sel_j],
                    torch.sin(theta_sel) * target_radius_sel,
                    torch.cos(theta_sel) * target_radius_sel,
                ], dim=-1)
                orig_scan_sel = quad_center_zyxs[sel_i, sel_j]
                target_scan_pieces = []
                for start in range(0, target_spiral_zyx_sel.shape[0], chunk):
                    target_scan_pieces.append(slice_to_spiral_transform.inv(target_spiral_zyx_sel[start : start + chunk]))
                target_scan_sel = torch.cat(target_scan_pieces, dim=0) if len(target_scan_pieces) > 1 else target_scan_pieces[0]
                scan_distances_sel = torch.linalg.norm(target_scan_sel - orig_scan_sel, dim=-1)
                scan_in_band[sel_i, sel_j] = scan_distances_sel <= scan_tolerance

            satisfied_quad_mask = satisfied_quad_mask & scan_in_band & in_roi_valid_quad_mask

            # Per-quad output-mesh winding index, derived from the raw (per-row) target
            # shifted-radius. NaN entries (quads without a target set) become -1.
            target_winding_idx_full = torch.where(
                torch.isnan(target_raw_shifted_all),
                torch.full_like(target_raw_shifted_all, -1.),
                torch.round(target_raw_shifted_all / dr),
            ).to(torch.int64)
            target_winding_idx_per_patch[patch_index] = target_winding_idx_full.cpu()

            # Boundary = in-ROI-valid quads with at least one 4-neighbor that is
            # out-of-bounds or not in in_roi_valid_quad_mask.
            padded = torch.nn.functional.pad(in_roi_valid_quad_mask, (1, 1, 1, 1), value=False)
            all_neighbors_in = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
            boundary_quad_mask = in_roi_valid_quad_mask & ~all_neighbors_in

            total_valid_quads = int(in_roi_valid_quad_mask.sum().item())
            satisfied_quad_masks[patch_index] = satisfied_quad_mask.cpu()
            if total_valid_quads == 0:
                continue
            num_satisfied_quads = int(satisfied_quad_mask.sum().item())
            satisfied_areas[patch_index] = float(patch.area) * num_satisfied_quads / max(total_full_valid_quads, 1)
            satisfied_patches[patch_index] = num_satisfied_quads >= metrics_config['satisfied_patch_quad_fraction'] * total_valid_quads
            num_boundary_quads = int(boundary_quad_mask.sum().item())
            if num_boundary_quads > 0:
                num_satisfied_boundary_quads = int((boundary_quad_mask & satisfied_quad_mask).sum().item())
                boundary_satisfied_patches[patch_index] = num_satisfied_boundary_quads >= metrics_config['boundary_satisfied_patch_quad_fraction'] * num_boundary_quads

    return satisfied_patches, satisfied_areas, total_areas, satisfied_quad_masks, boundary_satisfied_patches, target_winding_idx_per_patch


def get_unattached_pcl_satisfied_counts(slice_to_spiral_transform, dr_per_winding, pcl_strips):
    # For each unattached pcl, treat its id-sorted points as a strip (so theta=0
    # crossings can be unwrapped, mirroring the patch row-walk in
    # get_patch_satisfied_areas), pick the snapped median normalised shifted-radius
    # as the target winding, then count points that satisfy both the same spiral-
    # space radius tolerance and the same scan-space distance tolerance used for
    # quad satisfaction. Returns three values: (satisfied_count_per_pcl,
    # total_count_per_pcl, per_point_satisfaction) — the first two are 1-D int64
    # tensors, and per_point_satisfaction is a list of CPU bool tensors (one per
    # pcl, of length N for that pcl; empty pcls get an empty tensor).
    spiral_tolerance = dr_per_winding.detach() * metrics_config['satisfaction_radius_tolerance']
    scan_tolerance = metrics_config['satisfaction_distance_tolerance']
    dr = dr_per_winding.detach()
    device = dr_per_winding.device

    satisfied_counts = torch.zeros(len(pcl_strips), dtype=torch.int64)
    total_counts = torch.zeros(len(pcl_strips), dtype=torch.int64)
    per_point_satisfaction = []
    with torch.no_grad():
        for k, strip in enumerate(pcl_strips):
            N = strip['zyxs'].shape[0]
            total_counts[k] = N
            if N == 0:
                per_point_satisfaction.append(torch.zeros([0], dtype=torch.bool))
                continue
            zyxs = torch.from_numpy(strip['zyxs']).to(device=device, dtype=torch.float32)
            windings = torch.from_numpy(strip['windings']).to(device=device, dtype=torch.float32)

            spiral_zyxs = slice_to_spiral_transform(zyxs)
            theta, _, shifted_radii = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
            unwrapped_shifted = _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding)

            normalised_radii = unwrapped_shifted - windings * dr
            target_normalised = torch.round(normalised_radii.median() / dr) * dr
            target_shifted = target_normalised + windings * dr
            spiral_in_band = (unwrapped_shifted - target_shifted).abs() <= spiral_tolerance

            target_radii = target_shifted + theta / (2 * np.pi) * dr
            target_spiral_zyxs = torch.stack([
                spiral_zyxs[..., 0],
                torch.sin(theta) * target_radii,
                torch.cos(theta) * target_radii,
            ], dim=-1)
            target_scroll_zyxs = slice_to_spiral_transform.inv(target_spiral_zyxs)
            scan_distances = torch.linalg.norm(target_scroll_zyxs - zyxs, dim=-1)
            scan_in_band = scan_distances <= scan_tolerance

            satisfied = spiral_in_band & scan_in_band
            satisfied_counts[k] = int(satisfied.sum().item())
            per_point_satisfaction.append(satisfied.cpu())
    return satisfied_counts, total_counts, per_point_satisfaction


def get_face_indices(h, w):
    indices = torch.arange(h * w).view(h, w)
    top_left = indices[:-1, :-1].flatten()
    top_right = indices[:-1, 1:].flatten()
    bottom_left = indices[1:, :-1].flatten()
    bottom_right = indices[1:, 1:].flatten()
    return torch.cat([
        torch.stack([bottom_left, top_left, top_right], dim=1),
        torch.stack([bottom_left, top_right, bottom_right], dim=1)
    ], dim=0)


@torch.inference_mode
def _get_working_set_winding_range(slice_to_spiral_transform, dr_per_winding, working_set_patches):
    """Returns (min_winding_idx, max_winding_idx) — inclusive min and exclusive max
    integer winding indices covered by the given patches' in-ROI valid quad centers.
    Winding indices are computed by transforming quad centers to spiral space and
    rounding shifted_radius/dr_per_winding to the nearest integer (matching the
    convention in overlay_patches_on_slices)."""
    device = dr_per_winding.device
    dr = dr_per_winding.detach()
    min_w = None
    max_w = None
    for patch in working_set_patches:
        patch_zyxs = patch.zyxs.to(device=device, dtype=torch.float32)
        if patch_zyxs.shape[0] < 2 or patch_zyxs.shape[1] < 2:
            continue
        valid_quad_mask = patch.valid_quad_mask.to(device=device)
        quad_center_zyxs = (
            patch_zyxs[:-1, :-1]
            + patch_zyxs[1:, :-1]
            + patch_zyxs[:-1, 1:]
            + patch_zyxs[1:, 1:]
        ) / 4
        quad_zs = torch.stack([
            patch_zyxs[:-1, :-1, 0],
            patch_zyxs[1:, :-1, 0],
            patch_zyxs[:-1, 1:, 0],
            patch_zyxs[1:, 1:, 0],
        ], dim=0)
        quad_touches_roi = (quad_zs.amax(dim=0) >= z_begin) & (quad_zs.amin(dim=0) < z_end)
        mask = valid_quad_mask & quad_touches_roi
        if not mask.any():
            continue
        zyxs = quad_center_zyxs[mask]
        chunk = 65536
        spiral_pieces = []
        for start in range(0, zyxs.shape[0], chunk):
            spiral_pieces.append(slice_to_spiral_transform(zyxs[start:start + chunk]))
        spiral_zyxs = torch.cat(spiral_pieces, dim=0) if len(spiral_pieces) > 1 else spiral_pieces[0]
        _, _, shifted_radius = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
        winding_indices = (shifted_radius / dr).round().to(torch.int64).clamp_min(0)
        patch_min = int(winding_indices.min().item())
        patch_max = int(winding_indices.max().item())
        min_w = patch_min if min_w is None else min(min_w, patch_min)
        max_w = patch_max if max_w is None else max(max_w, patch_max)
    cfg_min, cfg_max = cfg['output_winding_range']
    if min_w is None:
        return cfg_min, cfg_max
    margin = cfg['output_winding_margin']
    return max(min_w - margin, cfg_min), min(max_w + 1 + margin, cfg_max)


def _rasterize_triangles_into_mesh(
    tri_uvs,  # (T, 3, 2) — (u, v) per triangle vertex
    tri_scrolls,  # (T, 3, 3) — scroll zyx per triangle vertex
    tri_target_w,  # (T,) — output winding index per triangle
    scroll_zyxs,  # (num_zs, total_thetas, 3) — modified in place
    winding_offsets_t,  # (num_windings + 1,) cumulative theta-offset per winding
    num_thetas_t,  # (num_windings,) — num theta points per output winding
):
    device = scroll_zyxs.device
    T = tri_uvs.shape[0]
    if T == 0:
        return

    num_zs = scroll_zyxs.shape[0]
    v_lim_per_tri = num_thetas_t[tri_target_w]  # (T,)

    u_min = torch.floor(tri_uvs[..., 0].min(dim=-1).values).to(torch.long)
    u_max = torch.ceil(tri_uvs[..., 0].max(dim=-1).values).to(torch.long)
    v_min = torch.floor(tri_uvs[..., 1].min(dim=-1).values).to(torch.long)
    v_max = torch.ceil(tri_uvs[..., 1].max(dim=-1).values).to(torch.long)
    u_min = u_min.clamp(min=0, max=num_zs - 1)
    u_max = u_max.clamp(min=0, max=num_zs - 1)
    v_min = v_min.clamp(min=0)
    v_max = torch.minimum(v_max, v_lim_per_tri - 1)
    valid_bbox = (u_min <= u_max) & (v_min <= v_max)

    bbox_h = (u_max - u_min + 1).clamp(min=1)
    bbox_w = (v_max - v_min + 1).clamp(min=1)
    max_h = int(bbox_h[valid_bbox].max().item()) if valid_bbox.any() else 1
    max_w = int(bbox_w[valid_bbox].max().item()) if valid_bbox.any() else 1

    du_grid, dv_grid = torch.meshgrid(
        torch.arange(max_h, device=device),
        torch.arange(max_w, device=device),
        indexing='ij',
    )
    us = u_min[:, None, None] + du_grid[None]  # (T, max_h, max_w)
    vs = v_min[:, None, None] + dv_grid[None]
    in_bbox = (
        (du_grid[None] < bbox_h[:, None, None])
        & (dv_grid[None] < bbox_w[:, None, None])
        & valid_bbox[:, None, None]
    )

    pts = torch.stack([us.float(), vs.float()], dim=-1)
    a = tri_uvs[:, 0]
    b = tri_uvs[:, 1]
    c = tri_uvs[:, 2]
    v0 = b - a
    v1 = c - a
    v2 = pts - a[:, None, None]
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0[:, None, None]).sum(-1)
    d21 = (v2 * v1[:, None, None]).sum(-1)
    denom = d00 * d11 - d01 * d01
    nonzero = denom.abs() >= 1e-9
    denom_safe = torch.where(nonzero, denom, torch.ones_like(denom))
    beta = (d11[:, None, None] * d20 - d01[:, None, None] * d21) / denom_safe[:, None, None]
    gamma = (d00[:, None, None] * d21 - d01[:, None, None] * d20) / denom_safe[:, None, None]
    alpha = 1 - beta - gamma
    inside = (alpha >= -1e-6) & (beta >= -1e-6) & (gamma >= -1e-6) & nonzero[:, None, None]
    mask = in_bbox & inside
    if not mask.any():
        return

    sa = tri_scrolls[:, 0][:, None, None, :]
    sb = tri_scrolls[:, 1][:, None, None, :]
    sc = tri_scrolls[:, 2][:, None, None, :]
    interp = alpha[..., None] * sa + beta[..., None] * sb + gamma[..., None] * sc

    sel = mask.reshape(-1)
    target_u = us.reshape(-1)[sel]
    target_v_local = vs.reshape(-1)[sel]
    target_w_flat = tri_target_w[:, None, None].expand(-1, max_h, max_w).reshape(-1)[sel]
    target_v_global = winding_offsets_t[target_w_flat] + target_v_local
    scroll_zyxs[target_u, target_v_global] = interp.reshape(-1, 3)[sel]


@torch.inference_mode
def _build_snapped_overlay(
    scroll_zyxs,  # (num_zs, total_thetas, 3) — modified in place
    num_thetas_by_winding,  # list[int]
    z0,  # scalar — spiral z corresponding to scroll_zyxs[0]
    grid_spacing,  # int — z and theta-arc-length step in spiral coords
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    satisfied_patches,
    boundary_satisfied_patches,
    target_winding_idx_per_patch,
):
    device = scroll_zyxs.device
    dr = dr_per_winding.detach()
    num_windings = len(num_thetas_by_winding)
    winding_offsets_t = torch.cat([
        torch.zeros([1], dtype=torch.long, device=device),
        torch.cumsum(torch.tensor(num_thetas_by_winding, dtype=torch.long, device=device), dim=0),
    ])
    num_thetas_t = torch.tensor(num_thetas_by_winding, dtype=torch.long, device=device)

    all_quad_uvs = []
    all_quad_scrolls = []
    all_target_w = []
    chunk = 65536

    for patch_idx, patch in enumerate(patches):
        # Gate on patch-level satisfaction: snap every quad of an overall- or
        # boundary-satisfied patch (even the ones that individually fail the
        # spiral/scan tolerance), as long as the satisfaction pass figured out
        # which integer winding they sit on (target_winding_idx_per_patch >= 0).
        if not (bool(satisfied_patches[patch_idx]) or bool(boundary_satisfied_patches[patch_idx])):
            continue
        target_winding_idx = target_winding_idx_per_patch[patch_idx].to(device)
        quad_mask = (target_winding_idx >= 0) & (target_winding_idx < num_windings)
        if not quad_mask.any():
            continue

        patch_zyxs = patch.zyxs.to(device=device, dtype=torch.float32)
        Hv, Wv = patch_zyxs.shape[:2]

        def chunked_transform(flat):
            pieces = []
            for s in range(0, flat.shape[0], chunk):
                pieces.append(slice_to_spiral_transform(flat[s : s + chunk]))
            return torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]

        vertex_spiral = chunked_transform(patch_zyxs.reshape(-1, 3)).reshape(Hv, Wv, 3)
        v_theta, _, v_shifted = get_theta_and_radii(vertex_spiral[..., 1:], dr_per_winding)
        v_winding_raw = (v_shifted / dr).round().to(torch.int64)

        quad_center_scroll = (
            patch_zyxs[:-1, :-1]
            + patch_zyxs[1:, :-1]
            + patch_zyxs[:-1, 1:]
            + patch_zyxs[1:, 1:]
        ) / 4
        center_spiral = chunked_transform(quad_center_scroll.reshape(-1, 3)).reshape(*quad_center_scroll.shape)
        c_theta, _, _ = get_theta_and_radii(center_spiral[..., 1:], dr_per_winding)

        qi, qj = torch.where(quad_mask)
        w_target = target_winding_idx[qi, qj].to(torch.float32)
        # Reference full angle at the quad center, anchored to the satisfaction
        # pass's target winding so unsatisfied quads (whose own raw c_shifted may
        # round to a neighbouring winding) still snap into the target's frame.
        ref_full = c_theta[qi, qj] + w_target * (2 * np.pi)

        # Vertex indices for each quad corner: 0=(i,j), 1=(i,j+1), 2=(i+1,j), 3=(i+1,j+1)
        vi = torch.stack([qi, qi, qi + 1, qi + 1], dim=-1)
        vj = torch.stack([qj, qj + 1, qj, qj + 1], dim=-1)
        vert_spiral = vertex_spiral[vi, vj]  # (N, 4, 3)
        vert_scroll = patch_zyxs[vi, vj]  # (N, 4, 3)
        vert_theta = v_theta[vi, vj]  # (N, 4)
        vert_w_raw = v_winding_raw[vi, vj].to(torch.float32)
        vert_full = vert_theta + vert_w_raw * (2 * np.pi)
        # Snap each vertex's full angle to be within pi of the center's reference.
        diff = vert_full - ref_full[:, None]
        vert_full_snapped = vert_full - torch.round(diff / (2 * np.pi)) * (2 * np.pi)

        # UV coords in the target winding's mesh.
        u_coords = (vert_spiral[..., 0] - z0) / grid_spacing  # (N, 4)
        theta_step_per_quad = grid_spacing / ((w_target + 0.5) * dr)  # (N,)
        v_coords = (vert_full_snapped - w_target[:, None] * (2 * np.pi)) / theta_step_per_quad[:, None]  # (N, 4)

        all_quad_uvs.append(torch.stack([u_coords, v_coords], dim=-1))
        all_quad_scrolls.append(vert_scroll)
        all_target_w.append(target_winding_idx[qi, qj])

    if not all_quad_uvs:
        return

    quad_uvs = torch.cat(all_quad_uvs, dim=0)  # (Nq, 4, 2)
    quad_scrolls = torch.cat(all_quad_scrolls, dim=0)  # (Nq, 4, 3)
    quad_target_w = torch.cat(all_target_w, dim=0)  # (Nq,)
    Nq = quad_uvs.shape[0]

    # Split each quad into 2 triangles along the (0,3) diagonal:
    #   v00=(i,j) (0), v01=(i,j+1) (1), v10=(i+1,j) (2), v11=(i+1,j+1) (3)
    # Triangle order: (v00, v01, v11), (v00, v11, v10)
    tri_local = torch.tensor([[0, 1, 3], [0, 3, 2]], device=device, dtype=torch.long)
    quad_repeat = torch.arange(Nq, device=device).repeat_interleave(2)
    tri_local_flat = tri_local.unsqueeze(0).expand(Nq, -1, -1).reshape(-1, 3)
    tri_uvs = quad_uvs[quad_repeat[:, None].expand(-1, 3), tri_local_flat]
    tri_scrolls = quad_scrolls[quad_repeat[:, None].expand(-1, 3), tri_local_flat]
    tri_target_w = quad_target_w[quad_repeat]

    _rasterize_triangles_into_mesh(
        tri_uvs, tri_scrolls, tri_target_w,
        scroll_zyxs, winding_offsets_t, num_thetas_t,
    )


@torch.inference_mode
def save_mesh(slice_to_spiral_transform, dr_per_winding, working_set_patches, out_path, name='mesh'):

    min_winding_idx, max_winding_idx = _get_working_set_winding_range(slice_to_spiral_transform, dr_per_winding, working_set_patches)
    print(f'save_mesh {name}: adaptive winding range [{min_winding_idx}, {max_winding_idx})')
    grid_spacing = cfg['output_step_size'] // downsample_factor  # voxels in downsampled volume
    z_margin = cfg['flow_bounds_z_margin']
    spiral_yxs_by_winding = get_spiral_yxs(max_winding_idx, dr_per_winding, grid_spacing, group_by_winding=True)
    num_thetas_by_winding = [len(yxs_for_winding) for yxs_for_winding in spiral_yxs_by_winding]
    spiral_yxs = torch.cat(spiral_yxs_by_winding, dim=0)
    z0 = z_begin - z_margin
    spiral_zs = torch.arange(z0, z_end + z_margin, grid_spacing, dtype=torch.float32, device=spiral_yxs.device)
    spiral_zyxs = torch.cat([spiral_zs[:, None, None].expand(-1, spiral_yxs.shape[0], 1), spiral_yxs[None, :, :].expand(spiral_zs.shape[0], -1, 2)], dim=-1)
    scroll_zyxs = slice_to_spiral_transform.inv(spiral_zyxs)

    # Snapped variant: replace cells covered by quads of overall- or boundary-satisfied
    # patches with patch-derived points, interpolated bilinearly across each quad in the
    # flattened-spiral UV coords of its target winding.
    snapped_scroll_zyxs = scroll_zyxs.clone()
    satisfied_patches, _, _, _, boundary_satisfied_patches, target_winding_idx_per_patch = get_patch_satisfied_areas(
        slice_to_spiral_transform, dr_per_winding, working_set_patches,
    )
    _build_snapped_overlay(
        snapped_scroll_zyxs, num_thetas_by_winding, z0, grid_spacing,
        slice_to_spiral_transform, dr_per_winding,
        working_set_patches,
        satisfied_patches, boundary_satisfied_patches, target_winding_idx_per_patch,
    )

    step_size = grid_spacing * downsample_factor
    out_dir = f'{out_path}/meshes/{name}'
    os.makedirs(out_dir, exist_ok=True)
    for uuid_suffix, variant_zyxs in [('', scroll_zyxs), ('_snapped', snapped_scroll_zyxs)]:
        offset = 0
        for winding_idx, num_thetas in enumerate(tqdm(num_thetas_by_winding, desc=f'saving winding patches ({name}{uuid_suffix})')):
            if num_thetas >= 2 and winding_idx >= min_winding_idx:
                winding_zyxs = (variant_zyxs[:, offset:offset + num_thetas] * downsample_factor).cpu().numpy().astype(np.float32)
                save_tifxyz(
                    winding_zyxs,
                    out_dir,
                    uuid=f'w{winding_idx:03d}{uuid_suffix}',
                    step_size=step_size,
                    voxel_size_um=voxel_size_um * downsample_factor,
                    source=f'fit_spiral {name}{uuid_suffix}',
                )
            offset += num_thetas


def _compute_patch_lines_by_slice(patches, slice_zs):
    all_zs = np.array(slice_zs)
    patch_ids_str = '_'.join(sorted(str(id(p)) for p in patches))
    hashed_patches = hashlib.sha256(bytes(patch_ids_str, 'ascii')).hexdigest()[:8]
    cache_filename = f'{cache_path}/patches-{hashed_patches}_lines_v3_ds-{downsample_factor}_slices-{hash(tuple(all_zs))}.pkl'

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as fp:
            lines_by_slice = pickle.load(fp)
    else:
        lines_by_slice = [[] for _ in all_zs]
        for patch_idx, patch in enumerate(tqdm(patches, desc='slicing patches')):
            mesh = patch.to_trimesh()
            if mesh.faces.shape[0] == 0:
                continue
            patch_min_z, patch_max_z = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
            relevant_zs_mask = (all_zs >= patch_min_z) & (all_zs <= patch_max_z)
            relevant_zs = all_zs[relevant_zs_mask]
            if len(relevant_zs) == 0:
                continue
            lines, to_3ds, face_indices = trimesh.intersections.mesh_multiplane(mesh, [0, 0, 0], [0, 0, 1], relevant_zs)
            # lines has one entry per slice; each entry is a tensor of lines represented by pairs of yx points, indexed [line-idx, start/end, yx]
            # face_indices[slice] gives the trimesh face index per line; we map back through the valid-quad filter to a quad index within the patch grid.
            assert all((to_3d[:3, :3] == np.eye(3)).all() for to_3d in to_3ds)
            assert (to_3ds[:, 2, 3] == relevant_zs).all()
            valid_quads_flat = patch.valid_quad_mask.flatten().numpy()
            num_quads_grid = valid_quads_flat.shape[0]
            face_mask = np.concatenate([valid_quads_flat, valid_quads_flat], axis=0)
            filtered_face_to_full = np.where(face_mask)[0]
            relevant_indices = np.where(relevant_zs_mask)[0]
            for local_idx, global_idx in enumerate(relevant_indices):
                if lines[local_idx] is not None and len(lines[local_idx]) > 0:
                    line_face_indices = np.asarray(face_indices[local_idx], dtype=np.int64)
                    line_quad_indices = (filtered_face_to_full[line_face_indices] % num_quads_grid).astype(np.int64)
                    lines_by_slice[global_idx].append((patch_idx, lines[local_idx], line_quad_indices))
        os.makedirs(cache_path, exist_ok=True)
        with open(cache_filename, 'wb') as fp:
            pickle.dump(lines_by_slice, fp)

    return lines_by_slice


def overlay_patches_on_slices(patches, slice_zs, slice_shape):
    color_slices = torch.zeros([len(slice_zs), *slice_shape, 3], dtype=torch.uint8)
    quad_label_slices = torch.zeros(color_slices.shape[:-1], dtype=torch.int32, device=color_slices.device)
    lines_by_slice = _compute_patch_lines_by_slice(patches, slice_zs)
    patch_colors = [tuple(np.random.randint(100, 256, size=3).tolist()) for _ in patches]

    # Each patch contributes (H-1)*(W-1) grid quads (row-major, including invalid ones — they
    # never appear as lines, but indexing this way matches the flat satisfied_quad_mask layout).
    quad_offsets = np.zeros(len(patches) + 1, dtype=np.int64)
    for patch_idx, patch in enumerate(patches):
        h, w = patch.zyxs.shape[:2]
        quad_offsets[patch_idx + 1] = quad_offsets[patch_idx] + (h - 1) * (w - 1)

    for slice_idx in tqdm(range(len(slice_zs)), desc='rasterising patch lines'):
        slice_lines = lines_by_slice[slice_idx]
        color_img = Image.fromarray(np.zeros(color_slices.shape[1:4], dtype=np.uint8))
        quad_label_img = Image.fromarray(np.zeros(quad_label_slices.shape[1:3], dtype=np.int32), mode='I')
        color_draw = ImageDraw.Draw(color_img)
        quad_label_draw = ImageDraw.Draw(quad_label_img)
        for patch_idx, lines, line_quad_indices in slice_lines:
            color = patch_colors[patch_idx] if patch_idx < len(patch_colors) else (255, 255, 255)
            offset = int(quad_offsets[patch_idx])
            for line_xy, quad_idx in zip(lines, line_quad_indices):
                line_points = line_xy.flatten().tolist()
                color_draw.line(line_points, fill=color)
                quad_label_draw.line(line_points, fill=int(offset + int(quad_idx) + 1))
        color_slices[slice_idx] = torch.from_numpy(np.array(color_img))
        quad_label_slices[slice_idx] = torch.from_numpy(np.array(quad_label_img))
    return color_slices, quad_label_slices, quad_offsets


def get_winding_positions_on_radials(slice_z, thetas, max_radius, slice_to_spiral_transform, dr_per_winding, z_to_umbilicus_yx):
    theta_slice, radius_slice = torch.meshgrid(thetas, torch.arange(1., max_radius), indexing='ij')
    radials_yx_slice = torch.from_numpy(z_to_umbilicus_yx(slice_z.cpu()).astype(np.float32)) + torch.stack([torch.sin(theta_slice), torch.cos(theta_slice)], dim=-1) * radius_slice[..., None]
    radials_zyx_slice = torch.cat([slice_z.expand(radials_yx_slice.shape[:2])[..., None], radials_yx_slice.cuda()], dim=-1)
    radials_zyx_spiral = slice_to_spiral_transform(radials_zyx_slice)
    _, _, inner_winding_idx, _ = get_bounding_windings(radials_zyx_spiral[..., 1:], dr_per_winding)
    radii_by_radial = []
    yxs_by_radial = []
    winding_indices_by_radial = []
    for radial_idx in range(inner_winding_idx.shape[0]):
        winding_change_indices = torch.where(torch.diff(inner_winding_idx[radial_idx], prepend=inner_winding_idx[radial_idx, :1]))[0].cpu()
        radii_by_radial.append(radius_slice[radial_idx, winding_change_indices])
        yxs_by_radial.append(radials_yx_slice[radial_idx, winding_change_indices])
        winding_indices_by_radial.append(inner_winding_idx[radial_idx, winding_change_indices])
    return radii_by_radial, yxs_by_radial, winding_indices_by_radial


@torch.inference_mode
def save_overlay(
    spiral_and_transform,
    flow_min_corner_spiral_zyx, flow_max_corner_spiral_zyx,
    zs_for_visualisation, all_zs, slice_yx,
    scroll_slices_for_visualisation, prediction_slices_for_visualisation,
    quad_label_map, quad_status_flat,
    unattached_pcl_strips, unattached_pcl_per_point_satisfied, unattached_pcl_fully_satisfied,
    umbilicus_zyx, z_to_umbilicus_yx,
    out_path, suffix
):

    device = slice_yx.device
    slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()

    # TODO: maybe use the smoothed umbilicus here, to avoid weird swirls appearing
    flow_corners_zyx = slice_to_spiral_transform.inv(torch.stack([flow_min_corner_spiral_zyx, flow_max_corner_spiral_zyx], dim=0).to(torch.float32)).to(torch.int64)
    flow_min_corner_zyx = flow_corners_zyx.amin(dim=0)
    flow_max_corner_zyx = flow_corners_zyx.amax(dim=0)

    def draw_boxes(canvas):
        def draw_box(min_corner_yx, max_corner_yx):
            min_corner_yx = torch.clip(min_corner_yx, torch.zeros_like(min_corner_yx), torch.tensor(canvas.shape[:2], device=min_corner_yx.device) - 1)
            max_corner_yx = torch.clip(max_corner_yx, torch.zeros_like(max_corner_yx), torch.tensor(canvas.shape[:2], device=max_corner_yx.device) - 1)
            canvas[min_corner_yx[0] : max_corner_yx[0], min_corner_yx[1]: min_corner_yx[1] + 1] = 150
            canvas[min_corner_yx[0] : max_corner_yx[0], max_corner_yx[1]: max_corner_yx[1] + 1] = 150
            canvas[min_corner_yx[0] : min_corner_yx[0] + 1, min_corner_yx[1]: max_corner_yx[1]] = 150
            canvas[max_corner_yx[0] : max_corner_yx[0] + 1, min_corner_yx[1]: max_corner_yx[1]] = 150
        draw_box(flow_min_corner_zyx[1:], flow_max_corner_zyx[1:])

    def overlay_on_predictions(spiral, slice, mask, name):
        spiral_colour = torch.tensor([200, 20, 20], device=slice.device)
        canvas = slice.to(torch.float32) * (1 - spiral[..., None]) + spiral_colour * spiral[..., None]
        canvas = canvas.to(torch.uint8).cpu().numpy()
        Image.fromarray(canvas).save(f'{out_path}/spiral_on_{name}_{suffix}.png', compress_level=3)

    def overlay_on_patch_satisfaction(spiral, spiral_zyx, label_map, slice_z, name):
        # quad_status_flat: 0=patch-and-quad not satisfied, 1=quad-only satisfied, 2=patch overall satisfied, 3=not in working set
        status = quad_status_flat.to(device=spiral.device, dtype=torch.int64)
        label_map = label_map.to(device=spiral.device, dtype=torch.int64)
        num_labels = status.numel()
        status_palette = torch.tensor([
            [200, 0, 0],  # 0: red — quad not satisfied (patch not overall satisfied)
            [230, 140, 0],  # 1: orange — quad satisfied but patch not overall satisfied
            [255, 200, 0],  # 2: yellow — patch overall satisfied
            [160, 160, 160],  # 3: gray — patch not in working set
        ], dtype=torch.uint8, device=spiral.device)
        colour_table = torch.zeros([num_labels + 1, 3], dtype=torch.uint8, device=spiral.device)
        colour_table[1:] = status_palette[status]
        # Per-winding spiral colour: cycle through `num_winding_hues` hues so each winding gets a constant colour
        dr_per_winding = spiral_and_transform.get_dr_per_winding()
        _, _, shifted_radius = get_theta_and_radii(spiral_zyx[..., 1:], dr_per_winding)
        winding_idx = (shifted_radius / dr_per_winding).round().to(torch.int64).clamp_min(0)
        num_winding_hues = 6
        # Cycle hues over [yellow, pink] to avoid satisfaction colours
        hue_min, hue_max = 1.5 / 6, 5.25 / 6
        hue_fraction = hue_min + (winding_idx % num_winding_hues).to(torch.float32) / num_winding_hues * (hue_max - hue_min)
        hue = hue_fraction * 2 * np.pi
        hsv = torch.stack([hue, torch.full_like(hue, 0.5), torch.ones_like(hue)])
        spiral_colours = kornia.color.hsv_to_rgb(hsv).permute(1, 2, 0) * 255
        canvas = spiral_colours * spiral[..., None]
        patch_mask = (label_map > 0)[..., None]
        canvas = torch.where(patch_mask, colour_table[label_map].to(torch.float32), canvas)
        canvas = canvas.to(torch.uint8).cpu().numpy()
        image = Image.fromarray(canvas)
        if unattached_pcl_strips:
            draw = ImageDraw.Draw(image)
            pcl_z_window = 4
            point_radius = 1
            pcl_palette = [(200, 0, 0), (230, 140, 0), (255, 200, 0)]  # matches status_palette[0:3]
            for k, strip in enumerate(unattached_pcl_strips):
                zyxs = strip['zyxs']
                in_slab = np.abs(zyxs[:, 0] - float(slice_z)) <= pcl_z_window
                if not in_slab.any():
                    continue
                fully = bool(unattached_pcl_fully_satisfied[k].item())
                per_point_sat = unattached_pcl_per_point_satisfied[k].numpy()
                for idx in np.nonzero(in_slab)[0]:
                    y = float(zyxs[idx, 1])
                    x = float(zyxs[idx, 2])
                    if fully:
                        colour = pcl_palette[2]
                    elif per_point_sat[idx]:
                        colour = pcl_palette[1]
                    else:
                        colour = pcl_palette[0]
                    draw.ellipse(
                        [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                        fill=colour,
                    )
        image.save(f'{out_path}/spiral_on_{name}_{suffix}.png', compress_level=3)

    def overlay_on_scroll(slice_zyx, spiral_zyx, spiral_density, slice, name):
        slice_min = slice[slice > 0].amin() if (slice > 0).any() else 0
        slice_normalised = (slice - slice_min) / (slice.amax() - slice_min) * (slice > 0)
        spiral_density_normalised = spiral_density / spiral_density.amax()
        theta, relative_yx = get_theta(spiral_zyx[..., 1:])
        # 1. Coloured spiral overlaid on scroll
        theta_colours = kornia.color.hsv_to_rgb(torch.stack([theta, *[torch.ones_like(theta)] * 2])).permute(1, 2, 0) * 0.5
        spiral_density_coloured = spiral_density_normalised[..., None].expand(-1, -1, 3) * theta_colours
        canvas = slice_normalised[..., None].expand(-1, -1, 3) * (1. - spiral_density_normalised[..., None]) + spiral_density_coloured
        canvas *= (slice > 0)[..., None]
        canvas = (canvas * 255).to(torch.uint8)
        draw_boxes(canvas)
        canvas = Image.fromarray(canvas.cpu().numpy())
        draw = ImageDraw.Draw(canvas)
        _, yxs_by_radial, winding_indices_by_radial = get_winding_positions_on_radials(
            slice_z=slice_zyx[0, 0, :1],
            thetas=torch.arange(torch.pi / 8, 2 * torch.pi, torch.pi / 4),
            max_radius=slice_zyx[..., 1:].amax(),
            slice_to_spiral_transform=slice_to_spiral_transform,
            dr_per_winding=spiral_and_transform.get_dr_per_winding(),
            z_to_umbilicus_yx=z_to_umbilicus_yx,
        )
        for radial_idx in range(len(yxs_by_radial)):
            for idx in range(winding_indices_by_radial[radial_idx].shape[0]):
                marker_yx = yxs_by_radial[radial_idx][idx]
                if (marker_yx > 0).all() and (marker_yx < torch.tensor(slice.shape)).all() and slice[*marker_yx.to(torch.int64)] > 0:
                    winding_idx = int(winding_indices_by_radial[radial_idx][idx].item())
                    if winding_idx > 0 and winding_idx % 5 == 0:
                        draw.point(tuple(marker_yx)[::-1])
                        draw.text(
                            tuple(marker_yx)[::-1],
                            str(winding_idx)
                        )
        canvas.save(f'{out_path}/spiral_on_{name}_{suffix}.png', compress_level=3)

    for vis_slice_idx, slice_z in enumerate(tqdm(zs_for_visualisation, desc='visualising slices')):
        slice_zyx = torch.cat([torch.full([*slice_yx.shape[:2], 1], slice_z, device=device), slice_yx], dim=-1)

        spiral_zyx = slice_to_spiral_transform(slice_zyx)
        spiral_density = spiral_and_transform.get_spiral_density(spiral_zyx)
        slice = scroll_slices_for_visualisation[vis_slice_idx].to(device)
        # overlay_on_scroll(slice_zyx, spiral_zyx, spiral_density, slice, f'scroll_s{slice_z * downsample_factor:05}')
        # overlay_on_predictions(spiral_density, prediction_slices_for_visualisation[vis_slice_idx].to(device), slice > 0., f'pred_s{slice_z * downsample_factor:05}')
        overlay_on_patch_satisfaction(spiral_density, spiral_zyx, quad_label_map[vis_slice_idx], slice_z, f'patches_s{slice_z * downsample_factor:05}')


def fit_spiral_3d(scroll_zarr, patches_dict, point_collections, unattached_pcl_strips, z_to_umbilicus_yx, out_path, pass_seed_patch_id, pass_idx=1):
    patches_list = list(patches_dict.values())
    patch_ids = list(patches_dict.keys())
    patch_sampling_probabilities = _prepare_patch_sampling_cache(patches_list, cfg['patch_loss_z_margin'])

    num_patches = len(patches_list)
    is_global_mode = cfg['working_set_mode'] == 'global'
    is_progressive_fixed_mode = cfg['working_set_mode'] == 'progressive_fixed'

    if is_global_mode:
        progressive_patch_order = list(range(num_patches))
        print(f'pass {pass_idx}: global mode — fitting all {num_patches} patches at once')
    else:
        if pass_seed_patch_id not in patches_dict:
            raise KeyError(f'pass_seed_patch_id {pass_seed_patch_id!r} not found among loaded patches')
        seed_patch_idx = patch_ids.index(pass_seed_patch_id)
        mode_label = 'progressive_fixed' if is_progressive_fixed_mode else 'progressive'
        print(f'pass {pass_idx}: building {mode_label} patch ordering from seed {pass_seed_patch_id} ({num_patches} patches in pool)')
        progressive_patch_order = _build_progressive_patch_order(patches_list, seed_patch_idx)

    num_slices_for_visualisation = 20
    rendering_slices_downsample_factor = 2  # stride the scroll by this along zyx for rendering

    device = torch.device('cuda')

    all_zs = np.arange(z_begin, z_end)
    zs_for_visualisation = np.linspace(z_begin, z_end - 1, min(num_slices_for_visualisation, z_end - 1 - z_begin), dtype=np.int64)

    umbilicus_zyx = torch.from_numpy(np.concatenate([all_zs[:, None], z_to_umbilicus_yx(all_zs)], axis=-1).astype(np.float32)).to(device)

    all_zs = torch.from_numpy(all_zs).to(device)

    if scroll_zarr is not None:
        subvolume_shape = tuple([z_end - z_begin, *scroll_zarr.shape[1:]])
        print('loading slices for visualisation')
        scroll_slices_for_visualisation = (torch.from_numpy(scroll_zarr[zs_for_visualisation]).to(torch.float32) / np.iinfo(scroll_zarr.dtype).max * 0.75 * 255).to(torch.uint8)
        scroll_slices_for_rendering = (torch.from_numpy(scroll_zarr[z_begin : z_end : rendering_slices_downsample_factor, ::rendering_slices_downsample_factor, ::rendering_slices_downsample_factor]).to(torch.int32) // (np.iinfo(scroll_zarr.dtype).max // 255)).to(torch.uint8)
    else:
        subvolume_shape = [z_end - z_begin, 32693 // 4 // downsample_factor, 32693 // 4 // downsample_factor]
        scroll_slices_for_visualisation = torch.zeros([len(zs_for_visualisation), *subvolume_shape[1:]])
        scroll_slices_for_rendering = None

    prediction_slices_for_visualisation, quad_label_map, _quad_offsets = overlay_patches_on_slices(patches_list, zs_for_visualisation, subvolume_shape[1:])

    slice_yx = torch.stack(torch.meshgrid(
        torch.arange(subvolume_shape[1], dtype=torch.float32),
        torch.arange(subvolume_shape[2], dtype=torch.float32),
        indexing='ij'
    ), axis=-1).to(device)

    if scroll_zarr is not None:
        # TODO: it's a bit nasty to use the *visualisation* slices here
        z_to_roi_min_max_yx = {
            z: torch.tensor([
                [slice.amax(dim=1).nonzero().amin(), slice.amax(dim=0).nonzero().amin()],
                [slice.amax(dim=1).nonzero().amax(), slice.amax(dim=0).nonzero().amax()]
            ])
            for z, slice in zip(zs_for_visualisation, scroll_slices_for_visualisation)
        }
    else:
        z_to_roi_min_max_yx = {0: torch.tensor([[640, 1020], [1100, 1420]])}
    # FIXME: this assumes the scale in scroll-space is the same as that in spiral-space, which is only true at the start of optimisation
    #  The only way to deal with this cleanly is to revert to supporting the flow field on scroll-space (but then the flow field no 
    #  longer shifts with the umbilicus)
    flow_field_radius = torch.stack([min_max_yx for min_max_yx in z_to_roi_min_max_yx.values()]).diff(dim=1).amax().item() / 2
    flow_min_corner_spiral_zyx = torch.tensor([z_begin - cfg['flow_bounds_z_margin'], -flow_field_radius, -flow_field_radius], dtype=torch.int64, device=device)
    flow_max_corner_spiral_zyx = torch.tensor([z_end + cfg['flow_bounds_z_margin'], flow_field_radius, flow_field_radius], dtype=torch.int64, device=device)

    num_training_steps = cfg['num_training_steps']

    spiral_and_transform = SpiralAndTransform(flow_integration_steps=cfg['num_flow_integration_steps'], flow_integration_solver=cfg['flow_integration_solver'], umbilicus_zyx=umbilicus_zyx, flow_min_corner_zyx=flow_min_corner_spiral_zyx, flow_max_corner_zyx=flow_max_corner_spiral_zyx)
    spiral_and_transform.to(device)

    optimiser = torch.optim.Adam(spiral_and_transform.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1.e-8, fused=True)
    if cfg['exp_lr_schedule']:
        gamma = cfg['lr_final_factor'] ** (1.0 / max(1, num_training_steps))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: 1.)

    def save_model(suffix):
        torch.save([spiral_and_transform.state_dict(), optimiser.state_dict()], f'{out_path}/checkpoint_{suffix}.ckpt')

    def load_model(path):
        transformed_spiral_state, optimiser_state = torch.load(path, map_location='cpu')
        spiral_and_transform.load_state_dict(transformed_spiral_state)
        optimiser.load_state_dict(optimiser_state)

    resume_path = os.environ.get('FIT_SPIRAL_RESUME_PATH')
    start_iteration = int(os.environ.get('FIT_SPIRAL_RESUME_STEP', '0'))
    if resume_path:
        print(f'resuming from {resume_path} at iteration {start_iteration}')
        load_model(resume_path)
        for _ in range(start_iteration):
            lr_scheduler.step()

    if False:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=2, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(f'{out_path}/profile.out'),
            record_shapes=True,
            with_stack=True,
        )
        profiler.start()
    else:
        profiler = None

    def save_overlay_and_print_satisfaction(suffix):
        satisfied_patches, satisfied_areas, total_areas, satisfied_quad_masks, boundary_satisfied_patches, target_winding_idx_per_patch = get_patch_satisfied_areas(slice_to_spiral_transform, dr_per_winding, patches_list, verbose=True)
        satisfied_count = satisfied_patches.sum().item()
        boundary_satisfied_count = boundary_satisfied_patches.sum().item()
        total_count = satisfied_patches.numel()
        satisfied_ratio = satisfied_count / max(total_count, 1)
        print(f'satisfied_patches = {satisfied_count}/{total_count} ({satisfied_ratio * 100:.1f}%)')
        boundary_satisfied_ratio = boundary_satisfied_count / max(total_count, 1)
        print(f'boundary_satisfied_patches = {boundary_satisfied_count}/{total_count} ({boundary_satisfied_ratio * 100:.1f}%)')
        satisfied_area = float(satisfied_areas.sum().item())
        total_area = float(total_areas.sum().item())
        satisfied_area_ratio = satisfied_area / max(total_area, 1e-9)
        print(f'satisfied_area = {satisfied_area:.1f}/{total_area:.1f} ({satisfied_area_ratio * 100:.1f}%)')
        unattached_pcl_per_point_satisfied = []
        unattached_pcl_fully_satisfied = torch.zeros(len(unattached_pcl_strips), dtype=torch.bool)
        if unattached_pcl_strips:
            unattached_pcl_satisfied_counts, unattached_pcl_total_counts, unattached_pcl_per_point_satisfied = get_unattached_pcl_satisfied_counts(slice_to_spiral_transform, dr_per_winding, unattached_pcl_strips)
            unattached_pcl_fully_satisfied = (unattached_pcl_satisfied_counts == unattached_pcl_total_counts)
            fully_satisfied_pcls = int(unattached_pcl_fully_satisfied.sum().item())
            num_pcls = len(unattached_pcl_strips)
            fully_satisfied_ratio = fully_satisfied_pcls / max(num_pcls, 1)
            print(f'satisfied_unattached_pcls = {fully_satisfied_pcls}/{num_pcls} ({fully_satisfied_ratio * 100:.1f}%)')
            satisfied_points = int(unattached_pcl_satisfied_counts.sum().item())
            total_points = int(unattached_pcl_total_counts.sum().item())
            satisfied_point_ratio = satisfied_points / max(total_points, 1)
            print(f'satisfied_unattached_pcl_points = {satisfied_points}/{total_points} ({satisfied_point_ratio * 100:.1f}%)')
        # Flatten per-patch (H-1, W-1) masks in patch order to match the rasteriser's quad-id offsets,
        # then combine with patch-level overall satisfaction into a 0/1/2 status per quad.
        if satisfied_quad_masks:
            satisfied_quads_flat = torch.cat([m.flatten() for m in satisfied_quad_masks])
            quads_per_patch = torch.tensor([m.numel() for m in satisfied_quad_masks], dtype=torch.int64)
            overall_satisfied_per_quad = satisfied_patches.to(torch.bool).repeat_interleave(quads_per_patch)
            in_working_set_per_patch = torch.zeros(len(patches_list), dtype=torch.bool)
            in_working_set_per_patch[list(working_set_indices)] = True
            in_working_set_per_quad = in_working_set_per_patch.repeat_interleave(quads_per_patch)
        else:
            satisfied_quads_flat = torch.zeros([0], dtype=torch.bool)
            overall_satisfied_per_quad = torch.zeros([0], dtype=torch.bool)
            in_working_set_per_quad = torch.zeros([0], dtype=torch.bool)
        quad_status_flat = torch.where(
            ~in_working_set_per_quad,
            torch.full_like(satisfied_quads_flat, 3, dtype=torch.int64),
            torch.where(
                overall_satisfied_per_quad,
                torch.full_like(satisfied_quads_flat, 2, dtype=torch.int64),
                satisfied_quads_flat.to(torch.int64),
            ),
        )
        if os.environ.get('FIT_SPIRAL_SKIP_FITTED_OVERLAY') != '1':
            save_overlay(
                spiral_and_transform,
                flow_min_corner_spiral_zyx, flow_max_corner_spiral_zyx,
                zs_for_visualisation, all_zs, slice_yx,
                scroll_slices_for_visualisation, prediction_slices_for_visualisation,
                quad_label_map, quad_status_flat,
                unattached_pcl_strips, unattached_pcl_per_point_satisfied, unattached_pcl_fully_satisfied,
                umbilicus_zyx, z_to_umbilicus_yx,
                out_path, suffix
            )
            working_set_patches_list = [patches_list[i] for i in working_set_indices]
            save_mesh(slice_to_spiral_transform, dr_per_winding, working_set_patches_list, out_path, name=suffix)

    slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()
    dr_per_winding = spiral_and_transform.get_dr_per_winding()

    working_set_size = num_patches if is_global_mode else 1
    working_set_indices = progressive_patch_order[:working_set_size]
    working_set_probabilities = _get_working_set_probabilities(patch_sampling_probabilities, working_set_indices)
    working_set_patches_dict = {patch_ids[i]: patches_list[i] for i in working_set_indices}

    working_set_log = open(f'{out_path}/working-set.txt', 'a', buffering=1)
    if is_global_mode:
        print(f'step {start_iteration}: working set initialised with all {num_patches} patches (global mode)')
        working_set_log.write(f'=== pass {pass_idx}: {num_patches} candidate patches, global mode ===\n')
        working_set_log.write(f'step {start_iteration}: initialised with all {num_patches} patches\n')
    else:
        mode_label = 'progressive_fixed' if is_progressive_fixed_mode else 'progressive'
        print(f'step {start_iteration}: working set initialised with {working_set_size}/{num_patches} patches (seed {patch_ids[working_set_indices[0]]}, {mode_label} mode)')
        working_set_log.write(f'=== pass {pass_idx}: {num_patches} candidate patches, seed {pass_seed_patch_id}, {mode_label} mode ===\n')
        working_set_log.write(f'step {start_iteration}: initialised with seed patch {patch_ids[working_set_indices[0]]} (size {working_set_size}/{num_patches})\n')

    # Snapshot of the most recent state at which the working set was fully satisfied —
    # i.e. the model state taken just before each successful expansion. If the pass
    # ends without absorbing every candidate, the optimisation phase since the final
    # expansion is an unsuccessful attempt that may have degraded earlier windings,
    # so save_outputs() can revert to this snapshot.
    last_known_good_state = None
    last_known_good_ws_size = None
    last_known_good_iteration = None

    for iteration in tqdm(range(start_iteration, num_training_steps)):

        slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()
        dr_per_winding = spiral_and_transform.get_dr_per_winding()

        if is_progressive_fixed_mode:
            should_grow = (
                working_set_size < num_patches
                and iteration > start_iteration
                and iteration % cfg['progressive_fixed_add_interval'] == 0
            )
            if should_grow:
                last_known_good_state = (
                    copy.deepcopy(spiral_and_transform.state_dict()),
                    copy.deepcopy(optimiser.state_dict()),
                )
                last_known_good_ws_size = working_set_size
                last_known_good_iteration = iteration
                working_set_size += 1
                working_set_indices = progressive_patch_order[:working_set_size]
                working_set_probabilities = _get_working_set_probabilities(patch_sampling_probabilities, working_set_indices)
                working_set_patches_dict = {patch_ids[i]: patches_list[i] for i in working_set_indices}
                print(f'step {iteration}: working set grew to {working_set_size}/{num_patches} patches (added {patch_ids[working_set_indices[-1]]}, fixed schedule)')
                working_set_log.write(f'step {iteration}: added patch {patch_ids[working_set_indices[-1]]} (size {working_set_size}/{num_patches}, fixed schedule)\n')
        elif (
            working_set_size < num_patches
            and iteration > start_iteration
            and iteration % cfg['working_set_check_interval'] == 0
        ):
            with torch.no_grad():
                working_set_patches_list = [patches_list[i] for i in working_set_indices]
                working_set_satisfied, *_ = get_patch_satisfied_areas(slice_to_spiral_transform, dr_per_winding, working_set_patches_list)
            if bool(working_set_satisfied.all().item()):
                last_known_good_state = (
                    copy.deepcopy(spiral_and_transform.state_dict()),
                    copy.deepcopy(optimiser.state_dict()),
                )
                last_known_good_ws_size = working_set_size
                last_known_good_iteration = iteration
                working_set_size += 1
                working_set_indices = progressive_patch_order[:working_set_size]
                working_set_probabilities = _get_working_set_probabilities(patch_sampling_probabilities, working_set_indices)
                working_set_patches_dict = {patch_ids[i]: patches_list[i] for i in working_set_indices}
                print(f'step {iteration}: working set grew to {working_set_size}/{num_patches} patches (added {patch_ids[working_set_indices[-1]]})')
                working_set_log.write(f'step {iteration}: added patch {patch_ids[working_set_indices[-1]]} (size {working_set_size}/{num_patches})\n')

        losses = {}

        # TODO
        # patches may be >1 winding; thus the track radius loss needs fixing to not 'force apart' windings
        # for pcl's, edges link patches, and relative w-n should be applied everywhere over the patch
        # ...but for multi-winding patches this is not directly true, again need to adjust for winding changes
        # also note the existing track loss does not correctly handle >1 winding tracks crossing theta=0
        # for single-winding patches, and local regions of larger ones, the relative numbering *should* hold everywhere
        #  on the patch -- but need to account properly for theta=0

        compute_patch_dt = iteration > cfg['loss_start_patch_dt']
        patch_radius_loss, umbilicus_loss, patch_dt_loss = get_patch_and_umbilicus_losses(
            slice_to_spiral_transform,
            dr_per_winding,
            cfg['num_patches_per_step'],
            cfg['num_patches_per_step_for_dt'],
            patches_list,
            working_set_probabilities,
            umbilicus_zyx,
            compute_dt=compute_patch_dt,
        )
        losses['patch_radius'] = patch_radius_loss * cfg['loss_weight_patch_radius']
        losses['patch_dt'] = patch_dt_loss * cfg['loss_weight_patch_dt']

        if cfg['loss_weight_patch_stretch'] > 0 or cfg['loss_weight_patch_normals'] > 0:
            patch_stretch_loss, patch_normals_loss = get_patch_stretch_and_normals_loss(
                slice_to_spiral_transform,
                cfg['regularisation_num_points'],
                patches_list,
                working_set_probabilities,
            )
            losses['patch_stretch'] = patch_stretch_loss * cfg['loss_weight_patch_stretch']
            losses['patch_normals'] = patch_normals_loss * cfg['loss_weight_patch_normals']

        if cfg['loss_weight_winding_number'] > 0 and point_collections:
            losses['winding_number'] = get_patch_winding_number_loss(slice_to_spiral_transform, dr_per_winding, working_set_patches_dict, point_collections) * cfg['loss_weight_winding_number']

        if (cfg['loss_weight_unattached_pcl_radius'] > 0 or cfg['loss_weight_unattached_pcl_dt'] > 0) and unattached_pcl_strips:
            unattached_pcl_radius_loss, unattached_pcl_dt_loss = get_unattached_pcl_strip_losses(
                slice_to_spiral_transform,
                dr_per_winding,
                unattached_pcl_strips,
                cfg['unattached_pcl_num_per_step'],
                cfg['unattached_pcl_num_points_per_step'],
                compute_dt=compute_patch_dt,
            )
            losses['unattached_pcl_radius'] = unattached_pcl_radius_loss * cfg['loss_weight_unattached_pcl_radius']
            losses['unattached_pcl_dt'] = unattached_pcl_dt_loss * cfg['loss_weight_unattached_pcl_dt']

        losses['umbilicus'] = umbilicus_loss * cfg['loss_weight_umbilicus']

        loss = sum(losses.values())

        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        lr_scheduler.step()
        if profiler is not None:
            profiler.step()

        if iteration % 200 == 0:
            # Only sync to CPU and log when we actually print, avoiding a per-iter
            # GPU->CPU sync that would otherwise stall CPU/GPU overlap.
            print(f'step {iteration}: loss = {loss.item():.1f}, ' + ', '.join(f'{name} = {value.item():.1f}' for name, value in losses.items()))
            if loss.isnan().item():
                print('aborting due to NaN')
                return
            wandb.log({
                'total_loss': loss.item(),
                **{name + '_loss': value for name, value in losses.items()},
            })

    working_set_log.write(f'pass {pass_idx} ended: final working set size {working_set_size}/{num_patches}\n')
    working_set_log.close()

    working_set_patch_ids = [patch_ids[i] for i in working_set_indices]
    last_added_patch_id = working_set_patch_ids[-1]
    pool_fully_absorbed = (working_set_size == num_patches)

    if not pool_fully_absorbed and last_known_good_state is not None:
        print(f'reverting model state to last-known-good (step {last_known_good_iteration}, working set size {last_known_good_ws_size}) for saving outputs')
        model_state, opt_state = last_known_good_state
        spiral_and_transform.load_state_dict(model_state)
        optimiser.load_state_dict(opt_state)
        with torch.no_grad():
            slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()
            dr_per_winding = spiral_and_transform.get_dr_per_winding()

    suffix = 'fitted' if pool_fully_absorbed else f'fitted_p{pass_idx}_ws{working_set_size}'
    save_overlay_and_print_satisfaction(suffix)
    save_model(suffix)

    return {
        'working_set_size': working_set_size,
        'working_set_patch_ids': working_set_patch_ids,
        'last_added_patch_id': last_added_patch_id,
        'num_patches': num_patches,
        'pool_fully_absorbed': pool_fully_absorbed,
        'progressive_patch_order_ids': [patch_ids[i] for i in progressive_patch_order],
    }


def load_patches(patches_path, segment_id_filter=lambda s: True):
    patches = {}
    failed_count = 0
    for entry in sorted(os.listdir(patches_path)):
        segment_path = os.path.join(patches_path, entry)
        meta_path = os.path.join(segment_path, 'meta.json')
        if not os.path.isdir(segment_path) or not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            print(f'Warning: failed to read {meta_path}: {e}')
            continue
        if meta.get('format') != 'tifxyz' or not segment_id_filter(entry):
            continue
        try:
            patches[entry] = load_tifxyz(segment_path)
        except Exception as e:
            print(f'Failed to load segment {entry}: {e}')
            failed_count += 1
    if not patches:
        raise RuntimeError('No patches could be loaded')
    print(f'Loaded {len(patches)} patches, {failed_count} failed')
    return patches


def process_point_collections_cached(patches, point_collections, tolerance=10.0):
    patch_ids_str = '_'.join(sorted(patches.keys()))
    hashed_patches = hashlib.sha256(bytes(patch_ids_str, 'ascii')).hexdigest()[:8]

    pcl_locations = []
    for collection_id in sorted(point_collections.keys()):
        collection = point_collections[collection_id]
        for point_id in sorted(collection['points'].keys()):
            point = collection['points'][point_id]
            pcl_locations.append(tuple(point['p']))
    hashed_pcls = hashlib.sha256(pickle.dumps(pcl_locations)).hexdigest()[:8]

    cache_filename = f'{cache_path}/point_patch_links-{hashed_patches}_{hashed_pcls}_tol-{tolerance}.pkl'

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as fp:
            point_collections_cached = pickle.load(fp)
        point_collections.clear()
        point_collections.update(point_collections_cached)
    else:
        _process_point_collections(patches, point_collections, tolerance)
        os.makedirs(cache_path, exist_ok=True)
        with open(cache_filename, 'wb') as fp:
            pickle.dump(point_collections, fp)


def main():

    np.random.seed(cfg['random_seed'])
    torch.random.manual_seed(cfg['random_seed'])

    umbilicus = umbilicus_z_to_yx(downsample_factor)

    if scroll_zarr_path:
        print('loading volume zarr')
        scroll_zarr_array = zarr.open(scroll_zarr_path, mode='r')
    else:
        scroll_zarr_array = None

    patches = load_patches(patches_path, segment_id_filter=lambda s: 'monster' not in s)

    cross_patch_point_collections = {}
    unattached_point_collections = {}
    next_id = 0
    for target, paths in (
        (cross_patch_point_collections, cross_patch_pcl_json_paths),
        (unattached_point_collections, unattached_pcl_json_paths),
    ):
        for path in paths:
            loaded = load_point_collection(path) or {}
            for pcl in loaded.values():
                pcl['source_file'] = path
                target[next_id] = pcl
                next_id += 1

    process_point_collections_cached(patches, cross_patch_point_collections)

    for patch in patches.values():
        patch.scale *= downsample_factor
        patch.zyxs /= downsample_factor
        patch.valid_zyxs /= downsample_factor
        patch.area /= downsample_factor ** 2
    for pcl_dict in (cross_patch_point_collections, unattached_point_collections):
        for pcl in pcl_dict.values():
            for point in pcl['points'].values():
                point['zyx'] = np.array([point['p'][2], point['p'][1], point['p'][0]]) / downsample_factor

    def patch_intersects_z_roi(patch):
        zs = patch.valid_zyxs[..., 0]
        if zs.numel() == 0:
            return False
        return bool(((zs >= z_begin) & (zs < z_end)).any().item())

    def pcl_intersects_z_roi(pcl):
        for point in pcl['points'].values():
            z = point['zyx'][0]
            if z_begin <= z < z_end:
                return True
        return False

    dropped_patch_ids = [pid for pid, patch in patches.items() if not patch_intersects_z_roi(patch)]
    for pid in dropped_patch_ids:
        del patches[pid]

    # For unattached pcls, keep only the longest contiguous subrange (in id-sorted
    # order) of points whose zs lie within [z_begin - margin, z_end + margin); drop
    # the pcl entirely if fewer than 2 points remain.
    z_margin = cfg['patch_loss_z_margin']
    dropped_unattached_pcl_count = 0
    for pid in list(unattached_point_collections.keys()):
        pcl = unattached_point_collections[pid]
        sorted_items = sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
        best_start, best_end = 0, 0
        run_start = 0
        for i, (_, point) in enumerate(sorted_items):
            z = point['zyx'][0]
            if z_begin - z_margin <= z < z_end + z_margin:
                if i + 1 - run_start > best_end - best_start:
                    best_start, best_end = run_start, i + 1
            else:
                run_start = i + 1
        kept_items = sorted_items[best_start:best_end]
        if len(kept_items) < 2:
            del unattached_point_collections[pid]
            dropped_unattached_pcl_count += 1
        else:
            pcl['points'] = dict(kept_items)

    dropped_cross_patch_pcl_ids = [pid for pid, pcl in cross_patch_point_collections.items() if not pcl_intersects_z_roi(pcl)]
    for pid in dropped_cross_patch_pcl_ids:
        del cross_patch_point_collections[pid]
    print(f'dropped {len(dropped_patch_ids)} patches, {len(dropped_cross_patch_pcl_ids)} cross-patch pcls, and {dropped_unattached_pcl_count} unattached pcls outside z-roi [{z_begin}, {z_end})')

    normalise_pcl_winding_annotations(cross_patch_point_collections)
    normalise_pcl_winding_annotations(unattached_point_collections)
    # Group each cross-patch pcl's attached points by patch, for the
    # winding-number loss. Patches are ordered by the first attached point that
    # hits them when scanning the pcl's points in int(json-key) order; within
    # each patch, points are also in int(key) order.
    for pcl in cross_patch_point_collections.values():
        points_by_patch = {}
        for _, point in sorted(pcl['points'].items(), key=lambda kv: int(kv[0])):
            if 'on_patch' not in point:
                continue
            pid = point['on_patch']['id']
            if pid not in patches:
                continue
            points_by_patch.setdefault(pid, []).append(point)
        pcl['points_by_patch'] = points_by_patch
    unattached_pcl_strips = _prepare_unattached_pcl_strips(unattached_point_collections)
    print(
        f'pcls: {len(cross_patch_point_collections)} cross-patch, '
        f'{len(unattached_pcl_strips)} unattached'
    )

    out_base_dir = os.environ.get('FIT_SPIRAL_OUT_DIR', './out')
    out_path = f'{out_base_dir}/{datetime.date.today()}_{scroll_name}_slice-{z_begin * downsample_factor}-{z_end * downsample_factor}_{len(patches)}-patch_{cfg["working_set_mode"]}'
    if not wandb.run.name.startswith('dummy-'):
        out_path += '_' + wandb.run.name
    run_tag = os.environ.get('FIT_SPIRAL_RUN_TAG')
    if run_tag:
        out_path += f'_{run_tag}'
    os.makedirs(out_path, exist_ok=True)

    if cfg['working_set_mode'] == 'global':
        fit_spiral_3d(
            scroll_zarr_array,
            patches,
            list(cross_patch_point_collections.values()),
            unattached_pcl_strips,
            umbilicus,
            out_path,
            pass_seed_patch_id=None,
            pass_idx=1,
        )
        return

    if cfg['working_set_mode'] == 'progressive_fixed':
        fit_spiral_3d(
            scroll_zarr_array,
            patches,
            list(cross_patch_point_collections.values()),
            unattached_pcl_strips,
            umbilicus,
            out_path,
            pass_seed_patch_id=seed_patch_id,
            pass_idx=1,
        )
        return

    pass_pool_patches = dict(patches)
    current_seed_patch_id = seed_patch_id
    pass_idx = 0
    while True:
        pass_idx += 1
        result = fit_spiral_3d(
            scroll_zarr_array,
            pass_pool_patches,
            list(cross_patch_point_collections.values()),
            unattached_pcl_strips,
            umbilicus,
            out_path,
            pass_seed_patch_id=current_seed_patch_id,
            pass_idx=pass_idx,
        )

        ws_size = result['working_set_size']
        ws_ids = set(result['working_set_patch_ids'])
        last_added_patch_id = result['last_added_patch_id']
        progressive_order_ids = result['progressive_patch_order_ids']
        leftover_ids = [pid for pid in pass_pool_patches if pid not in ws_ids]
        pool_fully_absorbed = result['pool_fully_absorbed']
        no_progress = (ws_size <= 1)

        del result
        torch.cuda.empty_cache()

        if pool_fully_absorbed:
            print(f'all {len(patches)} patches covered after {pass_idx} pass(es)')
            break

        if no_progress:
            # The seed couldn't bootstrap any growth even with a fresh model — drop it
            # permanently and retry with the next-nearest candidate from this pass's
            # progressive ordering (= the leftover patch nearest to the failed seed).
            dropped_seed = current_seed_patch_id
            next_seed = next((pid for pid in progressive_order_ids[1:] if pid != dropped_seed), None)
            if next_seed is None:
                print(f'pass {pass_idx} made no progress and no other candidate seed remains; stopping with {len(leftover_ids)} patches unfit')
                break
            next_pool_ids = [pid for pid in pass_pool_patches if pid != dropped_seed]
            pass_pool_patches = {pid: patches[pid] for pid in next_pool_ids}
            current_seed_patch_id = next_seed
            print(f'pass {pass_idx} made no progress; dropping seed {dropped_seed} and retrying with seed {current_seed_patch_id} ({len(next_pool_ids)} candidates remaining)')
            continue

        next_pool_ids = leftover_ids + [last_added_patch_id]
        pass_pool_patches = {pid: patches[pid] for pid in next_pool_ids}
        current_seed_patch_id = last_added_patch_id
        print(f'starting pass {pass_idx + 1}: {len(next_pool_ids)} candidate patches (seed {current_seed_patch_id})')


if __name__ == '__main__':
    config = dict(default_config)
    config.update(get_env_config_overrides())
    wandb.init(project='scrolls', config=config)
    cfg = wandb.config
    main()
