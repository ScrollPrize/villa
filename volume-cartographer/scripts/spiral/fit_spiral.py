
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
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
from tqdm import tqdm
import pyro.distributions
from einops import rearrange
from torchdiffeq import odeint

from lasagna_data import prepare_lasagna_volume
from tifxyz import load_tifxyz
from geom_utils import interp1d
from point_collection import load_point_collection, link_points_to_patches, link_points_to_patches_pointcache, can_use_surface_index_backend, normalise_pcl_winding_annotations
from tracks import (
    get_track_losses,
    get_track_satisfied_counts_in_chunks,
    load_tracks_from_dbm,
    prepare_main_phase_tracks,
    render_spiral_on_tracks_for_slice,
)
from umbilicus import thaumato_umbilicus_z_to_yx, json_umbilicus_z_to_yx
from sample_spiral import (
    canonical_winding_samples,
    get_bounding_windings,
    get_spiral_points,
    get_theta,
    get_theta_and_radii,
    get_winding_xy,
)
from spiral_helpers import (
    compute_winding_range_and_input_extents,
    erode_patch_valid_region,
    load_patches,
    save_mesh,
    scale_counts_for_z_range,
    _huber_abs,
    _infer_shell_outer_winding_idx,
    _warn_if_inputs_exceed_flow_bounds,
)
import sample_spiral
from satistfaction_metrics import (
    get_patch_satisfied_areas as _get_patch_satisfied_areas,
    get_unattached_pcl_satisfied_counts as _get_unattached_pcl_satisfied_counts,
    metrics_config,
)


# PHercParis4
dataset_path = '/ephemeral/paul/spiral/dataset'
scroll_zarr_path = None
normal_nx_zarr_path = f'{dataset_path}/normals/las_008_nx.ome.zarr'
normal_ny_zarr_path = f'{dataset_path}/normals/las_008_ny.ome.zarr'
grad_mag_zarr_path = '/ephemeral/paul/spiral/las_008_grad_mag.ome.zarr'
normal_zarr_group = '4'
pcl_json_paths = [
    f'{dataset_path}/rel_windings_new_reordered.json',
    f'{dataset_path}/new_same_wind.json',
]
patches_path = f'{dataset_path}/patches'
unverified_patches_path = f'{dataset_path}/unproofed_patches_trimmed_deduped'
run_tag = os.environ.get('FIT_SPIRAL_RUN_TAG')
shell_path = f'{dataset_path}/s1_2um_outer'
tracks_dbm_path = f'{dataset_path}/tracks/2um_ds2_ps256_surf_v2.dbm'
spiral_outward_sense = 'CW'  # CW | ACW
umbilicus_z_to_yx = lambda f: json_umbilicus_z_to_yx(f'{dataset_path}/umbilicus.json', downsample_factor=f)
scroll_name = 's1'
z_begin, z_end = 7000, 16500
voxel_size_um = 2.4 * 4  # before downsampling

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

cache_path = '../cache'
downsample_factor = 4
z_begin //= downsample_factor
z_end //= downsample_factor

default_config = {
    'random_seed': 1,
    'learning_rate': 3.e-5,
    'exp_lr_schedule': True,
    'lr_final_factor': 0.3,
    'num_training_steps': 30_000,
    'num_flow_integration_steps': 3,
    'flow_integration_solver': 'rk4',
    'num_flow_timesteps': 1,
    'flow_bounds_z_margin': 40,
    'flow_bounds_radius': 800,
    'flow_voxel_resolution': 4,
    'flow_field_type': 'cartesian',  # 'cartesian' or 'cylindrical'
    'flow_field_high_res_lr_scale_initial': 3.0e-1,
    'flow_field_high_res_lr_scale_final': 3.0e-1,
    'flow_field_high_res_lr_ramp_start_step': 0,
    'flow_field_high_res_lr_ramp_steps': 1,
    'gap_expander_logit_resolution': 6,
    'gap_expander_num_windings': 130,
    'gap_expander_lr_scale': 0.3,
    'linear_z_resolution': 12,
    'initial_dr_per_winding': 4.,
    'patch_radius_loss_margin': 0.025,
    'patch_radius_loss_inv': False,
    'patch_loss_z_margin': 0,
    'patch_dt_norm_p': 0.5,
    'patch_dt_within_patch_norm_p': 3.0,
    'patch_dt_loss_margin': 0.025,
    'patch_radius_within_norm_p': 3.0,  # >1 emphasises worst within-track points in the radius loss
    'num_patches_per_step': 360,
    'num_patches_per_step_for_dt': 240,
    'num_points_per_patch': 800,
    'erode_patches': 2,  # if >0, erode every patch's valid region (verified + unverified) by this many grid cells
    'unverified_patch_radius_loss_margin': 0.025,
    'unverified_patch_radius_loss_inv': False,
    'unverified_patch_radius_within_norm_p': 3.0,
    'unverified_patch_dt_norm_p': 0.5,
    'unverified_patch_dt_within_patch_norm_p': 3.0,
    'unverified_patch_dt_loss_margin': 0.025,
    'unverified_num_patches_per_step': 120,
    'unverified_num_patches_per_step_for_dt': 80,
    'unverified_num_points_per_patch': 800,
    'unverified_patch_exclusion_radius': 16.0,  # mask unverified-patch vertices within this of trusted geometry (downsampled voxels)
    'rel_winding_num_pcls': 48,
    'rel_winding_num_patch_pairs_per_pcl': 4,
    'rel_winding_adjacent_patches_only': True,
    'abs_winding_num_pcls': 48,
    'abs_winding_num_points_per_pcl': 4,
    'unattached_pcl_num_per_step': 84,
    'unattached_pcl_num_points_per_step': 32,
    'unattached_pcl_min_point_spacing': 4.,
    'track_num_per_step': 48000,
    'track_num_points_per_step': 24,
    'track_exclusion_radius': 0.0,
    'track_radius_target': 'mean',
    'track_radius_loss_margin': 0.025,
    'track_radius_within_norm_p': 6.0,  # >1 emphasises worst within-track point in the radius loss (1.0 = mean)
    'track_dt_within_track_norm_p': 3.0,  # within a track; -> inf strongly penalises isolated badly-aligned points
    'track_dt_norm_p': 0.5,  # across tracks; -> 0 prefers many fully-satisfied tracks (winner-take-all snapping)
    'track_dt_loss_margin': 0.025,
    'dense_normals_num_points': 60_000,
    'regularisation_num_points': 4500,
    'grad_mag_encode_scale': 1000.0,
    'grad_mag_factor': 0.25 * 4,  # 4 maps from 2um to 8um
    'spacing_integration_steps': 8,
    'loss_weight_patch_radius': 32.e0,
    'loss_weight_patch_dt': 16.e0,
    'loss_weight_unverified_patch_radius': 8.e0,
    'loss_weight_unverified_patch_dt': 4.e0,
    'loss_weight_rel_winding': 20.,
    'loss_weight_abs_winding': 20.,
    'loss_weight_unattached_pcl_radius': 8.e0,
    'loss_weight_unattached_pcl_dt': 16.e0,
    'loss_weight_track_radius': 200.,
    'loss_weight_track_dt': 40.,
    'loss_weight_sym_dirichlet': 10.0,
    'loss_weight_dense_normals': 1.e2,
    'loss_weight_dense_spacing': 8.,
    'loss_weight_umbilicus': 5.,
    'loss_weight_shell_outer': 4.0,
    'loss_weight_shell_patch_radius': 0.0,
    'weight_decay_gap_expander': 1.e-2,
    'weight_decay_flow_field': 0.0,
    'loss_start_patch_dt': 25_000,
    'loss_start_track_dt': 10_000,
    'loss_start_unverified_patch_dt': None,  # None => fall back to loss_start_patch_dt
    'dt_progressive_windings': False,  # gate the DT losses (patch, track, unattached-pcl) to grow outwards across windings
    'dt_progressive_inner_winding': 20,  # outer-winding cutoff when each DT loss first turns on
    'dt_progressive_steps': 50_000,  # steps to grow the cutoff from start_winding to shell_outer_winding_idx
    'dt_progressive_exponent': 1.0,  # warp on the time fraction; 1.0 = linear in winding, <1 = slower later (~0.5 ≈ constant area rate)
    'output_first_winding': 10,
    'output_winding_margin': 4,
    'output_step_size': 20,
    'shell_outer_winding_idx': 130,
    'shell_outer_winding_margin': 10,
    'shell_num_samples': 24576,
    'shell_num_theta_bins': 720,
    'shell_huber_delta': 4.0,
    'shell_table_smooth_sigma_z': 1.0,
    'shell_table_smooth_sigma_theta': 1.0,
    'shell_min_confidence': 0.25,
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


# The per-step object-sample counts above are tuned for the z 7000-16500 range
# (~9500 un-downsampled slices). For a smaller/larger z-range each loss term sees
# proportionally fewer/more objects, so scale_counts_for_z_range() scales these
# counts linearly with the number of slices (points-PER-object stays fixed).
reference_z_range_num_slices = 9500
z_range_scaled_count_keys = (
    'num_patches_per_step',
    'num_patches_per_step_for_dt',
    'unverified_num_patches_per_step',
    'unverified_num_patches_per_step_for_dt',
    'rel_winding_num_pcls',
    'abs_winding_num_pcls',
    'unattached_pcl_num_per_step',
    'track_num_per_step',
    'dense_normals_num_points',
    'regularisation_num_points',
    'shell_num_samples',
)



def get_spiral_density(relative_yx, dr_per_winding=10., sigma=3., winding_range=None):
    if winding_range is None:
        winding_range = (cfg['output_first_winding'], float('inf'))
    return sample_spiral.get_spiral_density(relative_yx, dr_per_winding=dr_per_winding, sigma=sigma, winding_range=winding_range)


def shell_losses_enabled():
    return (
        cfg['loss_weight_shell_outer'] > 0
        or cfg['loss_weight_shell_patch_radius'] > 0
    )



def _masked_mean(values, mask):
    mask_f = mask.to(values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp(min=1.)


class ShellPolarMap:

    def __init__(self, shell_patch, z_to_umbilicus_yx, z_min, z_max, num_theta_bins, device):
        self.z_min = int(z_min)
        self.z_max = int(z_max)
        self.num_theta_bins = int(num_theta_bins)
        self.device = device

        shell_zyxs = shell_patch.valid_zyxs.cpu().numpy().astype(np.float32, copy=False)
        in_z = (shell_zyxs[:, 0] >= self.z_min) & (shell_zyxs[:, 0] <= self.z_max)
        shell_zyxs = shell_zyxs[in_z]
        if len(shell_zyxs) == 0:
            raise RuntimeError(f'shell has no valid points in z range [{self.z_min}, {self.z_max}]')

        centres_yx = z_to_umbilicus_yx(shell_zyxs[:, 0]).astype(np.float32)
        rel_yx = shell_zyxs[:, 1:] - centres_yx
        theta = np.mod(np.arctan2(rel_yx[:, 0], rel_yx[:, 1]), 2 * np.pi)
        radius = np.linalg.norm(rel_yx, axis=-1)

        num_z = self.z_max - self.z_min + 1
        z_idx = np.rint(shell_zyxs[:, 0] - self.z_min).astype(np.int64).clip(0, num_z - 1)
        theta_idx = np.floor(theta / (2 * np.pi) * self.num_theta_bins).astype(np.int64) % self.num_theta_bins

        radius_sum = np.zeros([num_z, self.num_theta_bins], dtype=np.float64)
        counts = np.zeros([num_z, self.num_theta_bins], dtype=np.float64)
        np.add.at(radius_sum, (z_idx, theta_idx), radius)
        np.add.at(counts, (z_idx, theta_idx), 1.0)
        valid = counts > 0
        if not valid.any():
            raise RuntimeError('shell polar table has no occupied bins')

        radius_mean = np.zeros_like(radius_sum, dtype=np.float32)
        radius_mean[valid] = (radius_sum[valid] / counts[valid]).astype(np.float32)

        valid_ext = np.concatenate([valid, valid, valid], axis=1)
        radius_ext = np.concatenate([radius_mean, radius_mean, radius_mean], axis=1)
        nearest_indices = scipy.ndimage.distance_transform_edt(~valid_ext, return_distances=False, return_indices=True)
        filled_ext = radius_ext[nearest_indices[0], nearest_indices[1]]
        filled = filled_ext[:, self.num_theta_bins:2 * self.num_theta_bins]

        sigma = (cfg['shell_table_smooth_sigma_z'], cfg['shell_table_smooth_sigma_theta'])
        if sigma[0] > 0 or sigma[1] > 0:
            smooth_ext = np.concatenate([filled, filled, filled], axis=1)
            smooth_ext = scipy.ndimage.gaussian_filter(smooth_ext, sigma=sigma, mode=('nearest', 'wrap'))
            filled = smooth_ext[:, self.num_theta_bins:2 * self.num_theta_bins]

        confidence = scipy.ndimage.gaussian_filter(valid.astype(np.float32), sigma=sigma, mode=('nearest', 'wrap'))
        if confidence.max() > 0:
            confidence = confidence / confidence.max()

        radius_with_wrap = np.concatenate([filled, filled[:, :1]], axis=1).astype(np.float32)
        confidence_with_wrap = np.concatenate([confidence, confidence[:, :1]], axis=1).astype(np.float32)

        self.lookup_table = torch.from_numpy(
            np.stack([radius_with_wrap, confidence_with_wrap], axis=0)
        ).to(device=device)

        z_coords = np.arange(self.z_min, self.z_max + 1, dtype=np.float32)
        self.umbilicus_zyx = torch.from_numpy(
            np.concatenate([z_coords[:, None], z_to_umbilicus_yx(z_coords).astype(np.float32)], axis=-1)
        ).to(device=device)

        occupied = int(valid.sum())
        total = int(valid.size)
        print(
            f'shell polar table: {num_z} z bins x {self.num_theta_bins} theta bins, '
            f'{occupied}/{total} occupied ({occupied / max(total, 1) * 100:.1f}%)'
        )

    def lookup(self, scan_zyx):
        centre_yx = interp1d(scan_zyx[..., 0].contiguous(), self.umbilicus_zyx[:, :1], self.umbilicus_zyx[:, 1:])
        rel_yx = scan_zyx[..., 1:] - centre_yx
        theta, rel_yx = get_theta(rel_yx)
        radius = torch.linalg.norm(rel_yx, dim=-1)

        z_normalised = (scan_zyx[..., 0] - self.z_min) / (self.z_max - self.z_min) * 2 - 1
        theta_normalised = theta / (2 * torch.pi) * 2 - 1
        grid = torch.stack([theta_normalised, z_normalised], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            self.lookup_table[None],
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        ).view(2, -1)
        target_radius = sampled[0].view(scan_zyx.shape[:-1])
        confidence = sampled[1].view(scan_zyx.shape[:-1])
        in_z = (scan_zyx[..., 0] >= self.z_min) & (scan_zyx[..., 0] <= self.z_max)
        valid = in_z & (confidence >= cfg['shell_min_confidence'])
        return target_radius, radius, confidence, valid


def get_shell_outer_loss(shell_map, slice_to_spiral_transform, dr_per_winding, outer_winding_idx):
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if shell_map is None or outer_winding_idx is None:
        return zero, {}

    num_samples = max(1, int(cfg['shell_num_samples']))
    huber_delta = torch.as_tensor(cfg['shell_huber_delta'], device=device, dtype=torch.float32)

    outer_spiral = canonical_winding_samples([outer_winding_idx], num_samples, dr_per_winding, device, z_begin, z_end)[0]
    outer_scan = slice_to_spiral_transform.inv(outer_spiral)

    target_r, scan_r, confidence, valid = shell_map.lookup(outer_scan)
    residual = scan_r - target_r
    shell_outer_loss = _masked_mean(_huber_abs(residual, huber_delta), valid)

    metrics = {}
    with torch.no_grad():
        if valid.any():
            abs_residual = residual[valid].abs()
            metrics = {
                'shell_outer_error_mean': abs_residual.mean(),
                'shell_outer_error_p95': torch.quantile(abs_residual, 0.95),
                'shell_confidence_mean': confidence[valid].mean(),
            }

    return shell_outer_loss, metrics


class CartesianFlowField(nn.Module):

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

    def get_sampler(self, t):
        # Returns a callable mapping normalised zyx points in [0, 1] to flow velocity at time t.
        # Materialises the flow as a [3, Z, Y, X] cartesian tensor of zyx vector components once
        # and reuses it across the (e.g. RK4) integrator's many sample calls.
        lr_flow, hr_flow = self.flows[0], self.flows[1]
        hr_shape = tuple(hr_flow.shape[2:])
        if cfg['num_flow_timesteps'] == 1:
            # Time-invariant: HR flow is already at the target resolution, so skip interpolating it.
            lr_upsampled = F.interpolate(lr_flow, size=hr_shape, mode='trilinear')[0] * self.flow_scales[0]
            field = lr_upsampled + hr_flow[0] * self.flow_scales[1]
        else:
            t_scaled = (t.clamp(-1. + 1.e-4, 1. - 1.e-4) + 1) / 2 * (cfg['num_flow_timesteps'] - 1)
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
        if cfg['num_flow_timesteps'] == 1:
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
        if cfg['flow_field_type'] == 'cylindrical':
            from flow_fields import CylindricalFlowField
            self.flow_field = CylindricalFlowField(
                flow_resolution,
                lr_scale_factor=cfg['flow_field_high_res_lr_scale_initial'],
                num_flow_timesteps=cfg['num_flow_timesteps'],
            )
        else:
            self.flow_field = CartesianFlowField(
                flow_resolution,
                lr_scale_factor=cfg['flow_field_high_res_lr_scale_initial'],
            )

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

    def get_spiral_density(self, spiral_zyx, winding_range=None):
        return get_spiral_density(spiral_zyx[..., 1:], dr_per_winding=self.get_dr_per_winding(), sigma=1., winding_range=winding_range) * self.spiral_intensity


def run_containing_index(mask_1d: np.ndarray, idx: int) -> tuple[int, int] | None:
    """Return (start, end) of the contiguous True run containing idx."""
    padded = np.concatenate([[False], mask_1d, [False]])
    diff = np.diff(padded.astype(int))  # diff will be +1 at start of runs, -1 at end of runs
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0] - 1
    run_idx = np.searchsorted(run_starts, idx, side='right') - 1
    return run_starts[run_idx], run_ends[run_idx] + 1


def _build_line_runs(line_valid):
    # Returns (los, his) arrays of contiguous True runs in a 1-D bool array.
    padded = np.concatenate([[False], line_valid, [False]]).astype(np.int8)
    diff = np.diff(padded)
    los = np.where(diff == 1)[0].astype(np.int64)
    his = np.where(diff == -1)[0].astype(np.int64)
    return los, his


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

        # Precompute, per row and per column, the contiguous valid-quad runs so the
        # per-iteration sampler can skip the run_containing_index work. We keep the
        # original "row uniform, run-within-row weighted by length" distribution by
        # indexing runs per row/col.
        def _runs_per_line(mask_np, fixed_axis, valid_lines):
            # Returns parallel lists indexed 0..len(valid_lines)-1:
            # los_list[k], his_list[k], cum_list[k] are arrays of run lo/hi/cum-length
            # for the k'th valid line (mask_np row if fixed_axis==0 else column).
            los_list, his_list, cum_list = [], [], []
            for r in valid_lines:
                line = mask_np[r] if fixed_axis == 0 else mask_np[:, r]
                los, his = _build_line_runs(line)
                los_list.append(los)
                his_list.append(his)
                cum_list.append(np.cumsum(his - los))
            return los_list, his_list, cum_list

        patch._h_runs_los, patch._h_runs_his, patch._h_runs_cum = _runs_per_line(
            in_roi_quad_mask_np, 0, patch._sampling_valid_quad_rows
        )
        patch._v_runs_los, patch._v_runs_his, patch._v_runs_cum = _runs_per_line(
            in_roi_quad_mask_np, 1, patch._sampling_valid_quad_cols
        )

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


class PatchGpuAtlas:
    """All patches' (H, W, 3) zyxs grids packed into one flat GPU tensor, so
    fractional-(i, j) bilinear lookups can run as a single batched gather instead
    of per-patch CPU dispatch."""

    def __init__(self, patches_by_id, device='cuda'):
        flat_pieces = []
        offsets = [0]
        widths = []
        heights = []
        valid_in_roi_pieces = []
        for p in patches_by_id.values():
            z = p.zyxs  # (H, W, 3) on CPU
            H, W = z.shape[:2]
            z_flat = z.reshape(-1, 3).to(device=device, dtype=torch.float32)
            flat_pieces.append(z_flat)
            offsets.append(offsets[-1] + H * W)
            widths.append(W)
            heights.append(H)
            valid_flat = p.valid_vertex_mask.reshape(-1).to(device=device)
            z_in_roi = (z_flat[:, 0] >= z_begin) & (z_flat[:, 0] < z_end)
            valid_in_roi_pieces.append(z_flat[valid_flat & z_in_roi])
        self.zyxs_flat = torch.cat(flat_pieces, dim=0)
        self.offsets = torch.tensor(offsets, device=device, dtype=torch.int64)  # (N+1,)
        self.widths = torch.tensor(widths, device=device, dtype=torch.int64)  # (N,)
        self.heights = torch.tensor(heights, device=device, dtype=torch.int64)  # (N,)
        self.id_to_idx = {pid: i for i, pid in enumerate(patches_by_id.keys())}
        # Flat (N, 3) of every patch vertex with valid_vertex_mask & z-in-ROI, kept on GPU so
        # callers (track-exclusion masking, snap anchors) can do batched pairwise comparisons
        # without re-walking patches on the CPU each time.
        self.valid_in_roi_zyxs = (
            torch.cat(valid_in_roi_pieces, dim=0) if valid_in_roi_pieces
            else torch.empty([0, 3], device=device, dtype=torch.float32)
        )

    def memory_mb(self):
        return self.zyxs_flat.numel() * 4 / 1e6

    def lookup(self, patch_idx_per_sample, ijs):
        # patch_idx_per_sample: (...,) int64 on GPU
        # ijs: (..., 2) float on GPU
        # returns (..., 3) on GPU. Caller must ensure floor(ij) lies on a valid quad.
        base = self.offsets[patch_idx_per_sample]
        W = self.widths[patch_idx_per_sample]
        ij = ijs.to(torch.float32)
        i0 = ij[..., 0].floor().to(torch.int64)
        j0 = ij[..., 1].floor().to(torch.int64)
        di = (ij[..., 0] - i0.to(torch.float32)).unsqueeze(-1)
        dj = (ij[..., 1] - j0.to(torch.float32)).unsqueeze(-1)
        flat_tl = base + i0 * W + j0
        flat_tr = flat_tl + 1
        flat_bl = flat_tl + W
        flat_br = flat_bl + 1
        z = self.zyxs_flat
        tl = z[flat_tl]
        tr = z[flat_tr]
        bl = z[flat_bl]
        br = z[flat_br]
        top = tl + (tr - tl) * dj
        bottom = bl + (br - bl) * dj
        return top + (bottom - top) * di


def _aggregate_dt_track_losses(track_losses, across_p, active_mask=None):
    # Power-mean across tracks/patches: ((sum x^p) / n)^(1/p). When `active_mask` is given
    # (progressive DT gating), only the masked-in tracks contribute and n is the number active;
    # returns a zero scalar when none are active.
    if active_mask is not None:
        track_losses = track_losses[active_mask]
    if track_losses.numel() == 0:
        return torch.zeros([], device=track_losses.device)
    return ((track_losses ** across_p).sum() / track_losses.numel()) ** (1 / across_p)


def _progressive_dt_active_mask(snapped_winding, dr_per_winding, dt_max_winding):
    # Boolean mask over tracks/patches whose snapped spiral-space winding index is within the
    # progressive cutoff (see get_progressive_dt_max_winding); None when gating is disabled.
    # `snapped_winding` is the per-track round(median(shifted_radius)/dr)*dr target (sampled in
    # scroll space, transformed to spiral space upstream); we divide dr_per_winding back out to
    # recover the integer winding index.
    if dt_max_winding is None:
        return None
    winding_idx = (snapped_winding / dr_per_winding).detach()
    return winding_idx <= dt_max_winding


def _sample_patch_tracks(slice_to_spiral_transform, dr_per_winding, patches, patch_atlas, patch_indices, extra_zyxs=None, num_points_per_patch=None):
    if len(patch_indices) == 0:
        raise ValueError('Expected at least one patch index')

    # For each patch, we take one row and one column. _sample_strip_ijs picks a contiguous
    # subrange of each so _unwrap_track_shifted_radii can reliably handle theta=0 crossings
    # between consecutive sorted samples.

    # TODO: instead of 'strict' horizontal & vertical strips, could/should take wiggly strips that take a mostly-horizontal
    #  or mostly-vertical patch between distant points, skirting around gaps/holes; important for long, ragged traces

    if num_points_per_patch is None:
        num_points_per_patch = cfg['num_points_per_patch']
    num_points_per_direction = num_points_per_patch // 2
    N = len(patch_indices)

    P = num_points_per_direction
    horizontal_ijs_by_patch = np.empty([N, P, 2], dtype=np.float32)
    vertical_ijs_by_patch = np.empty([N, P, 2], dtype=np.float32)
    rand = np.random.random
    randint = np.random.randint
    fixed_jitters_h = rand(N).astype(np.float32)
    fixed_jitters_v = rand(N).astype(np.float32)
    var_jitters_h = rand((N, P)).astype(np.float32)
    var_jitters_v = rand((N, P)).astype(np.float32)
    for n, patch_idx in enumerate(patch_indices):
        patch = patches[patch_idx]

        # Horizontal: pick a row uniformly from rows-with-valid-quads, then pick a run
        # within that row weighted by length (matches original `np.random.choice(flatnonzero)`).
        rows_h = patch._sampling_valid_quad_rows
        k = randint(rows_h.shape[0])
        row_idx = rows_h[k]
        cum_h = patch._h_runs_cum[k]
        total_h = cum_h[-1]
        if cum_h.shape[0] == 1:
            r = 0
        else:
            r = np.searchsorted(cum_h, randint(total_h), side='right')
        lo_h = patch._h_runs_los[k][r]
        hi_h = patch._h_runs_his[k][r]
        run_len_h = hi_h - lo_h
        coords_h = np.sort(np.random.choice(run_len_h, P, replace=P > run_len_h))
        horizontal_ijs_by_patch[n, :, 0] = row_idx + fixed_jitters_h[n]
        horizontal_ijs_by_patch[n, :, 1] = lo_h + coords_h + var_jitters_h[n]

        # Vertical: same but with rows/cols swapped (fixed-coord is the column).
        cols_v = patch._sampling_valid_quad_cols
        k = randint(cols_v.shape[0])
        col_idx = cols_v[k]
        cum_v = patch._v_runs_cum[k]
        total_v = cum_v[-1]
        if cum_v.shape[0] == 1:
            r = 0
        else:
            r = np.searchsorted(cum_v, randint(total_v), side='right')
        lo_v = patch._v_runs_los[k][r]
        hi_v = patch._v_runs_his[k][r]
        run_len_v = hi_v - lo_v
        coords_v = np.sort(np.random.choice(run_len_v, P, replace=P > run_len_v))
        vertical_ijs_by_patch[n, :, 1] = col_idx + fixed_jitters_v[n]
        vertical_ijs_by_patch[n, :, 0] = lo_v + coords_v + var_jitters_v[n]

    # Batched bilinear interp on GPU: ijs are guaranteed to fall on valid quads by the
    # _sample_strip_ijs sampler (it draws i0/j0 from `_sampling_valid_quad_*`), so we
    # skip the per-call validity check used by patch.ij_to_zyx.
    combined_ijs_np = np.stack([horizontal_ijs_by_patch, vertical_ijs_by_patch], axis=0)  # (2, N, P, 2)
    combined_ijs_gpu = torch.from_numpy(combined_ijs_np).cuda(non_blocking=True)
    patch_indices_gpu = torch.from_numpy(np.ascontiguousarray(patch_indices, dtype=np.int64)).cuda(non_blocking=True)
    patch_idx_per_sample = patch_indices_gpu[None, :, None].expand(2, N, num_points_per_direction)
    all_slice_zyxs = patch_atlas.lookup(patch_idx_per_sample, combined_ijs_gpu)

    # When the caller has extra points (umbilicus, shell, ...), pack them into the same
    # forward ODE call to amortise the per-call overhead.
    patches_flat = all_slice_zyxs.reshape(-1, 3)
    if extra_zyxs is not None:
        combined_spiral = slice_to_spiral_transform(torch.cat([patches_flat, extra_zyxs], dim=0))
        n_patch_pts = patches_flat.shape[0]
        all_spiral_zyxs = combined_spiral[:n_patch_pts].reshape(*all_slice_zyxs.shape)
        extra_spiral = combined_spiral[n_patch_pts:]
    else:
        all_spiral_zyxs = slice_to_spiral_transform(patches_flat).reshape(*all_slice_zyxs.shape)
        extra_spiral = None

    all_theta, _, all_shifted_radii = get_theta_and_radii(all_spiral_zyxs[..., 1:], dr_per_winding)
    all_shifted_radii = _unwrap_track_shifted_radii(all_theta, all_shifted_radii, dr_per_winding)

    return all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii, extra_spiral



def _patch_radius_and_dt_losses(
    slice_to_spiral_transform, dr_per_winding,
    all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii,
    num_patches_for_radius, num_patches_for_dt, compute_dt, dt_max_winding,
    radius_loss_margin, radius_loss_inv, radius_within_norm_p,
    dt_loss_margin, dt_norm_p, dt_within_patch_norm_p,
):
    # Shared radius + DT patch losses, operating on pre-sampled row/column tracks
    # (all_*; see _sample_patch_tracks). Pulled out of get_patch_and_umbilicus_losses so the
    # same loss can serve both the verified and the untrusted ('unverified') patch sets with
    # independent hyperparameters. Returns (mean_radius_deviation, patch_dt_loss).
    radius_hinge_margin = dr_per_winding.detach() * radius_loss_margin
    dt_hinge_margin = dr_per_winding.detach() * dt_loss_margin

    # Each patch row/col should lie at constant shifted-radius.
    radius_shifted_radii = all_shifted_radii[:, :num_patches_for_radius]
    if radius_loss_inv:
        # Express the loss in scroll space like the DT loss below: construct target
        # spiral-space points at the track's mean shifted-radius (continuous, not snapped
        # to an integer winding) but with each point's own z and theta, transform back to
        # scroll space, and penalise the distance from the original sampled points.
        radius_slice_zyxs = all_slice_zyxs[:, :num_patches_for_radius]
        radius_spiral_zyxs = all_spiral_zyxs[:, :num_patches_for_radius]
        radius_theta = all_theta[:, :num_patches_for_radius]

        mean_shifted_radii = radius_shifted_radii.mean(dim=-1, keepdim=True)
        radius_target_radii = mean_shifted_radii + radius_theta / (2 * np.pi) * dr_per_winding
        radius_target_spiral_zyxs = torch.stack([
            radius_spiral_zyxs[..., 0],
            torch.sin(radius_theta) * radius_target_radii,
            torch.cos(radius_theta) * radius_target_radii,
        ], dim=-1).detach()

        radius_target_scroll_zyxs = slice_to_spiral_transform.inv(radius_target_spiral_zyxs.reshape(-1, 3)).reshape(*radius_target_spiral_zyxs.shape)

        radius_point_distances = torch.linalg.norm(radius_slice_zyxs - radius_target_scroll_zyxs, dim=-1)
        mean_radius_deviation = F.relu(radius_point_distances - radius_hinge_margin).mean()
    else:
        # Penalise deviation from the track's mean shifted-radius directly in spiral space.
        mean_radii = radius_shifted_radii.mean(dim=-1, keepdim=True)
        radius_deviations = (radius_shifted_radii - mean_radii).abs()
        radius_deviations_hinge = F.relu(radius_deviations - radius_hinge_margin)
        if radius_within_norm_p == 1.0:
            mean_radius_deviation = radius_deviations_hinge.mean()
        else:
            d = radius_deviations_hinge + 1.e-5
            per_track = (d ** radius_within_norm_p).mean(dim=-1) ** (1.0 / radius_within_norm_p)
            mean_radius_deviation = per_track.mean()

    if compute_dt:
        dt_slice_zyxs = all_slice_zyxs[:, :num_patches_for_dt]
        dt_spiral_zyxs = all_spiral_zyxs[:, :num_patches_for_dt]
        dt_theta = all_theta[:, :num_patches_for_dt]
        dt_shifted_radii = all_shifted_radii[:, :num_patches_for_dt]

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
        point_distances = F.relu(point_distances - dt_hinge_margin) + 1.e-5  # epsilon to avoid NaN in p-norm backward
        track_losses = (point_distances ** dt_within_patch_norm_p).mean(dim=-1) ** (1 / dt_within_patch_norm_p)
        # Progressive DT: only patches whose snapped winding is within the current cutoff contribute.
        active_mask = _progressive_dt_active_mask(target_shifted_radii.squeeze(-1), dr_per_winding, dt_max_winding)
        patch_dt_loss = _aggregate_dt_track_losses(track_losses, dt_norm_p, active_mask)
    else:
        patch_dt_loss = torch.zeros([], device=dr_per_winding.device)

    return mean_radius_deviation, patch_dt_loss


def get_patch_and_umbilicus_losses(slice_to_spiral_transform, dr_per_winding, num_patches_for_radius, num_patches_for_dt, patches, patch_atlas, patch_sampling_probabilities, umbilicus_zyx, compute_dt=True, shell_valid_zyxs=None, shell_outer_winding_idx=None, dt_max_winding=None):

    # Sample once and share the tracks between the radius and DT losses; the loss using
    # fewer patches takes a prefix of the larger sample.
    num_patches_to_sample = max(num_patches_for_radius, num_patches_for_dt) if compute_dt else num_patches_for_radius
    patch_indices = np.random.choice(len(patches), num_patches_to_sample, p=patch_sampling_probabilities, replace=True)

    n_umb = umbilicus_zyx.shape[0]
    if shell_valid_zyxs is not None:
        num_shell_samples = min(int(cfg['shell_num_samples']), shell_valid_zyxs.shape[0])
        sample_idx = torch.randint(shell_valid_zyxs.shape[0], (num_shell_samples,), device=shell_valid_zyxs.device)
        extra_zyxs = torch.cat([umbilicus_zyx, shell_valid_zyxs[sample_idx]], dim=0)
    else:
        extra_zyxs = umbilicus_zyx

    all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii, extra_spiral = _sample_patch_tracks(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        patch_atlas,
        patch_indices,
        extra_zyxs,
    )
    umbilicus_spiral = extra_spiral[:n_umb]
    shell_spiral_zyxs = extra_spiral[n_umb:] if shell_valid_zyxs is not None else None

    mean_radius_deviation, patch_dt_loss = _patch_radius_and_dt_losses(
        slice_to_spiral_transform, dr_per_winding,
        all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii,
        num_patches_for_radius, num_patches_for_dt, compute_dt, dt_max_winding,
        cfg['patch_radius_loss_margin'], cfg['patch_radius_loss_inv'], cfg['patch_radius_within_norm_p'],
        cfg['patch_dt_loss_margin'], cfg['patch_dt_norm_p'], cfg['patch_dt_within_patch_norm_p'],
    )

    # Umbilicus should map to the spiral origin (yx ≈ 0)
    umbilicus_loss = umbilicus_spiral[..., 1:].abs().mean()

    if shell_spiral_zyxs is not None:
        radius_hinge_margin = dr_per_winding.detach() * cfg['patch_radius_loss_margin']
        _, _, shell_shifted_radii = get_theta_and_radii(shell_spiral_zyxs[..., 1:], dr_per_winding)
        shell_target = dr_per_winding * float(shell_outer_winding_idx)
        shell_patch_radius_loss = F.relu((shell_shifted_radii - shell_target).abs() - radius_hinge_margin).mean()
    else:
        shell_patch_radius_loss = torch.zeros([], device=dr_per_winding.device)

    return mean_radius_deviation, umbilicus_loss, patch_dt_loss, shell_patch_radius_loss


def get_unverified_patch_losses(slice_to_spiral_transform, dr_per_winding, num_patches_for_radius, num_patches_for_dt, patches, patch_atlas, patch_sampling_probabilities, compute_dt=True, dt_max_winding=None):
    # Radius + DT losses for the untrusted 'unverified' patch set. Same machinery as the
    # verified patches (shared _sample_patch_tracks + _patch_radius_and_dt_losses) but with the
    # independent unverified_* hyperparameters and no umbilicus/shell extras. These patches are
    # masked away near trusted geometry upstream (see _mask_patches_near_trusted_geometry), so
    # they only constrain regions the verified inputs don't cover.
    num_patches_to_sample = max(num_patches_for_radius, num_patches_for_dt) if compute_dt else num_patches_for_radius
    patch_indices = np.random.choice(len(patches), num_patches_to_sample, p=patch_sampling_probabilities, replace=True)

    all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii, _ = _sample_patch_tracks(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        patch_atlas,
        patch_indices,
        num_points_per_patch=cfg['unverified_num_points_per_patch'],
    )

    return _patch_radius_and_dt_losses(
        slice_to_spiral_transform, dr_per_winding,
        all_slice_zyxs, all_spiral_zyxs, all_theta, all_shifted_radii,
        num_patches_for_radius, num_patches_for_dt, compute_dt, dt_max_winding,
        cfg['unverified_patch_radius_loss_margin'], cfg['unverified_patch_radius_loss_inv'], cfg['unverified_patch_radius_within_norm_p'],
        cfg['unverified_patch_dt_loss_margin'], cfg['unverified_patch_dt_norm_p'], cfg['unverified_patch_dt_within_patch_norm_p'],
    )


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


def get_patch_rel_winding_loss(slice_to_spiral_transform, dr_per_winding, patches_dict, patch_atlas, point_collections):
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
    # regressing it onto winding_diff * dr_per_winding. If the selected PCL
    # points (adjacent mode) or the PCL chain between them (non-adjacent mode)
    # crosses theta=0, adjust the expected delta by that branch-cut jump.

    num_points_per_strip = cfg['num_points_per_patch'] // 2
    num_strips_per_pcl = 4
    num_strips_per_pair = 2 * num_strips_per_pcl  # 8

    # Each entry: (ls1, ls2, pid1, pid2, winding_diff, pcl_chain_zyxs), where
    # ls* is a list of 4 L-shape ij strips and pcl_chain_zyxs is ordered p1 -> p2.
    strip_pairs = []

    # Single-point pcls (possible only for winding_is_absolute pcls) can't form a
    # cross-patch pair, so exclude them from the candidate pool before sampling.
    candidate_pcls = [pcl for pcl in point_collections if len(pcl['points']) > 1]
    num_pcls_per_step = min(cfg['rel_winding_num_pcls'], len(candidate_pcls))
    if num_pcls_per_step <= 0:
        return torch.zeros([], device='cuda')
    selected_idxs = np.random.choice(len(candidate_pcls), num_pcls_per_step, replace=False)
    selected_pcls = [candidate_pcls[i] for i in selected_idxs]

    for pcl in selected_pcls:
        sorted_pcl_points = None
        sorted_pcl_point_idx = None
        if not cfg['rel_winding_adjacent_patches_only']:
            sorted_pcl_points = [
                point for _, point in sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
            ]
            sorted_pcl_point_idx = {id(point): idx for idx, point in enumerate(sorted_pcl_points)}

        # Pair patches either only with their immediate neighbour in the pcl's
        # patch ordering (first-seen order; built in main()),
        # or with every other patch.
        if cfg['rel_winding_adjacent_patches_only']:
            cross_pairs = [(p1, p2) for p1, p2 in zip(pcl['points_by_patch'], list(pcl['points_by_patch'])[1:])]
        else:
            cross_pairs = list(itertools.combinations(pcl['points_by_patch'], r=2))
        if not cross_pairs:
            continue

        num_pairs_for_pcl = min(len(cross_pairs), cfg['rel_winding_num_patch_pairs_per_pcl'])
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

            if cfg['rel_winding_adjacent_patches_only']:
                pcl_chain = [p1, p2]
            else:
                idx1, idx2 = sorted_pcl_point_idx[id(p1)], sorted_pcl_point_idx[id(p2)]
                if idx1 <= idx2:
                    pcl_chain = sorted_pcl_points[idx1:idx2 + 1]
                else:
                    pcl_chain = list(reversed(sorted_pcl_points[idx2:idx1 + 1]))
            pcl_chain_zyxs = np.stack([point['zyx'] for point in pcl_chain], axis=0).astype(np.float32)
            strip_pairs.append((ls1, ls2, pid1, pid2, winding_diff, pcl_chain_zyxs))

    if not strip_pairs:
        return torch.zeros([], device='cuda')

    # Flatten: 8 strips per pair, ordered as p1's 4 strips followed by p2's 4 strips.
    total_strips = len(strip_pairs) * num_strips_per_pair
    flat_ijs = np.empty([total_strips, num_points_per_strip, 2], dtype=np.float32)
    flat_pids = []
    for k, (ls1, ls2, pid1, pid2, _, _) in enumerate(strip_pairs):
        base = k * num_strips_per_pair
        for s, strip in enumerate(ls1):
            flat_ijs[base + s] = strip
        for s, strip in enumerate(ls2):
            flat_ijs[base + num_strips_per_pcl + s] = strip
        flat_pids.extend([pid1] * num_strips_per_pcl + [pid2] * num_strips_per_pcl)

    # Batched GPU bilinear interp across all strips.
    patch_idx_per_strip_np = np.fromiter(
        (patch_atlas.id_to_idx[pid] for pid in flat_pids),
        dtype=np.int64,
        count=total_strips,
    )
    patch_idx_per_strip_gpu = torch.from_numpy(patch_idx_per_strip_np).cuda(non_blocking=True)
    ijs_gpu = torch.from_numpy(flat_ijs).cuda(non_blocking=True)
    patch_idx_per_sample = patch_idx_per_strip_gpu[:, None].expand(total_strips, num_points_per_strip)
    flat_zyxs = patch_atlas.lookup(patch_idx_per_sample, ijs_gpu)

    # Mask out strip samples whose z falls outside [z_begin - margin, z_end + margin).
    # Computed before unwrapping but applied after, since _unwrap_track_shifted_radii
    # needs the full sequential strip to stitch theta=0 crossings.
    z_margin = cfg['patch_loss_z_margin']
    z_mask = (flat_zyxs[..., 0] >= z_begin - z_margin) & (flat_zyxs[..., 0] < z_end + z_margin)

    flat_spiral = slice_to_spiral_transform(flat_zyxs.reshape(-1, 3)).reshape(*flat_zyxs.shape)
    theta, _, shifted_radii = get_theta_and_radii(flat_spiral[..., 1:], dr_per_winding)
    shifted_radii = _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding)

    # [num_pairs, 8, num_points_per_strip] -> pool each side's 4 strips into a single set.
    shifted_radii = shifted_radii.reshape(len(strip_pairs), num_strips_per_pair, num_points_per_strip)
    z_mask = z_mask.reshape(len(strip_pairs), num_strips_per_pair, num_points_per_strip)
    num_points_per_side = num_strips_per_pcl * num_points_per_strip
    p1_r = shifted_radii[:, :num_strips_per_pcl].reshape(len(strip_pairs), num_points_per_side)
    p2_r = shifted_radii[:, num_strips_per_pcl:].reshape(len(strip_pairs), num_points_per_side)
    m1 = z_mask[:, :num_strips_per_pcl].reshape(len(strip_pairs), num_points_per_side)
    m2 = z_mask[:, num_strips_per_pcl:].reshape(len(strip_pairs), num_points_per_side)

    winding_diffs = torch.tensor(
        [sp[4] for sp in strip_pairs],
        device='cuda',
        dtype=torch.float32,
    )
    pcl_seam_adjustments = []
    for _, _, _, _, _, pcl_chain_zyxs in strip_pairs:
        chain_zyxs = torch.from_numpy(pcl_chain_zyxs).cuda(non_blocking=True)
        chain_spiral = slice_to_spiral_transform(chain_zyxs)
        chain_theta, _, _ = get_theta_and_radii(chain_spiral[..., 1:], dr_per_winding)
        zero_shifted = torch.zeros_like(chain_theta)
        chain_adjustments = _unwrap_track_shifted_radii(chain_theta, zero_shifted, dr_per_winding)
        pcl_seam_adjustments.append(chain_adjustments[-1])
    pcl_seam_adjustments = torch.stack(pcl_seam_adjustments)
    expected_diff = ((winding_diffs * dr_per_winding) - pcl_seam_adjustments)[:, None, None]

    diff = p2_r[:, :, None] - p1_r[:, None, :]
    pair_mask = m2[:, :, None] & m1[:, None, :]
    err = (diff - expected_diff).abs()
    return (err * pair_mask).sum() / pair_mask.sum().clamp(min=1)


def get_patch_abs_winding_loss(slice_to_spiral_transform, dr_per_winding, patches_dict, patch_atlas, point_collections):
    # For PCL points carrying an absolute winding annotation (only pcls flagged
    # metadata.winding_is_absolute), pin the spiral shifted-radius at each annotated
    # point to its absolute target, winding_annotation * dr_per_winding (the spiral has
    # radius 0 at winding 0 and grows at dr_per_winding, so shifted_radius == winding *
    # dr_per_winding). This mirrors get_patch_rel_winding_loss, but anchors each point's
    # absolute winding instead of regressing a pair's winding difference: we sample some
    # absolute-winding pcls, some attached points within each, build 4 L-shaped strips
    # per point (sampled in traversal order so _unwrap_track_shifted_radii can stitch
    # theta=0 seam crossings), then drive every in-roi strip sample's shifted radius to
    # the point's target. Each L starts at the annotated point, so its unwrapped
    # shifted-radius keeps the true absolute scale at the anchor.

    num_points_per_strip = cfg['num_points_per_patch'] // 2
    num_strips_per_point = 4

    # Each entry: (ls, pid, winding_annotation) where ls is a list of 4 L-shape ij strips.
    strips = []

    abs_pcls = [pcl for pcl in point_collections if pcl.get('metadata', {}).get('winding_is_absolute', False)]
    num_pcls_per_step = min(cfg['abs_winding_num_pcls'], len(abs_pcls))
    if num_pcls_per_step <= 0:
        return torch.zeros([], device='cuda')
    selected_idxs = np.random.choice(len(abs_pcls), num_pcls_per_step, replace=False)
    selected_pcls = [abs_pcls[i] for i in selected_idxs]

    for pcl in selected_pcls:
        # An absolute-winding pcl's attached points, flattened across its patches.
        attached = [p for pts in pcl['points_by_patch'].values() for p in pts]
        if not attached:
            continue
        num_points_for_pcl = min(len(attached), cfg['abs_winding_num_points_per_pcl'])
        chosen = np.random.choice(len(attached), num_points_for_pcl, replace=False)
        for idx in chosen:
            p = attached[idx]
            pid = p['on_patch']['id']
            i, j = int(p['on_patch']['ij'][0]), int(p['on_patch']['ij'][1])
            ls = _sample_l_shapes_at_ij(patches_dict[pid], i, j, num_points_per_strip)
            if ls is None:
                continue
            strips.append((ls, pid, p['winding_annotation']))

    if not strips:
        return torch.zeros([], device='cuda')

    # Flatten: 4 strips per annotated point.
    total_strips = len(strips) * num_strips_per_point
    flat_ijs = np.empty([total_strips, num_points_per_strip, 2], dtype=np.float32)
    flat_pids = []
    for k, (ls, pid, _) in enumerate(strips):
        base = k * num_strips_per_point
        for s, strip in enumerate(ls):
            flat_ijs[base + s] = strip
        flat_pids.extend([pid] * num_strips_per_point)

    # Batched GPU bilinear interp across all strips.
    patch_idx_per_strip_np = np.fromiter(
        (patch_atlas.id_to_idx[pid] for pid in flat_pids),
        dtype=np.int64,
        count=total_strips,
    )
    patch_idx_per_strip_gpu = torch.from_numpy(patch_idx_per_strip_np).cuda(non_blocking=True)
    ijs_gpu = torch.from_numpy(flat_ijs).cuda(non_blocking=True)
    patch_idx_per_sample = patch_idx_per_strip_gpu[:, None].expand(total_strips, num_points_per_strip)
    flat_zyxs = patch_atlas.lookup(patch_idx_per_sample, ijs_gpu)

    # Mask out strip samples whose z falls outside [z_begin - margin, z_end + margin).
    # Computed before unwrapping but applied after, since _unwrap_track_shifted_radii
    # needs the full sequential strip to stitch theta=0 crossings.
    z_margin = cfg['patch_loss_z_margin']
    z_mask = (flat_zyxs[..., 0] >= z_begin - z_margin) & (flat_zyxs[..., 0] < z_end + z_margin)

    flat_spiral = slice_to_spiral_transform(flat_zyxs.reshape(-1, 3)).reshape(*flat_zyxs.shape)
    theta, _, shifted_radii = get_theta_and_radii(flat_spiral[..., 1:], dr_per_winding)
    shifted_radii = _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding)

    # [num_points, 4, num_points_per_strip] -> pool each point's 4 strips into one set.
    num_samples_per_point = num_strips_per_point * num_points_per_strip
    shifted_radii = shifted_radii.reshape(len(strips), num_samples_per_point)
    mask = z_mask.reshape(len(strips), num_samples_per_point)

    winding_annotations = torch.tensor(
        [s[2] for s in strips],
        device='cuda',
        dtype=torch.float32,
    )
    target_shifted = (winding_annotations * dr_per_winding)[:, None]

    err = (shifted_radii - target_shifted).abs()
    return (err * mask).sum() / mask.sum().clamp(min=1)


class _UnattachedPclStripList(list):
    """List of unattached-pcl strip dicts, with a slot for an attached `.flat`
    GPU bundle that batched satisfaction / winding-range computations reuse."""
    pass


def _prepare_unattached_pcl_strips(pcls_dict, min_point_spacing):
    # For each unattached pcl, materialise an id-sorted strip of point zyxs and the
    # corresponding winding annotations. Strips with <2 points are dropped.
    # If min_point_spacing > 0, decimate each strip greedily along its id-sorted order
    # so consecutive kept points are at least min_point_spacing apart in 3D scroll space.
    # The first and last points are always kept.
    prepared = _UnattachedPclStripList()
    for pcl_id, pcl in pcls_dict.items():
        sorted_items = sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
        if len(sorted_items) < 2:
            continue
        zyxs = np.stack([p['zyx'] for _, p in sorted_items], axis=0).astype(np.float32)
        windings = np.array([p['winding_annotation'] for _, p in sorted_items], dtype=np.float32)
        if min_point_spacing > 0 and len(zyxs) > 2:
            keep = [0]
            last_kept = zyxs[0]
            for i in range(1, len(zyxs) - 1):
                if np.linalg.norm(zyxs[i] - last_kept) >= min_point_spacing:
                    keep.append(i)
                    last_kept = zyxs[i]
            keep.append(len(zyxs) - 1)
            zyxs = zyxs[keep]
            windings = windings[keep]
        prepared.append({
            'id': pcl_id,
            'name': pcl.get('name'),
            'source_file': pcl.get('source_file'),
            'zyxs': zyxs,
            'windings': windings,
        })
    return prepared


def _build_strip_flat_bundle(strip_arrays, device):
    # Concatenate per-strip (zyxs, windings) arrays into one flat GPU tensor so the
    # downstream computations can run a single transform call plus segmented reductions
    # instead of per-strip Python loops. `strip_arrays` is a sequence of
    # `(zyxs_np, windings_np)` pairs. Returns None when there are no points.
    pairs = list(strip_arrays)
    if len(pairs) == 0:
        return None
    lengths_np = np.fromiter((len(z) for z, _ in pairs), dtype=np.int64, count=len(pairs))
    starts_np = np.empty(len(pairs) + 1, dtype=np.int64)
    starts_np[0] = 0
    np.cumsum(lengths_np, out=starts_np[1:])
    total = int(starts_np[-1])
    if total == 0:
        return None
    zyxs_flat = np.concatenate([z for z, _ in pairs], axis=0).astype(np.float32, copy=False)
    windings_flat = np.concatenate([w for _, w in pairs], axis=0).astype(np.float32, copy=False)
    strip_id_np = np.repeat(np.arange(len(pairs), dtype=np.int64), lengths_np)
    return {
        'zyxs': torch.from_numpy(zyxs_flat).to(device=device),
        'windings': torch.from_numpy(windings_flat).to(device=device),
        'strip_id': torch.from_numpy(strip_id_np).to(device=device),
        'starts': torch.from_numpy(starts_np).to(device=device),
        'starts_cpu': torch.from_numpy(starts_np),
        'lengths': torch.from_numpy(lengths_np).to(device=device),
        'lengths_cpu': torch.from_numpy(lengths_np),
        'num_strips': len(pairs),
        'total': total,
    }


def get_or_build_unattached_pcl_flat(pcl_strips, device):
    # Reuse a cached `.flat` bundle on the strip list when available (set up at the
    # top of fit_spiral_3d); otherwise build it now and try to cache for next call.
    flat = getattr(pcl_strips, 'flat', None)
    if flat is None and len(pcl_strips) > 0:
        flat = _build_strip_flat_bundle(((s['zyxs'], s['windings']) for s in pcl_strips), device)
        try:
            pcl_strips.flat = flat
        except AttributeError:
            pass
    return flat


def _decode_uint8_normal_component(value):
    return (value - 128.0) / 127.0


def get_radial_normal_in_scroll_space(slice_to_spiral_transform, scroll_zyx, spiral_zyx=None, epsilon=6.0):
    # At each scroll-space point, pull the spiral-space cylinder normal (the outward radial
    # direction normalize(spiral_yx)) back to scroll space as a covector, J^T n_spiral, where
    # J = d(spiral) / d(scroll) is estimated by central differences. This is the geometrically
    # correct transport of a surface normal (covector) -- unlike a tangent-vector pushforward J n.
    # Returns the normalised scroll-space normal direction (num_points, 3) in zyx.
    #
    # Gradient flows through the transform parameters via the Jacobian only; the sample positions
    # (scroll_zyx) and the radial direction are held fixed, matching the dense-normals loss. If the
    # forward image spiral_zyx is supplied it is reused for the radial direction (and treated as a
    # constant); otherwise it is computed here from scroll_zyx.
    device = scroll_zyx.device
    num_points = scroll_zyx.shape[0]
    scroll_zyx = scroll_zyx.detach()

    basis_zyx = torch.eye(3, device=device, dtype=scroll_zyx.dtype) * epsilon
    scroll_plus = (scroll_zyx[None, :, :] + basis_zyx[:, None, :]).reshape(-1, 3)
    scroll_minus = (scroll_zyx[None, :, :] - basis_zyx[:, None, :]).reshape(-1, 3)
    if spiral_zyx is None:
        combined_spiral = slice_to_spiral_transform(torch.cat([scroll_zyx, scroll_plus, scroll_minus], dim=0))
        spiral_zyx = combined_spiral[:num_points]
        spiral_plus, spiral_minus = combined_spiral[num_points:].chunk(2, dim=0)
    else:
        spiral_plus, spiral_minus = slice_to_spiral_transform(torch.cat([scroll_plus, scroll_minus], dim=0)).chunk(2, dim=0)

    spiral_outward_yx = F.normalize(spiral_zyx[:, 1:].detach(), dim=-1)
    spiral_outward_zyx = torch.cat([torch.zeros_like(spiral_outward_yx[:, :1]), spiral_outward_yx], dim=-1)

    spiral_plus = spiral_plus.view(3, num_points, 3)
    spiral_minus = spiral_minus.view(3, num_points, 3)
    jacobian_columns = (spiral_plus - spiral_minus) / (2.0 * epsilon)  # scroll basis axis, point, spiral zyx
    return F.normalize((jacobian_columns * spiral_outward_zyx[None, :, :]).sum(dim=-1).transpose(0, 1), dim=-1)


def sample_spiral_surface_frame(dr_per_winding, outer_winding_idx, num_points):
    # Sample points from discrete spiral windings embedded in spiral yx (over the z-ROI) and return
    # each point's orthonormal in-surface frame in spiral space: e1 = z-axis, e2 = the winding tangent.
    # Winding indices are sampled with probability proportional to their approximate circumference,
    # which is the simple large-radius approximation to uniform area over the wound surface. The inner
    # core is excluded because there is no scroll surface there.
    # Returns (spiral_zyx, e1, e2), each (num_points, 3) in zyx.
    device = dr_per_winding.device
    winding_weights = torch.arange(1, int(outer_winding_idx), device=device, dtype=dr_per_winding.dtype) + 0.5
    winding_idx = torch.multinomial(winding_weights, num_points, replacement=True).to(dr_per_winding.dtype) + 1.0
    theta = torch.rand([num_points], device=device) * (2 * torch.pi)
    radius = (winding_idx + theta / (2 * torch.pi)) * dr_per_winding.detach()
    z = torch.empty([num_points], device=device).uniform_(float(z_begin), float(z_end - 1))
    spiral_zyx = torch.stack([z, torch.sin(theta) * radius, torch.cos(theta) * radius], dim=-1)

    dr_dtheta = dr_per_winding.detach() / (2 * torch.pi)
    tangent_y = torch.cos(theta) * radius + torch.sin(theta) * dr_dtheta
    tangent_x = -torch.sin(theta) * radius + torch.cos(theta) * dr_dtheta
    tangential_yx = F.normalize(torch.stack([tangent_y, tangent_x], dim=-1), dim=-1)
    e1 = F.pad(torch.zeros_like(tangential_yx), (1, 0), value=1.)  # (1, 0, 0) -> z-axis
    e2 = F.pad(tangential_yx, (1, 0), value=0.)  # (0, ty, tx)
    return spiral_zyx, e1, e2


def get_lasagna_losses(slice_to_spiral_transform, dr_per_winding, lasagna_volume, outer_winding_idx, num_points, epsilon=2.0):
    # Sample points uniformly over the spiral cylinder (a disk of radius
    # dr_per_winding * outer_winding_idx in spiral yx, over the z-ROI). Two losses are computed:
    #   (normals) the spiral radial covector at each sample is pulled back to scroll space via
    #             central-difference J^T (a normal is a covector, not a finite-length displacement)
    #             and matched in direction to the precomputed nx/ny scroll-space normal.
    #   (spacing) at each sample, shift inward and outward by dr_per_winding/2 along the spiral
    #             radial direction (so the two endpoints span exactly one winding in spiral
    #             space), map both endpoints to scroll space, and integrate the winding-density
    #             field (grad_mag, windings per voxel) along the scroll-space segment between
    #             them. grad_mag is a density, not a distance, so the number of windings the
    #             segment actually crosses is the line integral of that density along it; for a
    #             correct fit the integral equals 1 (one winding). The density is decoded from
    #             grad_mag and converted from base-volume to current-grid voxels via
    #             downsample_factor.
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if lasagna_volume is None or outer_winding_idx is None:
        return zero, zero

    volume = lasagna_volume['volume']  # 3 (nx, ny, grad_mag), z, y, x  uint8
    z_size, y_size, x_size = lasagna_volume['shape']
    z_origin = lasagna_volume['z_origin']

    dr = dr_per_winding.detach()
    r_max = dr * float(outer_winding_idx)
    r_min = dr  # inner endpoint sits at radius - dr/2 >= dr/2 > 0
    theta = torch.rand([num_points], device=device) * (2 * torch.pi)
    radius = torch.sqrt(torch.rand([num_points], device=device) * (r_max ** 2 - r_min ** 2) + r_min ** 2)
    z = torch.empty([num_points], device=device).uniform_(float(z_begin), float(z_end - 1))
    sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
    spiral_zyx = torch.stack([z, sin_theta * radius, cos_theta * radius], dim=-1)
    radius_inner = radius - dr / 2
    radius_outer = radius + dr / 2
    spiral_inner = torch.stack([z, sin_theta * radius_inner, cos_theta * radius_inner], dim=-1)
    spiral_outer = torch.stack([z, sin_theta * radius_outer, cos_theta * radius_outer], dim=-1)

    scroll_samples = slice_to_spiral_transform.inv(torch.cat([spiral_inner, spiral_outer, spiral_zyx], dim=0))
    scroll_inner, scroll_outer, scroll_center = scroll_samples.chunk(3, dim=0)
    scroll_displacement = scroll_outer - scroll_inner  # spans exactly one winding in spiral space
    scroll_segment_length = torch.linalg.norm(scroll_displacement, dim=-1).clamp(min=1.e-8)

    # Look up the precomputed scroll-space targets at the midpoint of the displacement (the
    # geometric centre of the one-winding step in scroll space).
    scroll_mid = ((scroll_inner + scroll_outer) / 2).detach()
    sample_zyx = scroll_mid.round().long()
    zi = sample_zyx[:, 0] - z_origin
    yi = sample_zyx[:, 1]
    xi = sample_zyx[:, 2]
    in_bounds = (zi >= 0) & (zi < z_size) & (yi >= 0) & (yi < y_size) & (xi >= 0) & (xi < x_size)
    zi = zi.clamp(0, z_size - 1)
    yi = yi.clamp(0, y_size - 1)
    xi = xi.clamp(0, x_size - 1)
    nx_u8 = volume[0, zi, yi, xi]
    ny_u8 = volume[1, zi, yi, xi]
    normal_weight = (((nx_u8 != 0) | (ny_u8 != 0)) & in_bounds).float()
    nx = _decode_uint8_normal_component(nx_u8.float())
    ny = _decode_uint8_normal_component(ny_u8.float())
    nz = torch.sqrt((1. - nx * nx - ny * ny).clamp(min=0.))
    target_normal = F.normalize(torch.stack([nz, ny, nx], dim=-1), dim=-1)  # zyx

    scroll_normal = get_radial_normal_in_scroll_space(slice_to_spiral_transform, scroll_center, spiral_zyx=spiral_zyx, epsilon=epsilon)
    normals_residual = 1. - (scroll_normal * target_normal).sum(dim=-1).abs()
    normals_loss = (normals_residual * normal_weight).sum() / normal_weight.sum().clamp(min=1)

    # grad_mag encodes a winding density (windings per base-volume voxel); the decode factor below
    # also rescales it to current-grid windings/voxel. The number of windings actually crossed by
    # the one-winding scroll-space segment (scroll_inner -> scroll_outer) is the line integral of
    # this density along it, so we sample the density at evenly spaced midpoints along the segment
    # and accumulate density * dl (a midpoint Riemann sum). For a correct fit the integral equals 1.
    density_decode = cfg['grad_mag_factor'] / cfg['grad_mag_encode_scale'] * downsample_factor
    num_steps = int(cfg['spacing_integration_steps'])
    step_frac = (torch.arange(num_steps, device=device).float() + 0.5) / num_steps  # midpoints in [0, 1]
    # [num_points, num_steps, 3] scroll-space samples along scroll_inner -> scroll_outer
    integration_zyx = scroll_inner[:, None, :] + step_frac[None, :, None] * scroll_displacement[:, None, :]
    int_idx = integration_zyx.detach().round().long()
    izi = int_idx[..., 0] - z_origin
    iyi = int_idx[..., 1]
    ixi = int_idx[..., 2]
    int_in_bounds = (izi >= 0) & (izi < z_size) & (iyi >= 0) & (iyi < y_size) & (ixi >= 0) & (ixi < x_size)
    izi = izi.clamp(0, z_size - 1)
    iyi = iyi.clamp(0, y_size - 1)
    ixi = ixi.clamp(0, x_size - 1)
    grad_mag_u8 = volume[2, izi, iyi, ixi]  # [num_points, num_steps]
    sample_valid = (grad_mag_u8 != 0) & int_in_bounds
    density = grad_mag_u8.float() * density_decode  # current-grid windings/voxel
    # dl is the per-step scroll-space length (current-grid voxels); gradient flows through it so the
    # loss can stretch/compress the mapping until the integrated winding count matches.
    dl = scroll_segment_length / num_steps
    integrated_windings = (density * sample_valid.float()).sum(dim=-1) * dl
    # Only score samples whose whole segment lies inside the valid field; a partially covered path
    # would under-integrate and unfairly compare against 1.
    spacing_weight = sample_valid.all(dim=-1).float()
    spacing_residual = (integrated_windings - 1.).abs()
    spacing_loss = (spacing_residual * spacing_weight).sum() / spacing_weight.sum().clamp(min=1)

    return normals_loss, spacing_loss


def get_unattached_pcl_strip_losses(
    slice_to_spiral_transform,
    dr_per_winding,
    pcl_strips,
    num_pcls_per_step,
    num_points_per_pcl,
    compute_dt,
    dt_max_winding=None,
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

    flat = get_or_build_unattached_pcl_flat(pcl_strips, device)
    if flat is None or flat['total'] == 0:
        return zero, zero

    starts_cpu = flat['starts_cpu'].numpy()
    sampled_flat_indices = np.empty([num_to_sample, num_points_per_pcl], dtype=np.int64)
    for k, pcl_idx in enumerate(chosen):
        strip = pcl_strips[pcl_idx]
        N = len(strip['zyxs'])
        coords = np.sort(np.random.choice(N, num_points_per_pcl, replace=num_points_per_pcl > N))
        sampled_flat_indices[k] = starts_cpu[pcl_idx] + coords

    sampled_flat_indices_t = torch.from_numpy(sampled_flat_indices).to(device=device)
    zyxs_t = flat['zyxs'][sampled_flat_indices_t]
    winding_t = flat['windings'][sampled_flat_indices_t]

    spiral_zyxs = slice_to_spiral_transform(zyxs_t.reshape(-1, 3)).reshape(*zyxs_t.shape)
    theta, _, shifted_radii = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
    shifted_radii = _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding)

    # Normalise so a pcl with mixed annotations still reads as a single 'strip'.
    normalised_radii = shifted_radii - winding_t * dr_per_winding

    radius_hinge_margin = dr_per_winding.detach() * cfg['patch_radius_loss_margin']
    dt_hinge_margin = dr_per_winding.detach() * cfg['patch_dt_loss_margin']

    mean_radii = normalised_radii.mean(dim=-1, keepdim=True)
    radius_deviations = (normalised_radii - mean_radii).abs()
    radius_loss = F.relu(radius_deviations - radius_hinge_margin).mean()

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
    point_distances = F.relu(point_distances - dt_hinge_margin) + 1.e-5
    track_losses = (point_distances ** within_p).mean(dim=-1) ** (1 / within_p)
    # Progressive DT: only strips whose snapped (raw, spiral-space) winding is within the current
    # cutoff contribute. Use shifted_radii (the strip's actual spiral position), not normalised_radii.
    strip_snapped_winding = torch.round(shifted_radii.median(dim=-1).values / dr_per_winding) * dr_per_winding
    active_mask = _progressive_dt_active_mask(strip_snapped_winding, dr_per_winding, dt_max_winding)
    dt_loss = _aggregate_dt_track_losses(track_losses, across_p, active_mask)

    return radius_loss, dt_loss


def get_symmetric_dirichlet_loss(slice_to_spiral_transform, dr_per_winding, outer_winding_idx, num_points, epsilon=1.0):
    # In-surface symmetric Dirichlet energy of the spiral<->scroll map, evaluated at points sampled
    # uniformly over the spiral cylinder (see sample_spiral_surface_frame).
    # At each point we take the orthonormal in-surface frame (e1, e2) in spiral space, map it to scroll
    # space through the inverse transform by finite differences to get its scroll-space image (a, b), and
    # form the 2x2 induced metric G = [[a.a, a.b], [a.b, b.b]]. The energy ||J||_F^2 + ||J^{-1}||_F^2 =
    # tr(G) + tr(G^{-1}) = (s1^2 + s2^2) + (1/s1^2 + 1/s2^2) is minimised (value 4) at an in-surface
    # isometry and diverges as the map degenerates (singular value -> 0 or inf), acting as a barrier
    # against in-surface collapse / element flips. We subtract 4 so the reported value is 0 at rest.
    device = dr_per_winding.device
    if outer_winding_idx is None:
        return torch.zeros([], device=device)

    spiral_zyx, e1, e2 = sample_spiral_surface_frame(dr_per_winding, outer_winding_idx, num_points)

    spiral_shift_1 = spiral_zyx + e1 * epsilon
    spiral_shift_2 = spiral_zyx + e2 * epsilon
    combined_spiral = torch.cat([spiral_zyx, spiral_shift_1, spiral_shift_2], dim=0)
    combined_scroll = slice_to_spiral_transform.inv(combined_spiral)
    scroll_zyx, scroll_shift_1, scroll_shift_2 = combined_scroll.chunk(3, dim=0)

    a = (scroll_shift_1 - scroll_zyx) / epsilon
    b = (scroll_shift_2 - scroll_zyx) / epsilon
    g11 = (a * a).sum(dim=-1)
    g22 = (b * b).sum(dim=-1)
    g12 = (a * b).sum(dim=-1)
    trace_g = g11 + g22
    det_g = g11 * g22 - g12 * g12
    # Energy is tr(G) + tr(G^{-1}) = (s1^2 + s2^2) + (1/s1^2 + 1/s2^2), regularised per-eigenvalue so a
    # vanishing singular value contributes a finite-but-large 1/(lambda+eps) barrier. We compute the
    # regularised inverse-eigenvalue sum directly from trace_g, det_g via the algebraic identity
    #   1/(l1+eps) + 1/(l2+eps) = ((l1+eps) + (l2+eps)) / ((l1+eps)(l2+eps))
    #                           = (trace_g + 2*eps) / (det_g + eps*trace_g + eps**2)
    inverse_eps = 1e-3
    inverse_term = (trace_g + 2.0 * inverse_eps) / (det_g + inverse_eps * trace_g + inverse_eps ** 2)
    energy = (trace_g + inverse_term - 4.0).clamp(min=0.0)
    # Per-sample cap so a single near-degenerate sample doesn't dominate the batch mean / gradient.
    energy = energy.clamp(max=1.e2)
    return energy.mean()



def get_patch_satisfied_areas(slice_to_spiral_transform, dr_per_winding, patches, verbose=False):
    return _get_patch_satisfied_areas(
        slice_to_spiral_transform, dr_per_winding, patches, z_begin, z_end, verbose=verbose,
    )


def get_unattached_pcl_satisfied_counts(slice_to_spiral_transform, dr_per_winding, pcl_strips):
    return _get_unattached_pcl_satisfied_counts(
        slice_to_spiral_transform, dr_per_winding, pcl_strips, get_or_build_unattached_pcl_flat,
    )



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
    winding_range,
    tracks,
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
        # quad_status_flat: 0=patch-and-quad not satisfied, 1=quad-only satisfied, 2=patch overall satisfied
        status = quad_status_flat.to(device=spiral.device, dtype=torch.int64)
        label_map = label_map.to(device=spiral.device, dtype=torch.int64)
        num_labels = status.numel()
        status_palette = torch.tensor([
            [200, 0, 0],  # 0: red — quad not satisfied (patch not overall satisfied)
            [230, 140, 0],  # 1: orange — quad satisfied but patch not overall satisfied
            [255, 200, 0],  # 2: yellow — patch overall satisfied
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

    def visualise_field(spiral_zyx, name):
        # Show the slice→spiral field as RGB: spiral (z, y, x) normalised into [0, 1] per channel
        # using the value range across the slice.
        values = spiral_zyx.to(torch.float32).reshape(-1, 3)
        lo = values.amin(dim=0)
        hi = values.amax(dim=0)
        normalised = ((spiral_zyx.to(torch.float32) - lo) / (hi - lo).clamp_min(1e-6)).clamp(0., 1.)
        canvas = (normalised * 255).to(torch.uint8)
        Image.fromarray(canvas.cpu().numpy()).save(f'{out_path}/{name}_{suffix}.png', compress_level=3)

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

    dr_per_winding = spiral_and_transform.get_dr_per_winding()
    for vis_slice_idx, slice_z in enumerate(tqdm(zs_for_visualisation, desc='visualising slices')):
        spiral_zyx, spiral_density = _rasterize_spiral_for_slice(
            spiral_and_transform, slice_to_spiral_transform, slice_yx, slice_z, winding_range,
        )
        slice = scroll_slices_for_visualisation[vis_slice_idx].to(device)
        # overlay_on_scroll(slice_zyx, spiral_zyx, spiral_density, slice, f'scroll_s{slice_z * downsample_factor:05}')
        # overlay_on_predictions(spiral_density, prediction_slices_for_visualisation[vis_slice_idx].to(device), slice > 0., f'pred_s{slice_z * downsample_factor:05}')
        overlay_on_patch_satisfaction(spiral_density, spiral_zyx, quad_label_map[vis_slice_idx], slice_z, f'patches_s{slice_z * downsample_factor:05}')
        if tracks:
            render_spiral_on_tracks_for_slice(
                spiral_zyx, spiral_density, dr_per_winding,
                slice_z, tracks, [],
                out_path, suffix, downsample_factor,
            )
        if os.environ.get('FIT_SPIRAL_SAVE_DISPLACEMENT') == '1':
            slice_zyx = torch.cat([torch.full([*slice_yx.shape[:2], 1], slice_z, device=device), slice_yx], dim=-1)
            visualise_field(spiral_zyx - slice_zyx, f'displacement_s{slice_z * downsample_factor:05}')


def _collect_anchor_scroll_zyxs(patch_atlas, unattached_pcl_strips, device):
    # Concatenate every in-ROI patch vertex (sourced from PatchGpuAtlas.valid_in_roi_zyxs,
    # already on GPU) and every in-ROI unattached-pcl strip point as scroll-space zyxs on
    # `device`. The result is the trusted point cloud used to mask out track and unverified
    # patch points near already-constrained regions.
    pieces = [patch_atlas.valid_in_roi_zyxs]
    for strip in unattached_pcl_strips:
        zyxs = torch.from_numpy(strip['zyxs']).to(device=device, dtype=torch.float32)
        in_roi = (zyxs[..., 0] >= z_begin) & (zyxs[..., 0] < z_end)
        if in_roi.any():
            pieces.append(zyxs[in_roi])
    return torch.cat(pieces, dim=0)


def _build_anchor_kdtree(anchor_zyx):
    # Build a cKDTree over the scroll-space anchor points (CPU) for fixed-radius
    # nearest-neighbour queries. Accepts a torch tensor or ndarray; returns None when
    # there are no anchors.
    if anchor_zyx is None:
        return None
    if isinstance(anchor_zyx, torch.Tensor):
        anchor_np = anchor_zyx.detach().cpu().numpy()
    else:
        anchor_np = np.asarray(anchor_zyx)
    if anchor_np.shape[0] == 0:
        return None
    return cKDTree(np.ascontiguousarray(anchor_np, dtype=np.float32))


def _points_far_from_anchors_mask(points_zyx, anchor_tree, threshold):
    # Returns a boolean numpy mask, True where the point is farther than `threshold`
    # from every anchor. `points_zyx` is an (N, 3) array (numpy or tensor); `anchor_tree` is
    # a cKDTree built by _build_anchor_kdtree (or None). Uses a fixed-radius nearest-neighbour
    # query — O(N log A), parallel across cores — instead of the full pairwise distance matrix.
    if isinstance(points_zyx, torch.Tensor):
        points_np = points_zyx.detach().cpu().numpy()
    else:
        points_np = np.asarray(points_zyx)
    points_np = np.ascontiguousarray(points_np, dtype=np.float32)
    if threshold <= 0 or anchor_tree is None:
        return np.ones(points_np.shape[0], dtype=bool)
    # query returns dist == inf for points with no anchor within distance_upper_bound.
    dist, _ = anchor_tree.query(points_np, k=1, distance_upper_bound=float(threshold), workers=-1)
    return np.isinf(dist)


def _mask_patches_near_trusted_geometry(patches_dict, anchor_tree, radius):
    # For each patch in `patches_dict`, invalidate (set zyxs -> -1) every currently-valid vertex
    # lying within `radius` (scroll space, downsampled voxels) of any trusted-geometry anchor in
    # `anchor_tree`, then re-derive the patch's masks/area. Patches left with no valid quad are
    # dropped. This is the patch analogue of the DBM-track exclusion in tracks.py:
    # untrusted patches only constrain regions the trusted inputs don't already cover, so they
    # can't fight verified geometry. Mutates and returns a possibly-smaller dict; a non-positive
    # radius or empty tree leaves everything intact.
    if not patches_dict or anchor_tree is None or radius <= 0:
        return patches_dict
    kept = {}
    n_masked_vertices = 0
    n_dropped = 0
    for pid, patch in patches_dict.items():
        zyxs_np = patch.zyxs.reshape(-1, 3).cpu().numpy()
        valid_flat = patch.valid_vertex_mask.reshape(-1).cpu().numpy()
        far = _points_far_from_anchors_mask(zyxs_np, anchor_tree, radius)  # True => keep
        invalidate = valid_flat & ~far
        if invalidate.any():
            mask2d = torch.from_numpy(invalidate.reshape(patch.zyxs.shape[:2]))
            patch.zyxs[mask2d] = -1.0
            n_masked_vertices += int(invalidate.sum())
            new_valid_vertex = torch.any(patch.zyxs != -1, dim=-1)
            new_valid_quad = (
                new_valid_vertex[:-1, :-1] & new_valid_vertex[1:, :-1]
                & new_valid_vertex[:-1, 1:] & new_valid_vertex[1:, 1:]
            )
            if not bool(new_valid_quad.any()):
                n_dropped += 1
                continue
            patch.__post_init__()  # re-derive valid_vertex_mask / valid_quad_mask / area / valid_zyxs
        kept[pid] = patch
    print(
        f'unverified patches: masked {n_masked_vertices} vertices near trusted geometry '
        f'(radius {radius:.1f}), dropped {n_dropped} fully-masked patches; {len(kept)} remain'
    )
    return kept


@torch.no_grad()
def _rasterize_spiral_for_slice(spiral_and_transform, slice_to_spiral_transform, slice_yx, slice_z, winding_range):
    # Transform a full slice from scroll into spiral space (in chunks to bound peak VRAM)
    # and evaluate the per-winding spiral density there. Returns (spiral_zyx, spiral_density).
    device = slice_yx.device
    slice_zyx = torch.cat([
        torch.full([*slice_yx.shape[:2], 1], float(slice_z), device=device),
        slice_yx,
    ], dim=-1)
    slice_zyx_flat = slice_zyx.reshape(-1, 3)
    chunk = 65536
    spiral_pieces = []
    for start in range(0, slice_zyx_flat.shape[0], chunk):
        spiral_pieces.append(slice_to_spiral_transform(slice_zyx_flat[start:start + chunk]))
    spiral_zyx = torch.cat(spiral_pieces, dim=0).reshape(*slice_zyx.shape)
    spiral_density = spiral_and_transform.get_spiral_density(spiral_zyx, winding_range=winding_range)
    return spiral_zyx, spiral_density


def get_flow_field_high_res_lr_scale(iteration):
    # Factor multiplying the high-resolution flow logits, which scales down their effective
    # learning rate relative to the main LR (kept <= 1 so the hi-res LR stays bounded by the
    # main LR). Ramps linearly from _initial to _final over _ramp_steps steps, starting at
    # _ramp_start_step; constant when _initial == _final.
    initial = cfg['flow_field_high_res_lr_scale_initial']
    final = cfg['flow_field_high_res_lr_scale_final']
    start_step = cfg['flow_field_high_res_lr_ramp_start_step']
    ramp_steps = max(1, int(cfg['flow_field_high_res_lr_ramp_steps']))
    frac = min(1., max(0., (iteration - start_step) / ramp_steps))
    return min(1., initial + frac * (final - initial))


def get_progressive_dt_max_winding(iteration, dt_start_step, shell_outer_winding_idx):
    # When `dt_progressive_windings` is set, the DT losses (patch, track, unattached-pcl) only act
    # on tracks/patches whose snapped spiral-space winding is <= the returned cutoff. The cutoff
    # grows outwards from `dt_progressive_inner_winding` (when the DT loss first turns on, at
    # `dt_start_step`) to `shell_outer_winding_idx` over `dt_progressive_steps` steps, so the
    # constraint expands across windings even after it has started. Returns None to disable gating
    # (include everything) -- when the feature is off, or no outer winding is known.
    #
    # The membership test lives in spiral space, but tracks/patches are sampled in scroll space;
    # callers reuse the per-track snapped winding (round(median(shifted_radius)/dr)) already needed
    # for the DT target, so deciding inclusion needs no extra transform (only a handful of points).
    #
    # `dt_progressive_exponent` warps the linear time fraction f -> f**exponent before mapping to
    # the winding cutoff. exponent == 1 grows the winding index (radius) linearly; exponent < 1 is
    # concave (fast early, slow late), so the outermost windings -- which gain area/volume
    # quadratically -- expand more slowly and get more time to catch up (~0.5 ≈ constant
    # area-introduction rate); exponent > 1 is the opposite.
    if not cfg['dt_progressive_windings'] or shell_outer_winding_idx is None:
        return None
    span = max(1, int(cfg['dt_progressive_steps']))
    f = min(1., max(0., (iteration - dt_start_step) / span))
    exponent = float(cfg['dt_progressive_exponent'])
    f_warped = f ** exponent if exponent != 1.0 else f
    w_inner = float(cfg['dt_progressive_inner_winding'])
    w_outer = float(shell_outer_winding_idx)
    return w_inner + (w_outer - w_inner) * f_warped


def fit_spiral_3d(scroll_zarr, patches_dict, point_collections, unattached_pcl_strips, tracks, lasagna_volume, shell_patch, z_to_umbilicus_yx, out_path, unverified_patches_dict=None):
    patches_list = list(patches_dict.values())
    patch_sampling_probabilities = _prepare_patch_sampling_cache(patches_list, cfg['patch_loss_z_margin'])

    num_patches = len(patches_list)
    print(f'fitting {num_patches} patches')

    patch_atlas = PatchGpuAtlas(patches_dict, device='cuda')
    print(f'patch GPU atlas: {patch_atlas.memory_mb():.1f} MB')

    num_slices_for_visualisation = 20
    rendering_slices_downsample_factor = 2  # stride the scroll by this along zyx for rendering

    device = torch.device('cuda')

    # Untrusted 'unverified' patches: mask away wherever they fall near trusted geometry (verified
    # patch vertices + pcl strips, same anchor cloud used for snap-anchors / track-exclusion), then
    # build their own sampling cache + GPU atlas. They feed only their own radius/DT losses.
    unverified_patches_dict = unverified_patches_dict or {}
    unverified_patches_list = []
    unverified_patch_sampling_probabilities = None
    unverified_patch_atlas = None
    if unverified_patches_dict:
        trusted_anchor_scroll = _collect_anchor_scroll_zyxs(patch_atlas, unattached_pcl_strips, device)
        trusted_anchor_tree = _build_anchor_kdtree(trusted_anchor_scroll)
        unverified_patches_dict = _mask_patches_near_trusted_geometry(
            unverified_patches_dict, trusted_anchor_tree, float(cfg['unverified_patch_exclusion_radius']),
        )
    if unverified_patches_dict:
        unverified_patches_list = list(unverified_patches_dict.values())
        unverified_patch_sampling_probabilities = _prepare_patch_sampling_cache(unverified_patches_list, cfg['patch_loss_z_margin'])
        unverified_patch_atlas = PatchGpuAtlas(unverified_patches_dict, device='cuda')
        print(f'fitting {len(unverified_patches_list)} unverified patches; atlas {unverified_patch_atlas.memory_mb():.1f} MB')

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

    # Load the resume checkpoint (if any) before constructing the model. The
    # model's parameter tensors are shaped by the z-range it was trained with,
    # so when resuming we must build them with the checkpoint's z-range -
    # otherwise the shapes won't match and load_state_dict will fail. This only
    # affects the model's flow-field domain; the optimisation continues to use
    # the current z_begin/z_end for sampling, losses and rendering.
    resume_path = os.environ.get('FIT_SPIRAL_RESUME_PATH')
    start_iteration = int(os.environ.get('FIT_SPIRAL_RESUME_STEP', '0'))
    resume_checkpoint = None
    model_z_begin, model_z_end = z_begin, z_end
    if resume_path:
        resume_checkpoint = torch.load(resume_path, map_location='cpu')
        if isinstance(resume_checkpoint, dict) and 'z_begin' in resume_checkpoint:
            model_z_begin, model_z_end = resume_checkpoint['z_begin'], resume_checkpoint['z_end']
            if (model_z_begin, model_z_end) != (z_begin, z_end):
                print(f'using checkpoint z-range [{model_z_begin}, {model_z_end}) for model parameter shapes (optimisation z-range is [{z_begin}, {z_end}))')
                assert z_begin >= model_z_begin and z_end <= model_z_end, (
                    f'optimisation z-range [{z_begin}, {z_end}) extends beyond the checkpoint '
                    f"model z-range [{model_z_begin}, {model_z_end}); the flow field has no "
                    'parameters outside its domain. Narrow z_begin/z_end to fit within the '
                    'checkpoint range, or train from scratch with the wider range.'
                )

    flow_field_radius = cfg['flow_bounds_radius']
    flow_min_corner_spiral_zyx = torch.tensor([model_z_begin - cfg['flow_bounds_z_margin'], -flow_field_radius, -flow_field_radius], dtype=torch.int64, device=device)
    flow_max_corner_spiral_zyx = torch.tensor([model_z_end + cfg['flow_bounds_z_margin'], flow_field_radius, flow_field_radius], dtype=torch.int64, device=device)

    num_training_steps = cfg['num_training_steps']

    spiral_and_transform = SpiralAndTransform(flow_integration_steps=cfg['num_flow_integration_steps'], flow_integration_solver=cfg['flow_integration_solver'], umbilicus_zyx=umbilicus_zyx, flow_min_corner_zyx=flow_min_corner_spiral_zyx, flow_max_corner_zyx=flow_max_corner_spiral_zyx)
    spiral_and_transform.to(device)

    shell_map = None
    shell_outer_winding_idx = None
    shell_valid_zyxs_gpu = None
    if shell_patch is not None and shell_losses_enabled():
        if cfg['loss_weight_shell_outer'] > 0:
            shell_map = ShellPolarMap(
                shell_patch,
                z_to_umbilicus_yx,
                z_min=z_begin - cfg['flow_bounds_z_margin'],
                z_max=z_end + cfg['flow_bounds_z_margin'],
                num_theta_bins=cfg['shell_num_theta_bins'],
                device=device,
            )
        if cfg['loss_weight_shell_patch_radius'] > 0:
            shell_valid_zyxs_gpu = shell_patch.valid_zyxs.to(device=device, dtype=torch.float32)
        initial_transform = spiral_and_transform.get_slice_to_spiral_transform()
        initial_dr = spiral_and_transform.get_dr_per_winding()
        if cfg['shell_outer_winding_idx'] is None:
            shell_outer_winding_idx = _infer_shell_outer_winding_idx(
                initial_transform,
                initial_dr,
                patches_list,
                unattached_pcl_strips,
                cfg,
                z_begin,
                z_end,
                get_or_build_unattached_pcl_flat,
            )
            print(f'inferred shell_outer_winding_idx = {shell_outer_winding_idx}')
        else:
            shell_outer_winding_idx = int(cfg['shell_outer_winding_idx'])
            print(f'using configured shell_outer_winding_idx = {shell_outer_winding_idx}')
        min_gap_expander_num_windings = shell_outer_winding_idx + 3
        if cfg['gap_expander_num_windings'] < min_gap_expander_num_windings:
            print(
                f'WARNING: shell_outer_winding_idx {shell_outer_winding_idx} requires '
                f'gap_expander_num_windings >= {min_gap_expander_num_windings}, got '
                f'gap_expander_num_windings {cfg["gap_expander_num_windings"]}; '
                'increase gap_expander_num_windings or lower shell_outer_winding_idx'
            )

    flow_field_params = list(spiral_and_transform.flow_field.parameters())
    gap_expander_params = list(spiral_and_transform.gap_expander_params.parameters())
    linear_params = [spiral_and_transform.linear_logits]
    grouped_ids = {id(p) for p in flow_field_params + gap_expander_params + linear_params}
    other_params = [p for p in spiral_and_transform.parameters() if id(p) not in grouped_ids]
    param_groups = [
        {'params': other_params, 'weight_decay': 0.0},
        {'params': linear_params, 'weight_decay': 0.0},
        {'params': gap_expander_params, 'weight_decay': cfg['weight_decay_gap_expander']},
        {'params': flow_field_params, 'weight_decay': cfg['weight_decay_flow_field']},
    ]
    optimiser = torch.optim.AdamW(param_groups, lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1.e-8, fused=True)
    if cfg['exp_lr_schedule']:
        gamma = cfg['lr_final_factor'] ** (1.0 / max(1, num_training_steps))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: 1.)

    def save_model(suffix):
        torch.save({
            'spiral_and_transform': spiral_and_transform.state_dict(),
            'optimiser': optimiser.state_dict(),
            'cfg': dict(cfg),
            'z_begin': z_begin,
            'z_end': z_end,
        }, f'{out_path}/checkpoint_{suffix}.ckpt')

    def load_model(checkpoint):
        transformed_spiral_state, optimiser_state = checkpoint['spiral_and_transform'], checkpoint['optimiser']
        spiral_and_transform.load_state_dict(transformed_spiral_state)
        optimiser.load_state_dict(optimiser_state)

    if resume_path:
        print(f'resuming from {resume_path} at iteration {start_iteration}')
        load_model(resume_checkpoint)
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
        if tracks:
            # Free the patch/pcl eval tensors before the (much larger) track eval,
            # and chunk the track eval so the full track set does not have to be
            # materialised on the GPU at once. Guard against OOM so that a failure
            # to compute the secondary track metric never prevents the mesh/overlay
            # from being saved.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                track_satisfied_counts, track_total_counts = get_track_satisfied_counts_in_chunks(
                    slice_to_spiral_transform, dr_per_winding, tracks, metrics_config,
                )
                track_fully_satisfied = (track_satisfied_counts == track_total_counts)
                fully_satisfied_tracks = int(track_fully_satisfied.sum().item())
                num_valid_tracks = int(track_total_counts.numel())
                fully_satisfied_track_ratio = fully_satisfied_tracks / max(num_valid_tracks, 1)
                print(f'satisfied_tracks = {fully_satisfied_tracks}/{num_valid_tracks} ({fully_satisfied_track_ratio * 100:.1f}%)')
                track_satisfied_points = int(track_satisfied_counts.sum().item())
                track_total_points = int(track_total_counts.sum().item())
                track_satisfied_point_ratio = track_satisfied_points / max(track_total_points, 1)
                print(f'satisfied_track_points = {track_satisfied_points}/{track_total_points} ({track_satisfied_point_ratio * 100:.1f}%)')
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print('WARNING: skipped satisfied_tracks metric (CUDA OOM during track evaluation)')
        # Unverified patches are reported entirely separately so they never inflate the verified
        # satisfaction numbers.
        unverified_patch_satisfaction_entries = []
        if unverified_patches_list:
            u_satisfied, u_sat_areas, u_tot_areas, _, _, _ = get_patch_satisfied_areas(
                slice_to_spiral_transform, dr_per_winding, unverified_patches_list, verbose=False,
            )
            u_count = int(u_satisfied.sum().item())
            u_total = u_satisfied.numel()
            u_ratio = u_count / max(u_total, 1)
            print(f'unverified_satisfied_patches = {u_count}/{u_total} ({u_ratio * 100:.1f}%)')
            u_sat_area = float(u_sat_areas.sum().item())
            u_tot_area = float(u_tot_areas.sum().item())
            u_area_ratio = u_sat_area / max(u_tot_area, 1e-9)
            print(f'unverified_satisfied_area = {u_sat_area:.1f}/{u_tot_area:.1f} ({u_area_ratio * 100:.1f}%)')
            for pid, sat_area_t, tot_area_t in zip(unverified_patches_dict.keys(), u_sat_areas.tolist(), u_tot_areas.tolist()):
                fraction = sat_area_t / tot_area_t if tot_area_t > 0 else 0.0
                unverified_patch_satisfaction_entries.append({
                    'id': pid,
                    'satisfied_area': sat_area_t,
                    'total_area': tot_area_t,
                    'fraction': fraction,
                })
            unverified_patch_satisfaction_entries.sort(key=lambda e: e['fraction'])

        patch_ids = list(patches_dict.keys())
        patch_satisfaction_entries = []
        for pid, sat_area_t, tot_area_t in zip(patch_ids, satisfied_areas.tolist(), total_areas.tolist()):
            fraction = sat_area_t / tot_area_t if tot_area_t > 0 else 0.0
            patch_satisfaction_entries.append({
                'id': pid,
                'satisfied_area': sat_area_t,
                'total_area': tot_area_t,
                'fraction': fraction,
            })
        patch_satisfaction_entries.sort(key=lambda e: e['fraction'])
        pcl_satisfaction_entries = []
        if unattached_pcl_strips:
            sat_counts = unattached_pcl_satisfied_counts.tolist()
            tot_counts = unattached_pcl_total_counts.tolist()
            for strip, sc, tc in zip(unattached_pcl_strips, sat_counts, tot_counts):
                fraction = sc / tc if tc > 0 else 0.0
                pcl_satisfaction_entries.append({
                    'id': strip.get('id'),
                    'name': strip.get('name'),
                    'source_file': strip.get('source_file'),
                    'satisfied_points': int(sc),
                    'total_points': int(tc),
                    'fraction': fraction,
                })
            pcl_satisfaction_entries.sort(key=lambda e: e['fraction'])
        with open(f'{out_path}/satisfied_{suffix}.json', 'w') as f:
            json.dump({
                'patches': patch_satisfaction_entries,
                'pcls': pcl_satisfaction_entries,
                'unverified_patches': unverified_patch_satisfaction_entries,
            }, f, indent=2)
        # Flatten per-patch (H-1, W-1) masks in patch order to match the rasteriser's quad-id offsets,
        # then combine with patch-level overall satisfaction into a 0/1/2 status per quad.
        if satisfied_quad_masks:
            satisfied_quads_flat = torch.cat([m.flatten() for m in satisfied_quad_masks])
            quads_per_patch = torch.tensor([m.numel() for m in satisfied_quad_masks], dtype=torch.int64)
            overall_satisfied_per_quad = satisfied_patches.to(torch.bool).repeat_interleave(quads_per_patch)
        else:
            satisfied_quads_flat = torch.zeros([0], dtype=torch.bool)
            overall_satisfied_per_quad = torch.zeros([0], dtype=torch.bool)
        quad_status_flat = torch.where(
            overall_satisfied_per_quad,
            torch.full_like(satisfied_quads_flat, 2, dtype=torch.int64),
            satisfied_quads_flat.to(torch.int64),
        )
        if os.environ.get('FIT_SPIRAL_SKIP_SAVE_OVERLAY') != '1':
            winding_range, patch_extents, pcl_extents = compute_winding_range_and_input_extents(
                slice_to_spiral_transform, dr_per_winding, patches_list, unattached_pcl_strips,
                cfg, z_begin, z_end, get_or_build_unattached_pcl_flat,
            )
            _warn_if_inputs_exceed_flow_bounds(
                list(patches_dict.keys()), patch_extents,
                unattached_pcl_strips, pcl_extents,
                flow_field_radius,
                cfg,
            )
            save_overlay(
                spiral_and_transform,
                flow_min_corner_spiral_zyx, flow_max_corner_spiral_zyx,
                zs_for_visualisation, all_zs, slice_yx,
                scroll_slices_for_visualisation, prediction_slices_for_visualisation,
                quad_label_map, quad_status_flat,
                unattached_pcl_strips, unattached_pcl_per_point_satisfied, unattached_pcl_fully_satisfied,
                umbilicus_zyx, z_to_umbilicus_yx,
                winding_range,
                tracks,
                out_path, suffix
            )
        if os.environ.get('FIT_SPIRAL_SKIP_SAVE_MESH') != '1':
            save_mesh(
                slice_to_spiral_transform, dr_per_winding, patches_list, unattached_pcl_strips,
                out_path, cfg, z_begin, z_end, downsample_factor, voxel_size_um,
                get_or_build_unattached_pcl_flat, get_patch_satisfied_areas,
                run_tag=run_tag, name=suffix,
            )

    prepared_main_tracks = None
    if (cfg['loss_weight_track_radius'] > 0 or cfg['loss_weight_track_dt'] > 0) and tracks:
        trusted_anchor_scroll = _collect_anchor_scroll_zyxs(patch_atlas, unattached_pcl_strips, device)
        prepared_main_tracks = prepare_main_phase_tracks(
            tracks, trusted_anchor_scroll,
            float(cfg['track_exclusion_radius']), device,
        )
    slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()
    dr_per_winding = spiral_and_transform.get_dr_per_winding()

    for iteration in tqdm(range(start_iteration, num_training_steps)):

        spiral_and_transform.flow_field.flow_scales[1] = get_flow_field_high_res_lr_scale(iteration)

        slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()
        dr_per_winding = spiral_and_transform.get_dr_per_winding()

        losses = {}
        log_metrics = {}
        log_metrics['flow_field_high_res_lr_scale'] = spiral_and_transform.flow_field.flow_scales[1]

        compute_patch_dt = iteration > cfg['loss_start_patch_dt']
        track_dt_start = cfg['loss_start_patch_dt'] if cfg['loss_start_track_dt'] is None else cfg['loss_start_track_dt']
        compute_track_dt = iteration > track_dt_start
        unverified_patch_dt_start = cfg['loss_start_patch_dt'] if cfg['loss_start_unverified_patch_dt'] is None else cfg['loss_start_unverified_patch_dt']
        compute_unverified_patch_dt = iteration > unverified_patch_dt_start

        # Progressive-outward DT gating: winding cutoff that grows from the respective DT start
        # step. Falls back to the configured shell_outer_winding_idx when shell losses are off so
        # the feature still works; None => no gating.
        dt_progressive_outer = shell_outer_winding_idx if shell_outer_winding_idx is not None else cfg['shell_outer_winding_idx']
        patch_dt_max_winding = get_progressive_dt_max_winding(iteration, cfg['loss_start_patch_dt'], dt_progressive_outer)
        track_dt_max_winding = get_progressive_dt_max_winding(iteration, track_dt_start, dt_progressive_outer)
        unverified_patch_dt_max_winding = get_progressive_dt_max_winding(iteration, unverified_patch_dt_start, dt_progressive_outer)
        if patch_dt_max_winding is not None:
            log_metrics['patch_dt_max_winding'] = patch_dt_max_winding
        if track_dt_max_winding is not None:
            log_metrics['track_dt_max_winding'] = track_dt_max_winding

        patch_radius_loss, umbilicus_loss, patch_dt_loss, shell_patch_radius_loss = get_patch_and_umbilicus_losses(
            slice_to_spiral_transform,
            dr_per_winding,
            cfg['num_patches_per_step'],
            cfg['num_patches_per_step_for_dt'],
            patches_list,
            patch_atlas,
            patch_sampling_probabilities,
            umbilicus_zyx,
            compute_dt=compute_patch_dt,
            shell_valid_zyxs=shell_valid_zyxs_gpu,
            shell_outer_winding_idx=shell_outer_winding_idx,
            dt_max_winding=patch_dt_max_winding,
        )
        losses['patch_radius'] = patch_radius_loss * cfg['loss_weight_patch_radius']
        losses['patch_dt'] = patch_dt_loss * cfg['loss_weight_patch_dt']
        if shell_valid_zyxs_gpu is not None:
            losses['shell_patch_radius'] = shell_patch_radius_loss * cfg['loss_weight_shell_patch_radius']

        if unverified_patch_atlas is not None and (cfg['loss_weight_unverified_patch_radius'] > 0 or cfg['loss_weight_unverified_patch_dt'] > 0):
            unverified_patch_radius_loss, unverified_patch_dt_loss = get_unverified_patch_losses(
                slice_to_spiral_transform,
                dr_per_winding,
                cfg['unverified_num_patches_per_step'],
                cfg['unverified_num_patches_per_step_for_dt'],
                unverified_patches_list,
                unverified_patch_atlas,
                unverified_patch_sampling_probabilities,
                compute_dt=compute_unverified_patch_dt,
                dt_max_winding=unverified_patch_dt_max_winding,
            )
            losses['unverified_patch_radius'] = unverified_patch_radius_loss * cfg['loss_weight_unverified_patch_radius']
            losses['unverified_patch_dt'] = unverified_patch_dt_loss * cfg['loss_weight_unverified_patch_dt']

        if cfg['loss_weight_sym_dirichlet'] > 0:
            sym_dirichlet_loss = get_symmetric_dirichlet_loss(
                slice_to_spiral_transform,
                dr_per_winding,
                shell_outer_winding_idx,
                cfg['regularisation_num_points'],
            )
            losses['sym_dirichlet'] = sym_dirichlet_loss * cfg['loss_weight_sym_dirichlet']

        if cfg['loss_weight_rel_winding'] > 0 and point_collections:
            losses['rel_winding'] = get_patch_rel_winding_loss(slice_to_spiral_transform, dr_per_winding, patches_dict, patch_atlas, point_collections) * cfg['loss_weight_rel_winding']

        if cfg['loss_weight_abs_winding'] > 0 and point_collections:
            losses['abs_winding'] = get_patch_abs_winding_loss(slice_to_spiral_transform, dr_per_winding, patches_dict, patch_atlas, point_collections) * cfg['loss_weight_abs_winding']

        if (cfg['loss_weight_dense_normals'] > 0 or cfg['loss_weight_dense_spacing'] > 0) and lasagna_volume is not None:
            dense_normals_loss, dense_spacing_loss = get_lasagna_losses(
                slice_to_spiral_transform,
                dr_per_winding,
                lasagna_volume,
                shell_outer_winding_idx,
                cfg['dense_normals_num_points'],
            )
            losses['dense_normals'] = dense_normals_loss * cfg['loss_weight_dense_normals']
            losses['dense_spacing'] = dense_spacing_loss * cfg['loss_weight_dense_spacing']

        if (cfg['loss_weight_unattached_pcl_radius'] > 0 or cfg['loss_weight_unattached_pcl_dt'] > 0) and unattached_pcl_strips:
            unattached_pcl_radius_loss, unattached_pcl_dt_loss = get_unattached_pcl_strip_losses(
                slice_to_spiral_transform,
                dr_per_winding,
                unattached_pcl_strips,
                cfg['unattached_pcl_num_per_step'],
                cfg['unattached_pcl_num_points_per_step'],
                compute_dt=compute_patch_dt,
                dt_max_winding=patch_dt_max_winding,
            )
            losses['unattached_pcl_radius'] = unattached_pcl_radius_loss * cfg['loss_weight_unattached_pcl_radius']
            losses['unattached_pcl_dt'] = unattached_pcl_dt_loss * cfg['loss_weight_unattached_pcl_dt']

        if prepared_main_tracks is not None:
            track_radius_loss, track_dt_loss = get_track_losses(
                slice_to_spiral_transform,
                dr_per_winding,
                prepared_main_tracks,
                cfg,
                compute_dt=compute_track_dt,
                dt_max_winding=track_dt_max_winding,
            )
            losses['track_radius'] = track_radius_loss * cfg['loss_weight_track_radius']
            losses['track_dt'] = track_dt_loss * cfg['loss_weight_track_dt']

        shell_metrics = {}
        if shell_map is not None:
            shell_outer_loss, shell_metrics = get_shell_outer_loss(
                shell_map,
                slice_to_spiral_transform,
                dr_per_winding,
                shell_outer_winding_idx,
            )
            losses['shell_outer'] = shell_outer_loss * cfg['loss_weight_shell_outer']

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
                **shell_metrics,
                **log_metrics,
            })

    suffix = 'fitted'
    save_model(suffix)
    save_overlay_and_print_satisfaction(suffix)



def link_points_to_patches_cached(
    patches,
    point_collections,
    tolerance=10.0,
    surface_index_tolerance=None,
    distance_scale=1.0,
):
    patch_ids_str = '_'.join(sorted(patches.keys()))
    hashed_patches = hashlib.sha256(bytes(patch_ids_str, 'ascii')).hexdigest()[:8]

    backend = 'surface-index' if surface_index_tolerance is not None and can_use_surface_index_backend(patches) else 'torch'
    # Per-point cache: key by patch set + linking config only (NOT the pcl set and NOT
    # annotations). Changing the --pcl set, editing annotations, or moving a few points
    # only links the points whose POSITION is new; everything else is reused, and a
    # re-run with no new positions does no linking work (the surface index is never
    # rebuilt). The pcl-content hash is intentionally dropped from the filename.
    cache_filename = (
        f'{cache_path}/point_links-{hashed_patches}'
        f'_backend-{backend}_tol-{tolerance}_idx-tol-{surface_index_tolerance}_dscale-{distance_scale}.pkl'
    )

    point_link_cache = {}
    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as fp:
            point_link_cache = pickle.load(fp)
    n_cached_before = len(point_link_cache)

    link_points_to_patches_pointcache(
        patches,
        point_collections,
        tolerance,
        surface_index_tolerance=surface_index_tolerance,
        distance_scale=distance_scale,
        cache=point_link_cache,
    )

    if len(point_link_cache) != n_cached_before:
        os.makedirs(cache_path, exist_ok=True)
        with open(cache_filename, 'wb') as fp:
            pickle.dump(point_link_cache, fp)


def prepare_point_collections(patches):
    # Load all pcls, transform into (downsampled) voxel space, link every point to patches,
    # and split into cross-patch / unattached sets. `patches` must already be downsampled
    # and filtered to the z-roi.

    point_collections = {}
    next_id = 0
    for pattern in pcl_json_paths:
        expanded = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        for path in expanded:
            loaded = load_point_collection(path) or {}
            for pcl in loaded.values():
                pcl['source_file'] = path
                point_collections[next_id] = pcl
                next_id += 1

    for pcl in point_collections.values():
        for point in pcl['points'].values():
            point['zyx'] = np.array([point['p'][2], point['p'][1], point['p'][0]]) / downsample_factor

    def pcl_intersects_z_roi(pcl):
        for point in pcl['points'].values():
            z = point['zyx'][0]
            if z_begin <= z < z_end:
                return True
        return False

    # Drop pcls lying entirely outside the z-roi before linking points to patches.
    dropped_pcl_ids = [pid for pid, pcl in point_collections.items() if not pcl_intersects_z_roi(pcl)]
    for pid in dropped_pcl_ids:
        del point_collections[pid]
    print(f'dropped {len(dropped_pcl_ids)} pcls outside z-roi [{z_begin}, {z_end})')

    # Link every point of every pcl to patches (adds 'on_patch' to attached points).
    link_points_to_patches_cached(
        patches,
        point_collections,
        tolerance=10.0 / downsample_factor,
        surface_index_tolerance=10.0 / downsample_factor,
        distance_scale=1.0,
    )

    # Classify each pcl from how its points attach to patches:
    #  - >= 2 attached points => acts as a cross-patch pcl (winding-number loss), using only
    #    its attached points (grouped by patch below);
    #  - >= 1 unattached point => acts as an unattached pcl (unattached loss), using the
    #    entire pcl.
    # A pcl can fall into both sets. When it does, the unattached entry is an independent copy
    # so its z-roi trimming / annotation normalisation cannot perturb the cross-patch entry's
    # points_by_patch (which is built from all attached points, regardless of z).
    # Exception: pcls flagged metadata.winding_is_absolute carry absolute winding annotations
    # and are always consumed as cross-patch pcls (never unattached), retained even when they
    # hold a single point. We only *warn* on any of their points that failed to attach to a
    # patch -- those points carry no winding target and are simply dropped (they never enter
    # points_by_patch) -- and assert that every *attached* point carries an explicit, positive
    # winding annotation (an absolute pcl must not fall back to winding 0), and (once grouped
    # below) that no patch holds more than one of their points.
    cross_patch_point_collections = {}
    unattached_point_collections = {}
    for pid, pcl in point_collections.items():
        num_attached = sum(1 for point in pcl['points'].values() if 'on_patch' in point)
        num_unattached = len(pcl['points']) - num_attached
        if pcl.get('metadata', {}).get('winding_is_absolute', False):
            if num_unattached > 0:
                print(
                    f'WARNING: winding_is_absolute pcl {pid} ({pcl.get("name")!r}) has '
                    f'{num_unattached} of {len(pcl["points"])} points not attached to any patch; '
                    f'dropping the unattached points'
                )
            # Validate only the attached points -- unattached ones are dropped above and never
            # enter points_by_patch, so their annotations are irrelevant.
            attached_points = [point for point in pcl['points'].values() if 'on_patch' in point]
            num_unannotated = sum(1 for point in attached_points if not np.isfinite(point['winding_annotation']))
            assert num_unannotated == 0, (
                f'winding_is_absolute pcl {pid} ({pcl.get("name")!r}) has {num_unannotated} of '
                f'{len(attached_points)} attached points without a winding annotation; absolute pcls '
                f'must give every winding number explicitly'
            )
            num_non_positive = sum(1 for point in attached_points if point['winding_annotation'] <= 0)
            assert num_non_positive == 0, (
                f'winding_is_absolute pcl {pid} ({pcl.get("name")!r}) has {num_non_positive} of '
                f'{len(attached_points)} attached points with a non-positive winding annotation; '
                f'absolute winding numbers must be > 0'
            )
            cross_patch_point_collections[pid] = pcl
            continue
        if num_attached >= 2:
            cross_patch_point_collections[pid] = pcl
        if num_unattached >= 1:
            unattached_point_collections[pid] = copy.deepcopy(pcl) if num_attached >= 2 else pcl

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
    if dropped_unattached_pcl_count:
        print(f'dropped {dropped_unattached_pcl_count} unattached pcls with <2 points in z-roi')

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
    unattached_pcl_strips = _prepare_unattached_pcl_strips(unattached_point_collections, cfg['unattached_pcl_min_point_spacing'])
    print(
        f'pcls: {len(cross_patch_point_collections)} cross-patch, '
        f'{len(unattached_pcl_strips)} unattached'
    )

    return cross_patch_point_collections, unattached_pcl_strips



def prepare_patches():

    def scale_patch(patch):
        patch.scale *= downsample_factor
        patch.zyxs /= downsample_factor
        patch.valid_zyxs /= downsample_factor
        patch.area /= downsample_factor ** 2

    def patch_intersects_z_roi(patch):
        zs = patch.valid_zyxs[..., 0]
        if zs.numel() == 0:
            return False
        return bool(((zs >= z_begin) & (zs < z_end)).any().item())

    def load_scale_and_roi_filter(path, label):
        loaded = load_patches(path)
        for patch in loaded.values():
            scale_patch(patch)
        erode_cells = int(cfg['erode_patches'])
        if erode_cells > 0:
            fully_eroded = [pid for pid, patch in loaded.items() if not erode_patch_valid_region(patch, erode_cells)]
            for pid in fully_eroded:
                del loaded[pid]
            print(f'eroded {label} patches by {erode_cells} cells; dropped {len(fully_eroded)} fully-eroded patches')
        dropped = [pid for pid, patch in loaded.items() if not patch_intersects_z_roi(patch)]
        for pid in dropped:
            del loaded[pid]
        print(f'dropped {len(dropped)} {label} patches outside z-roi [{z_begin}, {z_end})')
        return loaded

    patches = load_scale_and_roi_filter(patches_path, 'verified')

    unverified_patches = {}
    if unverified_patches_path is not None:
        unverified_patches = load_scale_and_roi_filter(unverified_patches_path, 'unverified')

    shell_patch = None
    if shell_losses_enabled():
        if not shell_path:
            raise RuntimeError('shell losses are enabled, but FIT_SPIRAL_SHELL_PATH is not set')
        shell_patch = load_tifxyz(shell_path)
        scale_patch(shell_patch)
        print(f'loaded shell from {shell_path}: {shell_patch.valid_zyxs.shape[0]} valid points')

    return patches, unverified_patches, shell_patch


def main():

    np.random.seed(cfg['random_seed'])
    torch.random.manual_seed(cfg['random_seed'])

    umbilicus = umbilicus_z_to_yx(downsample_factor)

    if scroll_zarr_path:
        print('loading volume zarr')
        scroll_zarr_array = zarr.open(scroll_zarr_path, mode='r')
    else:
        scroll_zarr_array = None

    patches, unverified_patches, shell_patch = prepare_patches()
    cross_patch_point_collections, unattached_pcl_strips = prepare_point_collections(patches)
    lasagna_volume = prepare_lasagna_volume(
        scroll_zarr_array,
        use_normals=cfg['loss_weight_dense_normals'] > 0,
        use_spacing=cfg['loss_weight_dense_spacing'] > 0,
        normal_nx_zarr_path=normal_nx_zarr_path,
        normal_ny_zarr_path=normal_ny_zarr_path,
        grad_mag_zarr_path=grad_mag_zarr_path,
        normal_zarr_group=normal_zarr_group,
        z_begin=z_begin,
        z_end=z_end,
        downsample_factor=downsample_factor,
    )

    if tracks_dbm_path is not None:
        print(f'loading tracks from {tracks_dbm_path}')
        tracks = load_tracks_from_dbm(tracks_dbm_path, z_begin, z_end, downsample_factor)
        print(f'loaded {len(tracks)} tracks within z-roi [{z_begin}, {z_end})')
    else:
        tracks = None

    out_base_dir = os.environ.get('FIT_SPIRAL_OUT_DIR', './out')
    out_path = f'{out_base_dir}/{datetime.date.today()}_{scroll_name}_slice-{z_begin * downsample_factor}-{z_end * downsample_factor}_{len(patches)}-patch'
    if not wandb.run.name.startswith('dummy-'):
        out_path += '_' + wandb.run.name
    if run_tag:
        out_path += f'_{run_tag}'
    os.makedirs(out_path, exist_ok=True)

    fit_spiral_3d(
        scroll_zarr_array,
        patches,
        list(cross_patch_point_collections.values()),
        unattached_pcl_strips,
        tracks,
        lasagna_volume,
        shell_patch,
        umbilicus,
        out_path,
        unverified_patches_dict=unverified_patches,
    )


if __name__ == '__main__':
    config = dict(default_config)
    config.update(get_env_config_overrides())
    z_range_scale, z_range_num_slices = scale_counts_for_z_range(
        config, z_begin, z_end, downsample_factor,
        reference_z_range_num_slices, z_range_scaled_count_keys,
    )
    print(
        f'scaled per-step counts by {z_range_scale:.3f} for the {z_range_num_slices}-slice '
        f'z-range [{z_begin * downsample_factor}, {z_end * downsample_factor}) '
        f'(reference {reference_z_range_num_slices} slices):\n  '
        + '\n  '.join(f'{k}={config[k]}' for k in z_range_scaled_count_keys)
    )
    wandb.init(project='scrolls', config=config)
    cfg = wandb.config
    main()
