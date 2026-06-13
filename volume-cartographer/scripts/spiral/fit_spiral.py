
import os
import dbm
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
import colorsys
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

from tifxyz import load_tifxyz, save_tifxyz
from geom_utils import interp1d
from point_collection import load_point_collection, link_points_to_patches, can_use_surface_index_backend, normalise_pcl_winding_annotations
from umbilicus import thaumato_umbilicus_z_to_yx, json_umbilicus_z_to_yx


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
    'track_loss_use_em': False,
    'track_em_start_step': 3000,
    'track_em_reassign_interval': 500,
    'track_em_assignment': 'mode',  # 'mode' or 'median'
    'track_em_min_confidence': 0.5,
    'track_em_unassigned_radius_weight': 0.0,
    'track_abs_radius_aggregation': 'hinge_mean',  # 'hinge_mean' or 'coverage'
    'track_coverage_reduce': 'softmin',  # 'softmin' or 'mean'
    'track_coverage_radius_tol': 0.5,
    'track_coverage_tau_init': 0.5,
    'track_coverage_tau_final': 0.05,
    'normals_num_points': 6000,
    'pcl_normals_num_points': 12000,
    'pcl_normals_sample_radius': 1,
    'dense_normals_num_points': 60_000,
    'regularisation_num_points': 4500,
    'grad_mag_encode_scale': 1000.0,
    'grad_mag_factor': 0.25 * 4,  # 4 maps from 2um to 8um
    'spacing_integration_steps': 8,
    'patch_stretch_loss_norm': 'L2',
    'loss_weight_patch_radius': 32.e0,
    'loss_weight_uv_distance': 0.,
    'loss_weight_patch_dt': 16.e0,
    'loss_weight_unverified_patch_radius': 8.e0,
    'loss_weight_unverified_patch_dt': 4.e0,
    'loss_weight_rel_winding': 20.,
    'loss_weight_abs_winding': 20.,
    'loss_weight_unattached_pcl_radius': 8.e0,
    'loss_weight_unattached_pcl_dt': 16.e0,
    'loss_weight_track_radius': 200.,
    'loss_weight_track_dt': 40.,
    'loss_weight_track_coverage': 0.0,
    'loss_weight_patch_stretch': 0.0,
    'loss_weight_bending': 0.0,
    'loss_weight_sym_dirichlet': 10.0,
    'loss_weight_patch_normals': 0.0,
    'loss_weight_pcl_normals': 0.0,
    'loss_weight_dense_normals': 1.e2,
    'loss_weight_dense_spacing': 8.,
    'loss_weight_umbilicus': 5.,
    'loss_weight_shell_outer': 4.0,
    'loss_weight_shell_no_cross': 0.0,
    'loss_weight_shell_z_drift': 0.0,
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
    'num_snapping_subpasses': 0,
    'snapping_steps_per_subpass': 500,
    'snapping_num_anchor_points': 2000,
    'snapping_num_track_points': 2000,
    'snapping_tracks_lower_fraction': 0.5,
    'snapping_tracks_upper_fraction': 0.95,
    'snapping_track_exclusion_radius': 8.0,
    'snapping_max_lr': 2.e-5,
    'loss_weight_snap_anchor': 100.0,
    'loss_weight_snap_tracks': 20.0,
    'output_first_winding': 10,
    'output_winding_margin': 4,
    'output_step_size': 20,
    'shell_outer_winding_idx': 130,
    'shell_outer_winding_margin': 10,
    'shell_num_samples': 24576,
    'shell_num_theta_bins': 720,
    'shell_near_outer_num_windings': 3,
    'shell_huber_delta': 4.0,
    'shell_no_cross_margin': 2.0,
    'shell_table_smooth_sigma_z': 1.0,
    'shell_table_smooth_sigma_theta': 1.0,
    'shell_min_confidence': 0.25,
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
    'normals_num_points',
    'pcl_normals_num_points',
    'dense_normals_num_points',
    'regularisation_num_points',
    'shell_num_samples',
)


def scale_counts_for_z_range(config):
    # z_begin/z_end have already been divided by downsample_factor; multiply back so
    # the scale is expressed in (and matches) un-downsampled slice counts.
    num_slices = (z_end - z_begin) * downsample_factor
    scale = num_slices / reference_z_range_num_slices
    for key in z_range_scaled_count_keys:
        config[key] = max(1, round(config[key] * scale))
    return scale, num_slices


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


def get_spiral_density(relative_yx, dr_per_winding=10., sigma=3., winding_range=None):

    if winding_range is None:
        min_w, max_w = cfg['output_first_winding'], float('inf')
    else:
        min_w, max_w = winding_range
    theta, radius, inner_winding, outer_winding = get_bounding_windings(relative_yx, dr_per_winding)
    def evaluate_kernel(winding_idx):
        winding_xy = get_winding_xy(winding_idx, theta, dr_per_winding)
        distance = torch.linalg.norm(winding_xy.flip(-1) - relative_yx, dim=-1)
        kernel = torch.exp(-distance ** 2 / sigma ** 2)
        kernel = torch.where((winding_idx >= min_w) & (winding_idx < max_w), kernel, torch.zeros_like(kernel))
        return kernel
    result = evaluate_kernel(inner_winding) + evaluate_kernel(outer_winding)
    return result.clip(0., 1.)


def shell_losses_enabled():
    return (
        cfg['loss_weight_shell_outer'] > 0
        or cfg['loss_weight_shell_no_cross'] > 0
        or cfg['loss_weight_shell_z_drift'] > 0
        or cfg['loss_weight_shell_patch_radius'] > 0
    )


def _huber_abs(residual, delta):
    abs_residual = residual.abs()
    return torch.where(
        abs_residual <= delta,
        0.5 * residual ** 2 / delta,
        abs_residual - 0.5 * delta,
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


def _canonical_winding_samples(winding_indices, num_samples, dr_per_winding, device):
    winding_indices_t = torch.as_tensor(winding_indices, device=device, dtype=torch.float32)
    theta = torch.rand([len(winding_indices), num_samples], device=device) * (2 * torch.pi)
    z = torch.empty([len(winding_indices), num_samples], device=device).uniform_(float(z_begin), float(z_end - 1))
    radius = (winding_indices_t[:, None] + theta / (2 * torch.pi)) * dr_per_winding
    return torch.stack([
        z,
        torch.sin(theta) * radius,
        torch.cos(theta) * radius,
    ], dim=-1)


def get_shell_losses(shell_map, slice_to_spiral_transform, dr_per_winding, outer_winding_idx):
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if shell_map is None or outer_winding_idx is None:
        return zero, zero, zero, {}

    num_samples = max(1, int(cfg['shell_num_samples']))
    huber_delta = torch.as_tensor(cfg['shell_huber_delta'], device=device, dtype=torch.float32)

    outer_spiral = _canonical_winding_samples([outer_winding_idx], num_samples, dr_per_winding, device)[0]
    near_count = max(1, int(cfg['shell_near_outer_num_windings']))
    first_w = max(0, int(outer_winding_idx) - near_count + 1)
    winding_indices = list(range(first_w, int(outer_winding_idx) + 1))
    barrier_spiral = _canonical_winding_samples(winding_indices, max(1, num_samples // len(winding_indices)), dr_per_winding, device)

    outer_count = outer_spiral.shape[0]
    combined_scan = slice_to_spiral_transform.inv(torch.cat([
        outer_spiral,
        barrier_spiral.reshape(-1, 3),
    ], dim=0))
    outer_scan = combined_scan[:outer_count]
    barrier_scan = combined_scan[outer_count:].reshape(*barrier_spiral.shape)

    target_r, scan_r, confidence, valid = shell_map.lookup(outer_scan)
    residual = scan_r - target_r
    shell_outer_loss = _masked_mean(_huber_abs(residual, huber_delta), valid)
    shell_z_drift_loss = _masked_mean(_huber_abs(outer_scan[..., 0] - outer_spiral[..., 0], huber_delta), valid)

    barrier_target_r, barrier_scan_r, _, barrier_valid = shell_map.lookup(barrier_scan)
    violation = F.relu(barrier_scan_r - barrier_target_r - cfg['shell_no_cross_margin'])
    shell_no_cross_loss = _masked_mean(_huber_abs(violation, huber_delta), barrier_valid)

    metrics = {}
    with torch.no_grad():
        if valid.any():
            abs_residual = residual[valid].abs()
            metrics = {
                'shell_outer_error_mean': abs_residual.mean(),
                'shell_outer_error_p95': torch.quantile(abs_residual, 0.95),
                'shell_z_drift_mean': (outer_scan[..., 0] - outer_spiral[..., 0]).abs()[valid].mean(),
                'shell_confidence_mean': confidence[valid].mean(),
            }
        if barrier_valid.any():
            metrics['shell_no_cross_violation_p95'] = torch.quantile(violation[barrier_valid], 0.95)
            metrics['shell_no_cross_violation_fraction'] = (violation[barrier_valid] > 0).to(torch.float32).mean()

    return shell_outer_loss, shell_no_cross_loss, shell_z_drift_loss, metrics


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
        # Sparse-gradient sampling state (see get_sampler / _SparseAccumTrilinearSample):
        # the most recent grad-enabled field materialisation, and the shared buffer its
        # samplers' backward passes accumulate the field gradient into.
        self._pending_field = None
        self._field_grad_acc = None

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
            if not field.requires_grad:
                return lambda y: sample_field(y, field)
            # Training path: sample a detached view through _SparseAccumTrilinearSample so
            # each backward node costs O(points) instead of O(field). The field gradient
            # lands in _field_grad_acc and is chained into the flow parameters once per
            # step by apply_accumulated_field_grad(). Only valid time-invariant: with
            # multiple timesteps each sampler would materialise a different field, and the
            # single pending graph could not chain them all.
            if self._field_grad_acc is None or self._field_grad_acc.shape != field.shape:
                self._field_grad_acc = torch.zeros_like(field)
            else:
                self._field_grad_acc.zero_()
            self._pending_field = field
            field_detached = field.detach()
            acc = self._field_grad_acc

            def sample(normalised_zyx):
                flat = normalised_zyx.reshape(-1, 3)
                return _SparseAccumTrilinearSample.apply(flat, field_detached, acc).view(*normalised_zyx.shape[:-1], 3)

            return sample
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

    def apply_accumulated_field_grad(self):
        # Backpropagate the sparse-accumulated field gradient through the (linear)
        # LR-upsample + HR-scale materialisation into the flow parameters. Call once per
        # optimisation step, after loss.backward() and before optimiser.step(); no-op if
        # no grad-enabled sampler was built since the last call.
        if self._pending_field is not None:
            self._pending_field.backward(gradient=self._field_grad_acc)
            self._pending_field = None


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

    def __init__(self, resolution, spatial_scale_factor=6, lr_scale_factor=1.e-1):
        # resolution is interpreted as the equivalent cartesian (Z, Y, X) voxel shape; the
        # cylindrical lattice sizes are derived from it.
        super().__init__()
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
            nn.Parameter(torch.zeros([cfg['num_flow_timesteps'], 3, nz_lr, int(lr_offsets[-1])])),
            nn.Parameter(torch.zeros([cfg['num_flow_timesteps'], 3, nz_hr, int(hr_offsets[-1])])),
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
        if cfg['num_flow_timesteps'] == 1:
            lr_field = self.flows[0][0]
            hr_field = self.flows[1][0]
        else:
            t_scaled = (t.clamp(-1. + 1.e-4, 1. - 1.e-4) + 1) / 2 * (cfg['num_flow_timesteps'] - 1)
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

    def apply_accumulated_field_grad(self):
        # The cylindrical lattice is sampled directly via gathers on its parameters (no
        # materialised cartesian field), so there is nothing to chain; this exists to
        # mirror CartesianFlowField's interface for the training loop.
        pass


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


class _SparseAccumTrilinearSample(torch.autograd.Function):
    # Trilinear sampling of a (zyx, z, y, x) field at normalised [0, 1] zyx points,
    # matching sample_field / F.grid_sample(align_corners=True, padding_mode='border')
    # exactly, but with the field gradient scattered sparsely (8 index_add_ calls of
    # O(num_points)) into a shared dense accumulation buffer instead of autograd's
    # fresh field-sized dense gradient per backward node. Each RK4 integration makes
    # 12 sampler calls and a training step makes ~14 transform invocations, so the
    # per-node zero + accumulate over the ~300M-element field otherwise dominates the
    # whole step. The buffer is chained into the flow parameters once per step via
    # CartesianFlowField.apply_accumulated_field_grad().

    @staticmethod
    def _corners(pts, field):
        # Shared forward/backward index + weight setup. align_corners=True maps
        # [0, 1] -> [0, size - 1]; border padding clamps the coordinate.
        shape = torch.tensor(field.shape[1:], device=pts.device, dtype=pts.dtype)  # z, y, x
        coord_raw = pts * (shape - 1)
        coord = coord_raw.clamp(min=torch.zeros_like(shape), max=shape - 1)
        lo = coord.floor().clamp(max=shape - 2).to(torch.int64)
        frac = coord - lo.to(coord.dtype)
        Y, X = field.shape[2], field.shape[3]
        base = (lo[:, 0] * Y + lo[:, 1]) * X + lo[:, 2]
        return coord_raw, shape, frac, base, (Y * X, X, 1)

    @staticmethod
    def forward(ctx, pts, field, acc):
        # pts :: n, 3 normalised zyx in [0, 1]; field :: 3, z, y, x (detached); acc like field
        ctx.set_materialize_grads(False)
        _, _, frac, base, strides = _SparseAccumTrilinearSample._corners(pts, field)
        flat_field = field.reshape(3, -1)
        out = torch.zeros(pts.shape[0], 3, device=pts.device, dtype=pts.dtype)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = ((frac[:, 0] if dz else 1 - frac[:, 0])
                         * (frac[:, 1] if dy else 1 - frac[:, 1])
                         * (frac[:, 2] if dx else 1 - frac[:, 2]))
                    idx = base + dz * strides[0] + dy * strides[1] + dx * strides[2]
                    out += flat_field[:, idx].T * w[:, None]
        ctx.save_for_backward(pts, field)
        ctx.acc = acc  # mutated in backward, so it cannot go through save_for_backward
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None
        pts, field = ctx.saved_tensors
        coord_raw, shape, frac, base, strides = _SparseAccumTrilinearSample._corners(pts, field)
        flat_field = field.reshape(3, -1)
        acc_flat = ctx.acc.reshape(3, -1)
        grad_coord = torch.zeros_like(pts)
        f = [(1 - frac[:, 0], frac[:, 0]), (1 - frac[:, 1], frac[:, 1]), (1 - frac[:, 2], frac[:, 2])]
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = f[0][dz] * f[1][dy] * f[2][dx]
                    idx = base + dz * strides[0] + dy * strides[1] + dx * strides[2]
                    acc_flat.index_add_(1, idx, (grad_out * w[:, None]).T)
                    v_dot_g = (flat_field[:, idx].T * grad_out).sum(dim=-1)
                    grad_coord[:, 0] += v_dot_g * ((1.0 if dz else -1.0) * f[1][dy] * f[2][dx])
                    grad_coord[:, 1] += v_dot_g * ((1.0 if dy else -1.0) * f[0][dz] * f[2][dx])
                    grad_coord[:, 2] += v_dot_g * ((1.0 if dx else -1.0) * f[0][dz] * f[1][dy])
        # grid_sample's border padding passes zero gradient to clipped coordinates
        unclipped = (coord_raw >= 0) & (coord_raw <= shape - 1)
        return grad_coord * unclipped.to(grad_coord.dtype) * (shape - 1), None, None


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


def expm_2x2(L):
    # Closed-form matrix exponential for (..., 2, 2) matrices:
    #   exp(L) = e^m (cosh(s) I + sinh(s)/s (L - m I)),  m = tr(L)/2,  s^2 = ((a-d)/2)^2 + bc
    # (cos/sin when s^2 < 0). Exact, unlike torch.linalg.matrix_exp's Pade route, and avoids
    # that op's per-call host synchronisation, which otherwise stalls the CPU's ability to
    # queue ahead of the GPU on every transform invocation.
    # sqrt has an infinite derivative at 0, and torch.where's backward turns the resulting
    # 0 * inf into NaN, so the small-|s2| branch must compute on a safe substitute while the
    # series branch stays a polynomial in s2 (this matters at init, where L == 0 exactly).
    a, b = L[..., 0, 0], L[..., 0, 1]
    c, d = L[..., 1, 0], L[..., 1, 1]
    m = 0.5 * (a + d)
    s2 = (0.5 * (a - d)) ** 2 + b * c
    small = s2.abs() < 1e-8
    s = torch.where(small, torch.ones_like(s2), s2).abs().sqrt()
    pos = s2 >= 0
    cosh_term = torch.where(small, 1.0 + s2 / 2.0, torch.where(pos, torch.cosh(s), torch.cos(s)))
    sinc_term = torch.where(small, 1.0 + s2 / 6.0, torch.where(pos, torch.sinh(s), torch.sin(s)) / s)
    em = torch.exp(m)
    f_diag = em * cosh_term
    f_off = em * sinc_term
    out = torch.empty_like(L)
    out[..., 0, 0] = f_diag + f_off * (a - m)
    out[..., 0, 1] = f_off * b
    out[..., 1, 0] = f_off * c
    out[..., 1, 1] = f_diag + f_off * (d - m)
    return out


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
        flow_field_cls = {'cartesian': CartesianFlowField, 'cylindrical': CylindricalFlowField}[cfg['flow_field_type']]
        self.flow_field = flow_field_cls(flow_resolution, lr_scale_factor=cfg['flow_field_high_res_lr_scale_initial'])

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
    # regressing it onto winding_diff * dr_per_winding.

    num_points_per_strip = cfg['num_points_per_patch'] // 2
    num_strips_per_pcl = 4
    num_strips_per_pair = 2 * num_strips_per_pcl  # 8

    # Each entry: (ls1, ls2, pid1, pid2, winding_diff) where ls* is a list of 4 L-shape ij strips
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
    expected_diff = (winding_diffs * dr_per_winding)[:, None, None]

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
    # downstream metrics can run a single transform call plus segmented reductions
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


def _get_or_build_unattached_pcl_flat(pcl_strips, device):
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


def _sample_zarr_block_at_zyx(array, zyx, radius):
    center = np.rint(zyx).astype(np.int64)
    shape = np.array(array.shape, dtype=np.int64)
    if np.any(center < 0) or np.any(center >= shape):
        return None
    lo = np.maximum(center - radius, 0)
    hi = np.minimum(center + radius + 1, shape)
    block = array[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if block.size == 0:
        return None
    return block


def _decode_uint8_normal_component(value):
    return (value - 128.0) / 127.0


def _sample_zarr_normal_zyx_at_zyx(nx_array, ny_array, zyx, radius):
    nx_u8 = _sample_zarr_block_at_zyx(nx_array, zyx, radius)
    ny_u8 = _sample_zarr_block_at_zyx(ny_array, zyx, radius)
    if nx_u8 is None or ny_u8 is None:
        return None
    if nx_u8.shape != ny_u8.shape:
        raise ValueError(f'nx/ny normal sample shapes differ: {nx_u8.shape} vs {ny_u8.shape}')

    valid = (nx_u8 != 0) | (ny_u8 != 0)
    if not valid.any():
        return None

    nx = _decode_uint8_normal_component(nx_u8.astype(np.float32, copy=False))[valid]
    ny = _decode_uint8_normal_component(ny_u8.astype(np.float32, copy=False))[valid]
    nz = np.sqrt(np.maximum(0.0, 1.0 - nx * nx - ny * ny)).astype(np.float32, copy=False)

    if radius <= 0 or nx.size == 1:
        n_xyz = np.array([float(nx.mean()), float(ny.mean()), float(nz.mean())], dtype=np.float32)
    else:
        mat = np.array([
            [np.mean(nx * nx), np.mean(nx * ny), np.mean(nx * nz)],
            [np.mean(nx * ny), np.mean(ny * ny), np.mean(ny * nz)],
            [np.mean(nx * nz), np.mean(ny * nz), np.mean(nz * nz)],
        ], dtype=np.float32)
        vals, vecs = np.linalg.eigh(mat)
        n_xyz = vecs[:, int(np.argmax(vals))].astype(np.float32)
        if n_xyz[2] < 0:
            n_xyz = -n_xyz

    norm = float(np.linalg.norm(n_xyz))
    if not np.isfinite(norm) or norm <= 0:
        return None
    n_xyz /= norm
    return np.array([n_xyz[2], n_xyz[1], n_xyz[0]], dtype=np.float32)


def _collect_pcl_normal_zyxs(point_collections, unattached_pcl_strips):
    zyxs = []
    seen = set()

    def add_zyx(zyx):
        zyx = np.asarray(zyx, dtype=np.float32)
        if not (z_begin <= zyx[0] < z_end):
            return
        key = tuple(np.round(zyx, 3))
        if key in seen:
            return
        seen.add(key)
        zyxs.append(zyx)

    for pcl in point_collections:
        for point in pcl['points'].values():
            if 'zyx' in point:
                add_zyx(point['zyx'])

    for strip in unattached_pcl_strips:
        for zyx in strip['zyxs']:
            add_zyx(zyx)

    if not zyxs:
        return np.zeros([0, 3], dtype=np.float32)
    return np.stack(zyxs, axis=0).astype(np.float32, copy=False)


def prepare_pcl_normal_samples(point_collections, unattached_pcl_strips, scroll_zarr):
    if cfg['loss_weight_pcl_normals'] <= 0:
        return None
    if not normal_nx_zarr_path or not normal_ny_zarr_path:
        raise RuntimeError('PCL normal loss is enabled, but FIT_SPIRAL_NORMAL_NX_ZARR_PATH/FIT_SPIRAL_NORMAL_NY_ZARR_PATH is not set')

    print(f'loading PCL normal zarrs group {normal_zarr_group}')
    nx_root = zarr.open(normal_nx_zarr_path, mode='r')
    ny_root = zarr.open(normal_ny_zarr_path, mode='r')
    nx_array = nx_root[normal_zarr_group]
    ny_array = ny_root[normal_zarr_group]
    if nx_array.shape != ny_array.shape:
        raise ValueError(f'nx/ny normal zarr shapes differ: {nx_array.shape} vs {ny_array.shape}')

    if scroll_zarr is not None:
        expected_shape = tuple(np.ceil(np.array(scroll_zarr.shape, dtype=np.float64) / downsample_factor).astype(np.int64))
        if tuple(nx_array.shape) != expected_shape:
            print(
                f'WARNING: normal zarr shape {nx_array.shape} does not match '
                f'ceil(scroll_zarr.shape / downsample_factor) {expected_shape}'
            )
        else:
            print(f'normal zarr shape {nx_array.shape} matches current optimisation grid {expected_shape}')

    multiscales = dict(nx_root.attrs).get('multiscales', [])
    if multiscales:
        datasets = {d.get('path'): d for d in multiscales[0].get('datasets', [])}
        scale = datasets.get(normal_zarr_group, {}).get('coordinateTransformations', [{}])[0].get('scale')
        if scale is not None:
            print(f'normal zarr group {normal_zarr_group} coordinate scale: {scale}')

    zyxs = _collect_pcl_normal_zyxs(point_collections, unattached_pcl_strips)
    if len(zyxs) == 0:
        print('no PCL points available for normal loss')
        return None

    radius = int(cfg['pcl_normals_sample_radius'])
    sampled_zyxs = []
    sampled_normals = []
    for zyx in tqdm(zyxs, 'sampling PCL normals'):
        normal_zyx = _sample_zarr_normal_zyx_at_zyx(nx_array, ny_array, zyx, radius)
        if normal_zyx is None:
            continue
        sampled_zyxs.append(zyx)
        sampled_normals.append(normal_zyx)

    if not sampled_zyxs:
        print(f'no valid PCL normals sampled from {len(zyxs)} PCL points')
        return None

    sampled_zyxs = np.stack(sampled_zyxs, axis=0).astype(np.float32, copy=False)
    sampled_normals = np.stack(sampled_normals, axis=0).astype(np.float32, copy=False)
    print(f'PCL normals: sampled {len(sampled_zyxs)}/{len(zyxs)} points using radius {radius}')
    return {
        'zyxs': sampled_zyxs,
        'normals': sampled_normals,
    }


def _pcl_normal_samples_to_gpu(pcl_normal_samples, device):
    if pcl_normal_samples is None:
        return None
    return {
        'zyxs': torch.from_numpy(pcl_normal_samples['zyxs']).to(device=device),
        'normals': torch.from_numpy(pcl_normal_samples['normals']).to(device=device),
    }


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


def get_pcl_normals_loss(slice_to_spiral_transform, pcl_normal_samples, num_points, epsilon=6.0):
    if pcl_normal_samples is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.zeros([], device=device)
    n = pcl_normal_samples['zyxs'].shape[0]
    if n == 0:
        return torch.zeros([], device=pcl_normal_samples['zyxs'].device)

    device = pcl_normal_samples['zyxs'].device
    sampled = np.random.choice(n, num_points, replace=n < num_points)
    sampled_t = torch.from_numpy(sampled).to(device=device, dtype=torch.long)
    scroll_zyx = pcl_normal_samples['zyxs'][sampled_t]
    target_normal = F.normalize(pcl_normal_samples['normals'][sampled_t], dim=-1)

    scroll_normal = get_radial_normal_in_scroll_space(slice_to_spiral_transform, scroll_zyx, epsilon=epsilon)
    return (1. - (scroll_normal * target_normal).sum(dim=-1).abs()).mean()


def prepare_lasagna_volume(scroll_zarr):
    # Densely load the precomputed nx/ny normal-component and grad_mag (windings-per-base-voxel)
    # zarrs over the z-ROI into a compact uint8 volume that can be pushed to the GPU and sampled by
    # get_lasagna_losses. The raw bytes are kept as-is (no radius averaging, no decode) and
    # only rescaled to floats at sample time; channel layout is (nx_u8, ny_u8, grad_mag_u8), nz is
    # reconstructed as sqrt(1 - nx^2 - ny^2), and per-channel validity is derived from
    # (nx_u8 != 0) | (ny_u8 != 0) for the normal direction and (grad_mag_u8 != 0) for the spacing.
    if cfg['loss_weight_dense_normals'] <= 0 and cfg['loss_weight_dense_spacing'] <= 0:
        return None

    use_normals = cfg['loss_weight_dense_normals'] > 0
    use_spacing = cfg['loss_weight_dense_spacing'] > 0
    if use_normals and (not normal_nx_zarr_path or not normal_ny_zarr_path):
        raise RuntimeError('dense normal loss is enabled, but one of the nx/ny zarr paths is not set')
    if use_spacing and not grad_mag_zarr_path:
        raise RuntimeError('dense spacing loss is enabled, but grad_mag zarr path is not set')

    print(f'loading lasagna zarrs group {normal_zarr_group}')
    nx_array = ny_array = grad_mag_array = None
    reference_shape = None
    if use_normals:
        nx_root = zarr.open(normal_nx_zarr_path, mode='r')
        ny_root = zarr.open(normal_ny_zarr_path, mode='r')
        nx_array = nx_root[normal_zarr_group]
        ny_array = ny_root[normal_zarr_group]
        if nx_array.shape != ny_array.shape:
            raise ValueError(f'nx/ny normal zarr shapes differ: {nx_array.shape} vs {ny_array.shape}')
        reference_shape = nx_array.shape
    if use_spacing:
        grad_mag_root = zarr.open(grad_mag_zarr_path, mode='r')
        grad_mag_array = grad_mag_root[normal_zarr_group]
        if reference_shape is None:
            reference_shape = grad_mag_array.shape
        elif grad_mag_array.shape != reference_shape:
            raise ValueError(f'grad_mag zarr shape {grad_mag_array.shape} differs from dense normal shape {reference_shape}')

    if scroll_zarr is not None:
        expected_shape = tuple(np.ceil(np.array(scroll_zarr.shape, dtype=np.float64) / downsample_factor).astype(np.int64))
        if tuple(reference_shape) != expected_shape:
            print(
                f'WARNING: lasagna zarr shape {reference_shape} does not match '
                f'ceil(scroll_zarr.shape / downsample_factor) {expected_shape}'
            )

    z_size = int(reference_shape[0])
    z_lo = max(0, z_begin)
    z_hi = min(z_size, z_end)
    if z_hi <= z_lo:
        raise RuntimeError(f'lasagna z-ROI [{z_lo}, {z_hi}) is empty (zarr z size {z_size})')

    roi_shape = (z_hi - z_lo, reference_shape[1], reference_shape[2])
    print(f'loading lasagna for z in [{z_lo}, {z_hi}) (shape {roi_shape[0]}, {roi_shape[1]}, {roi_shape[2]})')
    nx_u8 = np.ascontiguousarray(nx_array[z_lo:z_hi], dtype=np.uint8) if use_normals else np.zeros(roi_shape, dtype=np.uint8)
    ny_u8 = np.ascontiguousarray(ny_array[z_lo:z_hi], dtype=np.uint8) if use_normals else np.zeros(roi_shape, dtype=np.uint8)
    grad_mag_u8 = np.ascontiguousarray(grad_mag_array[z_lo:z_hi], dtype=np.uint8) if use_spacing else np.zeros(roi_shape, dtype=np.uint8)
    volume = np.stack([nx_u8, ny_u8, grad_mag_u8], axis=0)  # 3 (nx, ny, grad_mag), z, y, x  uint8
    print(f'lasagna: loaded {volume.nbytes / 1e9:.2f} GB volume {volume.shape}')
    volume = torch.from_numpy(volume).to(device='cuda')
    return {
        'volume': volume,
        'z_origin': z_lo,
        'shape': tuple(volume.shape[1:]),  # z, y, x
    }


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

    flat = _get_or_build_unattached_pcl_flat(pcl_strips, device)
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


def get_patch_stretch_and_normals_loss(slice_to_spiral_transform, num_points, patches, patch_sampling_probabilities, epsilon=1.0):
    # Sample patch points and enforce, at each point, two local rigidity properties of the scroll->spiral transform:
    #   (stretch) epsilon-steps along the discrete patch tangents (+i and +j neighbors in patch coordinates) preserve
    #             length in spiral space;
    #   (normals) the patch surface normal -- the cross-product of the +i and +j patch coordinate deltas -- matches the
    #             outward radial direction in spiral space. This is checked in scroll space: the spiral-space radial
    #             direction is pulled back to scroll space as a covector via J^T (see get_radial_normal_in_scroll_space)
    #             and compared against the measured normal there.
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
    combined_scroll = torch.cat([scroll_zyx, scroll_shift_i, scroll_shift_j], dim=0)
    combined_spiral = slice_to_spiral_transform(combined_scroll)
    spiral_zyx, spiral_shift_i, spiral_shift_j = combined_spiral.chunk(3, dim=0)

    stretch_i = torch.linalg.norm(spiral_shift_i - spiral_zyx, dim=-1) - epsilon
    stretch_j = torch.linalg.norm(spiral_shift_j - spiral_zyx, dim=-1) - epsilon
    if cfg['patch_stretch_loss_norm'] == 'L2':
        stretch_residual = 0.5 * (stretch_i ** 2 + stretch_j ** 2)
    elif cfg['patch_stretch_loss_norm'] == 'L1':
        stretch_residual = 0.5 * (stretch_i.abs() + stretch_j.abs())
    else:
        raise ValueError(f"patch_stretch_loss_norm must be 'L1' or 'L2', got {cfg['patch_stretch_loss_norm']!r}")

    predicted_normal = get_radial_normal_in_scroll_space(slice_to_spiral_transform, scroll_zyx)
    normals_residual = 1. - (predicted_normal * scroll_normal).sum(dim=-1).abs()

    denom = mask.sum().clamp(min=1)
    stretch_loss = (stretch_residual * mask).sum() / denom
    normals_loss = (normals_residual * mask).sum() / denom
    return stretch_loss, normals_loss


def get_bending_loss(slice_to_spiral_transform, dr_per_winding, outer_winding_idx, num_points, epsilon=1.0):
    # Extrinsic bending penalty on the scroll-space image of the spiral surface, evaluated at points
    # sampled uniformly over the spiral cylinder (see sample_spiral_surface_frame).
    # At each point we take the orthonormal in-surface frame (e1, e2) in spiral space, form
    # collinear triples (centre, centre +- e * epsilon) along it, map them to scroll space through the
    # inverse transform, and take the second difference of the resulting scroll positions along each
    # frame axis (a discrete directional second derivative of the map). We then project that onto the
    # scroll-space surface normal (from cross of the pushed-forward frame vectors) and penalise its
    # magnitude squared, so non-isometric stretching of the parameterisation (which lives in the
    # tangent plane and is covered by the symmetric Dirichlet term) does not contribute.
    device = dr_per_winding.device
    if outer_winding_idx is None:
        return torch.zeros([], device=device)

    spiral_center, e1, e2 = sample_spiral_surface_frame(dr_per_winding, outer_winding_idx, num_points)

    spiral_1_prev = spiral_center - e1 * epsilon
    spiral_1_next = spiral_center + e1 * epsilon
    spiral_2_prev = spiral_center - e2 * epsilon
    spiral_2_next = spiral_center + e2 * epsilon
    combined_spiral = torch.cat([spiral_center, spiral_1_prev, spiral_1_next, spiral_2_prev, spiral_2_next], dim=0)
    combined_scroll = slice_to_spiral_transform.inv(combined_spiral)
    scroll_center, scroll_1_prev, scroll_1_next, scroll_2_prev, scroll_2_next = combined_scroll.chunk(5, dim=0)

    # Pushforward of e1, e2 via central differences (scroll-space tangent vectors).
    tangent_1 = (scroll_1_next - scroll_1_prev) / (2.0 * epsilon)
    tangent_2 = (scroll_2_next - scroll_2_prev) / (2.0 * epsilon)
    normal = torch.linalg.cross(tangent_1, tangent_2, dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    second_diff_1 = (scroll_1_next + scroll_1_prev - 2.0 * scroll_center) / (epsilon ** 2)
    second_diff_2 = (scroll_2_next + scroll_2_prev - 2.0 * scroll_center) / (epsilon ** 2)
    bending_1 = ((second_diff_1 * normal).sum(dim=-1)) ** 2
    bending_2 = ((second_diff_2 * normal).sum(dim=-1)) ** 2
    return (bending_1 + bending_2).mean()


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


def _build_strip_spiral_context(slice_to_spiral_transform, dr_per_winding, flat, num_strips):
    # Shared front-half of the per-strip satisfaction pass: given a flat bundle
    # (from `_build_strip_flat_bundle`), transform points into spiral space,
    # unwrap theta across strip boundaries, and produce the per-point normalised
    # shifted-radius (`unwrapped_shifted - windings * dr`). Returns
    # `(ctx, lengths_cpu, num_strips)` where `ctx` is None when there are no
    # points; downstream target-winding selectors (median / mode) operate on
    # `ctx['normalised_radii']` and feed the picked per-strip target through
    # `_strip_satisfaction_from_target`.
    spiral_tolerance = dr_per_winding.detach() * metrics_config['satisfaction_radius_tolerance']
    scan_tolerance = metrics_config['satisfaction_distance_tolerance']
    dr = dr_per_winding.detach()
    device = dr_per_winding.device
    S = num_strips

    if flat is None or flat['total'] == 0:
        lengths_cpu = flat['lengths_cpu'] if flat is not None else torch.zeros(S, dtype=torch.int64)
        return None, lengths_cpu, S

    chunk = 65536

    def transform_in_chunks(zyxs, fn):
        if zyxs.shape[0] <= chunk:
            return fn(zyxs)
        pieces = []
        for st in range(0, zyxs.shape[0], chunk):
            pieces.append(fn(zyxs[st:st + chunk]))
        return torch.cat(pieces, dim=0)

    zyxs = flat['zyxs']
    windings = flat['windings']
    strip_id = flat['strip_id']
    starts = flat['starts']
    lengths = flat['lengths']
    lengths_cpu = flat['lengths_cpu']
    T = flat['total']

    with torch.no_grad():
        spiral_zyxs = transform_in_chunks(zyxs, slice_to_spiral_transform)
        theta, _, shifted_radii = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)

        # Segmented version of _unwrap_track_shifted_radii: build
        # adjustments via a global cumsum where step_adj is zeroed across strip
        # boundaries, then subtract each strip's start value so each strip
        # starts at 0 in its own frame.
        if T > 1:
            theta_d = theta.detach()
            diffs = theta_d[1:] - theta_d[:-1]
            same_strip = strip_id[1:] == strip_id[:-1]
            step_adj = (
                (diffs > np.pi).to(shifted_radii.dtype)
                - (diffs < -np.pi).to(shifted_radii.dtype)
            ) * dr
            step_adj = torch.where(same_strip, step_adj, torch.zeros_like(step_adj))
            cumsum_inner = torch.cumsum(step_adj, dim=0)
            cumsum_flat = torch.cat([
                torch.zeros(1, device=device, dtype=cumsum_inner.dtype),
                cumsum_inner,
            ], dim=0)
            adjustments = cumsum_flat - cumsum_flat[starts[:-1][strip_id]]
        else:
            adjustments = torch.zeros_like(shifted_radii)
        unwrapped_shifted = shifted_radii + adjustments

        normalised_radii = unwrapped_shifted - windings * dr

    ctx = {
        'spiral_tolerance': spiral_tolerance,
        'scan_tolerance': scan_tolerance,
        'dr': dr,
        'device': device,
        'S': S,
        'T': T,
        'transform_in_chunks': transform_in_chunks,
        'slice_to_spiral_transform': slice_to_spiral_transform,
        'zyxs': zyxs,
        'windings': windings,
        'strip_id': strip_id,
        'starts': starts,
        'lengths': lengths,
        'lengths_cpu': lengths_cpu,
        'spiral_zyxs': spiral_zyxs,
        'theta': theta,
        'shifted_radii': shifted_radii,
        'adjustments': adjustments,
        'unwrapped_shifted': unwrapped_shifted,
        'normalised_radii': normalised_radii,
    }
    return ctx, lengths_cpu, S


def _strip_satisfaction_from_target(ctx, target_normalised_per_strip):
    # Given a per-strip target normalised shifted-radius, count points whose
    # spiral-space radius and scan-space distance both fall within the
    # satisfaction tolerances. Returns
    # `(satisfied_counts_cpu, per_point_satisfaction_cpu_list)`.
    dr = ctx['dr']
    device = ctx['device']
    S = ctx['S']
    strip_id = ctx['strip_id']
    windings = ctx['windings']
    theta = ctx['theta']
    adjustments = ctx['adjustments']
    unwrapped_shifted = ctx['unwrapped_shifted']
    spiral_zyxs = ctx['spiral_zyxs']
    zyxs = ctx['zyxs']
    lengths_cpu = ctx['lengths_cpu']
    spiral_tolerance = ctx['spiral_tolerance']
    scan_tolerance = ctx['scan_tolerance']
    transform_in_chunks = ctx['transform_in_chunks']
    slice_to_spiral_transform = ctx['slice_to_spiral_transform']

    with torch.no_grad():
        target_normalised = target_normalised_per_strip[strip_id]
        target_shifted = target_normalised + windings * dr
        spiral_in_band = (unwrapped_shifted - target_shifted).abs() <= spiral_tolerance

        target_radii = target_shifted - adjustments + theta / (2 * np.pi) * dr
        target_spiral_zyxs = torch.stack([
            spiral_zyxs[..., 0],
            torch.sin(theta) * target_radii,
            torch.cos(theta) * target_radii,
        ], dim=-1)
        target_scroll_zyxs = transform_in_chunks(target_spiral_zyxs, slice_to_spiral_transform.inv)
        scan_distances = torch.linalg.norm(target_scroll_zyxs - zyxs, dim=-1)
        scan_in_band = scan_distances <= scan_tolerance

        satisfied = spiral_in_band & scan_in_band

        satisfied_counts_dev = torch.zeros(S, dtype=torch.int64, device=device)
        satisfied_counts_dev.scatter_add_(0, strip_id, satisfied.to(torch.int64))
        satisfied_counts = satisfied_counts_dev.cpu()

        per_point_satisfaction = list(torch.split(satisfied.cpu(), lengths_cpu.tolist()))

    return satisfied_counts, per_point_satisfaction


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
    #
    # All strips are processed in a single batched pass: points are concatenated
    # into one flat (T, 3) tensor, the scan->spiral transform runs once over
    # everything, then unwrap / median / satisfaction are done with segmented
    # cumsum and a single composite-key sort (no Python-level per-strip loop).
    flat = _get_or_build_unattached_pcl_flat(pcl_strips, dr_per_winding.device)
    ctx, lengths_cpu, S = _build_strip_spiral_context(
        slice_to_spiral_transform, dr_per_winding, flat, len(pcl_strips),
    )
    if ctx is None:
        per_point = [torch.zeros([int(n.item())], dtype=torch.bool) for n in lengths_cpu]
        return torch.zeros(S, dtype=torch.int64), lengths_cpu.clone(), per_point

    dr = ctx['dr']

    with torch.no_grad():
        medians = _segmented_median_per_strip(ctx)
        target_normalised_per_strip = torch.round(medians / dr) * dr

    satisfied_counts, per_point_satisfaction = _strip_satisfaction_from_target(ctx, target_normalised_per_strip)
    return satisfied_counts, lengths_cpu.clone(), per_point_satisfaction


def _segmented_median_per_strip(ctx):
    # Segmented median: sort the flat values with a composite key
    # (strip_id-major, normalised_radii-minor) so values for each strip end
    # up contiguous and sorted within their range. Per-strip median is then
    # at start + (length - 1) // 2 (matching torch.median's lower-median
    # convention for even lengths). Float64 keeps headroom against
    # strip_id * val_range overflow for hundreds-of-thousands of strips.
    normalised_radii = ctx['normalised_radii']
    strip_id = ctx['strip_id']
    starts = ctx['starts']
    lengths = ctx['lengths']
    S = ctx['S']
    device = ctx['device']
    if normalised_radii.numel() == 0:
        return torch.zeros(S, dtype=normalised_radii.dtype, device=device)

    val_min = normalised_radii.min().to(torch.float64)
    val_max = normalised_radii.max().to(torch.float64)
    val_range = (val_max - val_min) + 1.0
    composite = (
        strip_id.to(torch.float64) * val_range
        + (normalised_radii.to(torch.float64) - val_min)
    )
    order = torch.argsort(composite)
    sorted_norm = normalised_radii[order]
    median_indices = starts[:-1] + (lengths - 1) // 2
    return sorted_norm[median_indices]


def _mode_winding_per_strip(strip_id, winding_idx_per_point, S, device):
    # Per strip, return the discrete winding index (round(normalised / dr))
    # that the most points fall on. Ties are broken by the smaller winding
    # index. Strips with no points get 0.
    mode_winding_per_strip = torch.zeros(S, dtype=torch.int64, device=device)
    if winding_idx_per_point.numel() == 0:
        return mode_winding_per_strip

    w_min = winding_idx_per_point.min()
    w_max = winding_idx_per_point.max()
    w_span = (w_max - w_min + 1).to(torch.int64)
    composite = strip_id.to(torch.int64) * w_span + (winding_idx_per_point - w_min).to(torch.int64)
    sorted_comp, _ = torch.sort(composite)
    unique_comp, counts = torch.unique_consecutive(sorted_comp, return_counts=True)
    u_strip = unique_comp // w_span
    u_widx = (unique_comp % w_span) + w_min

    # Pick per-strip row with max count (smallest winding wins ties) via a
    # composite-sort: key = strip-major, then count-descending, then
    # winding-ascending. The first row for each strip after sorting is the
    # winner.
    counts_max = counts.max().to(torch.int64)
    widx_min = u_widx.min().to(torch.int64)
    widx_max = u_widx.max().to(torch.int64)
    widx_span = (widx_max - widx_min + 1).to(torch.int64)
    key = (
        u_strip * ((counts_max + 1) * widx_span)
        + (counts_max - counts.to(torch.int64)) * widx_span
        + (u_widx.to(torch.int64) - widx_min)
    )
    order = torch.argsort(key)
    sorted_strip = u_strip[order]
    sorted_widx = u_widx[order]
    new_strip = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        sorted_strip[1:] != sorted_strip[:-1],
    ])
    first_idx = torch.nonzero(new_strip, as_tuple=False).squeeze(-1)
    mode_winding_per_strip[sorted_strip[first_idx]] = sorted_widx[first_idx].to(torch.int64)
    return mode_winding_per_strip


def get_track_satisfied_counts(slice_to_spiral_transform, dr_per_winding, tracks):
    # Tracks are unannotated strips. For each track, infer the integer winding it
    # most often lies near, then reuse the same per-point spiral-space and scan-
    # space satisfaction checks as unattached PCLs. Returns values aligned to
    # `valid_track_indices`, since tracks with fewer than two points are skipped.
    device = dr_per_winding.device
    if not tracks:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty, empty, [], empty

    valid_track_indices = [i for i, track in enumerate(tracks) if len(track) >= 2]
    if not valid_track_indices:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty, empty, [], empty

    strip_arrays = [
        (
            np.asarray(tracks[i], dtype=np.float32),
            np.zeros(len(tracks[i]), dtype=np.float32),
        )
        for i in valid_track_indices
    ]
    flat = _build_strip_flat_bundle(strip_arrays, device)
    ctx, lengths_cpu, S = _build_strip_spiral_context(
        slice_to_spiral_transform, dr_per_winding, flat, len(valid_track_indices),
    )
    valid_track_indices_t = torch.tensor(valid_track_indices, dtype=torch.int64)
    if ctx is None:
        per_point = [torch.zeros([int(n.item())], dtype=torch.bool) for n in lengths_cpu]
        return valid_track_indices_t, torch.zeros(S, dtype=torch.int64), lengths_cpu.clone(), per_point, torch.zeros(S, dtype=torch.int64)

    dr = ctx['dr']
    strip_id = ctx['strip_id']
    normalised_radii = ctx['normalised_radii']

    with torch.no_grad():
        winding_idx_per_point = torch.round(normalised_radii / dr).to(torch.int64)
        mode_winding_per_strip = _mode_winding_per_strip(strip_id, winding_idx_per_point, S, device)
        target_normalised_per_strip = mode_winding_per_strip.to(dr.dtype) * dr

    satisfied_counts, per_point_satisfaction = _strip_satisfaction_from_target(ctx, target_normalised_per_strip)
    return (
        valid_track_indices_t,
        satisfied_counts,
        lengths_cpu.clone(),
        per_point_satisfaction,
        mode_winding_per_strip.cpu(),
    )


def get_track_satisfied_counts_in_chunks(slice_to_spiral_transform, dr_per_winding, tracks, chunk_size=500_000):
    # Memory-safe wrapper around get_track_satisfied_counts for very large track
    # sets (the full z-range has tens of millions of tracks, whose flat point
    # tensors do not fit in GPU memory at once). Every per-track quantity
    # (winding-mode, unwrap, satisfaction) is independent across tracks, so
    # splitting the track list into contiguous chunks of whole tracks and
    # concatenating the per-track results yields identical satisfied/total
    # counts to a single pass. Returns only the two tensors the metric needs:
    # (satisfied_counts, total_counts), each 1-D int64 over the valid tracks.
    sat_parts, tot_parts = [], []
    for start in range(0, len(tracks), chunk_size):
        chunk = tracks[start:start + chunk_size]
        _, sat, tot, _, _ = get_track_satisfied_counts(slice_to_spiral_transform, dr_per_winding, chunk)
        sat_parts.append(sat)
        tot_parts.append(tot)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not sat_parts:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty
    return torch.cat(sat_parts), torch.cat(tot_parts)


def get_partially_supported_tracks(
    slice_to_spiral_transform, dr_per_winding, tracks,
    lower_fraction=0.5, upper_fraction=0.95,
):
    # For each track, infer a target winding from the *mode* of the per-point
    # winding indices in spiral space (i.e. the winding the track is most often
    # near to — `round(normalised_shifted_radius / dr)` per point, then the
    # most frequent value per track), measure the fraction of points within
    # the satisfaction tolerances of that target winding, and return
    # `(track, mode_winding_idx)` pairs whose fraction falls in
    # `[lower_fraction, upper_fraction]`. The tolerances are
    # `metrics_config['satisfaction_radius_tolerance']` (spiral space) and
    # `metrics_config['satisfaction_distance_tolerance']` (scan space) — same
    # as patch / pcl satisfaction elsewhere.
    # Also returns `(num_below, num_above)`: counts of tracks discarded for
    # falling below `lower_fraction` or above `upper_fraction` respectively.
    kept_tracks = [t for t in tracks if len(t) >= 2]
    device = dr_per_winding.device
    strip_arrays = [
        (t.astype(np.float32, copy=False), np.zeros(len(t), dtype=np.float32))
        for t in kept_tracks
    ]
    flat = _build_strip_flat_bundle(strip_arrays, device)
    ctx, lengths_cpu, S = _build_strip_spiral_context(
        slice_to_spiral_transform, dr_per_winding, flat, len(kept_tracks),
    )
    if ctx is None:
        return [], 0, 0

    dr = ctx['dr']
    strip_id = ctx['strip_id']
    normalised_radii = ctx['normalised_radii']

    with torch.no_grad():
        winding_idx_per_point = torch.round(normalised_radii / dr).to(torch.int64)
        mode_winding_per_strip = _mode_winding_per_strip(strip_id, winding_idx_per_point, S, device)
        target_normalised_per_strip = mode_winding_per_strip.to(dr.dtype) * dr

    satisfied_counts, _ = _strip_satisfaction_from_target(ctx, target_normalised_per_strip)
    satisfied_counts_list = satisfied_counts.tolist()
    lengths_list = lengths_cpu.tolist()
    mode_winding_list = mode_winding_per_strip.cpu().tolist()

    partially_supported = []
    num_below = 0
    num_above = 0
    for track, length, sat, mode_winding in zip(
        kept_tracks, lengths_list, satisfied_counts_list, mode_winding_list,
    ):
        if length == 0:
            continue
        fraction = sat / length
        if fraction < lower_fraction:
            num_below += 1
        elif fraction > upper_fraction:
            num_above += 1
        else:
            partially_supported.append((track, mode_winding))
    return partially_supported, num_below, num_above


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
def _compute_winding_range_and_input_extents(slice_to_spiral_transform, dr_per_winding, patches, unattached_pcl_strips=()):
    """Transforms each patch's valid in-ROI quad centers and each unattached pcl strip's
    in-ROI points into spiral space, and computes:
      - output_winding_range = (min_winding_idx, max_winding_idx) — inclusive min and
        exclusive max winding indices, using `(shifted_radius / dr).round().clamp_min(0)`
        (the convention in overlay_patches_on_slices); min clamped to
        cfg['output_first_winding'], cfg['output_winding_margin'] applied on both sides.
      - patch_extents: list parallel to `patches` of (max_radius, max_winding) per input,
        or (None, None) when no valid in-ROI samples.
      - pcl_extents: list parallel to `unattached_pcl_strips`, same shape."""
    device = dr_per_winding.device
    dr = dr_per_winding.detach()
    min_w = None
    max_w = None

    def update_from_winding_indices(winding_indices):
        nonlocal min_w, max_w
        local_min = int(winding_indices.min().item())
        local_max = int(winding_indices.max().item())
        min_w = local_min if min_w is None else min(min_w, local_min)
        max_w = local_max if max_w is None else max(max_w, local_max)

    chunk = 65536

    def transform_in_chunks(zyxs):
        spiral_pieces = []
        for start in range(0, zyxs.shape[0], chunk):
            spiral_pieces.append(slice_to_spiral_transform(zyxs[start:start + chunk]))
        return torch.cat(spiral_pieces, dim=0) if len(spiral_pieces) > 1 else spiral_pieces[0]

    patch_extents = [(None, None)] * len(patches)
    for patch_index, patch in enumerate(patches):
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
        spiral_zyxs = transform_in_chunks(quad_center_zyxs[mask])
        _, radius, shifted_radius = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
        winding_indices = (shifted_radius / dr).round().to(torch.int64).clamp_min(0)
        update_from_winding_indices(winding_indices)
        patch_extents[patch_index] = (
            float(radius.max().item()),
            int(winding_indices.max().item()),
        )

    pcl_extents = [(None, None)] * len(unattached_pcl_strips)
    if unattached_pcl_strips:
        flat = _get_or_build_unattached_pcl_flat(unattached_pcl_strips, device)
        if flat is not None and flat['total'] > 0:
            zyxs = flat['zyxs']
            strip_id = flat['strip_id']
            num_strips = flat['num_strips']
            in_roi = (zyxs[:, 0] >= z_begin) & (zyxs[:, 0] < z_end)
            if in_roi.any():
                zyxs_roi = zyxs[in_roi]
                strip_id_roi = strip_id[in_roi]
                spiral_zyxs = transform_in_chunks(zyxs_roi)
                _, radius, shifted_radius = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
                winding_indices = (shifted_radius / dr).round().to(torch.int64).clamp_min(0)
                update_from_winding_indices(winding_indices)
                per_strip_max_r = torch.zeros(num_strips, dtype=torch.float32, device=device)
                per_strip_max_w = torch.full((num_strips,), -1, dtype=torch.int64, device=device)
                strip_has_roi = torch.zeros(num_strips, dtype=torch.bool, device=device)
                per_strip_max_r.scatter_reduce_(0, strip_id_roi, radius.to(torch.float32), reduce='amax')
                per_strip_max_w.scatter_reduce_(0, strip_id_roi, winding_indices, reduce='amax')
                strip_has_roi.scatter_(0, strip_id_roi, torch.ones_like(strip_id_roi, dtype=torch.bool))
                per_strip_max_r_cpu = per_strip_max_r.cpu().tolist()
                per_strip_max_w_cpu = per_strip_max_w.cpu().tolist()
                strip_has_roi_cpu = strip_has_roi.cpu().tolist()
                for k in range(num_strips):
                    if strip_has_roi_cpu[k]:
                        pcl_extents[k] = (per_strip_max_r_cpu[k], per_strip_max_w_cpu[k])

    first_winding = cfg['output_first_winding']
    if min_w is None:
        output_winding_range = (first_winding, first_winding)
    else:
        margin = cfg['output_winding_margin']
        output_winding_range = (max(min_w - margin, first_winding), max_w + 1 + margin)
    return output_winding_range, patch_extents, pcl_extents


def _infer_shell_outer_winding_idx(slice_to_spiral_transform, dr_per_winding, patches, unattached_pcl_strips):
    _, patch_extents, pcl_extents = _compute_winding_range_and_input_extents(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        unattached_pcl_strips,
    )
    observed_max = None
    for _, max_w in itertools.chain(patch_extents, pcl_extents):
        if max_w is None:
            continue
        observed_max = max_w if observed_max is None else max(observed_max, max_w)
    if observed_max is None:
        observed_max = cfg['output_first_winding']
    return int(observed_max + cfg['shell_outer_winding_margin'])


def _warn_if_inputs_exceed_flow_bounds(patch_ids, patch_extents, unattached_pcl_strips, pcl_extents, flow_field_radius):
    gap_expander_num_windings = cfg['gap_expander_num_windings']

    over_radius_patches = []
    over_winding_patches = []
    for pid, (max_r, max_w) in zip(patch_ids, patch_extents):
        if max_r is None:
            continue
        if max_r > flow_field_radius:
            over_radius_patches.append((pid, max_r))
        if max_w >= gap_expander_num_windings:
            over_winding_patches.append((pid, max_w))

    over_radius_pcls = []
    over_winding_pcls = []
    for k, (strip, (max_r, max_w)) in enumerate(zip(unattached_pcl_strips, pcl_extents)):
        if max_r is None:
            continue
        name = strip.get('name') or strip.get('id') or strip.get('source_file') or f'#{k}'
        if max_r > flow_field_radius:
            over_radius_pcls.append((name, max_r))
        if max_w >= gap_expander_num_windings:
            over_winding_pcls.append((name, max_w))

    def _print_offenders(kind, value_label, threshold_label, patches, pcls, fmt):
        if not (patches or pcls):
            return
        print(f'WARNING: {len(patches)} patch(es) and {len(pcls)} unattached pcl(s) have {value_label} exceeding {threshold_label}:')
        for pid, v in sorted(patches, key=lambda e: -e[1])[:10]:
            print(f'  patch {pid}: max {kind} {fmt(v)}')
        if len(patches) > 10:
            print(f'  ... and {len(patches) - 10} more patches')
        for name, v in sorted(pcls, key=lambda e: -e[1])[:10]:
            print(f'  pcl {name}: max {kind} {fmt(v)}')
        if len(pcls) > 10:
            print(f'  ... and {len(pcls) - 10} more pcls')

    _print_offenders(
        'spiral radius', 'spiral-space radius', f'flow_bounds_radius ({flow_field_radius})',
        over_radius_patches, over_radius_pcls, lambda v: f'{v:.1f}',
    )
    _print_offenders(
        'winding idx', 'winding index', f'gap_expander_num_windings ({gap_expander_num_windings})',
        over_winding_patches, over_winding_pcls, lambda v: f'{v}',
    )


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

    chunk_size = 16384
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        valid_c = valid_bbox[s:e]
        if not valid_c.any():
            continue
        u_min_c = u_min[s:e]
        v_min_c = v_min[s:e]
        bbox_h_c = bbox_h[s:e]
        bbox_w_c = bbox_w[s:e]
        max_h = int(bbox_h_c[valid_c].max().item())
        max_w = int(bbox_w_c[valid_c].max().item())

        du_grid, dv_grid = torch.meshgrid(
            torch.arange(max_h, device=device),
            torch.arange(max_w, device=device),
            indexing='ij',
        )
        us = u_min_c[:, None, None] + du_grid[None]  # (n, max_h, max_w)
        vs = v_min_c[:, None, None] + dv_grid[None]
        in_bbox = (
            (du_grid[None] < bbox_h_c[:, None, None])
            & (dv_grid[None] < bbox_w_c[:, None, None])
            & valid_c[:, None, None]
        )

        tri_uvs_c = tri_uvs[s:e]
        pts = torch.stack([us.float(), vs.float()], dim=-1)
        a = tri_uvs_c[:, 0]
        b = tri_uvs_c[:, 1]
        c = tri_uvs_c[:, 2]
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
            continue

        tri_scrolls_c = tri_scrolls[s:e]
        sa = tri_scrolls_c[:, 0][:, None, None, :]
        sb = tri_scrolls_c[:, 1][:, None, None, :]
        sc = tri_scrolls_c[:, 2][:, None, None, :]
        interp = alpha[..., None] * sa + beta[..., None] * sb + gamma[..., None] * sc

        sel = mask.reshape(-1)
        target_u = us.reshape(-1)[sel]
        target_v_local = vs.reshape(-1)[sel]
        target_w_flat = tri_target_w[s:e][:, None, None].expand(-1, max_h, max_w).reshape(-1)[sel]
        target_v_global = winding_offsets_t[target_w_flat] + target_v_local
        scroll_zyxs[target_u, target_v_global] = interp.reshape(-1, 3)[sel]


@torch.inference_mode
def _build_spliced_overlay(
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
def save_mesh(slice_to_spiral_transform, dr_per_winding, patches, unattached_pcl_strips, out_path, name='mesh'):

    (min_winding_idx, max_winding_idx), _, _ = _compute_winding_range_and_input_extents(slice_to_spiral_transform, dr_per_winding, patches, unattached_pcl_strips)
    if cfg['shell_outer_winding_idx'] is not None:
        max_winding_idx = min(max_winding_idx, cfg['shell_outer_winding_idx'])
    print(f'save_mesh {name}: winding range [{min_winding_idx}, {max_winding_idx})')
    grid_spacing = cfg['output_step_size'] // downsample_factor  # voxels in downsampled volume
    z_margin = cfg['flow_bounds_z_margin']
    spiral_yxs_by_winding = get_spiral_yxs(max_winding_idx, dr_per_winding, grid_spacing, group_by_winding=True)
    num_thetas_by_winding = [len(yxs_for_winding) for yxs_for_winding in spiral_yxs_by_winding]
    spiral_yxs = torch.cat(spiral_yxs_by_winding, dim=0)
    z0 = z_begin - z_margin
    spiral_zs = torch.arange(z0, z_end + z_margin, grid_spacing, dtype=torch.float32, device=spiral_yxs.device)
    spiral_zyxs = torch.cat([spiral_zs[:, None, None].expand(-1, spiral_yxs.shape[0], 1), spiral_yxs[None, :, :].expand(spiral_zs.shape[0], -1, 2)], dim=-1)
    chunk = 65536
    flat_spiral_zyxs = spiral_zyxs.reshape(-1, 3)
    scroll_pieces = []
    for start in range(0, flat_spiral_zyxs.shape[0], chunk):
        scroll_pieces.append(slice_to_spiral_transform.inv(flat_spiral_zyxs[start : start + chunk]))
    scroll_zyxs = torch.cat(scroll_pieces, dim=0).reshape(*spiral_zyxs.shape)

    # Mark vertices whose scroll-space z falls outside the [z_begin, z_end) ROI as invalid
    # (-1, -1, -1), but allow splicing to still replace these.
    out_of_roi = (scroll_zyxs[..., 0] < z_begin) | (scroll_zyxs[..., 0] >= z_end)
    scroll_zyxs[out_of_roi] = -1.0

    # Spliced variant: replace cells covered by quads of overall- or boundary-satisfied
    # patches with patch-derived points, interpolated bilinearly across each quad in the
    # flattened-spiral UV coords of its target winding.
    spliced_scroll_zyxs = scroll_zyxs.clone()
    satisfied_patches, _, _, _, boundary_satisfied_patches, target_winding_idx_per_patch = get_patch_satisfied_areas(
        slice_to_spiral_transform, dr_per_winding, patches,
    )
    _build_spliced_overlay(
        spliced_scroll_zyxs, num_thetas_by_winding, z0, grid_spacing,
        slice_to_spiral_transform, dr_per_winding,
        patches,
        satisfied_patches, boundary_satisfied_patches, target_winding_idx_per_patch,
    )

    step_size = grid_spacing * downsample_factor
    tag_suffix = f'_{run_tag}' if run_tag else ''
    out_dir = f'{out_path}/meshes/{name}{tag_suffix}'
    os.makedirs(out_dir, exist_ok=True)
    for uuid_suffix, variant_zyxs in [('', scroll_zyxs), ('_spliced', spliced_scroll_zyxs)]:
        offset = 0
        for winding_idx, num_thetas in enumerate(tqdm(num_thetas_by_winding, desc=f'saving winding patches ({name}{uuid_suffix})')):
            if num_thetas >= 2 and winding_idx >= min_winding_idx:
                winding_slice = variant_zyxs[:, offset:offset + num_thetas]
                invalid_mask = (winding_slice == -1.0).all(dim=-1).cpu().numpy()
                winding_zyxs = (winding_slice * downsample_factor).cpu().numpy().astype(np.float32)
                winding_zyxs[invalid_mask] = -1.0
                save_tifxyz(
                    winding_zyxs,
                    out_dir,
                    uuid=f'w{winding_idx:03d}{uuid_suffix}{tag_suffix}',
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
            _render_spiral_on_tracks_for_slice(
                spiral_zyx, spiral_density, dr_per_winding,
                slice_z, tracks, [],
                out_path, suffix,
            )
        if os.environ.get('FIT_SPIRAL_SAVE_DISPLACEMENT') == '1':
            slice_zyx = torch.cat([torch.full([*slice_yx.shape[:2], 1], slice_z, device=device), slice_yx], dim=-1)
            visualise_field(spiral_zyx - slice_zyx, f'displacement_s{slice_z * downsample_factor:05}')


def _collect_anchor_scroll_zyxs(patch_atlas, unattached_pcl_strips, device):
    # Concatenate every in-ROI patch vertex (sourced from PatchGpuAtlas.valid_in_roi_zyxs,
    # already on GPU) and every in-ROI unattached-pcl strip point as scroll-space zyxs on
    # `device`. The result is the trusted point cloud used both to build snap anchors and to
    # mask out track points near already-constrained regions.
    pieces = [patch_atlas.valid_in_roi_zyxs]
    for strip in unattached_pcl_strips:
        zyxs = torch.from_numpy(strip['zyxs']).to(device=device, dtype=torch.float32)
        in_roi = (zyxs[..., 0] >= z_begin) & (zyxs[..., 0] < z_end)
        if in_roi.any():
            pieces.append(zyxs[in_roi])
    return torch.cat(pieces, dim=0)


def _build_snap_anchors(slice_to_spiral_transform, patch_atlas, unattached_pcl_strips, device):
    # Collect every patch valid in-ROI vertex and every unattached-pcl strip point
    # (in-ROI) as scroll-space zyxs, transform them once to spiral space with the
    # current transform, and return both flat tensors so the snapping phase can
    # later sample a subset and anchor them back to these stored spiral targets.
    anchor_scroll = _collect_anchor_scroll_zyxs(patch_atlas, unattached_pcl_strips, device)
    if anchor_scroll.shape[0] == 0:
        return anchor_scroll, torch.empty([0, 3], device=device)
    chunk = 65536
    with torch.no_grad():
        out_pieces = []
        for st in range(0, anchor_scroll.shape[0], chunk):
            out_pieces.append(slice_to_spiral_transform(anchor_scroll[st:st + chunk]))
        anchor_spiral = torch.cat(out_pieces, dim=0).detach()
    return anchor_scroll, anchor_spiral


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


def _track_points_far_from_anchors_mask(track_zyx, anchor_tree, threshold):
    # Returns a boolean numpy mask, True where the track point is farther than `threshold`
    # from every anchor. `track_zyx` is an (N, 3) array (numpy or tensor); `anchor_tree` is
    # a cKDTree built by _build_anchor_kdtree (or None). Uses a fixed-radius nearest-neighbour
    # query — O(N log A), parallel across cores — instead of the full pairwise distance matrix.
    if isinstance(track_zyx, torch.Tensor):
        track_np = track_zyx.detach().cpu().numpy()
    else:
        track_np = np.asarray(track_zyx)
    track_np = np.ascontiguousarray(track_np, dtype=np.float32)
    if threshold <= 0 or anchor_tree is None:
        return np.ones(track_np.shape[0], dtype=bool)
    # query returns dist == inf for points with no anchor within distance_upper_bound.
    dist, _ = anchor_tree.query(track_np, k=1, distance_upper_bound=float(threshold), workers=-1)
    return np.isinf(dist)


def _mask_patches_near_trusted_geometry(patches_dict, anchor_tree, radius):
    # For each patch in `patches_dict`, invalidate (set zyxs -> -1) every currently-valid vertex
    # lying within `radius` (scroll space, downsampled voxels) of any trusted-geometry anchor in
    # `anchor_tree`, then re-derive the patch's masks/area. Patches left with no valid quad are
    # dropped. This is the patch analogue of the track-exclusion in _prepare_main_phase_tracks:
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
        far = _track_points_far_from_anchors_mask(zyxs_np, anchor_tree, radius)  # True => keep
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


def _prepare_main_phase_tracks(tracks, patch_atlas, unattached_pcl_strips, exclusion_radius, device):
    # Drop every track point that lies within `exclusion_radius` of any patch vertex / pcl
    # strip point, then drop tracks left with fewer than 2 points. Returns a flat per-point
    # zyx tensor plus per-track offsets and lengths, all on `device`.
    if not tracks:
        return None
    print('removing tracks near patches')
    anchor_scroll = _collect_anchor_scroll_zyxs(patch_atlas, unattached_pcl_strips, device)
    anchor_tree = _build_anchor_kdtree(anchor_scroll)
    flat_zyx_np = np.concatenate([t.astype(np.float32) for t in tracks], axis=0)
    track_id_np = np.concatenate([
        np.full(len(t), i, dtype=np.int64) for i, t in enumerate(tracks)
    ])
    keep_np = _track_points_far_from_anchors_mask(flat_zyx_np, anchor_tree, exclusion_radius)
    flat_zyx_np = flat_zyx_np[keep_np]
    track_id_np = track_id_np[keep_np]
    num_tracks_orig = len(tracks)
    new_lengths = np.bincount(track_id_np, minlength=num_tracks_orig)
    surviving = np.where(new_lengths >= 2)[0]
    print(f'kept {len(surviving)} / {len(tracks)} tracks')
    if len(surviving) == 0:
        return None
    old_to_new = -np.ones(num_tracks_orig, dtype=np.int64)
    old_to_new[surviving] = np.arange(len(surviving))
    new_id = old_to_new[track_id_np]
    keep2 = new_id >= 0
    flat_zyx_np = flat_zyx_np[keep2]
    new_id = new_id[keep2]
    # Stable sort by new track id so same-track points end up contiguous; the within-track
    # ordering is preserved.
    sort_idx = np.argsort(new_id, kind='stable')
    flat_zyx_np = flat_zyx_np[sort_idx]
    lengths_new = new_lengths[surviving].astype(np.int64)
    offsets_new = np.concatenate([[0], np.cumsum(lengths_new)]).astype(np.int64)
    print(
        f'track radius loss: {len(surviving)}/{num_tracks_orig} tracks survive exclusion '
        f'(radius {exclusion_radius:.1f}); {int(lengths_new.sum())} points retained'
    )
    return {
        'flat_zyx': torch.from_numpy(flat_zyx_np).to(device=device),
        'offsets': torch.from_numpy(offsets_new).to(device=device),
        'lengths': torch.from_numpy(lengths_new).to(device=device),
    }


def _track_flat_bundle(prepared_tracks, device):
    cached = prepared_tracks.get('_flat_bundle')
    if cached is not None:
        return cached

    offsets = prepared_tracks['offsets']
    lengths = prepared_tracks['lengths']
    total = int(prepared_tracks['flat_zyx'].shape[0])
    S = int(lengths.numel())
    strip_id = torch.repeat_interleave(torch.arange(S, device=device), lengths)
    bundle = {
        'zyxs': prepared_tracks['flat_zyx'],
        'windings': torch.zeros(total, device=device),
        'strip_id': strip_id,
        'starts': offsets,
        'starts_cpu': offsets.cpu(),
        'lengths': lengths,
        'lengths_cpu': lengths.cpu(),
        'num_strips': S,
        'total': total,
    }
    prepared_tracks['_flat_bundle'] = bundle
    return bundle


@torch.no_grad()
def compute_track_em_assignment(slice_to_spiral_transform, dr_per_winding, track_flat, cfg):
    ctx, lengths_cpu, S = _build_strip_spiral_context(
        slice_to_spiral_transform, dr_per_winding, track_flat, track_flat['num_strips'],
    )
    if ctx is None:
        device = dr_per_winding.device
        return {
            'W': torch.zeros(S, dtype=torch.int64, device=device),
            'm_p': torch.zeros(0, dtype=torch.int64, device=device),
            'confidence': torch.zeros(S, dtype=dr_per_winding.dtype, device=device),
            'valid': torch.zeros(S, dtype=torch.bool, device=device),
        }

    dr = ctx['dr']
    device = ctx['device']
    normalised_radii = ctx['normalised_radii']
    winding_idx_per_point = torch.round(normalised_radii / dr).to(torch.int64)
    if cfg['track_em_assignment'] == 'mode':
        W = _mode_winding_per_strip(ctx['strip_id'], winding_idx_per_point, S, device)
    elif cfg['track_em_assignment'] == 'median':
        W = torch.round(_segmented_median_per_strip(ctx) / dr).to(torch.int64)
    else:
        raise ValueError(f"track_em_assignment must be 'mode' or 'median', got {cfg['track_em_assignment']!r}")

    m_p = torch.round(ctx['adjustments'] / dr).to(torch.int64)
    target_normalised_per_strip = W.to(dr.dtype) * dr
    satisfied_counts, _ = _strip_satisfaction_from_target(ctx, target_normalised_per_strip)
    satisfied_counts = satisfied_counts.to(device=device)
    confidence = satisfied_counts.to(dr.dtype) / ctx['lengths'].clamp(min=1).to(dr.dtype)
    valid = confidence >= float(cfg['track_em_min_confidence'])
    return {
        'W': W,
        'm_p': m_p,
        'confidence': confidence,
        'valid': valid,
    }


def _sample_prepared_track_points(prepared_tracks, candidate_track_ids, num_tracks_per_step, num_points_per_track):
    flat_zyx = prepared_tracks['flat_zyx']
    offsets = prepared_tracks['offsets']
    lengths = prepared_tracks['lengths']
    device = flat_zyx.device
    num_tracks = int(lengths.numel())
    if num_tracks == 0 or num_tracks_per_step <= 0 or num_points_per_track <= 0:
        return None

    if candidate_track_ids is None:
        num_candidates = num_tracks
        k = min(int(num_tracks_per_step), num_candidates)
        track_idx = torch.randint(num_candidates, (k,), device=device)
    else:
        if candidate_track_ids.numel() == 0:
            return None
        num_candidates = int(candidate_track_ids.numel())
        k = min(int(num_tracks_per_step), num_candidates)
        candidate_sel = torch.randint(num_candidates, (k,), device=device)
        track_idx = candidate_track_ids[candidate_sel]

    track_lengths_sample = lengths[track_idx]
    track_offsets_sample = offsets[track_idx]
    point_idx_within = (
        torch.rand([k, num_points_per_track], device=device)
        * track_lengths_sample[:, None].to(torch.float32)
    ).to(torch.int64)
    point_idx_within, _ = torch.sort(point_idx_within, dim=-1)
    flat_idx = (track_offsets_sample[:, None] + point_idx_within).reshape(-1)
    sampled_scroll = flat_zyx[flat_idx].view(k, num_points_per_track, 3)
    return track_idx, flat_idx, sampled_scroll


def _same_radius_loss_for_shifted_radii(shifted_radii, dr_per_winding):
    radius_hinge_margin = dr_per_winding.detach() * cfg['track_radius_loss_margin']
    if cfg['track_radius_target'] == 'mean':
        radius_target_per_track = shifted_radii.mean(dim=-1, keepdim=True)
    elif cfg['track_radius_target'] == 'median':
        radius_target_per_track = shifted_radii.median(dim=-1, keepdim=True).values
    else:
        raise ValueError(f"track_radius_target must be 'mean' or 'median', got {cfg['track_radius_target']!r}")
    deviations = (shifted_radii - radius_target_per_track).abs()
    hinged = F.relu(deviations - radius_hinge_margin)
    within_p = cfg['track_radius_within_norm_p']
    if within_p == 1.0:
        return hinged.mean()
    # Emphasise the worst within-track point: the satisfied_tracks metric requires
    # ALL points in the spiral band, so a per-track p-norm (p>1) over the residuals
    # targets the binding point. +1e-5 mirrors the dt loss, avoiding the x**(1/p)
    # zero-gradient singularity.
    per_track = ((hinged + 1.e-5) ** within_p).mean(dim=-1) ** (1.0 / within_p)
    return per_track.mean()


def _same_radius_loss_for_track_subset(
    slice_to_spiral_transform,
    dr_per_winding,
    prepared_tracks,
    candidate_track_ids,
    num_tracks_per_step,
    num_points_per_track,
):
    zero = torch.zeros([], device=dr_per_winding.device)
    sample = _sample_prepared_track_points(
        prepared_tracks, candidate_track_ids, num_tracks_per_step, num_points_per_track,
    )
    if sample is None:
        return zero
    _, _, sampled_scroll = sample
    k = sampled_scroll.shape[0]
    M = sampled_scroll.shape[1]
    sampled_spiral = slice_to_spiral_transform(sampled_scroll.reshape(-1, 3)).reshape(k, M, 3)
    theta, _, shifted_radii = get_theta_and_radii(sampled_spiral[..., 1:], dr_per_winding)
    shifted_radii = _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding)
    return _same_radius_loss_for_shifted_radii(shifted_radii, dr_per_winding)


def _coverage_loss(residual, tol, tau, reduce='softmin'):
    tau_t = torch.as_tensor(tau, device=residual.device, dtype=residual.dtype).clamp(min=1.e-8)
    if reduce == 'softmin':
        worst = tau_t * torch.logsumexp(residual / tau_t, dim=-1)
        s = torch.sigmoid((tol - worst) / tau_t)
    elif reduce == 'mean':
        s = torch.sigmoid((tol - residual) / tau_t).mean(dim=-1)
    else:
        raise ValueError(f"track_coverage_reduce must be 'softmin' or 'mean', got {reduce!r}")
    return (1.0 - s).mean()


def _coverage_tau(iteration, cfg):
    tau_init = float(cfg['track_coverage_tau_init'])
    tau_final = float(cfg['track_coverage_tau_final'])
    if tau_init <= 0 or tau_final <= 0:
        raise ValueError('track_coverage_tau_init and track_coverage_tau_final must be positive')
    span = max(1, int(cfg['num_training_steps']) - int(cfg['track_em_start_step']))
    f = min(1.0, max(0.0, (iteration - int(cfg['track_em_start_step'])) / span))
    return tau_init * (tau_final / tau_init) ** f


def get_track_em_losses(
    slice_to_spiral_transform,
    dr_per_winding,
    prepared_tracks,
    em_state,
    num_tracks_per_step,
    num_points_per_track,
    tau,
    cfg,
):
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if prepared_tracks is None or em_state is None:
        fallback = _same_radius_loss_for_track_subset(
            slice_to_spiral_transform, dr_per_winding, prepared_tracks, None,
            num_tracks_per_step, num_points_per_track,
        ) if prepared_tracks is not None else zero
        return {'track_radius': fallback, 'track_dt': zero, 'track_coverage': zero}

    valid_track_ids = torch.nonzero(em_state['valid'], as_tuple=False).squeeze(-1)
    invalid_track_ids = torch.nonzero(~em_state['valid'], as_tuple=False).squeeze(-1)

    assigned_radius_loss = zero
    dt_loss = zero
    coverage_loss = zero
    sample = _sample_prepared_track_points(
        prepared_tracks, valid_track_ids, num_tracks_per_step, num_points_per_track,
    )
    if sample is not None:
        track_idx, flat_idx, sampled_scroll = sample
        k = sampled_scroll.shape[0]
        M = sampled_scroll.shape[1]
        m_p = em_state['m_p'][flat_idx].view(k, M).to(dr_per_winding.dtype)
        W = em_state['W'][track_idx].to(dr_per_winding.dtype)[:, None]
        sampled_spiral = slice_to_spiral_transform(sampled_scroll.reshape(-1, 3)).reshape(k, M, 3)
        theta, _, raw_shifted = get_theta_and_radii(sampled_spiral[..., 1:], dr_per_winding)
        unwrapped = raw_shifted + m_p * dr_per_winding
        abs_residual = (unwrapped - W * dr_per_winding).abs()

        if cfg['track_abs_radius_aggregation'] == 'hinge_mean':
            radius_hinge_margin = dr_per_winding.detach() * cfg['track_radius_loss_margin']
            assigned_radius_loss = F.relu(abs_residual - radius_hinge_margin).mean()
        elif cfg['track_abs_radius_aggregation'] == 'coverage':
            radius_tol = dr_per_winding.detach() * cfg['track_coverage_radius_tol']
            assigned_radius_loss = _coverage_loss(
                abs_residual, radius_tol, tau, reduce=cfg['track_coverage_reduce'],
            )
        else:
            raise ValueError(
                "track_abs_radius_aggregation must be 'hinge_mean' or 'coverage', "
                f"got {cfg['track_abs_radius_aggregation']!r}"
            )

        target_raw_radius = (W - m_p) * dr_per_winding + theta / (2 * np.pi) * dr_per_winding
        target_spiral_zyxs = torch.stack([
            sampled_spiral[..., 0],
            torch.sin(theta) * target_raw_radius,
            torch.cos(theta) * target_raw_radius,
        ], dim=-1).detach()
        target_scroll_zyxs = slice_to_spiral_transform.inv(
            target_spiral_zyxs.reshape(-1, 3),
        ).reshape(k, M, 3)

        dt_hinge_margin = dr_per_winding.detach() * cfg['track_dt_loss_margin']
        point_distances = torch.linalg.norm(sampled_scroll - target_scroll_zyxs, dim=-1)
        point_distances = F.relu(point_distances - dt_hinge_margin) + 1.e-5
        within_p = cfg['track_dt_within_track_norm_p']
        across_p = cfg['track_dt_norm_p']
        track_losses = (point_distances ** within_p).mean(dim=-1) ** (1 / within_p)
        dt_loss = ((track_losses ** across_p).sum() / track_losses.numel()) ** (1 / across_p)

        if cfg['loss_weight_track_coverage'] > 0:
            coverage_loss = _coverage_loss(
                point_distances,
                torch.as_tensor(metrics_config['satisfaction_distance_tolerance'], device=device, dtype=point_distances.dtype),
                tau,
                reduce=cfg['track_coverage_reduce'],
            )

    fallback_radius_loss = zero
    if invalid_track_ids.numel() > 0 and cfg['track_em_unassigned_radius_weight'] > 0:
        fallback_radius_loss = _same_radius_loss_for_track_subset(
            slice_to_spiral_transform,
            dr_per_winding,
            prepared_tracks,
            invalid_track_ids,
            num_tracks_per_step,
            num_points_per_track,
        )

    radius_loss = assigned_radius_loss + cfg['track_em_unassigned_radius_weight'] * fallback_radius_loss
    return {'track_radius': radius_loss, 'track_dt': dt_loss, 'track_coverage': coverage_loss}


def get_track_losses(slice_to_spiral_transform, dr_per_winding, prepared_tracks, num_tracks_per_step, num_points_per_track, compute_dt=True, dt_max_winding=None):
    # Sample K tracks (with replacement) and M points per track, transform to spiral space, and
    # compute two losses analogous to the patch radius/DT losses: (1) each point's deviation
    # from the track's mean shifted-radius beyond the radius hinge margin; (2) each point
    # should snap to its target integer winding, with the target taken from the snapped
    # track median.
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if prepared_tracks is None:
        return zero, zero
    sample = _sample_prepared_track_points(prepared_tracks, None, num_tracks_per_step, num_points_per_track)
    if sample is None:
        return zero, zero
    _, _, sampled_scroll = sample
    k = sampled_scroll.shape[0]
    M = sampled_scroll.shape[1]
    sampled_spiral = slice_to_spiral_transform(sampled_scroll.reshape(-1, 3)).reshape(k, M, 3)
    theta, _, shifted_radii = get_theta_and_radii(sampled_spiral[..., 1:], dr_per_winding)
    shifted_radii = _unwrap_track_shifted_radii(theta, shifted_radii, dr_per_winding)
    dt_hinge_margin = dr_per_winding.detach() * cfg['track_dt_loss_margin']
    radius_loss = _same_radius_loss_for_shifted_radii(shifted_radii, dr_per_winding)

    if not compute_dt:
        return radius_loss, zero

    target_shifted_radii = torch.round(shifted_radii.median(dim=-1, keepdim=True).values / dr_per_winding) * dr_per_winding
    target_radii = target_shifted_radii + theta / (2 * np.pi) * dr_per_winding
    target_spiral_zyxs = torch.stack([
        sampled_spiral[..., 0],
        torch.sin(theta) * target_radii,
        torch.cos(theta) * target_radii,
    ], dim=-1).detach()
    target_scroll_zyxs = slice_to_spiral_transform.inv(target_spiral_zyxs.reshape(-1, 3)).reshape(*target_spiral_zyxs.shape)

    within_p = cfg['track_dt_within_track_norm_p']
    across_p = cfg['track_dt_norm_p']
    point_distances = torch.linalg.norm(sampled_scroll - target_scroll_zyxs, dim=-1)
    point_distances = F.relu(point_distances - dt_hinge_margin) + 1.e-5
    track_losses = (point_distances ** within_p).mean(dim=-1) ** (1 / within_p)
    # Progressive DT: only tracks whose snapped winding is within the current cutoff contribute.
    active_mask = _progressive_dt_active_mask(target_shifted_radii.squeeze(-1), dr_per_winding, dt_max_winding)
    dt_loss = _aggregate_dt_track_losses(track_losses, across_p, active_mask)

    return radius_loss, dt_loss


def _build_snap_track_set(slice_to_spiral_transform, dr_per_winding, tracks, device, lower_fraction, upper_fraction, anchor_tree, exclusion_radius):
    # Run get_partially_supported_tracks against the current transform and turn the
    # returned (track, mode_winding) pairs into two flat GPU tensors: per-point
    # scroll zyx and per-point target winding index. Points within `exclusion_radius`
    # of any anchor (cached patch vertex / unattached pcl strip point, indexed by the
    # prebuilt `anchor_tree`) are dropped,
    # since those locations are already well-constrained by the anchor loss.
    # Also returns the subset of the partially-supported tracks that retained at
    # least one point after the exclusion filter — i.e. the tracks actually being
    # snapped to — for visualisation, plus the counts of tracks discarded for
    # being below / above the satisfaction-fraction thresholds.
    # Returns (None, None, 0, [], num_below, num_above) when no tracks remain.
    with torch.no_grad():
        partial, num_below, num_above = get_partially_supported_tracks(
            slice_to_spiral_transform, dr_per_winding, tracks,
            lower_fraction=lower_fraction, upper_fraction=upper_fraction,
        )
    if not partial:
        return None, None, 0, [], num_below, num_above
    zyx_pieces = []
    target_pieces = []
    track_lengths = []
    for track, mode_winding in partial:
        t = np.asarray(track, dtype=np.float32)
        zyx_pieces.append(t)
        target_pieces.append(np.full((len(t),), mode_winding, dtype=np.float32))
        track_lengths.append(len(t))
    flat_zyx_np = np.concatenate(zyx_pieces, axis=0)
    flat_target_np = np.concatenate(target_pieces, axis=0)
    keep_cpu = _track_points_far_from_anchors_mask(flat_zyx_np, anchor_tree, exclusion_radius)
    flat_zyx = torch.from_numpy(flat_zyx_np[keep_cpu]).to(device=device)
    flat_target = torch.from_numpy(flat_target_np[keep_cpu]).to(device=device)
    snapped_tracks = []
    offset = 0
    for (track, _), length in zip(partial, track_lengths):
        if keep_cpu[offset:offset + length].any():
            snapped_tracks.append(track)
        offset += length
    if flat_zyx.shape[0] == 0:
        return None, None, 0, [], num_below, num_above
    return flat_zyx, flat_target, len(partial), snapped_tracks, num_below, num_above


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


def _render_spiral_on_tracks_for_slice(
    spiral_zyx, spiral_density, dr_per_winding,
    slice_z, all_tracks, snapped_tracks,
    out_path, name_suffix,
):
    # Per-slice render of the spiral density (per-winding hued) with track points within
    # a narrow z slab drawn on top; points from tracks in `snapped_tracks` use a brighter
    # palette so it's easy to see which tracks are currently being snapped to.
    z_window = 5
    point_radius = 1
    target_ids = {id(t) for t in snapped_tracks}

    def track_colour(track, is_target):
        # Stable per-track hue from id(track); saturation/value picks the bright
        # ("snap target") vs grey-ish ("not enabled") palette.
        hue = ((id(track) * 2654435761) & 0xFFFFFFFF) / 2 ** 32
        sat, val = (0.9, 1.0) if is_target else (0.35, 0.75)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return (int(r * 255), int(g * 255), int(b * 255))

    _, _, shifted_radius = get_theta_and_radii(spiral_zyx[..., 1:], dr_per_winding)
    winding_idx = (shifted_radius / dr_per_winding).round().to(torch.int64).clamp_min(0)
    num_winding_hues = 6
    hue_min, hue_max = 1.5 / 6, 5.25 / 6
    hue_fraction = hue_min + (winding_idx % num_winding_hues).to(torch.float32) / num_winding_hues * (hue_max - hue_min)
    hue = hue_fraction * 2 * np.pi
    hsv = torch.stack([hue, torch.full_like(hue, 0.5), torch.ones_like(hue)])
    spiral_colours = kornia.color.hsv_to_rgb(hsv).permute(1, 2, 0) * 255
    canvas = (spiral_colours * spiral_density[..., None]).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)

    # Draw non-target tracks first so target tracks paint on top.
    for is_target in (False, True):
        for track in all_tracks:
            if (id(track) in target_ids) != is_target:
                continue
            zs = track[:, 0]
            in_slab = np.abs(zs.astype(np.float32) - float(slice_z)) <= z_window
            if not in_slab.any():
                continue
            colour = track_colour(track, is_target)
            for idx in np.nonzero(in_slab)[0]:
                y = float(track[idx, 1])
                x = float(track[idx, 2])
                draw.ellipse(
                    [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                    fill=colour,
                )
    image.save(f'{out_path}/spiral_on_tracks_s{int(slice_z) * downsample_factor:05}_{name_suffix}.png', compress_level=3)


def save_spiral_on_tracks_overlay(
    spiral_and_transform,
    slice_yx,
    zs_for_visualisation,
    all_tracks,
    snapped_tracks,
    winding_range,
    out_path,
    name_suffix,
):
    slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()
    dr_per_winding = spiral_and_transform.get_dr_per_winding()
    for slice_z in zs_for_visualisation:
        spiral_zyx, spiral_density = _rasterize_spiral_for_slice(
            spiral_and_transform, slice_to_spiral_transform, slice_yx, slice_z, winding_range,
        )
        _render_spiral_on_tracks_for_slice(
            spiral_zyx, spiral_density, dr_per_winding,
            slice_z, all_tracks, snapped_tracks,
            out_path, name_suffix,
        )


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


def fit_spiral_3d(scroll_zarr, patches_dict, point_collections, unattached_pcl_strips, tracks, pcl_normal_samples, lasagna_volume, shell_patch, z_to_umbilicus_yx, out_path, unverified_patches_dict=None):
    patches_list = list(patches_dict.values())
    patch_sampling_probabilities = _prepare_patch_sampling_cache(patches_list, cfg['patch_loss_z_margin'])

    num_patches = len(patches_list)
    print(f'fitting {num_patches} patches')

    patch_atlas = PatchGpuAtlas(patches_dict, device='cuda')
    print(f'patch GPU atlas: {patch_atlas.memory_mb():.1f} MB')

    num_slices_for_visualisation = 20
    rendering_slices_downsample_factor = 2  # stride the scroll by this along zyx for rendering

    device = torch.device('cuda')
    pcl_normal_samples = _pcl_normal_samples_to_gpu(pcl_normal_samples, device)

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
        if (
            cfg['loss_weight_shell_outer'] > 0
            or cfg['loss_weight_shell_no_cross'] > 0
            or cfg['loss_weight_shell_z_drift'] > 0
        ):
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
                    slice_to_spiral_transform, dr_per_winding, tracks,
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
            winding_range, patch_extents, pcl_extents = _compute_winding_range_and_input_extents(
                slice_to_spiral_transform, dr_per_winding, patches_list, unattached_pcl_strips,
            )
            _warn_if_inputs_exceed_flow_bounds(
                list(patches_dict.keys()), patch_extents,
                unattached_pcl_strips, pcl_extents,
                flow_field_radius,
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
            save_mesh(slice_to_spiral_transform, dr_per_winding, patches_list, unattached_pcl_strips, out_path, name=suffix)

    prepared_main_tracks = None
    if (cfg['loss_weight_track_radius'] > 0 or cfg['loss_weight_track_dt'] > 0) and tracks:
        prepared_main_tracks = _prepare_main_phase_tracks(
            tracks, patch_atlas, unattached_pcl_strips,
            float(cfg['track_exclusion_radius']), device,
        )
    use_track_em = bool(cfg['track_loss_use_em']) and prepared_main_tracks is not None
    track_flat = _track_flat_bundle(prepared_main_tracks, device) if use_track_em else None
    track_em_state = None

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

        if cfg['loss_weight_patch_stretch'] > 0 or cfg['loss_weight_patch_normals'] > 0:
            patch_stretch_loss, patch_normals_loss = get_patch_stretch_and_normals_loss(
                slice_to_spiral_transform,
                cfg['regularisation_num_points'],
                patches_list,
                patch_sampling_probabilities,
            )
            losses['patch_stretch'] = patch_stretch_loss * cfg['loss_weight_patch_stretch']
            losses['patch_normals'] = patch_normals_loss * cfg['loss_weight_patch_normals']

        if cfg['loss_weight_bending'] > 0:
            bending_loss = get_bending_loss(
                slice_to_spiral_transform,
                dr_per_winding,
                shell_outer_winding_idx,
                cfg['regularisation_num_points'],
            )
            losses['bending'] = bending_loss * cfg['loss_weight_bending']

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

        if cfg['loss_weight_pcl_normals'] > 0 and pcl_normal_samples is not None:
            losses['pcl_normals'] = get_pcl_normals_loss(
                slice_to_spiral_transform,
                pcl_normal_samples,
                cfg['pcl_normals_num_points'],
            ) * cfg['loss_weight_pcl_normals']

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
            if use_track_em and iteration >= int(cfg['track_em_start_step']):
                reassign_interval = max(1, int(cfg['track_em_reassign_interval']))
                should_reassign = (
                    track_em_state is None
                    or iteration == start_iteration
                    or (iteration - int(cfg['track_em_start_step'])) % reassign_interval == 0
                )
                if should_reassign:
                    prev_W = None if track_em_state is None else track_em_state['W']
                    track_em_state = compute_track_em_assignment(
                        slice_to_spiral_transform, dr_per_winding, track_flat, cfg,
                    )
                    valid = track_em_state['valid']
                    total_tracks = int(valid.numel())
                    valid_tracks = int(valid.sum().item())
                    mean_conf = float(track_em_state['confidence'].mean().item()) if total_tracks > 0 else 0.0
                    changed_msg = ''
                    if prev_W is not None and prev_W.shape == track_em_state['W'].shape:
                        changed = int((prev_W != track_em_state['W']).sum().item())
                        changed_msg = f', changed_windings = {changed}'
                    W_valid = track_em_state['W'][valid].detach().cpu()
                    if W_valid.numel() > 0:
                        uniq, counts = torch.unique(W_valid, return_counts=True)
                        order = torch.argsort(counts, descending=True)
                        top = ', '.join( f'{int(uniq[i].item())}' for i in order[:8] )
                    else:
                        top = 'none'
                    print(
                        f'track EM step {iteration}: valid = {valid_tracks}/{total_tracks}, '
                        f'mean_conf = {mean_conf:.3f}{changed_msg}, top_windings = {top}'
                    )
                    log_metrics['track_em_valid_tracks'] = valid_tracks
                    log_metrics['track_em_total_tracks'] = total_tracks
                    log_metrics['track_em_mean_confidence'] = mean_conf

                tau = _coverage_tau(iteration, cfg)
                log_metrics['track_em_tau'] = tau
                track_losses = get_track_em_losses(
                    slice_to_spiral_transform,
                    dr_per_winding,
                    prepared_main_tracks,
                    track_em_state,
                    cfg['track_num_per_step'],
                    cfg['track_num_points_per_step'],
                    tau,
                    cfg,
                )
                losses['track_radius'] = track_losses['track_radius'] * cfg['loss_weight_track_radius']
                losses['track_dt'] = track_losses['track_dt'] * cfg['loss_weight_track_dt']
                losses['track_coverage'] = track_losses['track_coverage'] * cfg['loss_weight_track_coverage']
            else:
                track_radius_loss, track_dt_loss = get_track_losses(
                    slice_to_spiral_transform,
                    dr_per_winding,
                    prepared_main_tracks,
                    cfg['track_num_per_step'],
                    cfg['track_num_points_per_step'],
                    compute_dt=(not use_track_em) and compute_track_dt,
                    dt_max_winding=track_dt_max_winding,
                )
                losses['track_radius'] = track_radius_loss * cfg['loss_weight_track_radius']
                losses['track_dt'] = track_dt_loss * cfg['loss_weight_track_dt']

        shell_metrics = {}
        if shell_map is not None:
            shell_outer_loss, shell_no_cross_loss, shell_z_drift_loss, shell_metrics = get_shell_losses(
                shell_map,
                slice_to_spiral_transform,
                dr_per_winding,
                shell_outer_winding_idx,
            )
            losses['shell_outer'] = shell_outer_loss * cfg['loss_weight_shell_outer']
            losses['shell_no_cross'] = shell_no_cross_loss * cfg['loss_weight_shell_no_cross']
            losses['shell_z_drift'] = shell_z_drift_loss * cfg['loss_weight_shell_z_drift']

        losses['umbilicus'] = umbilicus_loss * cfg['loss_weight_umbilicus']

        loss = sum(losses.values())

        loss.backward()
        spiral_and_transform.flow_field.apply_accumulated_field_grad()
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

    num_subpasses = int(cfg['num_snapping_subpasses'])
    steps_per_subpass = max(1, int(cfg['snapping_steps_per_subpass']))
    num_snapping_steps = num_subpasses * steps_per_subpass
    if num_subpasses > 0:
        print(f'starting snapping phase: {num_subpasses} sub-passes x {steps_per_subpass} steps')

        # Snapshot every patch valid in-ROI vertex and every PCL strip point in
        # spiral space; the snapping phase pulls them back towards these targets.
        slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()
        anchor_scroll, anchor_spiral_target = _build_snap_anchors(
            slice_to_spiral_transform, patch_atlas, unattached_pcl_strips, device,
        )
        print(f'snap anchors: {anchor_scroll.shape[0]} points')
        anchor_tree = _build_anchor_kdtree(anchor_scroll)

        snap_anchor_n = int(cfg['snapping_num_anchor_points'])
        snap_track_n = int(cfg['snapping_num_track_points'])
        snap_lower = float(cfg['snapping_tracks_lower_fraction'])
        snap_upper = float(cfg['snapping_tracks_upper_fraction'])
        snap_exclusion_radius = float(cfg['snapping_track_exclusion_radius'])
        snap_lr_min = float(cfg['learning_rate']) * float(cfg['lr_final_factor'])
        snap_lr_max = float(cfg['snapping_max_lr'])

        snap_iter = 0
        pbar = tqdm(total=num_snapping_steps)
        for subpass_idx in range(num_subpasses):
            slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()
            dr_per_winding = spiral_and_transform.get_dr_per_winding()

            snap_track_scroll, snap_track_target_winding, num_partial, snap_target_tracks, num_below, num_above = _build_snap_track_set(
                slice_to_spiral_transform, dr_per_winding, tracks, device,
                lower_fraction=snap_lower, upper_fraction=snap_upper,
                anchor_tree=anchor_tree, exclusion_radius=snap_exclusion_radius,
            )
            num_points = 0 if snap_track_scroll is None else snap_track_scroll.shape[0]
            print(f'snap iter {snap_iter}: working set has {num_partial} tracks ({num_points} points); discarded {num_below} below and {num_above} above satisfaction thresholds')
            if os.environ.get('FIT_SPIRAL_SKIP_SAVE_OVERLAY') != '1':
                snap_winding_range, _, _ = _compute_winding_range_and_input_extents(
                    slice_to_spiral_transform, dr_per_winding, patches_list, unattached_pcl_strips,
                )
                save_spiral_on_tracks_overlay(
                    spiral_and_transform,
                    slice_yx,
                    zs_for_visualisation,
                    tracks,
                    snap_target_tracks,
                    snap_winding_range,
                    out_path,
                    f'iter{snap_iter:04}',
                )

            for inner_iter in range(steps_per_subpass):
                frac = inner_iter / (steps_per_subpass - 1)
                lr = snap_lr_min + (snap_lr_max - snap_lr_min) * 0.5 * (1.0 - np.cos(2.0 * np.pi * frac))
                for pg in optimiser.param_groups:
                    pg['lr'] = lr

                slice_to_spiral_transform = spiral_and_transform.get_slice_to_spiral_transform()
                dr_per_winding = spiral_and_transform.get_dr_per_winding()

                losses = {}

                n_anchor = anchor_scroll.shape[0]
                k = min(snap_anchor_n, n_anchor)
                sel = torch.randint(n_anchor, (k,), device=device)
                sample_scroll = anchor_scroll[sel]
                sample_target_spiral = anchor_spiral_target[sel]
                current_spiral = slice_to_spiral_transform(sample_scroll)
                anchor_loss = (current_spiral - sample_target_spiral).pow(2).sum(dim=-1).mean()
                losses['snap_anchor'] = anchor_loss * cfg['loss_weight_snap_anchor']

                if snap_track_scroll is not None:
                    n_track = snap_track_scroll.shape[0]
                    k = min(snap_track_n, n_track)
                    sel = torch.randint(n_track, (k,), device=device)
                    sample_scroll = snap_track_scroll[sel]
                    sample_target_winding = snap_track_target_winding[sel]
                    sample_spiral = slice_to_spiral_transform(sample_scroll)
                    _, _, sample_shifted_radii = get_theta_and_radii(sample_spiral[..., 1:], dr_per_winding)
                    target_shifted_radii = sample_target_winding * dr_per_winding
                    track_loss = (sample_shifted_radii - target_shifted_radii).abs().mean()
                    losses['snap_tracks'] = track_loss * cfg['loss_weight_snap_tracks']

                umbilicus_spiral = slice_to_spiral_transform(umbilicus_zyx)
                umbilicus_loss = umbilicus_spiral[..., 1:].abs().mean()
                losses['umbilicus'] = umbilicus_loss * cfg['loss_weight_umbilicus']

                if cfg['loss_weight_sym_dirichlet'] > 0:
                    sym_dirichlet_loss = get_symmetric_dirichlet_loss(
                        slice_to_spiral_transform,
                        dr_per_winding,
                        shell_outer_winding_idx,
                        cfg['regularisation_num_points'],
                    )
                    losses['sym_dirichlet'] = sym_dirichlet_loss * cfg['loss_weight_sym_dirichlet']

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

                loss = sum(losses.values())

                loss.backward()
                spiral_and_transform.flow_field.apply_accumulated_field_grad()
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)

                if snap_iter % 50 == 0:
                    print(f'snap step {snap_iter}: loss = {loss.item():.1f}, lr = {lr:.2e}, ' + ', '.join(f'{name} = {value.item():.1f}' for name, value in losses.items()))
                    if loss.isnan().item():
                        print('aborting due to NaN')
                        return
                    wandb.log({
                        'snap_total_loss': loss.item(),
                        'snap_lr': lr,
                        **{f'snap_{name}_loss': value.item() for name, value in losses.items()},
                    })

                snap_iter += 1
                pbar.update(1)
        pbar.close()

    if num_subpasses > 0:
        suffix = 'snapped'
        save_model(suffix)
        save_overlay_and_print_satisfaction(suffix)


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


def link_points_to_patches_cached(
    patches,
    point_collections,
    tolerance=10.0,
    surface_index_tolerance=None,
    distance_scale=1.0,
):
    patch_ids_str = '_'.join(sorted(patches.keys()))
    hashed_patches = hashlib.sha256(bytes(patch_ids_str, 'ascii')).hexdigest()[:8]

    pcl_locations = []
    for collection_id in sorted(point_collections.keys()):
        collection = point_collections[collection_id]
        for point_id in sorted(collection['points'].keys()):
            point = collection['points'][point_id]
            pcl_locations.append(tuple(point['p']))
    hashed_pcls = hashlib.sha256(pickle.dumps(pcl_locations)).hexdigest()[:8]

    backend = 'surface-index' if surface_index_tolerance is not None and can_use_surface_index_backend(patches) else 'torch'
    cache_filename = (
        f'{cache_path}/point_patch_links-{hashed_patches}_{hashed_pcls}'
        f'_backend-{backend}_tol-{tolerance}_idx-tol-{surface_index_tolerance}.pkl'
    )

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as fp:
            point_collections_cached = pickle.load(fp)
        point_collections.clear()
        point_collections.update(point_collections_cached)
    else:
        link_points_to_patches(
            patches,
            point_collections,
            tolerance,
            surface_index_tolerance=surface_index_tolerance,
            distance_scale=distance_scale,
        )
        os.makedirs(cache_path, exist_ok=True)
        with open(cache_filename, 'wb') as fp:
            pickle.dump(point_collections, fp)


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
    # hold a single point; we assert that every one of their points attached to a patch and
    # carries an explicit winding annotation (an absolute pcl must not fall back to winding 0),
    # and (once grouped below) that no patch holds more than one of their points.
    cross_patch_point_collections = {}
    unattached_point_collections = {}
    for pid, pcl in point_collections.items():
        num_attached = sum(1 for point in pcl['points'].values() if 'on_patch' in point)
        num_unattached = len(pcl['points']) - num_attached
        if pcl.get('metadata', {}).get('winding_is_absolute', False):
            assert num_unattached == 0, (
                f'winding_is_absolute pcl {pid} ({pcl.get("name")!r}) has {num_unattached} of '
                f'{len(pcl["points"])} points not attached to any patch; expected all attached'
            )
            num_unannotated = sum(1 for point in pcl['points'].values() if not np.isfinite(point['winding_annotation']))
            assert num_unannotated == 0, (
                f'winding_is_absolute pcl {pid} ({pcl.get("name")!r}) has {num_unannotated} of '
                f'{len(pcl["points"])} points without a winding annotation; absolute pcls must '
                f'give every winding number explicitly'
            )
            num_non_positive = sum(1 for point in pcl['points'].values() if point['winding_annotation'] <= 0)
            assert num_non_positive == 0, (
                f'winding_is_absolute pcl {pid} ({pcl.get("name")!r}) has {num_non_positive} of '
                f'{len(pcl["points"])} points with a non-positive winding annotation; absolute '
                f'winding numbers must be > 0'
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
        # An absolute-winding pcl carries one absolute winding per patch (one sheet ->
        # one winding); two annotated points on the same patch would hand the abs-winding
        # loss conflicting targets for the same sheet, so require at most one per patch.
        if pcl.get('metadata', {}).get('winding_is_absolute', False):
            multi_point_patches = {pid: len(pts) for pid, pts in points_by_patch.items() if len(pts) > 1}
            assert not multi_point_patches, (
                f'winding_is_absolute pcl ({pcl.get("name")!r}) attaches multiple points to '
                f'patches {multi_point_patches}; each absolute pcl must attach at most one '
                f'point per patch'
            )
    unattached_pcl_strips = _prepare_unattached_pcl_strips(unattached_point_collections, cfg['unattached_pcl_min_point_spacing'])
    print(
        f'pcls: {len(cross_patch_point_collections)} cross-patch, '
        f'{len(unattached_pcl_strips)} unattached'
    )

    return cross_patch_point_collections, unattached_pcl_strips


def load_tracks_from_dbm(path, z_lo, z_hi):
    # Load tracks written by extract_surface_tracks.py. Each DBM value
    # is a pickled list of (N, 3) int32 zyx arrays; we keep only tracks that
    # lie entirely within [z_lo, z_hi).
    z_lo_raw = z_lo * downsample_factor
    z_hi_raw = z_hi * downsample_factor
    tracks = []
    with dbm.open(path, 'r') as db:
        for key in tqdm(db.keys(), desc='loading tracks'):
            entries = pickle.loads(db[key])
            if not entries:
                continue
            # Vectorize the per-track z min/max across the whole key: concatenate
            # every (non-empty) track's z column and reduce per segment with
            # reduceat, rather than calling .min()/.max() once per track .
            idx = [i for i in range(len(entries)) if len(entries[i])]
            if not idx:
                continue
            lengths = np.fromiter((len(entries[i]) for i in idx), dtype=np.intp, count=len(idx))
            zcat = np.concatenate([entries[i][:, 0] for i in idx])
            offsets = np.zeros(len(idx), dtype=np.intp)
            np.cumsum(lengths[:-1], out=offsets[1:])
            zmins = np.minimum.reduceat(zcat, offsets)
            zmaxs = np.maximum.reduceat(zcat, offsets)
            keep = (zmins >= z_lo_raw) & (zmaxs < z_hi_raw)
            for j in np.nonzero(keep)[0]:
                tracks.append(entries[idx[j]].astype(np.float32) / downsample_factor)
    return tracks


def erode_patch_valid_region(patch, num_cells):
    """Erode the patch's valid-vertex region inward by `num_cells` grid cells,
    marking the removed vertices invalid (zyxs = -1) and re-deriving the patch.

    Returns True if any valid quad survives, False if the patch was fully eroded
    (in which case the caller should drop it).
    """
    valid = patch.valid_vertex_mask.cpu().numpy()
    # border_value=0 so the patch outline is treated as invalid, eroding edges too.
    eroded = scipy.ndimage.binary_erosion(valid, iterations=num_cells, border_value=0)
    remove = valid & ~eroded
    if not remove.any():
        return True
    patch.zyxs[torch.from_numpy(remove)] = -1.0
    new_valid_vertex = torch.any(patch.zyxs != -1, dim=-1)
    new_valid_quad = (
        new_valid_vertex[:-1, :-1] & new_valid_vertex[1:, :-1]
        & new_valid_vertex[:-1, 1:] & new_valid_vertex[1:, 1:]
    )
    if not bool(new_valid_quad.any()):
        return False
    patch.__post_init__()  # re-derive valid_vertex_mask / valid_quad_mask / area / valid_zyxs
    return True


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
    pcl_normal_samples = prepare_pcl_normal_samples(
        list(cross_patch_point_collections.values()),
        unattached_pcl_strips,
        scroll_zarr_array,
    )
    lasagna_volume = prepare_lasagna_volume(scroll_zarr_array)

    if tracks_dbm_path is not None:
        print(f'loading tracks from {tracks_dbm_path}')
        tracks = load_tracks_from_dbm(tracks_dbm_path, z_begin, z_end)
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
        pcl_normal_samples,
        lasagna_volume,
        shell_patch,
        umbilicus,
        out_path,
        unverified_patches_dict=unverified_patches,
    )


if __name__ == '__main__':
    config = dict(default_config)
    config.update(get_env_config_overrides())
    z_range_scale, z_range_num_slices = scale_counts_for_z_range(config)
    print(
        f'scaled per-step counts by {z_range_scale:.3f} for the {z_range_num_slices}-slice '
        f'z-range [{z_begin * downsample_factor}, {z_end * downsample_factor}) '
        f'(reference {reference_z_range_num_slices} slices):\n  '
        + '\n  '.join(f'{k}={config[k]}' for k in z_range_scaled_count_keys)
    )
    wandb.init(project='scrolls', config=config)
    cfg = wandb.config
    main()
