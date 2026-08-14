import os
import sys

# Expandable segments stop the CUDA caching allocator from ratcheting its
# reserved pool toward the VRAM ceiling under the variable-size per-step loss
# graphs (measured ~12 GB lower steady-state envelope on the s1 fit). Must be
# set before the allocator initialises; an explicit PYTORCH_CUDA_ALLOC_CONF in
# the environment always wins.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import copy
import gc
import json
import glob
from collections.abc import Mapping
import zarr
import torch
import wandb
import datetime
import time
import numpy as np
import scipy.ndimage
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm import tqdm

from ddp_helpers import (
    StepTimer,
    DistributedContext,
    allreduce_grads_,
    broadcast_model_params,
    configure_torch_threads_from_env,
    is_main_process,
    maybe_destroy_distributed,
    maybe_init_distributed,
    process_context,
)
from config import (CHECKPOINT_MODEL_SHAPE_KEYS, Config, FitConfig,
                    durable_config)
from fit_session import fit_input
from lasagna_data import prepare_lasagna_volume, prepare_surf_sdt_volume
from checkpoint_io import load_checkpoint_cpu
from influence import make_influence_state, subsample_rows
from native_spiral import load_native_spiral_sampling
from tifxyz import load_tifxyz
from geom_utils import bilinear_atlas_lookup, interp1d
from point_collection import (
    link_points_to_patches,
    load_point_collection,
    normalise_pcl_winding_annotations,
)
from dt_targets import (
    DtTargetCacheManager,
    compute_patch_dt_target_cache,
    compute_strip_dt_target_cache,
    prepare_patch_dt_target_samples,
)
from tracks import (
    PackedTrackCollection,
    configure_prepared_track_sampling,
    filter_tracks_to_outer_shell,
    get_track_satisfied_counts_in_chunks,
    iter_track_losses,
    load_track_crossing_cache,
    load_tracks_from_dbm,
    prepare_main_phase_tracks,
    validate_track_sampling_config,
)
from track_graph import TrackGraph
from umbilicus import thaumato_umbilicus_z_to_yx, json_umbilicus_z_to_yx
from sample_spiral import (
    get_spiral_points,
    get_theta,
    get_winding_xy,
)
import strip_path_pools
from losses import (
    build_pcl_sampling_strata,
    iter_lasagna_losses,
    get_patch_abs_winding_loss,
    get_patch_and_umbilicus_losses,
    get_patch_rel_winding_loss,
    get_shell_outer_loss,
    get_symmetric_dirichlet_loss,
    get_unattached_pcl_strip_losses,
    get_unverified_patch_losses,
)
from loss_maps import (LossMapRecorder, attach_loss_maps_to_manifest,
                       capture_loss_maps)
from sdt_losses import (
    aggregate_pair_counts,
    iter_phase_bundle_losses,
    phase_bundle_component_weights,
)
from spiral_helpers import (
    REFERENCE_Z_RANGE_NUM_SLICES,
    erode_patch_valid_region,
    load_patches,
    load_fiber_point_collection,
    load_fiber_point_collections,
    _decimate_ordered_points_min_spacing,
    resolve_fiber_links,
    build_link_components,
    merge_linked_point_collections,
    SequenceChain,
    scale_and_split_counts,
    _infer_shell_outer_winding_idx,
    _structurally_disabled_dense_weight_keys,
    resolve_outer_winding_idx_and_notes,
    patch_intersects_z_roi,
    save_combined_preview,
)
from satisfaction_metrics import (
    get_patch_satisfied_areas as _get_patch_satisfied_areas,
    get_unattached_pcl_satisfied_counts as _get_unattached_pcl_satisfied_counts,
    metrics_config,
    save_overlay_and_print_satisfaction,
)
from visualization import overlay_patches_on_slices
from transforms import SpiralAndTransform
from winding_supervision import (
    get_winding_inference_losses,
    load_winding_inference_store,
)
from spiral_progress import ProgressReporter, progress_or_null


configure_torch_threads_from_env()


# Fields of a surf-SDT fingerprint that describe where the store lives and how
# much of it has been built, rather than what it contains.
_SDT_COVERAGE_AND_LOCATION_KEYS = (
    'path', 'source', 'complete', 'z_range_working', 'built_z_ranges_working',
)


def comparable_sdt_fingerprint(fingerprint):
    """The content-identity subset of a surf-SDT fingerprint."""
    if not fingerprint:
        return None
    return {key: value for key, value in fingerprint.items()
            if key not in _SDT_COVERAGE_AND_LOCATION_KEYS}


class CheckpointVerdict:
    """The result of one CPU-side checkpoint preflight.

    A verdict is a value, not an exception: an all-rank load has to collect
    one from every rank and only then decide, so phase 1 must be able to
    refuse without unwinding anything.
    """

    __slots__ = ('accepted', 'reasons', 'completed_iterations', 'source')

    def __init__(self, accepted, reasons=(), completed_iterations=None,
                 source=''):
        self.accepted = bool(accepted)
        self.reasons = tuple(reasons)
        self.completed_iterations = completed_iterations
        self.source = source

    def message(self):
        source = f' {self.source}' if self.source else ''
        if self.accepted:
            return f'checkpoint{source} is compatible with this fit'
        return (f'checkpoint{source} is not compatible with this fit:\n  - '
                + '\n  - '.join(self.reasons))

    def to_dict(self):
        return {
            'accepted': self.accepted,
            'reasons': list(self.reasons),
            'completed_iterations': self.completed_iterations,
            'source': self.source,
        }

    def __repr__(self):
        return (f'CheckpointVerdict(accepted={self.accepted!r}, '
                f'reasons={self.reasons!r})')


def get_env_config_overrides():
    overrides_json = os.environ.get('FIT_SPIRAL_CONFIG_OVERRIDES')
    if not overrides_json:
        return {}
    overrides = json.loads(overrides_json)
    unknown_keys = sorted(set(overrides) - set(Config().as_dict()))
    if unknown_keys:
        raise KeyError(f'unknown FIT_SPIRAL_CONFIG_OVERRIDES keys: {unknown_keys}')
    return overrides


# The per-step object-sample counts are tuned for the z 7000-16500 range
# (~9500 full-resolution slices). For a smaller/larger z-range each loss term sees
# proportionally fewer/more objects, so scale_counts_for_z_range() scales these
# counts linearly with the number of slices (points-PER-object stays fixed).
class ShellPolarMap:

    def __init__(self, shell_patch, z_to_umbilicus_yx, z_min, z_max, num_theta_bins, device, *, config):
        self._config = config
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

        sigma = (config['shell_table_smooth_sigma_z'], config['shell_table_smooth_sigma_theta'])
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
        valid = in_z & (confidence >= self._config['shell_min_confidence'])
        return target_radius, radius, confidence, valid


class PatchAtlas:
    """All patches' (H, W, 3) zyxs grids packed into one flat tensor, batched
    per lookup instead of per-patch dispatch. The packed grids stay resident in
    host memory: the (i, j) samples are drawn on the CPU anyway, so the
    bilinear gather runs there (mostly-contiguous strip reads, a few MB per
    step) and only the interpolated points are uploaded to `device`. This
    keeps the atlas - which scales with the input patch count, not with any
    per-step budget - out of VRAM entirely."""

    def __init__(self, patches_by_id, device='cuda'):
        self.device = torch.device(device)
        flat_pieces = []
        offsets = [0]
        widths = []
        heights = []
        for p in patches_by_id.values():
            z = p.zyxs  # (H, W, 3) on CPU
            H, W = z.shape[:2]
            z_flat = z.reshape(-1, 3).to(dtype=torch.float32)
            flat_pieces.append(z_flat)
            offsets.append(offsets[-1] + H * W)
            widths.append(W)
            heights.append(H)
        self.zyxs_flat = (
            torch.cat(flat_pieces, dim=0)
            if flat_pieces
            else torch.empty([0, 3], dtype=torch.float32))
        self.offsets = torch.tensor(offsets, dtype=torch.int64)  # (N+1,)
        self.widths = torch.tensor(widths, dtype=torch.int64)  # (N,)
        self.heights = torch.tensor(heights, dtype=torch.int64)  # (N,)
        self.id_to_idx = {pid: i for i, pid in enumerate(patches_by_id.keys())}
        native = load_native_spiral_sampling()
        self.sampling_atlas = (
            native.PatchSamplingAtlas([
                np.ascontiguousarray(p._sampling_valid_quad_mask_np, dtype=bool)
                for p in patches_by_id.values()
            ])
            if native is not None and patches_by_id else None
        )

    def memory_mb(self):
        return self.zyxs_flat.numel() * 4 / 1e6

    def lookup(self, patch_idx_per_sample, ijs):
        # patch_idx_per_sample: (...,) int64 on CPU
        # ijs: (..., 2) float on CPU
        # Gathers and interpolates on the host-resident atlas and returns
        # (..., 3) on self.device. Caller must ensure floor(ij) lies on a
        # valid quad. Runs inside the batch prefetcher when that is enabled,
        # so both the gather and the upload happen a step ahead.
        assert not patch_idx_per_sample.is_cuda and not ijs.is_cuda, (
            'the atlas is host-resident: pass CPU indices/ijs; only the '
            'interpolated points are uploaded')
        zyxs = bilinear_atlas_lookup(
            self.zyxs_flat,
            self.offsets,
            self.widths,
            patch_idx_per_sample,
            ijs,
            heights=self.heights,
        )
        return zyxs.to(device=self.device, non_blocking=True)

    def append_patches(self, patches_by_id):
        """Append new patches without rebuilding the resident atlas.

        A host-side concatenation of just the new grids, so a resident
        interactive session can incorporate a handful of added patches in
        seconds.
        """
        if not patches_by_id:
            return
        flat_pieces = []
        offsets = [int(self.offsets[-1].item())]
        widths = []
        heights = []
        for pid, p in patches_by_id.items():
            if pid in self.id_to_idx:
                raise ValueError(f'Patch {pid!r} is already in the atlas')
            z = p.zyxs
            H, W = z.shape[:2]
            flat_pieces.append(z.reshape(-1, 3).to(dtype=torch.float32))
            offsets.append(offsets[-1] + H * W)
            widths.append(W)
            heights.append(H)
        new_flat = torch.cat(flat_pieces, dim=0)
        self.zyxs_flat = torch.cat([self.zyxs_flat, new_flat], dim=0)
        self.offsets = torch.cat([
            self.offsets,
            torch.tensor(offsets[1:], dtype=torch.int64),
        ])
        self.widths = torch.cat([
            self.widths, torch.tensor(widths, dtype=torch.int64)])
        self.heights = torch.cat([
            self.heights, torch.tensor(heights, dtype=torch.int64)])
        next_idx = len(self.id_to_idx)
        for pid in patches_by_id:
            self.id_to_idx[pid] = next_idx
            next_idx += 1
        masks = [
            np.ascontiguousarray(p._sampling_valid_quad_mask_np, dtype=bool)
            for p in patches_by_id.values()
        ]
        if self.sampling_atlas is not None:
            self.sampling_atlas.append(masks)
        else:
            native = load_native_spiral_sampling()
            if native is not None:
                self.sampling_atlas = native.PatchSamplingAtlas(masks)


class _UnattachedPclStripList(list):
    """List of unattached-pcl strip dicts, with a slot for an attached `.flat`
    GPU bundle that batched satisfaction / winding-range computations reuse."""
    pass


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


def get_progressive_dt_max_winding(cfg, iteration, dt_start_step, shell_outer_winding_idx):
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


def get_interactive_dt_resume_iteration(start_iteration, target_iteration,
                                        disabled_fraction=0.75):
    """Return the first iteration that may use DT losses after new inputs.

    Input incorporation happens at the start of an interactive run. Keep DT
    losses disabled for the requested fraction of that run so the radius-based
    losses can settle the newly added geometry before directional constraints
    resume.
    """
    run_iterations = max(0, int(target_iteration) - int(start_iteration))
    fraction = min(1.0, max(0.0, float(disabled_fraction)))
    return int(start_iteration) + int(run_iterations * fraction)


def unresolved_fiber_link_warning(new_fibers, *, use_links, use_pending_links,
                                  max_named=6):
    """Warn that uploaded fibers' cross-fiber links are inert this session.

    Cross-fiber links are resolved once, over the resident inputs
    (load_host_inputs): an uploaded fiber joins the pool as its own singleton
    component, so any branch the user drew on it does nothing until the input
    is committed and the fit rebuilt. Rather than let that pass silently, count
    the links that *would* be used after a rebuild -- so nothing is reported
    when links are configured off, and pending links only count when they are
    configured in -- and describe them.

    new_fibers is an iterable of (input_id, pcl). Returns the warning text, or
    None when there is nothing to warn about.
    """
    if not use_links:
        return None
    counted = []
    for input_id, pcl in new_fibers:
        num_links = sum(1 for branch in pcl.get('branches', ())
                        if use_pending_links or not branch['pending'])
        if num_links:
            counted.append((input_id, num_links))
    if not counted:
        return None
    named = ', '.join(f'{input_id} ({count})' for input_id, count in counted[:max_named])
    if len(counted) > max_named:
        named += f', and {len(counted) - max_named} more'
    return (
        f'{sum(count for _, count in counted)} cross-fiber link(s) on '
        f'{len(counted)} added fiber(s) are not used by this session: links are '
        f'resolved when the fit is built. Commit the inputs and rebuild the fit '
        f'to apply them. Fibers: {named}')


def get_dense_attachment_ramp(cfg, iteration):
    """Warm-up/ramp factor for the attachment weight, measured against the
    durable completed-iteration count so a resumed run continues the schedule
    instead of restarting it."""
    warmup = int(cfg['dense_attachment_warmup_steps'])
    ramp = int(cfg['dense_attachment_ramp_steps'])
    if iteration < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, (iteration - warmup + 1) / ramp)


def get_exponential_lr_at_step(
        initial_lr, final_factor, completed_steps, training_horizon):
    """LR on the absolute exponential curve for a completed-step count."""
    horizon = max(1, int(training_horizon))
    completed = max(0, int(completed_steps))
    gamma = float(final_factor) ** (1.0 / horizon)
    return float(initial_lr) * gamma ** completed


def get_flow_field_high_res_lr_scale(cfg, iteration):
    """Relative optimizer LR for the high-resolution flow lattices."""
    initial = cfg['model_flow_field_high_res_lr_scale_initial']
    final = cfg['model_flow_field_high_res_lr_scale_final']
    start_step = cfg['model_flow_field_high_res_lr_ramp_start_step']
    ramp_steps = max(
        1, int(cfg['model_flow_field_high_res_lr_ramp_steps']))
    fraction = min(
        1., max(0., (int(iteration) - int(start_step)) / ramp_steps))
    return min(1., float(initial) + fraction * (float(final) - float(initial)))


def set_optimizer_group_lr_scale(
        optimiser, lr_scheduler, *, group, reference_group, scale,
        initial_lr):
    """Set one parameter group's LR relative to another optimizer group."""
    scale = float(scale)
    group_index = next(
        index for index, candidate in enumerate(optimiser.param_groups)
        if candidate is group)
    group['lr_scale'] = scale
    group['initial_lr'] = float(initial_lr) * scale
    group['lr'] = float(reference_group['lr']) * scale
    lr_scheduler.base_lrs[group_index] = group['initial_lr']
    if len(lr_scheduler._last_lr) == len(optimiser.param_groups):
        lr_scheduler._last_lr[group_index] = group['lr']


def realign_optimizer_lr_schedule(
        optimiser, lr_scheduler, *, initial_lr, final_factor,
        completed_steps, training_horizon, exponential):
    """Realign an optimizer and scheduler to an absolute training step."""
    horizon = max(1, int(training_horizon))
    completed = max(0, int(completed_steps))
    initial_lr = float(initial_lr)

    if exponential:
        gamma = float(final_factor) ** (1.0 / horizon)
        if not isinstance(
                lr_scheduler, torch.optim.lr_scheduler.ExponentialLR):
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimiser, gamma=gamma)
        lr_scheduler.gamma = gamma
        aligned_lr = get_exponential_lr_at_step(
            initial_lr, final_factor, completed, horizon)
    else:
        if not isinstance(lr_scheduler, torch.optim.lr_scheduler.LambdaLR):
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimiser, lambda step: 1.)
        aligned_lr = initial_lr

    lr_scales = [
        float(group.get('lr_scale', 1.))
        for group in optimiser.param_groups
    ]
    base_lrs = [initial_lr * scale for scale in lr_scales]
    aligned_lrs = [aligned_lr * scale for scale in lr_scales]
    for group, base_lr, current_lr in zip(
            optimiser.param_groups, base_lrs, aligned_lrs):
        group['initial_lr'] = base_lr
        group['lr'] = current_lr
    lr_scheduler.base_lrs = base_lrs
    lr_scheduler.last_epoch = completed
    lr_scheduler._last_lr = aligned_lrs
    lr_scheduler._step_count = completed + 1
    return lr_scheduler, horizon


def _query_near_trusted_geometry(points_np, trusted_geometry_tree, threshold):
    # Returns True for each point with at least one trusted-geometry anchor
    # within `threshold`. query returns dist == inf for misses.
    points_np = np.ascontiguousarray(points_np, dtype=np.float32)
    dist, _ = trusted_geometry_tree.query(
        points_np,
        k=1,
        distance_upper_bound=float(threshold),
        workers=-1,
    )
    return np.isfinite(dist)


def _apply_unverified_patch_trusted_mask(patch, vertices_to_invalidate):
    if not vertices_to_invalidate.any():
        return 0, False

    invalid_mask_2d = torch.from_numpy(vertices_to_invalidate.reshape(patch.zyxs.shape[:2]))
    patch.zyxs[invalid_mask_2d] = -1.0
    n_masked = int(vertices_to_invalidate.sum())

    new_valid_vertex_mask = torch.any(patch.zyxs != -1, dim=-1)
    new_valid_quad_mask = (
        new_valid_vertex_mask[:-1, :-1]
        & new_valid_vertex_mask[1:, :-1]
        & new_valid_vertex_mask[:-1, 1:]
        & new_valid_vertex_mask[1:, 1:]
    )

    if not bool(new_valid_quad_mask.any()):
        return n_masked, True

    patch.__post_init__()
    return n_masked, False


def _mask_unverified_patches_near_trusted_geometry(
    unverified_patches,
    trusted_geometry_tree,
    threshold,
    max_query_points=2_000_000,
):
    if threshold <= 0 or trusted_geometry_tree is None:
        return dict(unverified_patches), 0, 0

    kept_unverified_patches = {}
    n_masked_vertices = 0
    n_dropped_patches = 0

    batch_entries = []
    batch_points = []
    batch_total = 0

    def flush_batch():
        nonlocal batch_entries, batch_points, batch_total
        nonlocal n_masked_vertices, n_dropped_patches

        if batch_total == 0:
            return

        points_np = batch_points[0] if len(batch_points) == 1 else np.concatenate(batch_points, axis=0)
        near_trusted = _query_near_trusted_geometry(points_np, trusted_geometry_tree, threshold)

        offset = 0
        for patch_id, patch, valid_indices in batch_entries:
            n_valid = len(valid_indices)
            patch_near_trusted = near_trusted[offset:offset + n_valid]
            offset += n_valid

            vertices_to_invalidate = np.zeros(patch.zyxs.shape[0] * patch.zyxs.shape[1], dtype=bool)
            vertices_to_invalidate[valid_indices[patch_near_trusted]] = True
            n_masked, dropped = _apply_unverified_patch_trusted_mask(patch, vertices_to_invalidate)
            n_masked_vertices += n_masked
            if dropped:
                n_dropped_patches += 1
            else:
                kept_unverified_patches[patch_id] = patch

        batch_entries = []
        batch_points = []
        batch_total = 0

    for patch_id, patch in unverified_patches.items():
        zyxs_flat = patch.zyxs.reshape(-1, 3).cpu().numpy()
        valid_flat = patch.valid_vertex_mask.reshape(-1).cpu().numpy()
        valid_indices = np.flatnonzero(valid_flat)

        if len(valid_indices) == 0:
            kept_unverified_patches[patch_id] = patch
            continue

        if len(valid_indices) > max_query_points:
            flush_batch()
            vertices_to_invalidate = np.zeros(len(valid_flat), dtype=bool)
            for start in range(0, len(valid_indices), max_query_points):
                chunk_indices = valid_indices[start:start + max_query_points]
                near_trusted = _query_near_trusted_geometry(
                    zyxs_flat[chunk_indices],
                    trusted_geometry_tree,
                    threshold,
                )
                vertices_to_invalidate[chunk_indices[near_trusted]] = True

            n_masked, dropped = _apply_unverified_patch_trusted_mask(patch, vertices_to_invalidate)
            n_masked_vertices += n_masked
            if dropped:
                n_dropped_patches += 1
            else:
                kept_unverified_patches[patch_id] = patch
            continue

        if batch_total + len(valid_indices) > max_query_points:
            flush_batch()

        batch_entries.append((patch_id, patch, valid_indices))
        batch_points.append(zyxs_flat[valid_indices])
        batch_total += len(valid_indices)

    flush_batch()
    return kept_unverified_patches, n_masked_vertices, n_dropped_patches


class FitContext:
    """Owner of all mutable state and resources for one spiral fit.

    State falls into four ownership classes:

    (a) immutable dataset/path descriptors;
    (b) host-prepared inputs and caches;
    (c) session-owned CUDA/device resources;
    (d) model, optimiser, scheduler, RNG, and iteration state.

    Threading rule: all methods that touch Torch or CUDA are executed only
    by the fitter thread for that rank; HTTP/coordinator threads submit
    commands instead of calling the context directly.
    """

    def __init__(self, config, *, scroll, paths, interactive_driver=None,
                 progress=None, resume_path=None, resume_step=0,
                 out_base_dir=None, run_tag=None, run_name=None,
                 cache_dir=None, storage_backend='sparse_cuda',
                 render_volume_scale=16, dist_context=None):
        # config is the explicit FitConfig (or any dict-style mapping of
        # fully resolved values) this fit reads; there is no module-global
        # fallback. The optimisation z window is Config's z_begin/z_end.
        #
        # scroll is the frozen ScrollSpec: physical facts of the scanned
        # scroll (name, voxel size, outward sense, umbilicus coordinate
        # scale, Lasagna/SDT groups and scale). paths is the resolved
        # SpiralInputPaths for this fit: the interactive runtime passes the
        # service's per-session selection; the CLI resolves the conventional
        # dataset layout (fit_session.conventional_input_paths).
        #
        # cache_dir / storage_backend / render_volume_scale are deployment
        # and presentation values, deliberately not part of the scroll file:
        # the CLI parses their FIT_SPIRAL_* environment defaults at its own
        # boundary, and the interactive runtime passes the service's values.
        #
        # The fit controls that used to arrive through FIT_SPIRAL_*
        # environment variables are constructor arguments now:
        #   resume_path / resume_step - checkpoint to restore and its legacy
        #     explicit step (the checkpoint's own completed_iterations wins);
        #   out_base_dir - parent directory for resolve_output_path();
        #   run_tag - optional suffix stamped into the output-directory name
        #     and the final overlay outputs;
        #   run_name - optional experiment-tracking run name appended to the
        #     output-directory name (the CLI passes wandb.run.name).
        #
        # interactive_driver marks a resident (interactive) session. The
        # setup/step paths read it for retention decisions (trusted-tree and
        # crossing-cache retention, store registration, interactive DT gating)
        # and checkpoint payloads read its requested_config/input_manifest.
        # The runtime drives the context itself; it never hands control back
        # through this reference. PR 3 replaces it with explicit session flags.
        self.config = config
        self.scroll = scroll
        self.paths = paths
        self.interactive_driver = interactive_driver
        self.progress = progress
        self.resume_path = resume_path or None
        self.resume_step = int(resume_step or 0)
        self.out_base_dir = out_base_dir if out_base_dir is not None else './out'
        self.run_tag = run_tag or None
        self.run_name = run_name
        # Who this process is in the job, as an explicit value. Callers that
        # joined a process group pass the context maybe_init_distributed()
        # returned; the default is the process context installed there (a
        # single-rank context when nothing joined one). Nothing below reads
        # RANK/WORLD_SIZE from the environment.
        self.dist = dist_context or process_context()

        # Scroll physical facts.
        self.scroll_name = scroll.name
        self.voxel_size_um = float(scroll.voxel_size_um)
        self.spiral_outward_sense = scroll.spiral_outward_sense
        self.normal_zarr_group = scroll.normal_zarr_group
        # The surf-SDT store's scale/encoding are read from the store's own
        # metadata, never from normal_zarr_group/lasagna_scale.
        self.surf_sdt_zarr_group = scroll.surf_sdt_zarr_group
        self.lasagna_scale = int(scroll.lasagna_scale)
        umbilicus_path = paths.umbilicus
        umbilicus_scale = float(scroll.umbilicus_coordinate_scale)
        self.umbilicus_z_to_yx = lambda: json_umbilicus_z_to_yx(
            umbilicus_path, coordinate_scale=umbilicus_scale)

        # Resolved input paths ('' means absent).
        self.scroll_zarr_path = paths.scroll_zarr or None
        self.normal_nx_zarr_path = paths.normal_x or None
        self.normal_ny_zarr_path = paths.normal_y or None
        self.grad_mag_zarr_path = paths.gradient_magnitude or None
        self.surf_sdt_zarr_path = paths.surf_sdt or None
        self.winding_inference_path = paths.winding_inference or None
        self.fibers_path = paths.fibers or None
        self.verified_patches_path = paths.verified_patches or None
        self.unverified_patches_path = paths.unverified_patches or None
        self.shell_path = paths.outer_shell or None
        self.tracks_dbm_path = paths.tracks_dbm or None
        self.pcl_input_specs = [
            (spec.path, spec.role.value if spec.role is not None else None)
            for spec in paths.pcls
        ]

        # Deployment/presentation values.
        self.cache_path = cache_dir if cache_dir is not None else (paths.cache_directory or None)
        self.lasagna_storage_backend = storage_backend
        self.render_volume_scale = int(render_volume_scale)

        self._lasagna_store = None
        self._scalar_stores = []
        # configure_losses() used to warm the geodesic strip-path worker
        # pool as an import-time side effect of installing the config.
        if config['patch_strip_sampling'] == 'dijkstra':
            strip_path_pools.warm_workers()

    # The optimisation z window lives in the fit configuration (its catalog
    # metadata records the full effect list); these properties are the one
    # reading point for the many z-window consumers below.
    @property
    def z_begin(self):
        return int(self.config['z_begin'])

    @property
    def z_end(self):
        return int(self.config['z_end'])

    def shell_losses_enabled(self):
        # The outer-shell enabling predicate is fit-input catalog data,
        # shared with request validation and run admission.
        return fit_input('outer_shell').required(self.config)

    def _load_patches_from_dir(self, path, label='patches'):
        progress = progress_or_null(self.progress)
        patches = {}
        entries = sorted(os.listdir(path))
        progress.begin(
            'loading', f'Loading {label}',
            step=0, total_steps=len(entries), unit='patches')
        for entry_number, entry in enumerate(entries, start=1):
            segment_path = os.path.join(path, entry)
            try:
                patches[entry] = load_tifxyz(segment_path)
            except Exception as e:
                print(f'Failed to load segment {entry}: {e}')
            progress.update(
                entry_number, detail=f'{len(patches):,} loaded')
        return patches

    def _prepare_patch_sampling_cache(self, patches):
        progress = progress_or_null(self.progress)
        progress.begin(
            'loading', 'Preparing patch sampling',
            step=0, total_steps=len(patches), unit='patches')
        native_sampling_available = load_native_spiral_sampling() is not None
        patch_areas = np.empty(len(patches), dtype=np.float32)
        for patch_idx, patch in enumerate(patches):
            # Use the quad-valid mask so bilinear interpolation at (row_idx+di, j+dj)
            # is well-defined for di, dj in [0, 1).
            valid_quad_mask_np = patch.valid_quad_mask.cpu().numpy()
            # Restrict sampling to quads whose representative z is in [z_begin, z_end),
            # so patch-loss tracks don't waste samples outside the optimisation ROI.
            zyxs_z_np = patch.zyxs[..., 0].cpu().numpy()
            quad_zs_np = (zyxs_z_np[:-1, :-1] + zyxs_z_np[1:, :-1] + zyxs_z_np[:-1, 1:] + zyxs_z_np[1:, 1:]) / 4
            z_in_roi_np = (
                    (quad_zs_np >= self.z_begin - self.config['patch_loss_z_margin'])
                    & (quad_zs_np < self.z_end + self.config['patch_loss_z_margin'])
            )
            in_roi_quad_mask_np = valid_quad_mask_np & z_in_roi_np
            if not in_roi_quad_mask_np.any():
                # Fallback if no quad falls in the ROI; should be rare since patches
                # entirely outside the z-ROI are dropped earlier.
                in_roi_quad_mask_np = valid_quad_mask_np
            patch._sampling_valid_quad_mask_np = in_roi_quad_mask_np
            if not native_sampling_available:
                patch._sampling_valid_quad_rows = np.flatnonzero(in_roi_quad_mask_np.any(axis=1))
                patch._sampling_valid_quad_cols = np.flatnonzero(in_roi_quad_mask_np.any(axis=0))

                # Python fallback: precompute, per row and per column, the
                # contiguous valid-quad runs. The native atlas owns an equivalent
                # packed representation and avoids these many small Python arrays.
                def _runs_per_line(mask_np, fixed_axis, valid_lines):
                    # Returns parallel lists indexed by valid line.

                    def _build_line_runs(line_valid):
                        padded = np.concatenate([[False], line_valid, [False]]).astype(np.int8)
                        diff = np.diff(padded)
                        los = np.where(diff == 1)[0].astype(np.int64)
                        his = np.where(diff == -1)[0].astype(np.int64)
                        return los, his

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
            progress.update(patch_idx + 1)

        inv_weights = patch_areas ** 0.5
        return inv_weights / inv_weights.sum()

    def _rebuild_pcl_sampling_strata(self):
        """Rebuild the per-family sampling strata in place.

        Mutates self.pcl_sampling_strata (never rebinds it) so every
        holder of the dict observes the rebuilt strata; called again by
        the interactive path whenever it appends pcls.
        """
        # Weight merged fiber-link components by their member-pcl count: each
        # draw samples at most sample_count_relative_winding_patch_pairs_per_pcl
        # patch pairs regardless of pcl size, so without this a component would
        # get ~1/N the pair-sampling pressure its N members had before merging.
        self.pcl_sampling_strata['cross_patch'] = build_pcl_sampling_strata(
            [pcl['sampling_group'] if len(pcl['points']) > 1 else None
             for pcl in self.cross_patch_pcls],
            self.config,
            member_weights=[len(pcl.get('link_member_cids', ())) or 1
                            for pcl in self.cross_patch_pcls],
        )
        self._rebuild_unattached_components()
        # Weight each component by its member count so a linked component keeps
        # the per-strip sampling pressure its members had before merging (each
        # sampled walk covers only one random path through the component).
        self.pcl_sampling_strata['unattached'] = build_pcl_sampling_strata(
            self.unattached_component_groups, self.config,
            member_weights=[len(members) for members in self.unattached_components])

    def _rebuild_unattached_components(self):
        """Rebuild the unattached link-component view in place.

        Each entry of self.unattached_components is the list of member strip
        indices connected by same-winding links (singletons for unlinked
        strips). The 'unattached' sampling strata index these components; each
        step the loss samples a chain *walk* through a chosen component -- along
        a strip, optionally hopping to the linked strip at each junction -- and
        applies the ordinary constant-winding strip target along the walk (the
        junction hop is a regular |dtheta| < pi step, so the sequential theta=0
        unwrap handles the seam). unattached_component_edges[c] lists component
        c's junctions as (strip_a, pos_a, strip_b, pos_b) with pos_* row indices
        into the strips' (decimated) point arrays.

        All three lists are mutated in place (never rebound) so the training-loop
        closures keep the same list objects across interactive rebuilds.
        """
        self.unattached_components.clear()
        self.unattached_component_groups.clear()
        self.unattached_component_edges.clear()
        strip_by_coll = {strip['id']: idx
                         for idx, strip in enumerate(self.unattached_pcl_strips)}
        # Strip-level view of the shared link_components decomposition. A link's
        # junction survives here only when both endpoints survived the z-roi trim
        # (they are exempt from decimation; see the strip loop in
        # load_host_inputs); a walk simply never crosses a dropped junction, so a
        # component with trimmed links degrades to islands sampled independently
        # within one entry.
        in_linked_component = set()
        for member_cids, member_links in self.link_components:
            comp_members = [strip_by_coll[cid] for cid in member_cids if cid in strip_by_coll]
            if not comp_members:
                continue
            in_linked_component.update(comp_members)
            edges = []
            for link in member_links:
                a = strip_by_coll.get(link['a_coll'])
                b = strip_by_coll.get(link['b_coll'])
                if a is None or b is None or a == b:
                    continue
                pos_a = self.unattached_pcl_strips[a].get('link_points', {}).get(link['a_point'])
                pos_b = self.unattached_pcl_strips[b].get('link_points', {}).get(link['b_point'])
                if pos_a is None or pos_b is None:
                    continue
                edges.append((a, pos_a, b, pos_b))
            self.unattached_components.append(comp_members)
            self.unattached_component_groups.append(
                self.unattached_strip_sampling_groups[comp_members[0]])
            self.unattached_component_edges.append(edges)
        for idx in range(len(self.unattached_pcl_strips)):
            if idx not in in_linked_component:
                self.unattached_components.append([idx])
                self.unattached_component_groups.append(
                    self.unattached_strip_sampling_groups[idx])
                self.unattached_component_edges.append([])

    def load_host_inputs(self):
        """Load and prepare every host-side input for a fit.

        Seeds the host RNG streams, then loads patches, point collections,
        fibers, tracks, and the outer shell; links and classifies PCLs;
        builds the sampling caches, the host-resident patch atlases, the
        trusted-geometry index, the interactive influence anchor stash and
        the whole-object DT target samples. Requires no device state: the
        patch atlases are host-resident (their `device` only selects where
        each lookup's interpolated points are delivered), and the CUDA
        stores, model, and optimiser are built later by
        build_device_state().

        Everything the device stages read from here they only read: nothing
        below the cut consumes or releases a host structure, so the model
        stage can be re-run against inputs this method loaded once.

        The loaded inputs stay readable as attributes afterwards, so
        analysis tools can construct a context, call this, and read:

        - patches: verified_patches (+ verified_patches_list,
          num_verified_patches), unverified_patches
          (+ unverified_patches_list), shell_patch
        - point collections: cross_patch_pcls, unattached_pcl_strips,
          unattached_strip_sampling_groups, pcl_sampling_strata, next_id,
          link_distance_tolerance, resolved_links, link_components,
          unattached_components / _component_groups / _component_edges
        - sampling: patch_sampling_probabilities, patch_atlas,
          unverified_patch_sampling_probabilities, unverified_patch_atlas
        - trusted geometry: trusted_geometry_tree,
          influence_anchor_geometry
        - tracks: tracks, track_families, track_source_ids,
          track_crossing_cache, track_graph, track_reload_source /
          _families / _source_ids, track_sampling_config, using_tracks,
          filter_tracks_by_shell
        - misc: umbilicus, scroll_zarr, shell_envelope,
          dense_spacing_mode, phase_mode, grad_mag_spacing_enabled
        """
        progress = progress_or_null(self.progress)

        np.random.seed(self.config['optimizer_random_seed'])
        torch.random.manual_seed(self.config['optimizer_random_seed'])
        progress.begin('loading', 'Loading umbilicus')
        umbilicus = self.umbilicus_z_to_yx()
        if self.scroll_zarr_path:
            progress.begin('loading', 'Opening scroll volume')
            print('loading volume zarr')
            scroll_zarr = zarr.open(self.scroll_zarr_path, mode='r')
        else:
            scroll_zarr = None

        # ==========================================================================
        # Patch loading and ROI filtering
        # ==========================================================================

        filter_tracks_by_shell = bool(self.tracks_dbm_path) and bool(self.shell_path)
        shell_patch = None
        if self.shell_losses_enabled() or filter_tracks_by_shell:
            if not self.shell_path:
                raise RuntimeError('shell losses are enabled, but no outer shell path is set')
            progress.begin('loading', 'Loading outer shell')
            shell_patch = load_tifxyz(self.shell_path)

        use_verified_patches = bool(self.verified_patches_path) and not self.config['input_disable_patches']
        use_unverified_patches = bool(self.unverified_patches_path) and not self.config['input_disable_patches']
        if not use_verified_patches and not use_unverified_patches:
            verified_patches = {}
            unverified_patches = {}
            print('skipping all verified/unverified patch loading')
        else:
            # An empty verified dir is allowed when unverified patches are supplied
            # (unverified-only ablations); both empty is a configuration error.
            verified_patches = (
                self._load_patches_from_dir(self.verified_patches_path, 'verified patches')
                if use_verified_patches and self.verified_patches_path else {}
            )
            unverified_patches = {}
            if use_unverified_patches and self.unverified_patches_path:
                unverified_patches = self._load_patches_from_dir(
                    self.unverified_patches_path, 'unverified patches')

        if (not verified_patches and not unverified_patches
                and (use_verified_patches or use_unverified_patches)):
            raise RuntimeError('No patches could be loaded')

        print(f" loaded {len(verified_patches)} patches")
        print(f" loaded {len(unverified_patches)} unverified patches")

        patch_filter_total = len(verified_patches) + len(unverified_patches)
        progress.begin(
            'loading', 'Filtering patches to fit region',
            step=0, total_steps=patch_filter_total, unit='patches')
        filtered_count = 0
        for patches in (verified_patches, unverified_patches):
            for patch_id, patch in list(patches.items()):
                try:
                    # we erode cells this distance from any invalid cell to catch annotation errors
                    # which are hard to detect at the edges of patches
                    cells_to_erode = patch.erosion_cells(self.config['patch_erode_patches'])
                    if cells_to_erode > 0:
                        if not erode_patch_valid_region(patch, cells_to_erode):
                            del patches[patch_id]
                            continue

                    # remove any patches which do not intersect with the roi we are fitting
                    if not patch_intersects_z_roi(patch, self.z_begin, self.z_end):
                        del patches[patch_id]
                        continue
                    # ROI testing may materialise the compact valid-coordinate view.
                    # Training retains the base grid and masks, so regenerate this view
                    # lazily only for a later exporter that actually requests it.
                    patch.release_derived_caches()
                finally:
                    filtered_count += 1
                    progress.update(filtered_count)

        # ==========================================================================
        # Point collection loading
        # ==========================================================================

        # Load all pcls in full-resolution voxel space, link every point to patches,
        # and split into cross-patch / unattached sets. Verified patches must already
        # be filtered to the z-roi.
        point_collections = {}
        next_id = 0
        input_specs = self.pcl_input_specs
        progress.begin(
            'loading', 'Loading point collections',
            step=0, total_steps=len(input_specs), unit='inputs')
        for spec_number, (pattern, explicit_role) in enumerate(input_specs, start=1):
            expanded = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
            for path in expanded:
                loaded = load_point_collection(path) or {}
                for pcl in loaded.values():
                    pcl['source_file'] = path
                    pcl['sampling_group'] = path
                    # Absolute-winding status is determined solely by the source file:
                    # only pcls loaded from abs_winding.json carry absolute winding
                    # numbers. Any metadata key in another file is ignored.
                    pcl.setdefault('metadata', {})['winding_is_absolute'] = (
                        explicit_role == 'absolute'
                        if explicit_role is not None
                        else os.path.basename(path) == 'abs_winding.json'
                    )
                    pcl['metadata']['input_role'] = explicit_role or (
                        'absolute' if os.path.basename(path) == 'abs_winding.json' else 'legacy'
                    )
                    point_collections[next_id] = pcl
                    next_id += 1
            progress.update(
                spec_number, detail=f'{len(point_collections):,} collections loaded')

        progress.begin('loading', 'Loading fiber point collections')
        fiber_point_collections, next_id = load_fiber_point_collections(
            self.fibers_path,
            next_id,
            min_point_spacing=self.config['pcl_fiber_min_point_spacing'],
        )
        # All fibers (horizontal, vertical, and merged link components) form one
        # sampling group, rather than one group per source file like the regular pcls.
        for pcl in fiber_point_collections.values():
            pcl['sampling_group'] = 'fibers'
        point_collections.update(fiber_point_collections)

        for pcl in point_collections.values():
            for point in pcl['points'].values():
                point['zyx'] = np.array([point['p'][2], point['p'][1], point['p'][0]], dtype=np.float32)

        def pcl_intersects_z_roi(pcl):
            for point in pcl['points'].values():
                z = point['zyx'][0]
                if self.z_begin <= z < self.z_end:
                    return True
            return False

        link_distance_tolerance = 2.5

        # ==========================================================================
        # Point-to-patch linking
        # ==========================================================================

        # Link every point of every pcl to patches (adds 'on_patch' to attached points).
        # Using the vc3d surface patch index, identify which pcl points lie on patch surfaces.
        # A point is considered on a patch surface if it is within link_distance_tolerance.
        # For general pcls, when multiple patches are within tolerance, prefer the largest
        # patch area and use distance only as a tie-break. Between-patches pcls connect
        # overlapping patches and attach only to their named patch pair, using nearest
        # distance within that pair.
        progress.begin(
            'loading', 'Linking points to patches',
            detail=(
                f'{len(point_collections):,} collections, '
                f'{len(verified_patches):,} patches'))
        link_points_to_patches(
            verified_patches,
            point_collections,
            tolerance=link_distance_tolerance,
            surface_index_tolerance=link_distance_tolerance,
            distance_scale=1.0,
            general_hit_policy='largest_area',
        )

        # Every pcl carries the uniform chain interface (pcl['chain'], see
        # spiral_helpers.Chain): a SequenceChain over the id-sorted point order
        # (lazy, so later point-dict replacements are picked up). Consumers go
        # through this and never assume id-sorted order is chain-valid.
        # merge_linked_point_collections below replaces link-connected members with
        # a merged component pcl carrying a graph-routing ComponentChain instead.
        for pcl in point_collections.values():
            pcl['chain'] = SequenceChain(pcl)

        # ==========================================================================
        # Cross-fiber links ("branches")
        # ==========================================================================
        # Resolve stored branch metadata into concrete point-to-point links. Handled
        # symmetrically for fibers and pcls (pcls simply carry no branches today). Two
        # effects downstream: link-connected collections merge into one cross-patch
        # "component pcl" carrying an explicit fiber graph (see
        # merge_linked_point_collections), so the rel-winding loss ties patches on
        # different fibers with delta 0 through chains that hop fibers at each junction
        # (an ordinary |dtheta| < pi chain step, theta=0-unwrapped like any other); and
        # the unattached-strip loss samples chain walks through the same junctions
        # (branching randomly at each one), pulling every linked fiber onto one shared
        # winding via the ordinary constant-radius-along-strip target.
        resolved_links = []
        if self.config['pcl_use_fiber_links']:
            resolved_links = resolve_fiber_links(
                point_collections,
                include_pending=self.config['pcl_use_pending_fiber_links'],
            )
            print(f'fiber links: {len(resolved_links)} resolved')
        # One shared component decomposition over the link graph, consumed twice:
        # the cross-patch merge below unions each component's attached points, and
        # _rebuild_unattached_components maps the same components onto unattached
        # strips for the walk-sampling loss.
        link_components = build_link_components(resolved_links)
        linked_cids = {cid for member_cids, _ in link_components for cid in member_cids}

        # ==========================================================================
        # Point collection classification
        # ==========================================================================

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
            # Link-component members join the unattached pool even when fully
            # attached: their strip is the geometry the component's chain walks (and
            # theta=0 unwrap) run through between the fibers they link.
            if num_unattached >= 1 or pid in linked_cids:
                unattached_point_collections[pid] = copy.deepcopy(pcl) if num_attached >= 2 else pcl

        # Merge each set of collections joined by cross-fiber links into one
        # cross-patch "component pcl" with an explicit graph over its members. The
        # merged pcl exposes the union of the members' attached points (so pairs of
        # patches on different fibers feed the rel-winding loss with delta 0) plus a
        # chain function that routes between any two points along the fibers, hopping
        # to the linked fiber at each junction. A junction hop is just another chain
        # step between nearly-coincident points (|dtheta| < pi), so the ordinary
        # theta=0 unwrap along the chain handles the seam. Members are removed
        # from the cross-patch pool (the merged pcl subsumes their within-fiber
        # pairs); their unattached role is untouched (handled by the walk-sampling in
        # the unattached-strip loss).
        if link_components:
            cross_patch_point_collections, num_merged = merge_linked_point_collections(
                point_collections, link_components, cross_patch_point_collections)
            if num_merged:
                print(f'fiber links: merged into {num_merged} cross-patch components')

        # For unattached pcls, keep only the longest contiguous subrange (in id-sorted
        # order) of points whose zs lie within [z_begin - margin, z_end + margin); drop
        # the pcl entirely if fewer than 2 points remain.
        z_margin = self.config['patch_loss_z_margin']
        dropped_unattached_pcl_count = 0
        for pid in list(unattached_point_collections.keys()):
            pcl = unattached_point_collections[pid]
            sorted_items = sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
            best_start, best_end = 0, 0
            run_start = 0
            for i, (_, point) in enumerate(sorted_items):
                z = point['zyx'][0]
                if self.z_begin - z_margin <= z < self.z_end + z_margin:
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
                if pid not in verified_patches:
                    continue
                points_by_patch.setdefault(pid, []).append(point)
            pcl['points_by_patch'] = points_by_patch
        unattached_pcl_strips = _UnattachedPclStripList()
        unattached_strip_sampling_groups = []  # parallel to unattached_pcl_strips
        min_point_spacing = self.config['pcl_unattached_pcl_min_point_spacing']
        # For each unattached pcl, materialise an id-sorted strip of point zyxs and the
        # corresponding winding annotations. Strips with <2 points are dropped.
        # If min_point_spacing > 0, decimate each strip greedily along its id-sorted order
        # so consecutive kept points are at least min_point_spacing apart in 3D scroll space.
        # The first and last points are always kept, as are cross-fiber link endpoints
        # (their strip positions are recorded in strip['link_points'], keyed by the
        # pcl-local point id, so _rebuild_unattached_components can place junctions).
        link_point_ids_by_coll = {}
        for link in resolved_links:
            link_point_ids_by_coll.setdefault(link['a_coll'], set()).add(link['a_point'])
            link_point_ids_by_coll.setdefault(link['b_coll'], set()).add(link['b_point'])
        for pcl_id, pcl in unattached_point_collections.items():
            sorted_items = sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
            if len(sorted_items) < 2:
                continue
            link_ids = link_point_ids_by_coll.get(pcl_id, ())
            force_keep = {pos for pos, (point_id, _) in enumerate(sorted_items)
                          if int(point_id) in link_ids}

            zyxs = np.stack([point['zyx'] for _, point in sorted_items], axis=0).astype(np.float32)
            windings = np.array([point['winding_annotation'] for _, point in sorted_items], dtype=np.float32)

            zyxs, keep = _decimate_ordered_points_min_spacing(
                zyxs, min_point_spacing, return_indices=True,
                force_keep=force_keep | {len(zyxs) - 1})
            windings = windings[keep]

            link_points = {
                int(sorted_items[orig_pos][0]): strip_pos
                for strip_pos, orig_pos in enumerate(keep)
                if orig_pos in force_keep
            }
            unattached_pcl_strips.append({
                'id': pcl_id,
                'name': pcl.get('name'),
                'source_file': pcl.get('source_file'),
                'zyxs': zyxs,
                'windings': windings,
                'link_points': link_points,
            })
            unattached_strip_sampling_groups.append(pcl['sampling_group'])

        cross_patch_pcls = list(cross_patch_point_collections.values())
        print(
            f'pcls: {len(cross_patch_pcls)} cross-patch, '
            f'{len(unattached_pcl_strips)} unattached'
        )
        if self.config['pcl_stratified_pcl_sampling'] or self.config['pcl_sampling_weights'] is not None:
            def _group_counts(groups):
                counts = {}
                for group in groups:
                    counts[group] = counts.get(group, 0) + 1
                entries = []
                for group, count in sorted(counts.items(), key=lambda kv: str(kv[0])):
                    key = os.path.splitext(os.path.basename(str(group)))[0]
                    if self.config['pcl_sampling_weights'] is None:
                        entries.append(f'{key}: {count}')
                    else:
                        entries.append(
                            f'{key} (w={self.config["pcl_sampling_weights"][key]}): {count}')
                return ', '.join(entries)
            print(f'  cross-patch sampling groups: {_group_counts(pcl["sampling_group"] for pcl in cross_patch_pcls)}')
            print(f'  unattached sampling groups: {_group_counts(unattached_strip_sampling_groups)}')

        # Per-step sampling pools for the rel-winding and unattached-strip losses:
        # pool indices grouped into strata by sampling group (see
        # build_pcl_sampling_strata; stratification is controlled by the legacy
        # boolean or the weighted config). Single-point pcls (possible only for
        # winding_is_absolute pcls) can't form a cross-patch pair, so they are
        # excluded from the rel-winding pool. Rebuilt whenever the interactive
        # path appends pcls (see _rebuild_pcl_sampling_strata).
        self.cross_patch_pcls = cross_patch_pcls
        self.unattached_pcl_strips = unattached_pcl_strips
        self.unattached_strip_sampling_groups = unattached_strip_sampling_groups
        self.resolved_links = resolved_links
        self.link_components = link_components
        # The unattached loss consumes strips through these link components; see
        # _rebuild_unattached_components, which fills them from link_components.
        self.unattached_components = []
        self.unattached_component_groups = []
        self.unattached_component_edges = []
        self.pcl_sampling_strata = {}
        self._rebuild_pcl_sampling_strata()

        # The strip arrays and cross-patch list are the compact training forms.
        # Drop the JSON-shaped source containers, especially the independent deep
        # copies made for PCLs that participate in both loss families.
        del point_collections, fiber_point_collections
        del unattached_point_collections, cross_patch_point_collections

        # ==========================================================================
        # dense-spacing mode, shell envelope, and tracks
        # ==========================================================================

        # Dense-spacing input contract. Checked before any asset paths so
        # an invalid mode fails as itself, not as a missing-file error.
        dense_spacing_mode = self.config['dense_spacing_mode']
        if dense_spacing_mode not in ('phase', 'grad_mag', 'winding_model'):
            raise ValueError(
                f'dense_spacing_mode={dense_spacing_mode!r} must be '
                "'phase', 'grad_mag', or 'winding_model'")
        phase_mode = dense_spacing_mode == 'phase'
        winding_model_mode = dense_spacing_mode == 'winding_model'
        grad_mag_spacing_enabled = (
            dense_spacing_mode == 'grad_mag'
            and self.config['loss_weight_dense_spacing'] > 0
        )
        shell_envelope = None
        if shell_patch is not None and filter_tracks_by_shell:
            progress.begin('loading', 'Building outer-shell lookup')
            shell_envelope = ShellPolarMap(
                shell_patch,
                umbilicus,
                z_min=self.z_begin - self.config['model_flow_bounds_z_margin'],
                z_max=self.z_end + self.config['model_flow_bounds_z_margin'],
                num_theta_bins=self.config['shell_num_theta_bins'],
                device='cpu',
                config=self.config,
            )

        track_sampling_config = validate_track_sampling_config(self.config)
        track_families = None
        track_source_ids = None
        track_crossing_cache = None
        track_graph = None
        track_reload_source = None
        track_reload_families = None
        track_reload_source_ids = None
        if self.tracks_dbm_path is not None:
            progress.begin(
                'loading', 'Resolving track store',
                detail=os.path.basename(self.tracks_dbm_path))
            print(f'loading tracks from {self.tracks_dbm_path}')
            if (track_sampling_config['crossing_precompute_max'] > 0
                    or track_sampling_config['crossing_mode'] == 'track_walk'):
                track_crossing_cache = load_track_crossing_cache(self.tracks_dbm_path)
                if track_crossing_cache is not None:
                    track_graph = TrackGraph(track_crossing_cache)
                    print(
                        f'built TrackGraph: {len(track_graph)} tracks, '
                        f'{track_graph.edge_count} crossings in '
                        f'{track_graph.build_seconds:.1f}s')
                    track_crossing_cache = None
                tracks, track_families, track_source_ids = load_tracks_from_dbm(
                    self.tracks_dbm_path, self.z_begin, self.z_end, return_families=True,
                    return_source_ids=True, progress=progress)
            else:
                tracks = load_tracks_from_dbm(
                    self.tracks_dbm_path, self.z_begin, self.z_end, progress=progress)
            track_reload_source = tracks
            track_reload_families = track_families
            track_reload_source_ids = track_source_ids
            if filter_tracks_by_shell:
                progress.begin(
                    'loading', 'Filtering tracks to outer shell',
                    detail=f'{len(tracks):,} tracks')
                tracks, track_families, kept_track_indices = filter_tracks_to_outer_shell(
                    tracks, shell_envelope, track_families, return_indices=True)
                if track_source_ids is not None:
                    track_source_ids = track_source_ids[kept_track_indices]
            print(f'loaded {len(tracks)} tracks within z-roi [{self.z_begin}, {self.z_end})')
        else:
            tracks = None

        # ==========================================================================
        # patch cache / atlas construction
        # ==========================================================================

        verified_patches_list = list(verified_patches.values())
        patch_sampling_probabilities = self._prepare_patch_sampling_cache(verified_patches_list)
        num_verified_patches = len(verified_patches_list)
        print(f'fitting {num_verified_patches} patches')

        progress.begin(
            'loading', 'Building verified-patch GPU atlas',
            detail=f'{len(verified_patches):,} patches')
        patch_atlas = PatchAtlas(verified_patches, device='cuda')
        print(f'patch atlas (host-resident): {patch_atlas.memory_mb():.1f} MB')

        # ==========================================================================================
        # trusted geometry (verified patches and pcls) kdtree / unverified patches + tracks masking
        # ==========================================================================================

        # The trusted point cloud is consumed only by a CPU cKDTree. Build it directly
        # on CPU instead of storing it in the atlas on CUDA, concatenating it again on
        # CUDA, and immediately copying it back here.
        verified_patches_and_pcls_cpu = []
        for patch in verified_patches_list:
            z_flat = patch.zyxs.reshape(-1, 3).to(dtype=torch.float32)
            valid_flat = patch.valid_vertex_mask.reshape(-1)
            z_in_roi = (z_flat[:, 0] >= self.z_begin) & (z_flat[:, 0] < self.z_end)
            if (valid_flat & z_in_roi).any():
                verified_patches_and_pcls_cpu.append(z_flat[valid_flat & z_in_roi])
        for strip in unattached_pcl_strips:
            zyxs = torch.from_numpy(strip['zyxs']).to(dtype=torch.float32)
            in_roi = (zyxs[..., 0] >= self.z_begin) & (zyxs[..., 0] < self.z_end)
            if in_roi.any():
                verified_patches_and_pcls_cpu.append(zyxs[in_roi])
        verified_patches_and_pcls_cpu = (
            torch.cat(verified_patches_and_pcls_cpu, dim=0)
            if verified_patches_and_pcls_cpu
            else torch.empty([0, 3], dtype=torch.float32)
        )

        unverified_patches_list = []
        unverified_patch_sampling_probabilities = None
        unverified_patch_atlas = None
        using_tracks = (
            (self.config['loss_weight_track_radius'] > 0 or self.config['loss_weight_track_dt'] > 0)
            and bool(tracks)
        )
        trusted_geometry_tree = None
        verified_patches_and_pcls_np = None

        # Untrusted 'unverified' patches: mask away wherever they fall near trusted geometry (verified
        # patch vertices + pcl strips, same anchor cloud used for snap-anchors / track-exclusion), then
        # build their own sampling cache + GPU atlas. They feed only their own radius/DT losses.
        if unverified_patches or using_tracks:
            # Build a cKDTree over the scroll-space anchor points (CPU) for fixed-radius
            # nearest-neighbour queries.
            verified_patches_and_pcls_np = verified_patches_and_pcls_cpu.numpy()
            verified_patches_and_pcls_np = np.ascontiguousarray(verified_patches_and_pcls_np, dtype=np.float32)
            if verified_patches_and_pcls_np.shape[0] > 0:
                progress.begin(
                    'loading', 'Building trusted-geometry index',
                    detail=f'{len(verified_patches_and_pcls_np):,} points')
                trusted_geometry_tree = cKDTree(verified_patches_and_pcls_np)

        if unverified_patches:
            # For each unverified patch, invalidate (set zyxs -> -1) every currently-valid vertex
            # lying within the exclusion radius of trusted geometry, then re-derive the patch's
            # masks/area. Patches left with no valid quad are dropped. This is the patch analogue
            # of the DBM-track exclusion in tracks.py: untrusted patches only constrain regions
            # the trusted inputs don't already cover, so they can't fight verified geometry.
            exclusion_radius = float(self.config['patch_unverified_patch_exclusion_radius'])
            progress.begin(
                'loading', 'Masking unverified patches',
                detail=f'{len(unverified_patches):,} patches')
            unverified_patches, n_masked_vertices, n_dropped_patches = (
                _mask_unverified_patches_near_trusted_geometry(
                    unverified_patches,
                    trusted_geometry_tree,
                    exclusion_radius,
                )
            )
            print(
                f'unverified patches: masked {n_masked_vertices} vertices near trusted geometry '
                f'(radius {exclusion_radius:.1f}), dropped {n_dropped_patches} fully-masked patches; '
                f'{len(unverified_patches)} remain'
            )

        if unverified_patches:
            unverified_patches_list = list(unverified_patches.values())
            unverified_patch_sampling_probabilities = self._prepare_patch_sampling_cache(unverified_patches_list)
            unverified_patch_atlas = PatchAtlas(unverified_patches, device='cuda')

        # Loaded host inputs, kept as inspectable attributes (ownership
        # class (b): host-prepared inputs and caches). cross_patch_pcls,
        # unattached_pcl_strips, unattached_strip_sampling_groups, and
        # pcl_sampling_strata were assigned above, before the strata build.
        self.umbilicus = umbilicus
        self.scroll_zarr = scroll_zarr
        self.filter_tracks_by_shell = filter_tracks_by_shell
        self.shell_patch = shell_patch
        self.shell_envelope = shell_envelope
        self.verified_patches = verified_patches
        self.unverified_patches = unverified_patches
        self.next_id = next_id
        self.link_distance_tolerance = link_distance_tolerance
        self.dense_spacing_mode = dense_spacing_mode
        self.phase_mode = phase_mode
        self.winding_model_mode = winding_model_mode
        self.grad_mag_spacing_enabled = grad_mag_spacing_enabled
        self.track_sampling_config = track_sampling_config
        self.tracks = tracks
        self.track_families = track_families
        self.track_source_ids = track_source_ids
        self.track_crossing_cache = track_crossing_cache
        self.track_graph = track_graph
        self.track_reload_source = track_reload_source
        self.track_reload_families = track_reload_families
        self.track_reload_source_ids = track_reload_source_ids
        self.verified_patches_list = verified_patches_list
        self.patch_sampling_probabilities = patch_sampling_probabilities
        self.num_verified_patches = num_verified_patches
        self.patch_atlas = patch_atlas
        self.using_tracks = using_tracks
        self.trusted_geometry_tree = trusted_geometry_tree
        self.unverified_patches_list = unverified_patches_list
        self.unverified_patch_sampling_probabilities = unverified_patch_sampling_probabilities
        self.unverified_patch_atlas = unverified_patch_atlas

        # A compact subsample of the trusted cloud seeds a future Run's
        # influence anchor bank. Keep it for every interactive session because
        # influence can be enabled or disabled independently on each Run
        # request. The generator is seeded explicitly, so the stash is
        # deterministic without perturbing the training RNG streams.
        self.influence_anchor_geometry = None
        if self.interactive_driver is not None:
            stash_generator = torch.Generator()
            stash_generator.manual_seed(int(self.config['optimizer_random_seed']))
            self.influence_anchor_geometry = subsample_rows(
                verified_patches_and_pcls_cpu,
                int(self.config['sample_count_influence_anchor_geometry_points']),
                stash_generator,
            ).clone()
        # The trusted cloud itself stays local: the cKDTree above and that
        # stash are all anything downstream reads it for, and consuming it
        # here rather than in build_device_state() is what leaves the device
        # stages with nothing of the host's to release.
        del verified_patches_and_pcls_cpu, verified_patches_and_pcls_np

        # ==========================================================================
        # Whole-object DT target caches (see dt_targets.py)
        # ==========================================================================

        # Deterministic sparse samples over each patch's own grid: host work on
        # host inputs, so it belongs here rather than beside the model that
        # eventually reads the caches these seed.
        if self.config['dt_target_mode'] not in ('strip_median', 'whole_object_quantile'):
            raise ValueError(f"dt_target_mode must be 'strip_median' or 'whole_object_quantile', got {self.config['dt_target_mode']!r}")
        self.dt_target_whole_object = self.config['dt_target_mode'] == 'whole_object_quantile'
        if self.dt_target_whole_object:
            progress.begin(
                'loading', 'Preparing distance-target samples',
                detail=(
                    f'{len(self.verified_patches_list) + len(self.unverified_patches_list):,} '
                    'patches'))
            prepare_patch_dt_target_samples(
                self.verified_patches_list, self.config['sample_count_patch_dt_target_points'], self.config['dt_target_max_stride'],
            )
            if self.unverified_patches_list:
                prepare_patch_dt_target_samples(
                    self.unverified_patches_list, self.config['sample_count_patch_dt_target_points'], self.config['dt_target_max_stride'],
                )

    def _phase_mode_active(self):
        return self.phase_mode and self.sdt_volume is not None and self.lasagna_volume is not None

    def _winding_model_mode_active(self):
        return self.winding_model_mode and self.winding_inference is not None

    def _warn_if_sdt_loss_inactive(self):
        # Run-mutable weights are read afresh every step, but the SDT-backed
        # components only exist in phase mode; make other sessions' nonzero
        # SDT-only weights a visible no-op. The native min-spacing
        # barrier is asset-independent and remains active in either mode.
        if self.phase_mode:
            return
        inactive = ['loss_weight_dense_spacing_count',
                    'loss_weight_dense_attachment']
        if not self.winding_model_mode:
            inactive.append('loss_weight_dense_spacing_density')
        for weight_key in inactive:
            if self.config[weight_key] > 0 and weight_key not in self._sdt_inactive_warned:
                self._sdt_inactive_warned.add(weight_key)
                print(f'WARNING: {weight_key} > 0 but dense_spacing_mode='
                      f'{self.dense_spacing_mode!r}; this component runs only as '
                      "part of the 'phase' bundle and is INACTIVE.")

    def _subsample_shell_radius_pool(self, patch):
        # The shell-patch radius loss draws sample_count_shell_samples random
        # shell points per step; keep a pool of exactly that size resident on
        # the GPU instead of the full shell cloud. A dedicated generator makes
        # the pool deterministic (identical across DDP ranks) without
        # perturbing the training RNG streams.
        pool_generator = torch.Generator()
        pool_generator.manual_seed(int(self.config['optimizer_random_seed']))
        return subsample_rows(
            patch.valid_zyxs, int(self.config['sample_count_shell_samples']), pool_generator,
        ).to(device=self.device, dtype=torch.float32)

    def _infer_outer_winding_idx_for_this_run(self):
        return _infer_shell_outer_winding_idx(
            self.spiral_and_transform.get_slice_to_spiral_transform(),
            self.spiral_and_transform.get_dr_per_winding(),
            self.verified_patches_list,
            self.unattached_pcl_strips,
            self.config,
            self.z_begin,
            self.z_end,
            get_or_build_unattached_pcl_flat,
        )

    def _warn_if_dense_losses_structurally_disabled(self):
        # Loss weights are run-mutable, so re-check each step and warn once.
        for weight_key in _structurally_disabled_dense_weight_keys(
                self.config, self.shell_outer_winding_idx):
            if weight_key not in self._dense_inactive_warned:
                self._dense_inactive_warned.add(weight_key)
                print(f'WARNING: {weight_key} > 0 but shell_outer_winding_idx '
                      'is unresolved (config key is None and no outer shell '
                      'inferred one); this loss samples the spiral out to that '
                      'winding and stays INACTIVE. Set shell_outer_winding_idx '
                      'to enable it (some of these losses also need the '
                      'phase/SDT assets, see any warnings above).')

    def _apply_high_res_lr_scale(self, iteration):
        scale = get_flow_field_high_res_lr_scale(self.config, iteration)
        low_res_group = next(
            group for group in self.optimiser.param_groups
            if any(param is self.low_res_flow_params[0] for param in group['params']))
        high_res_group = next(
            group for group in self.optimiser.param_groups
            if any(param is self.high_res_flow_params[0] for param in group['params']))
        set_optimizer_group_lr_scale(
            self.optimiser,
            self.lr_scheduler,
            group=high_res_group,
            reference_group=low_res_group,
            scale=scale,
            initial_lr=self.config['optimizer_learning_rate'],
        )
        return scale

    def _realign_lr_schedule(self, completed_steps):
        """Align optimizer/scheduler state to the current absolute horizon."""
        self.lr_scheduler, self.num_training_steps = realign_optimizer_lr_schedule(
            self.optimiser,
            self.lr_scheduler,
            initial_lr=self.config['optimizer_learning_rate'],
            final_factor=self.config['optimizer_lr_final_factor'],
            completed_steps=completed_steps,
            training_horizon=self.config['optimizer_num_training_steps'],
            exponential=self.config['optimizer_exp_lr_schedule'],
        )

    def build_device_state(self):
        """Allocate the session's device-resident state.

        Creates the CUDA-backed volume stores, the model, optimiser and LR
        scheduler, the prepared device track tables and the rest of the
        device-dependent setup, preserving the original inline order: the
        resume-checkpoint restore and the distributed reseed consume and
        overwrite the host RNG streams, so relative order is load-bearing.
        Requires load_host_inputs() to have run and self.out_path to be set.

        The two stages below are one ordinal, not a graph: everything the
        model stage needs from the store stage the store stage has already
        built, so a rebuild that only changes the model re-runs the second
        alone (see rebuild_model_state()).
        """
        self._build_store_state()
        self._build_model_state()

    def _build_store_state(self):
        """Materialise the Lasagna and surf-SDT brick pools.

        The expensive half of the device build, and the half nothing but the
        z window, the store paths and the dense-loss mode can invalidate.
        """
        interactive_driver = self.interactive_driver
        progress = progress_or_null(self.progress)

        # ==========================================================================
        # lasagna and SDT stores
        # ==========================================================================

        self.lasagna_volume = prepare_lasagna_volume(
            self.scroll_zarr,
            use_normals=(self.config['loss_weight_dense_normals'] > 0 or self.phase_mode),
            use_spacing=self.grad_mag_spacing_enabled,
            normal_nx_zarr_path=self.normal_nx_zarr_path,
            normal_ny_zarr_path=self.normal_ny_zarr_path,
            grad_mag_zarr_path=self.grad_mag_zarr_path,
            normal_zarr_group=self.normal_zarr_group,
            z_begin=self.z_begin,
            z_end=self.z_end,
            lasagna_scale=self.lasagna_scale,
            storage_backend=self.lasagna_storage_backend,
            cache_directory=self.cache_path,
            progress=progress,
        )
        if interactive_driver is not None and self.lasagna_volume:
            self._lasagna_store = self.lasagna_volume['store']

        # Surf-SDT store: a core input of the whole phase bundle (registration,
        # count, attachment), required in phase mode even when individual
        # sub-weights are zero so run-mutable weights can be adjusted (or zeroed
        # and re-raised) at run boundaries without a session reload.
        self.sdt_volume = None
        if self.phase_mode:
            if not self.surf_sdt_zarr_path or not os.path.exists(self.surf_sdt_zarr_path):
                raise RuntimeError(
                    "dense_spacing_mode='phase' requires the surf-SDT store: "
                    f'{self.surf_sdt_zarr_path!r}')
            if self.lasagna_volume is None:
                raise RuntimeError(
                    "dense_spacing_mode='phase' requires the dense normal stores "
                    'for band incidence/fragment handling')
            self.sdt_volume = prepare_surf_sdt_volume(
                self.surf_sdt_zarr_path,
                self.surf_sdt_zarr_group,
                z_begin=self.z_begin,
                z_end=self.z_end,
                cache_directory=self.cache_path,
                storage_backend=self.lasagna_storage_backend,
                progress=progress,
            )
            if interactive_driver is not None:
                self._scalar_stores.append(self.sdt_volume['store'])

        self.winding_inference = None
        if self.winding_model_mode:
            if (not self.winding_inference_path
                    or not os.path.isdir(self.winding_inference_path)):
                raise RuntimeError(
                    "dense_spacing_mode='winding_model' requires the compact "
                    f"winding-inference store: {self.winding_inference_path!r}")
            progress.begin(
                'loading', 'Loading winding-inference supervision',
                detail=os.path.basename(os.path.normpath(
                    self.winding_inference_path)))
            self.winding_inference = load_winding_inference_store(
                self.winding_inference_path,
                torch.device('cuda'),
                verify=os.environ.get(
                    'FIT_SPIRAL_VERIFY_WINDING_INFERENCE', '1') != '0',
                z_range=(self.z_begin, self.z_end),
            )
            print(
                'loaded winding inference: '
                f"{self.winding_inference.fingerprint['num_rays']:,} rays "
                f"({self.winding_inference.num_z_eligible_rays:,} intersect "
                f"z-range [{self.z_begin}, {self.z_end})), "
                f"{self.winding_inference.fingerprint['num_crossings']:,} "
                'crossings')

        self._sdt_inactive_warned = set()

    def _build_model_state(self):
        """Construct the model, the optimiser, and everything after them.

        Reads the host inputs and the stores; owns the umbilicus device
        tensors, the flow-field corners, the model and its resume, the shell
        loss structures, the optimiser and LR scheduler, the prepared device
        track tables, and the distributed bookkeeping. Nothing here consumes
        or releases a host structure, which is what lets rebuild_model_state()
        run it a second time.
        """
        interactive_driver = self.interactive_driver
        progress = progress_or_null(self.progress)

        self.num_slices_for_visualisation = self.config.get('output_num_slices_for_visualization', 20)
        self.device = torch.device('cuda')

        # The full z series is a model input. PNG-only slice grids and raster inputs
        # are prepared lazily at final export, and never in a resident VC3D session.
        all_zs = np.arange(self.z_begin, self.z_end)
        self.umbilicus_zyx = torch.from_numpy(
            np.concatenate([all_zs[:, None], self.umbilicus(all_zs)], axis=-1).astype(np.float32)).to(self.device)
        all_zs = torch.from_numpy(all_zs).to(self.device)

        # ==========================================================================
        # Model construction and resume
        # ==========================================================================

        # Load the resume checkpoint (if any) before constructing the model. The
        # model's parameter tensors are shaped by the z-range it was trained with,
        # so when resuming we must build them with the checkpoint's z-range -
        # otherwise the shapes won't match and load_state_dict will fail. This only
        # affects the model's flow-field domain; the optimisation continues to use
        # the current z_begin/z_end for sampling, losses and rendering.
        resume_path = self.resume_path
        self.start_iteration = int(self.resume_step)
        resume_checkpoint = None
        self.model_z_begin, self.model_z_end = self.z_begin, self.z_end
        if resume_path:
            progress.begin(
                'loading', 'Loading fit checkpoint',
                detail=os.path.basename(resume_path))
            resume_checkpoint = load_checkpoint_cpu(resume_path)
            # Only the model's parameter domain is read before construction: it
            # decides the parameter shapes, so it cannot wait for the preflight.
            # Every other invariant this checkpoint must satisfy is checked by
            # inspect_checkpoint() once the model, optimiser and scheduler
            # exist - the same implementation an in-session load-checkpoint
            # runs, against a model deliberately built to be exactly compatible
            # with the checkpoint's own domain.
            if isinstance(resume_checkpoint, dict) and 'z_begin' in resume_checkpoint:
                self.model_z_begin, self.model_z_end = resume_checkpoint['z_begin'], resume_checkpoint['z_end']
                if (self.model_z_begin, self.model_z_end) != (self.z_begin, self.z_end):
                    print(
                        f'using checkpoint z-range [{self.model_z_begin}, {self.model_z_end}) for model parameter shapes (optimisation z-range is [{self.z_begin}, {self.z_end}))')
                    assert self.z_begin >= self.model_z_begin and self.z_end <= self.model_z_end, (
                        f'optimisation z-range [{self.z_begin}, {self.z_end}) extends beyond the checkpoint '
                        f"model z-range [{self.model_z_begin}, {self.model_z_end}); the flow field has no "
                        'parameters outside its domain. Narrow z_begin/z_end to fit within the '
                        'checkpoint range, or train from scratch with the wider range.'
                    )

        self.flow_field_radius = self.config['model_flow_bounds_radius']
        self.flow_min_corner_spiral_zyx = torch.tensor(
            [self.model_z_begin - self.config['model_flow_bounds_z_margin'], -self.flow_field_radius, -self.flow_field_radius], dtype=torch.int64,
            device=self.device)
        self.flow_max_corner_spiral_zyx = torch.tensor(
            [self.model_z_end + self.config['model_flow_bounds_z_margin'], self.flow_field_radius, self.flow_field_radius], dtype=torch.int64,
            device=self.device)

        self.num_training_steps = self.config['optimizer_num_training_steps']
        # _save_model's default completed_iterations is the configured horizon at
        # setup time, unaffected by any later interactive schedule realignment
        # (preserving the former closure's def-time default binding).
        self._initial_num_training_steps = self.num_training_steps

        progress.begin('loading', 'Constructing spiral model')
        self.spiral_and_transform = SpiralAndTransform(
            flow_integration_steps=self.config['model_num_flow_integration_steps'],
            flow_integration_solver=self.config['model_flow_integration_solver'],
            umbilicus_zyx=self.umbilicus_zyx,
            flow_min_corner_zyx=self.flow_min_corner_spiral_zyx,
            flow_max_corner_zyx=self.flow_max_corner_spiral_zyx,
            config=self.config,
            spiral_outward_sense=self.spiral_outward_sense,
        )
        self.spiral_and_transform.to(self.device)

        # ==========================================================================
        # Shell loss setup
        # ==========================================================================

        self.shell_map = None
        self.shell_valid_zyxs_gpu = None

        shell_active = self.shell_patch is not None and self.shell_losses_enabled()
        if shell_active:
            if self.config['loss_weight_shell_outer'] > 0:
                self.shell_map = ShellPolarMap(
                    self.shell_patch,
                    self.umbilicus,
                    z_min=self.z_begin - self.config['model_flow_bounds_z_margin'],
                    z_max=self.z_end + self.config['model_flow_bounds_z_margin'],
                    num_theta_bins=self.config['shell_num_theta_bins'],
                    device=self.device,
                    config=self.config,
                )
            if self.config['loss_weight_shell_patch_radius'] > 0:
                self.shell_valid_zyxs_gpu = self._subsample_shell_radius_pool(self.shell_patch)

        # Dense losses sample out to this index even when shell losses are off.
        self.shell_outer_winding_idx, outer_winding_notes = resolve_outer_winding_idx_and_notes(
            self.config, shell_active, self._infer_outer_winding_idx_for_this_run)
        for note in outer_winding_notes:
            print(note)

        self._dense_inactive_warned = set()

        # ==========================================================================
        # Optimizer and checkpoint helpers
        # ==========================================================================

        # Keep every stage's low- and high-resolution lattices in distinct groups
        # so the HR learning-rate scale is an optimizer setting, not a multiplier
        # in the model's forward path.
        self.low_res_flow_params = [
            flow_field.flows[0]
            for flow_field in self.spiral_and_transform.flow_fields
        ]
        self.high_res_flow_params = [
            flow_field.flows[1]
            for flow_field in self.spiral_and_transform.flow_fields
        ]
        flow_field_params = self.low_res_flow_params + self.high_res_flow_params
        self.gap_expander_params = list(self.spiral_and_transform.gap_expander_params.parameters())
        linear_params = [self.spiral_and_transform.linear_logits]
        grouped_ids = {id(p) for p in flow_field_params + self.gap_expander_params + linear_params}
        other_params = [p for p in self.spiral_and_transform.parameters() if id(p) not in grouped_ids]
        initial_high_res_lr_scale = get_flow_field_high_res_lr_scale(self.config, 0)
        param_groups = [
            {'params': other_params, 'weight_decay': 0.0},
            {'params': linear_params, 'weight_decay': 0.0},
            {'params': self.gap_expander_params, 'weight_decay': self.config['optimizer_weight_decay_gap_expander']},
            {
                'params': self.low_res_flow_params,
                'weight_decay': self.config['optimizer_weight_decay_flow_field'],
                'lr_scale': 1.,
            },
            {
                'params': self.high_res_flow_params,
                'weight_decay': self.config['optimizer_weight_decay_flow_field'],
                'lr': self.config['optimizer_learning_rate'] * initial_high_res_lr_scale,
                'lr_scale': initial_high_res_lr_scale,
            },
        ]
        progress.begin('loading', 'Creating optimizer')
        self.optimiser = torch.optim.AdamW(param_groups, lr=self.config['optimizer_learning_rate'], betas=(0.9, 0.999), eps=1.e-8, fused=True)
        # Influence masks are scoped to one interactive Run request. They are
        # created from that run's pending inputs and discarded before its autosave.
        self.influence_state = None
        self.interactive_influence_loss_weight = 0.0
        self.interactive_influence_anchor_samples = 0
        if self.config['optimizer_exp_lr_schedule']:
            gamma = self.config['optimizer_lr_final_factor'] ** (1.0 / max(1, self.num_training_steps))
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser, gamma=gamma)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, lambda step: 1.)

        if resume_path:
            # Phase 1 of the same two-phase load an in-session checkpoint
            # request uses: inspect on the CPU and refuse before anything
            # resident is touched. Checkpoints written before schema version 2
            # predate several identity fields and are still accepted here, on
            # the CLI/startup path that has no live state to protect; an
            # in-session load refuses them.
            verdict = self.inspect_checkpoint(
                resume_checkpoint, source=resume_path, allow_legacy_schema=True)
            if not verdict.accepted:
                raise RuntimeError(verdict.message())
            print(f'resuming from {resume_path} at iteration '
                  f'{verdict.completed_iterations if verdict.completed_iterations is not None else self.start_iteration}')
            progress.begin(
                'loading', 'Restoring model and optimizer',
                detail=os.path.basename(resume_path))
            # Phase 2: apply. The LR realignment for a resident session is the
            # unconditional one below, so it is not requested twice here.
            self.apply_checkpoint(resume_checkpoint,
                                  fallback_iteration=self.start_iteration)
            # load_state_dict has moved the model and optimiser state to their
            # destination tensors.  Release the CPU-side archive mappings before
            # entering the resident training loop.
            del resume_checkpoint
            resume_checkpoint = None

        if interactive_driver is not None:
            # A checkpoint may carry the scheduler state from a shorter horizon.
            # The active session configuration is authoritative.
            self._realign_lr_schedule(self.start_iteration)

        progress.begin('loading', 'Synchronizing model across GPU workers')
        broadcast_model_params(self.spiral_and_transform, self.dist)

        if os.environ.get('FIT_SPIRAL_TORCH_PROFILE') == '1':
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=5, warmup=2, active=2, repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace(f'{self.out_path}/profile.out'),
                record_shapes=True,
                with_stack=True,
            )
            self.profiler.start()
        else:
            self.profiler = None

        # ==========================================================================
        # Track training inputs
        # ==========================================================================

        # A resident session drops the per-track input arrays once the tables
        # exist (release_setup_only_tracks), so on a model-stage rebuild there
        # is nothing left to prepare them from. Nothing is lost: the tables are
        # a function of the host tracks and of track_* settings, and no track
        # setting is on MODEL_STAGE_KEYS, so the tables already in hand are
        # exactly what preparing them again would produce.
        retained_tracks = self.using_tracks and self.tracks is None
        if not retained_tracks:
            self.prepared_main_tracks = None
            self.preview_extent_tracks = self.tracks
        if self.using_tracks and not retained_tracks:
            progress.begin(
                'loading', 'Preparing tracks for optimization',
                detail=f'{len(self.tracks):,} tracks')
            self.prepared_main_tracks = prepare_main_phase_tracks(
                self.tracks,
                None,
                float(self.config['track_exclusion_radius']),
                self.device,
                anchor_tree=self.trusted_geometry_tree,
                sampling_config=self.track_sampling_config,
                track_families=self.track_families,
                track_source_ids=self.track_source_ids,
                crossing_cache=self.track_crossing_cache,
                track_graph=self.track_graph,
            )
            # The sidecar CSR is setup-only. The prepared bundle now owns only its
            # fixed-width training tables, so release the whole-DB graph promptly.
            if interactive_driver is None:
                self.track_crossing_cache = None
                self.track_graph = None
            # With the usual zero exclusion radius, the training bundle already
            # contains every authoritative track point as one flat CPU tensor.  Reuse it
            # for preview bounds instead of walking millions of short NumPy tracks.
            if self.prepared_main_tracks is not None:
                input_track_points = (
                    int(self.tracks.selected_lengths.sum())
                    if isinstance(self.tracks, PackedTrackCollection)
                    else sum(len(track) for track in self.tracks))
                if self.prepared_main_tracks['flat_zyx_cpu'].shape[0] == input_track_points:
                    self.preview_extent_tracks = (self.prepared_main_tracks['flat_zyx_cpu'],)

        # The trusted cloud's double-precision cKDTree is setup-only data on a
        # one-shot fit; prepare_main_phase_tracks above is its last reader.
        # Track sampling retains its own compact offsets and coordinates.
        if interactive_driver is None:
            self.trusted_geometry_tree = None

        self.slice_to_spiral_transform = self.spiral_and_transform.get_slice_to_spiral_transform()
        self.dr_per_winding = self.spiral_and_transform.get_dr_per_winding()

        # Caches are recomputed lazily once the corresponding DT loss is active.
        # Updates are deterministic given the transform, so DDP ranks stay consistent.
        def report_first_dt_target_cache(kind, cache):
            if self.dist.is_main_process:
                message = (
                    f'dt-target[{kind}]: {int(cache["valid"].numel())} objects, '
                    f'{cache.get("num_points", 0)} points'
                )
                if 'main_component_fraction' in cache:
                    message += f', main-component fraction {cache["main_component_fraction"]:.3f}'
                print(message)

        self.dt_target_cache_manager = DtTargetCacheManager(
            self.config['dt_target_update_interval'], report_first_dt_target_cache,
        )

        if self.dist.is_distributed:
            np.random.seed(self.config['optimizer_random_seed'] + self.dist.rank)
            torch.manual_seed(self.config['optimizer_random_seed'] + self.dist.rank)
        self.dist_grad_params = list(self.spiral_and_transform.parameters())
        self.dist_grad_named = list(self.spiral_and_transform.named_parameters())
        if self.dist.is_main_process:
            n_params = sum(p.numel() for p in self.dist_grad_params)
            n_bytes = sum(p.numel() * p.element_size() for p in self.dist_grad_params)
            print(
                f'trainable parameters: {n_params:,} ({n_bytes / 1e6:.1f} MB) - '
                'gradient volume all-reduced every step in distributed mode'
            )
        self.step_timer = StepTimer(
            enabled=os.environ.get('FIT_SPIRAL_PROFILE_STEPS') == '1',
            report=self.dist.is_main_process,
        )
        self.nonfinite_grad_steps = torch.zeros((), device=self.dist_grad_params[0].device)
        self.nonfinite_grad_by_param = {name: torch.zeros((), device=p.device) for name, p in self.dist_grad_named}
        self.interactive_dt_resume_iteration = None

    # What _build_model_state() constructs, and therefore what
    # rebuild_model_state() releases before running it again. Written out
    # rather than derived, so device state added to the model stage without a
    # decision about its release is a visible omission instead of a leak.
    # prepared_main_tracks/preview_extent_tracks are deliberately absent: see
    # the retained_tracks note in _build_model_state.
    _MODEL_STAGE_ATTRIBUTES = (
        'num_slices_for_visualisation', 'device', 'umbilicus_zyx',
        'start_iteration', 'model_z_begin', 'model_z_end',
        'flow_field_radius', 'flow_min_corner_spiral_zyx',
        'flow_max_corner_spiral_zyx', 'num_training_steps',
        '_initial_num_training_steps', 'spiral_and_transform', 'shell_map',
        'shell_valid_zyxs_gpu', 'shell_outer_winding_idx',
        '_dense_inactive_warned', 'low_res_flow_params',
        'high_res_flow_params', 'gap_expander_params', 'optimiser',
        'influence_state', 'interactive_influence_loss_weight',
        'interactive_influence_anchor_samples', 'lr_scheduler', 'profiler',
        'slice_to_spiral_transform', 'dr_per_winding',
        'dt_target_cache_manager', 'dist_grad_params', 'dist_grad_named',
        'step_timer', 'nonfinite_grad_steps', 'nonfinite_grad_by_param',
        'interactive_dt_resume_iteration',
    )

    def rebuild_model_state(self):
        """Replace the model stage, keeping the host inputs and the stores.

        The cheap rebuild: a configuration change confined to
        config.MODEL_STAGE_KEYS reaches nothing load_host_inputs() or
        _build_store_state() produced, so this releases exactly what
        _build_model_state() owns and runs it again, against a model built
        from the current configuration and resumed from the current
        resume_path — the same session a full rebuild would have produced,
        without re-reading the dataset or re-materialising the brick pools.

        Fitter-thread only: the CUDA tensors released here are freed by the
        thread that owns them, as every other release in this class is.
        """
        if getattr(self, 'profiler', None) is not None:
            self.profiler.stop()
        for name in self._MODEL_STAGE_ATTRIBUTES:
            setattr(self, name, None)
        # The optimiser state and the flow-field lattices are the session's
        # largest allocations; hand the arena back before asking for their
        # replacements rather than peaking at twice the model.
        gc.collect()
        torch.cuda.empty_cache()
        self._build_model_state()

    # ==========================================================================
    # Checkpoint save/load
    # ==========================================================================

    def _checkpoint_payload(self, completed_iterations):
        return {
            'schema_version': 2,
            'completed_iterations': int(completed_iterations),
            'spiral_and_transform': self.spiral_and_transform.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'cfg': durable_config(self.config),
            'requested_config': durable_config(
                getattr(self.interactive_driver, 'requested_config', dict(self.config))),
            'resolved_config': durable_config(self.config),
            'lasagna_scale': self.lasagna_scale,
            'lasagna_group': self.normal_zarr_group,
            'surf_sdt_fingerprint': (
                self.sdt_volume['fingerprint'] if self.sdt_volume is not None else None),
            'winding_inference_fingerprint': (
                self.winding_inference.fingerprint
                if self.winding_inference is not None else None),
            # The model z-range, not the run window: a resumed session may
            # optimise a narrower window than the flow field covers, and
            # resume rebuilds parameter shapes from these values.
            'z_begin': self.model_z_begin,
            'z_end': self.model_z_end,
            'spiral_outward_sense': self.spiral_outward_sense,
            'numpy_rng_state': np.random.get_state(),
            'torch_cpu_rng_state': torch.random.get_rng_state(),
            'torch_cuda_rng_states': torch.cuda.get_rng_state_all(),
            'input_manifest': dict(getattr(self.interactive_driver, 'input_manifest', {})),
            'preview_first_winding': 10,
        }

    def save_checkpoint(self, path, completed_iterations):
        destination = os.path.abspath(path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        temporary = f'{destination}.tmp-{os.getpid()}-{time.time_ns()}'
        try:
            torch.save(self._checkpoint_payload(completed_iterations), temporary)
            # 'rb+' not 'rb': fsync on Windows (_commit) requires a writable descriptor.
            with open(temporary, 'rb+') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            try:
                directory_fd = os.open(os.path.dirname(destination), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
            return destination
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _save_model(self, suffix, completed_iterations=None):
        if completed_iterations is None:
            completed_iterations = self._initial_num_training_steps
        return self.save_checkpoint(f'{self.out_path}/checkpoint_{suffix}.ckpt', completed_iterations)

    def inspect_checkpoint(self, checkpoint, *, source='',
                           allow_legacy_schema=False):
        """Decide, on the CPU and without mutating anything, whether this
        checkpoint may be applied to this live fit.

        Phase 1 of the two-phase load. It reads only the CPU-side checkpoint
        mapping and the live context's already-constructed model, optimiser,
        scheduler and configuration; it writes nothing, so a refusal leaves
        the session exactly as it was. Every invariant is reported, not just
        the first, because the caller has to explain a refusal to a user who
        picked the wrong file.

        The rule is strict refusal over implicit rebuilding: a checkpoint that
        does not match the live model domain and structural configuration is
        refused here, and the explicit rebuild/new-fit path is what changes a
        model domain.

        ``allow_legacy_schema`` admits pre-v2 checkpoints, which do not carry
        the Lasagna group, outward sense, SDT fingerprint or model z-domain
        fields. Only the startup/CLI restore passes it: there is no live
        session to protect there, and the model is built from the checkpoint.
        """
        reasons = []
        if not isinstance(checkpoint, dict):
            return CheckpointVerdict(
                False, ('checkpoint is not a state dictionary',), source=source)

        # --- schema -------------------------------------------------------
        schema_version = int(checkpoint.get('schema_version', 1) or 1)
        modern = schema_version >= 2
        if not modern and not allow_legacy_schema:
            reasons.append(
                f'checkpoint schema version {schema_version} predates the '
                'identity fields an in-session load verifies (expected >= 2)')
        for key in ('spiral_and_transform', 'optimiser', 'cfg'):
            if checkpoint.get(key) is None:
                reasons.append(f'checkpoint has no {key!r} entry')

        # --- scroll / dataset identity ------------------------------------
        if checkpoint.get('lasagna_scale') != self.lasagna_scale:
            reasons.append(
                f'checkpoint lasagna_scale={checkpoint.get("lasagna_scale")!r} '
                f'does not match this fit ({self.lasagna_scale!r})')
        if modern:
            if checkpoint.get('lasagna_group') != self.normal_zarr_group:
                reasons.append(
                    f'checkpoint Lasagna group '
                    f'{checkpoint.get("lasagna_group")!r} does not match '
                    f'requested group {self.normal_zarr_group!r}')
            if checkpoint.get('spiral_outward_sense') != self.spiral_outward_sense:
                reasons.append(
                    f'checkpoint outward sense '
                    f'{checkpoint.get("spiral_outward_sense")!r} does not '
                    f'match requested sense {self.spiral_outward_sense!r}')
        checkpoint_dataset = str(
            (checkpoint.get('input_manifest') or {}).get('dataset_root') or '')
        dataset_root = str(getattr(self.paths, 'dataset_root', '') or '')
        if checkpoint_dataset and dataset_root and checkpoint_dataset != dataset_root:
            reasons.append(
                f'checkpoint was written against dataset {checkpoint_dataset!r}, '
                f'not {dataset_root!r}')

        # --- Lasagna / SDT store identity ---------------------------------
        # The SDT store is an independent input: the Lasagna group/scale checks
        # above do not cover it. Reject an unexpected change in its content
        # fingerprint whenever an SDT-driven loss is enabled. Paths may
        # legitimately move and coverage may legitimately grow (--resume
        # extension of an ROI-first build), so only the content-identity fields
        # compare - 'created'/'git_commit' are stamped once at store creation
        # and anchor the identity.
        if modern and self.phase_mode:
            checkpoint_fingerprint = comparable_sdt_fingerprint(
                checkpoint.get('surf_sdt_fingerprint'))
            current_fingerprint = comparable_sdt_fingerprint(
                self.sdt_volume['fingerprint']
                if self.sdt_volume is not None else None)
            if (checkpoint_fingerprint is not None
                    and checkpoint_fingerprint != current_fingerprint):
                reasons.append(
                    'checkpoint surf-SDT fingerprint does not match the '
                    'resolved store while an SDT-driven loss is enabled:'
                    f'\n      checkpoint: {checkpoint_fingerprint}'
                    f'\n      current:    {current_fingerprint}')

        if modern and self.winding_model_mode:
            checkpoint_fingerprint = checkpoint.get(
                'winding_inference_fingerprint')
            current_fingerprint = (
                self.winding_inference.fingerprint
                if self.winding_inference is not None else None)
            if checkpoint_fingerprint is None:
                print(
                    'WARNING: checkpoint has no winding-inference fingerprint; '
                    'the current store cannot be matched')
            elif checkpoint_fingerprint != current_fingerprint:
                reasons.append(
                    'checkpoint winding-inference fingerprint does not match '
                    'the resolved store while inference losses are enabled:'
                    f'\n      checkpoint: {checkpoint_fingerprint}'
                    f'\n      current:    {current_fingerprint}')

        # --- structural configuration -------------------------------------
        checkpoint_cfg = checkpoint.get('cfg')
        if isinstance(checkpoint_cfg, Mapping):
            # Checkpoints store the durable subset of the schema, so the key
            # set compares against that subset. z_begin/z_end joined the schema
            # after many checkpoints were written and are owned by the session
            # request either way, so exactly those two may be absent.
            durable_schema = set(durable_config(dict(self.config)))
            unknown = set(checkpoint_cfg) - durable_schema
            missing = durable_schema - set(checkpoint_cfg) - {'z_begin', 'z_end'}
            if unknown or missing:
                reasons.append(
                    'checkpoint configuration does not match the current '
                    f'schema (unknown: {sorted(unknown)}, '
                    f'missing: {sorted(missing)})')
            incompatible = [
                key for key in CHECKPOINT_MODEL_SHAPE_KEYS
                if key in checkpoint_cfg and checkpoint_cfg[key] != self.config[key]
            ]
            if incompatible:
                reasons.append(
                    'checkpoint model-shaping config mismatch: '
                    + ', '.join(
                        f'{key}={checkpoint_cfg[key]!r} != {self.config[key]!r}'
                        for key in incompatible))
        elif checkpoint_cfg is not None:
            reasons.append('checkpoint configuration is not a mapping')

        # --- model z-domain ------------------------------------------------
        if 'z_begin' in checkpoint and 'z_end' in checkpoint:
            domain = (int(checkpoint['z_begin']), int(checkpoint['z_end']))
            if domain != (int(self.model_z_begin), int(self.model_z_end)):
                reasons.append(
                    f'checkpoint model z-domain [{domain[0]}, {domain[1]}) is '
                    f'not the live model domain [{self.model_z_begin}, '
                    f'{self.model_z_end}); rebuild the fit to change the model '
                    'domain')
        elif not allow_legacy_schema:
            reasons.append(
                'checkpoint does not record a model z-domain, so it cannot be '
                'shown to match the live model')

        # --- model keys and tensor geometry --------------------------------
        model_state = checkpoint.get('spiral_and_transform')
        if isinstance(model_state, Mapping):
            live_state = self.spiral_and_transform.state_dict()
            unexpected = sorted(set(model_state) - set(live_state))
            absent = sorted(set(live_state) - set(model_state))
            if unexpected or absent:
                reasons.append(
                    f'checkpoint model keys differ (unexpected: {unexpected}, '
                    f'missing: {absent})')
            geometry = []
            for key in sorted(set(model_state) & set(live_state)):
                saved, live = model_state[key], live_state[key]
                saved_shape = tuple(getattr(saved, 'shape', ()))
                live_shape = tuple(live.shape)
                if saved_shape != live_shape:
                    geometry.append(f'{key} {saved_shape} != {live_shape}')
                elif getattr(saved, 'dtype', live.dtype) != live.dtype:
                    geometry.append(
                        f'{key} dtype {saved.dtype} != {live.dtype}')
            if geometry:
                reasons.append(
                    'checkpoint tensor geometry differs: ' + ', '.join(geometry))
        elif model_state is not None:
            reasons.append('checkpoint model state is not a mapping')

        # --- optimiser and scheduler compatibility -------------------------
        optimiser_state = checkpoint.get('optimiser')
        if isinstance(optimiser_state, Mapping):
            live_optimiser = self.optimiser.state_dict()
            saved_groups = list(optimiser_state.get('param_groups') or [])
            live_groups = list(live_optimiser.get('param_groups') or [])
            if len(saved_groups) != len(live_groups):
                reasons.append(
                    f'checkpoint optimiser has {len(saved_groups)} parameter '
                    f'groups, this fit has {len(live_groups)}')
            else:
                mismatched = [
                    index for index, (saved, live) in enumerate(
                        zip(saved_groups, live_groups))
                    if list(saved.get('params') or []) != list(live.get('params') or [])
                ]
                if mismatched:
                    reasons.append(
                        'checkpoint optimiser parameter groups do not cover '
                        f'the same parameters (groups {mismatched})')
        elif optimiser_state is not None:
            reasons.append('checkpoint optimiser state is not a mapping')

        scheduler_state = checkpoint.get('scheduler')
        if scheduler_state is not None:
            if not isinstance(scheduler_state, Mapping):
                reasons.append('checkpoint scheduler state is not a mapping')
            else:
                live_scheduler = self.lr_scheduler.state_dict()
                if set(scheduler_state) != set(live_scheduler):
                    reasons.append(
                        'checkpoint scheduler is a different kind of schedule '
                        f'(fields {sorted(scheduler_state)} != '
                        f'{sorted(live_scheduler)})')
                else:
                    saved_base = list(scheduler_state.get('base_lrs') or [])
                    live_base = list(live_scheduler.get('base_lrs') or [])
                    if len(saved_base) != len(live_base):
                        reasons.append(
                            f'checkpoint scheduler tracks {len(saved_base)} '
                            f'parameter groups, this fit has {len(live_base)}')

        completed = checkpoint.get('completed_iterations')
        return CheckpointVerdict(
            not reasons, tuple(reasons),
            completed_iterations=(None if completed is None else int(completed)),
            source=source)

    def apply_checkpoint(self, checkpoint, *, fallback_iteration=0,
                         realign_lr=False):
        """Phase 2: move a preflighted checkpoint into the live fit.

        Only ever called after :meth:`inspect_checkpoint` accepted this exact
        payload on every participating rank. It mutates model, optimiser,
        scheduler, RNG and iteration state; a failure here leaves a partially
        loaded optimiser, which is why callers treat it as fatal to the
        session rather than as a refusal.

        ``completed_iterations`` comes from the checkpoint - the durable step
        the fit actually reached - and the LR schedule is realigned to it
        rather than reset to zero.
        """
        embedded = checkpoint.get('completed_iterations')
        completed = int(fallback_iteration if embedded is None else embedded)
        self.start_iteration = completed
        self.load_checkpoint(checkpoint)
        if checkpoint.get('scheduler') is None:
            for _ in range(completed):
                self.lr_scheduler.step()
        self._restore_rng_state(checkpoint)
        if realign_lr:
            self._realign_lr_schedule(completed)
        return completed

    def _restore_rng_state(self, checkpoint):
        if checkpoint.get('numpy_rng_state') is not None:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if checkpoint.get('torch_cpu_rng_state') is not None:
            torch.random.set_rng_state(checkpoint['torch_cpu_rng_state'])
        if checkpoint.get('torch_cuda_rng_states') is not None:
            # The checkpoint holds one state per GPU on the machine that saved
            # it, which may not match this machine's device count.
            saved_cuda_states = checkpoint['torch_cuda_rng_states']
            local_device_count = torch.cuda.device_count()
            if len(saved_cuda_states) != local_device_count:
                print(f'checkpoint has {len(saved_cuda_states)} CUDA RNG states but '
                      f'{local_device_count} device(s) are visible; restoring the first '
                      f'{min(len(saved_cuda_states), local_device_count)}')
            for device_index, state in enumerate(saved_cuda_states[:local_device_count]):
                torch.cuda.set_rng_state(state, device_index)

    def load_checkpoint(self, checkpoint):
        transformed_spiral_state, optimiser_state = checkpoint['spiral_and_transform'], checkpoint['optimiser']
        self.spiral_and_transform.load_state_dict(transformed_spiral_state)
        self.optimiser.load_state_dict(optimiser_state)
        # Older checkpoints could have been saved while influence masking had
        # disabled gap weight decay. Influence state is no longer restored, so
        # restore the session configuration explicitly as well.
        gap_param = self.gap_expander_params[0]
        gap_group = next(group for group in self.optimiser.param_groups
                         if any(param is gap_param for param in group['params']))
        gap_group['weight_decay'] = self.config['optimizer_weight_decay_gap_expander']
        if checkpoint.get('scheduler') is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])

    def _rebuild_unverified_patch_inputs(self, exclusion_radius):
        """Reload only the unverified-patch pool for a Run-boundary mask edit."""
        if not self.unverified_patches_path:
            return {}, [], None, None
        candidates = self._load_patches_from_dir(self.unverified_patches_path)
        for patch_id, patch in list(candidates.items()):
            cells_to_erode = patch.erosion_cells(self.config['patch_erode_patches'])
            if (cells_to_erode > 0
                    and not erode_patch_valid_region(patch, cells_to_erode)):
                del candidates[patch_id]
                continue
            if not patch_intersects_z_roi(patch, self.z_begin, self.z_end):
                del candidates[patch_id]
                continue
            patch.release_derived_caches()
        candidates, n_masked, n_dropped = \
            _mask_unverified_patches_near_trusted_geometry(
                candidates, self.trusted_geometry_tree, exclusion_radius)
        print(
            f'unverified patches: remasked {n_masked} vertices near trusted '
            f'geometry (radius {exclusion_radius:.1f}), dropped {n_dropped}; '
            f'{len(candidates)} remain')
        candidate_list = list(candidates.values())
        probabilities = (
            self._prepare_patch_sampling_cache(candidate_list)
            if candidate_list else None)
        atlas = (
            PatchAtlas(candidates, device='cuda')
            if candidate_list else None)
        return candidates, candidate_list, probabilities, atlas

    def _prepare_png_visualization_inputs(self):
        zs = np.linspace(
            self.z_begin,
            self.z_end - 1,
            min(self.num_slices_for_visualisation, self.z_end - 1 - self.z_begin),
            dtype=np.int64,
        )
        if self.scroll_zarr is not None:
            subvolume_shape = (self.z_end - self.z_begin, *self.scroll_zarr.shape[1:])
            print('loading slices for visualisation')
            vis_zs = np.floor(zs / self.render_volume_scale).astype(np.int64)
            scroll_slices = (
                torch.from_numpy(self.scroll_zarr[vis_zs]).to(torch.float32)
                / np.iinfo(self.scroll_zarr.dtype).max * 0.75 * 255
            ).to(torch.uint8)
        else:
            subvolume_shape = (
                self.z_end - self.z_begin,
                int(np.ceil(32693 / self.render_volume_scale)),
                int(np.ceil(32693 / self.render_volume_scale)),
            )
            scroll_slices = torch.zeros([len(zs), *subvolume_shape[1:]])

        prediction_slices, quad_labels, _ = overlay_patches_on_slices(
            self.verified_patches_list,
            zs,
            subvolume_shape[1:],
            self.cache_path,
            canvas_scale=self.render_volume_scale,
        )
        yx = torch.stack(torch.meshgrid(
            torch.arange(subvolume_shape[1], dtype=torch.float32),
            torch.arange(subvolume_shape[2], dtype=torch.float32),
            indexing='ij',
        ), axis=-1).to(self.device) * self.render_volume_scale
        return zs, yx, scroll_slices, prediction_slices, quad_labels

    def clear_interactive_influence(self):
        """End the current Run request's influence window.

        Called by the runtime when an interactive Run reaches its target, and
        defensively before a new incorporation begins.
        """
        self.interactive_dt_resume_iteration = None
        if self.influence_state is None:
            self.interactive_influence_loss_weight = 0.0
            self.interactive_influence_anchor_samples = 0
            return
        self.influence_state.deactivate_(self.spiral_and_transform, self.optimiser)
        self.influence_state = None
        self.interactive_influence_loss_weight = 0.0
        self.interactive_influence_anchor_samples = 0

    def export_preview(self, generation_path, surface_id):
        progress = progress_or_null(self.progress)
        # Export has its own saved RNG envelope so pausing does not alter the
        # stochastic training sequence.
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()
        try:
            manifest = save_combined_preview(
                self.spiral_and_transform.get_slice_to_spiral_transform(),
                self.spiral_and_transform.get_dr_per_winding(),
                self.verified_patches_list,
                self.unattached_pcl_strips,
                generation_path,
                self.config,
                self.z_begin,
                self.z_end,
                self.voxel_size_um,
                get_or_build_unattached_pcl_flat,
                tracks=self.preview_extent_tracks,
                surface_id=surface_id,
                progress=progress,
            )
            diagnostic_weights = {
                name: self.config.get(f'loss_weight_{name}', 0.0)
                for name in (
                    'patch_radius', 'patch_dt',
                    'unverified_patch_radius', 'unverified_patch_dt',
                    'sym_dirichlet', 'rel_winding', 'abs_winding',
                    'dense_normals', 'dense_spacing', 'dense_attachment',
                    'unattached_pcl_radius', 'unattached_pcl_dt',
                    'track_radius', 'track_dt', 'shell_patch_radius',
                )
            }
            if self._phase_mode_active():
                diagnostic_weights['dense_spacing_phase'] = max(
                    float(self.config['loss_weight_dense_spacing']), 1.0)
                diagnostic_weights['dense_spacing_count'] = max(
                    float(self.config['loss_weight_dense_spacing_count']), 1.0)
            if self._winding_model_mode_active():
                diagnostic_weights['dense_spacing_winding_model_relative'] = max(
                    float(self.config['loss_weight_dense_spacing']), 1.0)
                diagnostic_weights['dense_spacing_winding_model_density'] = max(
                    float(self.config['loss_weight_dense_spacing_density']), 1.0)
            transform = self.spiral_and_transform.get_slice_to_spiral_transform()
            dr = self.spiral_and_transform.get_dr_per_winding()
            progress.begin(
                'exporting_preview', 'Computing preview diagnostics')
            recorder = LossMapRecorder(
                manifest,
                generation_path,
                z0=self.z_begin - int(self.config['model_flow_bounds_z_margin']),
                grid_spacing=int(self.config['output_step_size']),
                dr_per_winding=dr,
                weights=diagnostic_weights,
            )
            with torch.no_grad(), capture_loss_maps(recorder, suppress_errors=True):
                get_patch_and_umbilicus_losses(
                    transform, dr,
                    self.config['sample_count_patches_per_step'],
                    self.config['sample_count_patches_per_step_for_dt'],
                    self.verified_patches_list, self.patch_atlas,
                    self.patch_sampling_probabilities, self.umbilicus_zyx,
                    compute_dt=self.config['loss_weight_patch_dt'] > 0,
                    shell_valid_zyxs=self.shell_valid_zyxs_gpu,
                    shell_outer_winding_idx=self.shell_outer_winding_idx,
                    cfg=self.config,
                )
                if self.unverified_patch_atlas is not None:
                    get_unverified_patch_losses(
                        transform, dr,
                        self.config['sample_count_unverified_patches_per_step'],
                        self.config['sample_count_unverified_patches_per_step_for_dt'],
                        self.unverified_patches_list, self.unverified_patch_atlas,
                        self.unverified_patch_sampling_probabilities,
                        compute_dt=self.config['loss_weight_unverified_patch_dt'] > 0,
                        cfg=self.config,
                    )
                if self.config['loss_weight_sym_dirichlet'] > 0:
                    get_symmetric_dirichlet_loss(
                        transform, dr, self.shell_outer_winding_idx,
                        self.config['sample_count_regularisation_points'],
                        cfg=self.config, z_begin=self.z_begin, z_end=self.z_end)
                if self.config['loss_weight_rel_winding'] > 0 and self.cross_patch_pcls:
                    get_patch_rel_winding_loss(
                        transform, dr, self.verified_patches, self.patch_atlas,
                        self.cross_patch_pcls, self.pcl_sampling_strata['cross_patch'],
                        cfg=self.config, z_begin=self.z_begin, z_end=self.z_end)
                if self.config['loss_weight_abs_winding'] > 0 and self.cross_patch_pcls:
                    get_patch_abs_winding_loss(
                        transform, dr, self.verified_patches, self.patch_atlas,
                        self.cross_patch_pcls,
                        cfg=self.config, z_begin=self.z_begin, z_end=self.z_end)
                if self.lasagna_volume is not None:
                    for _loss_name, _loss_value in iter_lasagna_losses(
                            transform, dr, self.lasagna_volume,
                            self.shell_outer_winding_idx,
                            self.config['sample_count_dense_normal_points'],
                            compute_spacing=self.grad_mag_spacing_enabled,
                            cfg=self.config, z_begin=self.z_begin, z_end=self.z_end):
                        pass
                if self._phase_mode_active():
                    preview_generator = torch.Generator(device=dr.device)
                    preview_generator.manual_seed(0x243F6A88)
                    for _loss_name, _loss_value, _metrics in iter_phase_bundle_losses(
                            self.spiral_and_transform, transform, dr, self.sdt_volume,
                            self.lasagna_volume, self.shell_outer_winding_idx, self.config,
                            self.z_begin, self.z_end, generator=preview_generator):
                        pass
                if self._winding_model_mode_active():
                    preview_generator = torch.Generator(device=dr.device)
                    preview_generator.manual_seed(0x13198A2E)
                    get_winding_inference_losses(
                        transform, dr, self.winding_inference, self.config,
                        self.z_begin, self.z_end,
                        generator=preview_generator)
                if self.unattached_pcl_strips:
                    get_unattached_pcl_strip_losses(
                        transform, dr, self.unattached_pcl_strips,
                        self.unattached_components,
                        self.unattached_component_edges,
                        self.pcl_sampling_strata['unattached'],
                        get_or_build_unattached_pcl_flat,
                        self.config['sample_count_unattached_pcls_per_step'],
                        self.config['sample_count_unattached_pcl_points_per_step'],
                        compute_dt=self.config['loss_weight_unattached_pcl_dt'] > 0,
                        cfg=self.config,
                    )
                if self.prepared_main_tracks is not None:
                    for _loss_name, _loss_value in iter_track_losses(
                            transform, dr, self.prepared_main_tracks, self.config,
                            compute_dt=self.config['loss_weight_track_dt'] > 0):
                        pass
            # Per-pair aggregated crossing counts: mean_count - m per winding
            # pair, the measurement behind any future discrete
            # insert/remove/reindex operation (gradient descent cannot perform
            # those). Written next to the loss maps as a preview artifact.
            if self._phase_mode_active() and self.shell_outer_winding_idx is not None:
                try:
                    with torch.no_grad():
                        pair_rows = aggregate_pair_counts(
                            transform, dr, self.sdt_volume,
                            self.shell_outer_winding_idx, self.config, self.z_begin, self.z_end)
                    pair_table_name = 'dense_spacing_pair_counts.json'
                    with open(os.path.join(generation_path, pair_table_name),
                              'w', encoding='utf-8') as stream:
                        json.dump(pair_rows, stream, indent=1)
                    manifest = dict(manifest)
                    manifest['dense_spacing_pair_counts'] = pair_table_name
                except Exception as error:
                    print('WARNING: could not aggregate per-pair crossing counts: '
                          f'{type(error).__name__}: {error}')
            if recorder.error is not None:
                print('WARNING: could not generate Spiral loss overlays: '
                      f'{type(recorder.error).__name__}: {recorder.error}')
                return manifest
            try:
                entries = recorder.finish()
                return attach_loss_maps_to_manifest(manifest, generation_path, entries)
            except Exception as error:
                print('WARNING: could not publish Spiral loss overlays: '
                      f'{type(error).__name__}: {error}')
                return manifest
        finally:
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            torch.cuda.set_rng_state_all(cuda_states)

    def incorporate_interactive_inputs(self, records, influence_config=None, *,
                                       current_iteration, target_iteration):
        """Append uploaded ephemeral inputs to the resident fit structures.

        current_iteration/target_iteration describe the Run request queued
        alongside the new inputs (the runtime sets the target before this
        method runs at the pause boundary); they size the DT-free window.

        Runs on the fitter thread at a pause boundary. Incorporation is
        append-only: only the new items are loaded and validated, and they are
        concatenated onto the structures the fitter already holds (the patch
        GPU atlas, the sampling caches, the PCL strip list). Existing tensors
        and prepared samplers are reused untouched. The record order is the
        service's deterministic order, so a multi-rank session would append the
        same items in the same order on every rank.

        Returns the warnings this incorporation raised, for the runtime to
        publish on the session status (nothing here is fatal enough to refuse
        the inputs).
        """
        # Incorporation has its own saved RNG envelope so adding inputs does
        # not alter the stochastic training sequence (same discipline as the
        # interactive preview export).
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()
        try:
            # Be defensive about a previously interrupted boundary: a new
            # batch must never union with an earlier Run request's masks.
            self.clear_interactive_influence()
            run_cfg = dict(self.config)
            run_cfg.update(dict(influence_config or {}))
            new_patches = {}
            new_collections = {}
            # (input_id, pcl) per uploaded fiber, for the unresolved-link
            # warning below.
            new_fibers = []
            for record in records:
                kind = record.get('kind')
                path = record.get('path')
                input_id = record.get('id')
                if kind == 'patch':
                    if self.config['input_disable_patches']:
                        raise RuntimeError('disable_patches=True: this session takes no patches')
                    if input_id in self.verified_patches or input_id in new_patches:
                        raise RuntimeError(f'Patch {input_id!r} is already part of this session')
                    patch = load_tifxyz(path)
                    cells_to_erode = patch.erosion_cells(self.config['patch_erode_patches'])
                    if cells_to_erode > 0 and not erode_patch_valid_region(patch, cells_to_erode):
                        raise RuntimeError(f'Patch {input_id!r} has no valid quads after erosion')
                    if not patch_intersects_z_roi(patch, self.z_begin, self.z_end):
                        raise RuntimeError(
                            f'Patch {input_id!r} does not intersect the fitted z range '
                            f'[{self.z_begin}, {self.z_end})')
                    patch.release_derived_caches()
                    new_patches[input_id] = patch
                elif kind == 'fiber':
                    pcl = load_fiber_point_collection(
                        path, self.next_id, min_point_spacing=self.config['pcl_fiber_min_point_spacing'])
                    if pcl is None:
                        raise RuntimeError(f'Fiber {input_id!r} has no usable control points')
                    pcl['source_file'] = path
                    pcl.setdefault('metadata', {})['winding_is_absolute'] = False
                    pcl['metadata']['input_role'] = 'fiber'
                    pcl['sampling_group'] = 'fibers'
                    new_fibers.append((input_id, pcl))
                    new_collections[self.next_id] = pcl
                    self.next_id += 1
                elif kind == 'pcl':
                    role = record.get('role')
                    loaded = load_point_collection(path) or {}
                    if not loaded:
                        raise RuntimeError(f'PCL document {input_id!r} contains no collections')
                    for pcl in loaded.values():
                        pcl['source_file'] = path
                        pcl['sampling_group'] = path
                        pcl.setdefault('metadata', {})['winding_is_absolute'] = role == 'absolute'
                        pcl['metadata']['input_role'] = role
                        new_collections[self.next_id] = pcl
                        self.next_id += 1
                else:
                    raise RuntimeError(f'Unknown ephemeral input kind {kind!r}')

            # Weighted sampling intentionally requires every group to be named.
            # Validate uploaded groups before mutating any resident patch/PCL pools,
            # so a missing weight cannot leave a half-incorporated session behind.
            if new_collections and self.config['pcl_sampling_weights'] is not None:
                build_pcl_sampling_strata(
                    (pcl['sampling_group'] for pcl in new_collections.values()),
                    self.config)

            # ---- Patches: sampling caches, probabilities, atlas append ----
            if new_patches:
                for patch in new_patches.values():
                    self._prepare_patch_sampling_cache([patch])
                self.verified_patches.update(new_patches)
                self.verified_patches_list.extend(new_patches.values())
                areas = np.array([float(p.area) for p in self.verified_patches_list],
                                 dtype=np.float32)
                inv_weights = areas ** 0.5
                self.patch_sampling_probabilities = inv_weights / inv_weights.sum()
                self.patch_atlas.append_patches(new_patches)
                if self.config['dt_target_mode'] == 'whole_object_quantile':
                    prepare_patch_dt_target_samples(
                        list(new_patches.values()),
                        self.config['sample_count_patch_dt_target_points'], self.config['dt_target_max_stride'],
                    )

            # ---- Point collections: link, classify, strip-materialise ----
            if new_collections:
                for pcl in new_collections.values():
                    for point in pcl['points'].values():
                        point['zyx'] = np.array(
                            [point['p'][2], point['p'][1], point['p'][0]],
                            dtype=np.float32)
                    pcl['chain'] = SequenceChain(pcl)
                link_points_to_patches(
                    self.verified_patches,
                    new_collections,
                    tolerance=self.link_distance_tolerance,
                    surface_index_tolerance=self.link_distance_tolerance,
                    distance_scale=1.0,
                    general_hit_policy='largest_area',
                )
                new_cross_patch = {}
                new_unattached = {}
                for pid, pcl in new_collections.items():
                    num_attached = sum(1 for point in pcl['points'].values() if 'on_patch' in point)
                    num_unattached = len(pcl['points']) - num_attached
                    if pcl.get('metadata', {}).get('winding_is_absolute', False):
                        attached_points = [point for point in pcl['points'].values()
                                           if 'on_patch' in point]
                        if any(not np.isfinite(point['winding_annotation'])
                               or point['winding_annotation'] <= 0
                               for point in attached_points):
                            raise RuntimeError(
                                f'Absolute-winding pcl {pcl.get("name")!r} must annotate every '
                                f'attached point with a positive winding number')
                        new_cross_patch[pid] = pcl
                        continue
                    if num_attached >= 2:
                        new_cross_patch[pid] = pcl
                    if num_unattached >= 1:
                        new_unattached[pid] = copy.deepcopy(pcl) if num_attached >= 2 else pcl

                z_margin = self.config['patch_loss_z_margin']
                for pid in list(new_unattached.keys()):
                    pcl = new_unattached[pid]
                    sorted_items = sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
                    best_start, best_end = 0, 0
                    run_start = 0
                    for i, (_, point) in enumerate(sorted_items):
                        z = point['zyx'][0]
                        if self.z_begin - z_margin <= z < self.z_end + z_margin:
                            if i + 1 - run_start > best_end - best_start:
                                best_start, best_end = run_start, i + 1
                        else:
                            run_start = i + 1
                    kept_items = sorted_items[best_start:best_end]
                    if len(kept_items) < 2:
                        del new_unattached[pid]
                    else:
                        pcl['points'] = dict(kept_items)

                normalise_pcl_winding_annotations(new_cross_patch)
                normalise_pcl_winding_annotations(new_unattached)

                for pcl in new_cross_patch.values():
                    points_by_patch = {}
                    for _, point in sorted(pcl['points'].items(), key=lambda kv: int(kv[0])):
                        if 'on_patch' not in point:
                            continue
                        pid = point['on_patch']['id']
                        if pid not in self.verified_patches:
                            continue
                        points_by_patch.setdefault(pid, []).append(point)
                    pcl['points_by_patch'] = points_by_patch
                    self.cross_patch_pcls.append(pcl)

                min_point_spacing = self.config['pcl_unattached_pcl_min_point_spacing']
                for pcl_id, pcl in new_unattached.items():
                    sorted_items = sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
                    if len(sorted_items) < 2:
                        continue
                    zyxs = np.stack([point['zyx'] for _, point in sorted_items],
                                    axis=0).astype(np.float32)
                    windings = np.array(
                        [point['winding_annotation'] for _, point in sorted_items],
                        dtype=np.float32)
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
                    self.unattached_pcl_strips.append({
                        'id': pcl_id,
                        'name': pcl.get('name'),
                        'source_file': pcl.get('source_file'),
                        'zyxs': zyxs,
                        'windings': windings,
                    })
                    self.unattached_strip_sampling_groups.append(pcl.get('sampling_group'))
                # No 'link_points' on these strips: an uploaded fiber's
                # branches are not resolved (see unresolved_links above), so
                # each new strip is its own singleton component with no
                # junctions to walk.
                # The flat GPU bundle is derived from the strip list; drop it so
                # the next consumer rebuilds it including the appended strips.
                self.unattached_pcl_strips.flat = None
                # Sampling strata index into the (now longer) pools.
                self._rebuild_pcl_sampling_strata()

            if new_patches or new_collections:
                # Whole-object DT target caches index the (now longer) object
                # pools; force recomputation on next use.
                self.dt_target_cache_manager.reset()

            if run_cfg['influence_enabled'] and (new_patches or new_collections):
                self.influence_state = make_influence_state(run_cfg, torch.device('cuda'))
                self.influence_state.activate_or_extend_(
                    new_patches=new_patches,
                    new_collections=new_collections,
                    spiral_and_transform=self.spiral_and_transform,
                    optimiser=self.optimiser,
                    cfg=run_cfg,
                    z_begin=self.z_begin,
                    z_end=self.z_end,
                    anchor_geometry_zyx=self.influence_anchor_geometry,
                )
                self.interactive_influence_loss_weight = float(run_cfg['loss_weight_anchor'])
                self.interactive_influence_anchor_samples = int(
                    run_cfg['sample_count_influence_anchor_samples_per_step'])

            # The runtime sets the target before this method is drained at the
            # pause boundary, so this is exactly the iteration window requested
            # alongside the new inputs. Do not let a later incorporation
            # shorten an already-active DT-free window.
            dt_resume_iteration = get_interactive_dt_resume_iteration(
                current_iteration,
                target_iteration,
                run_cfg['influence_disable_dt_frac'],
            )
            self.interactive_dt_resume_iteration = dt_resume_iteration

            print(f'incorporated {len(new_patches)} patches and '
                  f'{len(new_collections)} point collections into the resident session; '
                  f'DT losses disabled until iteration {self.interactive_dt_resume_iteration}')

            warnings = []
            link_warning = unresolved_fiber_link_warning(
                new_fibers,
                use_links=self.config['pcl_use_fiber_links'],
                use_pending_links=self.config['pcl_use_pending_fiber_links'])
            if link_warning is not None:
                print(f'WARNING: {link_warning}')
                warnings.append(link_warning)
            return warnings
        finally:
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            torch.cuda.set_rng_state_all(cuda_states)

    def apply_config(self, config, path_changes=None, *, current_iteration):
        """Apply Run-scoped settings without replacing the resident fit.

        current_iteration is the session's durable completed-iteration count;
        an LR-schedule change is realigned at that step.
        """
        path_changes = dict(path_changes or {})
        changed = set(config)
        old_values = {key: self.config[key] for key in config}
        self.config.update(config)
        try:
            # Static path changes are rejected by the service. Shell atlas
            # construction settings are classified as prepared-input changes
            # and likewise never reach this live boundary; ordinary shell
            # parameters are just run configuration.
            rebuilt_tracks = None
            replace_prepared_tracks = False
            rebuilt_shell_map = self.shell_map
            rebuilt_shell_outer = self.shell_outer_winding_idx
            rebuilt_shell_valid = self.shell_valid_zyxs_gpu

            reprepare_tracks = bool(changed & {
                'track_max_tortuosity',
                'track_exclusion_radius',
            })
            if reprepare_tracks and rebuilt_tracks is None and self.tracks:
                rebuilt_tracks = prepare_main_phase_tracks(
                    self.tracks, None, float(self.config['track_exclusion_radius']),
                    self.device, anchor_tree=self.trusted_geometry_tree,
                    sampling_config=validate_track_sampling_config(self.config),
                    track_families=self.track_families,
                    track_source_ids=self.track_source_ids,
                    crossing_cache=self.track_crossing_cache,
                    track_graph=self.track_graph)
                replace_prepared_tracks = True

            target_tracks = (
                rebuilt_tracks
                if replace_prepared_tracks else self.prepared_main_tracks)
            if ({'track_length_bin_weights',
                 'track_max_track_crossing_per_step',
                 'track_min_walk_steps_per_track',
                 'track_max_walk_steps_per_track',
                 'track_min_walks_per_track',
                 'track_max_walks_per_track',
                 'track_walk_minimum_cycle_travel'}
                    & changed):
                configure_prepared_track_sampling(target_tracks, config)

            if 'patch_loss_z_margin' in changed:
                self.patch_sampling_probabilities = \
                    self._prepare_patch_sampling_cache(self.verified_patches_list)
                if self.unverified_patches_list:
                    self.unverified_patch_sampling_probabilities = \
                        self._prepare_patch_sampling_cache(
                            self.unverified_patches_list)
            if 'patch_unverified_patch_exclusion_radius' in changed:
                (rebuilt_unverified, rebuilt_unverified_list,
                 rebuilt_unverified_probabilities,
                 rebuilt_unverified_atlas) = \
                    self._rebuild_unverified_patch_inputs(float(
                        self.config['patch_unverified_patch_exclusion_radius']))
            else:
                rebuilt_unverified = self.unverified_patches
                rebuilt_unverified_list = self.unverified_patches_list
                rebuilt_unverified_probabilities = \
                    self.unverified_patch_sampling_probabilities
                rebuilt_unverified_atlas = self.unverified_patch_atlas

            # Loss weights are live settings. If a shell loss is enabled for
            # the first time, construct only the resident structure that loss
            # needs; disabling it releases that structure. Atlas-shaping
            # settings themselves require a full prepared-input rebuild.
            if 'loss_weight_shell_outer' in changed:
                rebuilt_shell_map = (
                    ShellPolarMap(
                        self.shell_patch, self.umbilicus,
                        z_min=self.z_begin - self.config['model_flow_bounds_z_margin'],
                        z_max=self.z_end + self.config['model_flow_bounds_z_margin'],
                        num_theta_bins=self.config['shell_num_theta_bins'],
                        device=self.device, config=self.config)
                    if self.config['loss_weight_shell_outer'] > 0 else None
                )
            if 'loss_weight_shell_patch_radius' in changed:
                rebuilt_shell_valid = (
                    self._subsample_shell_radius_pool(self.shell_patch)
                    if self.config['loss_weight_shell_patch_radius'] > 0 else None
                )
            if 'shell_outer_winding_idx' in changed:
                rebuilt_shell_outer = int(self.config['shell_outer_winding_idx'])

            dt_preparation_changed = bool(changed & {
                'dt_target_mode', 'dt_target_max_stride',
                'sample_count_patch_dt_target_points',
            })
            if dt_preparation_changed:
                self.dt_target_whole_object = (
                    self.config['dt_target_mode'] == 'whole_object_quantile')
                if self.dt_target_whole_object:
                    prepare_patch_dt_target_samples(
                        self.verified_patches_list,
                        self.config['sample_count_patch_dt_target_points'],
                        self.config['dt_target_max_stride'])
                    if self.unverified_patches_list:
                        prepare_patch_dt_target_samples(
                            self.unverified_patches_list,
                            self.config['sample_count_patch_dt_target_points'],
                            self.config['dt_target_max_stride'])
            if any(key.startswith('dt_') for key in changed) \
                    or dt_preparation_changed:
                self.dt_target_cache_manager.update_interval = max(
                    1, int(self.config['dt_target_update_interval']))
                self.dt_target_cache_manager.reset()
            if changed & {
                    'optimizer_exp_lr_schedule',
                    'optimizer_learning_rate',
                    'optimizer_lr_final_factor',
                    'optimizer_num_training_steps',
            }:
                self._realign_lr_schedule(current_iteration)
        except Exception:
            self.config.update(old_values)
            raise

        self.shell_map = rebuilt_shell_map
        self.shell_outer_winding_idx = rebuilt_shell_outer
        self.shell_valid_zyxs_gpu = rebuilt_shell_valid
        if replace_prepared_tracks:
            self.prepared_main_tracks = rebuilt_tracks
            self.preview_extent_tracks = (
                (self.prepared_main_tracks['flat_zyx_cpu'],)
                if self.prepared_main_tracks is not None else ())
        self.unverified_patches = rebuilt_unverified
        self.unverified_patches_list = rebuilt_unverified_list
        self.unverified_patch_sampling_probabilities = \
            rebuilt_unverified_probabilities
        self.unverified_patch_atlas = rebuilt_unverified_atlas

    def step(self, iteration):
        self.step_timer.start('fwd')
        flow_field_high_res_lr_scale = self._apply_high_res_lr_scale(iteration)

        # The tiny graph paths shared by every transform evaluation this
        # iteration (dr softplus, scaled linear logits, pinned gap logits) are
        # cut at detached leaves. Each loss family's backward then owns its
        # whole graph, so autograd can free the family's buffers as the
        # backward pass consumes them instead of retaining the full graph
        # (retain_graph) until the family is released. The leaf gradients
        # accumulated across families flow through the real shared paths once,
        # next to the flow-field gradient flush below.
        shared_transform_outputs = self.spiral_and_transform.get_shared_transform_tensors()
        shared_transform_leaves = tuple(
            output.detach().requires_grad_(True) for output in shared_transform_outputs)
        self.slice_to_spiral_transform = self.spiral_and_transform.get_slice_to_spiral_transform(
            shared=shared_transform_leaves)
        self.dr_per_winding = shared_transform_leaves[0]

        losses = {}
        log_metrics = {
            'flow_field_high_res_lr_scale': flow_field_high_res_lr_scale,
        }

        def backward_family(weighted_losses):
            """Accumulate one loss family's gradients, then release its graph."""
            family_loss = sum(weighted_losses.values())
            if family_loss.requires_grad:
                self.step_timer.stop('fwd')
                self.step_timer.start('bwd')
                # The paths shared with later families end at detached leaves
                # (shared_transform_leaves and the flow fields' internal
                # accumulators), so this family's graph is self-contained and
                # its buffers are freed as the backward pass consumes them.
                family_loss.backward()
                self.step_timer.stop('bwd')
                self.step_timer.start('fwd')
            for name, value in weighted_losses.items():
                losses[name] = value.detach()

        interactive_dt_suppressed = (
            self.interactive_dt_resume_iteration is not None
            and iteration < self.interactive_dt_resume_iteration
        )
        log_metrics['interactive_dt_suppressed'] = float(interactive_dt_suppressed)

        compute_patch_dt = not interactive_dt_suppressed and iteration > self.config['loss_start_patch_dt']
        track_dt_start = self.config['loss_start_patch_dt'] if self.config['loss_start_track_dt'] is None else self.config['loss_start_track_dt']
        compute_track_dt = not interactive_dt_suppressed and iteration > track_dt_start
        unverified_patch_dt_start = self.config['loss_start_patch_dt'] if self.config['loss_start_unverified_patch_dt'] is None else self.config['loss_start_unverified_patch_dt']
        compute_unverified_patch_dt = not interactive_dt_suppressed and iteration > unverified_patch_dt_start

        # Progressive-outward DT gating: winding cutoff that grows from the
        # respective DT start step. None means no gating.
        dt_progressive_outer = self.shell_outer_winding_idx
        patch_dt_max_winding = get_progressive_dt_max_winding(self.config, iteration, self.config['loss_start_patch_dt'], dt_progressive_outer)
        track_dt_max_winding = get_progressive_dt_max_winding(self.config, iteration, track_dt_start, dt_progressive_outer)
        unverified_patch_dt_max_winding = get_progressive_dt_max_winding(self.config, iteration, unverified_patch_dt_start, dt_progressive_outer)
        if patch_dt_max_winding is not None:
            log_metrics['patch_dt_max_winding'] = patch_dt_max_winding
        if track_dt_max_winding is not None:
            log_metrics['track_dt_max_winding'] = track_dt_max_winding

        patch_dt_target_cache = None
        unverified_patch_dt_target_cache = None
        unattached_pcl_dt_target_cache = None
        track_dt_target_cache = None
        if self.dt_target_whole_object:
            if compute_patch_dt and self.config['loss_weight_patch_dt'] > 0 and self.verified_patches_list:
                patch_dt_target_cache = self.dt_target_cache_manager.get('patch', iteration, lambda: compute_patch_dt_target_cache(
                    self.slice_to_spiral_transform, self.dr_per_winding,
                    self.verified_patches_list, self.patch_atlas, self.config['dt_target_floating_threshold'],
                ))
            if compute_unverified_patch_dt and self.config['loss_weight_unverified_patch_dt'] > 0 and self.unverified_patch_atlas is not None:
                unverified_patch_dt_target_cache = self.dt_target_cache_manager.get('unverified_patch', iteration, lambda: compute_patch_dt_target_cache(
                    self.slice_to_spiral_transform, self.dr_per_winding,
                    self.unverified_patches_list, self.unverified_patch_atlas, self.config['dt_target_floating_threshold'],
                ))
            if compute_patch_dt and self.config['loss_weight_unattached_pcl_dt'] > 0 and self.unattached_pcl_strips:
                pcl_flat = get_or_build_unattached_pcl_flat(self.unattached_pcl_strips, torch.device('cuda'))
                if pcl_flat is not None:
                    unattached_pcl_dt_target_cache = self.dt_target_cache_manager.get('unattached_pcl', iteration, lambda: compute_strip_dt_target_cache(
                        self.slice_to_spiral_transform, self.dr_per_winding,
                        pcl_flat['zyxs'], pcl_flat['starts'],
                        windings=pcl_flat['windings'],
                        floating_threshold=self.config['dt_target_floating_threshold'],
                        num_points_per_strip=self.config['sample_count_dt_target_points_per_strip'],
                        max_stride=self.config['dt_target_max_stride'],
                        max_total_points=20_000_000,
                    ))
            if compute_track_dt and self.config['loss_weight_track_dt'] > 0 and self.prepared_main_tracks is not None:
                track_dt_target_cache = self.dt_target_cache_manager.get('track', iteration, lambda: compute_strip_dt_target_cache(
                    self.slice_to_spiral_transform, self.dr_per_winding,
                    self.prepared_main_tracks['flat_zyx_cpu'], self.prepared_main_tracks['offsets'],
                    windings=None,
                    floating_threshold=self.config['dt_target_floating_threshold'],
                    num_points_per_strip=self.config['sample_count_dt_target_points_per_strip'],
                    max_stride=self.config['dt_target_max_stride'],
                    max_total_points=20_000_000,
                ))

        patch_loss_values = get_patch_and_umbilicus_losses(
            self.slice_to_spiral_transform,
            self.dr_per_winding,
            self.config['sample_count_patches_per_step'],
            self.config['sample_count_patches_per_step_for_dt'],
            self.verified_patches_list,
            self.patch_atlas,
            self.patch_sampling_probabilities,
            self.umbilicus_zyx,
            compute_dt=compute_patch_dt,
            shell_valid_zyxs=self.shell_valid_zyxs_gpu,
            shell_outer_winding_idx=self.shell_outer_winding_idx,
            dt_max_winding=patch_dt_max_winding,
            dt_target_cache=patch_dt_target_cache,
            cfg=self.config,
        )
        patch_family = {
            'patch_radius': patch_loss_values[0] * self.config['loss_weight_patch_radius'],
            'patch_dt': patch_loss_values[2] * self.config['loss_weight_patch_dt'],
            'umbilicus': patch_loss_values[1] * self.config['loss_weight_umbilicus'],
        }
        if self.shell_valid_zyxs_gpu is not None:
            patch_family['shell_patch_radius'] = patch_loss_values[3] * self.config['loss_weight_shell_patch_radius']
        backward_family(patch_family)
        del patch_family, patch_loss_values

        if self.unverified_patch_atlas is not None and (
            self.config['loss_weight_unverified_patch_radius'] > 0
            or self.config['loss_weight_unverified_patch_dt'] > 0
        ):
            unverified_loss_values = get_unverified_patch_losses(
                self.slice_to_spiral_transform,
                self.dr_per_winding,
                self.config['sample_count_unverified_patches_per_step'],
                self.config['sample_count_unverified_patches_per_step_for_dt'],
                self.unverified_patches_list,
                self.unverified_patch_atlas,
                self.unverified_patch_sampling_probabilities,
                compute_dt=compute_unverified_patch_dt,
                dt_max_winding=unverified_patch_dt_max_winding,
                dt_target_cache=unverified_patch_dt_target_cache,
                cfg=self.config,
            )
            backward_family({
                'unverified_patch_radius': unverified_loss_values[0] * self.config['loss_weight_unverified_patch_radius'],
                'unverified_patch_dt': unverified_loss_values[1] * self.config['loss_weight_unverified_patch_dt'],
            })
            del unverified_loss_values

        if self.config['loss_weight_sym_dirichlet'] > 0:
            backward_family({
                'sym_dirichlet': get_symmetric_dirichlet_loss(
                    self.slice_to_spiral_transform,
                    self.dr_per_winding,
                    self.shell_outer_winding_idx,
                    self.config['sample_count_regularisation_points'],
                    cfg=self.config, z_begin=self.z_begin, z_end=self.z_end,
                ) * self.config['loss_weight_sym_dirichlet'],
            })

        if self.config['loss_weight_rel_winding'] > 0 and self.cross_patch_pcls:
            backward_family({
                'rel_winding': get_patch_rel_winding_loss(
                    self.slice_to_spiral_transform,
                    self.dr_per_winding,
                    self.verified_patches,
                    self.patch_atlas,
                    self.cross_patch_pcls,
                    self.pcl_sampling_strata['cross_patch'],
                    cfg=self.config, z_begin=self.z_begin, z_end=self.z_end,
                ) * self.config['loss_weight_rel_winding'],
            })

        if self.config['loss_weight_abs_winding'] > 0 and self.cross_patch_pcls:
            backward_family({
                'abs_winding': get_patch_abs_winding_loss(
                    self.slice_to_spiral_transform,
                    self.dr_per_winding,
                    self.verified_patches,
                    self.patch_atlas,
                    self.cross_patch_pcls,
                    cfg=self.config, z_begin=self.z_begin, z_end=self.z_end,
                ) * self.config['loss_weight_abs_winding'],
            })

        if (
            (self.config['loss_weight_dense_normals'] > 0 or self.grad_mag_spacing_enabled)
            and self.lasagna_volume is not None
        ):
            for dense_loss_name, dense_loss_value in iter_lasagna_losses(
                self.slice_to_spiral_transform,
                self.dr_per_winding,
                self.lasagna_volume,
                self.shell_outer_winding_idx,
                self.config['sample_count_dense_normal_points'],
                compute_spacing=self.grad_mag_spacing_enabled,
                cfg=self.config, z_begin=self.z_begin, z_end=self.z_end,
            ):
                weight = (
                    self.config['loss_weight_dense_normals']
                    if dense_loss_name == 'dense_normals'
                    else self.config['loss_weight_dense_spacing']
                )
                backward_family({dense_loss_name: dense_loss_value * weight})
                # Release before the generator builds the next loss's graph,
                # or both large transform graphs are resident at peak.
                del dense_loss_value
            if self.lasagna_volume.get('backend') == 'sparse_cuda':
                log_metrics.update({
                    f'lasagna_{name}': value
                    for name, value in self.lasagna_volume['store'].last_timings.items()
                })

        self._warn_if_sdt_loss_inactive()
        self._warn_if_dense_losses_structurally_disabled()
        if self._winding_model_mode_active():
            inference_losses, inference_metrics = get_winding_inference_losses(
                self.slice_to_spiral_transform,
                self.dr_per_winding,
                self.winding_inference,
                self.config,
                self.z_begin,
                self.z_end,
            )
            backward_family({
                'dense_spacing_winding_model_relative': (
                    inference_losses['dense_spacing_winding_model_relative']
                    * self.config['loss_weight_dense_spacing']),
                'dense_spacing_winding_model_density': (
                    inference_losses['dense_spacing_winding_model_density']
                    * self.config['loss_weight_dense_spacing_density']),
            })
            log_metrics.update(inference_metrics)
            del inference_losses, inference_metrics
        phase_components_active = self._phase_mode_active()
        min_spacing_active = self.config['loss_weight_min_spacing'] > 0
        if phase_components_active or min_spacing_active:
            # SDT-backed phase components require phase mode; the native
            # min-spacing barrier does not. Weights are re-read every step so
            # the barrier can be enabled at a Run boundary in either mode.
            attachment_ramp = (
                get_dense_attachment_ramp(self.config, iteration)
                if phase_components_active else 0.0)
            if phase_components_active:
                log_metrics['dense_attachment_ramp'] = attachment_ramp
            component_weights = phase_bundle_component_weights(
                self.config, attachment_ramp)
            # Components tagged '_shared_graph' (count, phase, shared-batch
            # density) backpropagate through one central-ray graph; summing
            # them into a single backward traverses that graph once instead
            # of once per component. Untagged components (density supplement
            # chunks, min_spacing, attachment) keep their own backward so at
            # most one supplement-chunk graph is resident at a time.
            pending_shared = {}
            for component_name, component_loss, component_metrics in \
                    iter_phase_bundle_losses(
                        self.spiral_and_transform,
                        self.slice_to_spiral_transform,
                        self.dr_per_winding,
                        self.sdt_volume,
                        self.lasagna_volume,
                        self.shell_outer_winding_idx,
                        self.config,
                        self.z_begin,
                        self.z_end,
                        attachment_ramp=attachment_ramp,
                    ):
                weighted = (
                    component_loss * component_weights[component_name])
                if component_metrics.pop('_shared_graph', False):
                    pending_shared[component_name] = weighted
                else:
                    if pending_shared:
                        backward_family(pending_shared)
                        pending_shared = {}
                    backward_family({component_name: weighted})
                # Release before the generator builds the next component's
                # graph, or several large graphs are resident at peak.
                del component_loss, weighted
                log_metrics.update(component_metrics)
            if pending_shared:
                backward_family(pending_shared)
            del pending_shared
            if (phase_components_active
                    and self.lasagna_volume['backend'] == 'sparse_cuda'):
                log_metrics.update({
                    f'dense_spacing_phase_normal_{name}': value
                    for name, value in self.lasagna_volume['store'].last_timings.items()
                })
            if phase_components_active and self.sdt_volume['backend'] == 'sparse_cuda':
                log_metrics.update({
                    f'dense_spacing_phase_sdt_store_{name}': value
                    for name, value in self.sdt_volume['store'].last_timings.items()
                })

        if (
            (self.config['loss_weight_unattached_pcl_radius'] > 0 or self.config['loss_weight_unattached_pcl_dt'] > 0)
            and self.unattached_pcl_strips
        ):
            unattached_loss_values = get_unattached_pcl_strip_losses(
                self.slice_to_spiral_transform,
                self.dr_per_winding,
                self.unattached_pcl_strips,
                self.unattached_components,
                self.unattached_component_edges,
                self.pcl_sampling_strata['unattached'],
                get_or_build_unattached_pcl_flat,
                self.config['sample_count_unattached_pcls_per_step'],
                self.config['sample_count_unattached_pcl_points_per_step'],
                compute_dt=compute_patch_dt,
                dt_max_winding=patch_dt_max_winding,
                dt_target_cache=unattached_pcl_dt_target_cache,
                cfg=self.config,
            )
            backward_family({
                'unattached_pcl_radius': unattached_loss_values[0] * self.config['loss_weight_unattached_pcl_radius'],
                'unattached_pcl_dt': unattached_loss_values[1] * self.config['loss_weight_unattached_pcl_dt'],
            })
            del unattached_loss_values

        if self.prepared_main_tracks is not None:
            for track_loss_name, track_loss_value in iter_track_losses(
                self.slice_to_spiral_transform,
                self.dr_per_winding,
                self.prepared_main_tracks,
                self.config,
                compute_dt=compute_track_dt,
                dt_max_winding=track_dt_max_winding,
                dt_target_cache=track_dt_target_cache,
            ):
                weight = (
                    self.config['loss_weight_track_radius']
                    if track_loss_name == 'track_radius'
                    else self.config['loss_weight_track_dt']
                )
                backward_family({track_loss_name: track_loss_value * weight})
                # Release before the generator builds the next loss's graph,
                # or both large transform graphs are resident at peak.
                del track_loss_value

        shell_metrics = {}
        if self.shell_map is not None:
            shell_outer_loss, shell_metrics = get_shell_outer_loss(
                self.shell_map,
                self.slice_to_spiral_transform,
                self.dr_per_winding,
                self.shell_outer_winding_idx,
                cfg=self.config, z_begin=self.z_begin, z_end=self.z_end,
            )
            backward_family({
                'shell_outer': shell_outer_loss * self.config['loss_weight_shell_outer'],
            })
            del shell_outer_loss

        if (self.influence_state is not None and self.influence_state.active
                and self.interactive_influence_loss_weight > 0):
            backward_family({
                'anchor': self.influence_state.get_anchor_loss(
                    self.slice_to_spiral_transform,
                    self.dr_per_winding,
                    self.interactive_influence_anchor_samples,
                ) * self.interactive_influence_loss_weight,
            })

        loss = sum(losses.values())

        self.step_timer.stop('fwd')
        self.step_timer.start('bwd')
        # Flush every stage's sparse-accumulated field gradient into its parameters.
        for flow_field in self.spiral_and_transform.flow_fields:
            apply_accumulated_field_grad = getattr(flow_field, 'apply_accumulated_field_grad', None)
            if apply_accumulated_field_grad is not None:
                apply_accumulated_field_grad()
        # Propagate the leaf gradients the family backwards accumulated on the
        # shared transform paths through the real parameters, exactly once.
        shared_transform_pending = [
            (output, leaf.grad)
            for output, leaf in zip(shared_transform_outputs, shared_transform_leaves)
            if output.requires_grad and leaf.grad is not None
        ]
        if shared_transform_pending:
            torch.autograd.backward(
                [output for output, _ in shared_transform_pending],
                [grad for _, grad in shared_transform_pending],
            )
        self.step_timer.stop('bwd')
        self.step_timer.start('comm')
        allreduce_grads_(self.dist_grad_params, self.dist.world_size)
        self.step_timer.stop('comm')

        step_had_nonfinite = torch.zeros((), dtype=torch.bool, device=self.nonfinite_grad_steps.device)
        for name, p in self.dist_grad_named:
            if p.grad is not None:
                # aminmax propagates NaN and surfaces +/-inf through two scalar
                # reductions, avoiding the gradient-sized boolean temporaries
                # that (~torch.isfinite(grad)).any() allocates per parameter.
                grad_min, grad_max = torch.aminmax(p.grad)
                param_nonfinite = ~(torch.isfinite(grad_min) & torch.isfinite(grad_max))
                step_had_nonfinite |= param_nonfinite
                self.nonfinite_grad_by_param[name] += param_nonfinite.to(self.nonfinite_grad_steps.dtype)
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        self.nonfinite_grad_steps += step_had_nonfinite.to(self.nonfinite_grad_steps.dtype)

        if self.influence_state is not None and self.influence_state.active:
            # After the all-reduce and the accumulated-field-grad handoff, so
            # every rank masks identical averaged gradients on both flow paths.
            self.influence_state.apply_grad_masks_(self.spiral_and_transform)

        self.step_timer.start('opt')
        self.optimiser.step()
        self.step_timer.stop('opt')
        if self.influence_state is not None and self.influence_state.active:
            self.influence_state.apply_masked_gap_decay_(self.spiral_and_transform, self.optimiser)
        self.optimiser.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.step_timer.tick()
        self.step_timer.maybe_report(iteration)
        if self.profiler is not None:
            self.profiler.step()

        return loss, losses, log_metrics, shell_metrics

    def resolve_output_path(self):
        """Derive and create this run's output directory.

        Requires load_host_inputs() (the directory name records the verified
        patch count). Both drivers call this between load_host_inputs() and
        build_device_state().
        """
        out_base_dir = self.out_base_dir
        self.out_path = f'{out_base_dir}/{datetime.date.today()}_{self.scroll_name}_slice-{self.z_begin}-{self.z_end}_{self.num_verified_patches}-patch'
        if self.run_name is not None and not self.run_name.startswith('dummy-'):
            self.out_path += '_' + self.run_name
        if self.run_tag:
            self.out_path += f'_{self.run_tag}'
        os.makedirs(self.out_path, exist_ok=True)
        return self.out_path

    def release_setup_only_tracks(self):
        """Drop the per-track input arrays a resident session no longer needs.

        In the usual zero-exclusion case preview bounds reuse the prepared
        flat tensor, so the original list of per-track arrays is no longer
        needed after setup. Interactive-session memory policy; the headless
        driver keeps self.tracks for the final outputs.
        """
        if self.preview_extent_tracks is not self.tracks and self.track_reload_source is None:
            self.tracks = None

    def log_step_metrics(self, iteration, loss, losses, log_metrics, shell_metrics):
        """Print (and, when a wandb run exists, wandb-log) the per-loss-family
        values every 200 iterations.

        Shared by both drivers; wandb is an optional logging sink, so the
        wandb.log call only happens when the process has an active run (the
        CLI's wandb.init). Interactive sessions run without one, so only the
        print is observable there.
        """
        if iteration % 200 == 0:
            # Only sync to CPU and log when we actually print, avoiding a per-iter
            # GPU->CPU sync that would otherwise stall CPU/GPU overlap.
            if self.dist.is_main_process:
                print(f'step {iteration}: loss = {loss.item():.1f}, ' + ', '.join(f'{name} = {value.item():.1f}' for name, value in losses.items()))
                n_sanitised = int(self.nonfinite_grad_steps.item())
                if n_sanitised > 0:
                    per_param = sorted(
                        ((name, int(count.item())) for name, count in self.nonfinite_grad_by_param.items() if count.item() > 0),
                        key=lambda name_count: -name_count[1],
                    )
                    by_param = ', '.join(f'{name}: {count}' for name, count in per_param)
                    print(f'  ({n_sanitised} non-finite-gradient steps sanitised so far; by param: {by_param})')
                if wandb.run is not None:
                    wandb.log({
                        'total_loss': loss.item(),
                        'nonfinite_grad_steps': self.nonfinite_grad_steps.item(),
                        **{f'nonfinite_grad_steps/{name}': count.item() for name, count in self.nonfinite_grad_by_param.items()},
                        **{name + '_loss': value for name, value in losses.items()},
                        **shell_metrics,
                        **log_metrics,
                    })

    def run(self):
        """Drive one complete headless fit to the configured horizon.

        Interactive sessions are not driven here: spiral_runtime owns the
        context, its ready signal, and the resident optimizer loop.
        """
        progress = progress_or_null(self.progress)
        has_progress = self.progress is not None

        self.load_host_inputs()
        self.resolve_output_path()
        self.build_device_state()

        # ==========================================================================
        # Training loop
        # ==========================================================================

        progress.begin(
            'optimizing', 'Optimizing',
            step=0, total_steps=max(0, self.num_training_steps - self.start_iteration),
            unit='iterations')
        for iteration in tqdm(
                range(self.start_iteration, self.num_training_steps),
                disable=not self.dist.is_main_process or has_progress):
            loss, losses, log_metrics, shell_metrics = self.step(iteration)
            progress.update(iteration - self.start_iteration + 1)
            self.log_step_metrics(iteration, loss, losses, log_metrics, shell_metrics)

        # ==========================================================================
        # Final outputs
        # ==========================================================================

        suffix = 'fitted'
        if self.dist.is_main_process:
            progress.begin(
                'saving_checkpoint', 'Saving final checkpoint',
                detail=f'checkpoint_{suffix}.ckpt')
            self._save_model(suffix, self.num_training_steps)
            if self.config.get('output_save_png_visualizations', False):
                progress.begin(
                    'finalizing', 'Preparing final visualizations')
                (
                    zs_for_visualisation,
                    slice_yx,
                    scroll_slices_for_visualisation,
                    prediction_slices_for_visualisation,
                    quad_label_map,
                ) = self._prepare_png_visualization_inputs()
            else:
                zs_for_visualisation = None
                slice_yx = None
                scroll_slices_for_visualisation = None
                prediction_slices_for_visualisation = None
                quad_label_map = None
            progress.begin(
                'finalizing', 'Computing satisfaction metrics and outputs')
            save_overlay_and_print_satisfaction(
                suffix,
                spiral_and_transform=self.spiral_and_transform,
                slice_to_spiral_transform=self.slice_to_spiral_transform,
                dr_per_winding=self.dr_per_winding,
                patches_list=self.verified_patches_list,
                patches_dict=self.verified_patches,
                unattached_pcl_strips=self.unattached_pcl_strips,
                tracks=self.tracks,
                unverified_patches_list=self.unverified_patches_list,
                unverified_patches_dict=self.unverified_patches,
                out_path=self.out_path,
                cfg=self.config,
                z_begin=self.z_begin,
                z_end=self.z_end,
                flow_field_radius=self.flow_field_radius,
                flow_min_corner_spiral_zyx=self.flow_min_corner_spiral_zyx,
                flow_max_corner_spiral_zyx=self.flow_max_corner_spiral_zyx,
                zs_for_visualisation=zs_for_visualisation,
                slice_yx=slice_yx,
                scroll_slices_for_visualisation=scroll_slices_for_visualisation,
                prediction_slices_for_visualisation=prediction_slices_for_visualisation,
                quad_label_map=quad_label_map,
                z_to_umbilicus_yx=self.umbilicus,
                render_volume_scale=self.render_volume_scale,
                voxel_size_um=self.voxel_size_um,
                get_or_build_unattached_pcl_flat=get_or_build_unattached_pcl_flat,
                run_tag=self.run_tag,
                save_png_visualizations=self.config.get('output_save_png_visualizations', False),
                progress=progress,
            )
            progress.finish()
            progress.clear()

    def close(self):
        """Release the sparse volume stores this context owns.

        close() owns resource release and runs on the fitter thread for
        this rank (the runtime calls it when its session ends; the CLI
        process simply exits).
        """
        store, self._lasagna_store = self._lasagna_store, None
        if store is not None:
            store.close()
        while self._scalar_stores:
            self._scalar_stores.pop().close()
        self.winding_inference = None


def main(config, *, scroll, paths, progress=None, resume_path=None,
         resume_step=0, out_base_dir=None, run_tag=None, run_name=None,
         cache_dir=None, storage_backend='sparse_cuda',
         render_volume_scale=16, dist_context=None):
    """Run one headless fit over a fresh context (library entry point).

    config is the resolved FitConfig; scroll/paths are the ScrollSpec and
    resolved SpiralInputPaths; the remaining keywords mirror the FitContext
    constructor's explicit fit controls.
    """
    return FitContext(
        config,
        scroll=scroll,
        paths=paths,
        progress=progress,
        resume_path=resume_path,
        resume_step=resume_step,
        out_base_dir=out_base_dir,
        run_tag=run_tag,
        run_name=run_name,
        cache_dir=cache_dir,
        storage_backend=storage_backend,
        render_volume_scale=render_volume_scale,
        dist_context=dist_context,
    ).run()


if __name__ == '__main__':
    import argparse

    from fit_session import (conventional_input_paths, default_user_cache_dir,
                             load_scroll_spec)

    parser = argparse.ArgumentParser(
        description='Headless Spiral fit over one dataset root.')
    parser.add_argument(
        '--dataset', required=True,
        help='Dataset root holding the conventional Spiral layout and the '
             'spiral-scroll.json scroll specification')
    parser.add_argument(
        '--scroll-spec', default=None,
        help='Explicit scroll specification file '
             '(default: <dataset>/spiral-scroll.json)')
    parser.add_argument(
        '--cache', default=None,
        help='Directory for derived host caches, shared with the interactive '
             'service (default: $FIT_SPIRAL_CACHE_DIR if set, else '
             '$XDG_CACHE_HOME/vc3d/spiral, i.e. ~/.cache/vc3d/spiral)')
    cli_args = parser.parse_args()

    scroll_spec = load_scroll_spec(cli_args.dataset, cli_args.scroll_spec)
    input_paths = conventional_input_paths(cli_args.dataset, scroll_spec)

    # The CLI is a torchrun rendezvous boundary: this is where RANK /
    # WORLD_SIZE / LOCAL_RANK are read, once, into an explicit context that is
    # passed down to everything below.
    dist_context = DistributedContext.from_env()
    cli_progress = (ProgressReporter(stream=sys.stderr)
                    if dist_context.is_main_process else None)
    maybe_init_distributed(dist_context)
    try:
        config = Config().as_dict()
        config.update(get_env_config_overrides())
        z_range_scaled_count_keys = (
            'sample_count_patches_per_step',
            'sample_count_patches_per_step_for_dt',
            'sample_count_unverified_patches_per_step',
            'sample_count_unverified_patches_per_step_for_dt',
            'sample_count_relative_winding_pcls',
            'sample_count_absolute_winding_pcls',
            'sample_count_unattached_pcls_per_step',
            'sample_count_tracks_per_step',
            'sample_count_dense_normal_points',
            'sample_count_dense_spacing_pairs',
            'sample_count_dense_spacing_density_extra_pairs',
            'sample_count_winding_model_relative_pairs',
            'sample_count_winding_model_density_pairs',
            'sample_count_dense_attachment_points',
            'sample_count_regularisation_points',
            'sample_count_shell_samples',
        )
        z_range_scale, z_range_num_slices, split_divisor = scale_and_split_counts(
            config, config['z_begin'], config['z_end'],
            z_range_scaled_count_keys, world_size=dist_context.world_size)
        if dist_context.is_main_process:
            print(
                f'scaled per-step counts by {z_range_scale:.3f} for the {z_range_num_slices}-slice '
                f'z-range [{config["z_begin"]}, {config["z_end"]}) '
                f'(reference {REFERENCE_Z_RANGE_NUM_SLICES} slices):\n  '
                + '\n  '.join(f'{k}={config[k]}' for k in z_range_scaled_count_keys)
            )
            if dist_context.is_distributed:
                policy = f'split by {split_divisor}' if split_divisor > 1 else 'scale-up (full counts per rank)'
                print(f'distributed: world_size={dist_context.world_size}, per-step counts {policy}')

        # wandb is an optional logging sink only: the run records the config
        # and receives log_step_metrics payloads, but the fit reads its
        # configuration exclusively from the explicit FitConfig below.
        wandb_mode = os.environ.get('WANDB_MODE', 'disabled')
        if not dist_context.is_main_process:
            wandb_mode = 'disabled'
        wandb.init(project='scrolls', config=config, mode=wandb_mode)
        # The CLI boundary is where the FIT_SPIRAL_* fit controls are parsed;
        # FitContext itself no longer reads them.
        main(
            FitConfig(config),
            scroll=scroll_spec,
            paths=input_paths,
            progress=cli_progress,
            resume_path=os.environ.get('FIT_SPIRAL_RESUME_PATH'),
            resume_step=int(os.environ.get('FIT_SPIRAL_RESUME_STEP', '0')),
            out_base_dir=os.environ.get('FIT_SPIRAL_OUT_DIR'),
            run_tag=os.environ.get('FIT_SPIRAL_RUN_TAG'),
            run_name=wandb.run.name if wandb.run is not None else None,
            cache_dir=(cli_args.cache
                       or os.environ.get('FIT_SPIRAL_CACHE_DIR')
                       or default_user_cache_dir()),
            render_volume_scale=int(
                os.environ.get('FIT_SPIRAL_RENDER_VOLUME_SCALE', '16')),
            dist_context=dist_context,
        )
    finally:
        if cli_progress is not None:
            cli_progress.close()
        maybe_destroy_distributed(dist_context)
