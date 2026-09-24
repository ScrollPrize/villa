import os
import sys

# Expandable segments stop the CUDA caching allocator from ratcheting its
# reserved pool toward the VRAM ceiling under the variable-size per-step loss
# graphs (measured ~12 GB lower steady-state envelope on the s1 fit). Must be
# set before the allocator initialises; an explicit PYTORCH_CUDA_ALLOC_CONF in
# the environment always wins.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import copy
import concurrent.futures
import gc
import hashlib
import multiprocessing
import json
import glob
import math
import re
try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None
from collections.abc import Mapping
import zarr
import torch
import wandb
import datetime
import time
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
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
import pins as pins_module
from config import (CHECKPOINT_MODEL_SHAPE_KEYS, Config, FitConfig,
                    SHELL_ATLAS_KEYS)
from checkpoint_migrations import (expand_gap_checkpoint_capacity,
                                   tolerate_checkpoint_config)
from fit_session import (AUTOSAVE_INTERVAL_ITERATIONS, EDITABLE_PCL_ROLE_VALUES,
                         RUN_MUTABLE_PCL_ROLES,
                         fit_input, input_source_enabled, pcl_input_enabled,
                         pcl_role_toggle_key,
                         shell_losses_enabled, winding_inference_enabled)
from lazy_moment_adamw import LazyMomentAdamW, robust_clip_


def _startup_resource_suffix(started_at=None):
    """Small cross-platform startup timing/high-water diagnostic."""
    fields = []
    if started_at is not None:
        fields.append(f'{time.perf_counter() - started_at:.1f}s')
    if resource is not None:
        high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        bytes_high_water = high_water if sys.platform == 'darwin' else high_water * 1024
        fields.append(f'peak RSS {bytes_high_water / (1 << 30):.2f} GiB')
    return ', '.join(fields)


from lasagna_data import ensure_fit_sparse_stores, prepare_lasagna_volume
from checkpoint_io import load_checkpoint_cpu, model_state_sha256
from fiber_direction_samples import load_fiber_direction_samples
from spiral_sampling import load_spiral_sampling
from tifxyz import load_tifxyz, patch_from_payload
from geom_utils import bilinear_atlas_lookup, interp1d
from point_collection import (
    PatchLinkOptions,
    SIDE_BEHIND,
    SIDE_FRONT,
    link_points_to_patches,
    load_point_collection,
    normalise_pcl_winding_annotations,
    umbilicus_inward_direction,
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
    RADIAL_OFFSET_STRETCH_EPSILON,
    get_radial_covector_in_scroll_space,
    get_radial_normal_stretch,
    get_spiral_points,
    get_theta,
    get_winding_xy,
)
from losses import (
    get_pin_strain_loss,
    get_pair_agreement_loss,
    draw_abs_winding_rows,
    draw_rel_winding_rows,
    MissingPclSamplingWeightError,
    build_pcl_sampling_strata,
    pcl_sampling_group_weight,
    get_fiber_direction_loss,
    get_min_spacing_loss,
    iter_lasagna_losses,
    get_patch_abs_winding_loss,
    get_patch_and_umbilicus_losses,
    get_patch_rel_winding_loss,
    get_shell_outer_loss,
    get_symmetric_dirichlet_loss,
    get_unattached_pcl_strip_losses,
)
from loss_maps import (LossMapRecorder, attach_loss_maps_to_manifest,
                       capture_loss_maps)
from spiral_helpers import (
    REFERENCE_Z_RANGE_NUM_SLICES,
    erode_patch_valid_region,
    load_patch_payload_chunk,
    load_patches,
    load_fiber_point_collection,
    load_fiber_point_collections,
    classify_fiber_hv,
    fiber_collection_hv_tag,
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
    evaluate_patch_satisfaction_packed,
    get_patch_satisfied_areas as _get_patch_satisfied_areas,
    get_unattached_pcl_satisfied_counts as _get_unattached_pcl_satisfied_counts,
    metrics_config,
    save_overlay_and_print_satisfaction,
)
from visualization import overlay_patches_on_slices
from transforms import SpiralAndTransform, pinned_gap_stage
from theta_crossing_map import ThetaCrossingMap
from winding_supervision import (
    get_winding_inference_losses,
    load_winding_inference_store,
)
from spiral_progress import ProgressReporter, progress_or_null


configure_torch_threads_from_env()


def largest_patch_quad_component(mask):
    """Return only the largest 8-connected True component of a quad mask."""
    mask = np.asarray(mask, dtype=bool)
    component_labels, num_components = scipy.ndimage.label(
        mask, structure=np.ones((3, 3), dtype=bool))
    if num_components <= 1:
        return mask.copy()
    component_sizes = np.bincount(component_labels.reshape(-1))
    component_sizes[0] = 0
    return component_labels == int(component_sizes.argmax())


_HEADLESS_AUTOSAVE_INTERVAL = AUTOSAVE_INTERVAL_ITERATIONS
# Steps between re-checks of the conflict-based pin demotion against the
# current free map (patches that no longer conflict are pinned again).
PIN_DEMOTE_RECHECK_INTERVAL = 1000


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

    def to(self, device):
        # The table is built once on the host (the track filter reads it
        # there) and shared with the losses on the training device: a shallow
        # copy whose tensors live on ``device``.
        moved = copy.copy(self)
        moved.device = device
        moved.lookup_table = self.lookup_table.to(device=device)
        moved.umbilicus_zyx = self.umbilicus_zyx.to(device=device)
        return moved

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


def _make_patch_sampling_atlas(masks):
    """Construct the required sampler, rejecting missing or stale bindings."""
    spiral_sampling = load_spiral_sampling()
    atlas_type = (
        getattr(spiral_sampling, 'PatchSamplingAtlas', None)
        if spiral_sampling is not None else None)
    if atlas_type is None:
        raise RuntimeError(
            'Patch sampling requires vc_spiral.spiral_sampling.PatchSamplingAtlas; '
            'rebuild and install the Spiral native extensions')
    atlas = atlas_type(masks)
    required = (
        'sample_patch_points', 'valid_counts', 'total_valid_cells',
        'node_ijs', 'cell_node_ordinals', 'tree_chunk',
        'neighbor_chunk', 'memory_stats',
    )
    missing = [name for name in required if not hasattr(atlas, name)]
    if missing:
        raise RuntimeError(
            'The installed vc_spiral.spiral_sampling binding is out of date: '
            f'PatchSamplingAtlas is missing {", ".join(missing)}; rebuild and '
            'install the Spiral native extensions')
    return atlas


class PatchAtlas:
    """Patch (H, W, 3) zyx grids packed for batched lookup.

    Initial geometry is moved as one allocation by :meth:`materialize` during
    device setup. Later interactive appends use independent resident chunks so
    the original allocation never has to be copied. Sampling masks remain in
    the native/CPU sampling atlas.
    """

    def __init__(self, patches_by_id, device='cuda'):
        self.device = torch.device(device)
        offsets = [0]
        widths = []
        heights = []
        for p in patches_by_id.values():
            z = p.zyxs  # (H, W, 3) on CPU
            H, W = z.shape[:2]
            offsets.append(offsets[-1] + H * W)
            widths.append(W)
            heights.append(H)
        # Geometry remains in each Patch until materialize().  Concatenating
        # here used to retain a second full host copy for the entire session.
        self.zyxs_flat = None
        self._geometry_chunks = []
        self._packed_direct = True
        self._num_vertices = offsets[-1]
        self.offsets = torch.tensor(offsets, dtype=torch.int64)  # (N+1,)
        self.widths = torch.tensor(widths, dtype=torch.int64)  # (N,)
        self.heights = torch.tensor(heights, dtype=torch.int64)  # (N,)
        self.id_to_idx = {pid: i for i, pid in enumerate(patches_by_id.keys())}
        self._patches = list(patches_by_id.values())
        self._theta_node_start = None
        self._theta_node_ranges = []
        self._satisfaction_atlases = {}
        masks = [
            np.ascontiguousarray(p._sampling_valid_quad_mask_np, dtype=bool)
            for p in patches_by_id.values()
        ]
        self.sampling_atlas = (
            _make_patch_sampling_atlas(masks) if masks else None)
        self._quad_counts = (
            np.asarray(self.sampling_atlas.valid_counts(), dtype=np.int64)
            if self.sampling_atlas is not None
            else np.empty(0, dtype=np.int64))

    def memory_mb(self):
        return self._num_vertices * 3 * 4 / 1e6

    def topology_memory_stats(self):
        """Return printable native-topology stats, including for no patches."""
        if self.sampling_atlas is None:
            return {'num_valid_cells': 0, 'persistent_bytes': 0}
        return dict(self.sampling_atlas.memory_stats())

    @staticmethod
    def _materialize_geometry(patches, num_vertices, device):
        """Copy patch grids into one new resident chunk.

        A chunk is allocated at its final size and filled in place, so adding
        patches to an already-materialized atlas never needs a replacement
        allocation for the existing geometry.
        """
        resident = torch.empty(
            (num_vertices, 3), dtype=torch.float32, device=device)
        if device.type == 'cpu':
            offset = 0
            for patch in patches:
                piece = patch.zyxs.reshape(-1, 3).to(dtype=torch.float32)
                resident[offset:offset + len(piece)].copy_(piece)
                offset += len(piece)
            return resident

        # Amortise transfers without ever concatenating the whole atlas on
        # the host.  256 MiB bounds the temporary independently of dataset
        # size while preserving the exact float32 bytes and patch order.
        staging_limit = 256 * (1 << 20) // (3 * 4)
        pieces = []
        staged = 0
        offset = 0

        def flush():
            nonlocal pieces, staged, offset
            if not pieces:
                return
            staging = torch.cat(pieces, dim=0)
            resident[offset:offset + staged].copy_(
                staging, non_blocking=True)
            offset += staged
            pieces = []
            staged = 0

        for patch in patches:
            piece = patch.zyxs.reshape(-1, 3).to(dtype=torch.float32)
            if pieces and staged + len(piece) > staging_limit:
                flush()
            # A single unusually large patch is copied directly so the
            # advertised staging bound is not exceeded.
            if len(piece) > staging_limit:
                resident[offset:offset + len(piece)].copy_(
                    piece, non_blocking=True)
                offset += len(piece)
            else:
                pieces.append(piece)
                staged += len(piece)
        flush()
        return resident

    def materialize(self, device=None):
        """Build one resident geometry allocation with bounded host staging."""
        if device is not None:
            self.device = torch.device(device)
        if (self.zyxs_flat is not None
                and self.zyxs_flat.device == self.device):
            return self
        resident = self._materialize_geometry(
            self._patches, self._num_vertices, self.device)
        self.zyxs_flat = resident
        self.offsets = self.offsets.to(self.device)
        self.widths = self.widths.to(self.device)
        self.heights = self.heights.to(self.device)
        self._geometry_chunks = [{
            'zyxs_flat': resident,
            'offsets': self.offsets,
            'widths': self.widths,
            'heights': self.heights,
            'patch_start': 0,
            'patch_end': len(self._patches),
        }]
        return self

    def rebuild_sampling_atlas(self):
        """Rebuild CPU/native lookup metadata after sampling masks change."""
        masks = [
            np.ascontiguousarray(
                patch._sampling_valid_quad_mask_np, dtype=bool)
            for patch in self._patches
        ]
        self.sampling_atlas = (
            _make_patch_sampling_atlas(masks) if masks else None)
        self._quad_counts = (
            np.asarray(self.sampling_atlas.valid_counts(), dtype=np.int64)
            if self.sampling_atlas is not None
            else np.empty(0, dtype=np.int64))
        self._satisfaction_atlases.clear()

    def satisfaction_atlas(self, z_begin, z_end):
        """Return exact packed ROI quad metadata, cached by z interval."""
        key = (float(z_begin), float(z_end))
        cached = self._satisfaction_atlases.get(key)
        if cached is not None:
            return cached
        native = load_spiral_sampling()
        atlas_type = (getattr(native, 'PatchSatisfactionAtlas', None)
                      if native is not None else None)
        if atlas_type is None:
            raise RuntimeError(
                'Packed satisfaction requires '
                'vc_spiral.spiral_sampling.PatchSatisfactionAtlas; rebuild the '
                'Spiral native extensions')
        masks = [np.ascontiguousarray(
            patch.valid_quad_mask.cpu().numpy(), dtype=bool)
            for patch in self._patches]
        vertex_zs = [np.ascontiguousarray(
            patch.zyxs[..., 0].cpu().numpy(), dtype=np.float32)
            for patch in self._patches]
        cached = atlas_type(masks, vertex_zs, key[0], key[1])
        self._satisfaction_atlases[key] = cached
        return cached

    def vertex_zyxs(self, vertex_ids):
        """Gather exact atlas vertices by their stable packed vertex IDs."""
        if self.zyxs_flat is None:
            self.materialize(self.device)
        ids = torch.as_tensor(
            vertex_ids, dtype=torch.int64, device=self.zyxs_flat.device)
        if len(self._geometry_chunks) == 1 and self._packed_direct:
            return self.zyxs_flat[ids]
        result = torch.empty((*ids.shape, 3), dtype=torch.float32,
                             device=self.zyxs_flat.device)
        patch_indices = torch.bucketize(ids, self.offsets[1:], right=True)
        for chunk in self._geometry_chunks:
            selected = ((patch_indices >= chunk['patch_start'])
                        & (patch_indices < chunk['patch_end']))
            if selected.any():
                indices = patch_indices[selected]
                local = indices - chunk['patch_start']
                vertices = ids[selected] - self.offsets[indices] + chunk['offsets'][local]
                result[selected] = chunk['zyxs_flat'][vertices]
        return result

    def replaced(self, patches_by_id):
        """Prepare a new mapping, retaining unchanged device geometry storage.

        Sampling/topology belong to the new atlas. Geometry tensors are shared
        by reference; replaced grids get independent allocations. No old atlas
        or sampling object is mutated, including on preparation failure.
        """
        candidate = PatchAtlas(patches_by_id, self.device)
        if self.zyxs_flat is None:
            return candidate
        candidate._packed_direct = False
        candidate.offsets = candidate.offsets.to(self.device)
        candidate.widths = candidate.widths.to(self.device)
        candidate.heights = candidate.heights.to(self.device)
        old_storage = {}
        for chunk in self._geometry_chunks:
            for old_idx in range(chunk['patch_start'], chunk['patch_end']):
                old_storage[old_idx] = (chunk, old_idx - chunk['patch_start'])
        runs = []
        for new_idx, (pid, patch) in enumerate(patches_by_id.items()):
            old_idx = self.id_to_idx.get(pid)
            shared = (old_idx is not None
                      and patch.zyxs is self._patches[old_idx].zyxs)
            chunk, local = old_storage[old_idx] if shared else (None, None)
            if not runs or runs[-1][0] is not chunk:
                runs.append((chunk, []))
            runs[-1][1].append((new_idx, local, patch))
        for source, entries in runs:
            start = entries[0][0]
            end = entries[-1][0] + 1
            if source is None:
                offsets = candidate.offsets[start:end + 1] - candidate.offsets[start]
                geometry = self._materialize_geometry(
                    [entry[2] for entry in entries], int(offsets[-1]), self.device)
            else:
                indices = torch.tensor([entry[1] for entry in entries],
                                       device=self.device, dtype=torch.int64)
                offsets = source['offsets'][indices]
                geometry = source['zyxs_flat']
            candidate._geometry_chunks.append({
                'zyxs_flat': geometry, 'offsets': offsets,
                'widths': candidate.widths[start:end],
                'heights': candidate.heights[start:end],
                'patch_start': start, 'patch_end': end,
            })
        candidate.zyxs_flat = (candidate._geometry_chunks[0]['zyxs_flat']
                               if candidate._geometry_chunks
                               else torch.empty((0, 3), device=self.device))
        return candidate

    def lookup(self, patch_idx_per_sample, ijs):
        # Caller must ensure floor(ij) lies on a valid quad. CPU lookup remains
        # supported before materialisation and for CPU-only tests.
        if self.zyxs_flat is None:
            patch_indices = torch.as_tensor(
                patch_idx_per_sample, dtype=torch.int64, device='cpu')
            ijs_cpu = torch.as_tensor(ijs, dtype=torch.float32, device='cpu')
            shape = ijs_cpu.shape[:-1]
            flat_indices = patch_indices.expand(shape).reshape(-1)
            flat_ijs = ijs_cpu.reshape(-1, 2)
            output = torch.empty((len(flat_ijs), 3), dtype=torch.float32)
            for patch_idx in torch.unique(flat_indices).tolist():
                selected = flat_indices == patch_idx
                grid = self._patches[patch_idx].zyxs.to(dtype=torch.float32)
                selected_ijs = flat_ijs[selected]
                i0 = selected_ijs[:, 0].floor().to(torch.int64).clamp(
                    0, grid.shape[0] - 2)
                j0 = selected_ijs[:, 1].floor().to(torch.int64).clamp(
                    0, grid.shape[1] - 2)
                di = (selected_ijs[:, 0] - i0).unsqueeze(-1).clamp(0., 1.)
                dj = (selected_ijs[:, 1] - j0).unsqueeze(-1).clamp(0., 1.)
                tl = grid[i0, j0]
                tr = grid[i0, j0 + 1]
                bl = grid[i0 + 1, j0]
                br = grid[i0 + 1, j0 + 1]
                top = tl + (tr - tl) * dj
                bottom = bl + (br - bl) * dj
                output[selected] = top + (bottom - top) * di
            return output.reshape(*shape, 3).to(self.device)
        patch_idx_per_sample = patch_idx_per_sample.to(
            device=self.zyxs_flat.device, dtype=torch.int64, non_blocking=True)
        ijs = ijs.to(device=self.zyxs_flat.device, non_blocking=True)
        if len(self._geometry_chunks) == 1:
            chunk = self._geometry_chunks[0]
            return bilinear_atlas_lookup(
                chunk['zyxs_flat'], chunk['offsets'], chunk['widths'],
                patch_idx_per_sample, ijs, heights=chunk['heights'])

        # Appended geometry stays in independent resident chunks. Dispatch
        # samples to their owning chunk instead of joining all geometry into a
        # replacement atlas (which would briefly require roughly 2x VRAM).
        sample_shape = ijs.shape[:-1]
        patch_indices = patch_idx_per_sample.expand(sample_shape)
        zyxs = torch.empty((*sample_shape, 3), dtype=torch.float32,
                           device=self.zyxs_flat.device)
        for chunk in self._geometry_chunks:
            start = chunk['patch_start']
            selected = ((patch_indices >= start)
                        & (patch_indices < chunk['patch_end']))
            local_indices = patch_indices[selected] - start
            zyxs[selected] = bilinear_atlas_lookup(
                chunk['zyxs_flat'], chunk['offsets'], chunk['widths'],
                local_indices, ijs[selected], heights=chunk['heights'])
        return zyxs

    def register_theta_topology(self, crossing_map):
        """Register compact native patch trees and streamed neighbour edges."""
        if self.sampling_atlas is None:
            self._theta_node_start = crossing_map.register_nodes(
                0, lambda lo, hi: torch.empty((0, 3), dtype=torch.float32))
            self._theta_node_ranges = []
            return self._theta_node_start
        num_quads = int(self.sampling_atlas.total_valid_cells())

        def get_centres(lo, hi):
            resolved = self.sampling_atlas.node_ijs(
                np.arange(lo, hi, dtype=np.int64))
            idx = torch.from_numpy(np.asarray(
                resolved['patch_indices'], dtype=np.int64))
            ijs = torch.from_numpy(np.asarray(
                resolved['ijs'], dtype=np.float32))
            return self.lookup(idx, ijs + 0.5)

        start = crossing_map.register_nodes(num_quads, get_centres)
        self._theta_node_start = start
        ends = np.cumsum(self._quad_counts, dtype=np.int64) + start
        begins = np.concatenate([
            np.asarray([start], dtype=np.int64), ends[:-1]])
        self._theta_node_ranges = list(zip(begins.tolist(), ends.tolist()))

        def tree_chunk(lo, hi):
            chunk = self.sampling_atlas.tree_chunk(int(lo), int(hi))
            return (np.asarray(chunk['node_ordinals'], dtype=np.int64),
                    np.asarray(chunk['parent_ordinals'], dtype=np.int64),
                    np.asarray(chunk['exit_positions'], dtype=np.int64))

        def neighbor_chunk(cursor, slot_count):
            chunk = self.sampling_atlas.neighbor_chunk(
                int(cursor), int(slot_count))
            return (int(chunk['next_cursor']),
                    np.asarray(chunk['node_pairs'], dtype=np.int64))

        crossing_map.register_potential_source(
            start, num_quads, tree_chunk, neighbor_chunk)
        return start

    def patch_ids_for_theta_nodes(self, node_ids):
        """Return patches owning any of the supplied global theta node IDs."""
        nodes = torch.as_tensor(node_ids, dtype=torch.int64).cpu().numpy()
        if nodes.size == 0:
            return []
        nodes = np.unique(nodes)
        patch_ids = list(self.id_to_idx)
        found = []
        for patch_id, (lo, hi) in zip(patch_ids, self._theta_node_ranges):
            position = int(np.searchsorted(nodes, lo))
            if position < nodes.size and nodes[position] < hi:
                found.append(patch_id)
        return found

    def theta_node_ids(self, patch_indices, ijs):
        """Resolve fractional samples/path cells to registered quad-centre ids."""
        if self._theta_node_start is None:
            raise RuntimeError('patch theta topology has not been registered')
        patch_indices = np.broadcast_to(
            np.asarray(patch_indices, dtype=np.int64), np.asarray(ijs).shape[:-1])
        shape = patch_indices.shape
        cells = np.floor(np.asarray(ijs)).astype(np.int64).reshape(-1, 2)
        ordinals = self.sampling_atlas.cell_node_ordinals(
            np.ascontiguousarray(patch_indices.reshape(-1), dtype=np.int64),
            np.ascontiguousarray(cells, dtype=np.int64))
        return (np.asarray(ordinals, dtype=np.int64).reshape(shape)
                + self._theta_node_start)

    def theta_node_ids_from_ordinals(self, node_ordinals):
        """Resolve native sampler ordinals without a cell-to-node lookup."""
        if self._theta_node_start is None:
            raise RuntimeError('patch theta topology has not been registered')
        return (np.asarray(node_ordinals, dtype=np.int64)
                + self._theta_node_start)


class _UnattachedPclStripList(list):
    """List of unattached-pcl strip dicts, with a slot for an attached `.flat`
    GPU bundle that batched satisfaction / winding-range computations reuse."""
    pass


def stamp_loaded_pcl_metadata(pcl, path, explicit_role, resident_collection_id):
    """Record provenance and identity on one regular PCL as it is loaded.

    Every regular collection carries its catalog id as
    ``resident_collection_id``: derived training views (cross-patch views,
    unattached strips) are matched back to the catalog entry through it when
    a later live patch relinks resident points, whatever the collection's
    role. Editable roles additionally carry the logical identity a client
    uses to replace or delete the collection in place.
    """
    source_collection_id = pcl['id']
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
    pcl['metadata']['resident_collection_id'] = resident_collection_id
    # Editable collection ids are unique within their role file, rather
    # than globally across all PCL inputs.
    if explicit_role in EDITABLE_PCL_ROLE_VALUES:
        pcl['metadata'].update({
            'logical_input_kind': explicit_role,
            'logical_input_id': str(source_collection_id),
            'logical_input_revision': None,
        })
    return pcl


def catalog_copy_of_pcl(pcl):
    """Copy a regular PCL without derived chain and patch-view state."""
    return copy.deepcopy({
        key: value for key, value in pcl.items()
        if key not in ('chain', 'points_by_patch')
    })


def longest_run_in_z_window(sorted_items, z_begin, z_end, z_margin):
    """Return the longest contiguous id-sorted run inside the z window."""
    best_start = best_end = run_start = 0
    for position, (_, point) in enumerate(sorted_items):
        z = point['zyx'][0]
        if z_begin - z_margin <= z < z_end + z_margin:
            if position + 1 - run_start > best_end - best_start:
                best_start, best_end = run_start, position + 1
        else:
            run_start = position + 1
    return sorted_items[best_start:best_end]


def subsample_rows(points, max_points, generator):
    """Deterministically keep at most ``max_points`` rows of ``points``."""
    if points.shape[0] <= max_points:
        return points
    order = torch.randperm(points.shape[0], generator=generator)[:max_points]
    return points[order.sort().values]


def regular_unattached_strip(pcl_id, pcl, min_point_spacing):
    """Materialize the unattached-loss strip of a regular PCL."""
    sorted_items = sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
    if len(sorted_items) < 2:
        return None
    zyxs = np.stack(
        [point['zyx'] for _, point in sorted_items], axis=0).astype(np.float32)
    windings = np.array(
        [point['winding_annotation'] for _, point in sorted_items],
        dtype=np.float32)
    zyxs, keep = _decimate_ordered_points_min_spacing(
        zyxs, min_point_spacing, return_indices=True,
        force_keep={len(zyxs) - 1})
    metadata = pcl.get('metadata', {})
    return {
        'id': pcl_id,
        'name': pcl.get('name'),
        'source_file': pcl.get('source_file'),
        'zyxs': zyxs,
        'windings': windings[keep],
        'link_points': {},
        'logical_input_kind': metadata.get('logical_input_kind'),
        'logical_input_id': metadata.get('logical_input_id'),
        'logical_input_revision': metadata.get('logical_input_revision'),
    }


def materialize_fiber_fit_inputs(
        fiber_catalog, verified_patches, *, z_begin, z_end, z_margin,
        min_point_spacing, use_links=True, use_pending_links=False,
        vertical_min_z_fraction=0.8, vertical_min_auto_certainty=0.5,
        vertical_radial_offset=0.0):
    """Derive every CPU training view from the resident fiber catalog.

    ``fiber_catalog`` is ordered by logical input id and owns the canonical
    fiber containers. The returned cross-patch PCLs and strips refer to their
    geometry, but trimming and decimation never mutate the catalog's point
    membership. Re-running this function over the same catalog therefore
    produces the same graph and ordered training views.

    Each strip carries ``hv_tag`` ('V', 'H', or None; see
    spiral_helpers.classify_fiber_hv) and ``is_vertical`` for radial-offset updates.
    ``radial_offsets`` holds the
    per-point radial target offset in scroll voxels along the sheet normal,
    positive in the increasing-winding direction of the fitted spiral (the
    scan-space winding gradient, not the line to the umbilicus):
    ``vertical_radial_offset`` on vertical strips (their back-face position
    relative to the fitted sheet, away from the umbilicus), zero everywhere
    else.
    ``radial_offset_bake_scale`` carries each point's accumulated normal
    stretch through the frozen constraint-bake stack (1 before any bake),
    read from the catalog point dicts (see accumulate_radial_offset_bake_scale)
    so the physical offset can be expressed in the resident frame.
    """
    point_collections = {}
    for logical_id, pcl in fiber_catalog.items():
        logical_id = str(logical_id)
        metadata = pcl.setdefault('metadata', {})
        metadata['logical_input_kind'] = 'fiber'
        metadata['logical_input_id'] = logical_id
        metadata['winding_is_absolute'] = False
        metadata.setdefault('input_role', 'fiber')
        pcl.setdefault('file_basename', f'{logical_id}.json')
        pcl['sampling_group'] = 'fibers'
        pcl['chain'] = SequenceChain(pcl)
        for point in pcl['points'].values():
            point['collectionId'] = pcl['id']
            if 'zyx' not in point:
                point['zyx'] = np.asarray(
                    [point['p'][2], point['p'][1], point['p'][0]],
                    dtype=np.float32)
        if pcl['id'] in point_collections:
            raise ValueError(f'duplicate resident fiber collection id {pcl["id"]!r}')
        point_collections[pcl['id']] = pcl

    resolved_links = []
    if use_links:
        resolved_links = resolve_fiber_links(
            point_collections, include_pending=use_pending_links,
            assume_unannotated=True)
    link_components = build_link_components(resolved_links)
    catalog_position = {
        pcl['id']: position
        for position, pcl in enumerate(fiber_catalog.values())
    }
    link_components = [
        (sorted(member_cids, key=catalog_position.__getitem__), member_links)
        for member_cids, member_links in link_components
    ]
    link_components.sort(
        key=lambda component: min(
            catalog_position[cid] for cid in component[0]))
    linked_cids = {
        cid for member_cids, _ in link_components for cid in member_cids
    }

    cross_patch = {}
    strip_sources = []
    for cid, pcl in point_collections.items():
        num_attached = sum(
            1 for point in pcl['points'].values() if 'on_patch' in point)
        num_unattached = len(pcl['points']) - num_attached
        if num_attached >= 2:
            cross_patch[cid] = pcl
        if num_unattached >= 1 or cid in linked_cids:
            strip_sources.append((cid, pcl))

    if link_components:
        cross_patch, _ = merge_linked_point_collections(
            point_collections, link_components, cross_patch)

    # Fiber files carry no winding annotations. Their point dictionaries are
    # shared by the catalog and derived PCLs, so normalize them directly after
    # link resolution; subsequent materializations explicitly resolve catalog
    # fibers as unannotated above.
    for pcl in point_collections.values():
        pcl['has_winding_annotations'] = False
        for point in pcl['points'].values():
            point['winding_annotation'] = 0.0

    for pcl in cross_patch.values():
        points_by_patch = {}
        for _, point in sorted(pcl['points'].items(), key=lambda item: int(item[0])):
            if 'on_patch' not in point:
                continue
            patch_id = point['on_patch']['id']
            if patch_id in verified_patches:
                points_by_patch.setdefault(patch_id, []).append(point)
        pcl['points_by_patch'] = points_by_patch

    link_point_ids_by_coll = {}
    for link in resolved_links:
        link_point_ids_by_coll.setdefault(link['a_coll'], set()).add(
            link['a_point'])
        link_point_ids_by_coll.setdefault(link['b_coll'], set()).add(
            link['b_point'])

    strips = _UnattachedPclStripList()
    sampling_groups = []
    for cid, pcl in strip_sources:
        sorted_items = sorted(pcl['points'].items(), key=lambda item: int(item[0]))
        best_start = best_end = run_start = 0
        for position, (_, point) in enumerate(sorted_items):
            z = point['zyx'][0]
            if z_begin - z_margin <= z < z_end + z_margin:
                if position + 1 - run_start > best_end - best_start:
                    best_start, best_end = run_start, position + 1
            else:
                run_start = position + 1
        kept_items = sorted_items[best_start:best_end]
        if len(kept_items) < 2:
            continue

        link_ids = link_point_ids_by_coll.get(cid, ())
        force_keep = {
            position for position, (point_id, _) in enumerate(kept_items)
            if int(point_id) in link_ids
        }
        zyxs = np.stack(
            [point['zyx'] for _, point in kept_items], axis=0).astype(np.float32)
        windings = np.asarray(
            [point['winding_annotation'] for _, point in kept_items],
            dtype=np.float32)
        bake_scale = np.asarray(
            [point.get('radial_offset_bake_scale', 1.0)
             for _, point in kept_items],
            dtype=np.float32)
        zyxs, keep = _decimate_ordered_points_min_spacing(
            zyxs, min_point_spacing, return_indices=True,
            force_keep=force_keep | {len(zyxs) - 1})
        windings = windings[keep]
        bake_scale = bake_scale[keep]
        link_points = {
            int(kept_items[original_position][0]): strip_position
            for strip_position, original_position in enumerate(keep)
            if original_position in force_keep
        }
        metadata = pcl.get('metadata', {})
        hv_tag = classify_fiber_hv(
            metadata.get('hv_classification'), zyxs,
            min_z_fraction=vertical_min_z_fraction,
            min_auto_certainty=vertical_min_auto_certainty)
        strips.append({
            'id': cid,
            'name': pcl.get('name'),
            'source_file': pcl.get('source_file'),
            'zyxs': zyxs,
            'windings': windings,
            'radial_offsets': np.full(
                len(zyxs),
                float(vertical_radial_offset) if hv_tag == 'V' else 0.0,
                dtype=np.float32),
            'radial_offset_bake_scale': bake_scale,
            'link_points': link_points,
            'hv_tag': hv_tag,
            'is_vertical': hv_tag == 'V',
            'logical_input_kind': 'fiber',
            'logical_input_id': metadata.get('logical_input_id'),
            'logical_input_revision': metadata.get('logical_input_revision'),
        })
        sampling_groups.append(pcl['sampling_group'])

    return (list(cross_patch.values()), strips, sampling_groups,
            resolved_links, link_components)


def _build_strip_flat_bundle(strip_arrays, device):
    # Concatenate per-strip (zyxs, windings, radial_offsets[, bake_scale]) arrays
    # into one flat GPU tensor so the downstream computations can run a single
    # transform call plus segmented reductions instead of per-strip Python loops.
    # `strip_arrays` is a sequence of `(zyxs_np, windings_np, radial_offsets_np)`
    # triples or `(..., radial_offset_bake_scale_np)` quadruples; radial_offsets
    # may be None for zeros and the bake scale None for ones. The bundle's
    # `radial_offsets` is the physical offset (scroll voxels along the sheet
    # normal) times the per-point bake scale, i.e. the offset in the resident
    # frame's voxels; consumers convert it to spiral radius through the live
    # transform's local normal stretch. `has_radial_offsets` lets them skip
    # that work when every offset is zero. Returns None when there are no points.
    triples = [tuple(t) for t in strip_arrays]
    if len(triples) == 0:
        return None
    lengths_np = np.fromiter((len(t[0]) for t in triples), dtype=np.int64, count=len(triples))
    starts_np = np.empty(len(triples) + 1, dtype=np.int64)
    starts_np[0] = 0
    np.cumsum(lengths_np, out=starts_np[1:])
    total = int(starts_np[-1])
    if total == 0:
        return None
    zyxs_flat = np.concatenate([t[0] for t in triples], axis=0).astype(np.float32, copy=False)
    windings_flat = np.concatenate([t[1] for t in triples], axis=0).astype(np.float32, copy=False)
    offsets_flat = np.concatenate([
        t[2] if len(t) > 2 and t[2] is not None else np.zeros(len(t[0]), dtype=np.float32)
        for t in triples], axis=0).astype(np.float32, copy=False)
    if any(len(t) > 3 and t[3] is not None for t in triples):
        scale_flat = np.concatenate([
            t[3] if len(t) > 3 and t[3] is not None else np.ones(len(t[0]), dtype=np.float32)
            for t in triples], axis=0).astype(np.float32, copy=False)
        offsets_flat = offsets_flat * scale_flat
    strip_id_np = np.repeat(np.arange(len(triples), dtype=np.int64), lengths_np)
    return {
        'zyxs': torch.from_numpy(zyxs_flat).to(device=device),
        'windings': torch.from_numpy(windings_flat).to(device=device),
        'radial_offsets': torch.from_numpy(offsets_flat).to(device=device),
        'has_radial_offsets': bool(np.any(offsets_flat != 0)),
        'strip_id': torch.from_numpy(strip_id_np).to(device=device),
        'starts': torch.from_numpy(starts_np).to(device=device),
        'starts_cpu': torch.from_numpy(starts_np),
        'lengths': torch.from_numpy(lengths_np).to(device=device),
        'lengths_cpu': torch.from_numpy(lengths_np),
        'num_strips': len(triples),
        'total': total,
    }


def accumulate_radial_offset_bake_scale(slice_to_spiral_transform, zyxs,
                                        previous=None, *, device=None):
    """Fold one frozen epoch's normal stretch into a per-point bake scale.

    The vertical-fiber radial offset is a physical distance along the sheet
    normal in scroll voxels. A constraint bake rewrites the resident geometry
    through a frozen transform that is not an isometry, so the same distance
    measured in the baked frame is the scroll distance times that transform's
    local stretch along the normal, ``|J^T n|`` at the pre-bake point
    (sample_spiral.get_radial_normal_stretch). ``zyxs`` (N, 3) are the points
    *before* this bake in the frame the transform reads; ``previous`` (N,) is
    the scale accumulated over earlier epochs (None => ones). Returns the
    updated (N,) float32 array. Successive epochs multiply, which is exact
    when each epoch's stretch is evaluated for the normal direction that epoch
    saw (the radial direction of its own output frame, as here) and treats
    later epochs' rotation of that direction as second order.
    """
    stretch = get_radial_normal_stretch(
        slice_to_spiral_transform,
        torch.from_numpy(np.ascontiguousarray(zyxs, dtype=np.float32)),
        device=device).numpy()
    if previous is None:
        return stretch
    return np.asarray(previous, dtype=np.float32) * stretch


def inward_winding_direction(slice_to_spiral_transform, zyxs, *, device=None,
                             chunk_size=65536):
    """Unit vectors along the fitted spiral's decreasing-winding direction.

    At each input-frame point the scan-space gradient of the fitted winding
    is ``J^T n`` (sample_spiral.get_radial_covector_in_scroll_space); its
    negative points toward the neighbouring winding with the lower winding
    number, the "front" side the fiber link side rules refer to. ``zyxs``
    (N, 3) in the frame the transform reads; returns (N, 3) float64, zero
    where the gradient vanishes. Evaluated under no_grad, chunked, with each
    chunk staged to ``device`` when given.
    """
    points = torch.as_tensor(
        np.ascontiguousarray(zyxs, dtype=np.float32)).reshape(-1, 3)
    out = np.zeros((points.shape[0], 3), dtype=np.float64)
    if points.shape[0] == 0:
        return out
    target = device if device is not None else points.device
    with torch.no_grad():
        for start in range(0, points.shape[0], chunk_size):
            chunk = points[start:start + chunk_size].to(
                device=target, dtype=torch.float32)
            covector = get_radial_covector_in_scroll_space(
                slice_to_spiral_transform, chunk,
                epsilon=RADIAL_OFFSET_STRETCH_EPSILON)
            norm = torch.linalg.norm(covector, dim=-1, keepdim=True)
            direction = torch.where(
                norm > 0, -covector / norm.clamp_min(1e-12),
                torch.zeros_like(covector))
            out[start:start + chunk_size] = direction.cpu().numpy()
    return out


def get_or_build_unattached_pcl_flat(pcl_strips, device):
    # Reuse a cached `.flat` bundle on the strip list when available (set up at the
    # top of fit_spiral_3d); otherwise build it now and try to cache for next call.
    flat = getattr(pcl_strips, 'flat', None)
    if flat is None and len(pcl_strips) > 0:
        flat = _build_strip_flat_bundle(
            ((s['zyxs'], s['windings'], s.get('radial_offsets'),
              s.get('radial_offset_bake_scale')) for s in pcl_strips),
            device)
        try:
            pcl_strips.flat = flat
        except AttributeError:
            pass
    return flat


def get_run_dt_resume_iteration(run_start, requested_iterations,
                                last_fraction):
    """Return the first Run iteration eligible for directional DT losses.

    The final ``last_fraction`` of a Run is eligible. Flooring the suppressed
    prefix makes the eligible suffix ``ceil(iterations * last_fraction)``.
    """
    iterations = max(0, int(requested_iterations))
    fraction = min(1.0, max(0.0, float(last_fraction)))
    suppressed = math.floor(iterations * (1.0 - fraction))
    return int(run_start) + suppressed


def get_dt_loss_eligibility(cfg, iteration, run_dt_resume_iteration=None):
    """Return DT eligibility for every directional DT loss family."""
    run_eligible = (run_dt_resume_iteration is None
                    or iteration >= run_dt_resume_iteration)
    patch_start = cfg['loss_start_patch_dt']
    track_start = (patch_start if cfg['loss_start_track_dt'] is None
                   else cfg['loss_start_track_dt'])
    unattached_start = get_unattached_pcl_dt_start(cfg)
    return {
        'verified_patch': run_eligible and iteration > patch_start,
        'track': run_eligible and iteration > track_start,
        'unattached_pcl': run_eligible and iteration > unattached_start,
    }


def get_unattached_pcl_dt_start(cfg):
    """The unattached-PCL DT start: its own key, else the verified-patch start."""
    start = cfg['loss_start_unattached_pcl_dt']
    return cfg['loss_start_patch_dt'] if start is None else start


def unresolved_fiber_link_warning(fiber_catalog, *, use_links, use_pending_links,
                                  max_named=6):
    """Warn only for active catalog branches whose target fiber is absent."""
    if not use_links:
        return None
    resident_basenames = {
        pcl.get('file_basename') for pcl in fiber_catalog.values()
    }
    counted = []
    for input_id, pcl in fiber_catalog.items():
        missing = [
            branch['branch_file'] for branch in pcl.get('branches', ())
            if (use_pending_links or not branch['pending'])
            and branch['branch_file'] not in resident_basenames
        ]
        if missing:
            counted.append((input_id, missing))
    if not counted:
        return None
    named = ', '.join(
        f'{input_id} -> {", ".join(missing)}'
        for input_id, missing in counted[:max_named])
    if len(counted) > max_named:
        named += f', and {len(counted) - max_named} more'
    return (
        f'{sum(len(missing) for _, missing in counted)} active cross-fiber '
        f'branch(es) target absent resident fibers: {named}')


def get_exponential_lr_at_step(
        initial_lr, final_factor, completed_steps, training_horizon):
    """LR on the absolute exponential curve for a completed-step count."""
    horizon = max(1, int(training_horizon))
    completed = max(0, int(completed_steps))
    gamma = float(final_factor) ** (1.0 / horizon)
    return float(initial_lr) * gamma ** completed


def _checkpoint_lacks_pin_group(context, optimiser_state):
    """True when this fit has a pin_targets optimiser group but the saved
    optimiser state predates pins (exactly one group fewer)."""
    if not getattr(context, 'pin_target_params', None) or not isinstance(optimiser_state, Mapping):
        return False
    saved_groups = list(optimiser_state.get('param_groups') or [])
    return len(saved_groups) == len(context.optimiser.param_groups) - 1


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


def get_flow_field_low_res_lr_scale(cfg):
    """Relative optimizer LR for the low-resolution flow lattice."""
    value = cfg.get('model_flow_field_low_res_lr_scale')
    # Only a missing or null setting takes the default: 0 is a valid scale
    # that freezes the coarse lattice (a zero AdamW LR also disables its
    # decoupled weight decay).
    return 1.0 if value is None else float(value)


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
                 progress=None, resume_path=None,
                 out_base_dir=None, run_dir=None, run_tag=None, run_name=None,
                 cache_dir=None, storage_backend='sparse_cuda',
                 render_volume_scale=16, dist_context=None):
        # config is the explicit FitConfig (or any dict-style mapping of
        # fully resolved values) this fit reads; there is no module-global
        # fallback. The optimisation z window is Config's z_begin/z_end.
        #
        # scroll is the frozen ScrollSpec: physical facts of the scanned
        # scroll (name, voxel size, outward sense, umbilicus coordinate
        # scale, Lasagna group and scale). paths is the resolved
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
        #   resume_path - checkpoint to restore (its completed_iterations is
        #     the step the fit resumes at);
        #   out_base_dir - parent directory for resolve_output_path();
        #   run_dir - an exact existing output directory to reuse when a
        #     headless runner resumes an interrupted fit;
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
        # Whole-object patch targets use the theta map's unwrap frame, so both
        # caches intentionally have one authoritative refresh cadence. Keep
        # the older DT-named setting as a synchronized compatibility alias.
        self.config.update({
            'dt_target_update_interval': self.config[
                'theta_crossing_map_update_interval'],
        })
        self.scroll = scroll
        self.paths = paths
        # Role toggles edit the resident manifest, but must retain the
        # explicitly configured sources for subsequent enables.
        self._configured_pcl_sources = tuple(paths.pcls)
        self.interactive_driver = interactive_driver
        self.progress = progress
        self.resume_path = resume_path or None
        self.out_base_dir = out_base_dir if out_base_dir is not None else './out'
        self.run_dir = run_dir or None
        self.run_tag = run_tag or None
        self.run_name = run_name
        self.non_liftable_patch_paths = set()
        # Who this process is in the job, as an explicit value. Callers that
        # joined a process group pass the context maybe_init_distributed()
        # returned; the default is the process context installed there (a
        # single-rank context when nothing joined one). Nothing below reads
        # RANK/WORLD_SIZE from the environment.
        self.dist = dist_context or process_context()

        # Scroll physical facts.
        self.scroll_name = scroll.name
        self.voxel_size_um = float(scroll.voxel_size_um)
        self.base_shape_zyx = scroll.base_shape_zyx
        self.spiral_outward_sense = scroll.spiral_outward_sense
        self.normal_zarr_group = scroll.normal_zarr_group
        self.lasagna_scale = int(scroll.lasagna_scale)
        umbilicus_path = paths.umbilicus
        umbilicus_scale = float(scroll.umbilicus_coordinate_scale)
        self.umbilicus_z_to_yx = lambda: json_umbilicus_z_to_yx(
            umbilicus_path, coordinate_scale=umbilicus_scale)

        # Resolved input paths ('' means absent).
        self.scroll_zarr_path = paths.scroll_zarr or None
        use_normals = input_source_enabled(config, 'normals')
        self.normal_nx_zarr_path = (paths.normal_x or None) if use_normals else None
        self.normal_ny_zarr_path = (paths.normal_y or None) if use_normals else None
        self.grad_mag_zarr_path = (
            (paths.gradient_magnitude or None)
            if input_source_enabled(config, 'gradient_magnitude') else None)
        self.winding_inference_path = (
            (paths.winding_inference or None)
            if winding_inference_enabled(config) else None)
        self.fibers_path = (
            (paths.fibers or None)
            if input_source_enabled(config, 'fibers') else None)
        self.fiber_directions_path = (
            (paths.fiber_directions or None)
            if input_source_enabled(config, 'fiber_directions') else None)
        self.verified_patches_path = (
            (paths.verified_patches or None)
            if input_source_enabled(config, 'verified_patches') else None)
        self.shell_path = (
            (paths.outer_shell or None)
            if input_source_enabled(config, 'outer_shell') else None)
        self.tracks_dbm_path = (
            (paths.tracks_dbm or None)
            if input_source_enabled(config, 'tracks_dbm') else None)
        self.pcl_input_specs = [
            (spec.path, spec.role.value if spec.role is not None else None)
            for spec in paths.pcls
            if pcl_input_enabled(config, spec.role, spec.path)
        ]

        # Deployment/presentation values.
        self.cache_path = cache_dir if cache_dir is not None else (paths.cache_directory or None)
        self.lasagna_storage_backend = storage_backend
        self.render_volume_scale = int(render_volume_scale)

        self._lasagna_store = None
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
        return shell_losses_enabled(self.config)

    def outer_shell_required(self):
        # The outer-shell requirement is fit-input catalog data, shared with
        # request validation and run admission.
        return fit_input('outer_shell').required(self.config)

    def _load_patches_from_dir(self, path, label='patches'):
        started_at = time.perf_counter()
        progress = progress_or_null(self.progress)
        entries = sorted(os.listdir(path))
        filter_regex = self.config['patch_uuid_filter_regex']
        if filter_regex is not None:
            filtered = [e for e in entries if re.search(filter_regex, e)]
            print(f'patch filter regex {filter_regex!r} kept '
                  f'{len(filtered)}/{len(entries)} {label} entries')
            entries = filtered
        progress.begin(
            'loading', f'Loading and filtering {label}',
            step=0, total_steps=len(entries), unit='patches')

        # Patch decode is dominated by per-file Python overhead (PIL TIFF tag
        # parsing etc.), so threads serialize on the GIL — worker processes
        # are required to actually parallelize.  Each worker runs a few IO
        # threads so per-file latency (e.g. NFS round trips) overlaps with
        # decode; patches come back as numpy payloads (see patch_to_payload).
        world_size = max(1, self.dist.world_size)
        configured_workers = int(os.environ.get(
            'FIT_SPIRAL_PATCH_LOAD_WORKERS',
            max(1, min(16, os.cpu_count() or 1) // world_size)))
        num_workers = max(1, min(configured_workers, len(entries)))
        io_threads = max(1, int(os.environ.get(
            'FIT_SPIRAL_PATCH_LOAD_IO_THREADS', 4)))
        chunk_size = 256
        erode_cells_default = self.config['patch_erode_patches']

        results_by_entry = {}
        loaded = 0
        completed = 0

        def consume(chunk_results):
            nonlocal loaded, completed
            for entry, payload, error, reason in chunk_results:
                if payload is not None:
                    patch = patch_from_payload(payload)
                    patch._source_path = os.path.abspath(
                        os.path.join(path, entry))
                    results_by_entry[entry] = (patch, None, None)
                    loaded += 1
                else:
                    results_by_entry[entry] = (None, error, reason)
                completed += 1
            progress.update(completed, detail=f'{loaded:,} loaded')

        chunks = [entries[start:start + chunk_size]
                  for start in range(0, len(entries), chunk_size)]
        if num_workers == 1 or len(chunks) <= 1:
            for chunk in chunks:
                consume(load_patch_payload_chunk(
                    path, chunk, self.z_begin, self.z_end,
                    erode_cells_default, io_threads))
        else:
            # spawn/forkserver rather than fork: loader workers only touch
            # CPU code, and forking after torch/OpenMP threads exist is not
            # reliably safe.  Workers persist across chunks, so the import
            # cost is paid once per worker.
            mp_context = multiprocessing.get_context(
                'forkserver'
                if 'forkserver' in multiprocessing.get_all_start_methods()
                else 'spawn')
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(num_workers, len(chunks)),
                    mp_context=mp_context) as executor:
                futures = [
                    executor.submit(
                        load_patch_payload_chunk, path, chunk,
                        self.z_begin, self.z_end,
                        erode_cells_default, io_threads)
                    for chunk in chunks]
                for future in concurrent.futures.as_completed(futures):
                    consume(future.result())

        patches = {}
        dropped = {
            'z ROI prefilter': 0,
            'erosion': 0,
            'z ROI after erosion': 0,
        }
        for entry in entries:
            patch, error, reason = results_by_entry[entry]
            if error is not None:
                print(f'Failed to load segment {entry}: {error}')
            elif reason is not None:
                dropped[reason] += 1
            else:
                patches[entry] = patch
        if dropped['z ROI prefilter']:
            print(f"z ROI prefilter skipped {dropped['z ROI prefilter']}/"
                  f'{len(entries)} {label} entries before full TIFF decode')
        if dropped['erosion']:
            print(f"erosion removed {dropped['erosion']}/{len(entries)} "
                  f'{label} entries')
        if dropped['z ROI after erosion']:
            print(f"erosion moved {dropped['z ROI after erosion']}/"
                  f'{len(entries)} {label} entries outside the z ROI')
        print(f'loaded and filtered {len(patches):,}/{len(entries):,} {label} '
              f'({_startup_resource_suffix(started_at)})')
        return patches

    def _prepare_patch_sampling_cache(self, patches):
        progress = progress_or_null(self.progress)
        progress.begin(
            'loading', 'Preparing patch sampling',
            step=0, total_steps=len(patches), unit='patches')
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
            # Every patch loss and its cached theta lift use one connected
            # surface. Ignore detached islands rather than inventing an
            # integer winding offset between components. Eight-connectivity
            # matches both patch edge topology and the former Dijkstra graph.
            in_roi_quad_mask_np = largest_patch_quad_component(
                in_roi_quad_mask_np)
            patch._sampling_valid_quad_mask_np = in_roi_quad_mask_np
            patch_scale = np.asarray(
                patch.scale.detach().cpu()
                if hasattr(patch.scale, 'detach') else patch.scale,
                dtype=np.float64)
            patch._sampling_area = float(
                in_roi_quad_mask_np.sum() * (1.0 / patch_scale).prod())
            progress.update(patch_idx + 1)

        return self._patch_sampling_probabilities(patches)

    def _patch_sampling_probabilities(self, patches):
        if not patches:
            return None
        areas = np.asarray([
            float(getattr(patch, '_sampling_area', patch.area))
            for patch in patches
        ], dtype=np.float32)
        weights = areas ** self.config['patch_sampling_area_exponent']
        return weights / weights.sum()

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

    def _derive_point_inputs(self, verified_patches, point_collections,
                             fiber_point_collections, *, direction_source='umbilicus'):
        """Shared initial/replacement linking, classification and sampling.

        Callers provide private point dictionaries; this method attaches and
        normalizes them while deriving every regular/fiber training view.
        """
        progress = progress_or_null(self.progress)
        fiber_catalog = {
            str(pcl['metadata']['logical_input_id']): pcl
            for pcl in fiber_point_collections.values()
        }
        for pcl in point_collections.values():
            for point in pcl['points'].values():
                point['zyx'] = np.array([point['p'][2], point['p'][1], point['p'][0]], dtype=np.float32)

        def pcl_intersects_z_roi(pcl):
            for point in pcl['points'].values():
                z = point['zyx'][0]
                if self.z_begin <= z < self.z_end:
                    return True
            return False

        link_distance_tolerance = float(self.config['pcl_link_distance_tolerance'])

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
        # Startup uses the umbilicus until a fitted transform is available.
        # Live revisions select the direction for their current step, just
        # like _maybe_relink_fibers_for_direction.
        self._fiber_link_direction_source = direction_source
        link_options = self._patch_link_options(
            point_collections, 1.0, direction_source=direction_source)
        if link_options.side_rules:
            print(self._describe_fiber_link_side_rules(link_options))
        link_points_to_patches(
            verified_patches,
            point_collections,
            tolerance=link_distance_tolerance,
            surface_index_tolerance=link_distance_tolerance,
            distance_scale=1.0,
            general_hit_policy='largest_area',
            options=link_options,
        )

        # Keep pristine linked regular collections. Derived training views may
        # trim or decimate points, but live patches must still be offered every
        # resident point that has not already attached to a patch.
        regular_pcl_catalog = {
            pid: catalog_copy_of_pcl(pcl)
            for pid, pcl in point_collections.items()
            if pid not in fiber_point_collections
        }

        # Every pcl carries the uniform chain interface (pcl['chain'], see
        # spiral_helpers.Chain): a SequenceChain over the id-sorted point order
        # (lazy, so later point-dict replacements are picked up). Consumers go
        # through this and never assume id-sorted order is chain-valid.
        # merge_linked_point_collections below replaces link-connected members with
        # a merged component pcl carrying a graph-routing ComponentChain instead.
        for pcl in point_collections.values():
            pcl['chain'] = SequenceChain(pcl)

        # ==========================================================================
        # Regular point collection classification
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
            if pid in fiber_point_collections:
                continue
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
                if num_unannotated:
                    raise ValueError(
                        f'winding_is_absolute pcl {pid} ({pcl.get("name")!r}) has {num_unannotated} of '
                        f'{len(attached_points)} attached points without a winding annotation; absolute pcls '
                        f'must give every winding number explicitly'
                    )
                num_non_positive = sum(1 for point in attached_points if point['winding_annotation'] <= 0)
                if num_non_positive:
                    raise ValueError(
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
        # The first and last points are always kept. Fiber junction retention is
        # handled by materialize_fiber_fit_inputs below.
        for pcl_id, pcl in unattached_point_collections.items():
            sorted_items = sorted(pcl['points'].items(), key=lambda kv: int(kv[0]))
            if len(sorted_items) < 2:
                continue
            zyxs = np.stack([point['zyx'] for _, point in sorted_items], axis=0).astype(np.float32)
            windings = np.array([point['winding_annotation'] for _, point in sorted_items], dtype=np.float32)

            zyxs, keep = _decimate_ordered_points_min_spacing(
                zyxs, min_point_spacing, return_indices=True,
                force_keep={len(zyxs) - 1})
            windings = windings[keep]
            unattached_pcl_strips.append({
                'id': pcl_id,
                'name': pcl.get('name'),
                'source_file': pcl.get('source_file'),
                'zyxs': zyxs,
                'windings': windings,
                'link_points': {},
                'logical_input_kind': pcl.get('metadata', {}).get(
                    'logical_input_kind'),
                'logical_input_id': pcl.get('metadata', {}).get(
                    'logical_input_id'),
                'logical_input_revision': pcl.get('metadata', {}).get(
                    'logical_input_revision'),
            })
            unattached_strip_sampling_groups.append(pcl['sampling_group'])

        (fiber_cross_patch, fiber_strips, fiber_sampling_groups,
         resolved_links, link_components) = materialize_fiber_fit_inputs(
            fiber_catalog,
            verified_patches,
            z_begin=self.z_begin,
            z_end=self.z_end,
            z_margin=self.config['patch_loss_z_margin'],
            min_point_spacing=min_point_spacing,
            use_links=self.config['pcl_use_fiber_links'],
            use_pending_links=self.config['pcl_use_pending_fiber_links'],
            vertical_min_z_fraction=self.config['pcl_vertical_fiber_min_z_fraction'],
            vertical_min_auto_certainty=self.config['pcl_vertical_fiber_min_auto_certainty'],
            vertical_radial_offset=self._vertical_fiber_radial_offset(),
        )
        cross_patch_point_collections.update(
            {pcl['id']: pcl for pcl in fiber_cross_patch})
        unattached_pcl_strips.extend(fiber_strips)
        unattached_strip_sampling_groups.extend(fiber_sampling_groups)
        print(f'fiber links: {len(resolved_links)} resolved')
        link_warning = unresolved_fiber_link_warning(
            fiber_catalog,
            use_links=self.config['pcl_use_fiber_links'],
            use_pending_links=self.config['pcl_use_pending_fiber_links'])
        if link_warning is not None:
            print(f'WARNING: {link_warning}')

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
                            f'{key} (w={pcl_sampling_group_weight(group, self.config)}): {count}')
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
        self.fiber_catalog = fiber_catalog
        self.regular_pcl_catalog = regular_pcl_catalog
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
        self.link_distance_tolerance = link_distance_tolerance

    def check_cuda_ready(self):
        """Create the CUDA context before input reads increase host cache pressure."""
        progress_or_null(self.progress).begin('loading', 'Checking CUDA availability')
        try:
            # Use the current device selected by the distributed driver. Keep
            # the allocation alive through synchronization; do not consume RNG.
            probe = torch.empty(1, device='cuda')
            torch.cuda.synchronize()
            del probe
        except RuntimeError as exc:
            message = (
                'CUDA startup check failed before loading fit inputs '
                '(one-element allocation and synchronization). '
                'This failure is independent of the patch atlas size. '
                f'Original error: {exc}')
            is_oom = isinstance(exc, torch.OutOfMemoryError) or 'out of memory' in str(exc).lower()
            if is_oom and sys.platform == 'linux':
                # Diagnostics must not hide the original error if querying
                # device properties or Linux memory accounting also fails.
                try:
                    shared_memory = 'GB10' in torch.cuda.get_device_name()
                except Exception:
                    shared_memory = False
                if shared_memory:
                    message += (
                        '\nNVIDIA GB10 shares RAM with the CPU. Filesystem cache '
                        'pressure or physical-memory fragmentation can prevent '
                        'CUDA context allocation despite ample available RAM. '
                        'Stop the Spiral service, then try NVIDIA\'s manual '
                        'cache-flush workaround and restart the service: '
                        "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'. "
                        'This releases filesystem caches and may slow subsequent reads; '
                        'Spiral does not run it automatically.')
                    try:
                        with open('/proc/meminfo') as stream:
                            fields = [line.strip() for line in stream
                                      if line.split(':', 1)[0] in {
                                          'MemTotal', 'MemFree', 'MemAvailable',
                                          'Buffers', 'Cached'}]
                        message += '\nHost memory: ' + '; '.join(fields)
                    except OSError:
                        pass
            raise RuntimeError(message) from exc

    def load_host_inputs(self):
        """Load and prepare every host-side input for a fit.

        Seeds the host RNG streams, then loads patches, point collections,
        fibers, tracks, and the outer shell; links and classifies PCLs;
        builds the sampling caches, host-prepared patch atlases, the
        trusted-geometry index and the whole-object DT target samples. Requires no device state: the
        patch atlases are moved as part of device-state setup, and the CUDA
        stores, model, and optimiser are built later by
        build_device_state().

        Everything the device stages read from here they only read: nothing
        below the cut consumes or releases a host structure, so the model
        stage can be re-run against inputs this method loaded once.

        The loaded inputs stay readable as attributes afterwards, so
        analysis tools can construct a context, call this, and read:

        - patches: verified_patches (+ verified_patches_list,
          num_verified_patches), shell_patch
        - point collections: cross_patch_pcls, unattached_pcl_strips,
          unattached_strip_sampling_groups, pcl_sampling_strata, next_id,
          fiber_catalog, link_distance_tolerance, resolved_links, link_components,
          unattached_components / _component_groups / _component_edges
        - sampling: patch_sampling_probabilities, patch_atlas
        - trusted geometry: trusted_geometry_tree
        - tracks: tracks, track_families, track_source_ids,
          track_crossing_cache, track_graph, track_reload_source /
          _families / _source_ids, track_sampling_config, using_tracks,
          filter_tracks_by_shell
        - misc: umbilicus, scroll_zarr, shell_envelope,
          dense_spacing_mode, grad_mag_spacing_enabled
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

        progress.begin('loading', 'Loading fiber direction samples')
        fiber_direction_samples = None
        if (self.fiber_directions_path
                and os.path.isfile(self.fiber_directions_path)):
            fiber_direction_samples = load_fiber_direction_samples(
                self.fiber_directions_path, self.z_begin, self.z_end)
        elif self.config['loss_weight_fiber_directions'] > 0:
            if not self.config['input_use_fiber_directions']:
                raise RuntimeError(
                    'fiber-direction loss is enabled, but its input source is '
                    'off; set input_use_fiber_directions')
            raise RuntimeError(
                'fiber-direction loss is enabled, but its sample file '
                f'is missing: {self.fiber_directions_path!r}')
        if fiber_direction_samples is not None:
            print(f'fiber directions: {len(fiber_direction_samples["position_zyx"]):,} '
                  f'samples in z ROI')
        elif self.config['loss_weight_fiber_directions'] > 0:
            raise RuntimeError(
                f'fiber-direction sample file contains no points in '
                f'z ROI [{self.z_begin}, {self.z_end})')

        # ==========================================================================
        # Patch loading and ROI filtering
        # ==========================================================================

        filter_tracks_by_shell = bool(self.tracks_dbm_path) and bool(self.shell_path)
        shell_patch = None
        if self.outer_shell_required() or filter_tracks_by_shell:
            if not self.shell_path:
                raise RuntimeError(
                    'the outer shell is required by shell losses or winding-model '
                    'supervision, but no outer shell path is set')
            progress.begin('loading', 'Loading outer shell')
            shell_patch = load_tifxyz(self.shell_path)

        use_verified_patches = bool(self.verified_patches_path) and not self.config['input_disable_patches']
        if not use_verified_patches:
            verified_patches = {}
            print('skipping patch loading')
        else:
            verified_patches = self._load_patches_from_dir(
                self.verified_patches_path, 'verified patches')
            if not verified_patches:
                raise RuntimeError('No patches could be loaded')

        print(f" loaded {len(verified_patches)} patches")

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
                    stamp_loaded_pcl_metadata(pcl, path, explicit_role, next_id)
                    point_collections[next_id] = pcl
                    next_id += 1
            progress.update(
                spec_number, detail=f'{len(point_collections):,} collections loaded')

        progress.begin('loading', 'Loading fiber point collections')
        fiber_point_collections, next_id = load_fiber_point_collections(
            self.fibers_path,
            next_id,
            min_point_spacing=self.config['pcl_fiber_min_point_spacing'],
            base_shape_zyx=getattr(self, 'base_shape_zyx', None),
        )
        # All fibers (horizontal, vertical, and merged link components) form one
        # sampling group, rather than one group per source file like the regular pcls.
        for pcl in fiber_point_collections.values():
            pcl['sampling_group'] = 'fibers'
            source = pcl.get('source_file', '')
            logical_id = os.path.splitext(os.path.basename(source))[0]
            try:
                with open(source, 'rb') as stream:
                    revision = hashlib.sha256(stream.read()).hexdigest()
            except OSError:
                revision = None
            pcl.setdefault('metadata', {}).update({
                'logical_input_kind': 'fiber',
                'logical_input_id': logical_id,
                'logical_input_revision': revision,
            })
            # Staging paths are content-addressed; branch records address the
            # stable logical document name instead.
            pcl['file_basename'] = f'{logical_id}.json'
        fiber_catalog = {
            str(pcl['metadata']['logical_input_id']): pcl
            for pcl in fiber_point_collections.values()
        }
        point_collections.update(fiber_point_collections)

        if self.interactive_driver is not None:
            self._source_point_collections = {
                pid: catalog_copy_of_pcl(pcl) for pid, pcl in point_collections.items()
            }
            self._source_verified_patches = dict(verified_patches)
        self._derive_point_inputs(
            verified_patches, point_collections, fiber_point_collections)
        unattached_pcl_strips = self.unattached_pcl_strips
        link_distance_tolerance = self.link_distance_tolerance

        # ==========================================================================
        # dense-spacing mode, shell envelope, and tracks
        # ==========================================================================

        # Dense-spacing input contract. Checked before any asset paths so
        # an invalid mode fails as itself, not as a missing-file error.
        dense_spacing_mode = self.config['dense_spacing_mode']
        if dense_spacing_mode not in ('grad_mag', 'winding_model'):
            raise ValueError(
                f'dense_spacing_mode={dense_spacing_mode!r} must be '
                "'grad_mag' or 'winding_model'")
        winding_model_mode = winding_inference_enabled(self.config)
        grad_mag_spacing_enabled = (
            dense_spacing_mode == 'grad_mag'
            and input_source_enabled(self.config, 'gradient_magnitude')
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
            if track_sampling_config['crossing_precompute_max'] > 0:
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

        # Keep the verified atlas as a required session object even when this
        # run has no verified patches.  Device setup and theta-topology
        # registration deliberately consume an empty atlas without special
        # casing, and interactive patch incorporation can append to it later.
        patch_atlas = PatchAtlas(verified_patches, device='cuda')
        if verified_patches:
            print(f'patch atlas: {patch_atlas.memory_mb():.1f} MB')
            topology_stats = patch_atlas.topology_memory_stats()
            print(
                'compact patch topology: '
                f"{int(topology_stats['num_valid_cells']):,} quads, "
                f"{int(topology_stats['persistent_bytes']) / (1 << 30):.2f} GiB host "
                f"({_startup_resource_suffix()})")

        # ==========================================================================================
        # trusted geometry (verified patches and pcls) kdtree / tracks masking
        # ==========================================================================================

        # The trusted point cloud is consumed only by a CPU cKDTree. Build it directly
        # on CPU instead of storing it in the atlas on CUDA, concatenating it again on
        # CUDA, and immediately copying it back here.
        trusted_counts = []
        for patch in verified_patches_list:
            z_flat = patch.zyxs.reshape(-1, 3).to(dtype=torch.float32)
            valid_flat = patch.valid_vertex_mask.reshape(-1)
            z_in_roi = (z_flat[:, 0] >= self.z_begin) & (z_flat[:, 0] < self.z_end)
            trusted_counts.append(int((valid_flat & z_in_roi).sum()))
        for strip in unattached_pcl_strips:
            zyxs_np = np.asarray(strip['zyxs'])
            trusted_counts.append(int((
                (zyxs_np[..., 0] >= self.z_begin)
                & (zyxs_np[..., 0] < self.z_end)).sum()))
        verified_patches_and_pcls_cpu = torch.empty(
            (sum(trusted_counts), 3), dtype=torch.float32)
        trusted_offset = 0
        count_index = 0
        for patch in verified_patches_list:
            count = trusted_counts[count_index]
            count_index += 1
            if count:
                z_flat = patch.zyxs.reshape(-1, 3).to(dtype=torch.float32)
                valid_flat = patch.valid_vertex_mask.reshape(-1)
                selected = valid_flat & (z_flat[:, 0] >= self.z_begin) \
                    & (z_flat[:, 0] < self.z_end)
                verified_patches_and_pcls_cpu[
                    trusted_offset:trusted_offset + count].copy_(z_flat[selected])
                trusted_offset += count
        for strip in unattached_pcl_strips:
            count = trusted_counts[count_index]
            count_index += 1
            if count:
                zyxs = torch.from_numpy(strip['zyxs']).to(dtype=torch.float32)
                selected = (zyxs[..., 0] >= self.z_begin) \
                    & (zyxs[..., 0] < self.z_end)
                verified_patches_and_pcls_cpu[
                    trusted_offset:trusted_offset + count].copy_(zyxs[selected])
                trusted_offset += count

        using_tracks = (
            (self.config['loss_weight_track_radius'] > 0 or self.config['loss_weight_track_dt'] > 0)
            and bool(tracks)
        )
        trusted_geometry_tree = None
        verified_patches_and_pcls_np = None

        # The trusted anchor cloud (verified patch vertices + pcl strips) drives
        # the DBM-track exclusion in tracks.py.
        if using_tracks:
            # Build a cKDTree over the scroll-space anchor points (CPU) for fixed-radius
            # nearest-neighbour queries.
            verified_patches_and_pcls_np = verified_patches_and_pcls_cpu.numpy()
            verified_patches_and_pcls_np = np.ascontiguousarray(verified_patches_and_pcls_np, dtype=np.float32)
            if verified_patches_and_pcls_np.shape[0] > 0:
                progress.begin(
                    'loading', 'Building trusted-geometry index',
                    detail=f'{len(verified_patches_and_pcls_np):,} points')
                trusted_geometry_tree = cKDTree(verified_patches_and_pcls_np)


        # Loaded host inputs, kept as inspectable attributes (ownership
        # class (b): host-prepared inputs and caches). cross_patch_pcls,
        # unattached_pcl_strips, unattached_strip_sampling_groups, and
        # pcl_sampling_strata were assigned above, before the strata build.
        self.umbilicus = umbilicus
        self.scroll_zarr = scroll_zarr
        self.fiber_direction_samples = fiber_direction_samples
        self.filter_tracks_by_shell = filter_tracks_by_shell
        self.shell_patch = shell_patch
        self.shell_envelope = shell_envelope
        self.verified_patches = verified_patches
        self.next_id = next_id
        self.link_distance_tolerance = link_distance_tolerance
        self.dense_spacing_mode = dense_spacing_mode
        self.winding_model_mode = winding_model_mode
        self.dense_normals_enabled = input_source_enabled(
            self.config, 'normals')
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

        # The trusted cloud itself stays local: the cKDTree above is all
        # anything downstream reads it for, and consuming it here rather
        # than in build_device_state() is what leaves the device stages
        # with nothing of the host's to release.
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
                detail=f'{len(self.verified_patches_list):,} patches')
            prepare_patch_dt_target_samples(
                self.verified_patches_list, self.config['sample_count_patch_dt_target_points'], self.config['dt_target_max_stride'],
            )

    def _vertical_fiber_radial_offset(self):
        """Radial target offset (voxels) applied to vertical fiber strips; 0 when off."""
        if not self.config['pcl_vertical_fiber_radial_offset_enabled']:
            return 0.0
        return float(self.config['pcl_vertical_fiber_radial_offset_voxels'])

    # ---- Point-to-patch linking options -----------------------------------

    def _fiber_link_side_rules(self, collections):
        """Side rule per collection id (point_collection.SIDE_*) under
        pcl_fiber_link_side_filter: vertical fibers lie on the sheet's back
        and attach only to patches in front of them, horizontal fibers on its
        front and attach only to patches behind them. Untagged fibers and
        regular collections carry no rule."""
        if not self.config['pcl_fiber_link_side_filter']:
            return {}
        rules = {}
        for cid, pcl in collections.items():
            tag = fiber_collection_hv_tag(
                pcl,
                min_z_fraction=self.config['pcl_vertical_fiber_min_z_fraction'],
                min_auto_certainty=self.config[
                    'pcl_vertical_fiber_min_auto_certainty'])
            if tag == 'V':
                rules[cid] = SIDE_FRONT
            elif tag == 'H':
                rules[cid] = SIDE_BEHIND
        return rules

    def _fiber_link_inward_direction(self, source, *, umbilicus_z_to_yx=None):
        """The side rules' inward-direction function for ``source``:
        'umbilicus' (the line to the umbilicus in the current input frame,
        the z-axis after a constraint bake) or 'model' (the fitted spiral's
        decreasing-winding direction, see inward_winding_direction)."""
        if source == 'model':
            transform = self.spiral_and_transform.get_slice_to_spiral_transform()
            device = self.device

            def inward(zyxs):
                return inward_winding_direction(transform, zyxs, device=device)
            return inward
        if source != 'umbilicus':
            raise ValueError(f'unknown fiber link direction source {source!r}')
        if umbilicus_z_to_yx is None:
            umbilicus_z_to_yx = (
                self._zero_umbilicus_z_to_yx if getattr(self, "frozen_epochs", ())
                else self.umbilicus)
        return umbilicus_inward_direction(umbilicus_z_to_yx)

    def _fiber_views_from_catalog(self, fiber_catalog, cross_patch, strips,
                                  strip_groups, voxel_scale):
        """Replace the fiber-derived entries of the given training views with
        views materialised from ``fiber_catalog`` under the current config.

        Shared by live fiber incorporation/relinking and the Run-boundary
        re-materialisation. Returns the new ``(cross_patch, strips,
        strip_groups, resolved_links, link_components)``; the inputs are not
        mutated, so callers commit (or discard) the result as one unit.
        """
        cross_patch = [
            pcl for pcl in cross_patch
            if pcl.get('metadata', {}).get('input_role')
            not in {'fiber', 'fiber_link_component'}
            and pcl.get('metadata', {}).get('logical_input_kind') != 'fiber'
        ]
        retained = [
            (strip, group) for strip, group in zip(strips, strip_groups)
            if strip.get('logical_input_kind') != 'fiber'
        ]
        strips = [strip for strip, _ in retained]
        strip_groups = [group for _, group in retained]
        (fiber_cross_patch, fiber_strips, fiber_sampling_groups,
         resolved_links, link_components) = materialize_fiber_fit_inputs(
            fiber_catalog,
            self.verified_patches,
            z_begin=self.z_begin,
            z_end=self.z_end,
            z_margin=self.config['patch_loss_z_margin'],
            min_point_spacing=(
                self.config['pcl_unattached_pcl_min_point_spacing']
                * voxel_scale),
            use_links=self.config['pcl_use_fiber_links'],
            use_pending_links=self.config['pcl_use_pending_fiber_links'],
            vertical_min_z_fraction=self.config[
                'pcl_vertical_fiber_min_z_fraction'],
            vertical_min_auto_certainty=self.config[
                'pcl_vertical_fiber_min_auto_certainty'],
            vertical_radial_offset=self._vertical_fiber_radial_offset(),
        )
        return (cross_patch + list(fiber_cross_patch),
                strips + list(fiber_strips),
                strip_groups + list(fiber_sampling_groups),
                resolved_links, link_components)


    def _refresh_trusted_geometry(self):
        """Rebuild the trusted-geometry tree from the active patches and strips."""
        trusted = self._trusted_geometry_from_active_inputs()
        trusted_np = np.ascontiguousarray(
            trusted.cpu().numpy(), dtype=np.float32)
        self.trusted_geometry_tree = (
            cKDTree(trusted_np) if len(trusted_np) else None)


    def _commit_rederived_pcl_views(self, cross_patch, strips, strip_groups,
                                    resolved_links=None, link_components=None):
        """Swap re-derived point-collection views into the resident session.

        Run-boundary counterpart of the live-incorporation commit: the lists
        are replaced in place (every holder observes them), the cached flat
        strip bundle is dropped, and everything derived from the views is
        refreshed -- sampling strata, whole-object DT caches, the trusted
        geometry, and the patch/PCL theta topology.
        """
        self.cross_patch_pcls[:] = cross_patch
        self.unattached_pcl_strips[:] = strips
        self.unattached_strip_sampling_groups[:] = strip_groups
        self.unattached_pcl_strips.flat = None
        if resolved_links is not None:
            self.resolved_links[:] = resolved_links
            self.link_components[:] = link_components
        self._rebuild_pcl_sampling_strata()
        self.dt_target_cache_manager.reset()
        self._refresh_trusted_geometry()
        for warning in self._build_theta_crossing_map() or ():
            print(f'WARNING: {warning}')


    def _rematerialize_fiber_views(self):
        """Re-derive every fiber training view from the retained fiber catalog.

        Run-boundary path for the fiber-view settings (link usage, vertical
        classification thresholds, unattached decimation spacing): the
        catalog owns the canonical (already baked, if a reset happened)
        fiber geometry, so no reload or re-bake is needed.
        """
        views = self._fiber_views_from_catalog(
            self.fiber_catalog, list(self.cross_patch_pcls),
            list(self.unattached_pcl_strips),
            list(self.unattached_strip_sampling_groups),
            self._baked_voxel_scale())
        self._commit_rederived_pcl_views(*views)
        link_warning = unresolved_fiber_link_warning(
            self.fiber_catalog,
            use_links=self.config['pcl_use_fiber_links'],
            use_pending_links=self.config['pcl_use_pending_fiber_links'])
        if link_warning is not None:
            print(f'WARNING: {link_warning}')


    def _baked_voxel_scale(self):
        """Blunt scroll-voxel -> current-frame factor for metric thresholds.

        Link tolerances, decimation spacings and exclusion radii are tuned in
        scroll voxels; after a bake the resident geometry is expressed in
        units stretched by the accumulated median stretch of the frozen
        stack (see _baked_unit_scale). 1.0 before the first bake.
        """
        return (float(self._baked_unit_scale)
                if getattr(self, 'frozen_epochs', ()) else 1.0)


    def _patch_link_options(self, collections, voxel_scale, *,
                            direction_source=None, umbilicus_z_to_yx=None):
        """PatchLinkOptions for linking ``collections`` (resident id -> pcl)
        in the current input frame. ``voxel_scale`` converts scroll-voxel
        thresholds to that frame; ``direction_source`` defaults to the
        session's current fiber link direction source."""
        rules = self._fiber_link_side_rules(collections)
        inward = None
        if rules:
            if direction_source is None:
                direction_source = getattr(
                    self, '_fiber_link_direction_source', 'umbilicus')
            inward = self._fiber_link_inward_direction(
                direction_source, umbilicus_z_to_yx=umbilicus_z_to_yx)
        return PatchLinkOptions(
            window_points=int(self.config['pcl_link_window_points']),
            window_min_points=int(self.config['pcl_link_window_min_points']),
            side_rules=rules,
            inward_direction=inward,
            side_margin=(float(self.config['pcl_fiber_link_side_margin_voxels'])
                         * voxel_scale))

    def _describe_fiber_link_side_rules(self, options):
        rules = options.side_rules
        front = sum(1 for rule in rules.values() if rule == SIDE_FRONT)
        behind = len(rules) - front
        return (
            f'fiber link side rules: {front} vertical fiber(s) restricted to '
            f'patches in front, {behind} horizontal fiber(s) to patches behind '
            f'(margin {options.side_margin:g} voxels, inward direction from the '
            f'{getattr(self, "_fiber_link_direction_source", "umbilicus")}, '
            f'switching to the fitted winding at step '
            f'{int(self.config["pcl_fiber_link_model_direction_step"])})')

    def _desired_fiber_link_direction_source(self, iteration):
        threshold = int(self.config['pcl_fiber_link_model_direction_step'])
        return 'model' if iteration >= threshold else 'umbilicus'

    def _maybe_relink_fibers_for_direction(self, iteration):
        """Step hook for the fiber side rules' direction switch.

        Once ``iteration`` (completed steps, whether run here or restored from
        a checkpoint) reaches pcl_fiber_link_model_direction_step, every
        fiber is relinked once under the fitted winding direction; a
        checkpoint that lands before the step after such a switch relinks
        back under the umbilicus direction. No-op while the side filter is
        off or the session holds no fibers.
        """
        if not self.config['pcl_fiber_link_side_filter']:
            return
        if not getattr(self, 'fiber_catalog', None):
            return
        desired = self._desired_fiber_link_direction_source(iteration)
        if desired == getattr(self, '_fiber_link_direction_source', 'umbilicus'):
            return
        self._relink_fibers_to_patches(desired, iteration=iteration)

    def _relink_fibers_to_patches(self, direction_source, *, iteration=None):
        """Drop every resident fiber point's patch attachment and link the
        fiber catalog again against the verified patches, the side rules
        taking their inward direction from ``direction_source``, then
        re-materialise the fiber training views."""
        collections = {pcl['id']: pcl for pcl in self.fiber_catalog.values()}

        def attached_count():
            return sum(
                1 for pcl in collections.values()
                for point in pcl['points'].values() if 'on_patch' in point)

        before = attached_count()
        for pcl in collections.values():
            for point in pcl['points'].values():
                point.pop('on_patch', None)
        voxel_scale = self._baked_voxel_scale()
        tolerance = self.link_distance_tolerance * voxel_scale
        options = self._patch_link_options(
            collections, voxel_scale, direction_source=direction_source)
        link_points_to_patches(
            self.verified_patches,
            collections,
            tolerance=tolerance,
            surface_index_tolerance=tolerance,
            distance_scale=1.0,
            general_hit_policy='largest_area',
            options=options,
        )
        self._fiber_link_direction_source = direction_source
        dist = getattr(self, 'dist', None)
        if dist is None or dist.is_main_process:
            prefix = '' if iteration is None else f'step {iteration}: '
            print(f'{prefix}relinked {len(collections)} fiber(s) with the '
                  f'{direction_source} inward direction: {before} -> '
                  f'{attached_count()} attached points')
        self._rematerialize_fiber_views()

    def _refill_vertical_fiber_radial_offsets(self):
        """Re-derive every strip's per-point radial target offset from its
        retained vertical/horizontal tag and the current config.

        The offset (scroll voxels along the sheet normal) is written into each
        strip's ``radial_offsets`` array when the strip is materialised and
        then, times the strip's ``radial_offset_bake_scale``, concatenated
        into the cached ``.flat`` GPU bundle that the strip losses, DT
        targets, and satisfaction metric read. A Run-boundary change to the
        offset therefore refills those arrays, drops the cached bundle, and
        resets the DT target caches that were computed against the old
        offsets; the bake scale is untouched. Strips without a tag (regular
        point collections, horizontal fibers) stay at zero.
        """
        offset = self._vertical_fiber_radial_offset()
        for strip in self.unattached_pcl_strips:
            strip['radial_offsets'] = np.full(
                len(strip['zyxs']),
                offset if strip.get('is_vertical') else 0.0,
                dtype=np.float32)
        self.unattached_pcl_strips.flat = None
        self.dt_target_cache_manager.reset()

    def _winding_model_mode_active(self):
        return self.winding_model_mode and self.winding_inference is not None

    def _warn_if_density_loss_inactive(self):
        # Run-mutable weights are read afresh every step, but the density
        # component only exists in winding-model mode; make another
        # session's nonzero density weight a visible no-op. The native
        # min-spacing barrier is asset-independent and active in every mode.
        if self.winding_model_mode:
            return
        weight_key = 'loss_weight_dense_spacing_density'
        if self.config[weight_key] > 0 and weight_key not in self._density_inactive_warned:
            self._density_inactive_warned.add(weight_key)
            print(f'WARNING: {weight_key} > 0 but dense_spacing_mode='
                  f'{self.dense_spacing_mode!r}; this component runs only in '
                  "'winding_model' mode and is INACTIVE.")

    def _make_shell_polar_map(self):
        # A session that filtered its tracks against the shell already built
        # the identical table on the host (the atlas keys are frozen for it,
        # and shell_min_confidence is read live at lookup): share it.
        if self.shell_envelope is not None:
            return self.shell_envelope.to(self.device)
        return ShellPolarMap(
            self.shell_patch, self.umbilicus,
            z_min=self.z_begin - self.config['model_flow_bounds_z_margin'],
            z_max=self.z_end + self.config['model_flow_bounds_z_margin'],
            num_theta_bins=self.config['shell_num_theta_bins'],
            device=self.device, config=self.config)

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
                      'winding-model assets, see any warnings above).')

    def _apply_flow_group_settings(self, iteration):
        """Per-step optimizer settings of the two flow-lattice groups: their
        LR scales relative to the base group and the moment flags are read
        every step. The catalog exposes optimizer flags at run boundaries,
        but classifies model_ LR controls as requiring a model rebuild.
        Returns (low-res scale, high-res scale)."""
        high_res_scale = get_flow_field_high_res_lr_scale(self.config, iteration)
        low_res_scale = get_flow_field_low_res_lr_scale(self.config)
        # The base group (pitch and shell parameters) carries the scheduled
        # optimizer_learning_rate unscaled; both flow groups hang off it.
        base_group = self.optimiser.param_groups[0]
        low_res_group = next(
            group for group in self.optimiser.param_groups
            if any(param is self.low_res_flow_params[0] for param in group['params']))
        high_res_group = next(
            group for group in self.optimiser.param_groups
            if any(param is self.high_res_flow_params[0] for param in group['params']))
        # A loaded checkpoint's groups replace the live hyperparameters, so
        # the flags are re-applied here rather than kept in the group.
        lazy_moments = bool(self.config.get('optimizer_flow_lazy_moments', False))
        shared_second_moment = bool(
            self.config.get('optimizer_flow_shared_second_moment', False))
        clip_quantile = self.config.get(
            'optimizer_flow_shared_second_moment_clip_quantile', 0.99)
        for group in (low_res_group, high_res_group):
            group['lazy_moments'] = lazy_moments
            group['shared_second_moment'] = shared_second_moment
            group['shared_second_moment_clip_quantile'] = clip_quantile
        for group, scale in ((low_res_group, low_res_scale), (high_res_group, high_res_scale)):
            set_optimizer_group_lr_scale(
                self.optimiser,
                self.lr_scheduler,
                group=group,
                reference_group=base_group,
                scale=scale,
                initial_lr=self.config['optimizer_learning_rate'],
            )
        return low_res_scale, high_res_scale

    def _report_flow_grad_conditioning(self):
        """Log the flow gradient smoothing widths in lattice cells.

        Widths use scroll-voxel units of the flow frame. Very small cell-unit
        widths collapse to identity kernels. Print the conversion at build
        time; subsequent live changes do not refresh this startup report.
        """
        if not self.dist.is_main_process:
            return
        if not self.config.get('optimizer_flow_grad_smoothing', False):
            return
        along, across, low_res = self._flow_grad_smoothing_widths()
        print(self.spiral_and_transform.describe_flow_grad_smoothing(along, across, low_res))
        if not (self.config.get('optimizer_flow_lazy_moments', False)
                or self.config.get('optimizer_flow_shared_second_moment', False)):
            print('NOTE: flow gradient smoothing with per-cell Adam moments and '
                  'without optimizer_flow_lazy_moments: the smoothed tails reach '
                  'cells whose Adam second moment has decayed, so their first '
                  'step is far larger than the tail warrants')

    def _flow_grad_smoothing_widths(self):
        """(along-sheet, across-ring, low-res along-sheet) widths in voxels."""
        along = float(self.config['optimizer_flow_grad_smoothing_sigma_voxels'])
        across = float(self.config.get(
            'optimizer_flow_grad_smoothing_across_sigma_voxels', 0.0) or 0.0)
        low_res = float(self.config.get(
            'optimizer_flow_grad_smoothing_low_res_sigma_voxels', 0.0) or 0.0)
        return along, across, low_res

    def _sanitize_nonfinite_grads_(self):
        """Count and zero nonfinite entries in every distributed gradient.

        Bumps nonfinite_grad_steps once per step with any nonfinite gradient
        and nonfinite_grad_by_param per affected parameter, then replaces
        NaN and +/-inf with zero in place.
        """
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

    def _clip_flow_grads(self):
        """Robustly clip both flow lattices' gradients per stage (see
        lazy_moment_adamw.robust_clip_) and keep the thresholds and clipped
        fractions for the step log. A non-positive multiple leaves the
        gradients alone and clears the stats."""
        multiple = float(self.config.get('optimizer_flow_grad_clip_median_multiple', 0.0) or 0.0)
        self.flow_grad_clip_stats = {}
        if multiple <= 0.0:
            return
        for name, param in (('LR', self.low_res_flow_params[0]),
                            ('HR', self.high_res_flow_params[0])):
            if param.grad is None:
                continue
            stats = robust_clip_(param.grad, multiple)
            if stats is not None:
                self.flow_grad_clip_stats[name] = stats

    def _flow_conditioning_report(self):
        """Per-lattice, per-stage gradient-conditioning lines and metrics.

        Update root-mean-squares are converted from the lattices' normalised
        flow-box units to scroll voxels per component (a cylindrical
        lattice's components are z, radial, tangential; a Cartesian one's
        z, y, x), so runs with different smoothing widths or denominators
        are compared on how far they actually move the field per step. Reads
        device statistics, so call it only when logging.
        """
        stats = getattr(self.optimiser, 'conditioning_stats', {})
        clip_stats = getattr(self, 'flow_grad_clip_stats', {})
        lines, payload = [], {}
        if not stats and not clip_stats:
            return lines, payload
        ranges = (self.spiral_and_transform.flow_max_corner_zyx
                  - self.spiral_and_transform.flow_min_corner_zyx).to(torch.float64).cpu()
        cylindrical = self.config['model_flow_field_type'] in ('cylindrical', 'bspline_cylindrical')
        component_names = ('z', 'r', 't') if cylindrical else ('z', 'y', 'x')
        # Component 0 is z; the in-plane components (radial and tangential,
        # or y and x) both live in the square yx box.
        component_voxels = [float(ranges[0]), float(ranges[1]), float(ranges[2])]
        for name, param in (('LR', self.low_res_flow_params[0]),
                            ('HR', self.high_res_flow_params[0])):
            entry = stats.get(param)
            clip = clip_stats.get(name)
            if entry is None and clip is None:
                continue
            stages = int(param.shape[0]) if param.dim() >= 3 else 1
            cells_per_stage = param.numel() / stages
            for stage in range(stages):
                parts = []
                prefix = f'flow_cond/{name}/stage{stage}/'
                if entry is not None:
                    rms = entry['update_rms'][stage].cpu()
                    count = entry['update_count'][stage].cpu()
                    voxels = [float(rms[c]) * component_voxels[c] for c in range(rms.numel())]
                    parts.append('update rms vox ' + ' '.join(
                        f'{component_names[c]}={v:.4f}' for c, v in enumerate(voxels)))
                    updated = float(count.sum()) / cells_per_stage
                    parts.append(f'updated {100.0 * updated:.1f}%')
                    payload[prefix + 'updated_fraction'] = updated
                    for c, v in enumerate(voxels):
                        payload[prefix + f'update_rms_vox_{component_names[c]}'] = v
                    if entry['scale'] is not None:
                        scale = float(entry['scale'][stage])
                        parts.append(f'shared scale {scale:.3e}')
                        payload[prefix + 'shared_scale'] = scale
                if clip is not None:
                    threshold, fraction = clip
                    threshold = float(threshold[stage])
                    fraction = float(fraction[stage])
                    parts.append(f'clip thr {threshold:.3e} clipped {100.0 * fraction:.3f}%')
                    payload[prefix + 'clip_threshold'] = threshold
                    payload[prefix + 'clipped_fraction'] = fraction
                lines.append(f'  flow cond {name} stage{stage}: ' + ', '.join(parts))
        return lines, payload

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

    def _ensure_sparse_volume_stores(self, *, use_normals, progress):
        """Have rank zero build missing derived stores before any rank loads."""
        # The resident pools are derived inputs. A normal single-process fit
        # builds any missing ones itself; in DDP, rank zero builds once and
        # publishes any failure before the other ranks try to open the pools.
        build_error = None
        if not self.dist.is_distributed or self.dist.is_main_process:
            try:
                ensure_fit_sparse_stores(
                    use_normals=use_normals,
                    use_spacing=self.grad_mag_spacing_enabled,
                    normal_nx_zarr_path=self.normal_nx_zarr_path,
                    normal_ny_zarr_path=self.normal_ny_zarr_path,
                    grad_mag_zarr_path=self.grad_mag_zarr_path,
                    normal_zarr_group=self.normal_zarr_group,
                    progress=progress,
                )
            except Exception as exc:
                build_error = exc
        if self.dist.is_distributed:
            error_message = [
                None if build_error is None else
                f'{type(build_error).__name__}: {build_error}'
            ]
            torch.distributed.broadcast_object_list(error_message, src=0)
            if error_message[0] is not None:
                if build_error is not None:
                    raise build_error
                raise RuntimeError(
                    'rank 0 could not build the sparse volume stores: '
                    f'{error_message[0]}')
        elif build_error is not None:
            raise build_error

    def _build_store_state(self):
        """Materialise the Lasagna brick pools.

        The expensive half of the device build, and the half nothing but the
        z window, the store paths and the dense-loss mode can invalidate.
        """
        interactive_driver = self.interactive_driver
        progress = progress_or_null(self.progress)

        # ==========================================================================
        # lasagna stores
        # ==========================================================================

        use_normals = (self.dense_normals_enabled
                       and self.config['loss_weight_dense_normals'] > 0)
        self._ensure_sparse_volume_stores(
            use_normals=use_normals, progress=progress)

        self.lasagna_volume = prepare_lasagna_volume(
            self.scroll_zarr,
            use_normals=use_normals,
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

        self._density_inactive_warned = set()

    # ==========================================================================
    # Pinned winding radii
    # ==========================================================================

    def _pin_footprint_rule(self):
        cfg = self.config
        return pins_module.FootprintRule(
            spacing_factor=float(cfg.get('model_pin_kernel_spacing_factor', 1.5)),
            min_arc_voxels=float(cfg.get('model_pin_kernel_min_arc_voxels', 3.0)),
            max_theta_radians=float(cfg.get('model_pin_kernel_max_theta_radians', 0.25)),
            min_z_voxels=float(cfg.get('model_pin_kernel_min_z_voxels', 3.0)),
            max_z_voxels=float(cfg.get('model_pin_kernel_max_z_voxels', 200.0)))

    def _finalize_pin_registry_fresh(self):
        """Resolve the pin graph against the current unpinned model (every
        patch pinned; no demotion)."""
        model = self.spiral_and_transform
        with torch.no_grad():
            intermediate = model.get_slice_to_intermediate_transform()
            dr = model.get_dr_per_winding()
            gap_expander, _, _, _ = model._get_transform_parts(with_pins=False)

            def free_gap(theta, z, slot):
                table = gap_expander.get_transformed_winding_radii(theta, z)
                gaps = table.diff(dim=-1)
                idx = slot.clamp(0, gaps.shape[-1] - 1)
                return torch.gather(gaps, -1, idx[:, None]).squeeze(-1)

            return pins_module.finalize_registry(
                self.pin_graph,
                intermediate_transform=intermediate, dr=dr,
                canonical_transform=model.get_unpinned_slice_to_spiral_transform(),
                crossing_map=self.theta_crossing_map, patch_atlas=self.patch_atlas,
                footprint_rule=self._pin_footprint_rule(), free_gap_fn=free_gap,
                min_z=float(self.flow_min_corner_spiral_zyx[0]),
                max_z=float(self.flow_max_corner_spiral_zyx[0]),
                device=self.device)

    def _finalize_pin_registry(self):
        """Resolve the pin graph against the current (unpinned) model.

        The theta topology has just been refreshed under
        self.slice_to_spiral_transform for a newly constructed registry.
        A matching checkpoint restores its registry together with T: rebuilding
        integer offsets in the current theta frame would change the meaning of
        the saved targets after a seam crossing. Otherwise the new registry
        supplies the initial target estimates.
        """
        model = self.spiral_and_transform
        started_at = time.perf_counter()
        self._pin_registry_full = None
        if self.pin_targets_loaded:
            registry = model.pin_registry
            if registry.patch_index is None:
                # Registry saved before per-pin patch indices existed: the
                # same graph emits pins in the same order, so borrow them.
                fresh = self._finalize_pin_registry_fresh()
                if fresh.num_pins == registry.num_pins and torch.equal(fresh.zyx, registry.zyx):
                    registry.patch_index = fresh.patch_index
                    self._pin_registry_full = fresh
                elif self.dist.is_main_process:
                    print('WARNING: could not backfill pin patch indices from the constraint graph; '
                          'pins fall back to uniform subsampling')
        else:
            registry = self._finalize_pin_registry_fresh()
            # Demotion (at activation and on re-checks) starts from this.
            self._pin_registry_full = registry
        if registry.patch_index is not None and registry.patch_component.numel():
            # A patch is pinned only if it contributed pins (older registries
            # marked every graph patch).
            has_pins = torch.zeros(registry.patch_component.numel(), dtype=torch.bool,
                                   device=registry.patch_component.device)
            pinned_idx = registry.patch_index[registry.patch_index >= 0]
            has_pins[pinned_idx[pinned_idx < has_pins.numel()]] = True
            registry.patch_component = torch.where(has_pins, registry.patch_component,
                                                   torch.full_like(registry.patch_component, -1))
        model.set_pin_registry(registry, reset_targets=not self.pin_targets_loaded)
        if self.dist.is_main_process:
            print(registry.summary())
            report = registry.consistency_report
            if report.get('inconsistent_edges'):
                print(f'WARNING: pin registry has {report["inconsistent_edges"]} '
                      f'inconsistent constraint cycles '
                      f'(by edge kind: {report.get("inconsistent_edges_by_kind")}):')
                for comp, entries in list(report['components'].items())[:20]:
                    print(f'  component {comp}: ' + '; '.join(
                        f'{e["edge"]} ({e["kind"]}, mismatch {e["mismatch"]})'
                        for e in entries[:5]))
            for conflict in report.get('absolute_conflicts', []):
                print(f'WARNING: absolute winding conflict in component '
                      f'{conflict["component"]}: {conflict}')
            print(f'pin registry ready ({_startup_resource_suffix(started_at)})')
        warmup = int(self.config.get('model_pins_warmup_steps', 0) or 0)
        self.pins_activation_iteration = warmup
        if self.pin_targets_loaded and self.start_iteration >= warmup:
            model.pins_active = True
            self._apply_integer_pin_targets()
            self.slice_to_spiral_transform = model.get_slice_to_spiral_transform()

    def _draw_step_pin_patches(self):
        """Draw this step's patch set for the pins and the patch losses."""
        num = len(self.verified_patches_list)
        count = min(num, max(int(self.config['sample_count_patches_per_step']),
                             int(self.config['sample_count_patches_per_step_for_dt'])))
        probs = self.patch_sampling_probabilities
        drawn = np.random.choice(num, count, replace=False, p=probs)
        return np.unique(drawn)

    def _restrict_patch_probabilities(self, patch_indices):
        """Patch sampling probabilities supported only on ``patch_indices``."""
        num = len(self.verified_patches_list)
        base = (np.ones(num, dtype=np.float64) / num if self.patch_sampling_probabilities is None
                else np.asarray(self.patch_sampling_probabilities, dtype=np.float64))
        mask = np.zeros(num, dtype=np.float64)
        mask[np.asarray(patch_indices.cpu() if torch.is_tensor(patch_indices) else patch_indices)] = 1.0
        probs = base * mask
        if probs.sum() <= 0:
            probs = mask
        return probs / probs.sum()

    def _apply_integer_pin_targets(self):
        """Honour ``model_pin_targets_integer`` on the live targets.

        On: round the component targets to integers and freeze them, so
        every component is pinned onto an integer winding (the integer-
        snapping satisfaction metric then measures pin exactness directly
        and DT has nothing to do). Off: targets are trainable. Called at
        activation, when a resumed checkpoint restores active pins, and when
        the setting changes at a run boundary; ``requires_grad`` is not part
        of the state dict, so a resume must re-apply it.
        """
        model = self.spiral_and_transform
        if getattr(model, 'pin_targets', None) is None:
            return
        if self.config.get('model_pin_targets_integer', False):
            with torch.no_grad():
                model.pin_targets.copy_(torch.round(model.pin_targets))
            model.pin_targets.requires_grad_(False)
        else:
            model.pin_targets.requires_grad_(True)

    def unpinned_verified_patch_mask(self):
        """Bool per verified patch: True when the registry pins none of its
        quads for a benign reason (outside the flow z domain, no sampling-
        valid quads). These keep the soft constraint losses while pins replace
        them on pinned inputs. Patches excluded by the spread filter are not
        included. None when pins are off."""
        model = self.spiral_and_transform
        if not model._pins_enabled() or model.pin_registry is None:
            return None
        registry = model.pin_registry
        mask = registry.patch_component < 0
        if mask.numel() != len(self.verified_patches_list):
            return None
        excluded = registry.excluded_patches
        if excluded is not None and excluded.numel():
            # Spread-excluded patches are left out on purpose: no soft loss
            # either, or they would drag the flow toward sheets they do not
            # consistently describe.
            mask = mask.clone()
            mask[excluded[excluded < mask.numel()].to(mask.device)] = False
        return mask

    def _full_pin_registry(self):
        """The registry with every patch pinned (before any demotion). Kept
        from finalisation; a resumed checkpoint carries only the demoted
        registry, so it is rebuilt from the constraint graph on demand."""
        full = getattr(self, '_pin_registry_full', None)
        if full is None:
            full = self._finalize_pin_registry_fresh()
            self._pin_registry_full = full
        return full

    def _pair_agreement_pairs(self):
        """``(pairs, zyx)`` for the pair-agreement loss: cross-component
        patch-pin index pairs within ``loss_pair_agreement_tolerance_voxels``
        (pins.cross_component_pairs) over the full registry's points. Built
        once per model state (the pairs are geometric, independent of the
        model and of demotion) and cached; the registry itself is not kept.
        """
        cache = getattr(self, '_pair_agreement_cache', None)
        if cache is not None:
            return cache
        started_at = time.perf_counter()
        full = self._full_pin_registry()
        pairs = pins_module.cross_component_pairs(
            full.zyx, full.patch_index, full.component,
            tolerance=float(self.config['loss_pair_agreement_tolerance_voxels']),
            stride=int(self.config['loss_pair_agreement_stride']),
            max_pairs=int(self.config['loss_pair_agreement_max_pairs']),
            seed=int(self.config['optimizer_random_seed']))
        zyx = full.zyx.detach().clone()
        if self.dist.is_main_process:
            print(f'pair agreement: {pairs.shape[0]} cross-component pin pairs within '
                  f"{float(self.config['loss_pair_agreement_tolerance_voxels']):g} voxels "
                  f"(every {int(self.config['loss_pair_agreement_stride'])}th of {full.num_pins} pins, "
                  f"cap {int(self.config['loss_pair_agreement_max_pairs'])}; "
                  f'{time.perf_counter() - started_at:.1f}s)')
        self._pair_agreement_cache = (pairs, zyx)
        return self._pair_agreement_cache

    def _demote_conflicting_patches(self, iteration=None):
        """Leave unpinned the verified patches that contradict their
        neighbours (``model_pin_demote_conflicting_patches``); see
        pins.conflicting_patch_demotion. Runs at activation and, every
        ``PIN_DEMOTE_RECHECK_INTERVAL`` steps, again on the current free
        map starting from the full registry, so patches that no longer
        conflict are pinned again. The demoted set is recorded in the
        registry (a resumed checkpoint keeps it) and such patches get no soft
        constraint loss; each decision is appended to pin_demotion.jsonl.
        """
        if not self.config.get('model_pin_demote_conflicting_patches', False):
            return
        model = self.spiral_and_transform
        full = self._full_pin_registry()
        if full is None or full.patch_index is None or not full.num_pins:
            return
        current = model.pin_registry
        model.set_pin_registry(full, reset_targets=False)
        try:
            with torch.no_grad():
                T = model.effective_pin_targets()
                _, estimate = model.estimate_pin_targets(return_per_pin=True)
                pins_t = model.compute_pins(full=True)
                n = pins_t[:, 3] / model.get_dr_per_winding() - T[full.component]
            demoted, report = pins_module.conflicting_patch_demotion(
                full.zyx, full.patch_index, full.component, torch.round(n), estimate, T,
                tolerance=float(self.config.get('model_pin_demote_pair_tolerance_voxels', 30.0)),
                min_conflict_fraction=float(self.config.get('model_pin_demote_conflict_fraction', 0.05)),
                min_conflict_ratio=float(self.config.get('model_pin_demote_conflict_ratio', 0.0)))
        except Exception:
            model.set_pin_registry(current, reset_targets=False)
            raise
        previous = set(current.excluded_patches.tolist()) if current.excluded_patches is not None else set()
        now = set(int(p) for p in demoted)
        if len(demoted):
            demoted_t = torch.as_tensor(demoted, dtype=torch.int64, device=full.zyx.device)
            keep = ~torch.isin(full.patch_index, demoted_t)
            model.set_pin_registry(full.subset(keep, demoted_patches=demoted_t), reset_targets=False)
        # (No demotion: the full registry stays installed.)
        if self.dist.is_main_process:
            print(f'pin conflicts (step {iteration if iteration is not None else "activation"}): '
                  f'{report["inconsistent"]} of {report["pairs"]} cross-component pin pairs inconsistent; '
                  f'{len(demoted)} patch(es) demoted ({len(now - previous)} new, {len(previous - now)} reinstated)')
            names = [getattr(patch, 'uuid', None) for patch in self.verified_patches_list]
            entry = {
                'iteration': iteration, 'pairs': report['pairs'], 'inconsistent': report['inconsistent'],
                'demoted': [{**e, 'id': names[e['patch']] if e['patch'] < len(names) else None,
                             'neighbours': [{'patch': q, 'id': names[q] if q < len(names) else None, 'conflicting_pairs': c}
                                            for q, c in e['neighbours']]} for e in report['demoted']],
                'reinstated': sorted(previous - now),
            }
            out_path = getattr(self, 'out_path', None)
            if out_path:
                with open(os.path.join(out_path, 'pin_demotion.jsonl'), 'a') as handle:
                    handle.write(json.dumps(entry) + '\n')

    def _maybe_activate_pins(self, iteration):
        model = self.spiral_and_transform
        if (model.pin_registry is None or model.pins_active
                or self.pins_activation_iteration is None
                or iteration < self.pins_activation_iteration):
            return
        if not self.pin_targets_loaded:
            # T from the state the soft fit has reached.
            with torch.no_grad():
                model.pin_targets.copy_(model.estimate_pin_targets())
        self._apply_integer_pin_targets()
        self._demote_conflicting_patches(iteration)
        model.pins_active = True
        if self.dist.is_main_process:
            T = model.effective_pin_targets().detach()
            frac = (T - torch.round(T)).abs()
            print(f'pins activated at iteration {iteration}: {model.pin_registry.num_pins} pins, '
                  f'{model.pin_registry.num_components} components, '
                  f'|T - round(T)| mean {float(frac.mean()) if frac.numel() else 0.0:.3f}')
        # Whole-object DT targets now come from T; drop the cached medians.
        self.dt_target_cache_manager.reset()

    def _pinned_patch_dt_values(self):
        """Per verified patch, ``T_g + O_P`` (nan for unpinned patches)."""
        model = self.spiral_and_transform
        if not model._pins_enabled():
            return None
        registry = model.pin_registry
        T = model.effective_pin_targets().detach()
        values = torch.full([len(self.verified_patches_list)], float('nan'), device=self.device)
        has = registry.patch_component >= 0
        values[has] = T[registry.patch_component[has]] + registry.patch_offset[has].to(T.dtype)
        return values

    def _record_pin_target_grad(self, family, pins_leaf, seen_before):
        grad = pins_leaf.grad
        if grad is None:
            return
        registry = self.spiral_and_transform.pin_registry
        # The pins leaf holds the step's sampled subset; its component ids
        # live in the model's current pin view.
        component = self.spiral_and_transform._pin_view['component']
        increment = grad[:, 3] if seen_before is None else grad[:, 3] - seen_before[:, 3]
        dr = float(self.dr_per_winding.detach())
        per_component = torch.zeros(
            [registry.num_components], device=grad.device).index_add(
            0, component, increment) * dr
        self.pin_grad_by_family[family] = per_component

    def _pin_step_metrics(self, pins):
        """Guard/conflict/exactness diagnostics at the pins' own rays."""
        model = self.spiral_and_transform
        metrics = {}
        gap = pinned_gap_stage(self.slice_to_spiral_transform)
        if gap is not None:
            metrics.update(gap.pin_diagnostics(pins))
        metrics['pin_conflicts'] = float(len(model.pin_conflicts))
        metrics['pin_rebuilds'] = float(model._pin_rebuilds)
        T = model.effective_pin_targets().detach()
        if T.numel():
            frac = (T - torch.round(T)).abs()
            metrics['pin_T_frac_mean'] = float(frac.mean())
            metrics['pin_T_frac_max'] = float(frac.max())
        for family, grad in self.pin_grad_by_family.items():
            metrics[f'pin_T_grad_norm/{family}'] = float(grad.norm())
        self.pin_grad_by_family = {}
        return metrics

    def _export_transform(self):
        """The live transform for export; pinned with the full registry."""
        transform = self.spiral_and_transform.get_slice_to_spiral_transform()
        model = self.spiral_and_transform
        if model._pins_enabled():
            gap = pinned_gap_stage(transform)
            assert gap is not None, 'pins are active but the export transform is unpinned'
            assert gap.pin_table.num_registry_pins == model.pin_registry.num_pins, (
                'export transform must carry the full pin registry')
        return transform

    def _adapt_checkpoint_for_pins(self, checkpoint, model_state, optimiser_state):
        """Fit a checkpoint written with a different (or no) pin registry.

        ``pin_targets`` is kept only when the checkpoint's registry
        fingerprint matches this fit's constraint graph; otherwise it is
        dropped and re-estimated from the loaded model. A checkpoint from
        before pins existed gets an appended optimiser/scheduler group for T.
        """
        scheduler_state = checkpoint.get('scheduler')
        self.pin_targets_loaded = False
        if getattr(self.spiral_and_transform, 'pin_targets', None) is None:
            model_state = {k: v for k, v in model_state.items() if k != 'pin_targets'}
            return model_state, optimiser_state, scheduler_state
        model_state = dict(model_state)
        saved_T = model_state.get('pin_targets')
        saved_registry = checkpoint.get('pin_registry')
        fingerprint = saved_registry.get('fingerprint') if isinstance(saved_registry, Mapping) else None
        live = self.spiral_and_transform.pin_targets
        # T carries information only if the checkpoint was written with pins
        # active: before activation T has no gradient and still holds the
        # registry's estimate from the model state at construction, which is
        # stale once the soft fit has moved on. Such a checkpoint keeps its
        # registry but re-estimates T at activation (_maybe_activate_pins).
        # (Checkpoints predating the flag are taken as active: legacy behaviour.)
        saved_active = bool(checkpoint.get('pins_active', True))
        if (saved_T is not None and tuple(saved_T.shape) == tuple(live.shape)
                and fingerprint == self.pin_graph.fingerprint()):
            self.spiral_and_transform.set_pin_registry(
                pins_module.PinRegistry.from_state_dict(saved_registry, live.device),
                reset_targets=not saved_active)
            self.pin_targets_loaded = saved_active
            if not saved_active:
                model_state['pin_targets'] = self.spiral_and_transform.pin_targets.detach().clone().cpu()
        else:
            if saved_T is not None:
                print('checkpoint pin_targets do not match this fit\'s constraint '
                      'graph; re-estimating T from the loaded model')
            model_state['pin_targets'] = live.detach().clone().cpu()
            # Adam's moments belong to the old component identities, even
            # when the new target tensor happens to have the same shape.
            # Preserve every other parameter's state and the source archive.
            if (isinstance(optimiser_state, Mapping)
                    and not _checkpoint_lacks_pin_group(self, optimiser_state)):
                groups = optimiser_state.get('param_groups') or []
                if len(groups) == len(self.optimiser.param_groups):
                    pin_ids = set(groups[-1].get('params', []))
                    optimiser_state = dict(optimiser_state)
                    optimiser_state['state'] = {
                        key: value for key, value in optimiser_state.get('state', {}).items()
                        if key not in pin_ids}
        if _checkpoint_lacks_pin_group(self, optimiser_state):
            optimiser_state = dict(optimiser_state)
            groups = [dict(g) for g in optimiser_state.get('param_groups') or []]
            next_index = 1 + max(
                (idx for g in groups for idx in g.get('params', [])), default=-1)
            live_group = dict(self.optimiser.param_groups[-1])
            live_group.pop('params', None)
            groups.append({**live_group, 'params': [next_index]})
            optimiser_state['param_groups'] = groups
            if isinstance(scheduler_state, Mapping):
                scheduler_state = dict(scheduler_state)
                base = list(scheduler_state.get('base_lrs') or [])
                base.append(self.lr_scheduler.base_lrs[-1])
                scheduler_state['base_lrs'] = base
                last = list(scheduler_state.get('_last_lr') or [])
                if last:
                    last.append(live_group['lr'])
                    scheduler_state['_last_lr'] = last
        return model_state, optimiser_state, scheduler_state

    def _make_theta_crossing_map(self):
        """Construct the shared patch/PCL source topology."""
        crossing_map = ThetaCrossingMap(
            self.device,
            self.config['theta_crossing_map_update_interval'])
        self.patch_atlas.register_theta_topology(crossing_map)

        flat = get_or_build_unattached_pcl_flat(
            self.unattached_pcl_strips, self.device)
        if flat is not None:
            start = crossing_map.register_nodes(
                flat['total'],
                lambda lo, hi, points=flat['zyxs']: points[lo:hi])
            starts = flat['starts_cpu'].numpy()
            for strip_idx, strip in enumerate(self.unattached_pcl_strips):
                node_ids = start + np.arange(
                    starts[strip_idx], starts[strip_idx + 1], dtype=np.int64)
                strip['_theta_node_ids'] = node_ids
                if len(node_ids) > 1:
                    crossing_map.register_edges(
                        np.stack([node_ids[:-1], node_ids[1:]], axis=1))
            for edges in self.unattached_component_edges:
                junctions = []
                for strip_a, pos_a, strip_b, pos_b in edges:
                    junctions.append((
                        self.unattached_pcl_strips[strip_a]['_theta_node_ids'][pos_a],
                        self.unattached_pcl_strips[strip_b]['_theta_node_ids'][pos_b]))
                if junctions:
                    crossing_map.register_edges(junctions)

        points = []
        seen = set()
        for pcl in self.cross_patch_pcls:
            for point in pcl['points'].values():
                if id(point) not in seen:
                    seen.add(id(point))
                    points.append(point)
        if points:
            point_zyxs = torch.as_tensor(
                np.stack([p['zyx'] for p in points]).astype(np.float32),
                device=self.device)
            start = crossing_map.register_nodes(
                len(points), lambda lo, hi, values=point_zyxs: values[lo:hi])
            for local, point in enumerate(points):
                point['_theta_node_id'] = start + local
            for pcl in self.cross_patch_pcls:
                chain = list(pcl['chain'].iter_chain())
                if len(chain) > 1:
                    ids = np.fromiter(
                        (p['_theta_node_id'] for p in chain), dtype=np.int64)
                    crossing_map.register_edges(
                        np.stack([ids[:-1], ids[1:]], axis=1))

        return crossing_map

    def _patch_input_path(self, patch_id, patch, root):
        source_path = getattr(patch, '_source_path', None)
        if source_path is None:
            source_path = os.path.join(root, patch_id) if root else str(patch_id)
        return os.path.abspath(source_path)

    def _write_non_liftable_patch_report(self):
        """Atomically publish the cumulative rejected-patch path list."""
        if getattr(self, '_preparing_inputs', False):
            return
        if not self.dist.is_main_process or not hasattr(self, 'out_path'):
            return
        report_path = os.path.join(self.out_path, 'non_liftable_patches.txt')
        temporary_path = f'{report_path}.tmp-{os.getpid()}'
        with open(temporary_path, 'w', encoding='utf-8') as report:
            for path in sorted(self.non_liftable_patch_paths):
                report.write(f'{path}\n')
        os.replace(temporary_path, report_path)

    def _trusted_geometry_from_active_inputs(self):
        """Rebuild retained trusted geometry after rejecting verified patches."""
        pieces = []
        for patch in self.verified_patches_list:
            points = patch.zyxs.reshape(-1, 3).to(dtype=torch.float32).cpu()
            valid = patch.valid_vertex_mask.reshape(-1).cpu()
            in_roi = (points[:, 0] >= self.z_begin) & (points[:, 0] < self.z_end)
            if bool((valid & in_roi).any()):
                pieces.append(points[valid & in_roi])
        for strip in self.unattached_pcl_strips:
            points = torch.from_numpy(strip['zyxs']).to(dtype=torch.float32)
            in_roi = (points[:, 0] >= self.z_begin) & (points[:, 0] < self.z_end)
            if bool(in_roi.any()):
                pieces.append(points[in_roi])
        return torch.cat(pieces, dim=0) if pieces else torch.empty((0, 3))

    def _exclude_non_liftable_patches(self, verified_ids, report):
        """Remove inconsistent patches from every active patch sampling pool."""
        warnings = []
        rejected_verified = set(verified_ids)

        for patch_id in verified_ids:
            patch = self.verified_patches[patch_id]
            path = self._patch_input_path(
                patch_id, patch, self.verified_patches_path)
            self.non_liftable_patch_paths.add(path)
            warning = (
                f'non-liftable patch {path!r} has theta cycle inconsistencies; '
                'excluding it from this fit')
            print(f'WARNING: {warning}')
            warnings.append(warning)
            del self.verified_patches[patch_id]

        # Preserve list identities where possible because resident-session
        # loss closures may already hold them.
        self.verified_patches_list[:] = self.verified_patches.values()
        self.patch_sampling_probabilities = (
            self._patch_sampling_probabilities(self.verified_patches_list)
            if self.verified_patches_list else np.empty(0, dtype=np.float32))
        self.num_verified_patches = len(self.verified_patches_list)
        self.patch_atlas = self.patch_atlas.replaced(self.verified_patches)

        if rejected_verified and self.cross_patch_pcls:
            for pcl in self.cross_patch_pcls:
                points_by_patch = pcl.get('points_by_patch', {})
                for patch_id in rejected_verified:
                    points_by_patch.pop(patch_id, None)
            self._rebuild_pcl_sampling_strata()
        if rejected_verified:
            # Rejected attachments become eligible for a later live patch.
            for catalog in (self.regular_pcl_catalog, self.fiber_catalog):
                for pcl in catalog.values():
                    for point in pcl['points'].values():
                        attachment = point.get('on_patch')
                        if (attachment is not None and
                                attachment['id'] in rejected_verified):
                            del point['on_patch']

        if rejected_verified and self.interactive_driver is not None:
            # Track exclusion masks prepared later must not retain geometry
            # from a patch that the consistency gate rejected.
            self._refresh_trusted_geometry()

        if hasattr(self, 'dt_target_cache_manager'):
            self.dt_target_cache_manager.reset()
        self._write_non_liftable_patch_report()
        print(
            'WARNING: theta consistency gate rejected '
            f'{len(verified_ids)} patch(es) from '
            f'{report["inconsistent_edges"]} inconsistent edge(s)')
        return warnings

    def _enforce_theta_liftability(self):
        """Reject patches whose cached tree potential is path-dependent."""
        warnings = []
        while True:
            report, bad_nodes = \
                self.theta_crossing_map.potential_inconsistencies()
            if report['inconsistent_edges'] == 0:
                self._write_non_liftable_patch_report()
                return warnings
            verified_ids = self.patch_atlas.patch_ids_for_theta_nodes(bad_nodes)
            if not verified_ids:
                raise RuntimeError(
                    'theta consistency gate found inconsistent potential edges '
                    'that could not be attributed to a patch')
            warnings.extend(self._exclude_non_liftable_patches(
                verified_ids, report))
            self.theta_crossing_map = self._make_theta_crossing_map()
            self.theta_crossing_map.force_refresh(
                self.slice_to_spiral_transform)

    def _build_theta_crossing_map(self):
        """Rebuild, refresh, and validate shared patch/PCL theta topology."""
        started_at = time.perf_counter()
        self.theta_crossing_map = self._make_theta_crossing_map()
        self.theta_crossing_map.force_refresh(self.slice_to_spiral_transform)
        warnings = self._enforce_theta_liftability()
        print('theta topology ready '
              f'({_startup_resource_suffix(started_at)})')
        return warnings

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

        self.device = torch.device('cuda')
        progress.begin(
            'loading', 'Building verified-patch GPU atlas',
            detail=f'{len(self.verified_patches):,} patches')
        self.patch_atlas.materialize(self.device)

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
        self.start_iteration = 0
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

        # Pinned winding radii. The constraint
        # graph is model-independent, so its component count -- the size of
        # the pin_targets parameter T -- is known before the optimiser exists;
        # the registry itself (integer offsets, footprints) is finalised
        # against the theta topology below, and the pins are switched on at
        # model_pins_warmup_steps (see _maybe_activate_pins).
        self.pin_graph = None
        self.pin_targets_loaded = False
        self.pins_activation_iteration = None
        self.pin_grad_by_family = {}
        self._pair_agreement_cache = None
        if self.config.get('model_pins_enabled', False):
            progress.begin('loading', 'Building pin constraint graph')
            overlap_tolerance = float(self.config.get('model_pin_overlap_tolerance_voxels', 0.0) or 0.0)
            overlap_pairs = None
            if overlap_tolerance > 0:
                started_at = time.perf_counter()
                overlap_pairs = pins_module.patch_overlap_pairs(
                    self.patch_atlas, overlap_tolerance,
                    stride=int(self.config.get('model_pin_patch_grid_stride', 1)))
                if self.dist.is_main_process:
                    linked = len({(a, b) for a, _, b, _, _ in overlap_pairs})
                    print(f'pin overlaps: {linked} patch pairs within {overlap_tolerance:g} voxels '
                          f'({_startup_resource_suffix(started_at)})')
            self.pin_graph = pins_module.build_pin_graph(
                verified_patches=self.verified_patches,
                patch_atlas=self.patch_atlas,
                cross_patch_pcls=self.cross_patch_pcls,
                unattached_pcl_strips=self.unattached_pcl_strips,
                unattached_components=self.unattached_components,
                unattached_component_edges=self.unattached_component_edges,
                patch_grid_stride=self.config.get('model_pin_patch_grid_stride', 1),
                overlap_pairs=overlap_pairs,
                z_range=(float(self.flow_min_corner_spiral_zyx[0]),
                         float(self.flow_max_corner_spiral_zyx[0])),
            )
            print(self.pin_graph.summary())
            self.spiral_and_transform.init_pin_targets(self.pin_graph.num_components)

        # ==========================================================================
        # Outer-shell setup
        # ==========================================================================

        self.shell_map = None

        shell_active = self.shell_patch is not None and self.shell_losses_enabled()
        if (self.shell_patch is not None
                and (self.config['loss_weight_shell_outer'] > 0
                     or self.winding_model_mode)):
            self.shell_map = self._make_shell_polar_map()

        # Dense losses sample out to this index even when shell losses are off.
        self.shell_outer_winding_idx, outer_winding_notes = resolve_outer_winding_idx_and_notes(
            self.config, shell_active, self._infer_outer_winding_idx_for_this_run)
        for note in outer_winding_notes:
            print(note)

        self._dense_inactive_warned = set()

        # ==========================================================================
        # Optimizer and checkpoint helpers
        # ==========================================================================

        # Keep the low- and high-resolution lattices (every flow stage is a
        # slab of each) in distinct groups so the HR learning-rate scale is an
        # optimizer setting, not a multiplier in the model's forward path.
        self.low_res_flow_params = [self.spiral_and_transform.flow_field.flows[0]]
        self.high_res_flow_params = [self.spiral_and_transform.flow_field.flows[1]]
        flow_field_params = self.low_res_flow_params + self.high_res_flow_params
        self.gap_expander_params = list(self.spiral_and_transform.gap_expander_params.parameters())
        linear_params = [self.spiral_and_transform.linear_logits]
        pin_target_params = (
            [self.spiral_and_transform.pin_targets]
            if self.spiral_and_transform.pin_targets is not None else [])
        grouped_ids = {id(p) for p in flow_field_params + self.gap_expander_params + linear_params + pin_target_params}
        other_params = [p for p in self.spiral_and_transform.parameters() if id(p) not in grouped_ids]
        initial_high_res_lr_scale = get_flow_field_high_res_lr_scale(self.config, 0)
        initial_low_res_lr_scale = get_flow_field_low_res_lr_scale(self.config)
        param_groups = [
            {'params': other_params, 'weight_decay': 0.0},
            {'params': linear_params, 'weight_decay': 0.0},
            {'params': self.gap_expander_params, 'weight_decay': self.config['optimizer_weight_decay_gap_expander']},
            {
                'params': self.low_res_flow_params,
                'weight_decay': self.config['optimizer_weight_decay_flow_field'],
                'lr': self.config['optimizer_learning_rate'] * initial_low_res_lr_scale,
                'lr_scale': initial_low_res_lr_scale,
            },
            {
                'params': self.high_res_flow_params,
                'weight_decay': self.config['optimizer_weight_decay_flow_field'],
                'lr': self.config['optimizer_learning_rate'] * initial_high_res_lr_scale,
                'lr_scale': initial_high_res_lr_scale,
            },
        ]
        if pin_target_params:
            # T has its own group: one Adam step moves a component by at most
            # about optimizer_lr_pin_targets windings. Expressed as
            # a scale of the base LR so schedule realignment preserves it.
            pin_lr = float(self.config.get('optimizer_lr_pin_targets', 0.01))
            pin_lr_scale = pin_lr / float(self.config['optimizer_learning_rate'])
            param_groups.append({
                'params': pin_target_params,
                'weight_decay': 0.0,
                'lr': pin_lr,
                'lr_scale': pin_lr_scale,
            })
        self.pin_target_params = pin_target_params
        progress.begin('loading', 'Creating optimizer')
        # AdamW for every group; the flow groups may additionally be stepped
        # with lazy moments (optimizer_flow_lazy_moments, applied per step by
        # _apply_flow_group_settings), which keeps AdamW's state format.
        self.optimiser = LazyMomentAdamW(param_groups, lr=self.config['optimizer_learning_rate'], betas=(0.9, 0.999), eps=1.e-8, fused=True)
        self._report_flow_grad_conditioning()
        if self.config['optimizer_exp_lr_schedule']:
            gamma = self.config['optimizer_lr_final_factor'] ** (1.0 / max(1, self.num_training_steps))
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser, gamma=gamma)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, lambda step: 1.)

        if resume_path:
            # Phase 1 of the same two-phase load an in-session checkpoint
            # request uses: inspect on the CPU and refuse before anything
            # resident is touched.
            verdict = self.inspect_checkpoint(
                resume_checkpoint, source=resume_path)
            if not verdict.accepted:
                raise RuntimeError(verdict.message())
            print(f'resuming from {resume_path} at iteration '
                  f'{verdict.completed_iterations}')
            progress.begin(
                'loading', 'Restoring model and optimizer',
                detail=os.path.basename(resume_path))
            # Phase 2: apply. The LR realignment for a resident session is the
            # unconditional one below, so it is not requested twice here.
            self.apply_checkpoint(resume_checkpoint)
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
                progress=self.progress,
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
        self._build_theta_crossing_map()
        if self.pin_graph is not None:
            progress.begin('loading', 'Finalising pin registry')
            self._finalize_pin_registry()

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
            self.config['theta_crossing_map_update_interval'],
            report_first_dt_target_cache,
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
        # Per-lattice, per-stage (threshold, clipped fraction) of the last
        # step's robust flow-gradient clip, for the step log (see
        # _clip_flow_grads).
        self.flow_grad_clip_stats = {}
        self.run_dt_resume_iteration = None

    # What _build_model_state() constructs, and therefore what
    # rebuild_model_state() releases before running it again. Written out
    # rather than derived, so device state added to the model stage without a
    # decision about its release is a visible omission instead of a leak.
    # prepared_main_tracks/preview_extent_tracks are deliberately absent: see
    # the retained_tracks note in _build_model_state.
    _MODEL_STAGE_ATTRIBUTES = (
        'device', 'umbilicus_zyx',
        'start_iteration', 'model_z_begin', 'model_z_end',
        'flow_field_radius', 'flow_min_corner_spiral_zyx',
        'flow_max_corner_spiral_zyx', 'num_training_steps',
        '_initial_num_training_steps', 'spiral_and_transform', 'shell_map',
        'shell_outer_winding_idx',
        '_dense_inactive_warned', 'low_res_flow_params',
        'high_res_flow_params', 'gap_expander_params', 'optimiser',
        'lr_scheduler', 'profiler',
        'slice_to_spiral_transform', 'dr_per_winding',
        'dt_target_cache_manager', 'dist_grad_params', 'dist_grad_named',
        'step_timer', 'nonfinite_grad_steps', 'nonfinite_grad_by_param',
        'run_dt_resume_iteration', 'pin_graph', 'pin_target_params',
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

    def model_state_digest(self):
        """Content identity of the surface the resident model places.

        Stamped into every checkpoint and every raw preview manifest. Two
        artifacts carrying the same digest were produced from byte-identical
        live parameters, frozen-epoch stack and run window, so the host may
        re-show the flattened preview it already holds for a checkpoint it is
        asked to load instead of flattening the same surface again.
        """
        return model_state_sha256(
            self.spiral_and_transform.state_dict(),
            getattr(self, 'frozen_epochs', None) or (),
            getattr(self, 'z_begin', None), getattr(self, 'z_end', None))

    def _checkpoint_payload(self, completed_iterations):
        return {
            'schema_version': 2,
            'model_state_sha256': self.model_state_digest(),
            'completed_iterations': int(completed_iterations),
            'spiral_and_transform': self.spiral_and_transform.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'cfg': dict(self.config),
            'requested_config': dict(
                getattr(self.interactive_driver, 'requested_config', dict(self.config))),
            'resolved_config': dict(self.config),
            'lasagna_scale': self.lasagna_scale,
            'lasagna_group': self.normal_zarr_group,
            'base_shape_zyx': (
                list(self.base_shape_zyx)
                if self.base_shape_zyx is not None else None),
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
            # Pinned winding radii: the registry lets an offline exporter
            # rebuild the exact pinned transform without the fit inputs.
            'pin_registry': (
                self.spiral_and_transform.pin_registry.state_dict()
                if self.spiral_and_transform.pin_registry is not None else None),
            'pins_active': bool(self.spiral_and_transform.pins_active),
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

    def _maybe_save_headless_checkpoint(self, completed_iterations):
        """Refresh the final checkpoint path at each headless autosave boundary.

        The completed run writes this same path once more, so an interrupted
        fit leaves one resumable checkpoint while a successful fit leaves only
        the conventional final checkpoint.
        """
        if (completed_iterations % _HEADLESS_AUTOSAVE_INTERVAL != 0
                or completed_iterations >= self.num_training_steps
                or not self.dist.is_main_process):
            return None
        return self._save_model('fitted', completed_iterations)

    def inspect_checkpoint(self, checkpoint, *, source=''):
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
        """
        reasons = []
        if not isinstance(checkpoint, dict):
            return CheckpointVerdict(
                False, ('checkpoint is not a state dictionary',), source=source)
        try:
            # The stored configuration is brought onto the current schema
            # first (dropped and defaulted keys), and each edit is
            # reported; only an invalid value refuses. Capacity growth is
            # prefix-preserving: inspect the expanded copy without mutating
            # the caller's checkpoint. load_checkpoint() performs the same
            # normalisation and migration only after this verdict succeeds.
            checkpoint, notes = tolerate_checkpoint_config(checkpoint)
            for note in notes:
                print(f'NOTE: checkpoint {note}', flush=True)
            checkpoint = expand_gap_checkpoint_capacity(
                checkpoint, self.config['model_gap_expander_capacity_windings'])
        except ValueError as exc:
            reasons.append(str(exc))

        # --- schema -------------------------------------------------------
        schema_version = int(checkpoint.get('schema_version', 1) or 1)
        if schema_version < 2:
            reasons.append(
                f'checkpoint schema version {schema_version} predates the '
                'identity fields a load verifies (expected >= 2)')
        for key in ('spiral_and_transform', 'optimiser', 'cfg'):
            if checkpoint.get(key) is None:
                reasons.append(f'checkpoint has no {key!r} entry')

        # --- scroll / dataset identity ------------------------------------
        if checkpoint.get('lasagna_scale') != self.lasagna_scale:
            reasons.append(
                f'checkpoint lasagna_scale={checkpoint.get("lasagna_scale")!r} '
                f'does not match this fit ({self.lasagna_scale!r})')
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
        checkpoint_base_shape = checkpoint.get('base_shape_zyx')
        if checkpoint_base_shape is not None:
            if (not isinstance(checkpoint_base_shape, (list, tuple))
                    or len(checkpoint_base_shape) != 3
                    or any(type(value) is not int or value <= 0
                           for value in checkpoint_base_shape)):
                reasons.append(
                    'checkpoint base_shape_zyx is not a ZYX list of three positive integers')
            else:
                checkpoint_base_shape = tuple(checkpoint_base_shape)
                if self.base_shape_zyx is None:
                    reasons.append(
                        'checkpoint declares base_shape_zyx but this dataset scroll spec does not')
                elif checkpoint_base_shape != tuple(self.base_shape_zyx):
                    reasons.append(
                        f'checkpoint base_shape_zyx={list(checkpoint_base_shape)!r} '
                        f'does not match this dataset ({list(self.base_shape_zyx)!r})')
        checkpoint_dataset = str(
            (checkpoint.get('input_manifest') or {}).get('dataset_root') or '')
        dataset_root = str(getattr(self.paths, 'dataset_root', '') or '')
        if checkpoint_dataset and dataset_root and checkpoint_dataset != dataset_root:
            reasons.append(
                f'checkpoint was written against dataset {checkpoint_dataset!r}, '
                f'not {dataset_root!r}')

        # --- Winding-inference store identity ------------------------------
        if self.winding_model_mode:
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
            # tolerate_checkpoint_config() has already brought the key set
            # onto the schema, so a mismatch here is a normalisation bug.
            schema = set(self.config)
            unknown = set(checkpoint_cfg) - schema
            missing = schema - set(checkpoint_cfg)
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
            steps_key = 'model_num_flow_integration_steps'
            if (steps_key in checkpoint_cfg
                    and checkpoint_cfg[steps_key] != self.config[steps_key]):
                print(
                    f'NOTE: checkpoint was written with {steps_key}='
                    f'{checkpoint_cfg[steps_key]!r}; resuming with the live '
                    f'{self.config[steps_key]!r} perturbs the transform by '
                    'the integration discretisation difference')
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
        else:
            reasons.append(
                'checkpoint does not record a model z-domain, so it cannot be '
                'shown to match the live model')

        # --- model keys and tensor geometry --------------------------------
        model_state = checkpoint.get('spiral_and_transform')
        if isinstance(model_state, Mapping):
            live_state = self.spiral_and_transform.state_dict()
            # The pin_targets parameter is sized by the constraint registry,
            # which is rebuilt from the inputs: a mismatch is re-initialised
            # from the loaded model, not refused (see load_checkpoint).
            model_state = {k: v for k, v in model_state.items() if k != 'pin_targets'}
            live_state = {k: v for k, v in live_state.items() if k != 'pin_targets'}
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
            if _checkpoint_lacks_pin_group(self, optimiser_state):
                # A checkpoint written before pins existed: the pin group is
                # appended at load (fresh Adam state for T).
                live_groups = live_groups[:-1]
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
                    if _checkpoint_lacks_pin_group(self, checkpoint.get('optimiser')):
                        live_base = live_base[:-1]
                    if len(saved_base) != len(live_base):
                        reasons.append(
                            f'checkpoint scheduler tracks {len(saved_base)} '
                            f'parameter groups, this fit has {len(live_base)}')

        completed = checkpoint.get('completed_iterations')
        if completed is None:
            reasons.append('checkpoint has no completed_iterations entry')
        return CheckpointVerdict(
            not reasons, tuple(reasons),
            completed_iterations=(None if completed is None else int(completed)),
            source=source)

    def apply_checkpoint(self, checkpoint, *, realign_lr=False):
        """Phase 2: move a preflighted checkpoint into the live fit.

        Only ever called after :meth:`inspect_checkpoint` accepted this exact
        payload on every participating rank. It mutates model, optimiser,
        scheduler, RNG and iteration state; a failure here leaves a partially
        loaded optimiser, which is why callers treat it as fatal to the
        session rather than as a refusal.

        ``completed_iterations`` comes from the checkpoint - the durable step
        the fit actually reached (every checkpoint carries it; the preflight
        refuses one without) - and the LR schedule is realigned to it rather
        than reset to zero.
        """
        completed = int(checkpoint['completed_iterations'])
        self.start_iteration = completed
        self.load_checkpoint(checkpoint)
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
        checkpoint, _ = tolerate_checkpoint_config(checkpoint)
        checkpoint = expand_gap_checkpoint_capacity(
            checkpoint, self.config['model_gap_expander_capacity_windings'])
        transformed_spiral_state, optimiser_state = checkpoint['spiral_and_transform'], checkpoint['optimiser']
        transformed_spiral_state, optimiser_state, scheduler_state = \
            self._adapt_checkpoint_for_pins(checkpoint, transformed_spiral_state, optimiser_state)
        self.spiral_and_transform.load_state_dict(transformed_spiral_state)
        self.optimiser.load_state_dict(optimiser_state)
        # Stored optimiser hyperparameters yield to the session configuration.
        # The flow groups are reapplied at every run boundary
        # (_apply_flow_group_settings) and the learning rates by the schedule
        # realignment; the gap group's weight decay has no later owner, so a
        # checkpoint saved while it was altered must not carry that forward.
        gap_param = self.gap_expander_params[0]
        for group in self.optimiser.param_groups:
            if any(param is gap_param for param in group['params']):
                group['weight_decay'] = self.config['optimizer_weight_decay_gap_expander']
        if scheduler_state is not None:
            self.lr_scheduler.load_state_dict(scheduler_state)

    def _prepare_png_visualization_inputs(self):
        zs = np.linspace(
            self.z_begin,
            self.z_end - 1,
            min(int(self.config['output_num_slices_for_visualization']),
                self.z_end - 1 - self.z_begin),
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

    def _preview_splice_evaluation(self, live_transform, dr_per_winding,
                                   progress):
        """The patch-satisfaction evaluation the preview splice reads.

        The preview surface is spliced onto the satisfied patches like the
        final ``_spliced`` meshes. That needs the patches in true scroll
        space: after a constraint bake the resident inputs are in baked
        space and an interactive session keeps no pristine copies, so the
        splice is skipped (returning ``None``) until the frozen stack is
        gone rather than writing baked-space coordinates into the surface.
        """
        if getattr(self, 'frozen_epochs', ()):
            print(f'preview: patch splice skipped; resident inputs are in '
                  f'baked space after {len(self.frozen_epochs)} constraint '
                  'bake(s)')
            return None
        if not self.verified_patches_list:
            return None
        progress.begin(
            'exporting_preview', 'Evaluating patches for preview splice',
            detail=f'{len(self.verified_patches_list):,} patches')
        started = time.perf_counter()
        evaluation = evaluate_patch_satisfaction_packed(
            live_transform, dr_per_winding,
            self.verified_patches_list, self.patch_atlas,
            self.z_begin, self.z_end, include_splicing=True,
        )
        print(f'preview: evaluated {len(self.verified_patches_list):,} '
              f'patches for splice in {time.perf_counter() - started:.2f}s')
        return evaluation

    def periodic_satisfaction_metrics(self):
        """Patch satisfaction (strict and fractional profiles) on the live
        export transform, over every ROI quad centre of the verified patches.
        Costs about a minute on a production atlas; the headless loop calls it
        every ``output_satisfaction_log_interval`` steps so the curve of the
        metric the export reports is visible during the fit."""
        if not self.verified_patches_list:
            return {}
        with torch.no_grad():
            transform = self._export_transform()
            evaluation = evaluate_patch_satisfaction_packed(
                transform, self.dr_per_winding.detach() if torch.is_tensor(self.dr_per_winding)
                else self.spiral_and_transform.get_dr_per_winding().detach(),
                self.verified_patches_list, self.patch_atlas,
                self.z_begin, self.z_end, include_splicing=False)
        metrics = {}
        for name in ('strict', 'fractional'):
            profile = evaluation.profiles[name]
            total_area = float(profile.total_areas.sum())
            metrics[f'{name}_satisfied_area_fraction'] = float(profile.satisfied_areas.sum()) / max(total_area, 1e-9)
            metrics[f'{name}_satisfied_patches_fraction'] = float(profile.satisfied_patches.float().mean())
            metrics[f'{name}_satisfied_patches_area_weighted_fraction'] = (
                float(profile.total_areas[profile.satisfied_patches].sum()) / max(total_area, 1e-9))
        return metrics

    def configure_dt_loss_schedule(self, run_start, requested_iterations,
                                   schedule):
        """Install one Run's independent directional-DT eligibility window."""
        if not schedule['enabled']:
            self.run_dt_resume_iteration = None
            return
        self.run_dt_resume_iteration = get_run_dt_resume_iteration(
            run_start, requested_iterations, schedule['last_fraction'])
        if self.dist.is_main_process:
            print(
                'Run DT losses resume at iteration '
                f'{self.run_dt_resume_iteration} '
                f'(Run starts at {int(run_start)}, '
                f'{int(requested_iterations)} requested iterations)')

    def clear_dt_loss_schedule(self):
        """Remove the transient DT window."""
        self.run_dt_resume_iteration = None

    def clear_interactive_run_state(self):
        """Clear every transient state installed for one interactive Run."""
        self.clear_dt_loss_schedule()

    def export_preview(self, generation_path, surface_id, *, diagnostics=False):
        """Write one preview generation; optionally with its loss overlays.

        The diagnostics pass re-evaluates every enabled loss at full preview
        sample counts and splats each one into an overlay, which costs as much
        as the surface itself. It is off unless the client asked for it, so the
        ordinary "show me the surface" preview does not pay for overlays
        nobody opened.
        """
        progress = progress_or_null(self.progress)
        # Export has its own saved RNG envelope so pausing does not alter the
        # stochastic training sequence.
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()
        try:
            # The preview surface must land in true scroll space, so after
            # any constraint bake it is pulled back through the composed
            # frozen+live chain; the resident inputs that bound its winding
            # range are in baked space and are read through the live chain.
            live_transform = self._export_transform()
            dr_per_winding = self.spiral_and_transform.get_dr_per_winding()
            splice_evaluation = self._preview_splice_evaluation(
                live_transform, dr_per_winding, progress)
            manifest = save_combined_preview(
                live_transform,
                dr_per_winding,
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
                base_shape_zyx=self.base_shape_zyx,
                progress=progress,
                input_extent_transform=live_transform,
                patch_atlas=self.patch_atlas,
                patch_satisfaction_evaluation=splice_evaluation,
            )
            # The same identity the checkpoint written at this boundary
            # carries, so the host can pair the two later.
            manifest = dict(manifest)
            manifest['model_state_sha256'] = self.model_state_digest()
            if not diagnostics:
                return manifest
            diagnostic_weights = {
                name: self.config.get(f'loss_weight_{name}', 0.0)
                for name in (
                    'patch_radius', 'patch_dt',
                    'sym_dirichlet', 'rel_winding', 'abs_winding',
                    'dense_normals', 'dense_spacing',
                    'unattached_pcl_radius', 'unattached_pcl_dt',
                    'track_radius', 'track_dt',
                )
            }
            if self._winding_model_mode_active():
                diagnostic_weights['dense_spacing_winding_model_relative'] = max(
                    float(self.config['loss_weight_dense_spacing']), 1.0)
                diagnostic_weights['dense_spacing_winding_model_density'] = max(
                    float(self.config['loss_weight_dense_spacing_density']), 1.0)
            transform = self.spiral_and_transform.get_slice_to_spiral_transform()
            dr = self.spiral_and_transform.get_dr_per_winding()
            self.theta_crossing_map.force_refresh(transform)
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
                    crossing_map=self.theta_crossing_map,
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
                        crossing_map=self.theta_crossing_map,
                        cfg=self.config, z_begin=self.z_begin, z_end=self.z_end)
                if self.config['loss_weight_abs_winding'] > 0 and self.cross_patch_pcls:
                    get_patch_abs_winding_loss(
                        transform, dr, self.verified_patches, self.patch_atlas,
                        self.cross_patch_pcls,
                        crossing_map=self.theta_crossing_map,
                        cfg=self.config, z_begin=self.z_begin, z_end=self.z_end)
                if self.lasagna_volume is not None and (
                        (self.dense_normals_enabled
                         and self.config['loss_weight_dense_normals'] > 0)
                        or self.grad_mag_spacing_enabled):
                    for _loss_name, _loss_value in iter_lasagna_losses(
                            transform, dr, self.lasagna_volume,
                            self.shell_outer_winding_idx,
                            self.config['sample_count_dense_normal_points'],
                            compute_spacing=self.grad_mag_spacing_enabled,
                            compute_normals=(
                                self.dense_normals_enabled
                                and self.config['loss_weight_dense_normals'] > 0),
                            cfg=self.config, z_begin=self.z_begin, z_end=self.z_end):
                        pass
                if self._winding_model_mode_active():
                    preview_generator = torch.Generator(device=dr.device)
                    preview_generator.manual_seed(0x13198A2E)
                    get_winding_inference_losses(
                        transform, dr, self.winding_inference, self.shell_map,
                        self.config,
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
                        crossing_map=self.theta_crossing_map,
                        cfg=self.config,
                    )
                if self.prepared_main_tracks is not None:
                    for _loss_name, _loss_value in iter_track_losses(
                            transform, dr, self.prepared_main_tracks, self.config,
                            compute_dt=self.config['loss_weight_track_dt'] > 0):
                        pass
            # This block is a one-shot consumer: no later sampler call would
            # drain its deferred unset-potential verdicts, so resolve them
            # before the diagnostics are published.
            self.theta_crossing_map.assert_no_pending_potential_errors()
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
            self._release_export_arena()

    def _release_export_arena(self):
        """Hand the dead device arena back before the host publishes.

        Publication flattens this generation in a separate CUDA process while
        the fitter sits idle in ExportingPreview. By now every allocation the
        training step and this export made at their peaks is dead, but the
        caching allocator still reserves it, so the flatten opens onto a
        nearly full device and thrashes cudaMalloc against blocks nobody is
        using. What stays reserved after this is the live set: parameters,
        optimiser state and the resident brick pools.

        The cost is one re-acquisition through cudaMalloc on the next step,
        which is nothing beside a publication measured in minutes. The freed
        amount is printed because it is the only way to know whether the
        contention this avoids was worth avoiding.
        """
        if not torch.cuda.is_available():
            return
        reserved_before = torch.cuda.memory_reserved()
        gc.collect()
        torch.cuda.empty_cache()
        reserved_after = torch.cuda.memory_reserved()
        gib = 1024 ** 3
        print(
            f'preview export: released '
            f'{(reserved_before - reserved_after) / gib:.2f} GiB of cached '
            f'device memory before publication '
            f'({torch.cuda.memory_allocated() / gib:.2f} GiB live, '
            f'{reserved_after / gib:.2f} GiB still reserved)',
            flush=True)

    def prepare_input_changes(self, records, *,
                              current_iteration=0, target_iteration=0):
        """Prepare a complete revision batch without changing active state.

        Each record carries a stable logical ``id``, a ``kind``, a content
        ``path`` or ``deleted``, and (for baseline adoption) ``source_id`` and
        ``source_path``. Source point geometry is retained before attachments
        and normalization so removing a patch can undo derived attachments.
        The returned context shares the trained model and unchanged geometry;
        only input-derived fields are installed at the worker boundary.
        """
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()
        try:
            candidate = copy.copy(self)
            candidate._preparing_inputs = True
            candidate.non_liftable_patch_paths = set(self.non_liftable_patch_paths)
            candidate._source_verified_patches = dict(self._source_verified_patches)
            candidate._source_point_collections = {
                cid: catalog_copy_of_pcl(pcl)
                for cid, pcl in self._source_point_collections.items()
            }
            candidate._workspace_membership = dict(getattr(self, '_workspace_membership', {}))
            candidate.next_id = self.next_id
            seen = set()
            for record in records:
                logical_id = str(record['id'])
                if logical_id in seen:
                    raise ValueError('A batch cannot contain two revisions of one input')
                seen.add(logical_id)
                kind = record['kind']
                deleted = bool(record.get('deleted'))
                adopt = bool(record.get('adopt'))
                path = record.get('path')
                source_id = str(record.get('source_id', logical_id))
                if kind == 'patch':
                    source = candidate._source_verified_patches
                    if source_id != logical_id and source_id in source:
                        # Preserve the dataset patch name for between-patch
                        # annotations; catalog UUID is independent of that name.
                        patch_id = source_id
                    else:
                        patch_id = candidate._workspace_membership.get(logical_id, {}).get(
                            'resident_id', source_id)
                    if deleted:
                        source.pop(patch_id, None)
                    elif not adopt:
                        # Baseline adoption only binds workspace identities.
                        # The initial loader already selected usable geometry
                        # (z range, erosion, name filter, and input toggles).
                        # An absent baseline must stay excluded, not be read
                        # again through the stricter live-revision loader.
                        patch = load_tifxyz(path)
                        patch._source_path = os.path.abspath(path)
                        cells = patch.erosion_cells(self.config['patch_erode_patches'])
                        valid = cells <= 0 or erode_patch_valid_region(patch, cells)
                        intersects = patch_intersects_z_roi(patch, self.z_begin, self.z_end)
                        if not valid or not intersects:
                            reason = 'has no valid quads' if not valid else 'is outside the fitted z range'
                            raise ValueError(f'Patch {logical_id} {reason}')
                        source[patch_id] = patch
                    resident_id = patch_id
                elif kind in {'pcl', 'fiber'}:
                    source = candidate._source_point_collections
                    previous = candidate._workspace_membership.get(logical_id, {})
                    resident_id = previous.get('resident_id')
                    if resident_id is None:
                        for cid, pcl in source.items():
                            metadata = pcl.get('metadata', {})
                            same_source = os.path.realpath(pcl.get('source_file', '')) == os.path.realpath(
                                record.get('source_path') or path or '')
                            if kind == 'fiber':
                                match = metadata.get('logical_input_kind') == 'fiber' and (
                                    str(metadata.get('logical_input_id')) == source_id or same_source)
                            else:
                                match = (same_source and str(pcl.get('id')) == source_id
                                         and metadata.get('input_role') == record.get('role'))
                            if match:
                                resident_id = cid
                                break
                    if resident_id is None:
                        resident_id = candidate.next_id
                        candidate.next_id += 1
                    if deleted:
                        source.pop(resident_id, None)
                    elif adopt and kind == 'fiber' and resident_id not in source:
                        # Preserve startup exclusions (including disabled or
                        # malformed fibers) while retaining workspace identity.
                        pass
                    else:
                        if adopt and resident_id in source:
                            pcl = source[resident_id]
                            if kind == 'fiber':
                                pcl['source_file'] = path
                        elif kind == 'fiber':
                            pcl = load_fiber_point_collection(
                                path, resident_id,
                                min_point_spacing=self.config['pcl_fiber_min_point_spacing'],
                                base_shape_zyx=getattr(self, 'base_shape_zyx', None))
                            if pcl is None:
                                raise ValueError(f'Fiber {logical_id} has no usable control points')
                            pcl['sampling_group'] = 'fibers'
                            pcl['file_basename'] = f'{source_id}.json'
                            pcl['source_file'] = path
                            pcl.setdefault('metadata', {}).update(
                                input_role='fiber', winding_is_absolute=False)
                        else:
                            loaded = load_point_collection(path) or {}
                            if len(loaded) != 1:
                                raise ValueError('A logical PCL input contains exactly one collection')
                            pcl = next(iter(loaded.values()))
                            stamp_loaded_pcl_metadata(
                                pcl, record.get('source_path') or path, record.get('role'), resident_id)
                        pcl.setdefault('metadata', {}).update({
                            'workspace_input_id': logical_id,
                            'logical_input_kind': kind,
                            'logical_input_id': logical_id,
                            'logical_input_revision': record.get('revision'),
                        })
                        source[resident_id] = pcl
                else:
                    raise ValueError(f'Unknown input kind {kind!r}')
                candidate._workspace_membership[logical_id] = {
                    'kind': kind, 'revision': record.get('revision'),
                    'resident_id': resident_id, 'deleted': deleted,
                }

            candidate.verified_patches = {
                pid: copy.copy(patch) for pid, patch in candidate._source_verified_patches.items()
            } if input_source_enabled(self.config, 'verified_patches') else {}
            candidate.verified_patches_list = list(candidate.verified_patches.values())
            candidate.num_verified_patches = len(candidate.verified_patches_list)
            candidate.patch_sampling_probabilities = candidate._prepare_patch_sampling_cache(
                candidate.verified_patches_list)
            candidate.patch_atlas = self.patch_atlas.replaced(candidate.verified_patches)
            candidate._input_warnings = []
            while True:
                points, fibers = {}, {}
                for cid, pcl in candidate._source_point_collections.items():
                    metadata = pcl.get('metadata', {})
                    role = ('fiber' if metadata.get('logical_input_kind') == 'fiber'
                            else metadata.get('input_role'))
                    enabled = (input_source_enabled(self.config, 'fibers') if role == 'fiber'
                               else pcl_input_enabled(self.config, None if role == 'legacy' else role,
                                                      pcl.get('source_file', '')))
                    if enabled:
                        points[cid] = catalog_copy_of_pcl(pcl)
                        if role == 'fiber':
                            fibers[cid] = points[cid]
                candidate._derive_point_inputs(
                    candidate.verified_patches, points, fibers,
                    direction_source=candidate._desired_fiber_link_direction_source(current_iteration))
                candidate._refresh_trusted_geometry()
                if self.dt_target_whole_object:
                    prepare_patch_dt_target_samples(
                        candidate.verified_patches_list,
                        self.config['sample_count_patch_dt_target_points'], self.config['dt_target_max_stride'])
                candidate.dt_target_cache_manager = DtTargetCacheManager(
                    self.dt_target_cache_manager.update_interval,
                    self.dt_target_cache_manager.on_first_update)
                if self.using_tracks and float(self.config['track_exclusion_radius']) > 0:
                    if self.tracks is None:
                        raise ValueError('Original tracks are required to recompute exclusion masks')
                    candidate.prepared_main_tracks = prepare_main_phase_tracks(
                        self.tracks, None, float(self.config['track_exclusion_radius']), self.device,
                        anchor_tree=candidate.trusted_geometry_tree,
                        sampling_config=validate_track_sampling_config(self.config),
                        track_families=self.track_families,
                        track_source_ids=self.track_source_ids, crossing_cache=self.track_crossing_cache,
                        track_graph=self.track_graph, progress=self.progress)
                before_verified = set(candidate.verified_patches)
                candidate._input_warnings.extend(candidate._build_theta_crossing_map())
                rejected_verified = before_verified - set(candidate.verified_patches)
                for record in records:
                    # Sources predate startup theta validation. Adoption may
                    # repeat its exclusions; only live revisions must fail.
                    if (record['kind'] != 'patch' or record.get('deleted')
                            or record.get('adopt')):
                        continue
                    member = candidate._workspace_membership[str(record['id'])]
                    if member['resident_id'] in rejected_verified:
                        raise ValueError(f"Patch {record['id']} failed theta consistency validation")
                if not rejected_verified:
                    break
                # Theta rejection can detach points. Re-derive the point inputs
                # from their immutable sources before accepting this candidate,
                # with rejected patches excluded.
            # The optimiser, parameter tensors and the trained model stay shared.
            return candidate
        except MissingPclSamplingWeightError as exc:
            # Reject configuration errors without stopping the resident worker.
            raise ValueError(str(exc)) from exc
        finally:
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            torch.cuda.set_rng_state_all(cuda_states)

    def install_input_changes(self, candidate):
        """Install a successfully prepared candidate on the fitter thread."""
        ignored = {'_preparing_inputs', '_input_warnings'}
        updates = {key: value for key, value in vars(candidate).items()
                   if key not in ignored and value is not getattr(self, key, None)}
        self.__dict__.update(updates)
        self._write_non_liftable_patch_report()
        return list(candidate._input_warnings)

    def _regular_views_from_catalog_entries(self, entries, voxel_scale):
        """Derive the training views of regular (non-fiber) collections.

        ``entries`` maps resident id -> linked pcl container (a catalog copy,
        or a freshly uploaded collection). Each is classified from how its
        points attach: at least two attached points make a cross-patch pcl
        (points grouped by patch), at least one unattached point makes an
        unattached strip over the whole pcl (trimmed to the longest run in
        the z window, then decimated), and absolute-winding pcls are always
        cross-patch. A pcl in both sets gets an independent copy for the
        strip. Returns ``(cross_patch_pcls, strips, strip_groups)``; the
        containers in ``entries`` become the cross-patch views, so pass copies
        when the originals must stay pristine.
        """
        new_cross_patch = {}
        new_unattached = {}
        for pid, pcl in entries.items():
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
            kept_items = longest_run_in_z_window(
                sorted(pcl['points'].items(), key=lambda kv: int(kv[0])),
                self.z_begin, self.z_end, z_margin)
            if len(kept_items) < 2:
                del new_unattached[pid]
            else:
                pcl['points'] = dict(kept_items)

        normalise_pcl_winding_annotations(new_cross_patch)
        normalise_pcl_winding_annotations(new_unattached)

        cross_patch = []
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
            cross_patch.append(pcl)

        min_point_spacing = (
            self.config['pcl_unattached_pcl_min_point_spacing']
            * voxel_scale)
        strips, strip_groups = [], []
        for pcl_id, pcl in new_unattached.items():
            strip = regular_unattached_strip(
                pcl_id, pcl, min_point_spacing)
            if strip is None:
                continue
            strips.append(strip)
            strip_groups.append(pcl.get('sampling_group'))
        return cross_patch, strips, strip_groups

    def _relink_all_points_to_patches(self, *, iteration):
        """Run-boundary path for the linking settings that govern every point
        collection (pcl_link_distance_tolerance, pcl_link_window_points,
        pcl_link_window_min_points).

        Drops every resident point's patch attachment (regular catalog and
        fiber catalog), links both again against the verified patches under
        the current options -- the fiber side rules taking the inward
        direction ``iteration`` implies -- and re-derives every
        point-collection training view from the catalogs.
        """
        self.link_distance_tolerance = float(
            self.config['pcl_link_distance_tolerance'])
        regular = dict(getattr(self, 'regular_pcl_catalog', None) or {})
        fibers = {pcl['id']: pcl
                  for pcl in (getattr(self, 'fiber_catalog', None) or {}).values()}
        collections = {**regular, **fibers}
        if not collections:
            return
        before = sum(1 for pcl in collections.values()
                     for point in pcl['points'].values() if 'on_patch' in point)
        for pcl in collections.values():
            for point in pcl['points'].values():
                point.pop('on_patch', None)
        voxel_scale = self._baked_voxel_scale()
        tolerance = self.link_distance_tolerance * voxel_scale
        direction_source = self._desired_fiber_link_direction_source(iteration)
        options = self._patch_link_options(
            collections, voxel_scale, direction_source=direction_source)
        link_points_to_patches(
            self.verified_patches,
            collections,
            tolerance=tolerance,
            surface_index_tolerance=tolerance,
            distance_scale=1.0,
            general_hit_policy='largest_area',
            options=options,
        )
        self._fiber_link_direction_source = direction_source
        after = sum(1 for pcl in collections.values()
                    for point in pcl['points'].values() if 'on_patch' in point)
        dist = getattr(self, 'dist', None)
        if dist is None or dist.is_main_process:
            print(f'step {iteration}: relinked {len(regular)} regular '
                  f'collection(s) and {len(fibers)} fiber(s) to patches '
                  f'(tolerance {self.link_distance_tolerance:g}, window '
                  f'{options.window_points}/{options.window_min_points}): '
                  f'{before} -> {after} attached points')

        # Every regular view is re-derived from a fresh catalog copy (the
        # catalog stays pristine); the fiber views follow from the catalog.
        working = {}
        for cid, pcl in regular.items():
            copy_ = catalog_copy_of_pcl(pcl)
            copy_['chain'] = SequenceChain(copy_)
            working[cid] = copy_
        cross_patch, strips, strip_groups = (
            self._regular_views_from_catalog_entries(working, voxel_scale))
        views = self._fiber_views_from_catalog(
            self.fiber_catalog, cross_patch, strips, strip_groups, voxel_scale)
        self._commit_rederived_pcl_views(*views)
        link_warning = unresolved_fiber_link_warning(
            self.fiber_catalog,
            use_links=self.config['pcl_use_fiber_links'],
            use_pending_links=self.config['pcl_use_pending_fiber_links'])
        if link_warning is not None:
            print(f'WARNING: {link_warning}')

    def _rederive_regular_unattached_strips(self):
        min_point_spacing = self.config[
            'pcl_unattached_pcl_min_point_spacing']
        z_margin = self.config['patch_loss_z_margin']
        rederived = {}
        for pid, catalog_pcl in self.regular_pcl_catalog.items():
            if catalog_pcl.get('metadata', {}).get(
                    'winding_is_absolute', False):
                continue
            if all('on_patch' in point
                   for point in catalog_pcl['points'].values()):
                continue
            pcl = catalog_copy_of_pcl(catalog_pcl)
            kept_items = longest_run_in_z_window(
                sorted(pcl['points'].items(), key=lambda kv: int(kv[0])),
                self.z_begin, self.z_end, z_margin)
            if len(kept_items) < 2:
                continue
            pcl['points'] = dict(kept_items)
            rederived[pid] = pcl
        normalise_pcl_winding_annotations(rederived)
        strips, strip_groups = [], []
        for pid, pcl in rederived.items():
            strip = regular_unattached_strip(pid, pcl, min_point_spacing)
            if strip is not None:
                strips.append(strip)
                strip_groups.append(pcl.get('sampling_group'))
        for strip, group in zip(self.unattached_pcl_strips,
                                self.unattached_strip_sampling_groups):
            if strip.get('logical_input_kind') == 'fiber':
                strips.append(strip)
                strip_groups.append(group)
        self._commit_rederived_pcl_views(
            list(self.cross_patch_pcls), strips, strip_groups)

    def _prepare_fiber_reingest(self):
        """Check reload sources before Run settings mutate resident state."""
        records = []
        missing = []
        for logical_id, pcl in self.fiber_catalog.items():
            path = pcl.get('source_file')
            if not path or not os.path.isfile(path):
                missing.append(str(logical_id))
                continue
            records.append({
                'kind': 'fiber', 'path': path, 'id': str(logical_id),
                'source_id': os.path.splitext(pcl.get('file_basename', str(logical_id)))[0],
                'revision': pcl.get('metadata', {}).get(
                    'logical_input_revision'),
            })
        if missing:
            raise ValueError(
                'pcl_fiber_min_point_spacing cannot change at a Run boundary: '
                f'{len(missing)} resident fiber document(s) are no longer on '
                f'disk (e.g. {missing[:3]}); rebuild the fit instead')
        return records

    def _reingest_fiber_documents(self, records):
        if records:
            candidate = self.prepare_input_changes(records)
            warnings = self.install_input_changes(candidate)
            for warning in warnings or ():
                print(f'WARNING: {warning}')

    def apply_config(self, config, path_changes=None, *, current_iteration):
        """Apply Run-scoped settings without replacing the resident fit.

        current_iteration is the session's durable completed-iteration count;
        an LR-schedule change is realigned at that step.
        """
        path_changes = dict(path_changes or {})
        changed = set(config)
        if any(key.startswith('track_') for key in changed):
            # Validate combined bounds before changing participation or any
            # other resident derivation: restoring config alone cannot undo
            # an installed input candidate after a rejected Run.
            validate_track_sampling_config({**self.config, **config})
        cadence_keys = {
            'theta_crossing_map_update_interval',
            'dt_target_update_interval',
        }
        cadence_changed = changed & cadence_keys
        tracked = set(config) | (cadence_keys if cadence_changed else set())
        old_values = {key: self.config[key] for key in tracked}
        self.config.update(config)
        try:
            if 'model_pin_targets_integer' in changed and self.spiral_and_transform.pins_active:
                self._apply_integer_pin_targets()
            if changed & {'pcl_link_window_points', 'pcl_link_window_min_points'}:
                window = int(self.config['pcl_link_window_points'])
                minimum = int(self.config['pcl_link_window_min_points'])
                if not 1 <= minimum <= window:
                    raise ValueError(
                        'pcl_link_window_min_points must be between 1 and '
                        f'pcl_link_window_points ({window}), got {minimum}')
            fiber_catalog = getattr(self, 'fiber_catalog', None) or {}
            fiber_records = (
                self._prepare_fiber_reingest()
                if 'pcl_fiber_min_point_spacing' in changed and fiber_catalog
                else None)
            if cadence_changed:
                if cadence_keys <= changed and (
                        int(config['theta_crossing_map_update_interval'])
                        != int(config['dt_target_update_interval'])):
                    raise ValueError(
                        'theta_crossing_map_update_interval and '
                        'dt_target_update_interval are one shared cadence and '
                        'cannot be set to different values')
                cadence = int(
                    config['theta_crossing_map_update_interval']
                    if 'theta_crossing_map_update_interval' in changed
                    else config['dt_target_update_interval'])
                self.config.update({
                    'theta_crossing_map_update_interval': cadence,
                    'dt_target_update_interval': cadence,
                })
                changed.update(cadence_keys)
            # Static path changes are rejected by the service. Settings below
            # either read live or rebuild a derivation from retained inputs.
            rebuilt_tracks = None
            replace_prepared_tracks = False
            rebuilt_shell_map = self.shell_map
            rebuilt_shell_outer = self.shell_outer_winding_idx

            reprepare_tracks = bool(changed & {
                'track_max_tortuosity',
                'track_exclusion_radius',
                'track_crossing_precompute_max',
            })
            shell_atlas_keys = SHELL_ATLAS_KEYS
            if (changed & shell_atlas_keys
                    and getattr(self, 'shell_envelope', None) is not None):
                raise ValueError(
                    'the shell atlas settings filtered this session tracks '
                    'against the outer shell when they were loaded; rebuild '
                    'the fit to change them')
            if ('loss_weight_fiber_directions' in changed
                    and self.config['loss_weight_fiber_directions'] > 0
                    and getattr(self, 'fiber_direction_samples', None) is None):
                raise ValueError(
                    'loss_weight_fiber_directions > 0 needs resident '
                    'fiber-direction samples; rebuild with that input enabled')
            rebuild_strata = 'pcl_sampling_weights' in changed
            if rebuild_strata and self.config['pcl_sampling_weights'] is not None:
                build_pcl_sampling_strata(
                    [pcl['sampling_group'] for pcl in self.cross_patch_pcls]
                    + list(self.unattached_strip_sampling_groups),
                    self.config)
            if 'patch_loss_z_margin' in changed:
                self.patch_sampling_probabilities = \
                    self._prepare_patch_sampling_cache(self.verified_patches_list)
                self.patch_atlas.rebuild_sampling_atlas()
            elif 'patch_sampling_area_exponent' in changed:
                self.patch_sampling_probabilities = self._patch_sampling_probabilities(
                    self.verified_patches_list)

            # Loss weights are live settings. If a shell loss is enabled for
            # the first time, construct only the resident structure that loss
            # needs; disabling it releases that structure. Atlas-shaping
            # settings themselves require a full prepared-input rebuild.
            if 'loss_weight_shell_outer' in changed:
                rebuilt_shell_map = (
                    self._make_shell_polar_map()
                    if (self.shell_patch is not None
                        and (self.config['loss_weight_shell_outer'] > 0
                             or self.winding_model_mode)) else None
                )
            if (changed & (shell_atlas_keys - {'shell_min_confidence'})
                    and 'loss_weight_shell_outer' not in changed
                    and rebuilt_shell_map is not None):
                rebuilt_shell_map = self._make_shell_polar_map()
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
            if any(key.startswith('dt_') for key in changed) \
                    or dt_preparation_changed:
                self.dt_target_cache_manager.update_interval = max(
                    1, int(self.config[
                        'theta_crossing_map_update_interval']))
                self.dt_target_cache_manager.reset()
            if changed & {
                    'optimizer_exp_lr_schedule',
                    'optimizer_learning_rate',
                    'optimizer_lr_final_factor',
                    'optimizer_num_training_steps',
            }:
                self._realign_lr_schedule(current_iteration)

            fiber_view_keys = {
                'pcl_use_fiber_links',
                'pcl_use_pending_fiber_links',
                'pcl_unattached_pcl_min_point_spacing',
            }
            fiber_link_keys = {
                'pcl_fiber_link_side_filter',
                'pcl_fiber_link_side_margin_voxels',
                'pcl_fiber_link_model_direction_step',
            }
            # The load-time linking settings govern every point collection:
            # both catalogs are re-linked and every view re-derived.
            all_link_keys = {
                'pcl_link_distance_tolerance',
                'pcl_link_window_points',
                'pcl_link_window_min_points',
            }
            fiber_catalog = getattr(self, 'fiber_catalog', None) or {}
            regular_catalog = getattr(self, 'regular_pcl_catalog', None) or {}
            rederived_views = False
            relinked_everything = False
            if fiber_records is not None:
                self._reingest_fiber_documents(fiber_records)
                rederived_views = True
            if changed & all_link_keys and (fiber_catalog or regular_catalog):
                self._relink_all_points_to_patches(iteration=current_iteration)
                rederived_views = True
                relinked_everything = True
            elif changed & fiber_link_keys and fiber_catalog:
                # Re-links every resident fiber under the new side rules, with
                # the inward direction the current step implies (umbilicus
                # below pcl_fiber_link_model_direction_step, fitted winding
                # from there), then re-materialises the fiber views.
                self._relink_fibers_to_patches(
                    self._desired_fiber_link_direction_source(
                        current_iteration),
                    iteration=current_iteration)
                rederived_views = True
            elif changed & fiber_view_keys and fiber_catalog:
                self._rematerialize_fiber_views()
                rederived_views = True
            if ('pcl_unattached_pcl_min_point_spacing' in changed
                    and regular_catalog and not relinked_everything):
                self._rederive_regular_unattached_strips()
                rederived_views = True
            # Rebuild participation from retained source revisions after all
            # other view changes. Disabled roles keep their editable content.
            role_changes = changed & {pcl_role_toggle_key(role) for role in RUN_MUTABLE_PCL_ROLES}
            if role_changes:
                candidate = self.prepare_input_changes(
                    [], current_iteration=current_iteration,
                    target_iteration=current_iteration)
                self.install_input_changes(candidate)
                rederived_views = True
            # Prepare tracks against the final participation geometry and policy.
            if reprepare_tracks and rebuilt_tracks is None and self.tracks:
                rebuilt_tracks = prepare_main_phase_tracks(
                    self.tracks, None, float(self.config['track_exclusion_radius']),
                    self.device, anchor_tree=self.trusted_geometry_tree,
                    sampling_config=validate_track_sampling_config(self.config),
                    track_families=self.track_families,
                    track_source_ids=self.track_source_ids,
                    crossing_cache=self.track_crossing_cache,
                    track_graph=self.track_graph,
                    progress=self.progress)
                replace_prepared_tracks = True

            target_tracks = (
                rebuilt_tracks
                if replace_prepared_tracks else self.prepared_main_tracks)
            if ({'track_length_bin_weights',
                 'track_max_track_crossing_per_step'}
                    & changed):
                configure_prepared_track_sampling(target_tracks, config)

            if rebuild_strata and not rederived_views:
                self._rebuild_pcl_sampling_strata()
        except Exception:
            self.config.update(old_values)
            raise

        self.shell_map = rebuilt_shell_map
        if 'model_num_flow_integration_steps' in changed:
            self.spiral_and_transform.flow_integration_steps = int(
                self.config['model_num_flow_integration_steps'])
        self.shell_outer_winding_idx = rebuilt_shell_outer
        if replace_prepared_tracks:
            self.prepared_main_tracks = rebuilt_tracks
            self.preview_extent_tracks = (
                (self.prepared_main_tracks['flat_zyx_cpu'],)
                if self.prepared_main_tracks is not None else ())
        if changed & {
                'pcl_vertical_fiber_radial_offset_enabled',
                'pcl_vertical_fiber_radial_offset_voxels',
        }:
            self._refill_vertical_fiber_radial_offsets()
        if 'patch_loss_z_margin' in changed:
            self._build_theta_crossing_map()
            self.dt_target_cache_manager.reset()
        elif 'theta_crossing_map_update_interval' in changed:
            # refresh_if_due observes the changed interval on the next step and
            # resets its cadence without rebuilding immutable topology.
            self.theta_crossing_map.invalidate()
            self.dt_target_cache_manager.update_interval = max(
                1, int(self.config[
                    'theta_crossing_map_update_interval']))
            self.dt_target_cache_manager.reset()

    def _refresh_theta_crossing_map_for_step(self, iteration, transform):
        refreshed = self.theta_crossing_map.refresh_if_due(
            iteration, transform,
            self.config['theta_crossing_map_update_interval'])
        if refreshed:
            self._enforce_theta_liftability()
            # Patch targets use the theta map's root-relative frame. Reset all
            # whole-object targets here so every active cache is rebuilt later
            # in this same step, phase-locking both cache families.
            self.dt_target_cache_manager.reset()
        return refreshed

    def step(self, iteration):
        self.step_timer.start('fwd')
        flow_field_low_res_lr_scale, flow_field_high_res_lr_scale = (
            self._apply_flow_group_settings(iteration))

        # The tiny graph paths shared by every transform evaluation this
        # iteration (dr softplus, scaled linear logits, pinned gap logits) are
        # cut at detached leaves. Each loss family's backward then owns its
        # whole graph, so autograd can free the family's buffers as the
        # backward pass consumes them instead of retaining the full graph
        # (retain_graph) until the family is released. The leaf gradients
        # accumulated across families flow through the real shared paths once,
        # next to the flow-field gradient flush below.
        self._maybe_activate_pins(iteration)
        if (self.spiral_and_transform._pins_enabled()
                and self.pins_activation_iteration is not None
                and iteration > self.pins_activation_iteration
                and (iteration - self.pins_activation_iteration) % PIN_DEMOTE_RECHECK_INTERVAL == 0):
            self._demote_conflicting_patches(iteration)
        # Pin subset for this step: every pin of the patches the patch losses
        # will sample (drawn here, the losses then draw from that set), so the
        # pinned map is exact where it is evaluated rather than a thin uniform
        # subsample smeared over widened footprints.
        step_patch_probabilities = self.patch_sampling_probabilities
        rel_winding_rows = abs_winding_rows = None
        # Stage 3: while pins are active the constraint losses on pinned
        # inputs are replaced by the pin strain loss (config-gated).
        pins_replace = (self.spiral_and_transform._pins_enabled()
                        and bool(self.config.get('loss_pins_replace_constraint_losses', True)))
        weight_patch_radius = 0.0 if pins_replace else self.config['loss_weight_patch_radius']
        weight_rel_winding = 0.0 if pins_replace else self.config['loss_weight_rel_winding']
        weight_abs_winding = 0.0 if pins_replace else self.config['loss_weight_abs_winding']
        weight_unattached_radius = 0.0 if pins_replace else self.config['loss_weight_unattached_pcl_radius']
        # Inputs without pins keep their soft constraint losses when pins
        # replace them: verified patches the registry does not pin (spread
        # filter, flow z domain), PCL rows touching such a patch, and strips
        # with non-pin points (vertical fibers with radial offsets).
        unpinned_patch_mask = self.unpinned_verified_patch_mask() if pins_replace else None
        unpinned_patches = (torch.nonzero(unpinned_patch_mask, as_tuple=True)[0].cpu().numpy()
                            if unpinned_patch_mask is not None and bool(unpinned_patch_mask.any()) else None)
        if pins_replace and any(
                strip.get('radial_offsets') is not None and bool((np.asarray(strip['radial_offsets']) != 0).any())
                for strip in self.unattached_pcl_strips):
            weight_unattached_radius = self.config['loss_weight_unattached_pcl_radius']
        if self.spiral_and_transform._pins_enabled() and self.verified_patches_list:
            # The PCL winding losses are drawn here too, so the patches they
            # touch are pinned this step as well as the patch-loss patches.
            if self.config['loss_weight_rel_winding'] > 0 and self.cross_patch_pcls:
                rel_winding_rows = draw_rel_winding_rows(
                    self.verified_patches, self.patch_atlas, self.cross_patch_pcls,
                    self.pcl_sampling_strata['cross_patch'], self.config)
            if self.config['loss_weight_abs_winding'] > 0 and self.cross_patch_pcls:
                abs_winding_rows = draw_abs_winding_rows(
                    self.verified_patches, self.patch_atlas, self.cross_patch_pcls, self.config)
            pcl_patches = [idx for row in (rel_winding_rows or []) for idx in row['patch_indices']]
            pcl_patches += [row[0] for row in (abs_winding_rows or [])]
            self.spiral_and_transform.set_step_pin_patches(
                np.union1d(self._draw_step_pin_patches(), np.asarray(pcl_patches, dtype=np.int64)))
            if pins_replace:
                # Keep only the rows that touch an unpinned patch, at full weight.
                pinned = None if unpinned_patch_mask is None else ~unpinned_patch_mask.cpu()
                def touches_unpinned(indices):
                    return pinned is None or any(not bool(pinned[int(i)]) for i in indices)
                rel_winding_rows = [row for row in (rel_winding_rows or []) if touches_unpinned(row['patch_indices'])]
                abs_winding_rows = [row for row in (abs_winding_rows or []) if touches_unpinned(row[:1])]
                weight_rel_winding = self.config['loss_weight_rel_winding'] if rel_winding_rows else 0.0
                weight_abs_winding = self.config['loss_weight_abs_winding'] if abs_winding_rows else 0.0
        shared_transform_outputs = self.spiral_and_transform.get_shared_transform_tensors()
        active_patches = self.spiral_and_transform.active_step_pin_patches()
        if active_patches is not None and self.verified_patches_list:
            step_patch_probabilities = self._restrict_patch_probabilities(active_patches)
        shared_transform_leaves = tuple(
            output.detach().requires_grad_(True) for output in shared_transform_outputs)
        self.slice_to_spiral_transform = self.spiral_and_transform.get_slice_to_spiral_transform(
            shared=shared_transform_leaves)
        self.dr_per_winding = shared_transform_leaves[0]
        pins_leaf = shared_transform_leaves[3] if len(shared_transform_leaves) > 3 else None
        log_pins_this_step = pins_leaf is not None and iteration % 200 == 0
        pin_grad_seen = None
        theta_map_refreshed = self._refresh_theta_crossing_map_for_step(
            iteration,
            self.slice_to_spiral_transform)
        self._maybe_relink_fibers_for_direction(iteration)

        losses = {}
        log_metrics = {
            'flow_field_low_res_lr_scale': flow_field_low_res_lr_scale,
            'flow_field_high_res_lr_scale': flow_field_high_res_lr_scale,
        }

        def backward_family(weighted_losses):
            """Accumulate one loss family's gradients, then release its graph."""
            nonlocal pin_grad_seen
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
            if log_pins_this_step:
                # Per-family gradient on T: dL/dT_g = dr * sum_i
                # dL/dr_target_i over the component's pins, read off the pins
                # leaf as the increment this family added.
                self._record_pin_target_grad(
                    '+'.join(weighted_losses), pins_leaf, pin_grad_seen)
                pin_grad_seen = (
                    pins_leaf.grad.clone() if pins_leaf.grad is not None else None)

        run_dt_suppressed = (
            self.run_dt_resume_iteration is not None
            and iteration < self.run_dt_resume_iteration
        )
        log_metrics['run_dt_suppressed'] = float(run_dt_suppressed)

        dt_eligibility = get_dt_loss_eligibility(
            self.config, iteration, self.run_dt_resume_iteration)
        compute_patch_dt = dt_eligibility['verified_patch']
        compute_track_dt = dt_eligibility['track']
        compute_unattached_pcl_dt = dt_eligibility['unattached_pcl']

        patch_dt_target_cache = None
        unattached_pcl_dt_target_cache = None
        track_dt_target_cache = None
        if self.dt_target_whole_object:
            if compute_patch_dt and self.config['loss_weight_patch_dt'] > 0 and self.verified_patches_list:
                patch_dt_target_cache = self.dt_target_cache_manager.get('patch', iteration, lambda: compute_patch_dt_target_cache(
                    self.slice_to_spiral_transform, self.dr_per_winding,
                    self.verified_patches_list, self.patch_atlas,
                    self.theta_crossing_map,
                    self.config['dt_target_floating_threshold'],
                    pinned_values=self._pinned_patch_dt_values(),
                ))
            if compute_unattached_pcl_dt and self.config['loss_weight_unattached_pcl_dt'] > 0 and self.unattached_pcl_strips:
                pcl_flat = get_or_build_unattached_pcl_flat(self.unattached_pcl_strips, torch.device('cuda'))
                if pcl_flat is not None:
                    unattached_pcl_dt_target_cache = self.dt_target_cache_manager.get('unattached_pcl', iteration, lambda: compute_strip_dt_target_cache(
                        self.slice_to_spiral_transform, self.dr_per_winding,
                        pcl_flat['zyxs'], pcl_flat['starts'],
                        windings=pcl_flat['windings'],
                        radial_offsets=pcl_flat.get('radial_offsets'),
                        floating_threshold=self.config['dt_target_floating_threshold'],
                        num_points_per_strip=self.config['sample_count_dt_target_points_per_strip'],
                        max_stride=self.config['dt_target_max_stride'],
                        max_total_points=5_000_000,
                    ))
            if compute_track_dt and self.config['loss_weight_track_dt'] > 0 and self.prepared_main_tracks is not None:
                track_dt_target_cache = self.dt_target_cache_manager.get('track', iteration, lambda: compute_strip_dt_target_cache(
                    self.slice_to_spiral_transform, self.dr_per_winding,
                    self.prepared_main_tracks['flat_zyx_cpu'], self.prepared_main_tracks['offsets'],
                    windings=None,
                    floating_threshold=self.config['dt_target_floating_threshold'],
                    num_points_per_strip=self.config['sample_count_dt_target_points_per_strip'],
                    max_stride=self.config['dt_target_max_stride'],
                    max_total_points=5_000_000,
                ))

        # Stage 3: pin strain, evaluated on this step's pins leaf against the
        # step's own free gap map (see losses.get_pin_strain_loss).
        if pins_leaf is not None and self.config.get('loss_weight_pin_strain', 0.0) > 0:
            strain_loss, strain = get_pin_strain_loss(
                pins_leaf, pinned_gap_stage(self.slice_to_spiral_transform), self.dr_per_winding,
                margin=float(self.config.get('loss_margin_pin_strain', 0.0)))
            backward_family({'pin_strain': strain_loss * self.config['loss_weight_pin_strain']})
            if strain.numel() and iteration % 20 == 0:
                magnitude = strain.abs()
                log_metrics['pin_strain_median'] = float(magnitude.median())
                log_metrics['pin_strain_p90'] = float(torch.quantile(magnitude, 0.9))
                log_metrics['pin_strain_frac_over_margin'] = float(
                    (magnitude > float(self.config.get('loss_margin_pin_strain', 0.0))).float().mean())

        # Pair agreement on the free map (pinned_spiral_plan.md, "preparing
        # the warm-up"): nearby quad centres of different constraint
        # components must differ by a whole number of windings. Evaluated on
        # the unpinned transform (the step's own leaves) whether or not pins
        # are active, since the pin targets are read off the free map. The
        # pairs come from the pin constraint graph, so without pins it is off.
        weight_pair_agreement = float(self.config.get('loss_weight_pair_agreement', 0.0) or 0.0)
        if weight_pair_agreement > 0 and self.pin_graph is not None:
            pairs, pair_zyx = self._pair_agreement_pairs()
            count = min(int(self.config['sample_count_pair_agreement']), int(pairs.shape[0]))
            if count > 0:
                chosen = pairs[torch.randint(int(pairs.shape[0]), [count], device=pairs.device)]
                free_transform = (
                    self.slice_to_spiral_transform
                    if not self.spiral_and_transform._pins_enabled()
                    else self.spiral_and_transform.get_unpinned_slice_to_spiral_transform(
                        shared=shared_transform_leaves))
                pair_loss, pair_residual = get_pair_agreement_loss(
                    pair_zyx[chosen[:, 0]], pair_zyx[chosen[:, 1]], free_transform, self.dr_per_winding,
                    margin=float(self.config.get('loss_margin_pair_agreement', 0.0)))
                backward_family({'pair_agreement': pair_loss * weight_pair_agreement})
                if pair_residual.numel() and iteration % 20 == 0:
                    magnitude = pair_residual.abs()
                    log_metrics['pair_agreement_median'] = float(magnitude.median())
                    log_metrics['pair_agreement_frac_over_margin'] = float(
                        (magnitude > float(self.config.get('loss_margin_pair_agreement', 0.0))).float().mean())

        patch_loss_values = get_patch_and_umbilicus_losses(
            self.slice_to_spiral_transform,
            self.dr_per_winding,
            self.config['sample_count_patches_per_step'],
            self.config['sample_count_patches_per_step_for_dt'],
            self.verified_patches_list,
            self.patch_atlas,
            step_patch_probabilities,
            self.umbilicus_zyx,
            compute_dt=compute_patch_dt,
            dt_target_cache=patch_dt_target_cache,
            crossing_map=self.theta_crossing_map,
            cfg=self.config,
        )
        patch_family = {
            'umbilicus': patch_loss_values[1] * self.config['loss_weight_umbilicus'],
        }
        if self.verified_patches_list:
            patch_family.update({
                'patch_radius': (
                    patch_loss_values[0]
                    * weight_patch_radius),
                'patch_dt': (
                    patch_loss_values[2]
                    * self.config['loss_weight_patch_dt']),
            })
        if pins_replace and self.verified_patches_list:
            # The replaced constraint loss stays visible, unweighted.
            log_metrics['patch_radius_unweighted'] = float(patch_loss_values[0].detach())
        backward_family(patch_family)
        del patch_family, patch_loss_values
        if pins_replace and unpinned_patches is not None and self.config['loss_weight_patch_radius'] > 0:
            # Verified patches without pins keep the soft radius loss.
            unpinned_values = get_patch_and_umbilicus_losses(
                self.slice_to_spiral_transform,
                self.dr_per_winding,
                min(int(self.config['sample_count_patches_per_step']), int(unpinned_patches.size)),
                0,
                self.verified_patches_list,
                self.patch_atlas,
                self._restrict_patch_probabilities(unpinned_patches),
                self.umbilicus_zyx,
                compute_dt=False,
                shell_valid_zyxs=None,
                shell_outer_winding_idx=self.shell_outer_winding_idx,
                crossing_map=self.theta_crossing_map,
                cfg=self.config,
            )
            backward_family({'patch_radius_unpinned': unpinned_values[0] * self.config['loss_weight_patch_radius']})
            del unpinned_values


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

        if weight_rel_winding > 0 and self.cross_patch_pcls:
            backward_family({
                'rel_winding': get_patch_rel_winding_loss(
                    self.slice_to_spiral_transform,
                    self.dr_per_winding,
                    self.verified_patches,
                    self.patch_atlas,
                    self.cross_patch_pcls,
                    self.pcl_sampling_strata['cross_patch'],
                    crossing_map=self.theta_crossing_map,
                    cfg=self.config, z_begin=self.z_begin, z_end=self.z_end,
                    rows=rel_winding_rows,
                ) * weight_rel_winding,
            })

        if weight_abs_winding > 0 and self.cross_patch_pcls:
            backward_family({
                'abs_winding': get_patch_abs_winding_loss(
                    self.slice_to_spiral_transform,
                    self.dr_per_winding,
                    self.verified_patches,
                    self.patch_atlas,
                    self.cross_patch_pcls,
                    crossing_map=self.theta_crossing_map,
                    cfg=self.config, z_begin=self.z_begin, z_end=self.z_end,
                    rows=abs_winding_rows,
                ) * weight_abs_winding,
            })

        if (
            ((self.dense_normals_enabled
              and self.config['loss_weight_dense_normals'] > 0)
             or self.grad_mag_spacing_enabled)
            and self.lasagna_volume is not None
        ):
            for dense_loss_name, dense_loss_value in iter_lasagna_losses(
                self.slice_to_spiral_transform,
                self.dr_per_winding,
                self.lasagna_volume,
                self.shell_outer_winding_idx,
                self.config['sample_count_dense_normal_points'],
                compute_spacing=self.grad_mag_spacing_enabled,
                compute_normals=(
                    self.dense_normals_enabled
                    and self.config['loss_weight_dense_normals'] > 0),
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

        if (self.config['loss_weight_fiber_directions'] > 0
                and self.fiber_direction_samples is not None):
            fiber_direction_loss = get_fiber_direction_loss(
                self.slice_to_spiral_transform,
                self.fiber_direction_samples,
                self.config['sample_count_fiber_direction_points'],
                self.config['fiber_directions_finite_difference_epsilon'],
                self.dr_per_winding.device,
            )
            backward_family({
                'fiber_directions': fiber_direction_loss
                * self.config['loss_weight_fiber_directions']
            })

        self._warn_if_density_loss_inactive()
        self._warn_if_dense_losses_structurally_disabled()
        if self._winding_model_mode_active():
            inference_losses, inference_metrics = get_winding_inference_losses(
                self.slice_to_spiral_transform,
                self.dr_per_winding,
                self.winding_inference,
                self.shell_map,
                self.config,
                self.z_begin,
                self.z_end,
                # Metrics are only reported every 200 steps
                # (log_step_metrics); computing them every step costs one
                # full GPU-queue drain per .item().
                with_metrics=iteration % 200 == 0,
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
        # The native min-spacing barrier is asset-independent; its weight is
        # re-read every step so it can be enabled at a Run boundary in any
        # dense-spacing mode.
        min_spacing_weight = float(self.config['loss_weight_min_spacing'])
        if min_spacing_weight > 0:
            min_spacing_loss, min_spacing_metrics = get_min_spacing_loss(
                self.spiral_and_transform,
                self.shell_outer_winding_idx,
                self.config,
                self.z_begin,
                self.z_end,
                # Metrics are only reported every 200 steps; their .item()
                # reads each stall on the full GPU queue.
                with_metrics=iteration % 200 == 0,
            )
            backward_family({'min_spacing': min_spacing_loss * min_spacing_weight})
            log_metrics.update(min_spacing_metrics)
            del min_spacing_loss, min_spacing_metrics

        if (
            (weight_unattached_radius > 0
             or self.config['loss_weight_unattached_pcl_dt'] > 0)
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
                compute_dt=compute_unattached_pcl_dt,
                dt_target_cache=unattached_pcl_dt_target_cache,
                crossing_map=self.theta_crossing_map,
                cfg=self.config,
            )
            backward_family({
                'unattached_pcl_radius': unattached_loss_values[0] * weight_unattached_radius,
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
        if (self.shell_map is not None
                and self.config['loss_weight_shell_outer'] > 0):
            shell_outer_loss, shell_metrics = get_shell_outer_loss(
                self.shell_map,
                self.slice_to_spiral_transform,
                self.dr_per_winding,
                self.shell_outer_winding_idx,
                cfg=self.config, z_begin=self.z_begin, z_end=self.z_end,
                # Metrics are only reported every 200 steps; the block's
                # valid.any() stalls on the full GPU queue.
                with_metrics=iteration % 200 == 0,
            )
            backward_family({
                'shell_outer': shell_outer_loss * self.config['loss_weight_shell_outer'],
            })
            del shell_outer_loss

        loss = sum(losses.values())

        self.step_timer.stop('fwd')
        self.step_timer.start('bwd')
        if pins_leaf is not None and pins_leaf.grad is not None:
            # The pins' graph runs through the flow chain, so it must be
            # propagated before the field's accumulated gradient is flushed.
            torch.autograd.backward([shared_transform_outputs[3]], [pins_leaf.grad])
        # Flush the sparse-accumulated field gradient into the flow parameters.
        self.spiral_and_transform.flow_field.apply_accumulated_field_grad()
        # Propagate the leaf gradients the family backwards accumulated on the
        # shared transform paths through the real parameters, exactly once.
        shared_transform_pending = [
            (output, leaf.grad)
            for output, leaf in zip(shared_transform_outputs[:3], shared_transform_leaves[:3])
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

        # Detect and zero nonfinite gradients straight after the all-reduce
        # (identical on every rank) and before the clipping and smoothing:
        # clamping would turn an infinity into a finite bound and hide it
        # from these counters, and smoothing would spread a single NaN over
        # every cell within its kernel. This does not check for overflow in
        # subsequent arithmetic or repair invalid parameter/moment state.
        self._sanitize_nonfinite_grads_()

        # Clip after the all-reduce (identical gradients and statistics on
        # every rank) and before smoothing, so a cell with an unsatisfiable
        # loss is bounded before its spike is spread over its neighbours.
        self.step_timer.start('clip')
        self._clip_flow_grads()
        self.step_timer.stop('clip')

        if self.config.get('optimizer_flow_grad_smoothing', False):
            # After the all-reduce (smoothing is linear, and every rank then
            # smooths identical gradients).
            self.step_timer.start('smooth')
            self.spiral_and_transform.smooth_flow_grad_(*self._flow_grad_smoothing_widths())
            self.step_timer.stop('smooth')

        # The unset-potential hard error is deferred off the sampler hot path
        # (theta_crossing_map.winding_potentials); resolve every pending
        # verdict before mutating parameters so an invalid batch can never
        # complete an optimizer update.
        self.theta_crossing_map.assert_no_pending_potential_errors()
        self.step_timer.start('opt')
        self.optimiser.step()
        self.step_timer.stop('opt')
        self.optimiser.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.step_timer.tick()
        self.step_timer.maybe_report(iteration)
        if self.profiler is not None:
            self.profiler.step()
        if pins_leaf is not None:
            log_metrics['pins_active'] = 1.0
            if log_pins_this_step:
                log_metrics.update(self._pin_step_metrics(pins_leaf.detach()))
        elif self.pin_graph is not None:
            log_metrics['pins_active'] = 0.0

        return loss, losses, log_metrics, shell_metrics

    def resolve_output_path(self):
        """Derive and create this run's output directory.

        Requires load_host_inputs() (the directory name records the verified
        patch count). Both drivers call this between load_host_inputs() and
        build_device_state().
        """
        if self.run_dir is not None:
            self.out_path = self.run_dir
            os.makedirs(self.out_path, exist_ok=True)
            return self.out_path

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
                conditioning_lines, conditioning_payload = self._flow_conditioning_report()
                for line in conditioning_lines:
                    print(line)
                interval = int(self.config.get('output_satisfaction_log_interval', 0) or 0)
                if interval > 0 and iteration % interval == 0 and self.verified_patches_list:
                    started = time.perf_counter()
                    satisfaction = self.periodic_satisfaction_metrics()
                    log_metrics.update(satisfaction)
                    print('  satisfaction: ' + ', '.join(
                        f'{k.replace("_satisfied_", " ").replace("_fraction", "")} = {v * 100:.1f}%'
                        for k, v in satisfaction.items()) + f'  ({time.perf_counter() - started:.0f}s)')
                pin_keys = ('pin_inexact_fraction', 'pin_rays_with_violation_fraction',
                            'pin_order_violations', 'pin_anchors_evaluated', 'pin_conflicts',
                            'pin_strain_median', 'pin_strain_p90', 'pin_strain_frac_over_margin',
                            'pin_T_frac_mean', 'patch_radius_unweighted',
                            'pair_agreement_median', 'pair_agreement_frac_over_margin')
                pin_items = [(k, log_metrics[k]) for k in pin_keys if k in log_metrics]
                if pin_items:
                    print('  pins: ' + ', '.join(
                        f'{k.removeprefix("pin_")} = {v:.3f}' if isinstance(v, float) and abs(v) < 1e4
                        else f'{k.removeprefix("pin_")} = {v:.0f}' for k, v in pin_items))
                payload = {
                    'total_loss': loss.item(),
                    **conditioning_payload,
                    'nonfinite_grad_steps': self.nonfinite_grad_steps.item(),
                    **{f'nonfinite_grad_steps/{name}': count.item() for name, count in self.nonfinite_grad_by_param.items()},
                    **{name + '_loss': value for name, value in losses.items()},
                    **shell_metrics,
                    **log_metrics,
                }
                metrics_history = os.environ.get('FIT_SPIRAL_METRICS_HISTORY')
                if metrics_history:
                    scalar_payload = {
                        name: (value.item() if hasattr(value, 'item') else value)
                        for name, value in payload.items()
                    }
                    with open(metrics_history, 'a') as stream:
                        stream.write(json.dumps({
                            'iteration': iteration,
                            'metrics': scalar_payload,
                        }) + '\n')
                if wandb.run is not None:
                    if metrics_history:
                        wandb.log(payload, step=iteration)
                    else:
                        wandb.log(payload)

    def run(self):
        """Drive one complete headless fit to the configured horizon.

        Interactive sessions are not driven here: spiral_runtime owns the
        context, its ready signal, and the resident optimizer loop.
        """
        progress = progress_or_null(self.progress)
        has_progress = self.progress is not None

        self.check_cuda_ready()
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
            self._maybe_save_headless_checkpoint(iteration + 1)

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
                slice_to_spiral_transform=self._export_transform(),
                dr_per_winding=self.dr_per_winding,
                patches_list=self.verified_patches_list,
                patches_dict=self.verified_patches,
                patch_atlas=self.patch_atlas,
                unattached_pcl_strips=self.unattached_pcl_strips,
                tracks=self.tracks,
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
        self.winding_inference = None


def main(config, *, scroll, paths, progress=None, resume_path=None,
         out_base_dir=None, run_dir=None, run_tag=None,
         run_name=None,
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
        out_base_dir=out_base_dir,
        run_dir=run_dir,
        run_tag=run_tag,
        run_name=run_name,
        cache_dir=cache_dir,
        storage_backend=storage_backend,
        render_volume_scale=render_volume_scale,
        dist_context=dist_context,
    ).run()


def _wandb_init_kwargs(config, mode, environment):
    """Build CLI W&B settings, including an explicitly requested group."""
    kwargs = {
        'project': environment.get('WANDB_PROJECT', 'scrolls'),
        'entity': environment.get('WANDB_ENTITY'),
        'config': config,
        'mode': mode,
    }
    if environment.get('FIT_SPIRAL_BATCH_RUN') == '1':
        kwargs.update({
            'id': environment['WANDB_RUN_ID'],
            'name': environment['WANDB_NAME'],
            'resume': ('allow'
                       if environment.get('FIT_SPIRAL_WANDB_RESUME') == '1'
                       else 'never'),
        })
        group = environment.get('WANDB_RUN_GROUP')
        if group is not None:
            kwargs['group'] = group
    return kwargs


def _finish_wandb_run():
    """Synchronously release a CLI run without making logging a fit failure."""
    try:
        if wandb.run is None:
            return True
        wandb.finish()
    except Exception as exc:
        print(
            f'WARNING: could not finish W&B run cleanly: '
            f'{type(exc).__name__}: {exc}',
            file=sys.stderr,
            flush=True)
        return False
    return True


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
            'sample_count_relative_winding_pcls',
            'sample_count_absolute_winding_pcls',
            'sample_count_unattached_pcls_per_step',
            'sample_count_tracks_per_step',
            'sample_count_dense_normal_points',
            'sample_count_fiber_direction_points',
            'sample_count_winding_model_relative_pairs',
            'sample_count_winding_model_density_pairs',
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
        wandb_init_kwargs = _wandb_init_kwargs(
            config, wandb_mode, os.environ)
        wandb.init(**wandb_init_kwargs)
        # The CLI boundary is where the FIT_SPIRAL_* fit controls are parsed;
        # FitContext itself no longer reads them.
        main(
            FitConfig(config),
            scroll=scroll_spec,
            paths=input_paths,
            progress=cli_progress,
            resume_path=os.environ.get('FIT_SPIRAL_RESUME_PATH'),
            out_base_dir=os.environ.get('FIT_SPIRAL_OUT_DIR'),
            run_dir=os.environ.get('FIT_SPIRAL_RUN_DIR'),
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
        _finish_wandb_run()
        if cli_progress is not None:
            cli_progress.close()
        maybe_destroy_distributed(dist_context)
