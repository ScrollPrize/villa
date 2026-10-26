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
import math
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
from config import (CHECKPOINT_MODEL_SHAPE_KEYS, OBJECT_STORE_PCL_KEYS,
                    Config, FitConfig, SHELL_ATLAS_KEYS)
from checkpoint_migrations import (expand_gap_checkpoint_capacity,
                                   tolerate_checkpoint_config)
from devices import fit_device
from fit_session import (AUTOSAVE_INTERVAL_ITERATIONS,
                         fit_input, input_source_enabled,
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
from tifxyz import load_tifxyz
from geom_utils import bilinear_atlas_lookup, grid_sample_border, interp1d
from pcl_views import (
    _UnattachedPclStripList,
    vertical_fiber_radial_offset,
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
    crossing_cache_in_fit_frame,
    filter_tracks_to_outer_shell,
    get_track_satisfied_counts_in_chunks,
    iter_track_losses,
    load_track_crossing_cache,
    load_tracks_from_dbm,
    prepare_main_phase_tracks,
    track_crossing_cache_path,
    track_store_z_range,
    tracks_in_fit_frame,
    validate_track_sampling_config,
)
from track_graph import TrackGraph
from umbilicus import thaumato_umbilicus_z_to_yx, json_umbilicus_z_to_yx
from sample_spiral import (
    get_radial_normal_stretch,
    get_spiral_points,
    get_theta,
    get_winding_xy,
)
from losses import (
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
    largest_patch_quad_component,
    patch_sampling_quad_mask,
    scale_and_split_counts,
    _infer_shell_outer_winding_idx,
    _structurally_disabled_dense_weight_keys,
    resolve_outer_winding_idx_and_notes,
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
from transforms import SpiralAndTransform
from theta_crossing_map import ThetaCrossingMap
from winding_supervision import (
    get_winding_inference_losses,
    load_winding_inference_store,
)
from spiral_progress import ProgressReporter, progress_or_null


configure_torch_threads_from_env()


_HEADLESS_AUTOSAVE_INTERVAL = AUTOSAVE_INTERVAL_ITERATIONS


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
        sampled = grid_sample_border(
            self.lookup_table[None],
            grid,
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


# Host staging bound for one PatchAtlas geometry upload, in bytes.
PATCH_ATLAS_STAGING_BYTES = 256 * (1 << 20)


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
        staging_limit = PATCH_ATLAS_STAGING_BYTES // (3 * 4)
        pieces = []
        staged = 0
        offset = 0

        def flush():
            nonlocal pieces, staged, offset
            if not pieces:
                return
            staging = torch.cat(pieces, dim=0)
            resident[offset:offset + staged].copy_(
                staging, non_blocking=device.type == 'cuda')
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
                    piece, non_blocking=device.type == 'cuda')
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
        # MPS reads a pageable host source after a non_blocking copy returns,
        # and these sources are often freed temporaries; CUDA stages them first.
        non_blocking = self.zyxs_flat.device.type == 'cuda'
        patch_idx_per_sample = patch_idx_per_sample.to(
            device=self.zyxs_flat.device, dtype=torch.int64,
            non_blocking=non_blocking)
        ijs = ijs.to(device=self.zyxs_flat.device, non_blocking=non_blocking)
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


def subsample_rows(points, max_points, generator):
    """Deterministically keep at most ``max_points`` rows of ``points``."""
    if points.shape[0] <= max_points:
        return points
    order = torch.randperm(points.shape[0], generator=generator)[:max_points]
    return points[order.sort().values]


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


# Settings that shape the frozen object-store inputs (the link refresh, the
# materialized constraints or the point views derived from them). The inputs
# are frozen for a fit, so apply_config refuses them at a Run boundary.
_OBJECT_STORE_FROZEN_KEYS = OBJECT_STORE_PCL_KEYS | {
    'input_use_pcl_relative',
    'input_use_pcl_same_winding',
}


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
                 render_volume_scale=16, dist_context=None,
                 object_store_inputs=None):
        # config is the explicit FitConfig (or any dict-style mapping of
        # fully resolved values) this fit reads; there is no module-global
        # fallback. The optimisation z window is Config's z_begin/z_end.
        #
        # scroll is the frozen ScrollSpec: physical facts of the scanned
        # scroll (name, voxel size, outward sense, umbilicus coordinate
        # scale, Lasagna group and scale). paths is the resolved
        # SpiralInputPaths for this fit: the interactive runtime passes the
        # service's per-session selection; the CLI resolves the conventional
        # dataset layout (fit_session.conventional_input_paths). Its patch,
        # fiber and PCL entries are not read: those inputs come from the
        # object store (config object_store_path).
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
        self.interactive_driver = interactive_driver
        # object_store_inputs.FitInputs materialized from a frozen object-store
        # snapshot: patches and point constraints with the store's links. When
        # None, load_host_inputs refreshes and reads the store the config
        # names (object_store_path).
        self.object_store_inputs = object_store_inputs
        self.progress = progress
        self.resume_path = resume_path or None
        self.out_base_dir = out_base_dir if out_base_dir is not None else './out'
        self.run_dir = run_dir or None
        self.run_tag = run_tag or None
        self.run_name = run_name
        self.non_liftable_patch_ids = set()
        # Who this process is in the job, as an explicit value. Callers that
        # joined a process group pass the context maybe_init_distributed()
        # returned; the default is the process context installed there (a
        # single-rank context when nothing joined one). Nothing below reads
        # RANK/WORLD_SIZE from the environment.
        self.dist = dist_context or process_context()
        # The device every session tensor lives on (devices.fit_device).
        self.device = fit_device()

        # Scroll physical facts.
        self.scroll_name = scroll.name
        self.voxel_size_um = float(scroll.voxel_size_um)
        self.base_shape_zyx = scroll.base_shape_zyx
        self.spiral_outward_sense = scroll.spiral_outward_sense
        # Decides whether exports are flipped so the scroll top is row 0.
        self.z_direction_is_top_to_bottom = scroll.z_direction_is_top_to_bottom
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
        self.fiber_directions_path = (
            (paths.fiber_directions or None)
            if input_source_enabled(config, 'fiber_directions') else None)
        self.shell_path = (
            (paths.outer_shell or None)
            if input_source_enabled(config, 'outer_shell') else None)
        self.tracks_dbm_path = (
            (paths.tracks_dbm or None)
            if input_source_enabled(config, 'tracks_dbm') else None)

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

    def _prepare_patch_sampling_cache(self, patches):
        progress = progress_or_null(self.progress)
        progress.begin(
            'loading', 'Preparing patch sampling',
            step=0, total_steps=len(patches), unit='patches')
        for patch_idx, patch in enumerate(patches):
            in_roi_quad_mask_np = patch_sampling_quad_mask(
                patch, self.z_begin, self.z_end,
                self.config['patch_loss_z_margin'])
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
        # A collection's explicit sampling_member_weight (patch-overlap pairs,
        # which share one edge's weight) takes precedence, and pairs sharing a
        # sampling_unit (their edge) are drawn edge first, then pair.
        self.pcl_sampling_strata['cross_patch'] = build_pcl_sampling_strata(
            [pcl['sampling_group'] if len(pcl['points']) > 1 else None
             for pcl in self.cross_patch_pcls],
            self.config,
            member_weights=[pcl.get('sampling_member_weight')
                            or len(pcl.get('link_member_cids', ())) or 1
                            for pcl in self.cross_patch_pcls],
            member_units=[pcl.get('sampling_unit')
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

    def _install_point_views(self, cross_patch_pcls, unattached_pcl_strips,
                             unattached_strip_sampling_groups, *,
                             resolved_links, link_components):
        """Install derived point-collection views and their sampling strata."""
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
        self.resolved_links = resolved_links
        self.link_components = link_components
        # The unattached loss consumes strips through these link components; see
        # _rebuild_unattached_components, which fills them from link_components.
        self.unattached_components = []
        self.unattached_component_groups = []
        self.unattached_component_edges = []
        self.pcl_sampling_strata = {}
        self._rebuild_pcl_sampling_strata()

    def check_cuda_ready(self):
        """Create the CUDA context before input reads increase host cache pressure."""
        if self.device.type != 'cuda':
            return
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

        Seeds the host RNG streams, then reads patches, point collections,
        fibers and their relationships from the object store
        (object_store_inputs.load_fit_inputs: one link refresh, then a frozen
        snapshot), loads tracks and the outer shell, classifies the point
        constraints; builds the sampling caches, host-prepared patch atlases, the
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
        - object_store_inputs: the materialized snapshot (provenance)
        - point collections: cross_patch_pcls, unattached_pcl_strips,
          unattached_strip_sampling_groups, pcl_sampling_strata,
          resolved_links, link_components,
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
        # Outer shell and object-store inputs
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

        # Patches, PCLs and fibers come only from the object store, linked by
        # its detectors; nothing here reads them from files or links them.
        if self.object_store_inputs is None:
            if not self.config['object_store_path']:
                raise RuntimeError(
                    'patches, fibers and point collections are read from an '
                    'object store: set object_store_path')
            self.object_store_inputs = self._load_object_store_inputs()
        verified_patches = dict(self.object_store_inputs.patches)
        print(f' loaded {len(verified_patches)} patches from the object store')
        self._derive_object_store_point_inputs(
            verified_patches, self.object_store_inputs)
        unattached_pcl_strips = self.unattached_pcl_strips

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
        # Stored track ZYX = fit ZYX x this; crossings are rediscovered from
        # the loaded coordinates only while it is 1 (prepare_main_phase_tracks).
        self.track_frame_scale = 1
        if self.tracks_dbm_path is not None:
            progress.begin(
                'loading', 'Resolving track store',
                detail=os.path.basename(self.tracks_dbm_path))
            print(f'loading tracks from {self.tracks_dbm_path}')
            # Object-store fits declare the track frame; tracks then load in
            # their stored frame (ROI included) and convert to fit voxels.
            track_scale = (self._object_store_track_scale()
                           if self.object_store_inputs is not None else 1)
            self.track_frame_scale = track_scale
            track_z_lo, track_z_hi = track_store_z_range(
                self.z_begin, self.z_end, track_scale)
            if track_scale != 1:
                print(f'tracks: stored coordinates = fit voxels x {track_scale:g}')
            if track_sampling_config['crossing_precompute_max'] > 0:
                track_crossing_cache = crossing_cache_in_fit_frame(
                    load_track_crossing_cache(self.tracks_dbm_path), track_scale)
                if track_crossing_cache is None and track_scale != 1:
                    raise RuntimeError(
                        'the track crossing cache is unusable; with '
                        f'track_scale_factor {track_scale:g} crossings come '
                        'only from it')
                if track_crossing_cache is not None:
                    track_graph = TrackGraph(track_crossing_cache)
                    print(
                        f'built TrackGraph: {len(track_graph)} tracks, '
                        f'{track_graph.edge_count} crossings in '
                        f'{track_graph.build_seconds:.1f}s')
                    track_crossing_cache = None
                tracks, track_families, track_source_ids = load_tracks_from_dbm(
                    self.tracks_dbm_path, track_z_lo, track_z_hi, return_families=True,
                    return_source_ids=True, progress=progress)
            else:
                tracks = load_tracks_from_dbm(
                    self.tracks_dbm_path, track_z_lo, track_z_hi, progress=progress)
            tracks = tracks_in_fit_frame(tracks, track_scale)
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
        patch_atlas = PatchAtlas(verified_patches, device=self.device)
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

    def _object_store_track_scale(self):
        """The declared track_scale_factor of an object-store fit attaching a
        track store (stored track ZYX = fit ZYX x it); None without tracks."""
        if not self.tracks_dbm_path:
            return None
        scale = self.config['track_scale_factor']
        if scale is None:
            raise RuntimeError(
                'a track store is attached to an object-store fit: declare '
                'track_scale_factor (stored track coordinates = fit-volume '
                'voxel coordinates x it); it is never inferred')
        scale = float(scale)
        if scale != 1 and self.config['track_crossing_precompute_max'] > 0:
            # Exact-voxel crossing discovery works on integer stored
            # coordinates: crossings must come from the stored-frame cache,
            # and point-level exclusion would rediscover them on scaled ones.
            if not track_crossing_cache_path(self.tracks_dbm_path).is_file():
                raise RuntimeError(
                    f'track_scale_factor {scale:g} needs the track store\'s '
                    'crossing cache (build_track_crossings.py), found in the '
                    'stored frame; crossings are not rediscovered in fit voxels')
            if float(self.config['track_exclusion_radius']) > 0:
                raise RuntimeError(
                    f'track_scale_factor {scale:g} with track crossings needs '
                    'track_exclusion_radius 0: exclusion would rediscover '
                    'crossings from scaled coordinates')
        return scale

    def _lasagna_normal_inputs(self):
        """The fit's lasagna normals as object_store_inputs.fiber_normal_source
        takes them, or None when the fit loads none."""
        if not (self.normal_nx_zarr_path and self.normal_ny_zarr_path):
            return None
        return {'normal_x': self.normal_nx_zarr_path,
                'normal_y': self.normal_ny_zarr_path,
                'group': self.normal_zarr_group, 'scale': self.lasagna_scale}

    def _external_input_frames(self):
        """Frames of the inputs read outside the object store, for the input
        manifest. Each is declared in fit-volume voxels: the umbilicus by its
        coordinate scale, lasagna stores by their downsampling, tracks by
        track_scale_factor; the others by their existing contract."""
        frames = {'umbilicus': {
            'coordinate_scale': float(self.scroll.umbilicus_coordinate_scale)}}
        if self.normal_nx_zarr_path or self.grad_mag_zarr_path:
            frames['lasagna'] = {'group': str(self.normal_zarr_group),
                                 'downsampling': int(self.lasagna_scale)}
        if self.tracks_dbm_path:
            frames['tracks'] = {'scale_factor': self._object_store_track_scale()}
        for name, path in (('outer_shell', self.shell_path),
                           ('fiber_directions', self.fiber_directions_path),
                           ('winding_inference', self.winding_inference_path)):
            if path:
                frames[name] = {'units': 'fit-volume voxels'}
        return frames

    def _load_object_store_inputs(self):
        """Refresh and freeze this fit's object-store inputs
        (object_store_inputs.load_fit_inputs) and report them."""
        from object_store_inputs import load_fit_inputs
        progress_or_null(self.progress).begin(
            'loading', 'Reading object-store inputs',
            detail=os.path.basename(str(self.config['object_store_path'])))
        cache = (os.path.join(self.cache_path, 'object_store_strips')
                 if self.cache_path else None)
        return load_fit_inputs(
            self.config, lasagna=self._lasagna_normal_inputs(),
            external_frames=self._external_input_frames(), dist=self.dist,
            cache_directory=cache)

    def _derive_object_store_point_inputs(self, verified_patches, inputs):
        """Install point views derived from object-store attachments.

        Uses object_store_inputs.derive_point_views, as the constraints
        report does, so both apply the same validation.
        """
        from object_store_inputs import derive_point_views
        views = derive_point_views(inputs, verified_patches, self.config)
        for line in inputs.report() + views.report():
            print(line)
        self._install_point_views(
            views.cross_patch, views.strips, views.strip_groups,
            resolved_links=views.resolved_links,
            link_components=views.link_components)

    def _vertical_fiber_radial_offset(self):
        """Radial target offset (voxels) applied to vertical fiber strips; 0 when off."""
        return vertical_fiber_radial_offset(self.config)

    def _refresh_trusted_geometry(self):
        """Rebuild the trusted-geometry tree from the active patches and strips."""
        trusted = self._trusted_geometry_from_active_inputs()
        trusted_np = np.ascontiguousarray(
            trusted.cpu().numpy(), dtype=np.float32)
        self.trusted_geometry_tree = (
            cKDTree(trusted_np) if len(trusted_np) else None)

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
                  - self.spiral_and_transform.flow_min_corner_zyx).cpu().to(torch.float64)
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
            device=self.device,
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
                self.device,
                verify=os.environ.get(
                    'FIT_SPIRAL_VERIFY_WINDING_INFERENCE', '1') != '0',
                z_range=(self.z_begin, self.z_end),
                z_to_umbilicus_yx=self.umbilicus_z_to_yx(),
            )
            print(
                'loaded winding inference: '
                f"{self.winding_inference.fingerprint['num_rays']:,} rays "
                f"({self.winding_inference.num_z_eligible_rays:,} intersect "
                f"z-range [{self.z_begin}, {self.z_end})), "
                f"{self.winding_inference.fingerprint['num_crossings']:,} "
                'crossings, '
                f"{self.winding_inference.num_inward_rays:,} inward rays "
                'flipped')

        self._density_inactive_warned = set()

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

    def _write_non_liftable_patch_report(self):
        """Atomically publish the cumulative rejected patch UUID list."""
        if not self.dist.is_main_process or not hasattr(self, 'out_path'):
            return
        report_path = os.path.join(self.out_path, 'non_liftable_patches.txt')
        temporary_path = f'{report_path}.tmp-{os.getpid()}'
        with open(temporary_path, 'w', encoding='utf-8') as report:
            for patch_id in sorted(self.non_liftable_patch_ids):
                report.write(f'{patch_id}\n')
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
            self.non_liftable_patch_ids.add(patch_id)
            warning = (
                f'non-liftable patch {patch_id} has theta cycle inconsistencies; '
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
        grouped_ids = {id(p) for p in flow_field_params + self.gap_expander_params + linear_params}
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
                rediscover_crossings=getattr(self, 'track_frame_scale', 1) == 1,
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
        'run_dt_resume_iteration',
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
        if self.device.type == 'mps':
            torch.mps.empty_cache()
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
            'z_direction_is_top_to_bottom': self.z_direction_is_top_to_bottom,
            'numpy_rng_state': np.random.get_state(),
            'torch_cpu_rng_state': torch.random.get_rng_state(),
            'torch_cuda_rng_states': torch.cuda.get_rng_state_all(),
            'input_manifest': self._checkpoint_input_manifest(),
            'preview_first_winding': 10,
            **({'torch_mps_rng_state': torch.mps.get_rng_state()}
               if self.device.type == 'mps' else {}),
        }

    def _checkpoint_input_manifest(self):
        """The interactive input manifest, plus the frozen object-store
        inputs' manifest and digest when the fit holds them."""
        manifest = dict(getattr(self.interactive_driver, 'input_manifest', {}))
        inputs = getattr(self, 'object_store_inputs', None)
        if inputs is not None and inputs.manifest is not None:
            manifest['object_store'] = inputs.manifest
            manifest['object_store_digest'] = inputs.digest
        return manifest

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
        # Checkpoints from before the z direction was saved carry no key.
        if ('z_direction_is_top_to_bottom' in checkpoint
                and checkpoint['z_direction_is_top_to_bottom']
                != self.z_direction_is_top_to_bottom):
            reasons.append(
                f'checkpoint z direction (top to bottom: '
                f'{checkpoint["z_direction_is_top_to_bottom"]!r}) does not '
                f'match the scroll spec '
                f'({self.z_direction_is_top_to_bottom!r})')
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
        inputs = getattr(self, 'object_store_inputs', None)
        if inputs is not None and inputs.manifest is not None:
            from object_store_inputs import resume_problems
            reasons.extend(resume_problems(
                checkpoint.get('input_manifest'), inputs,
                allow_changed=self.config['object_store_allow_changed_inputs']))
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
        if (checkpoint.get('torch_mps_rng_state') is not None
                and self.device.type == 'mps'):
            torch.mps.set_rng_state(checkpoint['torch_mps_rng_state'])

    def load_checkpoint(self, checkpoint):
        checkpoint, _ = tolerate_checkpoint_config(checkpoint)
        checkpoint = expand_gap_checkpoint_capacity(
            checkpoint, self.config['model_gap_expander_capacity_windings'])
        transformed_spiral_state, optimiser_state = checkpoint['spiral_and_transform'], checkpoint['optimiser']
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
        if checkpoint.get('scheduler') is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])

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
        mps_state = torch.mps.get_rng_state() if self.device.type == 'mps' else None
        try:
            # The preview surface must land in true scroll space, so after
            # any constraint bake it is pulled back through the composed
            # frozen+live chain; the resident inputs that bound its winding
            # range are in baked space and are read through the live chain.
            live_transform = \
                self.spiral_and_transform.get_slice_to_spiral_transform()
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
                z_direction_is_top_to_bottom=self.z_direction_is_top_to_bottom,
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
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
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
        if self.device.type == 'mps':
            driver_before = torch.mps.driver_allocated_memory()
            gc.collect()
            torch.mps.empty_cache()
            print(
                f'preview export: released '
                f'{(driver_before - torch.mps.driver_allocated_memory()) / 1024 ** 3:.2f} GiB '
                f'of cached MPS memory before publication', flush=True)
            return
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

    def apply_config(self, config, path_changes=None, *, current_iteration):
        """Apply Run-scoped settings without replacing the resident fit.

        current_iteration is the session's durable completed-iteration count;
        an LR-schedule change is realigned at that step.
        """
        path_changes = dict(path_changes or {})
        changed = set(config)
        frozen = changed & _OBJECT_STORE_FROZEN_KEYS
        if frozen:
            raise ValueError(
                'object-store inputs are frozen for this fit; '
                f'{sorted(frozen)} would relink, reload or re-derive them')
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
                    progress=self.progress,
                    rediscover_crossings=getattr(self, 'track_frame_scale', 1) == 1)
                replace_prepared_tracks = True

            target_tracks = (
                rebuilt_tracks
                if replace_prepared_tracks else self.prepared_main_tracks)
            if ({'track_length_bin_weights',
                 'track_max_track_crossing_per_step'}
                    & changed):
                configure_prepared_track_sampling(target_tracks, config)

            if rebuild_strata:
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
        shared_transform_outputs = self.spiral_and_transform.get_shared_transform_tensors()
        shared_transform_leaves = tuple(
            output.detach().requires_grad_(True) for output in shared_transform_outputs)
        self.slice_to_spiral_transform = self.spiral_and_transform.get_slice_to_spiral_transform(
            shared=shared_transform_leaves)
        self.dr_per_winding = shared_transform_leaves[0]
        theta_map_refreshed = self._refresh_theta_crossing_map_for_step(
            iteration,
            self.slice_to_spiral_transform)

        losses = {}
        log_metrics = {
            'flow_field_low_res_lr_scale': flow_field_low_res_lr_scale,
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
                ))
            if compute_unattached_pcl_dt and self.config['loss_weight_unattached_pcl_dt'] > 0 and self.unattached_pcl_strips:
                pcl_flat = get_or_build_unattached_pcl_flat(self.unattached_pcl_strips, self.device)
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
                    * self.config['loss_weight_patch_radius']),
                'patch_dt': (
                    patch_loss_values[2]
                    * self.config['loss_weight_patch_dt']),
            })
        backward_family(patch_family)
        del patch_family, patch_loss_values


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
                    crossing_map=self.theta_crossing_map,
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
                    crossing_map=self.theta_crossing_map,
                    cfg=self.config, z_begin=self.z_begin, z_end=self.z_end,
                ) * self.config['loss_weight_abs_winding'],
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
            (self.config['loss_weight_unattached_pcl_radius'] > 0
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
        # Flush the sparse-accumulated field gradient into the flow parameters.
        self.spiral_and_transform.flow_field.apply_accumulated_field_grad()
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

    def _write_object_store_manifest(self):
        """Record the frozen object-store inputs' manifest beside the run's
        outputs (rank zero); checkpoints carry it too."""
        inputs = self.object_store_inputs
        if inputs is None or inputs.manifest is None or not self.dist.is_main_process:
            return
        path = os.path.join(self.out_path, 'object_store_inputs.json')
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump({'digest': inputs.digest, 'manifest': inputs.manifest},
                      stream, indent=1, sort_keys=True)
        print(f'object-store input manifest: {path}')

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
        self._write_object_store_manifest()
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
                slice_to_spiral_transform=self.slice_to_spiral_transform,
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
                z_direction_is_top_to_bottom=self.z_direction_is_top_to_bottom,
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
                             load_scroll_spec, scroll_spec_config_defaults)

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
        # A winding count in spiral-scroll.json replaces the generic default;
        # FIT_SPIRAL_CONFIG_OVERRIDES still wins over it.
        config.update(scroll_spec_config_defaults(scroll_spec, config))
        config.update(get_env_config_overrides())
        if not config['object_store_path']:
            raise SystemExit(
                'the command-line fit reads patches, fibers and point '
                'collections from an object store: set object_store_path '
                '(FIT_SPIRAL_CONFIG_OVERRIDES)')
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
