"""TifxyzLasagnaDataset — PyTorch Dataset for lasagna 3D UNet training from tifxyz surfaces.

Produces training patches where:
- CT volume crops are read from zarr (CPU)
- Surface masks and direction channels are voxelized from tifxyz grids (CPU)
- EDT, chain ordering, cos/grad_mag/validity derivation happens on GPU in the train step

Uses helpers from vesuvius neural tracing for patch finding, surface extraction,
and voxelization.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import vesuvius.tifxyz as tifxyz
from vesuvius.neural_tracing.datasets.patch_finding import find_world_chunk_patches
from vesuvius.neural_tracing.datasets.common import (
    ChunkPatch,
    open_zarr,
    _parse_z_range,
    _read_volume_crop_from_patch,
    _segment_overlaps_z_range,
    _trim_to_world_bbox,
    _upsample_world_triplet,
    voxelize_surface_grid_masked,
)

try:
    from numba import njit
except Exception:
    njit = None


TAG = "[tifxyz_lasagna_dataset]"


# ---------------------------------------------------------------------------
# Multi-channel trilinear splatting
# ---------------------------------------------------------------------------

if njit is not None:
    @njit(cache=True)
    def _splat_multichannel_trilinear_numba(points, values, size_z, size_y, size_x, n_channels):
        """Splat (N, n_channels) values at (N, 3) ZYX positions into a volume.

        Returns (n_channels, size_z, size_y, size_x) float32.
        """
        vox = np.zeros((n_channels, size_z, size_y, size_x), dtype=np.float32)
        weights = np.zeros((size_z, size_y, size_x), dtype=np.float32)
        n_points = points.shape[0]
        for i in range(n_points):
            pz = points[i, 0]
            py = points[i, 1]
            px = points[i, 2]
            if not (np.isfinite(pz) and np.isfinite(py) and np.isfinite(px)):
                continue

            z0 = int(np.floor(pz))
            y0 = int(np.floor(py))
            x0 = int(np.floor(px))
            dz = pz - z0
            dy = py - y0
            dx = px - x0

            for oz in range(2):
                zi = z0 + oz
                if zi < 0 or zi >= size_z:
                    continue
                wz = (1.0 - dz) if oz == 0 else dz
                if wz <= 0.0:
                    continue
                for oy in range(2):
                    yi = y0 + oy
                    if yi < 0 or yi >= size_y:
                        continue
                    wy = (1.0 - dy) if oy == 0 else dy
                    if wy <= 0.0:
                        continue
                    for ox in range(2):
                        xi = x0 + ox
                        if xi < 0 or xi >= size_x:
                            continue
                        wx = (1.0 - dx) if ox == 0 else dx
                        if wx <= 0.0:
                            continue
                        w = wz * wy * wx
                        weights[zi, yi, xi] += w
                        for c in range(n_channels):
                            vox[c, zi, yi, xi] += w * values[i, c]

        # Normalize by accumulated weight
        for zi in range(size_z):
            for yi in range(size_y):
                for xi in range(size_x):
                    if weights[zi, yi, xi] > 0:
                        for c in range(n_channels):
                            vox[c, zi, yi, xi] /= weights[zi, yi, xi]
        return vox
else:
    _splat_multichannel_trilinear_numba = None


def _splat_multichannel(points_zyx, values, crop_size):
    """Splat multi-channel values at 3D positions into a volume.

    Args:
        points_zyx: (N, 3) float32 — local ZYX positions
        values: (N, C) float32 — channel values per point
        crop_size: (Z, Y, X) int tuple

    Returns:
        (C, Z, Y, X) float32 — splatted volume
        (Z, Y, X) float32 — weight accumulator (>0 where splatted)
    """
    crop_size = tuple(int(v) for v in crop_size)
    N = points_zyx.shape[0]
    C = values.shape[1]

    if N == 0:
        return (
            np.zeros((C,) + crop_size, dtype=np.float32),
            np.zeros(crop_size, dtype=np.float32),
        )

    # Filter non-finite points
    finite = np.isfinite(points_zyx).all(axis=1) & np.isfinite(values).all(axis=1)
    points_zyx = np.ascontiguousarray(points_zyx[finite], dtype=np.float32)
    values = np.ascontiguousarray(values[finite], dtype=np.float32)

    if points_zyx.shape[0] == 0:
        return (
            np.zeros((C,) + crop_size, dtype=np.float32),
            np.zeros(crop_size, dtype=np.float32),
        )

    if _splat_multichannel_trilinear_numba is not None:
        vox = _splat_multichannel_trilinear_numba(
            points_zyx, values,
            crop_size[0], crop_size[1], crop_size[2], C,
        )
        # Recompute weight for the mask (any non-zero channel)
        weight = np.zeros(crop_size, dtype=np.float32)
        weight[np.any(np.abs(vox) > 0, axis=0)] = 1.0
        return vox, weight

    # Fallback: numpy (slower but functional)
    vox = np.zeros((C,) + crop_size, dtype=np.float32)
    weights = np.zeros(crop_size, dtype=np.float32)
    base = np.floor(points_zyx).astype(np.int64)
    frac = points_zyx - base.astype(np.float32)

    for oz in (0, 1):
        z_idx = base[:, 0] + oz
        wz = (1.0 - frac[:, 0]) if oz == 0 else frac[:, 0]
        for oy in (0, 1):
            y_idx = base[:, 1] + oy
            wy = (1.0 - frac[:, 1]) if oy == 0 else frac[:, 1]
            for ox in (0, 1):
                x_idx = base[:, 2] + ox
                wx = (1.0 - frac[:, 2]) if ox == 0 else frac[:, 2]
                w = wz * wy * wx
                valid = (
                    (w > 0)
                    & (z_idx >= 0) & (z_idx < crop_size[0])
                    & (y_idx >= 0) & (y_idx < crop_size[1])
                    & (x_idx >= 0) & (x_idx < crop_size[2])
                )
                if np.any(valid):
                    zi = z_idx[valid]
                    yi = y_idx[valid]
                    xi = x_idx[valid]
                    wv = w[valid].astype(np.float32)
                    np.add.at(weights, (zi, yi, xi), wv)
                    for c in range(C):
                        np.add.at(vox[c], (zi, yi, xi), wv * values[valid, c])

    # Normalize
    nonzero = weights > 0
    for c in range(C):
        vox[c][nonzero] /= weights[nonzero]
    return vox, (weights > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Direction channel encoding (numpy, for CPU splatting)
# ---------------------------------------------------------------------------

def _encode_dir_np(gx, gy):
    """Double-angle direction encoding (numpy)."""
    eps = 1e-8
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    r2 = gx * gx + gy * gy + eps
    cos2t = (gx * gx - gy * gy) / r2
    sin2t = 2.0 * gx * gy / r2
    d0 = 0.5 + 0.5 * cos2t
    d1 = 0.5 + 0.5 * (cos2t - sin2t) * inv_sqrt2
    return d0.astype(np.float32), d1.astype(np.float32)


def compute_direction_values(normals_zyx):
    """Compute 6 direction channel values from ZYX normals.

    Args:
        normals_zyx: (N, 3) or (H, W, 3) float32 — normals in ZYX order

    Returns:
        (N, 6) or (H, W, 6) float32 — [dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x]
    """
    orig_shape = normals_zyx.shape[:-1]
    normals = normals_zyx.reshape(-1, 3)
    nz, ny, nx = normals[:, 0], normals[:, 1], normals[:, 2]

    dir0_z, dir1_z = _encode_dir_np(nx, ny)   # Z-slices (XY plane)
    dir0_y, dir1_y = _encode_dir_np(nx, nz)   # Y-slices (XZ plane)
    dir0_x, dir1_x = _encode_dir_np(ny, nz)   # X-slices (YZ plane)

    result = np.stack([dir0_z, dir1_z, dir0_y, dir1_y, dir0_x, dir1_x], axis=-1)
    return result.reshape(*orig_shape, 6)


# ---------------------------------------------------------------------------
# Surface normal estimation from grid
# ---------------------------------------------------------------------------

def _estimate_grid_normals(zyx_grid):
    """Estimate surface normals from a (H, W, 3) ZYX grid via cross product of tangent vectors."""
    tangent_r = np.zeros_like(zyx_grid)
    tangent_r[:-1] = zyx_grid[1:] - zyx_grid[:-1]
    tangent_r[-1] = tangent_r[-2]
    tangent_c = np.zeros_like(zyx_grid)
    tangent_c[:, :-1] = zyx_grid[:, 1:] - zyx_grid[:, :-1]
    tangent_c[:, -1] = tangent_c[:, -2]
    normals = np.cross(tangent_r, tangent_c)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6
    return normals / norm


# ---------------------------------------------------------------------------
# Patch finding (world-chunk method)
# ---------------------------------------------------------------------------

def _find_patches_world_chunks(config, patch_size_zyx):
    """Find training patches using the world-chunk tiling method.

    Follows the pattern from dataset_rowcol_cond.py, using find_world_chunk_patches
    from the neural tracing pipeline.
    """
    # Defaults from dataset_defaults.py:50-62
    overlap_fraction = float(config.get("overlap_fraction", 0.0))
    min_span_ratio = float(config.get("min_span_ratio", 1.0))
    edge_touch_frac = float(config.get("edge_touch_frac", 0.1))
    edge_touch_min_count_base = int(config.get("edge_touch_min_count", 10))
    edge_touch_pad = int(config.get("edge_touch_pad", 0))
    min_points_per_wrap_base = int(config.get("min_points_per_wrap", 100))
    scale_normalize = bool(config.get("scale_normalize_patch_counts", True))
    ref_scale = int(config.get("patch_count_reference_scale", 0))
    bbox_pad_2d = int(config.get("bbox_pad_2d", 0))
    require_all_valid = bool(config.get("require_all_valid_in_bbox", True))
    skip_invalid = bool(config.get("skip_chunk_if_any_invalid", False))
    inner_bbox_fraction = float(config.get("inner_bbox_fraction", 0.7))
    force_recompute = bool(config.get("force_recompute_patches", False))
    chunk_pad = float(config.get("chunk_pad", 0.0))
    verbose = bool(config.get("verbose", False))

    target_size = tuple(int(v) for v in patch_size_zyx)

    patches = []
    for dataset_idx, dataset in enumerate(config["datasets"]):
        volume_path = dataset.get("volume_path")
        if volume_path is None:
            continue

        volume_scale = int(dataset["volume_scale"])
        segments_path = dataset.get("segments_path")
        if not segments_path:
            continue

        # Open zarr volume (handles local, S3, HTTPS with caching)
        volume_auth_json = dataset.get(
            "volume_auth_json", config.get("volume_auth_json")
        )
        volume = open_zarr(
            volume_path, scale=volume_scale,
            auth_json_path=volume_auth_json, config=config,
        )

        # Load and retarget segments. Per neural tracer convention, z_range
        # is specified in retargeted (volume) coordinate space — same space
        # as segments AFTER retarget — so we do NOT scale it here. Users
        # whose volume_scale differs from what their z_range targets should
        # either align volume_scale or update z_range to match.
        retarget_factor = 2 ** volume_scale
        z_range = _parse_z_range(dataset.get("z_range"))
        dataset_segments = list(tifxyz.load_folder(segments_path))
        scaled_segments = []
        dropped_by_z_range = 0
        for seg in dataset_segments:
            seg_scaled = seg.retarget(retarget_factor)
            if not _segment_overlaps_z_range(seg_scaled, z_range):
                dropped_by_z_range += 1
                continue
            seg_scaled.volume = volume
            scaled_segments.append(seg_scaled)

        if not scaled_segments:
            warnings.warn(
                f"No segments remain after z_range filtering for dataset_idx={dataset_idx} "
                f"(segments_path={segments_path}, z_range={z_range}); skipping."
            )
            continue

        # Scale-normalize patch counts (dataset_rowcol_cond.py:297-309)
        if scale_normalize:
            count_scale = float(2 ** (volume_scale - ref_scale))
            count_scale_sq = count_scale * count_scale
        else:
            count_scale_sq = 1.0

        min_points_per_wrap = max(
            1, int(round(min_points_per_wrap_base * count_scale_sq))
        )
        edge_touch_min_count = max(
            1, int(round(edge_touch_min_count_base * count_scale_sq))
        )

        # Find world-chunk patches
        cache_dir = Path(segments_path) / ".patch_cache"
        chunk_results = find_world_chunk_patches(
            segments=scaled_segments,
            target_size=target_size,
            overlap_fraction=overlap_fraction,
            min_span_ratio=min_span_ratio,
            edge_touch_frac=edge_touch_frac,
            edge_touch_min_count=edge_touch_min_count,
            edge_touch_pad=edge_touch_pad,
            min_points_per_wrap=min_points_per_wrap,
            bbox_pad_2d=bbox_pad_2d,
            require_all_valid_in_bbox=require_all_valid,
            skip_chunk_if_any_invalid=skip_invalid,
            inner_bbox_fraction=inner_bbox_fraction,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            verbose=verbose,
            chunk_pad=chunk_pad,
        )

        # Convert chunk dicts to ChunkPatch objects (dataset_rowcol_cond.py:332-349)
        for chunk in chunk_results:
            wraps_in_chunk = []
            for w in chunk["wraps"]:
                seg_idx = w["segment_idx"]
                wraps_in_chunk.append({
                    "segment": scaled_segments[seg_idx],
                    "bbox_2d": tuple(w["bbox_2d"]),
                    "wrap_id": w["wrap_id"],
                    "segment_idx": seg_idx,
                })

            patches.append(ChunkPatch(
                chunk_id=tuple(chunk["chunk_id"]),
                volume=volume,
                scale=volume_scale,
                world_bbox=tuple(chunk["bbox_3d"]),
                wraps=wraps_in_chunk,
                segments=scaled_segments,
            ))

    return patches


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TifxyzLasagnaDataset(Dataset):
    """Dataset that derives lasagna training channels from tifxyz surfaces.

    Each __getitem__ returns:
        - vol_crop: (1, Z, Y, X) float32 — z-score normalized CT crop
        - surface_masks: (N, Z, Y, X) float32 — per-surface binary voxelization
        - direction_channels: (6, Z, Y, X) float32 — splatted direction values
        - normals_valid: (1, Z, Y, X) float32 — where directions were splatted
        - num_surfaces: int
        - padding_mask: (1, Z, Y, X) float32 — where CT data exists

    GPU label derivation (EDT, chain, cos/grad_mag) happens in the train step
    via tifxyz_labels.compute_patch_labels().
    """

    def __init__(
        self,
        config: dict,
        apply_augmentation: bool = True,
    ):
        # Normalize patch_size to 3-element ZYX array
        patch_size_zyx = np.asarray(config["patch_size"], dtype=np.int32).reshape(-1)
        if patch_size_zyx.size == 1:
            patch_size_zyx = np.repeat(patch_size_zyx, 3)
        self.patch_size_zyx = patch_size_zyx

        self.max_surfaces_per_patch = int(config.get("max_surfaces_per_patch", 8))

        # Augmentation
        if apply_augmentation:
            from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
            self.augmentations = create_training_transforms(
                patch_size=tuple(int(v) for v in self.patch_size_zyx),
                no_spatial=False,
            )
        else:
            self.augmentations = None

        # Find patches using world-chunk method
        self.patches = _find_patches_world_chunks(config, self.patch_size_zyx)

        print(f"{TAG} loaded {len(self.patches)} patches")

    def __len__(self):
        return len(self.patches)

    def _extract_and_voxelize_wrap(self, patch, wrap, min_corner, crop_size):
        """Extract a wrap surface, voxelize it, and compute direction channels.

        Args:
            patch: ChunkPatch containing the volume and world_bbox
            wrap: dict with "segment", "bbox_2d", etc.
            min_corner: (3,) int64 array — ZYX origin of the crop
            crop_size: (Z, Y, X) int tuple

        Returns:
            surface_mask: (Z, Y, X) float32 — binary voxelization
            dir_points_local: (M, 3) float32 — local ZYX positions for splatting
            dir_values: (M, 6) float32 — direction channel values
        """
        crop_size_tuple = tuple(int(v) for v in crop_size)
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        empty_vals = np.zeros((0, 6), dtype=np.float32)
        empty_mask = np.zeros(crop_size_tuple, dtype=np.float32)

        seg = wrap["segment"]
        r_min, r_max, c_min, c_max = wrap["bbox_2d"]

        # Clamp to segment bounds
        seg_h, seg_w = seg._valid_mask.shape
        r_min = max(0, r_min)
        r_max = min(seg_h - 1, r_max)
        c_min = max(0, c_min)
        c_max = min(seg_w - 1, c_max)
        if r_max < r_min or c_max < c_min:
            return empty_mask, empty_pts, empty_vals

        # Read stored-resolution grid
        seg.use_stored_resolution()
        scale_y, scale_x = seg._scale
        x_s, y_s, z_s, valid_s = seg[r_min:r_max + 1, c_min:c_max + 1]
        if x_s.size == 0:
            return empty_mask, empty_pts, empty_vals

        # Skip wraps with invalid cells in stored grid (upsampling requires all-valid)
        if valid_s is not None and not valid_s.all():
            return empty_mask, empty_pts, empty_vals

        # Upsample to full resolution
        try:
            x_full, y_full, z_full = _upsample_world_triplet(
                x_s, y_s, z_s, scale_y, scale_x,
            )
        except ValueError:
            return empty_mask, empty_pts, empty_vals

        # Trim to world bbox
        trimmed = _trim_to_world_bbox(x_full, y_full, z_full, patch.world_bbox)
        if trimmed is None:
            return empty_mask, empty_pts, empty_vals
        x_full, y_full, z_full = trimmed

        # Build (H, W, 3) ZYX grid in world coordinates
        zyx_world = np.stack(
            [z_full, y_full, x_full], axis=-1,
        ).astype(np.float32)

        # Convert to local coordinates
        min_corner_f = min_corner.astype(np.float32)
        zyx_local = zyx_world - min_corner_f

        # Compute validity mask (finite + within crop bounds)
        valid = (
            np.isfinite(zyx_local).all(axis=-1)
            & (zyx_local[..., 0] >= -0.5)
            & (zyx_local[..., 0] < crop_size_tuple[0] - 0.5)
            & (zyx_local[..., 1] >= -0.5)
            & (zyx_local[..., 1] < crop_size_tuple[1] - 0.5)
            & (zyx_local[..., 2] >= -0.5)
            & (zyx_local[..., 2] < crop_size_tuple[2] - 0.5)
        )

        if not np.any(valid):
            return empty_mask, empty_pts, empty_vals

        # Voxelize using neural tracing's line-drawing rasterizer
        surface_mask = voxelize_surface_grid_masked(
            zyx_local, crop_size_tuple, valid,
        )

        # Compute normals from the grid
        normals_zyx = _estimate_grid_normals(zyx_world)

        # Direction encoding from normals
        valid_for_dir = valid & np.isfinite(normals_zyx).all(axis=-1)
        if not np.any(valid_for_dir):
            return surface_mask, empty_pts, empty_vals

        pts_local = zyx_local[valid_for_dir].astype(np.float32)
        dir_vals = compute_direction_values(
            normals_zyx[valid_for_dir].astype(np.float32),
        )

        return surface_mask, pts_local, dir_vals

    def __getitem__(self, idx):
        patch = self.patches[idx]

        z0, z1, y0, y1, x0, x1 = patch.world_bbox
        crop_size = tuple(int(v) for v in self.patch_size_zyx)
        # world_bbox is half-open: [z0, z1), so max_corner = min_corner + crop_size
        min_corner = np.array([z0, y0, x0], dtype=np.int64)
        max_corner = min_corner + np.array(crop_size, dtype=np.int64)

        # Read CT crop (z-score normalized)
        vol_crop = _read_volume_crop_from_patch(
            patch, crop_size=crop_size,
            min_corner=min_corner, max_corner=max_corner,
        )

        # Per-surface: voxelize mask and compute direction channels
        surface_masks = []
        all_dir_points = []
        all_dir_values = []

        for wrap in patch.wraps[:self.max_surfaces_per_patch]:
            mask, dir_pts, dir_vals = self._extract_and_voxelize_wrap(
                patch, wrap, min_corner, crop_size,
            )
            if np.any(mask > 0):
                surface_masks.append(mask)
                if dir_pts.shape[0] > 0:
                    all_dir_points.append(dir_pts)
                    all_dir_values.append(dir_vals)

        num_surfaces = len(surface_masks)

        # Stack surface masks: (N, Z, Y, X)
        if num_surfaces > 0:
            surface_masks_arr = np.stack(surface_masks, axis=0)
        else:
            surface_masks_arr = np.zeros((0,) + crop_size, dtype=np.float32)

        # Splat direction channels from all surfaces combined: (6, Z, Y, X)
        if all_dir_points:
            pts = np.concatenate(all_dir_points, axis=0)
            vals = np.concatenate(all_dir_values, axis=0)
            direction_channels, normals_valid_vol = _splat_multichannel(
                pts, vals, crop_size,
            )
        else:
            direction_channels = np.zeros((6,) + crop_size, dtype=np.float32)
            normals_valid_vol = np.zeros(crop_size, dtype=np.float32)

        # Padding mask: where CT data actually exists (non-zero after crop)
        padding_mask = np.ones(crop_size, dtype=np.float32)

        # Convert to tensors
        vol_crop_t = torch.as_tensor(
            np.asarray(vol_crop, dtype=np.float32)
        ).unsqueeze(0)  # (1, Z, Y, X)

        surface_masks_t = torch.as_tensor(surface_masks_arr, dtype=torch.float32)
        direction_channels_t = torch.as_tensor(direction_channels, dtype=torch.float32)
        normals_valid_t = torch.as_tensor(normals_valid_vol, dtype=torch.float32).unsqueeze(0)
        padding_mask_t = torch.as_tensor(padding_mask, dtype=torch.float32).unsqueeze(0)

        return {
            "image": vol_crop_t,                        # (1, Z, Y, X)
            "surface_masks": surface_masks_t,           # (N, Z, Y, X)
            "direction_channels": direction_channels_t, # (6, Z, Y, X)
            "normals_valid": normals_valid_t,           # (1, Z, Y, X)
            "num_surfaces": num_surfaces,
            "padding_mask": padding_mask_t,             # (1, Z, Y, X)
            "patch_info": {
                "segment_uuid": str(patch.wraps[0]["segment"].uuid) if patch.wraps else "",
                "world_bbox": patch.world_bbox,
                "idx": int(idx),
            },
        }


def collate_variable_surfaces(batch):
    """Custom collate_fn that handles variable numbers of surfaces per patch.

    Stacks fixed-size tensors normally, keeps surface_masks as a list.
    """
    images = torch.stack([b["image"] for b in batch])
    direction_channels = torch.stack([b["direction_channels"] for b in batch])
    normals_valid = torch.stack([b["normals_valid"] for b in batch])
    padding_masks = torch.stack([b["padding_mask"] for b in batch])
    num_surfaces = [b["num_surfaces"] for b in batch]
    surface_masks = [b["surface_masks"] for b in batch]
    patch_infos = [b["patch_info"] for b in batch]

    return {
        "image": images,                        # (B, 1, Z, Y, X)
        "surface_masks": surface_masks,         # list of (Ni, Z, Y, X) tensors
        "direction_channels": direction_channels,  # (B, 6, Z, Y, X)
        "normals_valid": normals_valid,         # (B, 1, Z, Y, X)
        "num_surfaces": num_surfaces,           # list of ints
        "padding_mask": padding_masks,          # (B, 1, Z, Y, X)
        "patch_info": patch_infos,
    }
