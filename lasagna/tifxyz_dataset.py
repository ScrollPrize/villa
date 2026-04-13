"""TifxyzLasagnaDataset — PyTorch Dataset for lasagna 3D UNet training from tifxyz surfaces.

Produces training patches where:
- CT volume crops are read from zarr (CPU)
- Surface masks and direction channels are voxelized from tifxyz grids (CPU)
- EDT, chain ordering, cos/grad_mag/validity derivation happens on GPU in the train step

Uses helpers from ink-detection/tifxyz_dataset/ for patch finding, surface sampling,
and voxelization.
"""
from __future__ import annotations

import math
import os
import sys
import warnings
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# Add ink-detection to path for tifxyz_dataset imports
_INK_DETECTION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ink-detection",
)
if _INK_DETECTION_DIR not in sys.path:
    sys.path.insert(0, _INK_DETECTION_DIR)

from tifxyz_dataset.common import (
    _normalize_patch_size_zyx,
    _read_volume_crop_from_patch_dict,
    _sample_patch_supervision_grid,
    _voxelize_surface_from_sampled_grid,
    _estimate_surface_normals_zyx,
)
from tifxyz_dataset.patch_finding import (
    _PATCH_CACHE_DEFAULT_FILENAME,
    find_patches,
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
        self.patch_size = config["patch_size"]
        self.patch_size_zyx = _normalize_patch_size_zyx(self.patch_size)

        self.surface_bbox_pad = float(config.get("surface_bbox_pad", 2.0))
        if self.surface_bbox_pad < 0.0:
            self.surface_bbox_pad = 0.0
        self.surface_interp_method = str(
            config.get("surface_interp_method", "catmull_rom")
        ).strip().lower()

        # Required by _sample_patch_supervision_grid for ink label loading
        # (returns empty when no labels exist, which is fine for lasagna)
        self.bg_dilate_distance = int(config.get("bg_dilate_distance", 0))
        self._segment_ink_label_path_by_uuid = {}

        self.overlap_fraction = float(config.get("overlap_fraction", 0.25))
        self.min_positive_fraction = float(config.get("min_positive_fraction", 0.0))
        self.min_span_ratio = float(config.get("min_span_ratio", 0.50))
        self.patch_finding_workers = int(config.get("patch_finding_workers", 4))
        self.patch_cache_force_recompute = bool(
            config.get("patch_cache_force_recompute", False)
        )
        self.patch_cache_filename = str(
            config.get("patch_cache_filename", _PATCH_CACHE_DEFAULT_FILENAME)
        )
        self.auto_fix_padding_multiples = [64, 256]
        self.max_surfaces_per_patch = int(config.get("max_surfaces_per_patch", 8))

        # Caches
        self._segment_grid_cache = {}
        self._segment_normal_cache = {}
        self._segment_world_bounds_cache = {}
        self._segment_ink_mask_cache = {}
        self._segment_positive_samples_cache = {}
        self._segment_positive_points_cache = {}

        # Augmentation
        if apply_augmentation:
            from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
            self.augmentations = create_training_transforms(
                patch_size=tuple(int(v) for v in self.patch_size_zyx),
                no_spatial=False,
            )
        else:
            self.augmentations = None

        # Find patches (no ink labels required)
        self.patches, self.patch_generation_stats = find_patches(
            config,
            patch_size_zyx=self.patch_size_zyx,
            overlap_fraction=self.overlap_fraction,
            min_positive_fraction=self.min_positive_fraction,
            min_span_ratio=self.min_span_ratio,
            patch_finding_workers=self.patch_finding_workers,
            patch_cache_force_recompute=self.patch_cache_force_recompute,
            patch_cache_filename=self.patch_cache_filename,
            auto_fix_padding_multiples=self.auto_fix_padding_multiples,
        )

        # Collect all segments per dataset for multi-surface overlap detection
        self._segments_by_dataset_idx = {}
        for patch in self.patches:
            dataset_idx = int(patch.get("dataset_idx", -1))
            segment = patch.get("segment")
            if segment is not None:
                seg_map = self._segments_by_dataset_idx.setdefault(dataset_idx, {})
                seg_map.setdefault(str(segment.uuid), segment)
        self._segments_by_dataset_idx = {
            int(k): tuple(v.values())
            for k, v in self._segments_by_dataset_idx.items()
        }

        print(f"{TAG} loaded {len(self.patches)} patches")

    def __len__(self):
        return len(self.patches)

    def _get_segment_stored_grid(self, segment):
        """Cache and return the stored-resolution grid for a segment."""
        segment_uuid = str(segment.uuid)
        cached = self._segment_grid_cache.get(segment_uuid)
        if cached is not None:
            return cached

        segment.use_stored_resolution()
        x_stored, y_stored, z_stored, valid_stored = segment[:, :]
        x_stored = np.asarray(x_stored, dtype=np.float32)
        y_stored = np.asarray(y_stored, dtype=np.float32)
        z_stored = np.asarray(z_stored, dtype=np.float32)
        valid_mask = np.asarray(valid_stored, dtype=bool)
        valid_mask &= np.isfinite(x_stored)
        valid_mask &= np.isfinite(y_stored)
        valid_mask &= np.isfinite(z_stored)

        cached = {
            "x": x_stored, "y": y_stored, "z": z_stored,
            "valid": valid_mask,
            "shape": (int(x_stored.shape[0]), int(x_stored.shape[1])),
        }
        self._segment_grid_cache[segment_uuid] = cached
        return cached

    def _get_segment_world_bounds(self, segment):
        """Get (z_min, z_max, y_min, y_max, x_min, x_max) for a segment."""
        segment_uuid = str(segment.uuid)
        if segment_uuid in self._segment_world_bounds_cache:
            return self._segment_world_bounds_cache[segment_uuid]

        grid = self._get_segment_stored_grid(segment)
        valid = np.asarray(grid["valid"], dtype=bool)
        if not np.any(valid):
            self._segment_world_bounds_cache[segment_uuid] = None
            return None

        z_vals = grid["z"][valid]
        y_vals = grid["y"][valid]
        x_vals = grid["x"][valid]
        bounds = (
            float(np.min(z_vals)), float(np.max(z_vals)),
            float(np.min(y_vals)), float(np.max(y_vals)),
            float(np.min(x_vals)), float(np.max(x_vals)),
        )
        self._segment_world_bounds_cache[segment_uuid] = bounds
        return bounds

    @staticmethod
    def _bounds_intersect(bounds, min_corner, max_corner):
        if bounds is None:
            return False
        min_c = np.asarray(min_corner, dtype=np.float32).reshape(3)
        max_c = np.asarray(max_corner, dtype=np.float32).reshape(3)
        z_min, z_max, y_min, y_max, x_min, x_max = [float(v) for v in bounds]
        return not (
            z_max < float(min_c[0]) or z_min >= float(max_c[0]) or
            y_max < float(min_c[1]) or y_min >= float(max_c[1]) or
            x_max < float(min_c[2]) or x_min >= float(max_c[2])
        )

    def _find_overlapping_segments(self, patch, min_corner, max_corner):
        """Find all segments from this dataset that overlap the patch bbox."""
        dataset_idx = int(patch.get("dataset_idx", -1))
        segments = self._segments_by_dataset_idx.get(dataset_idx, ())
        result = []
        for segment in segments:
            bounds = self._get_segment_world_bounds(segment)
            if self._bounds_intersect(bounds, min_corner, max_corner):
                result.append(segment)
            if len(result) >= self.max_surfaces_per_patch:
                break
        return result

    def _sample_and_voxelize_segment(self, segment, min_corner, max_corner, crop_size):
        """Sample a tifxyz segment within a patch and produce mask + direction channels.

        Returns:
            surface_mask: (Z, Y, X) float32 — binary voxelization
            dir_points_local: (M, 3) float32 — local ZYX positions for splatting
            dir_values: (M, 6) float32 — direction channel values
            normals_points_local: (M, 3) float32 — positions where normals are valid
        """
        crop_size_tuple = tuple(int(v) for v in crop_size)

        # Sample grid
        sampled = _sample_patch_supervision_grid(
            self, segment, min_corner=min_corner, max_corner=max_corner,
            extra_bbox_pad=0.0,
        )

        # Binary mask voxelization
        surface_mask = _voxelize_surface_from_sampled_grid(
            self, segment, min_corner=min_corner, max_corner=max_corner,
            crop_size=crop_size_tuple, sampled_grid=sampled,
        )

        # Direction channels from normals
        local_grid = sampled["local_grid"]
        in_patch = sampled["in_patch"]
        normals_zyx = sampled["normals_zyx"]
        normals_valid = sampled["normals_valid"]

        # Points with valid normals inside the patch
        valid_for_dir = in_patch & normals_valid
        if not np.any(valid_for_dir):
            empty_pts = np.zeros((0, 3), dtype=np.float32)
            empty_vals = np.zeros((0, 6), dtype=np.float32)
            return surface_mask, empty_pts, empty_vals

        pts_local = local_grid[valid_for_dir].astype(np.float32)
        normals_at_pts = normals_zyx[valid_for_dir].astype(np.float32)

        # Compute direction encoding
        dir_vals = compute_direction_values(normals_at_pts)

        return surface_mask, pts_local, dir_vals

    def __getitem__(self, idx):
        patch = self.patches[idx]

        z0, z1, y0, y1, x0, x1 = patch["world_bbox"]
        min_corner = np.array([z0, y0, x0], dtype=np.int32)
        max_corner = np.array([z1 + 1, y1 + 1, x1 + 1], dtype=np.int32)
        crop_size = tuple(int(v) for v in self.patch_size_zyx)

        # Read CT crop (z-score normalized)
        vol_crop = _read_volume_crop_from_patch_dict(
            patch, crop_size=crop_size,
            min_corner=min_corner, max_corner=max_corner,
        )

        # Find all overlapping surfaces
        segments = self._find_overlapping_segments(patch, min_corner, max_corner)

        # Per-surface: voxelize mask and compute direction channels
        surface_masks = []
        all_dir_points = []
        all_dir_values = []

        for segment in segments:
            mask, dir_pts, dir_vals = self._sample_and_voxelize_segment(
                segment, min_corner, max_corner, crop_size,
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
                "dataset_idx": int(patch.get("dataset_idx", -1)),
                "segment_uuid": str(patch.get("segment_uuid", "")),
                "world_bbox": patch["world_bbox"],
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
