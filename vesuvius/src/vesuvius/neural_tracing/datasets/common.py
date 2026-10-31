import torch
import numpy as np
import json
import os
from numba import njit
from dataclasses import dataclass
from pathlib import Path
from vc_reader import tifxyz
from vc_reader.reader import ZarrGroup, ZarrReader, load_basic_auth, open_node
from typing import Any, Dict, List, Tuple
import re
from functools import lru_cache
from scipy import ndimage
from vesuvius.image_proc.intensity.normalization import normalize_zscore
import tifffile
import warnings


_HTTP_PREFIXES = ('http://', 'https://')
_REMOTE_PREFIXES = (*_HTTP_PREFIXES, 's3://')


def _resolve_config_relative_path(path_value, config):
    """Resolve a possibly-relative path against the config file's directory."""
    if path_value is None:
        return None

    path_obj = Path(os.path.expanduser(str(path_value)))
    if path_obj.is_absolute():
        return path_obj

    config_dir = config.get('_config_dir') or config.get('config_dir')
    if config_dir:
        return Path(config_dir).expanduser() / path_obj
    return Path.cwd() / path_obj


def _volume_open_options(path, auth_json_path, config, threads):
    """vc_reader's open arguments from a dataset config: ``volume_cache_dir``
    (required for a remote volume), ``volume_cache_offline``, and HTTP basic
    auth from ``auth_json_path``."""
    path = str(path)
    config = {} if config is None else config
    options = {"threads": threads}
    if path.startswith(_REMOTE_PREFIXES):
        cache_dir = config.get('volume_cache_dir')
        if not cache_dir:
            raise ValueError(
                f"Remote volume path {path!r} requires 'volume_cache_dir' in config. "
                f"Set it to a local directory for caching, e.g. "
                f'{{"volume_cache_dir": "/data/zarr_cache"}}'
            )
        options["cache_dir"] = str(_resolve_config_relative_path(cache_dir, config) or cache_dir)
        options["offline"] = bool(config.get('volume_cache_offline', False))
    if auth_json_path and path.startswith(_HTTP_PREFIXES):
        options["auth"] = load_basic_auth(_resolve_config_relative_path(auth_json_path, config))
    return options


def open_zarr(path, scale=None, auth_json_path=None, config=None, threads=None):
    """One level of a volume: a local path, HTTP URL, or S3 URI."""
    return ZarrReader.open(
        str(path), level=0 if scale is None else int(scale),
        **_volume_open_options(path, auth_json_path, config, threads))


def open_zarr_group(path, auth_json_path=None, config=None, threads=None):
    """A volume's root group (all pyramid levels), or the array at ``path``."""
    return open_node(str(path), **_volume_open_options(path, auth_json_path, config, threads))


def _parse_z_range(z_range):
    if z_range is None:
        return None
    if not isinstance(z_range, (list, tuple)) or len(z_range) != 2:
        raise ValueError(f"dataset z_range must be [z_min, z_max], got {z_range!r}")
    z_min = float(z_range[0])
    z_max = float(z_range[1])
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        raise ValueError(f"dataset z_range must contain finite numbers, got {z_range!r}")
    if z_min > z_max:
        z_min, z_max = z_max, z_min
    return z_min, z_max


def _segment_z_bounds(seg):
    valid = seg.valid
    if not np.any(valid):
        return None
    if seg.bbox is not None:
        # Segment bbox is in XYZ order: (x_min, y_min, z_min, x_max, y_max, z_max)
        z_min = float(seg.bbox[2])
        z_max = float(seg.bbox[5])
    else:
        z_vals = seg.xyz[..., 2][valid]
        z_min = float(np.min(z_vals))
        z_max = float(np.max(z_vals))
    return z_min, z_max


def _segment_overlaps_z_range(seg, z_range):
    if z_range is None:
        return True
    z_bounds = _segment_z_bounds(seg)
    if z_bounds is None:
        return False
    seg_z_min, seg_z_max = z_bounds
    z_min, z_max = z_range
    return not (seg_z_min > z_max or seg_z_max < z_min)


def _extract_wrap_ids(name: str):
    if not name:
        return tuple()
    wrap_ids = sorted({int(m.group(1)) for m in re.finditer(r"w(\d+)", str(name))})
    return tuple(wrap_ids)


def _has_consecutive_wrap_ids(left_ids, right_ids):
    if not left_ids or not right_ids:
        return False
    right_set = set(right_ids)
    for wrap_id in left_ids:
        if (wrap_id - 1) in right_set or (wrap_id + 1) in right_set:
            return True
    return False


def _triplet_wraps_compatible(target_wrap_meta: dict, other_wrap_meta: dict):
    if target_wrap_meta["segment_idx"] == other_wrap_meta["segment_idx"]:
        return True
    return _has_consecutive_wrap_ids(
        target_wrap_meta.get("wrap_ids", tuple()),
        other_wrap_meta.get("wrap_ids", tuple()),
    )


def _wrap_bbox_has_overlap(mask: np.ndarray, bbox_2d):
    if mask is None:
        return False
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2 or mask_arr.size == 0:
        return False
    r_min, r_max, c_min, c_max = [int(v) for v in bbox_2d]
    h, w = mask_arr.shape
    r0 = max(0, r_min)
    r1 = min(h - 1, r_max)
    c0 = max(0, c_min)
    c1 = min(w - 1, c_max)
    if r1 < r0 or c1 < c0:
        return False
    return bool(np.any(mask_arr[r0:r1 + 1, c0:c1 + 1]))


def _wrap_chunk_has_overlap(mask: np.ndarray, wrap: dict, patch_world_bbox):
    """Return True when overlap-mask pixels for this wrap fall inside the chunk bbox."""
    if mask is None:
        return False
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2 or mask_arr.size == 0:
        return False

    seg = wrap.get("segment")
    if seg is None:
        return False

    r_min, r_max, c_min, c_max = [int(v) for v in wrap["bbox_2d"]]
    seg_h, seg_w = seg.shape
    mask_h, mask_w = mask_arr.shape
    r0 = max(0, r_min)
    r1 = min(seg_h - 1, mask_h - 1, r_max)
    c0 = max(0, c_min)
    c1 = min(seg_w - 1, mask_w - 1, c_max)
    if r1 < r0 or c1 < c0:
        return False

    overlap_local = mask_arr[r0:r1 + 1, c0:c1 + 1] > 0
    if not bool(np.any(overlap_local)):
        return False

    xyz_s = seg.xyz[r0:r1 + 1, c0:c1 + 1]
    x_s, y_s, z_s = xyz_s[..., 0], xyz_s[..., 1], xyz_s[..., 2]
    valid_s = seg.valid[r0:r1 + 1, c0:c1 + 1]

    z_min, z_max, y_min, y_max, x_min, x_max = patch_world_bbox
    in_chunk = (
        (z_s >= z_min) & (z_s < z_max) &
        (y_s >= y_min) & (y_s < y_max) &
        (x_s >= x_min) & (x_s < x_max)
    )
    in_chunk &= valid_s
    return bool(np.any(overlap_local & in_chunk))


def _filter_triplet_overlap_chunks(patches, config):
    """Drop triplet chunks based on configured overlap filtering mode.

    Modes:
      - bbox: drop if overlap_mask has hits inside the wrap bbox.
      - any_masked_pixel: drop if overlap_mask hits overlap with this wrap inside this chunk's world bbox.
    """
    mask_filename = str(config["triplet_overlap_mask_filename"])
    warn_missing_masks = bool(config["triplet_warn_missing_overlap_masks"])
    overlap_filter_mode = str(config.get("triplet_overlap_filter_mode", "bbox")).lower()
    warned_missing = set()
    kept = []
    kept_indices = []

    for patch_idx, patch in enumerate(patches):
        drop_chunk = False
        for wrap in patch.wraps:
            seg = wrap.get("segment")
            seg_path = getattr(seg, "path", None)
            if seg_path is None:
                continue
            seg_path = Path(seg_path)
            mask_path = seg_path / mask_filename
            mask_key = str(mask_path)

            if not mask_path.exists():
                if warn_missing_masks and mask_key not in warned_missing:
                    warned_missing.add(mask_key)
                    warnings.warn(
                        f"Triplet overlap mask not found at {mask_path}; treating as no-overlap for this wrap.",
                        RuntimeWarning,
                    )
                continue

            overlap_mask = np.asarray(tifffile.imread(str(mask_path))) > 0

            if overlap_filter_mode == "any_masked_pixel":
                has_overlap = _wrap_chunk_has_overlap(overlap_mask, wrap, patch.world_bbox)
            elif overlap_filter_mode == "bbox":
                has_overlap = _wrap_bbox_has_overlap(overlap_mask, wrap["bbox_2d"])
            else:
                raise ValueError(
                    "triplet_overlap_filter_mode must be one of {'bbox', 'any_masked_pixel'}, "
                    f"got {overlap_filter_mode!r}"
                )
            if has_overlap:
                drop_chunk = True
                break

        if not drop_chunk:
            kept.append(patch)
            kept_indices.append(int(patch_idx))

    return kept, tuple(kept_indices)


def _compute_wrap_order_stats(wrap):
    """Compute per-wrap medians used for triplet neighbor ordering."""
    seg = wrap["segment"]
    r_min, r_max, c_min, c_max = wrap["bbox_2d"]

    seg_h, seg_w = seg.shape
    r_min = max(0, r_min)
    r_max = min(seg_h - 1, r_max)
    c_min = max(0, c_min)
    c_max = min(seg_w - 1, c_max)
    if r_max < r_min or c_max < c_min:
        return None

    valid_s = seg.valid[r_min:r_max + 1, c_min:c_max + 1]
    if not valid_s.any():
        return None
    xyz_s = seg.xyz[r_min:r_max + 1, c_min:c_max + 1]
    x_vals = xyz_s[..., 0][valid_s]
    y_vals = xyz_s[..., 1][valid_s]

    if x_vals.size == 0 or y_vals.size == 0:
        return None

    return {
        "x_median": float(np.median(x_vals)),
        "y_median": float(np.median(y_vals)),
    }


def _read_volume_crop(volume_arr, crop_size, min_corner, max_corner,
                      image_normalization: str = "zscore"):
    """Read a CT crop from a zarr Array and normalize it.

    Args:
        volume_arr: ZarrReader to read from (a specific pyramid level).
        crop_size: (Z, Y, X) shape of the output crop.
        min_corner: (3,) int64 array — ZYX origin in volume coordinates.
        max_corner: (3,) int64 array — ZYX end (exclusive) in volume coordinates.
        image_normalization: ``"zscore"`` or ``"unit"`` (uint8/255).
    """
    vol_crop = np.zeros(crop_size, dtype=volume_arr.dtype)
    vol_shape = volume_arr.shape
    src_starts = np.maximum(min_corner, 0)
    src_ends = np.minimum(max_corner, np.array(vol_shape, dtype=np.int64))
    dst_starts = src_starts - min_corner
    dst_ends = dst_starts + (src_ends - src_starts)

    if np.all(src_ends > src_starts):
        vol_crop[
            dst_starts[0]:dst_ends[0],
            dst_starts[1]:dst_ends[1],
            dst_starts[2]:dst_ends[2],
        ] = volume_arr[
            src_starts[0]:src_ends[0],
            src_starts[1]:src_ends[1],
            src_starts[2]:src_ends[2],
        ]

    if image_normalization == "unit":
        return vol_crop.astype(np.float32) / 255.0
    if image_normalization == "zscore":
        return normalize_zscore(vol_crop)
    raise ValueError(
        f"Unknown image_normalization '{image_normalization}' "
        f"(expected 'zscore' or 'unit')."
    )


def _read_volume_crop_from_patch(patch, crop_size, min_corner, max_corner,
                                 image_normalization: str = "zscore"):
    """Read a CT crop from a patch's volume and normalize it.

    Thin wrapper around :func:`_read_volume_crop` that resolves the
    level from ``patch.volume`` / ``patch.scale``.
    """
    volume = patch.volume
    if isinstance(volume, ZarrGroup):
        volume = volume[str(patch.scale)]
    return _read_volume_crop(volume, crop_size, min_corner, max_corner,
                             image_normalization)


def _validate_result_tensors(result: dict, idx: int, enabled: bool):
    if not bool(enabled):
        return True
    for key, tensor in result.items():
        if not torch.is_tensor(tensor):
            print(f"WARNING: Non-tensor value for '{key}' at index {idx}, resampling...")
            return False
        if tensor.numel() == 0:
            print(f"WARNING: Empty tensor for '{key}' at index {idx}, resampling...")
            return False
        if not bool(torch.isfinite(tensor).all()):
            print(f"WARNING: Non-finite values in '{key}' at index {idx}, resampling...")
            return False
    return True


def _should_attempt_cond_local_perturb(config: dict, apply_perturbation: bool = True) -> bool:
    cfg = dict(config["cond_local_perturb"] or {})
    if not bool(cfg["enabled"]):
        return False
    if float(cfg["probability"]) <= 0.0:
        return False
    if not bool(apply_perturbation):
        return False
    return True


def _require_augmented_keypoints(augmented: dict, expected_shape, mode: str):
    augmented_keypoints = augmented.get("keypoints")
    expected_tuple = tuple(expected_shape)
    requirement = (
        "cond_local_perturb post-augmentation requires the augmentation pipeline to preserve "
        "keypoints when cond_surface_local is provided."
    )
    if augmented_keypoints is None:
        raise RuntimeError(
            f"{mode} augmentation did not return keypoints (expected shape {expected_tuple}); "
            f"{requirement}"
        )
    actual_tuple = tuple(augmented_keypoints.shape)
    if actual_tuple != expected_tuple:
        raise RuntimeError(
            f"{mode} augmentation returned keypoints with shape {actual_tuple}; expected "
            f"{expected_tuple}. {requirement}"
        )
    return augmented_keypoints


def _prepare_cond_surface_keypoints(cond_surface_local):
    if cond_surface_local is None:
        return None, None, None, True
    if cond_surface_local.ndim != 3 or int(cond_surface_local.shape[-1]) != 3:
        return None, None, None, False
    cond_surface_shape = tuple(int(s) for s in cond_surface_local.shape[:2])
    cond_surface_keypoints = cond_surface_local.reshape(-1, 3).contiguous()
    return cond_surface_local, cond_surface_shape, cond_surface_keypoints, True


def _fraction_within_distance(dist_map: np.ndarray, source_mask: np.ndarray, max_distance_voxels: float) -> float:
    dist = np.asarray(dist_map, dtype=np.float32)
    mask = np.asarray(source_mask, dtype=bool)
    if dist.shape != mask.shape:
        raise ValueError(
            f"dist_map shape must match source_mask shape, got {tuple(dist.shape)} vs {tuple(mask.shape)}"
        )
    if not bool(mask.any()):
        return 0.0
    vals = dist[mask]
    finite = np.isfinite(vals)
    vals = vals[finite]
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals <= float(max_distance_voxels)))


def _fraction_disp_within_distance(
    disp_np: np.ndarray,
    source_mask: np.ndarray,
    max_distance_voxels: float,
) -> float:
    """Fraction of masked voxels whose displacement magnitude is <= threshold."""
    disp = np.asarray(disp_np, dtype=np.float32)
    mask = np.asarray(source_mask, dtype=bool)
    if disp.ndim != 4 or disp.shape[0] != 3:
        raise ValueError(f"disp_np must have shape (3, D, H, W), got {tuple(disp.shape)}")
    if mask.shape != tuple(disp.shape[1:]):
        raise ValueError(
            "source_mask shape must match displacement spatial shape, "
            f"got {tuple(mask.shape)} vs {tuple(disp.shape[1:])}"
        )
    if not bool(mask.any()):
        return 0.0

    masked_disp = disp[:, mask]
    if masked_disp.size == 0:
        return 0.0
    finite = np.isfinite(masked_disp).all(axis=0)
    masked_disp = masked_disp[:, finite]
    if masked_disp.size == 0:
        return 0.0

    mags = np.sqrt(np.sum(masked_disp * masked_disp, axis=0, dtype=np.float32), dtype=np.float32)
    return float(np.mean(mags <= float(max_distance_voxels)))


def _triplet_close_contact_fractions(
    cond_mask: np.ndarray,
    behind_mask: np.ndarray,
    front_mask: np.ndarray,
    behind_disp_np: np.ndarray,
    front_disp_np: np.ndarray,
    max_distance_voxels: float,
):
    behind_disp = np.asarray(behind_disp_np, dtype=np.float32)
    front_disp = np.asarray(front_disp_np, dtype=np.float32)
    if behind_disp.ndim != 4 or behind_disp.shape[0] != 3:
        raise ValueError(f"behind_disp_np must have shape (3, D, H, W), got {tuple(behind_disp.shape)}")
    if front_disp.ndim != 4 or front_disp.shape[0] != 3:
        raise ValueError(f"front_disp_np must have shape (3, D, H, W), got {tuple(front_disp.shape)}")

    cond_behind_frac = _fraction_disp_within_distance(behind_disp, cond_mask, max_distance_voxels)
    cond_front_frac = _fraction_disp_within_distance(front_disp, cond_mask, max_distance_voxels)
    behind_to_front_frac = _fraction_disp_within_distance(front_disp, behind_mask, max_distance_voxels)
    front_to_behind_frac = _fraction_disp_within_distance(behind_disp, front_mask, max_distance_voxels)
    behind_front_frac = max(behind_to_front_frac, front_to_behind_frac)
    return cond_behind_frac, cond_front_frac, behind_front_frac


def create_band_mask(
    cond_bin_full: np.ndarray,
    d_front_work: np.ndarray,
    d_behind_work: np.ndarray,
    front_disp_work: np.ndarray,
    behind_disp_work: np.ndarray,
    band_pct: float,
    band_padding: float,
    cc_structure_26: np.ndarray,
    closing_structure_3: np.ndarray,
):
    """Build a dense slab mask between two wrap displacements for triplet band mode."""
    cond_bin = np.asarray(cond_bin_full, dtype=np.uint8)
    if cond_bin.sum() == 0:
        return None

    cond_mask = cond_bin > 0
    cond_to_front = d_front_work[cond_mask]
    cond_to_behind = d_behind_work[cond_mask]
    if cond_to_front.size == 0 or cond_to_behind.size == 0:
        return None

    # Inside points tend to have front/back displacement vectors pointing in
    # opposite directions (non-positive dot product).
    d_sum_work = d_front_work + d_behind_work
    cond_sum = (cond_to_front + cond_to_behind).astype(np.float32, copy=False)
    if cond_sum.size == 0:
        return None

    sum_threshold = float(np.percentile(cond_sum, band_pct)) + (2.0 * band_padding)
    vector_dot = np.sum(front_disp_work * behind_disp_work, axis=0, dtype=np.float32)
    dense_band = (vector_dot <= 0.0) & (d_sum_work <= sum_threshold)
    if not dense_band.any():
        return None

    # Remove isolated islands: keep only components connected to conditioning.
    labels, num_labels = ndimage.label(dense_band, structure=cc_structure_26)
    if num_labels <= 0:
        return None
    touching = np.unique(labels[cond_mask])
    touching = touching[touching > 0]
    if touching.size == 0:
        return None
    keep = np.zeros(num_labels + 1, dtype=bool)
    keep[touching] = True

    dense_band = keep[labels]
    # Fill tiny holes inside the slab.
    dense_band = ndimage.binary_closing(
        dense_band,
        structure=closing_structure_3,
        iterations=1,
    )
    if not dense_band.any():
        return None

    # Closing can create detached islands; keep only cond-connected components.
    labels, num_labels = ndimage.label(dense_band, structure=cc_structure_26)
    if num_labels <= 0:
        return None
    touching = np.unique(labels[cond_mask])
    touching = touching[touching > 0]
    if touching.size == 0:
        return None
    keep = np.zeros(num_labels + 1, dtype=bool)
    keep[touching] = True
    return keep[labels].astype(np.float32, copy=False)


def _compute_triplet_edt_bbox(
    cond_mask: np.ndarray,
    behind_mask: np.ndarray,
    front_mask: np.ndarray,
    padding_voxels: float,
):
    """Return padded (z,y,x) slices for triplet EDT compute region, or None if empty."""
    cond = np.asarray(cond_mask, dtype=bool)
    behind = np.asarray(behind_mask, dtype=bool)
    front = np.asarray(front_mask, dtype=bool)
    if cond.shape != behind.shape or cond.shape != front.shape:
        raise ValueError(
            "triplet EDT bbox masks must share shape, "
            f"got cond={tuple(cond.shape)} behind={tuple(behind.shape)} front={tuple(front.shape)}"
        )
    if cond.ndim != 3:
        raise ValueError(f"triplet EDT bbox masks must be 3D, got shape {tuple(cond.shape)}")

    union = cond | behind | front
    if not union.any():
        return None

    pad = int(np.ceil(max(0.0, float(padding_voxels))))
    zz, yy, xx = np.nonzero(union)
    d, h, w = union.shape

    z0 = max(0, int(zz.min()) - pad)
    y0 = max(0, int(yy.min()) - pad)
    x0 = max(0, int(xx.min()) - pad)
    z1 = min(d - 1, int(zz.max()) + pad)
    y1 = min(h - 1, int(yy.max()) + pad)
    x1 = min(w - 1, int(xx.max()) + pad)
    if z1 < z0 or y1 < y0 or x1 < x0:
        return None
    return (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))


@lru_cache(maxsize=5)
def _small_spherical_footprint(radius: int):
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    zz, yy, xx = np.meshgrid(offsets, offsets, offsets, indexing='ij')
    return (zz * zz + yy * yy + xx * xx) <= (radius * radius)


def edt_dilate_binary_mask(mask: np.ndarray, radius_voxels: float) -> np.ndarray:
    """Dilate binary mask by EDT thresholding; returns a boolean mask."""
    radius = float(radius_voxels)
    mask_bool = np.asarray(mask) > 0
    if radius <= 0.0 or not mask_bool.any():
        return mask_bool
    # Exact structuring-element dilation is faster than EDT for common small
    # integer radii (for example, triplet_gt_vector_dilation_radius=1.0).
    rounded_radius = int(round(radius))
    if abs(radius - rounded_radius) <= 1e-6 and rounded_radius <= 4:
        footprint = _small_spherical_footprint(rounded_radius)
        return ndimage.binary_dilation(mask_bool, structure=footprint, iterations=1)
    dist = ndimage.distance_transform_edt(~mask_bool)
    return dist <= radius


def _signed_distance_field(mask: np.ndarray) -> np.ndarray:
    """Signed distance using the legacy sign convention (positive inside)."""
    mask_bool = np.asarray(mask) > 0
    dist_outside = ndimage.distance_transform_edt(~mask_bool)
    dist_inside = ndimage.distance_transform_edt(mask_bool)
    return (dist_inside - dist_outside).astype(np.float32, copy=False)


def _upsample_world_surface(x_s, y_s, z_s, scale_y: float, scale_x: float):
    """Upsample (x, y, z) sampled grids, all present, by bicubic Hermite patches."""
    h_s, w_s = x_s.shape
    h_up = int(round(h_s / scale_y))
    w_up = int(round(w_s / scale_x))
    dense_rows = np.linspace(0, h_s - 1, h_up, dtype=np.float32)
    dense_cols = np.linspace(0, w_s - 1, w_up, dtype=np.float32)
    query_row, query_col = np.meshgrid(dense_rows, dense_cols, indexing="ij")

    surface = tifxyz.Surface(np.stack([x_s, y_s, z_s], axis=-1).astype(np.float32))
    xyz, valid = surface.points_at(query_row, query_col, interpolation="smooth")

    if not np.all(valid):
        invalid_count = int((~valid).sum())
        raise ValueError(
            "Invalid points from surface interpolation: "
            f"{invalid_count}/{valid.size}"
        )

    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def _trim_to_world_bbox(x_full, y_full, z_full, world_bbox):
    """Keep the minimal row/col slab that intersects the world bbox."""
    z_min, z_max, y_min, y_max, x_min, x_max = world_bbox
    in_bounds = (
        (z_full >= z_min) & (z_full < z_max) &
        (y_full >= y_min) & (y_full < y_max) &
        (x_full >= x_min) & (x_full < x_max)
    )
    if not in_bounds.any():
        return None

    valid_rows = np.any(in_bounds, axis=1)
    valid_cols = np.any(in_bounds, axis=0)
    row_idx = np.flatnonzero(valid_rows)
    col_idx = np.flatnonzero(valid_cols)
    if row_idx.size == 0 or col_idx.size == 0:
        return None

    r0, r1 = int(row_idx[0]), int(row_idx[-1])
    c0, c1 = int(col_idx[0]), int(col_idx[-1])
    return (
        x_full[r0:r1 + 1, c0:c1 + 1],
        y_full[r0:r1 + 1, c0:c1 + 1],
        z_full[r0:r1 + 1, c0:c1 + 1],
    )


@njit
def _draw_line_3d(volume: np.ndarray, z0: int, y0: int, x0: int, z1: int, y1: int, x1: int) -> None:
    """Draw a 3D line using Bresenham's algorithm. Modifies volume in-place."""
    dz = abs(z1 - z0)
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)

    sz = 1 if z0 < z1 else -1
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1

    # Determine the dominant axis
    if dx >= dy and dx >= dz:
        # X is dominant
        err_y = 2 * dy - dx
        err_z = 2 * dz - dx
        while x0 != x1:
            if 0 <= z0 < volume.shape[0] and 0 <= y0 < volume.shape[1] and 0 <= x0 < volume.shape[2]:
                volume[z0, y0, x0] = 1.0
            if err_y > 0:
                y0 += sy
                err_y -= 2 * dx
            if err_z > 0:
                z0 += sz
                err_z -= 2 * dx
            err_y += 2 * dy
            err_z += 2 * dz
            x0 += sx
    elif dy >= dx and dy >= dz:
        # Y is dominant
        err_x = 2 * dx - dy
        err_z = 2 * dz - dy
        while y0 != y1:
            if 0 <= z0 < volume.shape[0] and 0 <= y0 < volume.shape[1] and 0 <= x0 < volume.shape[2]:
                volume[z0, y0, x0] = 1.0
            if err_x > 0:
                x0 += sx
                err_x -= 2 * dy
            if err_z > 0:
                z0 += sz
                err_z -= 2 * dy
            err_x += 2 * dx
            err_z += 2 * dz
            y0 += sy
    else:
        # Z is dominant
        err_x = 2 * dx - dz
        err_y = 2 * dy - dz
        while z0 != z1:
            if 0 <= z0 < volume.shape[0] and 0 <= y0 < volume.shape[1] and 0 <= x0 < volume.shape[2]:
                volume[z0, y0, x0] = 1.0
            if err_x > 0:
                x0 += sx
                err_x -= 2 * dz
            if err_y > 0:
                y0 += sy
                err_y -= 2 * dz
            err_x += 2 * dx
            err_y += 2 * dy
            z0 += sz

    # Set the final point
    if 0 <= z1 < volume.shape[0] and 0 <= y1 < volume.shape[1] and 0 <= x1 < volume.shape[2]:
        volume[z1, y1, x1] = 1.0


@njit
def voxelize_surface_grid_into(
    volume: np.ndarray,
    zyx_grid: np.ndarray,
):
    """
    Voxelize a 2D grid of 3D points into an existing volume.

    Args:
        volume: (D, H, W) output volume to mutate in-place
        zyx_grid: (H, W, 3) array of ZYX coordinates in local crop space
    """
    n_rows, n_cols = zyx_grid.shape[0], zyx_grid.shape[1]

    # Draw horizontal lines (between adjacent columns)
    for r in range(n_rows):
        for c in range(n_cols - 1):
            z0 = int(round(zyx_grid[r, c, 0]))
            y0 = int(round(zyx_grid[r, c, 1]))
            x0 = int(round(zyx_grid[r, c, 2]))
            z1 = int(round(zyx_grid[r, c + 1, 0]))
            y1 = int(round(zyx_grid[r, c + 1, 1]))
            x1 = int(round(zyx_grid[r, c + 1, 2]))
            _draw_line_3d(volume, z0, y0, x0, z1, y1, x1)

    # Draw vertical lines (between adjacent rows)
    for r in range(n_rows - 1):
        for c in range(n_cols):
            z0 = int(round(zyx_grid[r, c, 0]))
            y0 = int(round(zyx_grid[r, c, 1]))
            x0 = int(round(zyx_grid[r, c, 2]))
            z1 = int(round(zyx_grid[r + 1, c, 0]))
            y1 = int(round(zyx_grid[r + 1, c, 1]))
            x1 = int(round(zyx_grid[r + 1, c, 2]))
            _draw_line_3d(volume, z0, y0, x0, z1, y1, x1)


@njit
def voxelize_surface_grid(
    zyx_grid: np.ndarray,
    crop_size: tuple,
) -> np.ndarray:
    """
    Voxelize a 2D grid of 3D points by drawing lines between adjacent points.

    Args:
        zyx_grid: (H, W, 3) array of ZYX coordinates in local crop space
        crop_size: (D, H, W) shape of output volume

    Returns:
        (D, H, W) uint8 binary volume with lines connecting adjacent grid points
    """
    volume = np.zeros(crop_size, dtype=np.uint8)
    voxelize_surface_grid_into(volume, zyx_grid)
    return volume

@njit
def voxelize_surface_grid_masked(
    zyx_grid: np.ndarray,
    crop_size: tuple,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Voxelize a 2D grid of 3D points while honoring a per-cell validity mask.

    Behavior:
    - Draw all valid points.
    - Draw row/col edges only when both endpoints are valid.

    Args:
        zyx_grid: (H, W, 3) array of ZYX coordinates in local crop space
        crop_size: (D, H, W) shape of output volume
        valid_mask: (H, W) bool/0-1 mask; True means the grid cell is valid

    Returns:
        (D, H, W) binary volume with masked surface rasterization
    """
    volume = np.zeros(crop_size, dtype=np.float32)
    n_rows, n_cols = zyx_grid.shape[0], zyx_grid.shape[1]

    # Draw valid points so isolated valid cells are still represented.
    for r in range(n_rows):
        for c in range(n_cols):
            if not valid_mask[r, c]:
                continue
            z = int(round(zyx_grid[r, c, 0]))
            y = int(round(zyx_grid[r, c, 1]))
            x = int(round(zyx_grid[r, c, 2]))
            if 0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]:
                volume[z, y, x] = 1.0

    # Draw horizontal lines between adjacent valid columns.
    for r in range(n_rows):
        for c in range(n_cols - 1):
            if not (valid_mask[r, c] and valid_mask[r, c + 1]):
                continue
            z0 = int(round(zyx_grid[r, c, 0]))
            y0 = int(round(zyx_grid[r, c, 1]))
            x0 = int(round(zyx_grid[r, c, 2]))
            z1 = int(round(zyx_grid[r, c + 1, 0]))
            y1 = int(round(zyx_grid[r, c + 1, 1]))
            x1 = int(round(zyx_grid[r, c + 1, 2]))
            _draw_line_3d(volume, z0, y0, x0, z1, y1, x1)

    # Draw vertical lines between adjacent valid rows.
    for r in range(n_rows - 1):
        for c in range(n_cols):
            if not (valid_mask[r, c] and valid_mask[r + 1, c]):
                continue
            z0 = int(round(zyx_grid[r, c, 0]))
            y0 = int(round(zyx_grid[r, c, 1]))
            x0 = int(round(zyx_grid[r, c, 2]))
            z1 = int(round(zyx_grid[r + 1, c, 0]))
            y1 = int(round(zyx_grid[r + 1, c, 1]))
            x1 = int(round(zyx_grid[r + 1, c, 2]))
            _draw_line_3d(volume, z0, y0, x0, z1, y1, x1)

    return volume

@dataclass
class Patch:
    """A single patch from the hierarchical tiling method."""
    seg: tifxyz.Surface                   # Reference to the segment
    volume: ZarrReader                    # zarr volume
    scale: float                          # volume_scale from config
    grid_bbox: Tuple[int, int, int, int]  # (row_min, row_max, col_min, col_max) in the tifxyz grid
    world_bbox: Tuple[float, ...]         # (z_min, z_max, y_min, y_max, x_min, x_max) in world coordinates (volume coordinates)


@dataclass
class ChunkPatch:
    """A world-space chunk containing one or more surface wraps.

    This is the output of the world-chunk tiling method. Each chunk can contain
    multiple wraps from potentially multiple segments.
    """
    chunk_id: Tuple[int, int, int]         # (cz, cy, cx) index in chunk grid
    volume: Any                            # ZarrReader or ZarrGroup
    scale: int                             # volume_scale from config
    world_bbox: Tuple[float, ...]          # (z_min, z_max, y_min, y_max, x_min, x_max)
    wraps: List[Dict]                      # [{"segment": tifxyz.Surface, "bbox_2d": tuple, "wrap_id": int, "segment_idx": int}, ...]
    segments: List[tifxyz.Surface]         # All segments (for lookup by segment_idx)
    dataset_idx: int = -1                  # index into config["datasets"] (-1 = unknown)
    dataset_name: str = ""                 # human-readable dataset name
    dataset_local_idx: int = -1            # per-dataset patch index (stable within a cache)
    volume_group: Any = None               # ZarrGroup for multi-level access (scale augmentation)
    cache_to_volume: Any = None            # (3,4) ndarray affine: cache-scale ZYX → volume reading-level ZYX (cross-volume transform)

    @property
    def wrap_count(self) -> int:
        """Number of wraps in this chunk."""
        return len(self.wraps)

    @property
    def has_multiple_wraps(self) -> bool:
        """Whether this chunk has more than one wrap."""
        return len(self.wraps) > 1

    @property
    def segment_ids(self) -> List[str]:
        """List of unique segment UUIDs in this chunk."""
        return list(set(w["segment"].uuid for w in self.wraps))



def make_gaussian_heatmap(coords, crop_size, sigma: float = 2.0, axis_1d=None):
    """
    Create a 3D gaussian heatmap centered at one or more coords.

    Uses sparse/scattered placement to avoid massive memory allocation.
    Only computes gaussian values within 3*sigma of each point.

    Args:
        coords: (N, 3) or (3,) tensor, or list of (3,) tensors - position(s) in crop-local coordinates (0 to crop_size-1)
        crop_size: int or tuple - size of the output volume
        sigma: float - gaussian standard deviation (default 2.0)
        axis_1d: ignored (kept for API compatibility)

    Returns:
        (D, H, W) tensor with gaussian(s) centered at coords.
        If multiple coords provided, heatmaps are combined using max.
    """
    # Handle inputs
    if isinstance(coords, list):
        if len(coords) == 0:
            if isinstance(crop_size, int):
                return torch.zeros(crop_size, crop_size, crop_size)
            else:
                return torch.zeros(*crop_size)
        coords = torch.stack(coords)

    if coords.dim() == 1:
        coords = coords.unsqueeze(0)

    # Determine output shape
    if isinstance(crop_size, int):
        shape = (crop_size, crop_size, crop_size)
    else:
        shape = tuple(crop_size)

    # Initialize output
    heatmap = torch.zeros(shape, dtype=torch.float32)

    # Radius to compute (3*sigma captures 99.7% of gaussian)
    radius = int(np.ceil(3 * sigma))

    # Precompute 1D gaussian values for efficiency
    r = torch.arange(-radius, radius + 1, dtype=torch.float32)
    gauss_1d = torch.exp(-r**2 / (2 * sigma**2))

    # Place gaussian at each point location
    for i in range(len(coords)):
        cz, cy, cx = coords[i]
        cz, cy, cx = int(round(cz.item())), int(round(cy.item())), int(round(cx.item()))

        # Compute bounds (clipped to volume)
        z0, z1 = max(0, cz - radius), min(shape[0], cz + radius + 1)
        y0, y1 = max(0, cy - radius), min(shape[1], cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(shape[2], cx + radius + 1)

        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue  # Point is outside the volume

        # Corresponding indices into gaussian kernel
        kz0, kz1 = z0 - (cz - radius), z1 - (cz - radius)
        ky0, ky1 = y0 - (cy - radius), y1 - (cy - radius)
        kx0, kx1 = x0 - (cx - radius), x1 - (cx - radius)

        # Compute local 3D gaussian via outer product of 1D gaussians
        local_gauss = gauss_1d[kz0:kz1, None, None] * gauss_1d[None, ky0:ky1, None] * gauss_1d[None, None, kx0:kx1]

        # Update with max (for overlapping gaussians)
        heatmap[z0:z1, y0:y1, x0:x1] = torch.maximum(
            heatmap[z0:z1, y0:y1, x0:x1],
            local_gauss
        )

    return heatmap


def compute_heatmap_targets(
    cond_direction: str,
    r_split: int, c_split: int,
    r_min_full: int, r_max_full: int,
    c_min_full: int, c_max_full: int,
    patch_seg,
    min_corner: np.ndarray,
    crop_size: tuple,
    step_size: int,
    step_count: int,
    sigma: float = 2.0,
    axis_1d: torch.Tensor = None,
) -> torch.Tensor:
    """
    Generate heatmap with gaussians at expected positions in the masked region.

    Samples a sparse grid of points with step_size spacing in both row and col.

    Args:
        cond_direction: One of "left", "right", "up", "down"
        r_split, c_split: Split boundary in UV grid
        r_min_full, r_max_full, c_min_full, c_max_full: Patch bounds in UV grid
        patch_seg: tifxyz Surface for indexing world coords
        min_corner: Crop origin in world coords (ZYX)
        crop_size: Output crop size tuple (D, H, W)
        step_size: Spacing between gaussians in UV grid units (both row and col)
        step_count: Number of steps to sample in the extrapolation direction
        sigma: Gaussian standard deviation
        axis_1d: Pre-computed axis tensor for efficiency (ignored, kept for API compat)

    Returns:
        (D, H, W) tensor with gaussians at expected positions
    """
    all_local_coords = []

    # Generate row indices with step_size spacing
    row_indices = list(range(r_min_full, r_max_full, step_size))

    if cond_direction == "left":
        for k in range(1, step_count + 1):
            col = c_split + k * step_size
            if col >= c_max_full:
                continue
            for row in row_indices:
                if not patch_seg.valid[row, col]:
                    continue
                world_zyx = patch_seg.xyz[row, col, ::-1].astype(np.float64)
                all_local_coords.append(world_zyx - min_corner)

    elif cond_direction == "right":
        for k in range(1, step_count + 1):
            col = c_split - k * step_size
            if col < c_min_full:
                continue
            for row in row_indices:
                if not patch_seg.valid[row, col]:
                    continue
                world_zyx = patch_seg.xyz[row, col, ::-1].astype(np.float64)
                all_local_coords.append(world_zyx - min_corner)

    elif cond_direction == "up":
        col_indices = list(range(c_min_full, c_max_full, step_size))
        for k in range(1, step_count + 1):
            row = r_split + k * step_size
            if row >= r_max_full:
                continue
            for col in col_indices:
                if not patch_seg.valid[row, col]:
                    continue
                world_zyx = patch_seg.xyz[row, col, ::-1].astype(np.float64)
                all_local_coords.append(world_zyx - min_corner)

    elif cond_direction == "down":
        col_indices = list(range(c_min_full, c_max_full, step_size))
        for k in range(1, step_count + 1):
            row = r_split - k * step_size
            if row < r_min_full:
                continue
            for col in col_indices:
                if not patch_seg.valid[row, col]:
                    continue
                world_zyx = patch_seg.xyz[row, col, ::-1].astype(np.float64)
                all_local_coords.append(world_zyx - min_corner)

    if not all_local_coords:
        print(f"[compute_heatmap_targets] No valid coords found for direction={cond_direction}")
        return None

    all_coords = np.stack(all_local_coords, axis=0)
    # Filter to in-bounds
    in_bounds = (
        (all_coords[:, 0] >= 0) & (all_coords[:, 0] < crop_size[0]) &
        (all_coords[:, 1] >= 0) & (all_coords[:, 1] < crop_size[1]) &
        (all_coords[:, 2] >= 0) & (all_coords[:, 2] < crop_size[2])
    )
    all_coords = all_coords[in_bounds]

    if len(all_coords) == 0:
        print(f"[compute_heatmap_targets] All coords out of bounds for direction={cond_direction}")
        return None

    coords_tensor = torch.from_numpy(all_coords).float()
    return make_gaussian_heatmap(coords_tensor, crop_size[0], sigma=sigma, axis_1d=axis_1d)
