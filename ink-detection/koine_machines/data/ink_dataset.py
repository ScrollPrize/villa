import json
from pathlib import Path
import random
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import cc3d
import torch
from torch.utils.data import Dataset
import numpy as np 
from numba import njit
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from koine_machines.augmentation.translation import maybe_translate_normal_pooled_crop_bbox
from koine_machines.common.common import (
    _read_bbox_with_padding,
    flat_patch_cache_path,
    flat_patch_finding_cache_token,
    load_flat_patch_cache,
    open_zarr,
    save_flat_patch_cache,
)
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
from vesuvius.image_proc.intensity.normalization import normalize_robust
import vesuvius.tifxyz as tifxyz
from koine_machines.data.patch import Patch
from koine_machines.data.normal_pooled_sample import (
    _build_normal_pooled_flat_metadata,
    _compute_normals_local_zyx_from_position_halo,
    _filter_support_components_by_active_supervision,
    _project_flat_patch_to_native_crop,
    _pack_normal_pooled_augmentation_data,
    _project_flat_labels_and_supervision_to_native_crop,
    _project_valid_surface_mask_to_native_crop,
    _select_flat_pixels_for_native_crop,
    _restore_normal_pooled_augmentation_data,
    _select_flat_pixels_for_native_crop_via_stored_resolution,
    _slice_support_halo_for_subwindow,
)
from koine_machines.data.native_crop import compute_native_crop_bbox_from_patch_points
from koine_machines.data.segment import Segment


_FULL_3D_SINGLE_WRAP_MODE = "full_3d_single_wrap"


_GEOMETRIC_TRANSFORM_KEYWORDS = (
    'Rot90', 'Mirror', 'Spatial', 'Affine', 'Elastic', 'CropPad', 'Resize',
)


def _is_geometric_transform(t) -> bool:
    """True iff the transform only changes spatial layout (no intensity change).

    Walks RandomTransform / OneOfTransform wrappers. A OneOf is geometric only
    if every branch is geometric — mixed groups fall to the photometric side
    so the label-generation image stays free of any intensity perturbation.
    """
    name = type(t).__name__
    if any(k in name for k in _GEOMETRIC_TRANSFORM_KEYWORDS):
        return True
    inner = getattr(t, 'transform', None)
    if inner is not None:
        return _is_geometric_transform(inner)
    list_of = getattr(t, 'list_of_transforms', None)
    if list_of:
        return all(_is_geometric_transform(s) for s in list_of)
    return False


def _split_augmentations_by_geometry(compose: ComposeTransforms):
    geo, photo = [], []
    for t in compose.transforms:
        (geo if _is_geometric_transform(t) else photo).append(t)
    return ComposeTransforms(geo), ComposeTransforms(photo)
_FULL_3D_LIKE_MODES = {"full_3d", _FULL_3D_SINGLE_WRAP_MODE}
_NATIVE_3D_MODES = {"normal_pooled_3d", *_FULL_3D_LIKE_MODES}
_REMOTE_VOLUME_PREFIXES = ("s3://", "http://", "https://")
_DEFAULT_FULL_3D_PROJECTION_HALF_THICKNESS = 1.0
_DEFAULT_IMAGE_NORMALIZATION = "robust_mad"
_IMAGE_NORMALIZATION_ALIASES = {
    "robust": "robust_mad",
    "robust_mad": "robust_mad",
    "mad": "robust_mad",
    "robust_percentile": "robust_percentile_span",
    "robust_percentile_span": "robust_percentile_span",
    "percentile_span": "robust_percentile_span",
    "minmax": "minmax",
    "min_max": "minmax",
    "percentile_minmax": "percentile_minmax",
    "percentile_min_max": "percentile_minmax",
    "clipped_minmax": "percentile_minmax",
    "clipped_min_max": "percentile_minmax",
    "none": "none",
    "identity": "none",
}


def _is_native_3d_mode(mode):
    return str(mode).strip().lower() in _NATIVE_3D_MODES


def _is_full_3d_like_mode(mode):
    return str(mode).strip().lower() in _FULL_3D_LIKE_MODES


def _includes_intersecting_segments_3d(mode):
    return str(mode).strip().lower() == "full_3d"


def _uses_surface_mask_channel(mode):
    return str(mode).strip().lower() in {"normal_pooled_3d", _FULL_3D_SINGLE_WRAP_MODE}


def _is_remote_volume_path(path):
    return str(path).lower().startswith(_REMOTE_VOLUME_PREFIXES)


def _coerce_volume_path(path):
    path_str = str(path)
    if _is_remote_volume_path(path_str):
        return path_str
    return Path(path_str)


def _image_normalization_config(config):
    normalization_config = (config or {}).get("image_normalization", _DEFAULT_IMAGE_NORMALIZATION)
    if isinstance(normalization_config, str):
        mode = normalization_config
        options = {}
    elif isinstance(normalization_config, dict):
        mode = normalization_config.get("mode", _DEFAULT_IMAGE_NORMALIZATION)
        options = normalization_config
    elif normalization_config is None:
        mode = _DEFAULT_IMAGE_NORMALIZATION
        options = {}
    else:
        raise ValueError(
            "image_normalization must be a string, object, or null, "
            f"got {type(normalization_config).__name__}"
        )

    normalized_mode = str(mode).strip().lower()
    normalized_mode = _IMAGE_NORMALIZATION_ALIASES.get(normalized_mode, normalized_mode)
    if normalized_mode not in {"robust_mad", "robust_percentile_span", "minmax", "percentile_minmax", "none"}:
        allowed = ", ".join(sorted(_IMAGE_NORMALIZATION_ALIASES))
        raise ValueError(f"Unsupported image_normalization mode {mode!r}; allowed: {allowed}")
    return normalized_mode, options


def _normalization_percentiles(options):
    lower = float(options.get("percentile_lower", 1.0))
    upper = float(options.get("percentile_upper", 99.0))
    if not (0.0 <= lower < upper <= 100.0):
        raise ValueError(
            "image_normalization percentiles must satisfy "
            f"0 <= lower < upper <= 100, got {lower!r}, {upper!r}"
        )
    return lower, upper


def _normalize_robust_percentile_span(image, *, percentile_lower=1.0, percentile_upper=99.0):
    arr = np.asarray(image).astype(np.float32, copy=False)
    if arr.size == 0:
        return arr

    lower_val = float(np.percentile(arr, percentile_lower))
    upper_val = float(np.percentile(arr, percentile_upper))
    np.clip(arr, lower_val, upper_val, out=arr)

    median = float(np.median(arr))
    scale = 0.5 * (upper_val - lower_val)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    arr -= median
    arr /= scale
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _normalize_minmax(image, *, percentile_lower=0.0, percentile_upper=100.0):
    arr = np.asarray(image).astype(np.float32, copy=False)
    if arr.size == 0:
        return arr

    lower_val = float(np.percentile(arr, percentile_lower))
    upper_val = float(np.percentile(arr, percentile_upper))
    np.clip(arr, lower_val, upper_val, out=arr)

    scale = upper_val - lower_val
    if not np.isfinite(scale) or scale < 1e-6:
        arr.fill(0.0)
        return arr

    arr -= lower_val
    arr /= scale
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _normalize_image_crop(image, config):
    mode, options = _image_normalization_config(config)
    if mode == "none":
        return np.asarray(image).astype(np.float32, copy=False)

    percentile_lower, percentile_upper = _normalization_percentiles(options)
    if mode == "robust_mad":
        return normalize_robust(
            image,
            percentile_lower=percentile_lower,
            percentile_upper=percentile_upper,
        )
    if mode == "robust_percentile_span":
        return _normalize_robust_percentile_span(
            image,
            percentile_lower=percentile_lower,
            percentile_upper=percentile_upper,
        )
    if mode == "minmax":
        return _normalize_minmax(image)
    if mode == "percentile_minmax":
        return _normalize_minmax(
            image,
            percentile_lower=percentile_lower,
            percentile_upper=percentile_upper,
        )

    raise AssertionError(f"Unhandled image normalization mode {mode!r}")


def _read_flat_surface_patch(volume, *, y0, y1, x0, x1):
    surface = int(volume.shape[0] // 2)
    patch, _ = _read_bbox_with_padding(
        volume,
        (surface, int(y0), int(x0), surface + 1, int(y1), int(x1)),
        fill_value=0,
    )
    return patch[0]


def _exclude_validation_voxels_from_training_supervision(
    supervision_patch,
    validation_patch,
    *,
    is_validation_patch=False,
):
    if is_validation_patch or validation_patch is None:
        return supervision_patch

    supervision_patch = np.asarray(supervision_patch)
    validation_patch = np.asarray(validation_patch)
    if supervision_patch.shape != validation_patch.shape:
        raise ValueError(
            "supervision_patch and validation_patch must have matching shapes, "
            f"got {tuple(supervision_patch.shape)} and {tuple(validation_patch.shape)}"
        )
    if supervision_patch.size == 0 or not np.any(validation_patch):
        return supervision_patch

    masked_supervision = np.array(supervision_patch, copy=True)
    masked_supervision[validation_patch > 0] = 0
    return masked_supervision


def _select_flat_pixels_for_native_crop(patch_zyxs, valid_mask, crop_bbox):
    patch_zyxs = np.asarray(patch_zyxs)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    crop_start = np.asarray(crop_bbox[:3], dtype=np.int64)
    crop_stop = np.asarray(crop_bbox[3:], dtype=np.int64)

    finite_mask = np.isfinite(patch_zyxs).all(axis=-1)
    within = valid_mask & finite_mask
    within &= patch_zyxs[..., 0] >= crop_start[0]
    within &= patch_zyxs[..., 0] < crop_stop[0]
    within &= patch_zyxs[..., 1] >= crop_start[1]
    within &= patch_zyxs[..., 1] < crop_stop[1]
    within &= patch_zyxs[..., 2] >= crop_start[2]
    within &= patch_zyxs[..., 2] < crop_stop[2]
    if not np.any(within):
        raise ValueError(f"crop_bbox {crop_bbox!r} does not intersect any valid flat tifxyz pixels")

    # We only need the enclosing row/column span, not every matching index.
    row_hits = np.any(within, axis=1)
    col_hits = np.any(within, axis=0)
    row_indices = np.flatnonzero(row_hits)
    col_indices = np.flatnonzero(col_hits)
    support_y0 = int(row_indices[0])
    support_y1 = int(row_indices[-1]) + 1
    support_x0 = int(col_indices[0])
    support_x1 = int(col_indices[-1]) + 1
    return (
        (support_y0, support_y1, support_x0, support_x1),
        patch_zyxs[support_y0:support_y1, support_x0:support_x1],
        within[support_y0:support_y1, support_x0:support_x1],
    )


def _select_flat_pixels_for_native_crop_via_stored_resolution(
    patch_tifxyz,
    crop_bbox,
    *,
    coarse_native_pad=20,
    coarse_patch_zyxs=None,
    coarse_valid=None,
    return_halo=False,
):
    coarse_native_pad = int(coarse_native_pad)
    coarse_crop_bbox = (
        int(crop_bbox[0]) - coarse_native_pad,
        int(crop_bbox[1]) - coarse_native_pad,
        int(crop_bbox[2]) - coarse_native_pad,
        int(crop_bbox[3]) + coarse_native_pad,
        int(crop_bbox[4]) + coarse_native_pad,
        int(crop_bbox[5]) + coarse_native_pad,
    )

    if coarse_patch_zyxs is None:
        coarse_patch_zyxs = np.asarray(
            patch_tifxyz.get_zyxs(stored_resolution=True),
            dtype=np.float32,
        )
    else:
        coarse_patch_zyxs = np.asarray(coarse_patch_zyxs, dtype=np.float32)

    if coarse_valid is None:
        coarse_valid = np.isfinite(coarse_patch_zyxs).all(axis=-1)
        coarse_valid &= (coarse_patch_zyxs >= 0).all(axis=-1)
    else:
        coarse_valid = np.asarray(coarse_valid, dtype=bool)

    (coarse_y0, coarse_y1, coarse_x0, coarse_x1), _, _ = _select_flat_pixels_for_native_crop(
        coarse_patch_zyxs,
        coarse_valid,
        coarse_crop_bbox,
    )

    stored_h, stored_w = (int(v) for v in coarse_patch_zyxs.shape[:2])
    full_h, full_w = (int(v) for v in patch_tifxyz.full_resolution_shape)
    if stored_h <= 0 or stored_w <= 0:
        raise ValueError(f"stored-resolution tifxyz grid must have positive shape, got {(stored_h, stored_w)!r}")

    factor_y = full_h / float(stored_h)
    factor_x = full_w / float(stored_w)

    # Expand by one stored cell before mapping back to full resolution so the
    # exact full-res refinement can't miss intersections near a coarse edge.
    coarse_y0 = max(0, coarse_y0 - 1)
    coarse_y1 = min(stored_h, coarse_y1 + 1)
    coarse_x0 = max(0, coarse_x0 - 1)
    coarse_x1 = min(stored_w, coarse_x1 + 1)

    full_y0 = max(0, int(np.floor(coarse_y0 * factor_y)))
    full_y1 = min(full_h, int(np.ceil(coarse_y1 * factor_y)))
    full_x0 = max(0, int(np.floor(coarse_x0 * factor_x)))
    full_x1 = min(full_w, int(np.ceil(coarse_x1 * factor_x)))

    full_x, full_y, full_z, full_valid = patch_tifxyz[full_y0:full_y1, full_x0:full_x1]
    full_patch_zyxs = np.stack([full_z, full_y, full_x], axis=-1)
    (local_y0, local_y1, local_x0, local_x1), support_patch_zyxs, support_valid = _select_flat_pixels_for_native_crop(
        full_patch_zyxs,
        full_valid,
        crop_bbox,
    )
    support_bbox = (
        full_y0 + local_y0,
        full_y0 + local_y1,
        full_x0 + local_x0,
        full_x0 + local_x1,
    )
    if not return_halo:
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
        )

    halo_local_y0 = max(0, local_y0 - 1)
    halo_local_y1 = min(full_patch_zyxs.shape[0], local_y1 + 1)
    halo_local_x0 = max(0, local_x0 - 1)
    halo_local_x1 = min(full_patch_zyxs.shape[1], local_x1 + 1)
    support_patch_zyxs_halo = full_patch_zyxs[halo_local_y0:halo_local_y1, halo_local_x0:halo_local_x1]
    support_valid_halo = full_valid[halo_local_y0:halo_local_y1, halo_local_x0:halo_local_x1]
    trim_slices = (
        slice(local_y0 - halo_local_y0, local_y1 - halo_local_y0),
        slice(local_x0 - halo_local_x0, local_x1 - halo_local_x0),
    )
    return (
        support_bbox,
        support_patch_zyxs,
        support_valid,
        support_patch_zyxs_halo,
        support_valid_halo,
        trim_slices,
    )


def _project_flat_patch_to_native_crop(flat_patch, patch_zyxs, valid_mask, crop_bbox):
    z0, y0, x0, z1, y1, x1 = (int(v) for v in crop_bbox)
    output = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.asarray(flat_patch).dtype)

    positive_mask = np.asarray(flat_patch) != 0
    valid_mask = np.asarray(valid_mask, dtype=bool) & positive_mask
    if not np.any(valid_mask):
        return output

    patch_zyxs = np.asarray(patch_zyxs)
    finite_mask = np.isfinite(patch_zyxs).all(axis=-1)
    valid_mask &= finite_mask
    if not np.any(valid_mask):
        return output

    mapped_zyxs = patch_zyxs[valid_mask].astype(np.int64, copy=False)
    local_zyxs = mapped_zyxs - np.asarray((z0, y0, x0), dtype=np.int32)
    within_crop = (
        (local_zyxs[:, 0] >= 0)
        & (local_zyxs[:, 0] < output.shape[0])
        & (local_zyxs[:, 1] >= 0)
        & (local_zyxs[:, 1] < output.shape[1])
        & (local_zyxs[:, 2] >= 0)
        & (local_zyxs[:, 2] < output.shape[2])
    )
    if not np.any(within_crop):
        return output

    local_zyxs = local_zyxs[within_crop]
    values = np.asarray(flat_patch)[valid_mask][within_crop]
    flat_indices = np.ravel_multi_index(local_zyxs.T, output.shape)
    np.maximum.at(output.reshape(-1), flat_indices, values)
    return output


@njit(cache=True)
def _mark_projected_voxel(output, z, y, x, z0, y0, x0):
    local_z = int(z) - z0
    local_y = int(y) - y0
    local_x = int(x) - x0
    if (
        local_z >= 0
        and local_z < output.shape[0]
        and local_y >= 0
        and local_y < output.shape[1]
        and local_x >= 0
        and local_x < output.shape[2]
    ):
        output[local_z, local_y, local_x] = 1


@njit(cache=True)
def _draw_projected_line(output, start_zyx, stop_zyx, z0, y0, x0):
    delta_z = stop_zyx[0] - start_zyx[0]
    delta_y = stop_zyx[1] - start_zyx[1]
    delta_x = stop_zyx[2] - start_zyx[2]
    steps = int(np.ceil(max(abs(delta_z), abs(delta_y), abs(delta_x))))
    if steps <= 0:
        _mark_projected_voxel(output, start_zyx[0], start_zyx[1], start_zyx[2], z0, y0, x0)
        return

    inv_steps = 1.0 / float(steps)
    for step in range(steps + 1):
        t = float(step) * inv_steps
        _mark_projected_voxel(
            output,
            start_zyx[0] + delta_z * t,
            start_zyx[1] + delta_y * t,
            start_zyx[2] + delta_x * t,
            z0,
            y0,
            x0,
        )


@njit(cache=True)
def _chebyshev_distance_zyx(start_zyx, stop_zyx):
    distance = abs(stop_zyx[0] - start_zyx[0])
    delta = abs(stop_zyx[1] - start_zyx[1])
    if delta > distance:
        distance = delta
    delta = abs(stop_zyx[2] - start_zyx[2])
    if delta > distance:
        distance = delta
    return distance


@njit(cache=True)
def _dense_interpolation_steps(distance_voxels):
    steps = int(np.ceil(float(distance_voxels) * 2.0))
    if steps < 1:
        return 1
    return steps


@njit(cache=True)
def _draw_projected_bilinear_patch(output, p00, p01, p10, p11, z0, y0, x0):
    row_distance = _chebyshev_distance_zyx(p00, p10)
    distance = _chebyshev_distance_zyx(p01, p11)
    if distance > row_distance:
        row_distance = distance
    col_distance = _chebyshev_distance_zyx(p00, p01)
    distance = _chebyshev_distance_zyx(p10, p11)
    if distance > col_distance:
        col_distance = distance

    row_steps = _dense_interpolation_steps(row_distance)
    col_steps = _dense_interpolation_steps(col_distance)
    inv_row_steps = 1.0 / float(row_steps)
    inv_col_steps = 1.0 / float(col_steps)

    for row_step in range(row_steps + 1):
        row_t = float(row_step) * inv_row_steps
        inv_row_t = 1.0 - row_t
        left_z = p00[0] * inv_row_t + p10[0] * row_t
        left_y = p00[1] * inv_row_t + p10[1] * row_t
        left_x = p00[2] * inv_row_t + p10[2] * row_t
        right_z = p01[0] * inv_row_t + p11[0] * row_t
        right_y = p01[1] * inv_row_t + p11[1] * row_t
        right_x = p01[2] * inv_row_t + p11[2] * row_t

        for col_step in range(col_steps + 1):
            col_t = float(col_step) * inv_col_steps
            inv_col_t = 1.0 - col_t
            _mark_projected_voxel(
                output,
                left_z * inv_col_t + right_z * col_t,
                left_y * inv_col_t + right_y * col_t,
                left_x * inv_col_t + right_x * col_t,
                z0,
                y0,
                x0,
            )


@njit(cache=True)
def _draw_projected_trilinear_cell(
    output,
    lower_p00,
    lower_p01,
    lower_p10,
    lower_p11,
    upper_p00,
    upper_p01,
    upper_p10,
    upper_p11,
    z0,
    y0,
    x0,
):
    offset_distance = _chebyshev_distance_zyx(lower_p00, upper_p00)
    distance = _chebyshev_distance_zyx(lower_p01, upper_p01)
    if distance > offset_distance:
        offset_distance = distance
    distance = _chebyshev_distance_zyx(lower_p10, upper_p10)
    if distance > offset_distance:
        offset_distance = distance
    distance = _chebyshev_distance_zyx(lower_p11, upper_p11)
    if distance > offset_distance:
        offset_distance = distance

    row_distance = _chebyshev_distance_zyx(lower_p00, lower_p10)
    distance = _chebyshev_distance_zyx(lower_p01, lower_p11)
    if distance > row_distance:
        row_distance = distance
    distance = _chebyshev_distance_zyx(upper_p00, upper_p10)
    if distance > row_distance:
        row_distance = distance
    distance = _chebyshev_distance_zyx(upper_p01, upper_p11)
    if distance > row_distance:
        row_distance = distance

    col_distance = _chebyshev_distance_zyx(lower_p00, lower_p01)
    distance = _chebyshev_distance_zyx(lower_p10, lower_p11)
    if distance > col_distance:
        col_distance = distance
    distance = _chebyshev_distance_zyx(upper_p00, upper_p01)
    if distance > col_distance:
        col_distance = distance
    distance = _chebyshev_distance_zyx(upper_p10, upper_p11)
    if distance > col_distance:
        col_distance = distance

    offset_steps = _dense_interpolation_steps(offset_distance)
    row_steps = _dense_interpolation_steps(row_distance)
    col_steps = _dense_interpolation_steps(col_distance)
    inv_offset_steps = 1.0 / float(offset_steps)
    inv_row_steps = 1.0 / float(row_steps)
    inv_col_steps = 1.0 / float(col_steps)

    for offset_step in range(offset_steps + 1):
        offset_t = float(offset_step) * inv_offset_steps
        inv_offset_t = 1.0 - offset_t

        p00_z = lower_p00[0] * inv_offset_t + upper_p00[0] * offset_t
        p00_y = lower_p00[1] * inv_offset_t + upper_p00[1] * offset_t
        p00_x = lower_p00[2] * inv_offset_t + upper_p00[2] * offset_t
        p01_z = lower_p01[0] * inv_offset_t + upper_p01[0] * offset_t
        p01_y = lower_p01[1] * inv_offset_t + upper_p01[1] * offset_t
        p01_x = lower_p01[2] * inv_offset_t + upper_p01[2] * offset_t
        p10_z = lower_p10[0] * inv_offset_t + upper_p10[0] * offset_t
        p10_y = lower_p10[1] * inv_offset_t + upper_p10[1] * offset_t
        p10_x = lower_p10[2] * inv_offset_t + upper_p10[2] * offset_t
        p11_z = lower_p11[0] * inv_offset_t + upper_p11[0] * offset_t
        p11_y = lower_p11[1] * inv_offset_t + upper_p11[1] * offset_t
        p11_x = lower_p11[2] * inv_offset_t + upper_p11[2] * offset_t

        for row_step in range(row_steps + 1):
            row_t = float(row_step) * inv_row_steps
            inv_row_t = 1.0 - row_t
            left_z = p00_z * inv_row_t + p10_z * row_t
            left_y = p00_y * inv_row_t + p10_y * row_t
            left_x = p00_x * inv_row_t + p10_x * row_t
            right_z = p01_z * inv_row_t + p11_z * row_t
            right_y = p01_y * inv_row_t + p11_y * row_t
            right_x = p01_x * inv_row_t + p11_x * row_t

            for col_step in range(col_steps + 1):
                col_t = float(col_step) * inv_col_steps
                inv_col_t = 1.0 - col_t
                _mark_projected_voxel(
                    output,
                    left_z * inv_col_t + right_z * col_t,
                    left_y * inv_col_t + right_y * col_t,
                    left_x * inv_col_t + right_x * col_t,
                    z0,
                    y0,
                    x0,
                )


@njit(cache=True)
def _projected_position_for_normal_offset(
    patch_zyxs,
    normals_local_zyx,
    row,
    col,
    offset,
    out_zyx,
):
    point_z = patch_zyxs[row, col, 0]
    point_y = patch_zyxs[row, col, 1]
    point_x = patch_zyxs[row, col, 2]
    normal_z = normals_local_zyx[row, col, 0]
    normal_y = normals_local_zyx[row, col, 1]
    normal_x = normals_local_zyx[row, col, 2]
    if (
        not np.isfinite(point_z)
        or not np.isfinite(point_y)
        or not np.isfinite(point_x)
        or not np.isfinite(normal_z)
        or not np.isfinite(normal_y)
        or not np.isfinite(normal_x)
    ):
        return False

    normal_mag = np.sqrt(
        normal_z * normal_z
        + normal_y * normal_y
        + normal_x * normal_x
    )
    if normal_mag <= 1e-6:
        return False

    inv_normal_mag = 1.0 / normal_mag
    out_zyx[0] = point_z + offset * normal_z * inv_normal_mag
    out_zyx[1] = point_y + offset * normal_y * inv_normal_mag
    out_zyx[2] = point_x + offset * normal_x * inv_normal_mag
    return True


@njit(cache=True)
def _project_binary_mask_along_normals_numba(
    flat_mask,
    patch_zyxs,
    normals_local_zyx,
    valid_mask,
    crop_start_zyx,
    output,
    half_thickness_voxels,
):
    z0 = int(crop_start_zyx[0])
    y0 = int(crop_start_zyx[1])
    x0 = int(crop_start_zyx[2])
    radius = int(np.ceil(half_thickness_voxels))
    current_zyx = np.empty((3,), dtype=np.float32)
    previous_offset_zyx = np.empty((3,), dtype=np.float32)
    neighbor_zyx = np.empty((3,), dtype=np.float32)
    right_zyx = np.empty((3,), dtype=np.float32)
    down_zyx = np.empty((3,), dtype=np.float32)
    diag_zyx = np.empty((3,), dtype=np.float32)
    previous_right_zyx = np.empty((3,), dtype=np.float32)
    previous_down_zyx = np.empty((3,), dtype=np.float32)
    previous_diag_zyx = np.empty((3,), dtype=np.float32)

    for row in range(flat_mask.shape[0]):
        for col in range(flat_mask.shape[1]):
            if flat_mask[row, col] == 0 or not valid_mask[row, col]:
                continue

            has_previous_offset = False
            has_previous_cell = False
            for offset_step in range(-radius, radius + 1):
                if abs(offset_step) > half_thickness_voxels + 1e-6:
                    continue

                offset = float(offset_step)
                if not _projected_position_for_normal_offset(
                    patch_zyxs,
                    normals_local_zyx,
                    row,
                    col,
                    offset,
                    current_zyx,
                ):
                    break

                _mark_projected_voxel(
                    output,
                    current_zyx[0],
                    current_zyx[1],
                    current_zyx[2],
                    z0,
                    y0,
                    x0,
                )
                if has_previous_offset:
                    _draw_projected_line(output, previous_offset_zyx, current_zyx, z0, y0, x0)

                right_ok = (
                    col + 1 < flat_mask.shape[1]
                    and flat_mask[row, col + 1] != 0
                    and valid_mask[row, col + 1]
                    and _projected_position_for_normal_offset(
                        patch_zyxs,
                        normals_local_zyx,
                        row,
                        col + 1,
                        offset,
                        right_zyx,
                    )
                )
                if right_ok:
                    _draw_projected_line(output, current_zyx, right_zyx, z0, y0, x0)

                down_ok = (
                    row + 1 < flat_mask.shape[0]
                    and flat_mask[row + 1, col] != 0
                    and valid_mask[row + 1, col]
                    and _projected_position_for_normal_offset(
                        patch_zyxs,
                        normals_local_zyx,
                        row + 1,
                        col,
                        offset,
                        down_zyx,
                    )
                )
                if down_ok:
                    _draw_projected_line(output, current_zyx, down_zyx, z0, y0, x0)

                diag_ok = (
                    row + 1 < flat_mask.shape[0]
                    and col + 1 < flat_mask.shape[1]
                    and flat_mask[row + 1, col + 1] != 0
                    and valid_mask[row + 1, col + 1]
                    and _projected_position_for_normal_offset(
                        patch_zyxs,
                        normals_local_zyx,
                        row + 1,
                        col + 1,
                        offset,
                        diag_zyx,
                    )
                )
                cell_ok = right_ok and down_ok and diag_ok
                if cell_ok:
                    _draw_projected_bilinear_patch(
                        output,
                        current_zyx,
                        right_zyx,
                        down_zyx,
                        diag_zyx,
                        z0,
                        y0,
                        x0,
                    )
                    if has_previous_cell:
                        _draw_projected_trilinear_cell(
                            output,
                            previous_offset_zyx,
                            previous_right_zyx,
                            previous_down_zyx,
                            previous_diag_zyx,
                            current_zyx,
                            right_zyx,
                            down_zyx,
                            diag_zyx,
                            z0,
                            y0,
                            x0,
                        )

                previous_offset_zyx[0] = current_zyx[0]
                previous_offset_zyx[1] = current_zyx[1]
                previous_offset_zyx[2] = current_zyx[2]
                if cell_ok:
                    previous_right_zyx[0] = right_zyx[0]
                    previous_right_zyx[1] = right_zyx[1]
                    previous_right_zyx[2] = right_zyx[2]
                    previous_down_zyx[0] = down_zyx[0]
                    previous_down_zyx[1] = down_zyx[1]
                    previous_down_zyx[2] = down_zyx[2]
                    previous_diag_zyx[0] = diag_zyx[0]
                    previous_diag_zyx[1] = diag_zyx[1]
                    previous_diag_zyx[2] = diag_zyx[2]
                has_previous_offset = True
                has_previous_cell = cell_ok


def _project_flat_binary_mask_along_normals_to_native_crop(
    flat_mask,
    patch_zyxs,
    normals_local_zyx,
    valid_mask,
    crop_bbox,
    *,
    half_thickness_voxels,
):
    half_thickness_voxels = float(half_thickness_voxels)
    if half_thickness_voxels < 0.0:
        raise ValueError(
            f"half_thickness_voxels must be >= 0, got {half_thickness_voxels!r}"
        )

    flat_mask = (np.asarray(flat_mask) > 0).astype(np.uint8, copy=False)
    if half_thickness_voxels <= 0.0:
        return _project_flat_patch_to_native_crop(
            flat_mask,
            patch_zyxs,
            valid_mask,
            crop_bbox,
        ) > 0

    if normals_local_zyx is None:
        raise ValueError("normals_local_zyx is required when projecting with thickness")

    patch_zyxs = np.asarray(patch_zyxs, dtype=np.float32)
    normals_local_zyx = np.asarray(normals_local_zyx, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    if patch_zyxs.shape[:2] != flat_mask.shape or patch_zyxs.shape[-1] != 3:
        raise ValueError(
            "patch_zyxs must have shape (*flat_mask.shape, 3), "
            f"got flat_mask={flat_mask.shape!r}, patch_zyxs={patch_zyxs.shape!r}"
        )
    if normals_local_zyx.shape[:2] != flat_mask.shape or normals_local_zyx.shape[-1] != 3:
        raise ValueError(
            "normals_local_zyx must have shape (*flat_mask.shape, 3), "
            f"got flat_mask={flat_mask.shape!r}, normals={normals_local_zyx.shape!r}"
        )
    if valid_mask.shape != flat_mask.shape:
        raise ValueError(
            "valid_mask must match flat_mask shape, "
            f"got flat_mask={flat_mask.shape!r}, valid_mask={valid_mask.shape!r}"
        )

    z0, y0, x0, z1, y1, x1 = (int(v) for v in crop_bbox)
    output = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.uint8)
    _project_binary_mask_along_normals_numba(
        np.ascontiguousarray(flat_mask),
        np.ascontiguousarray(patch_zyxs),
        np.ascontiguousarray(normals_local_zyx),
        np.ascontiguousarray(valid_mask),
        np.asarray((z0, y0, x0), dtype=np.int64),
        output,
        half_thickness_voxels,
    )
    return output > 0


def _support_normals_local_zyx(
    *,
    patch_tifxyz=None,
    support_bbox=None,
    support_patch_zyxs_halo=None,
    support_valid_halo=None,
    trim_slices=None,
):
    if patch_tifxyz is not None and hasattr(patch_tifxyz, "get_normals"):
        if support_bbox is None:
            raise ValueError("support_bbox is required when patch_tifxyz is provided")
        support_y0, support_y1, support_x0, support_x1 = (int(v) for v in support_bbox)
        nx, ny, nz = patch_tifxyz.get_normals(
            support_y0,
            support_y1,
            support_x0,
            support_x1,
        )
        return np.stack([nz, ny, nx], axis=-1).astype(np.float32, copy=False)

    if support_patch_zyxs_halo is None or support_valid_halo is None or trim_slices is None:
        raise ValueError(
            "support_patch_zyxs_halo, support_valid_halo, and trim_slices are required "
            "when patch_tifxyz normals are unavailable"
        )
    return _compute_normals_local_zyx_from_position_halo(
        support_patch_zyxs_halo,
        support_valid_halo,
        trim_slices,
    )


def _project_valid_surface_occupancy_to_native_crop(patch_zyxs, valid_mask, crop_bbox):
    flat_mask = np.ones(np.asarray(valid_mask).shape, dtype=np.float32)
    surface_occupancy = _project_flat_patch_to_native_crop(flat_mask, patch_zyxs, valid_mask, crop_bbox)
    surface_occupancy = surface_occupancy > 0
    return surface_occupancy.astype(np.float32, copy=False)


def _project_valid_surface_mask_to_native_crop(patch_zyxs, valid_mask, crop_bbox):
    surface_occupancy = _project_valid_surface_occupancy_to_native_crop(
        patch_zyxs,
        valid_mask,
        crop_bbox,
    ) > 0
    if not np.any(surface_occupancy):
        return surface_occupancy.astype(np.float32)

    max_distance_voxels = 10.0
    distance = distance_transform_edt(~surface_occupancy)
    surface_distance_field = np.clip(1.0 - (distance / max_distance_voxels), 0.0, 1.0)
    return surface_distance_field.astype(np.float32, copy=False)


def _maybe_select_flat_pixels_for_native_crop_via_stored_resolution(
    patch_tifxyz,
    crop_bbox,
    *,
    coarse_native_pad=20,
    coarse_patch_zyxs=None,
    coarse_valid=None,
):
    coarse_native_pad = int(coarse_native_pad)
    coarse_crop_bbox = (
        int(crop_bbox[0]) - coarse_native_pad,
        int(crop_bbox[1]) - coarse_native_pad,
        int(crop_bbox[2]) - coarse_native_pad,
        int(crop_bbox[3]) + coarse_native_pad,
        int(crop_bbox[4]) + coarse_native_pad,
        int(crop_bbox[5]) + coarse_native_pad,
    )

    if coarse_patch_zyxs is None:
        coarse_patch_zyxs = np.asarray(
            patch_tifxyz.get_zyxs(stored_resolution=True),
            dtype=np.float32,
        )
    else:
        coarse_patch_zyxs = np.asarray(coarse_patch_zyxs, dtype=np.float32)

    if coarse_valid is None:
        coarse_valid = np.isfinite(coarse_patch_zyxs).all(axis=-1)
        coarse_valid &= (coarse_patch_zyxs >= 0).all(axis=-1)
    else:
        coarse_valid = np.asarray(coarse_valid, dtype=bool)

    coarse_start = np.asarray(coarse_crop_bbox[:3], dtype=np.float32)
    coarse_stop = np.asarray(coarse_crop_bbox[3:], dtype=np.float32)
    coarse_hits = coarse_valid & np.isfinite(coarse_patch_zyxs).all(axis=-1)
    coarse_hits &= (coarse_patch_zyxs >= coarse_start).all(axis=-1)
    coarse_hits &= (coarse_patch_zyxs < coarse_stop).all(axis=-1)
    if not np.any(coarse_hits):
        return None

    row_indices = np.flatnonzero(np.any(coarse_hits, axis=1))
    col_indices = np.flatnonzero(np.any(coarse_hits, axis=0))
    coarse_y0 = int(row_indices[0])
    coarse_y1 = int(row_indices[-1]) + 1
    coarse_x0 = int(col_indices[0])
    coarse_x1 = int(col_indices[-1]) + 1

    stored_h, stored_w = (int(v) for v in coarse_patch_zyxs.shape[:2])
    full_h, full_w = (int(v) for v in patch_tifxyz.full_resolution_shape)
    if stored_h <= 0 or stored_w <= 0:
        raise ValueError(f"stored-resolution tifxyz grid must have positive shape, got {(stored_h, stored_w)!r}")

    factor_y = full_h / float(stored_h)
    factor_x = full_w / float(stored_w)

    coarse_y0 = max(0, coarse_y0 - 1)
    coarse_y1 = min(stored_h, coarse_y1 + 1)
    coarse_x0 = max(0, coarse_x0 - 1)
    coarse_x1 = min(stored_w, coarse_x1 + 1)

    full_y0 = max(0, int(np.floor(coarse_y0 * factor_y)))
    full_y1 = min(full_h, int(np.ceil(coarse_y1 * factor_y)))
    full_x0 = max(0, int(np.floor(coarse_x0 * factor_x)))
    full_x1 = min(full_w, int(np.ceil(coarse_x1 * factor_x)))

    full_x, full_y, full_z, full_valid = patch_tifxyz[full_y0:full_y1, full_x0:full_x1]
    full_patch_zyxs = np.stack([full_z, full_y, full_x], axis=-1)
    crop_start = np.asarray(crop_bbox[:3], dtype=np.float32)
    crop_stop = np.asarray(crop_bbox[3:], dtype=np.float32)
    hits = np.asarray(full_valid, dtype=bool) & np.isfinite(full_patch_zyxs).all(axis=-1)
    hits &= (full_patch_zyxs >= crop_start).all(axis=-1)
    hits &= (full_patch_zyxs < crop_stop).all(axis=-1)
    if not np.any(hits):
        return None

    row_indices = np.flatnonzero(np.any(hits, axis=1))
    col_indices = np.flatnonzero(np.any(hits, axis=0))
    local_y0 = int(row_indices[0])
    local_y1 = int(row_indices[-1]) + 1
    local_x0 = int(col_indices[0])
    local_x1 = int(col_indices[-1]) + 1
    support_bbox = (
        full_y0 + local_y0,
        full_y0 + local_y1,
        full_x0 + local_x0,
        full_x0 + local_x1,
    )
    return (
        support_bbox,
        full_patch_zyxs[local_y0:local_y1, local_x0:local_x1],
        hits[local_y0:local_y1, local_x0:local_x1],
    )


def _project_flat_labels_and_supervision_to_native_crop(
    *,
    support_patch_zyxs,
    support_valid,
    support_inklabels_flat_patch,
    support_supervision_flat_patch,
    crop_bbox,
    support_normals_local_zyx=None,
    label_projection_half_thickness=0.0,
    background_projection_half_thickness=0.0,
):
    support_inklabels = np.asarray(support_inklabels_flat_patch) > 0
    support_supervision = np.asarray(support_supervision_flat_patch) > 0
    support_background = support_supervision & ~support_inklabels

    inklabels_crop = _project_flat_binary_mask_along_normals_to_native_crop(
        support_inklabels,
        support_patch_zyxs,
        support_normals_local_zyx,
        support_valid,
        crop_bbox,
        half_thickness_voxels=label_projection_half_thickness,
    )
    background_crop = _project_flat_binary_mask_along_normals_to_native_crop(
        support_background,
        support_patch_zyxs,
        support_normals_local_zyx,
        support_valid,
        crop_bbox,
        half_thickness_voxels=background_projection_half_thickness,
    )
    background_crop &= ~inklabels_crop
    supervision_crop = inklabels_crop | background_crop
    return (
        inklabels_crop.astype(np.float32, copy=False),
        supervision_crop.astype(np.float32, copy=False),
    )


def _filter_support_by_flat_seeded_component(
    *,
    support_bbox,
    support_valid,
    support_supervision_flat_patch,
    patch_bbox,
    max_grid_distance=None,
):
    support_valid = np.asarray(support_valid, dtype=bool)
    if not np.any(support_valid):
        return support_valid

    support_y0, support_y1, support_x0, support_x1 = (int(v) for v in support_bbox)
    patch_y0, patch_y1, patch_x0, patch_x1 = (
        int(patch_bbox[1]),
        int(patch_bbox[4]),
        int(patch_bbox[2]),
        int(patch_bbox[5]),
    )

    row0 = max(0, patch_y0 - support_y0)
    row1 = min(support_y1 - support_y0, patch_y1 - support_y0)
    col0 = max(0, patch_x0 - support_x0)
    col1 = min(support_x1 - support_x0, patch_x1 - support_x0)

    seed_mask = np.zeros_like(support_valid, dtype=bool)
    if row1 > row0 and col1 > col0:
        seed_mask[row0:row1, col0:col1] = (
            np.asarray(support_supervision_flat_patch)[row0:row1, col0:col1] > 0
        )
    seed_mask &= support_valid

    if not np.any(seed_mask):
        seed_mask = (np.asarray(support_supervision_flat_patch) > 0) & support_valid
    if not np.any(seed_mask):
        return support_valid

    labeled_components, _ = ndimage.label(
        support_valid,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    kept_component_ids = np.unique(labeled_components[seed_mask])
    kept_component_ids = kept_component_ids[kept_component_ids > 0]
    if kept_component_ids.size == 0:
        return support_valid

    filtered_valid = np.isin(labeled_components, kept_component_ids)
    if max_grid_distance is not None:
        filtered_valid &= _filter_support_by_2d_distance_from_supervision(
            filtered_valid,
            seed_mask.astype(np.uint8, copy=False),
            max_grid_distance=max_grid_distance,
        )
    return filtered_valid


def _tighten_support_window(
    support_bbox,
    support_patch_zyxs,
    support_valid,
    support_inklabels_flat_patch,
    support_supervision_flat_patch,
):
    support_valid = np.asarray(support_valid, dtype=bool)
    if not np.any(support_valid):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    row_hits = np.any(support_valid, axis=1)
    col_hits = np.any(support_valid, axis=0)
    row_indices = np.flatnonzero(row_hits)
    col_indices = np.flatnonzero(col_hits)
    row_start = int(row_indices[0])
    row_stop = int(row_indices[-1]) + 1
    col_start = int(col_indices[0])
    col_stop = int(col_indices[-1]) + 1
    support_y0, _, support_x0, _ = (int(v) for v in support_bbox)
    tightened_bbox = (
        support_y0 + row_start,
        support_y0 + row_stop,
        support_x0 + col_start,
        support_x0 + col_stop,
    )
    return (
        tightened_bbox,
        np.asarray(support_patch_zyxs)[row_start:row_stop, col_start:col_stop],
        support_valid[row_start:row_stop, col_start:col_stop],
        np.asarray(support_inklabels_flat_patch)[row_start:row_stop, col_start:col_stop],
        np.asarray(support_supervision_flat_patch)[row_start:row_stop, col_start:col_stop],
    )


def _filter_support_by_2d_distance_from_supervision(
    support_valid,
    support_supervision_flat_patch,
    *,
    max_grid_distance,
):
    support_valid = np.asarray(support_valid, dtype=bool)
    if not np.any(support_valid):
        return support_valid

    max_grid_distance = float(max_grid_distance)
    if not np.isfinite(max_grid_distance) or max_grid_distance < 0:
        raise ValueError(
            f"max_grid_distance must be a finite value >= 0, got {max_grid_distance!r}"
        )

    supervision_seed_mask = (np.asarray(support_supervision_flat_patch) > 0) & support_valid
    if not np.any(supervision_seed_mask):
        return support_valid

    grid_distance = distance_transform_edt(~supervision_seed_mask)
    return support_valid & (grid_distance <= max_grid_distance)


def _slice_support_halo_for_subwindow(
    support_patch_zyxs_halo,
    support_valid_halo,
    trim_slices,
    base_support_bbox,
    subwindow_bbox,
):
    base_support_y0, _, base_support_x0, _ = (int(v) for v in base_support_bbox)
    subwindow_y0, subwindow_y1, subwindow_x0, subwindow_x1 = (int(v) for v in subwindow_bbox)
    row_start = subwindow_y0 - base_support_y0
    row_stop = subwindow_y1 - base_support_y0
    col_start = subwindow_x0 - base_support_x0
    col_stop = subwindow_x1 - base_support_x0

    halo_row_start = max(0, int(trim_slices[0].start) + row_start - 1)
    halo_row_stop = min(support_patch_zyxs_halo.shape[0], int(trim_slices[0].start) + row_stop + 1)
    halo_col_start = max(0, int(trim_slices[1].start) + col_start - 1)
    halo_col_stop = min(support_patch_zyxs_halo.shape[1], int(trim_slices[1].start) + col_stop + 1)

    return (
        np.asarray(support_patch_zyxs_halo)[halo_row_start:halo_row_stop, halo_col_start:halo_col_stop],
        np.asarray(support_valid_halo)[halo_row_start:halo_row_stop, halo_col_start:halo_col_stop],
        (
            slice(int(trim_slices[0].start) + row_start - halo_row_start, int(trim_slices[0].start) + row_stop - halo_row_start),
            slice(int(trim_slices[1].start) + col_start - halo_col_start, int(trim_slices[1].start) + col_stop - halo_col_start),
        ),
    )


def _filter_support_components_by_active_supervision(
    *,
    support_bbox,
    support_patch_zyxs,
    support_valid,
    support_inklabels_flat_patch,
    support_supervision_flat_patch,
    crop_bbox,
    patch_bbox,
    max_supervision_grid_distance=None,
):
    support_valid = np.asarray(support_valid, dtype=bool)
    if not np.any(support_valid):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    occupancy = _project_flat_patch_to_native_crop(
        np.ones(np.asarray(support_valid).shape, dtype=np.uint8),
        support_patch_zyxs,
        support_valid,
        crop_bbox,
    )
    occupancy = occupancy > 0
    if not np.any(occupancy):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    components = cc3d.connected_components(
        occupancy.astype(np.uint8, copy=False),
        connectivity=26,
    )
    supervision_native = _project_flat_patch_to_native_crop(
        (np.asarray(support_supervision_flat_patch) > 0).astype(np.uint8, copy=False),
        support_patch_zyxs,
        support_valid,
        crop_bbox,
    )
    kept_component_ids = np.unique(components[supervision_native > 0])
    kept_component_ids = kept_component_ids[kept_component_ids > 0]
    if kept_component_ids.size == 0:
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    patch_zyxs = np.asarray(support_patch_zyxs)
    finite_mask = np.isfinite(patch_zyxs).all(axis=-1)
    flat_valid = support_valid & finite_mask
    if not np.any(flat_valid):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    z0, y0, x0, z1, y1, x1 = (int(v) for v in crop_bbox)
    mapped_zyxs = patch_zyxs[flat_valid].astype(np.int64, copy=False)
    local_zyxs = mapped_zyxs - np.asarray((z0, y0, x0), dtype=np.int64)
    within_crop = (
        (local_zyxs[:, 0] >= 0)
        & (local_zyxs[:, 0] < (z1 - z0))
        & (local_zyxs[:, 1] >= 0)
        & (local_zyxs[:, 1] < (y1 - y0))
        & (local_zyxs[:, 2] >= 0)
        & (local_zyxs[:, 2] < (x1 - x0))
    )
    if not np.any(within_crop):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    row_indices, col_indices = np.nonzero(flat_valid)
    row_indices = row_indices[within_crop]
    col_indices = col_indices[within_crop]
    local_zyxs = local_zyxs[within_crop]
    component_ids = components[
        local_zyxs[:, 0],
        local_zyxs[:, 1],
        local_zyxs[:, 2],
    ]
    keep_flat = np.isin(component_ids, kept_component_ids)
    if not np.any(keep_flat):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    filtered_valid = np.zeros_like(support_valid, dtype=bool)
    filtered_valid[row_indices[keep_flat], col_indices[keep_flat]] = True
    filtered_valid = _filter_support_by_flat_seeded_component(
        support_bbox=support_bbox,
        support_valid=filtered_valid,
        support_supervision_flat_patch=support_supervision_flat_patch,
        patch_bbox=patch_bbox,
        max_grid_distance=max_supervision_grid_distance,
    )
    return _tighten_support_window(
        support_bbox,
        support_patch_zyxs,
        filtered_valid,
        support_inklabels_flat_patch,
        support_supervision_flat_patch,
    )


class InkDataset(Dataset):
    def __init__(self, config, do_augmentations=True, debug=False, patches=None, mode="flat", unlabeled_segments=None, segments=None):

        self.debug            = debug
        self.config           = config
        self.patch_size       = config['patch_size']
        self.discovery_mode   = str(config.get('patch_discovery_mode', 'labeled')).strip().lower()
        if self.discovery_mode == 'unlabeled':
            self.datasets = list(config.get('unlabeled_datasets') or [])
        else:
            self.datasets = list(config['datasets'])
        self.vol_auth         = config.get('volume_auth_json')
        self.num_workers      = config.get('dataloader_workers', 8)
        self.mode             = str(config.get('mode', 'flat')).strip().lower()
        self.do_augmentations = bool(do_augmentations)
        self.input_channels   = 1 + int(_uses_surface_mask_channel(self.mode))
        self.training_patches = []
        self.validation_patches = []
        self.unlabeled_patches = []
        self.unlabeled_segments = list(unlabeled_segments) if unlabeled_segments else []
        self.segments = []
        self._num_virtual_unlabeled = 0
        self._zarr_cache = {}
        self._tifxyz_cache = {}
        self._stored_resolution_zyx_cache = {}
        self._segments_by_native_volume_key = {}
  

        if self.do_augmentations:
            self.augmentations = create_training_transforms(self.patch_size)
        else:
            self.augmentations = None

        # When the trainer feeds the augmented image through DINO+UNet to make
        # pseudo-labels, photometric augs perturb DINO features. Split the
        # pipeline so we can snapshot a photometrically-clean (but geometrically
        # transformed) image as `image_for_label`.
        self._emit_image_for_label = bool(
            (config.get('dynamic_label') or {}).get('enabled', False)
        ) and self.do_augmentations and self.augmentations is not None
        if self._emit_image_for_label:
            self._geometric_augmentations, self._photometric_augmentations = (
                _split_augmentations_by_geometry(self.augmentations)
            )
        else:
            self._geometric_augmentations = None
            self._photometric_augmentations = None

        if patches is None:
            segments = list(self._gather_segments())
            self.segments = segments
            self._register_segments(segments)

            if self.discovery_mode == 'unlabeled':
                self.unlabeled_segments = segments
                self.patches = []
                return

            cache_path = flat_patch_cache_path(self.config)
            expected_patch_finding_key = flat_patch_finding_cache_token(self.config)
            segments_by_key = {
                seg.cache_key: seg
                for seg in segments
            }

            if cache_path.exists():
                cached_records = load_flat_patch_cache(cache_path)
                cached_patches = []
                cache_valid = True
                for record in cached_records:
                    if record.get('patch_finding_key') != expected_patch_finding_key:
                        cache_valid = False
                        break
                    cache_key = (
                        int(record['dataset_idx']),
                        str(record['segment_relpath']),
                        record['scale'],
                        str(record.get('inklabels_path', '')),
                        str(record.get('supervision_mask_path', '')),
                        str(record.get('validation_mask_path', '')),
                    )
                    segment = segments_by_key.get(cache_key)
                    if segment is None:
                        cache_valid = False
                        break
                    patch = Patch(
                        segment=segment,
                        bbox=tuple(record['bbox']),
                        is_validation=bool(record.get('is_validation', False)),
                        is_unlabeled=bool(record.get('is_unlabeled', False)),
                        supervision_mask_override=record.get('active_supervision_mask_path') or None,
                    )
                    cached_patches.append(patch)
                    if patch.is_unlabeled:
                        self.unlabeled_patches.append(patch)
                        self.training_patches.append(patch)
                    elif patch.is_validation:
                        self.validation_patches.append(patch)
                    else:
                        self.training_patches.append(patch)
                if cache_valid:
                    self.patches = cached_patches
                    return

            def _process_segment(seg):
                try:
                    seg._find_patches()
                except Exception as exc:
                    segment_id = (
                        getattr(seg, "segment_relpath", None)
                        or getattr(seg, "segment_name", None)
                        or getattr(seg, "segment_dir", None)
                    )
                    raise RuntimeError(
                        "Failed finding patches for "
                        f"dataset_idx={getattr(seg, 'dataset_idx', None)!r}, "
                        f"segment={segment_id!r}, "
                        f"image_volume={getattr(seg, 'image_volume', None)!r}, "
                        f"volume_scale={getattr(seg, 'scale', None)!r}, "
                        f"supervision_mask={getattr(seg, 'supervision_mask', None)!r}, "
                        f"inklabels={getattr(seg, 'inklabels', None)!r}, "
                        f"validation_mask={getattr(seg, 'validation_mask', None)!r}"
                    ) from exc
                return seg.training_patches, seg.validation_patches

            self.patches = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                for training_patches, validation_patches in tqdm(pool.map(_process_segment, segments), total=len(segments), desc='Finding patches'):
                    self.training_patches.extend(training_patches)
                    self.validation_patches.extend(validation_patches)
            self.unlabeled_patches = [patch for patch in self.training_patches if getattr(patch, 'is_unlabeled', False)]
            self.patches = self.training_patches + self.validation_patches
            save_flat_patch_cache(cache_path, self.patches)
        else:
            self.patches = list(patches)
            self.segments = (
                list(segments)
                if segments is not None
                else [patch.segment for patch in self.patches] + list(self.unlabeled_segments)
            )
            self.training_patches = [
                patch for patch in self.patches
                if not getattr(patch, 'is_validation', False)
            ]
            self.validation_patches = [patch for patch in self.patches if getattr(patch, 'is_validation', False)]
            self.unlabeled_patches = [patch for patch in self.patches if getattr(patch, 'is_unlabeled', False)]
            if self.unlabeled_segments:
                self._num_virtual_unlabeled = max(len(self.patches), 1000)
            self._register_segments(self.segments)

    @staticmethod
    def _native_volume_key(segment):
        dataset_idx = getattr(segment, "dataset_idx", None)
        return (
            None if dataset_idx is None else int(dataset_idx),
            str(getattr(segment, "image_volume", "")),
            str(getattr(segment, "scale", "")),
        )

    @staticmethod
    def _segment_identity_key(segment):
        dataset_idx = getattr(segment, "dataset_idx", None)
        return (
            None if dataset_idx is None else int(dataset_idx),
            str(getattr(segment, "segment_relpath", None) or getattr(segment, "segment_dir", None) or getattr(segment, "segment_name", "")),
        )

    def _full_3d_projection_half_thicknesses(self):
        full_3d_config = (getattr(self, "config", None) or {}).get('full_3d') or {}
        default_thickness = float(
            full_3d_config.get(
                'projection_half_thickness',
                _DEFAULT_FULL_3D_PROJECTION_HALF_THICKNESS,
            )
        )
        label_thickness = float(
            full_3d_config.get('label_projection_half_thickness', default_thickness)
        )
        background_thickness = float(
            full_3d_config.get(
                'background_projection_half_thickness',
                full_3d_config.get('supervision_projection_half_thickness', default_thickness),
            )
        )
        if label_thickness < 0.0 or background_thickness < 0.0:
            raise ValueError(
                "full_3d projection half-thickness values must be >= 0, "
                f"got label={label_thickness!r}, background={background_thickness!r}"
            )
        return label_thickness, background_thickness

    def _register_segments(self, segments):
        self._segments_by_native_volume_key = {}
        seen = set()
        for segment in segments:
            if segment is None:
                continue
            volume_key = self._native_volume_key(segment)
            identity_key = self._segment_identity_key(segment)
            dedupe_key = (volume_key, identity_key)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            self._segments_by_native_volume_key.setdefault(volume_key, []).append(segment)

    def _iter_other_native_segments(self, patch):
        base_key = self._segment_identity_key(patch.segment)
        for segment in self._segments_by_native_volume_key.get(self._native_volume_key(patch.segment), ()):
            if self._segment_identity_key(segment) != base_key:
                yield segment

    def _get_cached_zarr(self, path, *, resolution):
        cache_key = (str(path), str(resolution), str(self.vol_auth))
        volume = self._zarr_cache.get(cache_key)
        if volume is None:
            volume = open_zarr(path, resolution=resolution, auth=self.vol_auth)
            self._zarr_cache[cache_key] = volume
        return volume

    def _get_cached_tifxyz(self, segment_dir):
        cache_key = str(segment_dir)
        patch_tifxyz = self._tifxyz_cache.get(cache_key)
        if patch_tifxyz is None:
            patch_tifxyz = tifxyz.read_tifxyz(segment_dir)
            patch_tifxyz.use_full_resolution()
            self._tifxyz_cache[cache_key] = patch_tifxyz
        return patch_tifxyz

    def _get_cached_stored_resolution_zyxs(self, segment_dir, *, patch_tifxyz=None):
        cache_key = str(segment_dir)
        cached = self._stored_resolution_zyx_cache.get(cache_key)
        if cached is None:
            if patch_tifxyz is None:
                patch_tifxyz = self._get_cached_tifxyz(segment_dir)
            coarse_patch_zyxs = np.asarray(
                patch_tifxyz.get_zyxs(stored_resolution=True),
                dtype=np.float32,
            )
            coarse_valid = np.isfinite(coarse_patch_zyxs).all(axis=-1)
            coarse_valid &= (coarse_patch_zyxs >= 0).all(axis=-1)
            cached = (coarse_patch_zyxs, coarse_valid)
            self._stored_resolution_zyx_cache[cache_key] = cached
        return cached

    def _gather_segments(self):
        for dataset_idx, ds in enumerate(self.datasets):

            seg_path = Path(ds['segments_path'])

            for tifxyz_folder in sorted(seg_path.iterdir()):
                if not tifxyz_folder.is_dir() or tifxyz_folder.name == 'unused':
                    continue
                if not any(tifxyz_folder.rglob('x.tif')):
                    continue

                if _is_native_3d_mode(self.mode):
                    image_volume = _coerce_volume_path(ds['volume_path'])
                else:
                    image_volume = Path(str(tifxyz_folder) + "/" + tifxyz_folder.name + '.zarr')

                segment = Segment(
                    config=self.config,
                    image_volume=image_volume,
                    scale=ds['volume_scale'],
                    dataset_idx=dataset_idx,
                    segment_relpath=tifxyz_folder.relative_to(seg_path).as_posix(),
                    segment_dir=tifxyz_folder,
                    segment_name=tifxyz_folder.name,
                )
                if self.discovery_mode == 'unlabeled':
                    inklabels, supervision_mask, validation_mask = segment.discover_labels(
                        extension='.zarr',
                        required=False,
                    )
                else:
                    inklabels, supervision_mask, validation_mask = segment.discover_labels(extension='.zarr')

                if self.debug:
                    print(image_volume)
                    print(supervision_mask)
                    print(inklabels)
                    print(validation_mask)

                if not _is_remote_volume_path(image_volume) and not image_volume.exists():
                    if self.discovery_mode == 'unlabeled':
                        print(f"Skipping {tifxyz_folder.name}: missing image volume {image_volume}")
                        continue
                    raise ValueError(f"{tifxyz_folder.name} is missing its image volume")
                if self.discovery_mode != 'unlabeled' and not (
                    supervision_mask is not None
                    and inklabels is not None
                    and supervision_mask.exists()
                    and inklabels.exists()
                ):
                    raise ValueError(f"{tifxyz_folder.name} is missing required data. make sure the image volume, supervision mask, and labels exist")

                yield segment

    def get_labeled_unlabeled_patch_indices(self):
        labeled_indices = []
        unlabeled_indices = []
        for patch_idx, patch in enumerate(self.patches):
            if getattr(patch, 'is_validation', False):
                continue
            if getattr(patch, 'is_unlabeled', False):
                unlabeled_indices.append(int(patch_idx))
            else:
                labeled_indices.append(int(patch_idx))
        unlabeled_indices.extend(range(len(self.patches), len(self.patches) + self._num_virtual_unlabeled))
        return labeled_indices, unlabeled_indices

    def __len__(self):
        return len(self.patches) + self._num_virtual_unlabeled

    def _choose_replacement_patch_index(self, *, current_idx):
        if len(self.patches) <= 1:
            raise RuntimeError("Cannot resample an oversized normal pooled patch from a dataset with <= 1 patch")
        seed = int(self.config.get('seed', 0))
        rng = random.Random(seed + (int(current_idx) * 7919))
        while True:
            replacement_idx = rng.randrange(len(self.patches))
            if replacement_idx != int(current_idx):
                return replacement_idx

    def _generate_random_unlabeled_sample(self):
        expected_shape = tuple(int(v) for v in self.patch_size)
        patch_depth, patch_height, patch_width = expected_shape

        for _ in range(50):
            segment = random.choice(self.unlabeled_segments)
            image_vol = self._get_cached_zarr(segment.image_volume, resolution=segment.scale)
            surface = int(image_vol.shape[0] // 2)
            max_y = max(0, int(image_vol.shape[1]) - patch_height)
            max_x = max(0, int(image_vol.shape[2]) - patch_width)
            y0 = random.randint(0, max_y) if max_y > 0 else 0
            x0 = random.randint(0, max_x) if max_x > 0 else 0
            z0 = surface - patch_depth // 2
            bbox = (z0, y0, x0, z0 + patch_depth, y0 + patch_height, x0 + patch_width)

            image_crop, image_valid_slices = _read_bbox_with_padding(image_vol, bbox, fill_value=0)
            if image_crop.size > 0 and np.count_nonzero(image_crop) / image_crop.size >= 0.25:
                break

        image_crop = image_crop.astype(np.float32, copy=False)
        if image_valid_slices is not None:
            image_crop[image_valid_slices] = _normalize_image_crop(
                image_crop[image_valid_slices],
                self.config,
            )

        image_crop = torch.from_numpy(image_crop).float().unsqueeze(0)
        supervision_crop = torch.zeros(1, *expected_shape, dtype=torch.float32)
        inklabels_crop = torch.zeros(1, *expected_shape, dtype=torch.float32)

        data = {
            'image': image_crop,
            'inklabels': inklabels_crop,
            'supervision_mask': supervision_crop,
        }

        if self.do_augmentations and self.augmentations is not None:
            result = self.augmentations(**data)
        else:
            result = data

        result['is_unlabeled'] = torch.tensor(True, dtype=torch.bool)
        return result

    def _merge_intersecting_segment_labels_into_crop(
        self,
        *,
        patch,
        crop_bbox,
        inklabels_crop,
        supervision_crop,
    ):
        label_projection_half_thickness, background_projection_half_thickness = (
            self._full_3d_projection_half_thicknesses()
        )
        for segment in self._iter_other_native_segments(patch):
            if segment.segment_dir is None or segment.inklabels is None or segment.supervision_mask is None:
                continue

            patch_tifxyz = self._get_cached_tifxyz(segment.segment_dir)
            coarse_patch_zyxs, coarse_valid = self._get_cached_stored_resolution_zyxs(
                segment.segment_dir,
                patch_tifxyz=patch_tifxyz,
            )
            support_selection = _maybe_select_flat_pixels_for_native_crop_via_stored_resolution(
                patch_tifxyz,
                crop_bbox,
                coarse_patch_zyxs=coarse_patch_zyxs,
                coarse_valid=coarse_valid,
            )
            if support_selection is None:
                continue
            support_bbox, support_patch_zyxs, support_valid = support_selection

            support_y0, support_y1, support_x0, support_x1 = support_bbox
            active_supervision_path = (
                segment.validation_mask
                if patch.is_validation and segment.validation_mask is not None
                else segment.supervision_mask
            )
            if active_supervision_path is None:
                continue

            other_supervision = self._get_cached_zarr(active_supervision_path, resolution=segment.scale)
            support_supervision_flat_patch = _read_flat_surface_patch(
                other_supervision,
                y0=support_y0,
                y1=support_y1,
                x0=support_x0,
                x1=support_x1,
            )
            if (not patch.is_validation) and segment.validation_mask is not None:
                other_validation = self._get_cached_zarr(segment.validation_mask, resolution=segment.scale)
                support_validation_flat_patch = _read_flat_surface_patch(
                    other_validation,
                    y0=support_y0,
                    y1=support_y1,
                    x0=support_x0,
                    x1=support_x1,
                )
                support_supervision_flat_patch = _exclude_validation_voxels_from_training_supervision(
                    support_supervision_flat_patch,
                    support_validation_flat_patch,
                    is_validation_patch=patch.is_validation,
                )

            supervised_support_valid = (
                np.asarray(support_valid, dtype=bool)
                & (np.asarray(support_supervision_flat_patch) > 0)
            )
            if not np.any(supervised_support_valid):
                continue

            support_normals_local_zyx = _support_normals_local_zyx(
                patch_tifxyz=patch_tifxyz,
                support_bbox=support_bbox,
            )
            other_inklabels = self._get_cached_zarr(segment.inklabels, resolution=segment.scale)
            support_inklabels_flat_patch = _read_flat_surface_patch(
                other_inklabels,
                y0=support_y0,
                y1=support_y1,
                x0=support_x0,
                x1=support_x1,
            )
            other_inklabels_crop, other_supervision_crop = _project_flat_labels_and_supervision_to_native_crop(
                support_patch_zyxs=support_patch_zyxs,
                support_valid=supervised_support_valid,
                support_inklabels_flat_patch=support_inklabels_flat_patch,
                support_supervision_flat_patch=support_supervision_flat_patch,
                crop_bbox=crop_bbox,
                support_normals_local_zyx=support_normals_local_zyx,
                label_projection_half_thickness=label_projection_half_thickness,
                background_projection_half_thickness=background_projection_half_thickness,
            )
            np.maximum(inklabels_crop, other_inklabels_crop, out=inklabels_crop)
            np.maximum(supervision_crop, other_supervision_crop, out=supervision_crop)

        return inklabels_crop, supervision_crop

    def __getitem__(self, idx):
        requested_idx = int(idx)

        if requested_idx >= len(self.patches) and self.unlabeled_segments:
            return self._generate_random_unlabeled_sample()

        current_idx = requested_idx

        while True:
            patch = self.patches[current_idx]
            z0, y0, x0, z1, y1, x1 = patch.bbox
            expected_shape = tuple(int(v) for v in self.patch_size)
            crop_bbox = patch.bbox
            use_surface_mask = _uses_surface_mask_channel(self.mode)
            surface_mask = None
            normal_pooled_metadata = None
            inklabels_crop = None
            supervision_crop = None
            support_normals_local_zyx = None
            resample_idx = None
            resample_warning_message = None
            
            # this entire if block only applies if you're using a "3d" mode. it samples the supervision in 2d 'flat' space,
            # and extracts the crop using the same patch finding as the 2d patch finding code. the result of this is that sometimes this patch
            # does not occupy the full 3d crop (or more than the full 3d crop). we handle this by either padding (adding adjacent quads until we reach crop size)
            # or by cropping. the supervision mask is built by first doing a 3d connected components on the surface voxels, and then filtering once again to the 2d
            # connected components "in crop". the first may be unnecessary.
            # Modes with a surface-mask channel use an EDT clipped to a distance of 10.
            if _is_native_3d_mode(self.mode):
                image_vol = self._get_cached_zarr(patch.image_volume, resolution=patch.segment.scale)
                supervision_mask = self._get_cached_zarr(patch.supervision_mask, resolution=patch.segment.scale)
                inklabels = self._get_cached_zarr(patch.inklabels, resolution=patch.segment.scale)
                validation_mask = None
                if (not patch.is_validation) and patch.segment.validation_mask is not None:
                    validation_mask = self._get_cached_zarr(
                        patch.segment.validation_mask,
                        resolution=patch.segment.scale,
                    )
                patch_tifxyz = self._get_cached_tifxyz(patch.segment_dir)
                coarse_patch_zyxs, coarse_valid = self._get_cached_stored_resolution_zyxs(patch.segment_dir,patch_tifxyz=patch_tifxyz)

                flat_x, flat_y, flat_z, flat_valid = patch_tifxyz[y0:y1, x0:x1]
                patch_zyxs = np.stack([flat_z, flat_y, flat_x], axis=-1)
                try:
                    crop_bbox = compute_native_crop_bbox_from_patch_points(patch_zyxs,flat_valid,expected_shape)
                except ValueError as exc:
                    if str(exc) != "No valid tifxyz points found for patch":
                        raise
                    resample_idx = self._choose_replacement_patch_index(current_idx=current_idx)
                    resample_warning_message = (
                        f"Normal pooled patch had no valid tifxyz points "
                        f"for requested idx {requested_idx}, patch idx {current_idx}, "
                        f"segment {patch.segment.segment_name}; resampling idx {resample_idx}"
                    )
                if resample_idx is None:
                    supervision_flat_patch = _read_flat_surface_patch(supervision_mask,y0=y0,y1=y1,x0=x0,x1=x1)
                    
                    if validation_mask is not None:
                        validation_flat_patch = _read_flat_surface_patch(validation_mask,y0=y0,y1=y1,x0=x0,x1=x1)
                        supervision_flat_patch = _exclude_validation_voxels_from_training_supervision(
                            supervision_flat_patch,
                            validation_flat_patch,
                            is_validation_patch=patch.is_validation,
                        )
                        
                    if self.do_augmentations:
                        crop_bbox = maybe_translate_normal_pooled_crop_bbox(crop_bbox,patch_zyxs,flat_valid,supervision_flat_patch)
                    
                    (
                        base_support_bbox,
                        support_patch_zyxs,
                        support_valid,
                        support_patch_zyxs_halo,
                        support_valid_halo,
                        trim_slices,
                    ) = _select_flat_pixels_for_native_crop_via_stored_resolution(
                        patch_tifxyz,
                        crop_bbox,
                        coarse_patch_zyxs=coarse_patch_zyxs,
                        coarse_valid=coarse_valid,
                        return_halo=True,
                    )
                    support_y0, support_y1, support_x0, support_x1 = base_support_bbox
                    support_supervision_flat_patch = _read_flat_surface_patch(supervision_mask,y0=support_y0,y1=support_y1,x0=support_x0,x1=support_x1)
                    
                    if validation_mask is not None:
                        support_validation_flat_patch = _read_flat_surface_patch(validation_mask,y0=support_y0,y1=support_y1,x0=support_x0,x1=support_x1)
                        support_supervision_flat_patch = _exclude_validation_voxels_from_training_supervision(
                            support_supervision_flat_patch,
                            support_validation_flat_patch,
                            is_validation_patch=patch.is_validation,
                        )
                        
                    support_inklabels_flat_patch = _read_flat_surface_patch(inklabels,y0=support_y0,y1=support_y1,x0=support_x0,x1=support_x1)
                    pooling_config = self.config.get('normal_pooling') or {}
                    max_support_grid_distance = pooling_config.get('support_grid_max_distance', 64.0)
                    (
                        label_projection_half_thickness,
                        background_projection_half_thickness,
                    ) = self._full_3d_projection_half_thicknesses()
                    (
                        (support_y0, support_y1, support_x0, support_x1),
                        support_patch_zyxs,
                        support_valid,
                        support_inklabels_flat_patch,
                        support_supervision_flat_patch,
                    ) = _filter_support_components_by_active_supervision(
                        support_bbox=(support_y0, support_y1, support_x0, support_x1),
                        support_patch_zyxs=support_patch_zyxs,
                        support_valid=support_valid,
                        support_inklabels_flat_patch=support_inklabels_flat_patch,
                        support_supervision_flat_patch=support_supervision_flat_patch,
                        crop_bbox=crop_bbox,
                        patch_bbox=patch.bbox,
                        max_supervision_grid_distance=max_support_grid_distance,
                    )
                    (
                        support_patch_zyxs_halo,
                        support_valid_halo,
                        trim_slices,
                    ) = _slice_support_halo_for_subwindow(
                        support_patch_zyxs_halo,
                        support_valid_halo,
                        trim_slices,
                        base_support_bbox,
                        (support_y0, support_y1, support_x0, support_x1),
                    )
                    if self.mode != "normal_pooled_3d":
                        support_normals_local_zyx = _support_normals_local_zyx(
                            patch_tifxyz=patch_tifxyz,
                            support_bbox=(support_y0, support_y1, support_x0, support_x1),
                            support_patch_zyxs_halo=support_patch_zyxs_halo,
                            support_valid_halo=support_valid_halo,
                            trim_slices=trim_slices,
                        )

                    support_grid_shape = tuple(int(v) for v in support_valid.shape)
                    support_grid_side_limits = (int(expected_shape[1] * 4),int(expected_shape[2] * 4))
                    if support_grid_shape[0] > support_grid_side_limits[0] or support_grid_shape[1] > support_grid_side_limits[1]:
                        resample_idx = self._choose_replacement_patch_index(current_idx=current_idx)
                        resample_warning_message = (
                            f"Oversized normal pooled support grid {support_grid_shape!r} "
                            f"exceeded side limits {support_grid_side_limits!r} "
                            f"for requested idx {requested_idx}, patch idx {current_idx}, "
                            f"segment {patch.segment.segment_name}; resampling idx {resample_idx}"
                        )
                    else:
                        image_crop, image_valid_slices = _read_bbox_with_padding(image_vol, crop_bbox, fill_value=0)
                        if use_surface_mask:
                            surface_mask = _project_valid_surface_mask_to_native_crop(
                                support_patch_zyxs,
                                support_valid,
                                crop_bbox,
                            )
                        if self.mode == "normal_pooled_3d":
                            normal_pooled_metadata = _build_normal_pooled_flat_metadata(
                                support_patch_zyxs=support_patch_zyxs,
                                support_valid=support_valid,
                                support_patch_zyxs_halo=support_patch_zyxs_halo,
                                support_valid_halo=support_valid_halo,
                                trim_slices=trim_slices,
                                support_inklabels_flat_patch=support_inklabels_flat_patch,
                                support_supervision_flat_patch=support_supervision_flat_patch,
                                crop_bbox=crop_bbox,
                            )
                        else:
                            inklabels_crop, supervision_crop = _project_flat_labels_and_supervision_to_native_crop(
                                support_patch_zyxs=support_patch_zyxs,
                                support_valid=support_valid,
                                support_inklabels_flat_patch=support_inklabels_flat_patch,
                                support_supervision_flat_patch=support_supervision_flat_patch,
                                crop_bbox=crop_bbox,
                                support_normals_local_zyx=support_normals_local_zyx,
                                label_projection_half_thickness=label_projection_half_thickness,
                                background_projection_half_thickness=background_projection_half_thickness,
                            )
                            if _includes_intersecting_segments_3d(self.mode):
                                inklabels_crop, supervision_crop = self._merge_intersecting_segment_labels_into_crop(
                                    patch=patch,
                                    crop_bbox=crop_bbox,
                                    inklabels_crop=inklabels_crop,
                                    supervision_crop=supervision_crop,
                                )

            # for pooled 2d, this is the only block that applies (outside of potential resampling)
            else:
                image_vol = self._get_cached_zarr(patch.image_volume, resolution=patch.segment.scale)
                image_crop, image_valid_slices = _read_bbox_with_padding(image_vol, patch.bbox, fill_value=0)
                if getattr(patch, 'is_unlabeled', False):
                    supervision_crop = np.zeros(expected_shape, dtype=np.uint8)
                    inklabels_crop = np.zeros(expected_shape, dtype=np.uint8)
                else:
                    supervision_mask = self._get_cached_zarr(patch.supervision_mask, resolution=patch.segment.scale)
                    inklabels = self._get_cached_zarr(patch.inklabels, resolution=patch.segment.scale)
                    validation_mask = None
                    if (not patch.is_validation) and patch.segment.validation_mask is not None:
                        validation_mask = self._get_cached_zarr(patch.segment.validation_mask,resolution=patch.segment.scale)
                        
                    supervision_crop, _ = _read_bbox_with_padding(supervision_mask, patch.bbox, fill_value=0)
                    if validation_mask is not None:
                        validation_crop, _ = _read_bbox_with_padding(validation_mask, patch.bbox, fill_value=0)
                        supervision_crop = _exclude_validation_voxels_from_training_supervision(
                            supervision_crop,
                            validation_crop,
                            is_validation_patch=patch.is_validation,
                        )
                    inklabels_crop, _ = _read_bbox_with_padding(inklabels, patch.bbox, fill_value=0)

            if resample_idx is None:
                image_crop = image_crop.astype(np.float32, copy=False)

                # Capture an input mask BEFORE normalization, in raw-image units
                # (e.g. uint8 0-255). Used by the dynamic-label generator to
                # zero out UNet predictions over background voxels.
                image_mask_for_label_np = None
                if self._emit_image_for_label:
                    mask_threshold = float(
                        (self.config.get('dynamic_label') or {}).get('input_mask_threshold', 50.0)
                    )
                    image_mask_for_label_np = (image_crop > mask_threshold).astype(np.float32)

                if image_valid_slices is not None:
                    image_crop[image_valid_slices] = _normalize_image_crop(
                        image_crop[image_valid_slices],
                        self.config,
                    )

                arrays_to_validate = [("image", image_crop)]
                if self.mode != "normal_pooled_3d":
                    arrays_to_validate.extend(
                        [
                            ("supervision_mask", supervision_crop),
                            ("inklabels", inklabels_crop),
                        ]
                    )
                for name, array in arrays_to_validate:
                    if tuple(int(v) for v in array.shape) != expected_shape:
                        raise AssertionError(
                            f"{name} crop shape {tuple(int(v) for v in array.shape)} does not match "
                            f"requested patch size {expected_shape} for bbox {crop_bbox!r}"
                        )

                image_crop = torch.from_numpy(image_crop).float().unsqueeze(0)
                data = {'image': image_crop}

                if self.mode == "normal_pooled_3d":
                    assert normal_pooled_metadata is not None
                    data.update(normal_pooled_metadata)
                else:
                    assert inklabels_crop is not None
                    assert supervision_crop is not None
                    inklabels_crop = torch.from_numpy(inklabels_crop).float().unsqueeze(0)
                    supervision_crop = torch.from_numpy(supervision_crop).float().unsqueeze(0)
                    data.update({
                        'inklabels': inklabels_crop,
                        'supervision_mask': supervision_crop,
                    })

                if use_surface_mask and surface_mask is not None:
                    data['surface_mask'] = torch.from_numpy(surface_mask).float().unsqueeze(0)

                if image_mask_for_label_np is not None:
                    data['image_mask_for_label'] = torch.from_numpy(image_mask_for_label_np).float().unsqueeze(0)

                if self.do_augmentations and self.augmentations is not None:
                    if self.mode == "normal_pooled_3d":
                        augmentation_data, flat_valid_mask = _pack_normal_pooled_augmentation_data(data)
                        augmented = self.augmentations(**augmentation_data)
                        result = _restore_normal_pooled_augmentation_data(augmented, data, flat_valid_mask)
                    else:
                        augmentation_data = data
                        if _is_full_3d_like_mode(self.mode) and 'surface_mask' in data:
                            augmentation_data = dict(data)
                            augmentation_data['regression_keys'] = ['surface_mask']
                        if self._emit_image_for_label:
                            after_geo = self._geometric_augmentations(**augmentation_data)
                            image_for_label = after_geo['image'].clone()
                            mask_for_label = after_geo.pop('image_mask_for_label', None)
                            result = self._photometric_augmentations(**after_geo)
                            result['image_for_label'] = image_for_label
                            if mask_for_label is not None:
                                # Re-binarize after geometric augs in case any
                                # interpolation produced fractional values.
                                result['image_mask_for_label'] = (mask_for_label > 0.5).float()
                        else:
                            result = self.augmentations(**augmentation_data)
                else:
                    result = data
                    if image_mask_for_label_np is not None:
                        result['image_mask_for_label'] = (data['image_mask_for_label'] > 0.5).float()

            if resample_idx is not None:
                warnings.warn(
                    resample_warning_message,
                    RuntimeWarning,
                    stacklevel=2,
                )
                current_idx = resample_idx
                continue

            if isinstance(result, dict):
                result['is_unlabeled'] = torch.tensor(bool(getattr(patch, 'is_unlabeled', False)), dtype=torch.bool)

            return result

if __name__ == "__main__":
    import argparse
    from qtpy.QtWidgets import QPushButton

    parser = argparse.ArgumentParser(description="Visualize an InkDataset sample in napari.")
    parser.add_argument("config_path", help="Path to the dataset config JSON.")
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ds = InkDataset(config, do_augmentations=False)

    import napari

    viewer = napari.Viewer()

    state = {"current_index": 0}
    layers = {
        "image": None,
        "target": None,
        "supervision": None,
        "surface_mask": None,
    }

    def load_sample(index):
        data = ds[index]
        print(f"\nSample {index}")
        for k, v in data.items():
            print(f'{k:20s} shape={str(list(v.shape)):20s} dtype={str(v.dtype):15s} min={v.min().item():.4f}  max={v.max().item():.4f}')
        return data

    def render_sample(index):
        data = load_sample(index)
        image = data['image'][0].numpy()
        target_key = 'inklabels' if 'inklabels' in data else 'flat_target'
        supervision_key = 'supervision_mask' if 'supervision_mask' in data else 'flat_supervision'
        target = data[target_key].squeeze(0).numpy().astype(int)
        supervision = data[supervision_key].squeeze(0).numpy().astype(int)

        if layers["image"] is None:
            layers["image"] = viewer.add_image(image, name='image')
            layers["target"] = viewer.add_labels(target, name=target_key)
            layers["supervision"] = viewer.add_labels(supervision, name=supervision_key)
        else:
            layers["image"].data = image
            layers["target"].data = target
            layers["target"].name = target_key
            layers["supervision"].data = supervision
            layers["supervision"].name = supervision_key

        surface_mask = data.get('surface_mask')
        if surface_mask is not None:
            surface_mask_data = surface_mask.squeeze(0).numpy()
            if layers["surface_mask"] is None:
                layers["surface_mask"] = viewer.add_image(
                    surface_mask_data,
                    name='surface_mask',
                    contrast_limits=(0.0, 1.0),
                    colormap='cyan',
                    opacity=0.5,
                )
            else:
                layers["surface_mask"].data = surface_mask_data
        elif layers["surface_mask"] is not None:
            viewer.layers.remove(layers["surface_mask"])
            layers["surface_mask"] = None

        viewer.title = f"InkDataset sample {index}"

    def show_next_sample():
        if len(ds) <= 1:
            render_sample(state["current_index"])
            return

        next_index = state["current_index"]
        while next_index == state["current_index"]:
            next_index = random.randrange(len(ds))
        state["current_index"] = next_index
        render_sample(state["current_index"])

    next_button = QPushButton("Next")
    next_button.clicked.connect(show_next_sample)
    viewer.window.add_dock_widget(next_button, area="right", name="Sample Controls")

    render_sample(state["current_index"])

    napari.run()
