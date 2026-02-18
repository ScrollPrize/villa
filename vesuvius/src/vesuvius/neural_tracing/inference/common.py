import json
import os
import colorsys
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zarr

import vesuvius.tifxyz as tifxyz
from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid
from vesuvius.neural_tracing.datasets.extrapolation import _EXTRAPOLATION_METHODS, apply_degradation
from vesuvius.neural_tracing.inference.extrap_lookup import ExtrapLookupArrays

from vesuvius.neural_tracing.tifxyz import save_tifxyz


def _resolve_extrapolation_settings(args, runtime_config):
    cfg = runtime_config or {}

    method = args.extrapolation_method
    if method is None:
        method = str(cfg.get("extrapolation_method", "rbf"))

    degrade_prob = float(cfg.get("extrap_degrade_prob", 0.0))

    def _pair_from_cfg(key, default_pair):
        val = cfg.get(key, default_pair)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return (float(val[0]), float(val[1]))
        return (float(default_pair[0]), float(default_pair[1]))

    degrade_curvature_range = _pair_from_cfg("extrap_degrade_curvature_range", (0.001, 0.01))
    degrade_gradient_range = _pair_from_cfg("extrap_degrade_gradient_range", (0.05, 0.2))

    method_kwargs = {}
    if method in {"rbf", "rbf_edge_only"}:
        rbf_downsample_override = getattr(args, "rbf_downsample_factor", None)
        rbf_max_points_override = getattr(args, "rbf_max_points", None)
        rbf_downsample = int(
            rbf_downsample_override
            if rbf_downsample_override is not None
            else cfg.get("rbf_downsample_factor", 2)
        )
        edge_downsample_cfg = cfg.get("rbf_edge_downsample_factor", None)
        edge_downsample = rbf_downsample if edge_downsample_cfg is None else int(edge_downsample_cfg)
        method_kwargs["downsample_factor"] = edge_downsample if method == "rbf_edge_only" else rbf_downsample
        method_kwargs["rbf_max_points"] = (
            int(rbf_max_points_override)
            if rbf_max_points_override is not None
            else cfg.get("rbf_max_points")
        )
        # MUST RUN IN FLOAT64: RBF extrapolation in this pipeline is unstable at lower precision.
        method_kwargs["precision"] = cfg.get("rbf_precision", "float64")
        rbf_uv_domain = str(cfg.get("rbf_uv_domain", "full")).strip().lower()
        if rbf_uv_domain not in {"full", "stored_lattice"}:
            raise ValueError(
                "Unsupported rbf_uv_domain value. "
                "Expected one of: ['full', 'stored_lattice']"
            )
        phase_rc_cfg = cfg.get("rbf_uv_phase_rc", (0, 0))
        if not isinstance(phase_rc_cfg, (list, tuple)) or len(phase_rc_cfg) != 2:
            raise ValueError("rbf_uv_phase_rc must be a 2-element list/tuple.")
        method_kwargs["rbf_uv_domain"] = rbf_uv_domain
        method_kwargs["rbf_uv_phase_rc"] = (
            int(phase_rc_cfg[0]),
            int(phase_rc_cfg[1]),
        )
        rbf_scale = getattr(args, "rbf_scale", None)
        if rbf_scale is not None:
            preset = str(rbf_scale).strip().lower()
            if preset == "stored":
                # Stored-scale preset: run RBF in stored UV lattice and avoid
                # additional control-point downsampling on top of lattice scaling.
                method_kwargs["rbf_uv_domain"] = "stored_lattice"
                method_kwargs["precision"] = "float32"
                method_kwargs["downsample_factor"] = 1
            elif preset == "full":
                method_kwargs["rbf_uv_domain"] = "full"
                method_kwargs["precision"] = "float64"

    if method == "rbf_edge_only":
        method_kwargs["edge_band_frac"] = float(cfg.get("rbf_edge_band_frac", 0.10))
        method_kwargs["edge_band_cells"] = cfg.get("rbf_edge_band_cells")
        method_kwargs["edge_min_points"] = int(cfg.get("rbf_edge_min_points", 128))

    # Keep args aligned with resolved settings for existing debug prints/metadata.
    args.extrapolation_method = method

    return {
        "method": method,
        "degrade_prob": float(degrade_prob),
        "degrade_curvature_range": degrade_curvature_range,
        "degrade_gradient_range": degrade_gradient_range,
        "method_kwargs": method_kwargs,
    }


def resolve_tifxyz_params(args, model_config, volume_scale):
    tifxyz_step_size = args.tifxyz_step_size
    tifxyz_voxel_size_um = args.tifxyz_voxel_size_um

    if tifxyz_step_size is None:
        if model_config is not None:
            tifxyz_step_size = model_config.get(
                "step_size",
                model_config.get("heatmap_step_size", 10),
            )
        else:
            tifxyz_step_size = 10
    if tifxyz_voxel_size_um is None:
        meta_path = os.path.join(args.volume_path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "rt") as meta_fp:
                tifxyz_voxel_size_um = json.load(meta_fp).get("voxelsize", None)
    if tifxyz_voxel_size_um is None:
        tifxyz_voxel_size_um = 8.24

    # Keep output tifxyz spacing in stored-resolution units.
    # The old behavior multiplied by 2**volume_scale, which inflated output
    # spacing as inference scale increased.
    tifxyz_step_size = int(round(float(tifxyz_step_size)))
    return tifxyz_step_size, tifxyz_voxel_size_um


def _validate_named_method(merge_method, valid_methods, method_label):
    method = str(merge_method).strip().lower()
    if method not in valid_methods:
        raise ValueError(
            f"Unknown {method_label} '{merge_method}'. "
            f"Supported methods: {list(valid_methods)}"
        )
    return method


def _aggregate_pred_samples_to_uv_grid(pred_samples, base_uv_bounds=None, overlap_merge_method="mean"):
    """Merge list of (uv, world_pts) into a dense HxWx3 grid in UV space."""
    overlap_merge_method = _validate_named_method(
        overlap_merge_method, ("mean",), "overlap merge method"
    )

    non_empty_samples = []
    for uv, pts in pred_samples:
        uv_arr = np.asarray(uv)
        pts_arr = np.asarray(pts)
        if uv_arr.size == 0 or pts_arr.size == 0:
            continue
        non_empty_samples.append((uv_arr, pts_arr))

    pred_bounds = None
    if non_empty_samples:
        all_uv = np.concatenate([uv for uv, _ in non_empty_samples], axis=0)
        pred_bounds = (
            int(all_uv[:, 0].min()),
            int(all_uv[:, 1].min()),
            int(all_uv[:, 0].max()),
            int(all_uv[:, 1].max()),
        )

    if base_uv_bounds is None and pred_bounds is None:
        return np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0, 0), dtype=bool), (0, 0)

    if base_uv_bounds is None:
        uv_r_min, uv_c_min, uv_r_max, uv_c_max = pred_bounds
    else:
        uv_r_min, uv_c_min, uv_r_max, uv_c_max = (int(v) for v in base_uv_bounds)
        if pred_bounds is not None:
            uv_r_min = min(uv_r_min, pred_bounds[0])
            uv_c_min = min(uv_c_min, pred_bounds[1])
            uv_r_max = max(uv_r_max, pred_bounds[2])
            uv_c_max = max(uv_c_max, pred_bounds[3])

    h = uv_r_max - uv_r_min + 1
    w = uv_c_max - uv_c_min + 1
    if not non_empty_samples:
        return np.full((h, w, 3), -1.0, dtype=np.float32), np.zeros((h, w), dtype=bool), (uv_r_min, uv_c_min)

    grid_acc = np.zeros((h, w, 3), dtype=np.float64)
    grid_count = np.zeros((h, w), dtype=np.int32)

    for uv, pts in non_empty_samples:
        rows = uv[:, 0].astype(np.int64) - uv_r_min
        cols = uv[:, 1].astype(np.int64) - uv_c_min
        np.add.at(grid_acc, (rows, cols), pts.astype(np.float64))
        np.add.at(grid_count, (rows, cols), 1)

    grid_valid = grid_count > 0
    grid_zyxs = np.full((h, w, 3), -1.0, dtype=np.float32)
    grid_zyxs[grid_valid] = (grid_acc[grid_valid] / grid_count[grid_valid, np.newaxis]).astype(np.float32)

    return grid_zyxs, grid_valid, (uv_r_min, uv_c_min)


def resolve_extrapolation_settings(args, model_config=None, load_checkpoint_config_fn=None):
    runtime_config = {}
    if model_config is None and args.checkpoint_path and load_checkpoint_config_fn is not None:
        model_config, _ = load_checkpoint_config_fn(args.checkpoint_path)
    if model_config:
        runtime_config.update(model_config)

    config_path = getattr(args, "config_path", None)
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)
        if isinstance(config_data, dict):
            runtime_config.update(config_data)

    return _resolve_extrapolation_settings(args, runtime_config)


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _serialize_args(args):
    return {str(k): _json_safe(v) for k, v in vars(args).items()}


_INT32_INFO = np.iinfo(np.int32)


def _int_dtype_for_value_range(min_value, max_value, prefer_int32=True):
    min_i = int(min_value)
    max_i = int(max_value)
    if prefer_int32 and min_i >= int(_INT32_INFO.min) and max_i <= int(_INT32_INFO.max):
        return np.int32
    return np.int64


def _coerce_int_array_for_range(values, prefer_int32=True, default_dtype=np.int32, round_values=False):
    arr = np.asarray(values)
    if arr.size == 0:
        return np.asarray(arr, dtype=default_dtype)

    if arr.dtype.kind in "iu":
        min_value = int(arr.min())
        max_value = int(arr.max())
        out_dtype = _int_dtype_for_value_range(min_value, max_value, prefer_int32=prefer_int32)
        return arr.astype(out_dtype, copy=False)

    arr_num = np.rint(arr) if round_values else np.asarray(arr)
    finite = np.isfinite(arr_num)
    if not bool(finite.all()):
        raise ValueError("Expected finite numeric values for integer coercion.")
    min_value = int(arr_num.min())
    max_value = int(arr_num.max())
    out_dtype = _int_dtype_for_value_range(min_value, max_value, prefer_int32=prefer_int32)
    return arr_num.astype(out_dtype, copy=False)


def _coerce_uv_int_array(values, prefer_int32=True, default_dtype=np.int32):
    return _coerce_int_array_for_range(
        values,
        prefer_int32=prefer_int32,
        default_dtype=default_dtype,
        round_values=True,
    )


def _flat_index_dtype_for_shape(height, width, prefer_int32=True):
    h = max(0, int(height))
    w = max(0, int(width))
    if h == 0 or w == 0:
        return np.int32 if prefer_int32 else np.int64
    max_flat = h * w - 1
    return _int_dtype_for_value_range(0, max_flat, prefer_int32=prefer_int32)


_DIRECTION_SPECS = {
    "left": {
        "axis": "col",
        "edge_idx": -1,
        "growth_sign": 1,
        "opposite": "right",
    },
    "right": {
        "axis": "col",
        "edge_idx": 0,
        "growth_sign": -1,
        "opposite": "left",
    },
    "up": {
        "axis": "row",
        "edge_idx": -1,
        "growth_sign": 1,
        "opposite": "down",
    },
    "down": {
        "axis": "row",
        "edge_idx": 0,
        "growth_sign": -1,
        "opposite": "up",
    },
}


def _get_direction_spec(direction):
    spec = _DIRECTION_SPECS.get(direction)
    if spec is None:
        raise ValueError(f"Unknown direction '{direction}'")
    return spec


def _get_growth_context(grow_direction):
    # Growth semantics are encoded by the opposite conditioning side.
    cond_direction = _get_direction_spec(grow_direction)["opposite"]
    growth_spec = _get_direction_spec(cond_direction)
    return cond_direction, growth_spec


def _in_bounds_mask(coords, size):
    size = np.asarray(size)
    return (
        (coords[:, 0] >= 0) & (coords[:, 0] < size[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < size[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < size[2])
    )


def _points_to_voxels(points_local, crop_size):
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    crop_size_dtype = _int_dtype_for_value_range(
        0,
        int(crop_size_arr.max()) if crop_size_arr.size > 0 else 0,
        prefer_int32=True,
    )
    crop_size_idx = crop_size_arr.astype(crop_size_dtype, copy=False)
    vox = np.zeros(tuple(crop_size_arr.tolist()), dtype=np.float32)
    if points_local is None or len(points_local) == 0:
        return vox
    points_local_arr = np.asarray(points_local)
    finite_pts = np.isfinite(points_local_arr).all(axis=1)
    if not bool(finite_pts.any()):
        return vox
    coords = _coerce_int_array_for_range(
        points_local_arr[finite_pts],
        prefer_int32=True,
        default_dtype=np.int32,
        round_values=True,
    )
    coords = coords[_in_bounds_mask(coords, crop_size_idx)]
    if coords.size > 0:
        vox[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return vox


def _valid_surface_mask(zyx_grid):
    return np.isfinite(zyx_grid).all(axis=-1) & ~(zyx_grid == -1).all(axis=-1)


def _get_cond_edge(cond_zyxs, cond_direction, cond_valid=None):
    spec = _get_direction_spec(cond_direction)
    edge_idx = spec["edge_idx"]
    # For non-rectangular inputs, compute a per-line frontier over valid cells
    # instead of relying on one global extreme row/col index.
    if cond_valid is not None and np.asarray(cond_valid).shape == cond_zyxs.shape[:2]:
        valid = np.asarray(cond_valid, dtype=bool)
    else:
        valid = _valid_surface_mask(cond_zyxs)

    if not valid.any():
        out_len = cond_zyxs.shape[0] if spec["axis"] == "col" else cond_zyxs.shape[1]
        return np.full((out_len, 3), -1, dtype=cond_zyxs.dtype)

    if spec["axis"] == "col":
        n_rows, n_cols = valid.shape
        out = np.full((n_rows, 3), -1, dtype=cond_zyxs.dtype)
        any_valid = valid.any(axis=1)
        row_idx = np.arange(n_rows, dtype=np.int64)
        if edge_idx == 0 or (edge_idx == -1 and n_cols == 1):
            # leftmost valid column per row (first True)
            col_indices = np.argmax(valid, axis=1)
        else:
            # rightmost valid column per row (last True)
            col_indices = n_cols - 1 - np.argmax(valid[:, ::-1], axis=1)
        out[any_valid] = cond_zyxs[row_idx[any_valid], col_indices[any_valid], :]
        return out

    n_rows, n_cols = valid.shape
    out = np.full((n_cols, 3), -1, dtype=cond_zyxs.dtype)
    any_valid = valid.any(axis=0)
    col_idx = np.arange(n_cols, dtype=np.int64)
    if edge_idx == 0 or (edge_idx == -1 and n_rows == 1):
        # topmost valid row per column (first True)
        row_indices = np.argmax(valid, axis=0)
    else:
        # bottommost valid row per column (last True)
        row_indices = n_rows - 1 - np.argmax(valid[::-1, :], axis=0)
    out[any_valid] = cond_zyxs[row_indices[any_valid], col_idx[any_valid], :]
    return out


def get_cond_edge_bboxes(cond_zyxs, cond_direction, crop_size, overlap_frac=0.15, cond_valid=None):
    # Build center-out crop anchors along the conditioning edge. Each chunk grows
    # while its XYZ span still fits in one crop-sized bbox.
    edge = _get_cond_edge(cond_zyxs, cond_direction, cond_valid=cond_valid)

    edge_valid = ~(edge == -1).all(axis=1)
    if not edge_valid.any():
        return [], edge
    edge = edge[edge_valid]
    n_edge = edge.shape[0]
    if n_edge == 0:
        return [], edge

    crop_size_arr = np.asarray(crop_size, dtype=np.int64)

    overlap_frac = float(overlap_frac)
    overlap_frac = max(0.0, min(overlap_frac, 0.99))

    span_limit = crop_size_arr - 1

    def _chunk_ordered_indices(ordered_indices):
        chunks = []
        if len(ordered_indices) == 0:
            return chunks
        start = 0
        while start < len(ordered_indices):
            first_pt = edge[ordered_indices[start]]
            running_min = first_pt.copy()
            running_max = first_pt.copy()
            end = start + 1
            while end < len(ordered_indices):
                next_pt = edge[ordered_indices[end]]
                candidate_min = np.minimum(running_min, next_pt)
                candidate_max = np.maximum(running_max, next_pt)
                if np.all((candidate_max - candidate_min) <= span_limit):
                    running_min = candidate_min
                    running_max = candidate_max
                    end += 1
                    continue
                break
            chunk = ordered_indices[start:end]
            if len(chunk) == 0:
                break
            chunks.append(chunk)
            # Once a chunk reaches the side endpoint, further starts only create
            # nested tail chunks that heavily overlap and can quantize to duplicates.
            if end >= len(ordered_indices):
                break
            chunk_len = len(chunk)
            overlap_count = int(round(chunk_len * overlap_frac))
            # Slide by (chunk - overlap) so adjacent bboxes share context.
            step = max(1, chunk_len - overlap_count)
            start += step
        return chunks

    center_idx = n_edge // 2
    first_side = np.arange(center_idx, -1, -1, dtype=np.int64)

    first_chunks = _chunk_ordered_indices(first_side)
    seam_overlap_count = 0
    if first_chunks:
        seam_overlap_count = int(round(len(first_chunks[0]) * overlap_frac))
        seam_overlap_count = max(0, min(seam_overlap_count, center_idx + 1))
    second_start = max(0, center_idx + 1 - seam_overlap_count)
    second_side = np.arange(second_start, n_edge, dtype=np.int64)
    second_chunks = _chunk_ordered_indices(second_side)

    bboxes = []
    seen_bboxes = set()

    def _append_chunks(chunks):
        for chunk in chunks:
            pts = edge[chunk]
            center = (pts.min(axis=0) + pts.max(axis=0)) / 2
            # Align to voxel indices so inclusive bounds match a crop of size crop_size.
            half = (crop_size_arr - 1) / 2.0
            min_corner = np.floor(center - half).astype(np.int64)
            max_corner = min_corner + (crop_size_arr - 1)
            bbox = (
                int(min_corner[0]), int(max_corner[0]),
                int(min_corner[1]), int(max_corner[1]),
                int(min_corner[2]), int(max_corner[2]),
            )
            if bbox in seen_bboxes:
                continue
            seen_bboxes.add(bbox)
            bboxes.append(bbox)

    _append_chunks(first_chunks)
    _append_chunks(second_chunks)

    return bboxes, edge


def _resolve_segment_volume(segment, volume_scale=None):
    volume = segment.volume
    if isinstance(volume, zarr.Group):
        target_level = None
        if volume_scale is not None:
            target_level = int(volume_scale)
        else:
            extra = getattr(segment, "extra", None)
            if isinstance(extra, dict):
                for key in ("volume_scale", "vol_scale", "zarr_level", "volume_level", "level"):
                    if key in extra:
                        try:
                            target_level = int(extra[key])
                            break
                        except (TypeError, ValueError):
                            continue
        if target_level is None:
            target_level = 0
        level_key = str(target_level)
        if level_key in volume:
            return volume[level_key]
        numeric_levels = sorted([k for k in volume.keys() if k.isdigit()], key=int)
        if numeric_levels:
            level_ints = [int(k) for k in numeric_levels]
            nearest = min(level_ints, key=lambda v: abs(v - target_level))
            return volume[str(nearest)]
    return volume


def _bbox_to_min_corner_and_bounds_array(bbox):
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    min_corner = np.floor([z_min, y_min, x_min]).astype(np.int64)
    bounds_array = np.asarray(
        [
            [z_min, y_min, x_min],
            [z_max, y_max, x_max],
        ],
        dtype=np.int32,
    )
    return min_corner, bounds_array


def _crop_volume_from_min_corner(volume, min_corner, crop_size):
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    max_corner = min_corner + crop_size_arr
    vol_crop = np.zeros(tuple(crop_size_arr.tolist()), dtype=volume.dtype)
    vol_shape = np.array(volume.shape, dtype=np.int64)
    src_starts = np.maximum(min_corner, 0)
    src_ends = np.minimum(max_corner, vol_shape)
    dst_starts = src_starts - min_corner
    dst_ends = dst_starts + (src_ends - src_starts)

    if np.all(src_ends > src_starts):
        vol_crop[
            dst_starts[0]:dst_ends[0],
            dst_starts[1]:dst_ends[1],
            dst_starts[2]:dst_ends[2],
        ] = volume[
            src_starts[0]:src_ends[0],
            src_starts[1]:src_ends[1],
            src_starts[2]:src_ends[2],
        ]

    return vol_crop


def _compute_query_mask_span(cond_span, cond_pct):
    cond_span = int(max(1, cond_span))
    # cond_pct is conditioning fraction of the combined (cond + extrap) span.
    # Rearranging gives extrap span to query beyond the current boundary.
    if cond_pct <= 0:
        return cond_span
    total_span = max(cond_span + 1, int(round(cond_span / float(cond_pct))))
    return max(1, total_span - cond_span)


def _profile_section(profiler, name):
    if profiler is None:
        return nullcontext()
    return profiler.section(name)


def _build_uv_query_from_edge_band(uv_edge_pts, grow_direction, cond_pct):
    """
    Build extrapolation UVs from a per-line frontier over edge-conditioning UVs.
    """
    if uv_edge_pts is None or len(uv_edge_pts) == 0:
        return np.zeros((0, 0, 2), dtype=np.int32)

    _, direction = _get_growth_context(grow_direction)
    uv = _coerce_uv_int_array(uv_edge_pts, prefer_int32=True, default_dtype=np.int32)

    if direction["axis"] == "col":
        rows = uv[:, 0]
        cols = uv[:, 1]
        unique_rows, row_inv = np.unique(rows, return_inverse=True)
        if unique_rows.size == 0:
            return np.zeros((0, 0, 2), dtype=np.int32)

        row_min = np.full((unique_rows.size,), np.iinfo(uv.dtype).max, dtype=uv.dtype)
        row_max = np.full((unique_rows.size,), np.iinfo(uv.dtype).min, dtype=uv.dtype)
        np.minimum.at(row_min, row_inv, cols)
        np.maximum.at(row_max, row_inv, cols)

        cond_span = int(
            np.median(
                row_max.astype(np.int64, copy=False) - row_min.astype(np.int64, copy=False) + 1
            )
        )
        mask_w = _compute_query_mask_span(cond_span, cond_pct)
        row_val_min = int(unique_rows.min())
        row_val_max = int(unique_rows.max())
        if direction["growth_sign"] > 0:
            col_val_min = int(row_max.min()) + 1
            col_val_max = int(row_max.max()) + int(mask_w)
        else:
            col_val_min = int(row_min.min()) - int(mask_w)
            col_val_max = int(row_min.max()) - 1
        query_dtype = _int_dtype_for_value_range(
            min(row_val_min, col_val_min),
            max(row_val_max, col_val_max),
            prefer_int32=True,
        )

        unique_rows = unique_rows.astype(query_dtype, copy=False)
        row_min = row_min.astype(query_dtype, copy=False)
        row_max = row_max.astype(query_dtype, copy=False)
        offsets = np.arange(mask_w, dtype=query_dtype)

        if direction["growth_sign"] > 0:
            frontier = row_max
            query_cols = (frontier[:, None] + 1) + offsets[None, :]
        else:
            frontier = row_min
            query_cols = (frontier[:, None] - mask_w) + offsets[None, :]

        query_rows = np.repeat(unique_rows[:, None], mask_w, axis=1)
        return np.stack([query_rows, query_cols], axis=-1)

    rows = uv[:, 0]
    cols = uv[:, 1]
    unique_cols, col_inv = np.unique(cols, return_inverse=True)
    if unique_cols.size == 0:
        return np.zeros((0, 0, 2), dtype=np.int32)

    col_min = np.full((unique_cols.size,), np.iinfo(uv.dtype).max, dtype=uv.dtype)
    col_max = np.full((unique_cols.size,), np.iinfo(uv.dtype).min, dtype=uv.dtype)
    np.minimum.at(col_min, col_inv, rows)
    np.maximum.at(col_max, col_inv, rows)

    cond_span = int(
        np.median(
            col_max.astype(np.int64, copy=False) - col_min.astype(np.int64, copy=False) + 1
        )
    )
    mask_h = _compute_query_mask_span(cond_span, cond_pct)
    col_val_min = int(unique_cols.min())
    col_val_max = int(unique_cols.max())
    if direction["growth_sign"] > 0:
        row_val_min = int(col_max.min()) + 1
        row_val_max = int(col_max.max()) + int(mask_h)
    else:
        row_val_min = int(col_min.min()) - int(mask_h)
        row_val_max = int(col_min.max()) - 1
    query_dtype = _int_dtype_for_value_range(
        min(row_val_min, col_val_min),
        max(row_val_max, col_val_max),
        prefer_int32=True,
    )

    unique_cols = unique_cols.astype(query_dtype, copy=False)
    col_min = col_min.astype(query_dtype, copy=False)
    col_max = col_max.astype(query_dtype, copy=False)
    offsets = np.arange(mask_h, dtype=query_dtype)

    if direction["growth_sign"] > 0:
        frontier = col_max
        query_rows = (frontier[:, None] + 1) + offsets[None, :]
    else:
        frontier = col_min
        query_rows = (frontier[:, None] - mask_h) + offsets[None, :]

    query_cols = np.repeat(unique_cols[:, None], mask_h, axis=1)
    return np.stack([query_rows, query_cols], axis=-1)


def _build_edge_input_mask(cond_valid, cond_direction, edge_input_rowscols):
    cond_valid = np.asarray(cond_valid, dtype=bool)
    if cond_valid.ndim != 2:
        raise ValueError(f"cond_valid must be 2D, got shape {cond_valid.shape}")

    n_axis = int(edge_input_rowscols)
    if n_axis < 1:
        raise ValueError("edge_input_rowscols must be >= 1")

    spec = _get_direction_spec(cond_direction)
    if spec["axis"] == "col":
        if spec["edge_idx"] == -1:
            # rank valid cells from right edge toward left, per row
            rank = np.cumsum(cond_valid[:, ::-1], axis=1)[:, ::-1]
        else:
            # rank valid cells from left edge toward right, per row
            rank = np.cumsum(cond_valid, axis=1)
    else:
        if spec["edge_idx"] == -1:
            # rank valid cells from bottom edge toward top, per column
            rank = np.cumsum(cond_valid[::-1, :], axis=0)[::-1, :]
        else:
            # rank valid cells from top edge toward bottom, per column
            rank = np.cumsum(cond_valid, axis=0)

    return cond_valid & (rank > 0) & (rank <= n_axis)


def compute_extrapolation_infer(
    uv_cond,
    zyx_cond,
    uv_query,
    min_corner,
    crop_size,
    method="rbf",
    cond_direction=None,
    degrade_prob=0.0,
    degrade_curvature_range=(0.001, 0.01),
    degrade_gradient_range=(0.05, 0.2),
    skip_bounds_check=False,
    profiler=None,
    **method_kwargs,
):
    if method not in _EXTRAPOLATION_METHODS:
        available = list(_EXTRAPOLATION_METHODS.keys())
        raise ValueError(f"Unknown extrapolation method '{method}'. Available: {available}")

    uv_cond_flat = uv_cond.reshape(-1, 2)
    zyx_cond_flat = zyx_cond.reshape(-1, 3)
    uv_query_flat = uv_query.reshape(-1, 2)
    if uv_cond_flat.size == 0 or uv_query_flat.size == 0:
        return None

    extrapolate_fn = _EXTRAPOLATION_METHODS[method]
    with _profile_section(profiler, "iter_edge_extrapolate_call"):
        zyx_extrapolated = extrapolate_fn(
            uv_cond=uv_cond_flat,
            zyx_cond=zyx_cond_flat,
            uv_query=uv_query_flat,
            min_corner=min_corner,
            crop_size=crop_size,
            cond_direction=cond_direction,
            profiler=profiler,
            **method_kwargs,
        )

    min_corner_arr = np.asarray(min_corner, dtype=zyx_extrapolated.dtype)
    zyx_extrap_local_full = zyx_extrapolated - min_corner_arr[None, :]

    if degrade_prob > 0.0 and cond_direction is not None:
        uv_shape = uv_query.shape[:2]
        zyx_extrap_local_full, _ = apply_degradation(
            zyx_extrap_local_full,
            uv_shape,
            cond_direction,
            degrade_prob=degrade_prob,
            curvature_range=degrade_curvature_range,
            gradient_range=degrade_gradient_range,
        )
    extrap_coords_local = zyx_extrap_local_full
    extrap_surface = None
    if not skip_bounds_check:
        in_bounds = _in_bounds_mask(zyx_extrap_local_full, crop_size)
        if not in_bounds.any():
            return None

        extrap_coords_local = zyx_extrap_local_full[in_bounds]

        uv_query_shape = uv_query.shape[:2]
        zyx_grid_local = zyx_extrap_local_full.reshape(uv_query_shape + (3,))
        extrap_surface = voxelize_surface_grid(zyx_grid_local, crop_size)

    return {
        "extrap_coords_local": extrap_coords_local,
        "extrap_surface": extrap_surface,
    }


def compute_edge_one_shot_extrapolation(
    cond_zyxs,
    cond_valid,
    uv_cond,
    grow_direction,
    edge_input_rowscols,
    cond_pct,
    method="rbf",
    min_corner=None,
    crop_size=None,
    degrade_prob=0.0,
    degrade_curvature_range=(0.001, 0.01),
    degrade_gradient_range=(0.05, 0.2),
    skip_bounds_check=True,
    profiler=None,
    rbf_lattice_stride_rc=(1, 1),
    rbf_lattice_phase_rc=(0, 0),
    **method_kwargs,
):
    cond_zyxs = np.asarray(cond_zyxs)
    uv_cond = np.asarray(uv_cond)
    if cond_zyxs.shape[:2] != uv_cond.shape[:2]:
        raise ValueError(
            f"cond_zyxs and uv_cond must share HxW; got {cond_zyxs.shape[:2]} vs {uv_cond.shape[:2]}"
        )

    if cond_valid is not None and np.asarray(cond_valid).shape == cond_zyxs.shape[:2]:
        cond_valid_base = np.asarray(cond_valid, dtype=bool)
    else:
        cond_valid_base = _valid_surface_mask(cond_zyxs)

    if not cond_valid_base.any():
        return None

    cond_direction, _ = _get_growth_context(grow_direction)
    with _profile_section(profiler, "iter_edge_mask_build"):
        edge_seed_mask = _build_edge_input_mask(cond_valid_base, cond_direction, edge_input_rowscols)
    if not edge_seed_mask.any():
        return None

    # Build query span from the edge-input conditioning band, not the full grown
    # surface, so iterative runs do not blow up one-shot query allocations.
    edge_seed_uv = _coerce_uv_int_array(
        uv_cond[edge_seed_mask],
        prefer_int32=True,
        default_dtype=np.int32,
    )
    with _profile_section(profiler, "iter_edge_query_build"):
        query_uv_grid = _build_uv_query_from_edge_band(edge_seed_uv, grow_direction, cond_pct)
    if query_uv_grid.size == 0:
        return None

    edge_seed_world = cond_zyxs[edge_seed_mask]
    edge_seed_world_for_extrap = edge_seed_world
    method_kwargs = dict(method_kwargs)
    rbf_uv_domain = str(method_kwargs.pop("rbf_uv_domain", "full")).strip().lower()
    if rbf_uv_domain not in {"full", "stored_lattice"}:
        raise ValueError(
            f"Unsupported rbf_uv_domain '{rbf_uv_domain}'. "
            "Expected one of: ['full', 'stored_lattice']"
        )
    phase_override = method_kwargs.pop("rbf_uv_phase_rc", None)

    def _normalize_pair(pair, label, min_value=None):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{label} must be a 2-element list/tuple.")
        a = int(pair[0])
        b = int(pair[1])
        if min_value is not None and (a < min_value or b < min_value):
            raise ValueError(f"{label} elements must be >= {min_value}.")
        return (a, b)

    def _project_uv_to_stored_lattice(uv_arr, stride_rc, phase_rc):
        uv_int = _coerce_uv_int_array(
            uv_arr,
            prefer_int32=False,
            default_dtype=np.int64,
        ).astype(np.int64, copy=False)
        if uv_int.size == 0:
            return uv_int
        sub_r, sub_c = stride_rc
        phase_r, phase_c = phase_rc
        out = np.empty_like(uv_int, dtype=np.int64)
        out[..., 0] = np.floor_divide(uv_int[..., 0] - phase_r, sub_r)
        out[..., 1] = np.floor_divide(uv_int[..., 1] - phase_c, sub_c)
        return out

    def _aggregate_world_by_uv(uv_arr, world_arr):
        uv_int = np.asarray(uv_arr, dtype=np.int64)
        world = np.asarray(world_arr, dtype=np.float32)
        if uv_int.ndim != 2 or uv_int.shape[1] != 2:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
        if world.ndim != 2 or world.shape[1] != 3:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
        if uv_int.shape[0] == 0 or world.shape[0] == 0 or uv_int.shape[0] != world.shape[0]:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
        uniq_uv, inv = np.unique(uv_int, axis=0, return_inverse=True)
        world_sum = np.zeros((uniq_uv.shape[0], 3), dtype=np.float64)
        world_count = np.zeros((uniq_uv.shape[0],), dtype=np.int64)
        np.add.at(world_sum, inv, world.astype(np.float64, copy=False))
        np.add.at(world_count, inv, 1)
        world_mean = (world_sum / np.maximum(world_count[:, None], 1)).astype(np.float32, copy=False)
        return uniq_uv.astype(np.int64, copy=False), world_mean

    min_corner_arr = (
        np.asarray(min_corner, dtype=np.float64)
        if min_corner is not None
        else np.zeros(3, dtype=np.float64)
    )
    if min_corner_arr.shape != (3,):
        raise ValueError(f"min_corner must have shape (3,), got {min_corner_arr.shape}")

    if crop_size is None:
        if not skip_bounds_check:
            raise ValueError("crop_size is required when skip_bounds_check=False")
        crop_size_use = (1, 1, 1)
    else:
        crop_size_use = tuple(int(v) for v in crop_size)

    uv_cond_for_extrap = edge_seed_uv
    uv_query_for_extrap = query_uv_grid
    lattice_stride_rc = _normalize_pair(
        rbf_lattice_stride_rc,
        "rbf_lattice_stride_rc",
        min_value=1,
    )
    lattice_phase_rc = _normalize_pair(
        phase_override if phase_override is not None else rbf_lattice_phase_rc,
        "rbf lattice phase",
    )

    if method in {"rbf", "rbf_edge_only"} and rbf_uv_domain == "stored_lattice":
        uv_cond_stored = _project_uv_to_stored_lattice(
            edge_seed_uv,
            lattice_stride_rc,
            lattice_phase_rc,
        )
        uv_cond_stored, edge_seed_world_for_extrap = _aggregate_world_by_uv(uv_cond_stored, edge_seed_world)
        uv_query_for_extrap = _project_uv_to_stored_lattice(
            query_uv_grid,
            lattice_stride_rc,
            lattice_phase_rc,
        )
        uv_cond_for_extrap = uv_cond_stored

    extrap_result = compute_extrapolation_infer(
        uv_cond=uv_cond_for_extrap,
        zyx_cond=edge_seed_world_for_extrap,
        uv_query=uv_query_for_extrap,
        min_corner=min_corner_arr,
        crop_size=crop_size_use,
        method=method,
        cond_direction=cond_direction,
        degrade_prob=degrade_prob,
        degrade_curvature_range=degrade_curvature_range,
        degrade_gradient_range=degrade_gradient_range,
        skip_bounds_check=skip_bounds_check,
        profiler=profiler,
        **method_kwargs,
    )
    if extrap_result is None:
        extrapolated_local = np.zeros((0, 3), dtype=np.float32)
        extrap_surface = None
    else:
        extrapolated_local = np.asarray(extrap_result["extrap_coords_local"], dtype=np.float32)
        extrap_surface = extrap_result["extrap_surface"]

    if extrapolated_local.size == 0:
        extrapolated_world = np.zeros((0, 3), dtype=np.float32)
    else:
        extrapolated_world = extrapolated_local + min_corner_arr[None, :].astype(np.float32, copy=False)

    return {
        "cond_direction": cond_direction,
        "edge_seed_mask": edge_seed_mask,
        "edge_seed_uv": edge_seed_uv.astype(np.int64, copy=False),
        "edge_seed_world": edge_seed_world.astype(np.float32, copy=False),
        "query_uv_grid": query_uv_grid,
        "rbf_uv_domain": rbf_uv_domain,
        "rbf_lattice_stride_rc": lattice_stride_rc,
        "rbf_lattice_phase_rc": lattice_phase_rc,
        "min_corner": min_corner_arr,
        "extrapolated_local": extrapolated_local,
        "extrapolated_world": extrapolated_world,
        "extrap_surface": extrap_surface,
    }


def get_window_bounds_from_bboxes(zyxs, valid, bboxes, pad=2):
    h, w = zyxs.shape[:2]
    r_min, r_max = h, -1
    c_min, c_max = w, -1

    z, y, x = zyxs[..., 0], zyxs[..., 1], zyxs[..., 2]
    valid_rows, valid_cols = np.where(valid)
    if valid_rows.size == 0:
        return 0, h - 1, 0, w - 1
    valid_z = z[valid_rows, valid_cols]
    valid_y = y[valid_rows, valid_cols]
    valid_x = x[valid_rows, valid_cols]

    for bbox in bboxes:
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        hit_idx = np.flatnonzero(
            (valid_z >= z_min) & (valid_z <= z_max) &
            (valid_y >= y_min) & (valid_y <= y_max) &
            (valid_x >= x_min) & (valid_x <= x_max)
        )
        if hit_idx.size == 0:
            continue
        rows_hit = valid_rows[hit_idx]
        cols_hit = valid_cols[hit_idx]
        r0, r1 = rows_hit.min(), rows_hit.max()
        c0, c1 = cols_hit.min(), cols_hit.max()
        r_min = min(r_min, r0)
        r_max = max(r_max, r1)
        c_min = min(c_min, c0)
        c_max = max(c_max, c1)

    if r_max < r_min or c_max < c_min:
        return 0, h - 1, 0, w - 1

    r_min = max(0, r_min - pad)
    r_max = min(h - 1, r_max + pad)
    c_min = max(0, c_min - pad)
    c_max = min(w - 1, c_max + pad)
    return r_min, r_max, c_min, c_max


def _build_uv_grid(uv_offset, shape_hw):
    r0, c0 = uv_offset
    h, w = shape_hw
    rows = np.arange(r0, r0 + h, dtype=np.int64)
    cols = np.arange(c0, c0 + w, dtype=np.int64)
    return np.stack(np.meshgrid(rows, cols, indexing="ij"), axis=-1)


def _scale_to_subsample_stride(scale):
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid tifxyz scale: {scale}")
    return max(1, int(round(1.0 / scale)))


def _stored_to_full_bounds(tgt_segment, stored_bounds):
    r0_s, r1_s, c0_s, c1_s = stored_bounds
    scale_y, scale_x = tgt_segment._scale
    full_h, full_w = tgt_segment.full_resolution_shape
    sub_r = _scale_to_subsample_stride(scale_y)
    sub_c = _scale_to_subsample_stride(scale_x)
    # Convert stored grid indices to full-resolution UV indices using integer
    # stride math to avoid float drift (e.g. 0.10000000149) dropping edge voxels.
    r0_full = max(0, int(r0_s) * sub_r)
    # Upper bound is exclusive. Map inclusive stored max directly, then +1.
    r1_full = min(full_h, int(r1_s) * sub_r + 1)
    c0_full = max(0, int(c0_s) * sub_c)
    c1_full = min(full_w, int(c1_s) * sub_c + 1)
    return r0_full, r1_full, c0_full, c1_full


def _initialize_window_state(tgt_segment, full_bounds):
    r0_full, r1_full, c0_full, c1_full = full_bounds
    tgt_segment.use_full_resolution()
    x, y, z, valid = tgt_segment[r0_full:r1_full, c0_full:c1_full]

    window_zyxs = np.stack([z, y, x], axis=-1)
    return window_zyxs.copy(), valid.copy(), (r0_full, c0_full)


def setup_segment(args, volume):
    tifxyz_path = Path(args.tifxyz_path)
    if not tifxyz_path.exists():
        raise FileNotFoundError(f"tifxyz path not found: {tifxyz_path}")
    if not tifxyz_path.is_dir():
        raise NotADirectoryError(f"tifxyz path must be a directory: {tifxyz_path}")

    tgt_segment = tifxyz.read_tifxyz(tifxyz_path)
    retarget_factor = 2 ** args.volume_scale
    tgt_segment = tgt_segment.retarget(retarget_factor)
    tgt_segment.volume = volume

    tgt_segment.use_stored_resolution()
    x_s, y_s, z_s, valid_s = tgt_segment[:]
    stored_zyxs = np.stack([z_s, y_s, x_s], axis=-1)

    h_s, w_s = stored_zyxs.shape[:2]
    valid_rows = np.any(valid_s, axis=1)
    valid_cols = np.any(valid_s, axis=0)
    valid_dirs = []
    if valid_cols.sum() >= 2:
        valid_dirs.extend(["left", "right"])
    if valid_rows.sum() >= 2:
        valid_dirs.extend(["up", "down"])
    if not valid_dirs:
        raise RuntimeError("Segment too small to define a split direction.")
    grow_direction = args.grow_direction
    cond_direction, _ = _get_growth_context(grow_direction)
    if cond_direction not in valid_dirs:
        raise RuntimeError(
            f"Requested grow_direction '{args.grow_direction}' (cond_direction='{cond_direction}') "
            f"not available for this segment. Valid options: {valid_dirs}"
        )

    return tgt_segment, stored_zyxs, valid_s, grow_direction, h_s, w_s


def _agg_extrap_axis_metadata(grow_direction):
    grow_direction = str(grow_direction).lower()
    if grow_direction in {"left", "right"}:
        axis_idx = 1
        axis_name = "col"
    elif grow_direction in {"up", "down"}:
        axis_idx = 0
        axis_name = "row"
    else:
        raise ValueError(f"Unknown grow_direction '{grow_direction}'")
    near_to_far_desc = grow_direction in {"left", "up"}
    return axis_idx, axis_name, near_to_far_desc


def _finite_uvs_from_extrap_lookup(extrap_lookup):
    if extrap_lookup is None:
        return np.zeros((0, 2), dtype=np.int64)

    if not isinstance(extrap_lookup, ExtrapLookupArrays):
        raise TypeError(
            "extrap_lookup must be ExtrapLookupArrays or None; "
            f"got {type(extrap_lookup).__name__}"
        )

    if extrap_lookup.uv.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    finite = np.isfinite(extrap_lookup.world).all(axis=1)
    if not finite.any():
        return np.zeros((0, 2), dtype=np.int64)
    return extrap_lookup.uv[finite].astype(np.int64, copy=False)


def _select_extrap_uv_indices_for_sampling(uv_ordered, grow_direction, max_lines=None):
    uv_ordered = np.asarray(uv_ordered, dtype=np.int64)
    if uv_ordered.ndim != 2 or uv_ordered.shape[1] != 2 or uv_ordered.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    axis_idx, _, near_to_far_desc = _agg_extrap_axis_metadata(grow_direction)
    primary_axis_vals = uv_ordered[:, axis_idx]
    primary = -primary_axis_vals if near_to_far_desc else primary_axis_vals
    secondary = uv_ordered[:, 1 - axis_idx]
    order = np.lexsort((secondary, primary))
    ordered_idx = order.astype(np.int64, copy=False)
    uv_ordered = uv_ordered[ordered_idx]

    if max_lines is None:
        return ordered_idx

    depth_keep = int(max_lines)
    if depth_keep < 1:
        return np.zeros((0,), dtype=np.int64)

    # For ragged/non-rectangular fronts, keep near->far depth per boundary line
    # (per row for left/right, per col for up/down), not global axis values.
    boundary_axis_idx = 1 - axis_idx
    boundary_ids = uv_ordered[:, boundary_axis_idx]
    line_primary_vals = uv_ordered[:, axis_idx]
    line_primary = -line_primary_vals if near_to_far_desc else line_primary_vals
    original_pos = np.arange(uv_ordered.shape[0], dtype=np.int64)
    grouped_order = np.lexsort((original_pos, line_primary, boundary_ids))

    grouped_boundary = boundary_ids[grouped_order]
    if grouped_boundary.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    group_starts = np.flatnonzero(
        np.concatenate(([True], grouped_boundary[1:] != grouped_boundary[:-1]))
    )
    group_ends = np.concatenate((group_starts[1:], np.array([grouped_boundary.shape[0]], dtype=np.int64)))

    picked = []
    for start, end in zip(group_starts, group_ends):
        keep = min(depth_keep, int(end - start))
        if keep > 0:
            picked.append(grouped_order[start:start + keep])
    if not picked:
        return np.zeros((0,), dtype=np.int64)

    selected_order_idx = np.concatenate(picked, axis=0).astype(np.int64, copy=False)
    selected_uv = uv_ordered[selected_order_idx]
    sel_primary_vals = selected_uv[:, axis_idx]
    sel_primary = -sel_primary_vals if near_to_far_desc else sel_primary_vals
    sel_secondary = selected_uv[:, 1 - axis_idx]
    sel_order = np.lexsort((sel_secondary, sel_primary))
    selected_order_idx = selected_order_idx[sel_order]
    return ordered_idx[selected_order_idx].astype(np.int64, copy=False)


def _select_extrap_uvs_for_sampling(extrap_lookup, grow_direction, max_lines=None):
    uv_ordered = _finite_uvs_from_extrap_lookup(extrap_lookup)
    if uv_ordered.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    selected_idx = _select_extrap_uv_indices_for_sampling(
        uv_ordered,
        grow_direction,
        max_lines=max_lines,
    )
    if selected_idx.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return uv_ordered[selected_idx].astype(np.int64, copy=False)


def _print_agg_extrap_sampling_debug(samples, extrap_lookup, grow_direction, max_lines=None, verbose=True):
    if not verbose:
        return
    all_uv = _select_extrap_uvs_for_sampling(extrap_lookup, grow_direction, max_lines=None)
    sampled_uv = np.asarray(samples.get("uv", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    axis_idx, axis_name, _ = _agg_extrap_axis_metadata(grow_direction)
    total_uv = int(all_uv.shape[0])
    total_lines = int(np.unique(all_uv[:, axis_idx]).shape[0]) if total_uv > 0 else 0
    sampled = int(sampled_uv.shape[0])
    sampled_lines = int(np.unique(sampled_uv[:, axis_idx]).shape[0]) if sampled > 0 else 0
    stack_count = np.asarray(samples.get("stack_count", np.zeros((0,), dtype=np.uint32)), dtype=np.uint32)
    print("== Aggregated Extrapolation Stack Sampling ==")
    print(f"sampled aggregated-extrap UVs: {sampled}/{total_uv}")
    print(f"sampled {axis_name} lines (near->far): {sampled_lines}/{total_lines}")
    if max_lines is not None:
        print(f"line limit requested: {int(max_lines)}")
    if stack_count.size > 0:
        sc = stack_count.astype(np.float32, copy=False)
        print(f"stack-count min/max/mean: {int(sc.min())}/{int(sc.max())}/{sc.mean():.2f}")


def _save_merged_surface_tifxyz(args, merged, checkpoint_path, model_config, call_args,
                                 input_scale=None):
    merged_zyxs = np.asarray(merged.get("merged_zyxs"), dtype=np.float32)
    merged_valid = np.asarray(merged.get("merged_valid"), dtype=bool)
    if merged_zyxs.ndim != 3 or merged_zyxs.shape[-1] != 3:
        raise RuntimeError(f"Unexpected merged surface shape: {tuple(merged_zyxs.shape)}")
    if merged_valid.shape != merged_zyxs.shape[:2]:
        raise RuntimeError(
            "merged_valid shape must match merged_zyxs spatial dimensions: "
            f"{merged_valid.shape} vs {merged_zyxs.shape[:2]}"
        )

    merged_for_save = np.full_like(merged_zyxs, -1.0, dtype=np.float32)
    merged_for_save[merged_valid] = merged_zyxs[merged_valid]

    scale_factor = int(2 ** int(args.volume_scale))
    if scale_factor != 1:
        merged_for_save = np.where(
            (merged_for_save == -1).all(axis=-1, keepdims=True),
            -1.0,
            merged_for_save * scale_factor,
        )

    tifxyz_step_size, tifxyz_voxel_size_um = resolve_tifxyz_params(
        args, model_config, args.volume_scale
    )
    current_step = int(round(2 ** int(args.volume_scale)))
    if input_scale is not None:
        # Stored-resolution input: stride accounts for the scale factor
        stride_y = max(1, int(round(float(tifxyz_step_size) * input_scale[0] / max(1, current_step))))
        stride_x = max(1, int(round(float(tifxyz_step_size) * input_scale[1] / max(1, current_step))))
    else:
        # Full-resolution input (legacy)
        stride_y = max(1, int(round(float(tifxyz_step_size) / max(1, current_step))))
        stride_x = stride_y
    if stride_y > 1 or stride_x > 1:
        merged_for_save = merged_for_save[::stride_y, ::stride_x]

    overwrite_input_surface = bool(getattr(args, "overwrite_input_surface", False))
    if overwrite_input_surface:
        input_tifxyz_path = os.path.abspath(str(args.tifxyz_path))
        tifxyz_uuid = os.path.basename(os.path.normpath(input_tifxyz_path))
        if not tifxyz_uuid:
            raise RuntimeError(
                "--overwrite-input-surface requires --tifxyz-path to point to a tifxyz directory."
            )
        out_dir = os.path.dirname(input_tifxyz_path)
        if args.tifxyz_out_dir:
            print("--overwrite-input-surface set: ignoring --tifxyz-out-dir.")
    else:
        out_dir = args.tifxyz_out_dir if args.tifxyz_out_dir else str(Path(args.tifxyz_path).parent)
        ckpt_name = "no_ckpt" if checkpoint_path is None else os.path.splitext(os.path.basename(str(checkpoint_path)))[0]
        timestamp = datetime.now().strftime("%H%M%S")
        tifxyz_uuid = f"displacement_{ckpt_name}_{timestamp}"

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    uv_offset = merged.get("uv_offset", (0, 0))
    stored_projection = merged.get("stored_projection", {})
    stored_proj_mode = str(stored_projection.get("mode", "unknown"))
    stored_proj_stride = stored_projection.get("stride_rc", [None, None])
    stored_proj_phase = stored_projection.get("phase_rc", [None, None])
    source = str(checkpoint_path) if checkpoint_path else "inference/infer_global_extrap.py"
    save_tifxyz(
        merged_for_save,
        out_dir,
        tifxyz_uuid,
        step_size=tifxyz_step_size,
        voxel_size_um=tifxyz_voxel_size_um,
        source=source,
        additional_metadata={
            "grow_direction": args.grow_direction,
            "extrapolation_method": args.extrapolation_method,
            "uv_offset_rc": [int(uv_offset[0]), int(uv_offset[1])],
            "agg_extrap_lines": None if args.agg_extrap_lines is None else int(args.agg_extrap_lines),
            "stored_projection_mode": stored_proj_mode,
            "stored_projection_stride_rc": _json_safe(stored_proj_stride),
            "stored_projection_phase_rc": _json_safe(stored_proj_phase),
            "effective_step_size_used": int(tifxyz_step_size),
            "run_argv": list(sys.argv[1:]),
            "run_args": _json_safe(call_args),
        },
    )

    output_path = os.path.join(out_dir, tifxyz_uuid)
    print(f"Saved tifxyz to {output_path}")
    return output_path


def _print_bbox_crop_debug_table(bbox_crops, verbose=True):
    if not verbose:
        return
    if not bbox_crops:
        print("== BBox Crop Debug ==")
        print("No bbox crops.")
        return

    headers = ("bbox", "n_cond", "n_query", "n_extrap", "n_nonfinite")
    rows = []
    for crop in bbox_crops:
        bbox_idx = int(crop["bbox_idx"])
        n_cond = int(crop.get("n_cond", 0))
        n_query = int(crop.get("n_query", 0))
        n_extrap = int(crop.get("n_extrap", 0))
        n_nonfinite = int(max(n_query - n_extrap, 0))
        rows.append((bbox_idx, n_cond, n_query, n_extrap, n_nonfinite))

    widths = []
    for i, header in enumerate(headers):
        cell_width = max(len(header), *(len(str(row[i])) for row in rows))
        widths.append(cell_width)

    def _fmt(row):
        return " | ".join(str(row[i]).rjust(widths[i]) for i in range(len(headers)))

    print("== BBox Crop Debug ==")
    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(_fmt(row))

    total_q = int(sum(row[2] for row in rows))
    total_extrap = int(sum(row[3] for row in rows))
    total_nonfinite = int(sum(row[4] for row in rows))
    print(
        f"totals: queries={total_q}, extrapolated={total_extrap}, nonfinite={total_nonfinite}"
    )


def _agg_extrap_line_summary(extrap_lookup, grow_direction):
    uv_ordered = _select_extrap_uvs_for_sampling(extrap_lookup, grow_direction, max_lines=None)
    axis_idx, axis_name, _ = _agg_extrap_axis_metadata(grow_direction)
    if uv_ordered.shape[0] == 0:
        return {
            "axis_name": axis_name,
            "uv_count": 0,
            "line_count": 0,
            "near_axis_value": None,
            "far_axis_value": None,
        }

    axis_values = uv_ordered[:, axis_idx].astype(np.int64, copy=False)
    keep = np.ones((axis_values.shape[0],), dtype=bool)
    keep[1:] = axis_values[1:] != axis_values[:-1]
    axis_values = axis_values[keep]

    return {
        "axis_name": axis_name,
        "uv_count": int(uv_ordered.shape[0]),
        "line_count": int(axis_values.shape[0]),
        "near_axis_value": int(axis_values[0]) if axis_values.shape[0] > 0 else None,
        "far_axis_value": int(axis_values[-1]) if axis_values.shape[0] > 0 else None,
    }


def _print_iteration_summary(bbox_results, edge_extrapolation, extrap_lookup, grow_direction, verbose=True):
    if not verbose:
        return
    agg_summary = _agg_extrap_line_summary(extrap_lookup, grow_direction)
    extrap_uv_count = int(_finite_uvs_from_extrap_lookup(extrap_lookup).shape[0])
    print("== Extrapolation Summary ==")
    print(f"bboxes: {len(bbox_results)}")
    print(f"edge-seed uv count: {len(edge_extrapolation.get('edge_seed_uv', []))}")
    print(f"edge extrapolated uv count (aggregated): {extrap_uv_count}")
    print("== Aggregated Extrapolation ==")
    print(f"axis: {agg_summary['axis_name']}")
    print(f"available lines (near->far): {agg_summary['line_count']}")
    print(f"available uv count: {agg_summary['uv_count']}")
    near_axis = agg_summary["near_axis_value"]
    far_axis = agg_summary["far_axis_value"]
    if near_axis is not None:
        print(f"axis range near->far: {near_axis} -> {far_axis}")


def _show_napari(
    cond_zyxs,
    cond_valid,
    bbox_results,
    edge_extrapolation,
    extrap_lookup,
    disp_bbox=None,
    displaced=None,
    merged=None,
    downsample=8,
    point_size=1.0,
):
    try:
        import napari
    except Exception as exc:
        raise RuntimeError("--napari was set, but napari is not available.") from exc

    viewer = napari.Viewer(ndisplay=3)
    downsample = max(1, int(downsample))
    point_size = float(point_size)

    def _bbox_wireframe_segments(bbox):
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        corners = np.asarray(
            [
                [z_min, y_min, x_min],
                [z_min, y_min, x_max],
                [z_min, y_max, x_min],
                [z_min, y_max, x_max],
                [z_max, y_min, x_min],
                [z_max, y_min, x_max],
                [z_max, y_max, x_min],
                [z_max, y_max, x_max],
            ],
            dtype=np.float32,
        )
        edge_pairs = (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        )
        return [corners[[i0, i1]] for (i0, i1) in edge_pairs]

    def _downsample_points(points):
        pts = np.asarray(points)
        if pts.size == 0 or downsample <= 1:
            return pts
        return pts[::downsample]

    cond_full = cond_zyxs[cond_valid]
    cond_full = _downsample_points(cond_full)
    if cond_full.size > 0:
        viewer.add_points(cond_full, name="cond_full", size=point_size, face_color=[0.7, 0.7, 0.7], opacity=0.2)

    sampled_agg = None
    if isinstance(displaced, dict):
        sampled_agg = np.asarray(displaced.get("world", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    sampled_agg = _downsample_points(sampled_agg if sampled_agg is not None else np.zeros((0, 3), dtype=np.float32))
    if sampled_agg is not None and sampled_agg.size > 0:
        viewer.add_points(
            sampled_agg,
            name="agg_extrap_sampled",
            size=point_size,
            face_color=[0.0, 1.0, 1.0],
            opacity=0.8,
        )

    displaced_band = None
    if isinstance(displaced, dict):
        displaced_band = np.asarray(displaced.get("world_displaced", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    displaced_band = _downsample_points(displaced_band if displaced_band is not None else np.zeros((0, 3), dtype=np.float32))
    if displaced_band is not None and displaced_band.size > 0:
        viewer.add_points(
            displaced_band,
            name="agg_extrap_displaced",
            size=point_size,
            face_color=[1.0, 0.0, 1.0],
            opacity=0.9,
        )

    if isinstance(merged, dict):
        merged_zyxs = np.asarray(merged.get("merged_zyxs", np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float32)
        merged_valid = np.asarray(merged.get("merged_valid", np.zeros((0, 0), dtype=bool)), dtype=bool)
        if merged_zyxs.ndim == 3 and merged_zyxs.shape[-1] == 3 and merged_valid.shape == merged_zyxs.shape[:2]:
            merged_full = merged_zyxs[merged_valid]
            merged_full = _downsample_points(merged_full)
            if merged_full.size > 0:
                viewer.add_points(
                    merged_full,
                    name="merged_full_surface",
                    size=point_size,
                    face_color=[1.0, 0.2, 0.2],
                    opacity=0.25,
                )

    if disp_bbox is not None:
        viewer.add_shapes(
            _bbox_wireframe_segments(disp_bbox),
            shape_type="path",
            edge_color=[1.0, 1.0, 1.0, 1.0],
            edge_width=2,
            face_color="transparent",
            name="disp_stack_bbox",
            opacity=1.0,
        )

    n_bbox = max(len(bbox_results), 1)
    for item in bbox_results:
        idx = int(item["bbox_idx"])
        rgb = colorsys.hsv_to_rgb((idx / n_bbox) % 1.0, 1.0, 1.0)
        viewer.add_shapes(
            _bbox_wireframe_segments(item["bbox"]),
            shape_type="path",
            edge_color=[*rgb, 0.9],
            edge_width=1,
            face_color="transparent",
            name=f"bbox_{idx:03d}_wire",
            opacity=0.9,
        )
        if item["cond_world"].size > 0:
            cond_pts = _downsample_points(item["cond_world"])
            viewer.add_points(
                cond_pts,
                name=f"bbox_{idx:03d}_cond",
                size=point_size,
                face_color=list(rgb),
                opacity=0.6,
            )
        if item["extrap_world"].size > 0:
            extrap_pts = _downsample_points(item["extrap_world"])
            viewer.add_points(
                extrap_pts,
                name=f"bbox_{idx:03d}_extrap",
                size=point_size,
                face_color=list(rgb),
                symbol="ring",
                opacity=0.9,
            )
    edge_seed_world_points = edge_extrapolation.get("edge_seed_world") if edge_extrapolation is not None else None
    if edge_seed_world_points is not None and len(edge_seed_world_points) > 0:
        edge_seed_world_points = _downsample_points(edge_seed_world_points)
        viewer.add_points(
            edge_seed_world_points,
            name="edge_seed_world",
            size=point_size,
            face_color=[1.0, 1.0, 0.0],
            opacity=0.9,
        )

    lookup_world_points = np.zeros((0, 3), dtype=np.float32)
    if extrap_lookup is not None:
        world_attr = getattr(extrap_lookup, "world", None)
        if world_attr is not None:
            lookup_world_points = np.asarray(world_attr, dtype=np.float32)
    if lookup_world_points.size > 0:
        lookup_world_points = _downsample_points(lookup_world_points)
        viewer.add_points(
            lookup_world_points,
            name="aggregated_extrapolated_world",
            size=point_size,
            face_color=[1.0, 0.4, 0.0],
            opacity=0.6,
        )

    def _default_visible(layer_name):
        if layer_name in {
            "cond_full",
            "merged_full_surface",
            "agg_extrap_sampled",
            "agg_extrap_displaced",
            "edge_seed_world",
            "disp_stack_bbox",
        }:
            return True
        if layer_name.startswith("bbox_") and (
            layer_name.endswith("_wire")
        ):
            return True
        return False

    # Keep all layers available but show only the requested subset by default.
    for layer in viewer.layers:
        layer.visible = _default_visible(layer.name)

    napari.run()


class RunTimeProfiler:
    def __init__(self, enabled=False, device=None):
        self.enabled = bool(enabled)
        self._nodes = []
        self._root_id = self._new_node("__root__", parent_id=None)
        self._stack = []
        self._device = None
        self._use_cuda_sync = False
        if self.enabled and device is not None:
            self._device = torch.device(device)
            self._use_cuda_sync = self._device.type == "cuda" and torch.cuda.is_available()

    def _new_node(self, name, parent_id):
        node = {
            "name": str(name),
            "parent_id": parent_id,
            "children": [],
            "children_by_name": {},
            "total_s": 0.0,
            "self_s": 0.0,
            "count": 0,
        }
        self._nodes.append(node)
        return len(self._nodes) - 1

    def _get_or_create_child_node(self, parent_id, name):
        parent = self._nodes[parent_id]
        node_id = parent["children_by_name"].get(name)
        if node_id is None:
            node_id = self._new_node(name, parent_id=parent_id)
            parent["children_by_name"][name] = node_id
            parent["children"].append(node_id)
        return node_id

    def _sync_cuda(self):
        if self._use_cuda_sync:
            torch.cuda.synchronize(self._device)

    def sync(self):
        if not self.enabled:
            return
        self._sync_cuda()

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        name = str(name)
        parent_id = self._stack[-1]["node_id"] if self._stack else self._root_id
        node_id = self._get_or_create_child_node(parent_id, name)
        self._sync_cuda()
        start = time.perf_counter()
        self._stack.append({"node_id": node_id, "start": start, "child_elapsed": 0.0})
        try:
            yield
        finally:
            self._sync_cuda()
            end = time.perf_counter()
            frame = self._stack.pop()
            elapsed = end - frame["start"]
            exclusive_elapsed = max(0.0, elapsed - float(frame["child_elapsed"]))
            if self._stack:
                self._stack[-1]["child_elapsed"] += elapsed
            node = self._nodes[frame["node_id"]]
            node["total_s"] += elapsed
            node["self_s"] += exclusive_elapsed
            node["count"] += 1

    def _iter_nodes(self):
        for node_id, node in enumerate(self._nodes):
            if node_id == self._root_id:
                continue
            yield node_id, node

    def _top_level_total(self):
        root = self._nodes[self._root_id]
        return float(sum(float(self._nodes[node_id]["total_s"]) for node_id in root["children"]))

    def _build_summary_rows(self, total_runtime_s=None):
        total_base = (
            float(total_runtime_s)
            if total_runtime_s is not None and float(total_runtime_s) > 0.0
            else self._top_level_total()
        )
        rows = []

        def _visit(node_id, parent_total_s, prefix, is_last):
            node = self._nodes[node_id]
            total_s = float(node["total_s"])
            self_s = float(node["self_s"])
            child_s = max(0.0, total_s - self_s)
            count = int(node["count"])
            avg_s = total_s / max(count, 1)
            self_avg_s = self_s / max(count, 1)
            pct_parent = 0.0 if parent_total_s <= 0.0 else (100.0 * total_s / parent_total_s)
            pct_total = 0.0 if total_base <= 0.0 else (100.0 * total_s / total_base)
            branch = "`- " if is_last else "|- "
            rows.append(
                {
                    "section": f"{prefix}{branch}{node['name']}",
                    "calls": f"{count:d}",
                    "incl_s": f"{total_s:.3f}",
                    "avg_s": f"{avg_s:.3f}",
                    "child_s": f"{child_s:.3f}",
                    "self_s": f"{self_s:.3f}",
                    "self_avg_s": f"{self_avg_s:.3f}",
                    "pct_parent_incl": f"{pct_parent:.1f}%",
                    "pct_total_incl": f"{pct_total:.1f}%",
                }
            )
            child_prefix = prefix + ("   " if is_last else "|  ")
            children = node["children"]
            for idx, child_id in enumerate(children):
                _visit(
                    child_id,
                    total_s,
                    child_prefix,
                    is_last=(idx == (len(children) - 1)),
                )

        top_level_total = self._top_level_total()
        root_children = self._nodes[self._root_id]["children"]
        for idx, child_id in enumerate(root_children):
            _visit(
                child_id,
                top_level_total,
                prefix="",
                is_last=(idx == (len(root_children) - 1)),
            )
        return rows

    def total_profiled_time(self, inclusive=False):
        if not self.enabled:
            return 0.0
        field = "total_s" if inclusive else "self_s"
        return float(sum(float(node[field]) for _, node in self._iter_nodes()))

    def print_summary(self, total_runtime_s=None):
        if not self.enabled:
            return
        headers = (
            "section",
            "calls",
            "incl_s",
            "avg_s",
            "child_s",
            "self_s",
            "self_avg_s",
            "%parent_incl",
            "%total_incl",
        )
        align_right = (False, True, True, True, True, True, True, True, True)
        rows = self._build_summary_rows(total_runtime_s=total_runtime_s)
        print("== Performance Profile ==")
        print("(incl_s is inclusive and overlaps across nested rows; self_s is exclusive.)")
        if not rows:
            print("No profiled sections.")
        else:
            table_rows = [
                (
                    row["section"],
                    row["calls"],
                    row["incl_s"],
                    row["avg_s"],
                    row["child_s"],
                    row["self_s"],
                    row["self_avg_s"],
                    row["pct_parent_incl"],
                    row["pct_total_incl"],
                )
                for row in rows
            ]
            widths = []
            for idx, header in enumerate(headers):
                width = max(len(header), *(len(row[idx]) for row in table_rows))
                widths.append(width)

            def _fmt(cells):
                parts = []
                for idx, cell in enumerate(cells):
                    if align_right[idx]:
                        parts.append(str(cell).rjust(widths[idx]))
                    else:
                        parts.append(str(cell).ljust(widths[idx]))
                return " | ".join(parts)

            print(_fmt(headers))
            print("-+-".join("-" * width for width in widths))
            for row in table_rows:
                print(_fmt(row))
        if total_runtime_s is not None:
            print(f"total_runtime: {float(total_runtime_s):.3f}s")


_RuntimeProfiler = RunTimeProfiler
