import json
import os
import colorsys
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

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
        rbf_downsample = int(
            rbf_downsample_override
            if rbf_downsample_override is not None
            else cfg.get("rbf_downsample_factor", 2)
        )
        edge_downsample_cfg = cfg.get("rbf_edge_downsample_factor", None)
        edge_downsample = rbf_downsample if edge_downsample_cfg is None else int(edge_downsample_cfg)
        method_kwargs["downsample_factor"] = edge_downsample if method == "rbf_edge_only" else rbf_downsample
        method_kwargs["rbf_max_points"] = cfg.get("rbf_max_points")
        # MUST RUN IN FLOAT64: RBF extrapolation in this pipeline is unstable at lower precision.
        method_kwargs["precision"] = cfg.get("rbf_precision", "float64")

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
    tifxyz_step_size = int(round(float(tifxyz_step_size) * (2 ** volume_scale)))
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


def save_tifxyz_output(
    args,
    tgt_segment,
    pred_samples,
    tifxyz_uuid,
    tifxyz_step_size,
    tifxyz_voxel_size_um,
    checkpoint_path,
    cond_direction,
    grow_direction,
    volume_scale,
    overlap_merge_method="mean",
):
    tgt_segment.use_full_resolution()
    full_zyxs = tgt_segment.get_zyxs(stored_resolution=False)

    h_full, w_full = full_zyxs.shape[:2]
    pred_grid, pred_valid, (uv_r_min, uv_c_min) = _aggregate_pred_samples_to_uv_grid(
        pred_samples,
        base_uv_bounds=(0, 0, h_full - 1, w_full - 1),
        overlap_merge_method=overlap_merge_method,
    )
    full_pred_zyxs = np.full_like(pred_grid, -1.0, dtype=np.float32)
    r_off = -uv_r_min
    c_off = -uv_c_min
    full_pred_zyxs[r_off:r_off + h_full, c_off:c_off + w_full] = full_zyxs
    full_pred_zyxs[pred_valid] = pred_grid[pred_valid]

    scale_factor = 2 ** volume_scale
    if scale_factor != 1:
        full_pred_zyxs_out = np.where(
            (full_pred_zyxs == -1).all(axis=-1, keepdims=True),
            -1.0,
            full_pred_zyxs * scale_factor,
        )
    else:
        full_pred_zyxs_out = full_pred_zyxs

    # Downsample grid to the correct tifxyz density (step_size spacing in full-res UV).
    current_step_y = int(round(2 ** volume_scale))
    current_step_x = int(round(2 ** volume_scale))

    stride_y = int(round(float(tifxyz_step_size) / max(1, current_step_y)))
    stride_x = int(round(float(tifxyz_step_size) / max(1, current_step_x)))
    if stride_y > 1 or stride_x > 1:
        full_pred_zyxs_out = full_pred_zyxs_out[::max(1, stride_y), ::max(1, stride_x)]

    save_tifxyz(
        full_pred_zyxs_out,
        args.tifxyz_out_dir,
        tifxyz_uuid,
        step_size=tifxyz_step_size,
        voxel_size_um=tifxyz_voxel_size_um,
        source=str(checkpoint_path),
        additional_metadata={
            "grow_direction": grow_direction,
            "cond_direction": cond_direction,
            "extrapolation_method": args.extrapolation_method,
            "refine_steps": None if args.refine is None else int(args.refine) + 1,
        }
    )
    print(f"Saved tifxyz to {os.path.join(args.tifxyz_out_dir, tifxyz_uuid)}")


def _load_optional_json(path):
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _resolve_settings(args, model_config=None, load_checkpoint_config_fn=None):
    runtime_config = {}
    if model_config is None and args.checkpoint_path and load_checkpoint_config_fn is not None:
        model_config, _ = load_checkpoint_config_fn(args.checkpoint_path)
    if model_config:
        runtime_config.update(model_config)
    runtime_config.update(_load_optional_json(args.config_path))
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


def _print_stacked_displacement_debug(stacked, verbose=True):
    if not verbose:
        return
    if stacked is None:
        print("== Displacement Stack ==")
        print("No displacement stack available.")
        return

    count = np.asarray(stacked["count"])
    total_vox = int(count.size)
    covered = int((count > 0).sum())
    max_overlap = int(count.max()) if covered > 0 else 0
    bbox = stacked["bbox"]
    print("== Displacement Stack ==")
    print(f"bbox: z[{bbox[0]}, {bbox[1]}], y[{bbox[2]}, {bbox[3]}], x[{bbox[4]}, {bbox[5]}]")
    print(f"shape: {tuple(int(v) for v in stacked['shape'])}")
    print(f"covered voxels: {covered}/{total_vox}")
    print(f"max overlap count: {max_overlap}")


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

    uv_attr = getattr(extrap_lookup, "uv", None)
    world_attr = getattr(extrap_lookup, "world", None)
    if uv_attr is not None and world_attr is not None:
        uv = np.asarray(uv_attr, dtype=np.int64)
        world = np.asarray(world_attr, dtype=np.float32)
        if uv.ndim != 2 or uv.shape[1] != 2:
            return np.zeros((0, 2), dtype=np.int64)
        if world.ndim != 2 or world.shape[1] != 3:
            return np.zeros((0, 2), dtype=np.int64)
        if uv.shape[0] != world.shape[0]:
            return np.zeros((0, 2), dtype=np.int64)
        finite = np.isfinite(world).all(axis=1)
        if not finite.any():
            return np.zeros((0, 2), dtype=np.int64)
        return uv[finite].astype(np.int64, copy=False)

    return np.zeros((0, 2), dtype=np.int64)


def _select_extrap_uvs_for_sampling(extrap_lookup, grow_direction, max_lines=None):
    uv_ordered = _finite_uvs_from_extrap_lookup(extrap_lookup)
    if uv_ordered.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    axis_idx, _, near_to_far_desc = _agg_extrap_axis_metadata(grow_direction)
    if axis_idx == 1:
        primary = -uv_ordered[:, 1] if near_to_far_desc else uv_ordered[:, 1]
        secondary = uv_ordered[:, 0]
    else:
        primary = -uv_ordered[:, 0] if near_to_far_desc else uv_ordered[:, 0]
        secondary = uv_ordered[:, 1]
    order = np.lexsort((secondary, primary))
    uv_ordered = uv_ordered[order]

    if max_lines is None:
        return uv_ordered

    axis_values = np.unique(uv_ordered[:, axis_idx]).astype(np.int64)
    if near_to_far_desc:
        axis_values = axis_values[::-1]
    selected_axis_values = axis_values[:int(max_lines)]
    keep = np.isin(uv_ordered[:, axis_idx], selected_axis_values, assume_unique=False)
    return uv_ordered[keep]


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
        sc = stack_count.astype(np.float64, copy=False)
        print(f"stack-count min/max/mean: {int(sc.min())}/{int(sc.max())}/{sc.mean():.2f}")


def _print_sample_filter_debug(filter_stats, verbose=True):
    if not verbose:
        return
    print("== Sample Filter ==")
    n_input = int(filter_stats.get("n_input", 0))
    n_after_stack_count = int(filter_stats.get("n_after_stack_count", 0))
    print(f"input samples: {n_input}")
    print(f"after min-stack-count: {n_after_stack_count}")


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


def _print_iteration_summary(bbox_results, one_shot, extrap_lookup, grow_direction, verbose=True):
    if not verbose:
        return
    agg_summary = _agg_extrap_line_summary(extrap_lookup, grow_direction)
    extrap_uv_count = int(_finite_uvs_from_extrap_lookup(extrap_lookup).shape[0])
    print("== Extrapolation Summary ==")
    print(f"bboxes: {len(bbox_results)}")
    print(f"one-shot edge-input uv count: {len(one_shot.get('edge_uv', []))}")
    print(f"one-shot extrap uv count (aggregated): {extrap_uv_count}")
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
    one_shot,
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
    edge_cond = one_shot.get("edge_zyx") if one_shot is not None else None
    if edge_cond is not None and len(edge_cond) > 0:
        edge_cond = _downsample_points(edge_cond)
        viewer.add_points(
            edge_cond,
            name="one_shot_edge_cond",
            size=point_size,
            face_color=[1.0, 1.0, 0.0],
            opacity=0.9,
        )

    one_pts = np.zeros((0, 3), dtype=np.float32)
    if extrap_lookup is not None:
        world_attr = getattr(extrap_lookup, "world", None)
        if world_attr is not None:
            one_pts = np.asarray(world_attr, dtype=np.float32)
    if one_pts.size > 0:
        one_pts = _downsample_points(one_pts)
        viewer.add_points(
            one_pts,
            name="one_shot_agg_extrap",
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
            "one_shot_edge_cond",
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
        self._totals = {}
        self._counts = {}
        self._order = []
        self._device = None
        self._use_cuda_sync = False
        if self.enabled and device is not None:
            self._device = torch.device(device)
            self._use_cuda_sync = self._device.type == "cuda" and torch.cuda.is_available()

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
        self._sync_cuda()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync_cuda()
            elapsed = time.perf_counter() - start
            if name not in self._totals:
                self._totals[name] = 0.0
                self._counts[name] = 0
                self._order.append(name)
            self._totals[name] += elapsed
            self._counts[name] += 1

    def print_summary(self, total_runtime_s=None):
        if not self.enabled:
            return
        print("== Performance Profile ==")
        for name in self._order:
            total_s = float(self._totals.get(name, 0.0))
            count = int(self._counts.get(name, 0))
            avg_s = total_s / max(count, 1)
            print(f"{name}: {total_s:.3f}s ({count}x, avg {avg_s:.3f}s)")
        if total_runtime_s is not None:
            print(f"total_runtime: {float(total_runtime_s):.3f}s")


_RuntimeProfiler = RunTimeProfiler
