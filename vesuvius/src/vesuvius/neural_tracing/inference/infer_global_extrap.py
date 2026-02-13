import argparse
import colorsys
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm

from vesuvius.image_proc.intensity.normalization import normalize_zscore
from vesuvius.neural_tracing.inference.common import (
    _aggregate_pred_samples_to_uv_grid,
    _resolve_extrapolation_settings,
    resolve_tifxyz_params,
)
from vesuvius.neural_tracing.inference.displacement_tta import TTA_MERGE_METHODS
from vesuvius.neural_tracing.inference.infer_rowcol_split import (
    _bbox_to_min_corner_and_bounds_array,
    _build_model_inputs,
    _build_uv_grid,
    _build_uv_query_from_cond_points,
    _crop_volume_from_min_corner,
    _get_growth_context,
    _grid_in_bounds_mask,
    _initialize_window_state,
    _load_optional_json,
    _points_to_voxels,
    _predict_displacement,
    _resolve_segment_volume,
    _stored_to_full_bounds,
    compute_edge_one_shot_extrapolation,
    compute_window_and_split,
    get_cond_edge_bboxes,
    load_model,
    load_checkpoint_config,
    setup_segment,
)
from vesuvius.neural_tracing.tifxyz import save_tifxyz


def _parse_optional_tta_outlier_drop_thresh(value):
    text = str(value).strip()
    if text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--tta-outlier-drop-thresh must be a positive float or 'none'."
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run global edge extrapolation with optional displacement-model stacking."
    )
    parser.add_argument("--tifxyz-path", type=str, required=True)
    parser.add_argument("--volume-path", type=str, required=True)
    parser.add_argument("--volume-scale", type=int, default=1)
    parser.add_argument("--grow-direction", type=str, required=True, choices=["left", "right", "up", "down"])
    parser.add_argument("--cond-pct", type=float, default=0.50)
    parser.add_argument("--crop-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--window-pad", type=int, default=10)
    parser.add_argument("--bbox-overlap-frac", type=float, default=0.15)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--extrapolation-method", type=str, default=None)
    parser.add_argument(
        "--tifxyz-out-dir",
        type=str,
        default=None,
        help="Output directory for merged tifxyz. Defaults to parent dir of --tifxyz-path.",
    )
    parser.add_argument(
        "--tifxyz-step-size",
        type=int,
        default=None,
        help="Output tifxyz UV step size. If unset, inferred from checkpoint config/volume metadata.",
    )
    parser.add_argument(
        "--tifxyz-voxel-size-um",
        type=float,
        default=None,
        help="Output tifxyz voxel size in micrometers. If unset, inferred from volume metadata.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip displacement model forward pass even if --checkpoint-path is provided.",
    )
    parser.add_argument(
        "--extrap-only",
        action="store_true",
        help=(
            "Iterative extrapolation-only mode: skip model inference and stacked displacement "
            "sampling, and directly merge selected aggregated extrapolation rows/cols each iteration."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterative boundary growth passes to run.",
    )
    parser.add_argument(
        "--no-tta",
        dest="tta",
        action="store_false",
        help="Disable mirroring-based test-time augmentation (enabled by default).",
    )
    parser.add_argument(
        "--tta-merge-method",
        type=str,
        default="vector_geomedian",
        choices=TTA_MERGE_METHODS,
        help="How to merge mirrored TTA displacement predictions.",
    )
    parser.add_argument(
        "--tta-outlier-drop-thresh",
        type=_parse_optional_tta_outlier_drop_thresh,
        default=1.25,
        help="Outlier threshold multiplier for dropping inconsistent TTA variants; use 'none' to disable.",
    )
    parser.add_argument(
        "--tta-outlier-drop-min-keep",
        type=int,
        default=4,
        help="Minimum number of TTA variants to keep after outlier filtering.",
    )
    parser.add_argument(
        "--edge-input-rowscols",
        type=int,
        required=True,
        help="Number of edge rows/cols from conditioning region to use in one-shot extrapolation.",
    )
    parser.add_argument(
        "--agg-extrap-lines",
        type=int,
        default=None,
        help=(
            "Optional number of near->far rows/cols to sample from one-shot aggregated extrapolation. "
            "If unset, samples all available lines."
        ),
    )
    parser.add_argument(
        "--agg-extrap-min-stack-count",
        type=int,
        default=1,
        help="Minimum stacked-field support count required for an aggregated extrapolation sample (default: 1).",
    )
    parser.add_argument(
        "--napari-downsample",
        type=int,
        default=8,
        help="Point downsample stride for Napari visualization (default: 8).",
    )
    parser.add_argument(
        "--napari-point-size",
        type=float,
        default=1.0,
        help="Point size for all Napari point layers (default: 1).",
    )
    parser.add_argument("--napari", action="store_true", help="Visualize bbox/full-edge conditioning and extrapolation.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging and tables.",
    )
    parser.set_defaults(tta=True)
    args = parser.parse_args()

    if args.edge_input_rowscols < 1:
        parser.error("--edge-input-rowscols must be >= 1")
    if args.bbox_overlap_frac < 0.0 or args.bbox_overlap_frac >= 1.0:
        parser.error("--bbox-overlap-frac must be in [0, 1)")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.agg_extrap_lines is not None and args.agg_extrap_lines < 1:
        parser.error("--agg-extrap-lines must be >= 1 when provided")
    if args.agg_extrap_min_stack_count < 1:
        parser.error("--agg-extrap-min-stack-count must be >= 1")
    if args.napari_downsample < 1:
        parser.error("--napari-downsample must be >= 1")
    if args.napari_point_size <= 0:
        parser.error("--napari-point-size must be > 0")
    if args.tta_outlier_drop_thresh is not None and args.tta_outlier_drop_thresh <= 0:
        parser.error("--tta-outlier-drop-thresh must be > 0 when provided.")
    if args.tta_outlier_drop_min_keep < 1:
        parser.error("--tta-outlier-drop-min-keep must be >= 1.")
    if args.tifxyz_step_size is not None and args.tifxyz_step_size < 1:
        parser.error("--tifxyz-step-size must be >= 1 when provided.")
    if args.tifxyz_voxel_size_um is not None and args.tifxyz_voxel_size_um <= 0:
        parser.error("--tifxyz-voxel-size-um must be > 0 when provided.")
    return args


def _resolve_settings(args, model_config=None):
    runtime_config = {}
    if model_config is None and args.checkpoint_path:
        model_config, _ = load_checkpoint_config(args.checkpoint_path)
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


def _finite_uv_world(uv, world):
    if uv is None or world is None:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float32)
    uv = np.asarray(uv)
    world = np.asarray(world)
    if uv.size == 0 or world.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float32)
    keep = np.isfinite(world).all(axis=1)
    if not keep.any():
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float32)
    return uv[keep].astype(np.float64, copy=False), world[keep].astype(np.float32, copy=False)


def _build_bbox_crops(
    bboxes,
    tgt_segment,
    volume_scale,
    cond_zyxs,
    cond_valid,
    uv_cond,
    grow_direction,
    crop_size,
    cond_pct,
    one_map,
):
    cond_valid_base = np.asarray(cond_valid, dtype=bool)
    cond_zyxs64 = np.asarray(cond_zyxs, dtype=np.float64)
    crop_size = tuple(int(v) for v in crop_size)
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    volume_for_crops = _resolve_segment_volume(tgt_segment, volume_scale=volume_scale)
    one_map = one_map if isinstance(one_map, dict) else {}
    one_map_get = one_map.get

    bbox_crops = []

    for bbox_idx, bbox in enumerate(bboxes):
        min_corner, _ = _bbox_to_min_corner_and_bounds_array(bbox)
        min_corner32 = min_corner.astype(np.float32, copy=False)
        vol_crop = _crop_volume_from_min_corner(volume_for_crops, min_corner, crop_size)
        vol_crop = normalize_zscore(vol_crop)

        cond_grid_local = cond_zyxs64 - min_corner[None, None, :]
        cond_grid_valid = cond_valid_base.copy()
        cond_grid_valid &= _grid_in_bounds_mask(cond_grid_local, crop_size)

        cond_uv = uv_cond[cond_grid_valid].astype(np.float64, copy=False)
        cond_world = cond_zyxs[cond_grid_valid].astype(np.float32, copy=False)
        cond_local = cond_grid_local[cond_grid_valid].astype(np.float32, copy=False)
        cond_vox = _points_to_voxels(cond_local, crop_size)
        uv_query = _build_uv_query_from_cond_points(cond_uv, grow_direction, cond_pct)
        uv_query_flat = uv_query.reshape(-1, 2).astype(np.float64, copy=False)

        extrap_uv_list = []
        extrap_world_list = []
        for uv in uv_query_flat:
            uv_key = (int(uv[0]), int(uv[1]))
            pt = one_map_get(uv_key)
            if pt is None:
                continue
            pt_arr = np.asarray(pt)
            if not np.isfinite(pt_arr).all():
                continue
            extrap_uv_list.append((float(uv_key[0]), float(uv_key[1])))
            extrap_world_list.append(pt_arr.astype(np.float32, copy=False))

        if extrap_world_list:
            extrap_uv = np.asarray(extrap_uv_list, dtype=np.float64)
            extrap_world = np.asarray(extrap_world_list, dtype=np.float32)
            extrap_local = extrap_world - min_corner32[None, :]
        else:
            extrap_uv = np.zeros((0, 2), dtype=np.float64)
            extrap_world = np.zeros((0, 3), dtype=np.float32)
            extrap_local = np.zeros((0, 3), dtype=np.float32)
        extrap_vox = _points_to_voxels(extrap_local, crop_size)

        bbox_crops.append(
            {
                "bbox_idx": bbox_idx,
                "bbox": bbox,
                "min_corner": min_corner.astype(np.int64, copy=False),
                "crop_size": crop_size_arr.copy(),
                "volume": vol_crop,
                "cond_vox": cond_vox,
                "extrap_vox": extrap_vox,
                "extrap_uv": extrap_uv.astype(np.int64, copy=False),
                "cond_world_zyx": cond_world,
                "extrap_world_zyx": extrap_world,
                "n_cond": int(cond_uv.shape[0]),
                "n_query": int(uv_query_flat.shape[0]),
                "n_extrap": int(extrap_uv.shape[0]),
            }
        )

    return bbox_crops


def _build_bbox_results_from_crops(bbox_crops):
    bbox_results = []
    for crop in bbox_crops:
        extrap_world = crop["extrap_world_zyx"]
        bbox_results.append(
            {
                "bbox_idx": crop["bbox_idx"],
                "bbox": crop["bbox"],
                "cond_world": crop["cond_world_zyx"],
                "extrap_world": extrap_world,
            }
        )
    return bbox_results


def _run_bbox_displacement_inference(args, bbox_crops, model_state, verbose=True):
    expected_in_channels = int(model_state["expected_in_channels"])
    if expected_in_channels != 3:
        raise RuntimeError(
            f"Only 3-channel models are supported in infer_global_extrap. "
            f"Checkpoint expects in_channels={expected_in_channels}."
        )

    if len(bbox_crops) == 0:
        return []

    batch_size = int(args.batch_size)
    per_crop_fields = []
    use_tta = bool(getattr(args, "tta", True))
    desc = "displacement inference (TTA)" if use_tta else "displacement inference"

    n_batches = (len(bbox_crops) + batch_size - 1) // batch_size
    for batch_start in range(0, len(bbox_crops), batch_size):
        batch = bbox_crops[batch_start:batch_start + batch_size]
        batch_inputs = [
            _build_model_inputs(crop["volume"], crop["cond_vox"], crop["extrap_vox"])
            for crop in batch
        ]
        model_inputs = torch.cat(batch_inputs, dim=0).to(args.device)
        disp_pred = _predict_displacement(args, model_state, model_inputs, use_tta=use_tta)
        if disp_pred is None:
            raise RuntimeError("Model output did not contain 'displacement'.")

        if disp_pred.ndim != 5 or disp_pred.shape[1] < 3:
            raise RuntimeError(f"Unexpected displacement shape: {tuple(disp_pred.shape)}")

        # NumPy conversion does not support BF16 directly in some torch builds.
        disp_pred = (
            disp_pred[:, :3]
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        for i, crop in enumerate(batch):
            per_crop_fields.append(
                {
                    "bbox_idx": int(crop["bbox_idx"]),
                    "min_corner": np.asarray(crop["min_corner"], dtype=np.int64),
                    "displacement": disp_pred[i],
                }
            )
        done = (batch_start // batch_size) + 1
        if verbose:
            print(f"{desc}: batch {done}/{n_batches}")

    return per_crop_fields


def _stack_displacements_to_global(per_crop_fields):
    if len(per_crop_fields) == 0:
        return None

    min_corners = np.stack([item["min_corner"] for item in per_crop_fields], axis=0).astype(np.int64, copy=False)
    max_exclusive = []
    for item in per_crop_fields:
        min_corner = item["min_corner"]
        disp = np.asarray(item["displacement"])
        _, d, h, w = disp.shape
        max_exclusive.append(min_corner + np.asarray([d, h, w], dtype=np.int64))
    max_exclusive = np.stack(max_exclusive, axis=0).astype(np.int64, copy=False)

    global_min = min_corners.min(axis=0)
    global_max_exclusive = max_exclusive.max(axis=0)
    global_shape = tuple((global_max_exclusive - global_min).tolist())
    if any(v <= 0 for v in global_shape):
        return None

    disp_sum = np.zeros((3,) + global_shape, dtype=np.float32)
    disp_count = np.zeros(global_shape, dtype=np.uint32)

    for item in per_crop_fields:
        min_corner = item["min_corner"]
        disp = np.asarray(item["displacement"], dtype=np.float32)
        _, d, h, w = disp.shape
        start = (min_corner - global_min).astype(np.int64)
        z0, y0, x0 = int(start[0]), int(start[1]), int(start[2])
        z1, y1, x1 = z0 + int(d), y0 + int(h), x0 + int(w)

        disp_block = disp_sum[:, z0:z1, y0:y1, x0:x1]
        count_block = disp_count[z0:z1, y0:y1, x0:x1]

        finite_mask = np.isfinite(disp).all(axis=0)
        if finite_mask.all():
            disp_block += disp
            count_block += 1
        else:
            disp_block[:, finite_mask] += disp[:, finite_mask]
            count_block[finite_mask] += 1

    disp_global = np.zeros_like(disp_sum, dtype=np.float32)
    valid = disp_count > 0
    if valid.any():
        disp_global[:, valid] = disp_sum[:, valid] / disp_count[valid]

    bbox_inclusive = (
        int(global_min[0]),
        int(global_max_exclusive[0] - 1),
        int(global_min[1]),
        int(global_max_exclusive[1] - 1),
        int(global_min[2]),
        int(global_max_exclusive[2] - 1),
    )
    return {
        "displacement": disp_global,
        "count": disp_count,
        "min_corner": global_min.astype(np.int64, copy=False),
        "shape": np.asarray(global_shape, dtype=np.int64),
        "bbox": bbox_inclusive,
    }


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


def _empty_stack_samples():
    return {
        "uv": np.zeros((0, 2), dtype=np.int64),
        "world": np.zeros((0, 3), dtype=np.float32),
        "displacement": np.zeros((0, 3), dtype=np.float32),
        "stack_count": np.zeros((0,), dtype=np.uint32),
    }


def _agg_extrap_axis_metadata(grow_direction):
    axis_idx = 1 if grow_direction in {"left", "right"} else 0
    axis_name = "col" if axis_idx == 1 else "row"
    _, growth_spec = _get_growth_context(grow_direction)
    near_to_far_desc = int(growth_spec["growth_sign"]) < 0
    return axis_idx, axis_name, near_to_far_desc


def _finite_uv_from_one_map(one_map):
    if not one_map:
        return np.zeros((0, 2), dtype=np.int64)
    finite_uv = []
    for uv_key, pt in one_map.items():
        if pt is None:
            continue
        pt_arr = np.asarray(pt)
        if pt_arr.shape == (3,) and np.isfinite(pt_arr).all():
            finite_uv.append((int(uv_key[0]), int(uv_key[1])))
    if len(finite_uv) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(finite_uv, dtype=np.int64).reshape(-1, 2)


def _select_agg_extrap_uv_for_sampling(one_map, grow_direction, max_lines=None):
    uv_ordered = _finite_uv_from_one_map(one_map)
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
    axis_values = np.sort(axis_values)
    if near_to_far_desc:
        axis_values = axis_values[::-1]
    selected_axis_values = axis_values[: int(max_lines)]
    keep = np.isin(uv_ordered[:, axis_idx], selected_axis_values, assume_unique=False)
    return uv_ordered[keep]


def _sample_stacked_displacement_on_agg_extrap(
    stacked_displacement,
    one_map,
    grow_direction,
    max_lines=None,
):
    if stacked_displacement is None or not one_map:
        return _empty_stack_samples()

    sampled_uv = _select_agg_extrap_uv_for_sampling(one_map, grow_direction, max_lines=max_lines)
    if sampled_uv.shape[0] == 0:
        return _empty_stack_samples()

    disp = np.asarray(stacked_displacement["displacement"], dtype=np.float32)
    count = np.asarray(stacked_displacement["count"], dtype=np.uint32)
    min_corner = np.asarray(stacked_displacement["min_corner"], dtype=np.int64)
    shape = np.asarray(stacked_displacement["shape"], dtype=np.int64)

    sampled_world = np.asarray(
        [one_map[(int(uv[0]), int(uv[1]))] for uv in sampled_uv],
        dtype=np.float32,
    )
    coords_local = sampled_world.astype(np.float64, copy=False) - min_corner[None, :].astype(np.float64)
    in_bounds = (
        (coords_local[:, 0] >= 0.0) & (coords_local[:, 0] <= float(shape[0] - 1)) &
        (coords_local[:, 1] >= 0.0) & (coords_local[:, 1] <= float(shape[1] - 1)) &
        (coords_local[:, 2] >= 0.0) & (coords_local[:, 2] <= float(shape[2] - 1))
    )
    if not in_bounds.any():
        return _empty_stack_samples()

    sampled_uv = sampled_uv[in_bounds]
    sampled_world = sampled_world[in_bounds]
    coords_local = coords_local[in_bounds].astype(np.float32, copy=False)

    sampled_disp, sampled_count, valid_mask = _sample_trilinear_displacement_stack(
        disp,
        count,
        coords_local,
    )
    if not valid_mask.any():
        return _empty_stack_samples()

    sampled_uv = sampled_uv[valid_mask]
    sampled_world = sampled_world[valid_mask]
    sampled_disp = sampled_disp[valid_mask]
    sampled_count = sampled_count[valid_mask]

    return {
        "uv": sampled_uv.astype(np.int64, copy=False),
        "world": sampled_world.astype(np.float32, copy=False),
        "displacement": sampled_disp.astype(np.float32, copy=False),
        "stack_count": sampled_count.astype(np.uint32, copy=False),
    }


def _sample_agg_extrap_direct(one_map, grow_direction, max_lines=None):
    if not one_map:
        return _empty_stack_samples()

    sampled_uv = _select_agg_extrap_uv_for_sampling(one_map, grow_direction, max_lines=max_lines)
    if sampled_uv.shape[0] == 0:
        return _empty_stack_samples()

    sampled_world = np.asarray(
        [one_map[(int(uv[0]), int(uv[1]))] for uv in sampled_uv],
        dtype=np.float32,
    )
    keep = np.isfinite(sampled_world).all(axis=1)
    if not keep.any():
        return _empty_stack_samples()

    sampled_uv = sampled_uv[keep].astype(np.int64, copy=False)
    sampled_world = sampled_world[keep].astype(np.float32, copy=False)
    n = int(sampled_world.shape[0])
    return {
        "uv": sampled_uv,
        "world": sampled_world,
        "displacement": np.zeros((n, 3), dtype=np.float32),
        "stack_count": np.ones((n,), dtype=np.uint32),
    }


def _sample_trilinear_displacement_stack(disp, count, coords_local):
    if coords_local is None or len(coords_local) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0,), dtype=bool),
        )

    disp_t = torch.from_numpy(np.asarray(disp, dtype=np.float32)).unsqueeze(0)  # [1, 3, D, H, W]
    count_t = torch.from_numpy(np.asarray(count, dtype=np.float32)).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    valid_t = (count_t > 0).to(dtype=disp_t.dtype)
    disp_weighted_t = disp_t * valid_t

    coords_t = torch.from_numpy(np.asarray(coords_local, dtype=np.float32)).clone()
    _, _, d, h, w = disp_t.shape
    d_denom = max(int(d) - 1, 1)
    h_denom = max(int(h) - 1, 1)
    w_denom = max(int(w) - 1, 1)
    coords_t[:, 0] = 2.0 * coords_t[:, 0] / float(d_denom) - 1.0
    coords_t[:, 1] = 2.0 * coords_t[:, 1] / float(h_denom) - 1.0
    coords_t[:, 2] = 2.0 * coords_t[:, 2] / float(w_denom) - 1.0
    grid = coords_t[:, [2, 1, 0]].view(1, -1, 1, 1, 3)

    sampled_disp_weighted = F.grid_sample(
        disp_weighted_t,
        grid,
        mode="bilinear",
        align_corners=True,
    ).view(3, -1).permute(1, 0)
    sampled_valid_weight = F.grid_sample(
        valid_t,
        grid,
        mode="bilinear",
        align_corners=True,
    ).view(-1)
    sampled_count = F.grid_sample(
        count_t,
        grid,
        mode="bilinear",
        align_corners=True,
    ).view(-1)

    eps = 1e-6
    valid_mask = sampled_valid_weight > eps
    sampled_disp = torch.zeros_like(sampled_disp_weighted)
    if bool(valid_mask.any()):
        sampled_disp[valid_mask] = (
            sampled_disp_weighted[valid_mask] /
            sampled_valid_weight[valid_mask].unsqueeze(-1)
        )

    sampled_disp_np = sampled_disp.cpu().numpy().astype(np.float32, copy=False)
    sampled_count_np = np.rint(sampled_count.cpu().numpy()).astype(np.uint32, copy=False)
    valid_mask_np = valid_mask.cpu().numpy().astype(bool, copy=False)
    return sampled_disp_np, sampled_count_np, valid_mask_np


def _print_agg_extrap_sampling_debug(samples, one_map, grow_direction, max_lines=None, verbose=True):
    if not verbose:
        return
    all_uv = _select_agg_extrap_uv_for_sampling(one_map, grow_direction, max_lines=None)
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


def _filter_samples_min_stack_count(samples, min_stack_count=1):
    uv = np.asarray(samples.get("uv", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    world = np.asarray(samples.get("world", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    displacement = np.asarray(samples.get("displacement", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    stack_count = np.asarray(samples.get("stack_count", np.zeros((0,), dtype=np.uint32)), dtype=np.uint32)

    n_input = int(min(uv.shape[0], world.shape[0], displacement.shape[0], stack_count.shape[0]))
    if n_input == 0:
        return {
            "uv": np.zeros((0, 2), dtype=np.int64),
            "world": np.zeros((0, 3), dtype=np.float32),
            "displacement": np.zeros((0, 3), dtype=np.float32),
            "stack_count": np.zeros((0,), dtype=np.uint32),
        }, {"n_input": 0, "n_after_stack_count": 0}

    uv = uv[:n_input]
    world = world[:n_input]
    displacement = displacement[:n_input]
    stack_count = stack_count[:n_input]
    min_stack_count = max(1, int(min_stack_count))
    keep = stack_count >= np.uint32(min_stack_count)
    n_after_stack_count = int(keep.sum())
    return {
        "uv": uv[keep].astype(np.int64, copy=False),
        "world": world[keep].astype(np.float32, copy=False),
        "displacement": displacement[keep].astype(np.float32, copy=False),
        "stack_count": stack_count[keep].astype(np.uint32, copy=False),
    }, {"n_input": n_input, "n_after_stack_count": n_after_stack_count}


def _print_sample_filter_debug(filter_stats, verbose=True):
    if not verbose:
        return
    print("== Sample Filter ==")
    n_input = int(filter_stats.get("n_input", 0))
    n_after_stack_count = int(filter_stats.get("n_after_stack_count", 0))
    print(f"input samples: {n_input}")
    print(f"after min-stack-count: {n_after_stack_count}")


def _apply_displacement_and_print_stats(samples, verbose=True):
    world = np.asarray(samples.get("world", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    displacement = np.asarray(samples.get("displacement", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    uv = np.asarray(samples.get("uv", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    stack_count = np.asarray(samples.get("stack_count", np.zeros((0,), dtype=np.uint32)), dtype=np.uint32)

    if world.shape[0] == 0 or displacement.shape[0] == 0:
        if verbose:
            print("== Applied Displacement ==")
            print("No sampled points to apply displacement.")
        return {
            "uv": np.zeros((0, 2), dtype=np.int64),
            "world": np.zeros((0, 3), dtype=np.float32),
            "displacement": np.zeros((0, 3), dtype=np.float32),
            "world_displaced": np.zeros((0, 3), dtype=np.float32),
            "stack_count": np.zeros((0,), dtype=np.uint32),
        }

    world_displaced = world + displacement
    disp_norm = np.linalg.norm(displacement.astype(np.float64), axis=1)

    if verbose:
        print("== Applied Displacement ==")
        print(f"n points: {world.shape[0]}")
        print(
            "disp_norm min/max/mean/median: "
            f"{disp_norm.min():.4f} / {disp_norm.max():.4f} / {disp_norm.mean():.4f} / {np.median(disp_norm):.4f}"
        )

        axis_names = ("z", "y", "x")
        for axis_idx, axis_name in enumerate(axis_names):
            vals = world_displaced[:, axis_idx].astype(np.float64, copy=False)
            print(
                f"{axis_name} min/max/mean/median: "
                f"{vals.min():.4f} / {vals.max():.4f} / {vals.mean():.4f} / {np.median(vals):.4f}"
            )

    return {
        "uv": uv,
        "world": world,
        "displacement": displacement,
        "world_displaced": world_displaced.astype(np.float32, copy=False),
        "stack_count": stack_count,
    }


def _use_extrap_points_directly(samples, verbose=True):
    uv = np.asarray(samples.get("uv", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    world = np.asarray(samples.get("world", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    stack_count = np.asarray(samples.get("stack_count", np.zeros((0,), dtype=np.uint32)), dtype=np.uint32)

    n = int(min(uv.shape[0], world.shape[0], stack_count.shape[0]))
    if n == 0:
        if verbose:
            print("== Applied Displacement ==")
            print("Extrap-only mode: no extrapolated points selected to merge.")
        return {
            "uv": np.zeros((0, 2), dtype=np.int64),
            "world": np.zeros((0, 3), dtype=np.float32),
            "displacement": np.zeros((0, 3), dtype=np.float32),
            "world_displaced": np.zeros((0, 3), dtype=np.float32),
            "stack_count": np.zeros((0,), dtype=np.uint32),
        }

    uv = uv[:n].astype(np.int64, copy=False)
    world = world[:n].astype(np.float32, copy=False)
    stack_count = stack_count[:n].astype(np.uint32, copy=False)
    if verbose:
        print("== Applied Displacement ==")
        print("Extrap-only mode: bypassed displacement sampling; merging extrapolated points directly.")
        print(f"n points: {n}")
    return {
        "uv": uv,
        "world": world,
        "displacement": np.zeros((n, 3), dtype=np.float32),
        "world_displaced": world,
        "stack_count": stack_count,
    }


def _merge_displaced_points_into_full_surface(cond_zyxs, cond_valid, cond_uv_offset, displaced, verbose=True):
    merged_zyxs = np.asarray(cond_zyxs, dtype=np.float32).copy()
    merged_valid = np.asarray(cond_valid, dtype=bool).copy()

    uv = np.asarray(displaced.get("uv", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    pts = np.asarray(displaced.get("world_displaced", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)

    if uv.shape[0] == 0 or pts.shape[0] == 0:
        if verbose:
            print("== Merge Displaced Into Full Surface ==")
            print("No displaced points to merge.")
        return {
            "merged_zyxs": merged_zyxs,
            "merged_valid": merged_valid,
            "uv_offset": (int(cond_uv_offset[0]), int(cond_uv_offset[1])),
            "n_written": 0,
            "n_new_valid": 0,
            "n_overwrite_existing": 0,
            "n_out_of_bounds": 0,
            "n_nonfinite": 0,
        }

    r0, c0 = int(cond_uv_offset[0]), int(cond_uv_offset[1])
    h, w = merged_valid.shape

    n = min(int(uv.shape[0]), int(pts.shape[0]))
    uv_n = uv[:n].astype(np.int64, copy=False)
    pts_n = pts[:n].astype(np.float32, copy=False)
    if uv_n.size > 0:
        min_r = min(r0, int(uv_n[:, 0].min()))
        max_r = max(r0 + h - 1, int(uv_n[:, 0].max()))
        min_c = min(c0, int(uv_n[:, 1].min()))
        max_c = max(c0 + w - 1, int(uv_n[:, 1].max()))
        new_h = max_r - min_r + 1
        new_w = max_c - min_c + 1
        if new_h != h or new_w != w:
            expanded_zyxs = np.full((new_h, new_w, 3), -1.0, dtype=np.float32)
            expanded_valid = np.zeros((new_h, new_w), dtype=bool)

            rr0 = r0 - min_r
            cc0 = c0 - min_c
            expanded_zyxs[rr0:rr0 + h, cc0:cc0 + w] = merged_zyxs
            expanded_valid[rr0:rr0 + h, cc0:cc0 + w] = merged_valid

            merged_zyxs = expanded_zyxs
            merged_valid = expanded_valid
            r0, c0 = min_r, min_c
            h, w = new_h, new_w
            if verbose:
                print("== Merge Displaced Into Full Surface ==")
                print(
                    f"expanded UV canvas: shape ({cond_valid.shape[0]}, {cond_valid.shape[1]}) -> ({h}, {w}), "
                    f"offset ({int(cond_uv_offset[0])}, {int(cond_uv_offset[1])}) -> ({r0}, {c0})"
                )

    n_written = 0
    n_new_valid = 0
    n_overwrite_existing = 0
    if n > 0:
        finite_mask = np.isfinite(pts_n).all(axis=1)
        rr_all = uv_n[:, 0] - r0
        cc_all = uv_n[:, 1] - c0
        in_bounds_mask = (
            finite_mask &
            (rr_all >= 0) &
            (rr_all < h) &
            (cc_all >= 0) &
            (cc_all < w)
        )
        n_nonfinite = int((~finite_mask).sum())
        n_out_of_bounds = int((finite_mask & ~in_bounds_mask).sum())
        write_indices = np.nonzero(in_bounds_mask)[0]
    else:
        rr_all = np.zeros((0,), dtype=np.int64)
        cc_all = np.zeros((0,), dtype=np.int64)
        n_nonfinite = 0
        n_out_of_bounds = 0
        write_indices = np.zeros((0,), dtype=np.int64)

    for i in write_indices:
        rr = int(rr_all[i])
        cc = int(cc_all[i])
        if merged_valid[rr, cc]:
            n_overwrite_existing += 1
        else:
            n_new_valid += 1

        merged_zyxs[rr, cc] = pts_n[i]
        merged_valid[rr, cc] = True
        n_written += 1

    if verbose:
        print("== Merge Displaced Into Full Surface ==")
        print(f"written points: {n_written}")
        print(f"new valid points: {n_new_valid}")
        print(f"overwrote existing valid points: {n_overwrite_existing}")
        print(f"skipped out-of-bounds points: {n_out_of_bounds}")
        print(f"skipped nonfinite points: {n_nonfinite}")

    return {
        "merged_zyxs": merged_zyxs,
        "merged_valid": merged_valid,
        "uv_offset": (r0, c0),
        "n_written": int(n_written),
        "n_new_valid": int(n_new_valid),
        "n_overwrite_existing": int(n_overwrite_existing),
        "n_out_of_bounds": int(n_out_of_bounds),
        "n_nonfinite": int(n_nonfinite),
    }


def _save_merged_surface_tifxyz(args, merged, checkpoint_path, model_config, call_args):
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
    stride_y = max(1, int(round(float(tifxyz_step_size) / max(1, current_step))))
    stride_x = max(1, int(round(float(tifxyz_step_size) / max(1, current_step))))
    if stride_y > 1 or stride_x > 1:
        merged_for_save = merged_for_save[::stride_y, ::stride_x]

    out_dir = args.tifxyz_out_dir if args.tifxyz_out_dir else str(Path(args.tifxyz_path).parent)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ckpt_name = "no_ckpt" if checkpoint_path is None else os.path.splitext(os.path.basename(str(checkpoint_path)))[0]
    timestamp = datetime.now().strftime("%H%M%S")
    tifxyz_uuid = f"displacement_{ckpt_name}_{timestamp}"

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
            "agg_extrap_min_stack_count": int(args.agg_extrap_min_stack_count),
            "run_argv": list(sys.argv[1:]),
            "run_args": _json_safe(call_args),
        },
    )

    output_path = os.path.join(out_dir, tifxyz_uuid)
    print(f"Saved tifxyz to {output_path}")
    return output_path


def _boundary_axis_value(valid, uv_offset, grow_direction):
    valid = np.asarray(valid, dtype=bool)
    rows, cols = np.where(valid)
    if rows.size == 0:
        return None
    _, growth_spec = _get_growth_context(grow_direction)
    r0, c0 = int(uv_offset[0]), int(uv_offset[1])
    if growth_spec["axis"] == "row":
        axis_vals = rows.astype(np.int64) + r0
    else:
        axis_vals = cols.astype(np.int64) + c0
    if growth_spec["growth_sign"] > 0:
        return int(axis_vals.max())
    return int(axis_vals.min())


def _boundary_advanced(prev_boundary, next_boundary, grow_direction):
    if prev_boundary is None or next_boundary is None:
        return False
    _, growth_spec = _get_growth_context(grow_direction)
    if growth_spec["growth_sign"] > 0:
        return int(next_boundary) > int(prev_boundary)
    return int(next_boundary) < int(prev_boundary)


def _surface_to_uv_samples(grid, valid, uv_offset):
    rows, cols = np.where(np.asarray(valid, dtype=bool))
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    r0, c0 = int(uv_offset[0]), int(uv_offset[1])
    uv = np.stack(
        [
            rows.astype(np.int64, copy=False) + r0,
            cols.astype(np.int64, copy=False) + c0,
        ],
        axis=-1,
    )
    pts = np.asarray(grid, dtype=np.float32)[rows, cols].astype(np.float32, copy=False)
    keep = np.isfinite(pts).all(axis=1)
    return uv[keep], pts[keep]


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


def _grid_to_uv_world_dict(grid, valid, offset):
    rows, cols = np.where(valid)
    if rows.size == 0:
        return {}
    rows_abs = rows.astype(np.int64) + int(offset[0])
    cols_abs = cols.astype(np.int64) + int(offset[1])
    pts = np.asarray(grid)[rows, cols]
    finite_mask = np.isfinite(pts).all(axis=1)
    if not finite_mask.any():
        return {}
    rows_abs = rows_abs[finite_mask]
    cols_abs = cols_abs[finite_mask]
    pts = pts[finite_mask].astype(np.float32, copy=False)
    return {
        (int(rows_abs[i]), int(cols_abs[i])): pts[i]
        for i in range(rows_abs.shape[0])
    }


def _agg_extrap_line_summary(one_map, grow_direction):
    uv_ordered = _select_agg_extrap_uv_for_sampling(one_map, grow_direction, max_lines=None)
    axis_idx, axis_name, _ = _agg_extrap_axis_metadata(grow_direction)
    if uv_ordered.shape[0] == 0:
        return {
            "axis_name": axis_name,
            "uv_count": 0,
            "line_count": 0,
            "near_axis_value": None,
            "far_axis_value": None,
        }

    axis_values = []
    seen = set()
    for value in uv_ordered[:, axis_idx]:
        int_value = int(value)
        if int_value in seen:
            continue
        seen.add(int_value)
        axis_values.append(int_value)

    return {
        "axis_name": axis_name,
        "uv_count": int(uv_ordered.shape[0]),
        "line_count": int(len(axis_values)),
        "near_axis_value": axis_values[0] if axis_values else None,
        "far_axis_value": axis_values[-1] if axis_values else None,
    }


def _print_iteration_summary(bbox_results, one_shot, one_map, grow_direction, verbose=True):
    if not verbose:
        return
    agg_summary = _agg_extrap_line_summary(one_map, grow_direction)
    print("== Extrapolation Summary ==")
    print(f"bboxes: {len(bbox_results)}")
    print(f"one-shot edge-input uv count: {len(one_shot.get('edge_uv', []))}")
    print(f"one-shot extrap uv count (aggregated): {len(one_map)}")
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
    one_map,
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

    if one_map:
        one_pts = np.asarray(list(one_map.values()), dtype=np.float32)
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


def main():
    args = parse_args()
    call_args = _serialize_args(args)
    crop_size = tuple(int(v) for v in args.crop_size)

    extrap_only_mode = bool(args.extrap_only)
    run_model_inference = (
        (args.checkpoint_path is not None)
        and (not args.skip_inference)
        and (not extrap_only_mode)
    )
    model_state = None
    model_config = None
    checkpoint_path = None
    if run_model_inference:
        model_state = load_model(args)
        model_config = model_state["model_config"]
        checkpoint_path = model_state["checkpoint_path"]
        expected_in_channels = int(model_state["expected_in_channels"])
        if expected_in_channels != 3:
            raise RuntimeError(
                f"infer_global_extrap only supports 3-channel models; got in_channels={expected_in_channels}"
            )
    elif extrap_only_mode:
        if args.verbose:
            print("Running extrap-only iterative mode (--extrap-only set): skipping inference and stack sampling.")
        if args.agg_extrap_min_stack_count != 1:
            if args.verbose:
                print("--agg-extrap-min-stack-count is ignored in --extrap-only mode.")
        if args.checkpoint_path is not None:
            model_config, checkpoint_path = load_checkpoint_config(args.checkpoint_path)
    elif args.skip_inference:
        if args.verbose:
            print("Skipping displacement inference (--skip-inference set).")
        if args.checkpoint_path is not None:
            model_config, checkpoint_path = load_checkpoint_config(args.checkpoint_path)
    else:
        if args.verbose:
            print("Skipping displacement inference (no --checkpoint-path provided).")
        if args.checkpoint_path is not None:
            model_config, checkpoint_path = load_checkpoint_config(args.checkpoint_path)

    extrapolation_settings = _resolve_settings(args, model_config=model_config)

    volume = zarr.open_group(args.volume_path, mode="r")
    tgt_segment, stored_zyxs, valid_s, grow_direction, h_s, w_s = setup_segment(args, volume)
    cond_direction, _ = _get_growth_context(grow_direction)

    if int(args.iterations) > 1:
        tgt_segment.use_full_resolution()
        base_x, base_y, base_z, base_valid = tgt_segment[:]
        current_zyxs = np.stack([base_z, base_y, base_x], axis=-1).copy()
        current_valid = np.asarray(base_valid, dtype=bool).copy()
        current_uv_offset = (0, 0)
        if args.verbose:
            print("Using full input surface as initial conditioning for iterative growth.")
    else:
        r0_s, r1_s, c0_s, c1_s = compute_window_and_split(
            args, stored_zyxs, valid_s, grow_direction, h_s, w_s, crop_size
        )
        full_bounds = _stored_to_full_bounds(tgt_segment, (r0_s, r1_s, c0_s, c1_s))
        current_zyxs, current_valid, current_uv_offset = _initialize_window_state(tgt_segment, full_bounds)

    bbox_results = []
    one_shot = {
        "edge_uv": np.zeros((0, 2), dtype=np.float64),
        "edge_zyx": np.zeros((0, 3), dtype=np.float32),
        "uv_query_flat": np.zeros((0, 2), dtype=np.float64),
        "extrap_coords_world": np.zeros((0, 3), dtype=np.float32),
    }
    one_map = {}
    stacked_displacement = None
    displaced = {
        "uv": np.zeros((0, 2), dtype=np.int64),
        "world": np.zeros((0, 3), dtype=np.float32),
        "displacement": np.zeros((0, 3), dtype=np.float32),
        "world_displaced": np.zeros((0, 3), dtype=np.float32),
        "stack_count": np.zeros((0,), dtype=np.uint32),
    }

    n_iterations = int(args.iterations)
    iteration_pbar = None
    if args.verbose:
        iteration_iter = range(n_iterations)
    else:
        iteration_pbar = tqdm(range(n_iterations), total=n_iterations, desc="iterations", unit="iter")
        iteration_iter = iteration_pbar

    for iteration in iteration_iter:
        if args.verbose:
            print(f"[iteration {iteration + 1}/{n_iterations}]")
        cond_zyxs = current_zyxs
        cond_valid = current_valid
        cond_uv_offset = current_uv_offset
        uv_cond = _build_uv_grid(cond_uv_offset, cond_zyxs.shape[:2])

        bboxes, _ = get_cond_edge_bboxes(
            cond_zyxs,
            cond_direction,
            crop_size,
            overlap_frac=args.bbox_overlap_frac,
        )
        if len(bboxes) == 0:
            if args.verbose:
                print("No edge bboxes available at current boundary; stopping iterative growth.")
            elif iteration_pbar is not None:
                iteration_pbar.set_postfix_str("stopped: no edge bboxes", refresh=True)
            break

        one_shot = compute_edge_one_shot_extrapolation(
            cond_zyxs=cond_zyxs,
            cond_valid=cond_valid,
            uv_cond=uv_cond,
            grow_direction=grow_direction,
            edge_input_rowscols=args.edge_input_rowscols,
            cond_pct=args.cond_pct,
            method=extrapolation_settings["method"],
            min_corner=np.zeros(3, dtype=np.float64),
            crop_size=crop_size,
            degrade_prob=extrapolation_settings["degrade_prob"],
            degrade_curvature_range=extrapolation_settings["degrade_curvature_range"],
            degrade_gradient_range=extrapolation_settings["degrade_gradient_range"],
            skip_bounds_check=True,
            **extrapolation_settings["method_kwargs"],
        )
        if one_shot is None:
            one_shot = {
                "edge_uv": np.zeros((0, 2), dtype=np.float64),
                "edge_zyx": np.zeros((0, 3), dtype=np.float32),
                "uv_query_flat": np.zeros((0, 2), dtype=np.float64),
                "extrap_coords_world": np.zeros((0, 3), dtype=np.float32),
            }

        one_uv, one_world = _finite_uv_world(one_shot.get("uv_query_flat"), one_shot.get("extrap_coords_world"))
        one_samples = [(one_uv, one_world)] if len(one_uv) > 0 else []
        one_grid, one_valid, one_offset = _aggregate_pred_samples_to_uv_grid(one_samples)
        one_map = _grid_to_uv_world_dict(one_grid, one_valid, one_offset)

        bbox_crops = _build_bbox_crops(
            bboxes=bboxes,
            tgt_segment=tgt_segment,
            volume_scale=args.volume_scale,
            cond_zyxs=cond_zyxs,
            cond_valid=cond_valid,
            uv_cond=uv_cond,
            grow_direction=grow_direction,
            crop_size=crop_size,
            cond_pct=args.cond_pct,
            one_map=one_map,
        )
        _print_bbox_crop_debug_table(bbox_crops, verbose=args.verbose)
        bbox_results = _build_bbox_results_from_crops(bbox_crops)

        stacked_displacement = None
        if run_model_inference and model_state is not None:
            per_crop_fields = _run_bbox_displacement_inference(
                args,
                bbox_crops,
                model_state,
                verbose=args.verbose,
            )
            stacked_displacement = _stack_displacements_to_global(per_crop_fields)
        _print_stacked_displacement_debug(stacked_displacement, verbose=args.verbose)
        if extrap_only_mode:
            agg_samples = _sample_agg_extrap_direct(
                one_map,
                grow_direction,
                max_lines=args.agg_extrap_lines,
            )
        else:
            agg_samples = _sample_stacked_displacement_on_agg_extrap(
                stacked_displacement,
                one_map,
                grow_direction,
                max_lines=args.agg_extrap_lines,
            )

        _print_iteration_summary(
            bbox_results,
            one_shot,
            one_map,
            grow_direction,
            verbose=args.verbose,
        )
        _print_agg_extrap_sampling_debug(
            agg_samples,
            one_map,
            grow_direction,
            max_lines=args.agg_extrap_lines,
            verbose=args.verbose,
        )
        if extrap_only_mode:
            sample_filter_stats = {
                "n_input": int(np.asarray(agg_samples.get("uv", np.zeros((0, 2), dtype=np.int64))).shape[0]),
                "n_after_stack_count": int(np.asarray(agg_samples.get("uv", np.zeros((0, 2), dtype=np.int64))).shape[0]),
            }
        else:
            agg_samples, sample_filter_stats = _filter_samples_min_stack_count(
                agg_samples,
                min_stack_count=args.agg_extrap_min_stack_count,
            )
        _print_sample_filter_debug(sample_filter_stats, verbose=args.verbose)
        if extrap_only_mode:
            displaced = _use_extrap_points_directly(agg_samples, verbose=args.verbose)
        else:
            displaced = _apply_displacement_and_print_stats(agg_samples, verbose=args.verbose)

        prev_boundary = _boundary_axis_value(cond_valid, cond_uv_offset, grow_direction)
        merged_iter = _merge_displaced_points_into_full_surface(
            cond_zyxs,
            cond_valid,
            cond_uv_offset,
            displaced,
            verbose=args.verbose,
        )
        next_boundary = _boundary_axis_value(
            merged_iter["merged_valid"],
            merged_iter["uv_offset"],
            grow_direction,
        )
        current_zyxs = merged_iter["merged_zyxs"]
        current_valid = merged_iter["merged_valid"]
        current_uv_offset = merged_iter["uv_offset"]

        if int(merged_iter.get("n_new_valid", 0)) < 1:
            if args.verbose:
                print("No newly added valid points this iteration; stopping iterative growth.")
            elif iteration_pbar is not None:
                iteration_pbar.set_postfix_str("stopped: no new valid points", refresh=True)
            break
        if not _boundary_advanced(prev_boundary, next_boundary, grow_direction):
            if args.verbose:
                print("Boundary did not advance this iteration; stopping iterative growth.")
            elif iteration_pbar is not None:
                iteration_pbar.set_postfix_str("stopped: boundary unchanged", refresh=True)
            break

    if iteration_pbar is not None:
        iteration_pbar.close()

    # Merge the full iteratively grown surface onto the full input canvas so
    # output preserves all existing input points and includes all growth steps.
    grown_uv, grown_pts = _surface_to_uv_samples(current_zyxs, current_valid, current_uv_offset)
    tgt_segment.use_full_resolution()
    base_x, base_y, base_z, base_valid = tgt_segment[:]
    base_zyxs = np.stack([base_z, base_y, base_x], axis=-1)
    merged = _merge_displaced_points_into_full_surface(
        base_zyxs,
        base_valid,
        (0, 0),
        {
            "uv": grown_uv,
            "world_displaced": grown_pts,
        },
        verbose=args.verbose,
    )
    _save_merged_surface_tifxyz(args, merged, checkpoint_path, model_config, call_args)

    if args.napari:
        disp_bbox = None if stacked_displacement is None else stacked_displacement["bbox"]
        _show_napari(
            current_zyxs,
            current_valid,
            bbox_results,
            one_shot,
            one_map,
            disp_bbox=disp_bbox,
            displaced=displaced,
            merged=merged,
            downsample=args.napari_downsample,
            point_size=args.napari_point_size,
        )


if __name__ == "__main__":
    main()
