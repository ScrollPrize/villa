import argparse
import colorsys
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zarr

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
    compute_extrapolation_infer,
    compute_window_and_split,
    get_cond_edge_bboxes,
    load_model,
    load_checkpoint_config,
    setup_segment,
)
from vesuvius.neural_tracing.tifxyz import save_tifxyz


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
        type=float,
        default=1.25,
        help="Outlier threshold multiplier for dropping inconsistent TTA variants.",
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
        "--safe-band-lines",
        type=int,
        default=None,
        help=(
            "Optional number of contiguous safe-band rows/cols (near->far) to sample from the stacked "
            "displacement. If unset, samples all safe-band lines."
        ),
    )
    parser.add_argument(
        "--safe-line-min-coverage",
        type=float,
        default=0.995,
        help=(
            "Minimum per-line coverage ratio in one-shot support (inside bbox-union coverage) "
            "to accept a safe line "
            "(default: 0.995)."
        ),
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
    if args.safe_band_lines is not None and args.safe_band_lines < 1:
        parser.error("--safe-band-lines must be >= 1 when provided")
    if args.safe_line_min_coverage <= 0.0 or args.safe_line_min_coverage > 1.0:
        parser.error("--safe-line-min-coverage must be in (0, 1].")
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
    extrapolation_settings,
    cond_pct,
):
    cond_direction, _ = _get_growth_context(grow_direction)
    cond_valid_base = np.asarray(cond_valid, dtype=bool)
    crop_size = tuple(int(v) for v in crop_size)
    volume_for_crops = _resolve_segment_volume(tgt_segment, volume_scale=volume_scale)

    bbox_crops = []

    for bbox_idx, bbox in enumerate(bboxes):
        min_corner, _ = _bbox_to_min_corner_and_bounds_array(bbox)
        vol_crop = _crop_volume_from_min_corner(volume_for_crops, min_corner, crop_size)
        vol_crop = normalize_zscore(vol_crop)

        cond_grid_local = cond_zyxs.astype(np.float64, copy=False) - min_corner[None, None, :]
        cond_grid_valid = cond_valid_base.copy()
        cond_grid_valid &= _grid_in_bounds_mask(cond_grid_local, crop_size)

        cond_uv = uv_cond[cond_grid_valid].astype(np.float64, copy=False)
        cond_world = cond_zyxs[cond_grid_valid].astype(np.float32, copy=False)
        cond_local = cond_grid_local[cond_grid_valid].astype(np.float32, copy=False)
        cond_vox = _points_to_voxels(cond_local, crop_size)
        uv_query = _build_uv_query_from_cond_points(cond_uv, grow_direction, cond_pct)
        uv_query_flat = uv_query.reshape(-1, 2).astype(np.float64, copy=False)

        extrap_local = np.zeros((0, 3), dtype=np.float32)
        extrap_uv = np.zeros((0, 2), dtype=np.float64)
        extrap_world = np.zeros((0, 3), dtype=np.float32)
        if uv_query.size > 0 and len(cond_uv) > 0:
            extrap_result = compute_extrapolation_infer(
                uv_cond=cond_uv,
                zyx_cond=cond_world,
                uv_query=uv_query,
                min_corner=min_corner,
                crop_size=crop_size,
                method=extrapolation_settings["method"],
                cond_direction=cond_direction,
                degrade_prob=extrapolation_settings["degrade_prob"],
                degrade_curvature_range=extrapolation_settings["degrade_curvature_range"],
                degrade_gradient_range=extrapolation_settings["degrade_gradient_range"],
                skip_bounds_check=True,
                **extrapolation_settings["method_kwargs"],
            )
            if extrap_result is not None:
                extrap_local_full = np.asarray(extrap_result["extrap_coords_local"], dtype=np.float32)
                if extrap_local_full.shape[0] != uv_query_flat.shape[0]:
                    raise ValueError(
                        f"bbox {bbox_idx} extrapolation count mismatch: "
                        f"{extrap_local_full.shape[0]} coords for {uv_query_flat.shape[0]} UV queries"
                    )
                extrap_world_full = extrap_local_full + min_corner[None, :].astype(np.float32)
                extrap_valid = np.isfinite(extrap_world_full).all(axis=1)
                extrap_local = extrap_local_full[extrap_valid].astype(np.float32, copy=False)
                extrap_uv = uv_query_flat[extrap_valid].astype(np.float64, copy=False)
                extrap_world = extrap_world_full[extrap_valid].astype(np.float32, copy=False)
        extrap_vox = _points_to_voxels(extrap_local, crop_size)

        bbox_crops.append(
            {
                "bbox_idx": bbox_idx,
                "bbox": bbox,
                "min_corner": min_corner.astype(np.int64, copy=False),
                "crop_size": np.asarray(crop_size, dtype=np.int64),
                "volume": vol_crop,
                "cond_vox": cond_vox,
                "extrap_vox": extrap_vox,
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


def _run_bbox_displacement_inference(args, bbox_crops, model_state):
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
            for ch in range(3):
                disp_block[ch][finite_mask] += disp[ch][finite_mask]
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


def _print_stacked_displacement_debug(stacked):
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


def _sample_stacked_displacement_on_safe_band(stacked_displacement, one_map, safe_band, max_lines=None):
    if stacked_displacement is None or not one_map or not isinstance(safe_band, dict):
        return {
            "uv": np.zeros((0, 2), dtype=np.int64),
            "world": np.zeros((0, 3), dtype=np.float32),
            "displacement": np.zeros((0, 3), dtype=np.float32),
            "stack_count": np.zeros((0,), dtype=np.uint32),
        }

    safe_uv_ordered = _select_safe_uv_for_sampling(safe_band, max_lines=max_lines)
    if safe_uv_ordered.shape[0] == 0:
        return {
            "uv": np.zeros((0, 2), dtype=np.int64),
            "world": np.zeros((0, 3), dtype=np.float32),
            "displacement": np.zeros((0, 3), dtype=np.float32),
            "stack_count": np.zeros((0,), dtype=np.uint32),
        }

    disp = np.asarray(stacked_displacement["displacement"], dtype=np.float32)
    count = np.asarray(stacked_displacement["count"], dtype=np.uint32)
    min_corner = np.asarray(stacked_displacement["min_corner"], dtype=np.int64)
    shape = np.asarray(stacked_displacement["shape"], dtype=np.int64)

    sampled_uv = []
    sampled_world = []
    sampled_disp = []
    sampled_count = []

    # Sample nearest stack voxel for each safe-band world point.
    for uv_row, uv_col in safe_uv_ordered:
        uv = (int(uv_row), int(uv_col))
        pt = one_map.get(uv)
        if pt is None or not np.isfinite(pt).all():
            continue

        idx = np.rint(np.asarray(pt, dtype=np.float64)).astype(np.int64) - min_corner
        if np.any(idx < 0) or np.any(idx >= shape):
            continue

        z, y, x = int(idx[0]), int(idx[1]), int(idx[2])
        c = count[z, y, x]
        if c == 0:
            continue

        sampled_uv.append((int(uv[0]), int(uv[1])))
        sampled_world.append(np.asarray(pt, dtype=np.float32))
        sampled_disp.append(disp[:, z, y, x].astype(np.float32, copy=False))
        sampled_count.append(int(c))

    if len(sampled_uv) == 0:
        return {
            "uv": np.zeros((0, 2), dtype=np.int64),
            "world": np.zeros((0, 3), dtype=np.float32),
            "displacement": np.zeros((0, 3), dtype=np.float32),
            "stack_count": np.zeros((0,), dtype=np.uint32),
        }

    return {
        "uv": np.asarray(sampled_uv, dtype=np.int64),
        "world": np.asarray(sampled_world, dtype=np.float32),
        "displacement": np.asarray(sampled_disp, dtype=np.float32),
        "stack_count": np.asarray(sampled_count, dtype=np.uint32),
    }


def _select_safe_uv_for_sampling(safe_band, max_lines=None):
    if not isinstance(safe_band, dict):
        return np.zeros((0, 2), dtype=np.int64)

    safe_uv_ordered = np.asarray(
        safe_band.get("safe_uv_ordered", np.zeros((0, 2), dtype=np.int64)),
        dtype=np.int64,
    )
    if safe_uv_ordered.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    safe_uv_ordered = safe_uv_ordered.reshape(-1, 2)
    if max_lines is None:
        return safe_uv_ordered

    safe_axis_values = safe_band.get("safe_axis_values_near_to_far", [])
    axis_name = safe_band.get("axis_name")
    if axis_name not in {"row", "col"} or len(safe_axis_values) == 0:
        return safe_uv_ordered

    selected_axis_values = set(int(v) for v in safe_axis_values[: int(max_lines)])
    if axis_name == "col":
        keep = np.array([int(uv[1]) in selected_axis_values for uv in safe_uv_ordered], dtype=bool)
    else:
        keep = np.array([int(uv[0]) in selected_axis_values for uv in safe_uv_ordered], dtype=bool)
    return safe_uv_ordered[keep]


def _print_safe_band_sampling_debug(samples, safe_band, max_lines=None):
    if isinstance(safe_band, dict):
        safe_uv_ordered = np.asarray(safe_band.get("safe_uv_ordered", np.zeros((0, 2), dtype=np.int64)))
        safe_total = int(safe_uv_ordered.reshape(-1, 2).shape[0]) if safe_uv_ordered.size > 0 else 0
    else:
        safe_total = 0
    safe_lines_total = int(safe_band.get("safe_lines", 0)) if isinstance(safe_band, dict) else 0
    sampled = int(samples["uv"].shape[0])
    print("== Safe Band Stack Sampling ==")
    print(f"sampled safe-band UVs: {sampled}/{safe_total}")
    if max_lines is not None:
        used_lines = min(int(max_lines), safe_lines_total)
        print(f"sampled safe-band lines (near->far): {used_lines}/{safe_lines_total}")


def _apply_displacement_and_print_stats(samples):
    world = np.asarray(samples.get("world", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    displacement = np.asarray(samples.get("displacement", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    uv = np.asarray(samples.get("uv", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    stack_count = np.asarray(samples.get("stack_count", np.zeros((0,), dtype=np.uint32)), dtype=np.uint32)

    if world.shape[0] == 0 or displacement.shape[0] == 0:
        print("== Safe Band Applied Displacement ==")
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

    print("== Safe Band Applied Displacement ==")
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


def _merge_displaced_points_into_full_surface(cond_zyxs, cond_valid, cond_uv_offset, displaced):
    merged_zyxs = np.asarray(cond_zyxs, dtype=np.float32).copy()
    merged_valid = np.asarray(cond_valid, dtype=bool).copy()

    uv = np.asarray(displaced.get("uv", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    pts = np.asarray(displaced.get("world_displaced", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)

    if uv.shape[0] == 0 or pts.shape[0] == 0:
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
    uv_n = uv[:n]
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
            print("== Merge Displaced Into Full Surface ==")
            print(
                f"expanded UV canvas: shape ({cond_valid.shape[0]}, {cond_valid.shape[1]}) -> ({h}, {w}), "
                f"offset ({int(cond_uv_offset[0])}, {int(cond_uv_offset[1])}) -> ({r0}, {c0})"
            )

    n_written = 0
    n_new_valid = 0
    n_overwrite_existing = 0
    n_out_of_bounds = 0
    n_nonfinite = 0

    for i in range(n):
        pt = pts[i]
        if not np.isfinite(pt).all():
            n_nonfinite += 1
            continue

        rr = int(uv[i, 0]) - r0
        cc = int(uv[i, 1]) - c0
        if rr < 0 or rr >= h or cc < 0 or cc >= w:
            n_out_of_bounds += 1
            continue

        if merged_valid[rr, cc]:
            n_overwrite_existing += 1
        else:
            n_new_valid += 1

        merged_zyxs[rr, cc] = pt
        merged_valid[rr, cc] = True
        n_written += 1

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
            "safe_band_lines": None if args.safe_band_lines is None else int(args.safe_band_lines),
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


def _print_bbox_crop_debug_table(bbox_crops):
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
    pts = grid[rows, cols]
    out = {}
    for i in range(rows_abs.shape[0]):
        if np.isfinite(pts[i]).all():
            out[(int(rows_abs[i]), int(cols_abs[i]))] = pts[i].astype(np.float32, copy=False)
    return out


def _compute_safe_band(bbox_results, one_shot, one_map, grow_direction, min_line_coverage=1.0):
    one_uv = np.asarray(one_shot.get("uv_query_flat", np.zeros((0, 2), dtype=np.float64)))
    if one_uv.size == 0:
        return {
            "axis_name": None,
            "total_lines": 0,
            "safe_lines": 0,
            "safe_axis_values_near_to_far": [],
            "safe_uv_ordered": np.zeros((0, 2), dtype=np.int64),
            "near_axis_value": None,
            "farthest_safe_axis_value": None,
            "first_unsafe_axis_value": None,
            "first_unsafe_covered": 0,
            "first_unsafe_total": 0,
        }

    one_uv_i = one_uv.astype(np.int64, copy=False)

    bbox_bounds = []
    for item in bbox_results:
        z_min, z_max, y_min, y_max, x_min, x_max = item["bbox"]
        bbox_bounds.append(
            (
                float(z_min), float(z_max),
                float(y_min), float(y_max),
                float(x_min), float(x_max),
            )
        )
    bbox_bounds = tuple(bbox_bounds)

    def _in_bbox_union(pt):
        z, y, x = float(pt[0]), float(pt[1]), float(pt[2])
        for z_min, z_max, y_min, y_max, x_min, x_max in bbox_bounds:
            if (
                z >= z_min and z <= z_max and
                y >= y_min and y <= y_max and
                x >= x_min and x <= x_max
            ):
                return True
        return False

    # UVs that have finite one-shot predictions inside the 3D union of bbox bounds.
    uv_supported = set()
    for uv_key, pt in one_map.items():
        if np.isfinite(pt).all() and _in_bbox_union(pt):
            uv_supported.add((int(uv_key[0]), int(uv_key[1])))

    axis_idx = 1 if grow_direction in {"left", "right"} else 0
    axis_name = "col" if axis_idx == 1 else "row"
    near_to_far_desc = grow_direction in {"left", "up"}

    line_orth_values = {}
    for row, col in one_uv_i:
        axis_val = int(col) if axis_idx == 1 else int(row)
        orth_val = int(row) if axis_idx == 1 else int(col)
        line_orth_values.setdefault(axis_val, set()).add(orth_val)

    axis_values = sorted(line_orth_values.keys(), reverse=near_to_far_desc)
    if len(axis_values) == 0:
        return {
            "axis_name": axis_name,
            "total_lines": 0,
            "safe_lines": 0,
            "safe_axis_values_near_to_far": [],
            "safe_uv_ordered": np.zeros((0, 2), dtype=np.int64),
            "near_axis_value": None,
            "farthest_safe_axis_value": None,
            "first_unsafe_axis_value": None,
            "first_unsafe_covered": 0,
            "first_unsafe_total": 0,
        }

    safe_axis_values = []
    first_unsafe_axis_value = None
    first_unsafe_covered = 0
    first_unsafe_total = 0

    for axis_val in axis_values:
        orth_values = line_orth_values[axis_val]
        covered = 0
        for orth_val in orth_values:
            uv_key = (orth_val, axis_val) if axis_idx == 1 else (axis_val, orth_val)
            if uv_key in uv_supported:
                covered += 1
        required = int(np.ceil(float(min_line_coverage) * float(len(orth_values))))
        if covered >= max(1, required):
            safe_axis_values.append(axis_val)
        else:
            first_unsafe_axis_value = axis_val
            first_unsafe_covered = int(covered)
            first_unsafe_total = int(len(orth_values))
            break

    safe_uv_set = set()
    for axis_val in safe_axis_values:
        for orth_val in line_orth_values[axis_val]:
            uv_key = (orth_val, axis_val) if axis_idx == 1 else (axis_val, orth_val)
            safe_uv_set.add(uv_key)
    safe_uv_ordered = np.asarray(
        [
            (int(uv_key[0]), int(uv_key[1]))
            for uv_key in one_map.keys()
            if (int(uv_key[0]), int(uv_key[1])) in safe_uv_set
        ],
        dtype=np.int64,
    )
    if safe_uv_ordered.size == 0:
        safe_uv_ordered = np.zeros((0, 2), dtype=np.int64)

    return {
        "axis_name": axis_name,
        "total_lines": int(len(axis_values)),
        "safe_lines": int(len(safe_axis_values)),
        "safe_axis_values_near_to_far": [int(v) for v in safe_axis_values],
        "safe_uv_ordered": safe_uv_ordered,
        "near_axis_value": int(axis_values[0]),
        "farthest_safe_axis_value": None if len(safe_axis_values) == 0 else int(safe_axis_values[-1]),
        "first_unsafe_axis_value": None if first_unsafe_axis_value is None else int(first_unsafe_axis_value),
        "first_unsafe_covered": first_unsafe_covered,
        "first_unsafe_total": first_unsafe_total,
    }


def _print_iteration_summary(bbox_results, one_shot, one_map, safe_band):
    print("== Extrapolation Summary ==")
    print(f"bboxes: {len(bbox_results)}")
    print(f"one-shot edge-input uv count: {len(one_shot.get('edge_uv', []))}")
    print(f"one-shot extrap uv count (aggregated): {len(one_map)}")
    print("== Safe Band ==")
    print(f"safe axis: {safe_band.get('axis_name')}")
    print(f"safe lines (near->far contiguous): {safe_band.get('safe_lines', 0)}/{safe_band.get('total_lines', 0)}")
    safe_uv_ordered = np.asarray(safe_band.get("safe_uv_ordered", np.zeros((0, 2), dtype=np.int64)))
    safe_uv_count = int(safe_uv_ordered.reshape(-1, 2).shape[0]) if safe_uv_ordered.size > 0 else 0
    print(f"safe uv count: {safe_uv_count}")
    near_axis = safe_band.get("near_axis_value")
    far_axis = safe_band.get("farthest_safe_axis_value")
    if near_axis is not None:
        print(f"safe axis range near->far: {near_axis} -> {far_axis}")
    first_unsafe = safe_band.get("first_unsafe_axis_value")
    if first_unsafe is not None:
        covered = int(safe_band.get("first_unsafe_covered", 0))
        total = int(safe_band.get("first_unsafe_total", 0))
        print(f"first unsafe axis value: {first_unsafe} ({covered}/{total} covered)")


def _show_napari(
    cond_zyxs,
    cond_valid,
    bbox_results,
    one_shot,
    one_map,
    safe_band,
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

    displaced_band = None
    if isinstance(displaced, dict):
        displaced_band = np.asarray(displaced.get("world_displaced", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    displaced_band = _downsample_points(displaced_band if displaced_band is not None else np.zeros((0, 3), dtype=np.float32))
    if displaced_band is not None and displaced_band.size > 0:
        viewer.add_points(
            displaced_band,
            name="safe_band_displaced",
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
    safe_uv_ordered = (
        np.asarray(safe_band.get("safe_uv_ordered", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
        if isinstance(safe_band, dict)
        else np.zeros((0, 2), dtype=np.int64)
    )
    safe_uv_ordered = safe_uv_ordered.reshape(-1, 2) if safe_uv_ordered.size > 0 else np.zeros((0, 2), dtype=np.int64)
    if one_map and safe_uv_ordered.shape[0] > 0:
        safe_pts = np.asarray(
            [
                one_map[(int(uv[0]), int(uv[1]))]
                for uv in safe_uv_ordered
                if (int(uv[0]), int(uv[1])) in one_map and np.isfinite(one_map[(int(uv[0]), int(uv[1]))]).all()
            ],
            dtype=np.float32,
        )
        safe_pts = _downsample_points(safe_pts)
        if safe_pts.size > 0:
            viewer.add_points(
                safe_pts,
                name="one_shot_safe_band",
                size=point_size,
                face_color=[0.0, 1.0, 1.0],
                opacity=0.8,
            )

    def _default_visible(layer_name):
        if layer_name in {
            "cond_full",
            "merged_full_surface",
            "safe_band_displaced",
            "one_shot_edge_cond",
            "one_shot_safe_band",
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

    run_model_inference = (args.checkpoint_path is not None) and (not args.skip_inference)
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
    elif args.skip_inference:
        print("Skipping displacement inference (--skip-inference set).")
        if args.checkpoint_path is not None:
            model_config, checkpoint_path = load_checkpoint_config(args.checkpoint_path)
    else:
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
    safe_band = {
        "axis_name": None,
        "total_lines": 0,
        "safe_lines": 0,
        "safe_axis_values_near_to_far": [],
        "safe_uv_ordered": np.zeros((0, 2), dtype=np.int64),
    }
    stacked_displacement = None
    displaced = {
        "uv": np.zeros((0, 2), dtype=np.int64),
        "world": np.zeros((0, 3), dtype=np.float32),
        "displacement": np.zeros((0, 3), dtype=np.float32),
        "world_displaced": np.zeros((0, 3), dtype=np.float32),
        "stack_count": np.zeros((0,), dtype=np.uint32),
    }

    for iteration in range(int(args.iterations)):
        print(f"[iteration {iteration + 1}/{int(args.iterations)}]")
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
            print("No edge bboxes available at current boundary; stopping iterative growth.")
            break

        bbox_crops = _build_bbox_crops(
            bboxes=bboxes,
            tgt_segment=tgt_segment,
            volume_scale=args.volume_scale,
            cond_zyxs=cond_zyxs,
            cond_valid=cond_valid,
            uv_cond=uv_cond,
            grow_direction=grow_direction,
            crop_size=crop_size,
            extrapolation_settings=extrapolation_settings,
            cond_pct=args.cond_pct,
        )
        _print_bbox_crop_debug_table(bbox_crops)
        bbox_results = _build_bbox_results_from_crops(bbox_crops)

        stacked_displacement = None
        if run_model_inference and model_state is not None:
            per_crop_fields = _run_bbox_displacement_inference(args, bbox_crops, model_state)
            stacked_displacement = _stack_displacements_to_global(per_crop_fields)
        _print_stacked_displacement_debug(stacked_displacement)

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
        safe_band = _compute_safe_band(
            bbox_results,
            one_shot,
            one_map,
            grow_direction,
            min_line_coverage=float(args.safe_line_min_coverage),
        )
        safe_stack_samples = _sample_stacked_displacement_on_safe_band(
            stacked_displacement,
            one_map,
            safe_band,
            max_lines=args.safe_band_lines,
        )

        _print_iteration_summary(bbox_results, one_shot, one_map, safe_band)
        _print_safe_band_sampling_debug(safe_stack_samples, safe_band, max_lines=args.safe_band_lines)
        displaced = _apply_displacement_and_print_stats(safe_stack_samples)

        prev_boundary = _boundary_axis_value(cond_valid, cond_uv_offset, grow_direction)
        merged_iter = _merge_displaced_points_into_full_surface(
            cond_zyxs,
            cond_valid,
            cond_uv_offset,
            displaced,
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
            print("No newly added valid points this iteration; stopping iterative growth.")
            break
        if not _boundary_advanced(prev_boundary, next_boundary, grow_direction):
            print("Boundary did not advance this iteration; stopping iterative growth.")
            break

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
            safe_band,
            disp_bbox=disp_bbox,
            displaced=displaced,
            merged=merged,
            downsample=args.napari_downsample,
            point_size=args.napari_point_size,
        )


if __name__ == "__main__":
    main()
