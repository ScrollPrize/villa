import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm

from vesuvius.image_proc.intensity.normalization import normalize_zscore
from vesuvius.neural_tracing.inference.common import (
    _aggregate_pred_samples_to_uv_grid,
    _print_agg_extrap_sampling_debug,
    _print_bbox_crop_debug_table,
    _print_iteration_summary,
    _resolve_settings,
    _RuntimeProfiler,
    _save_merged_surface_tifxyz,
    _serialize_args,
    _show_napari,
)
from vesuvius.neural_tracing.inference.displacement_tta import TTA_MERGE_METHODS
from vesuvius.neural_tracing.inference.infer_rowcol_split import (
    _bbox_to_min_corner_and_bounds_array,
    _build_model_inputs,
    _build_uv_grid,
    _build_uv_query_from_cond_points,
    _crop_volume_from_min_corner,
    _get_growth_context,
    _initialize_window_state,
    _points_to_voxels,
    _predict_displacement,
    _resolve_segment_volume,
    _scale_to_subsample_stride,
    _stored_to_full_bounds,
    compute_edge_one_shot_extrapolation,
    get_cond_edge_bboxes,
    get_window_bounds_from_bboxes,
    load_model,
    load_checkpoint_config,
    setup_segment,
)


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
        "--rbf-downsample-factor",
        type=int,
        default=None,
        help=(
            "Override RBF downsample factor used by one-shot extrapolation. "
            "When set, takes precedence over checkpoint/config rbf_downsample_factor."
        ),
    )
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
        "--overwrite-input-surface",
        action="store_true",
        help=(
            "Overwrite the input tifxyz surface directory provided by --tifxyz-path "
            "instead of creating a new timestamped output surface."
        ),
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
        "--refine",
        type=int,
        default=None,
        help=(
            "Within each outer iteration, run N+1 staged fractional displacement updates when "
            "sampling the stacked displacement field. Each stage applies 1/(N+1) of the sampled "
            "displacement before re-sampling at updated coordinates."
        ),
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
        "--lines-to-keep",
        dest="agg_extrap_lines",
        type=int,
        default=None,
        help=(
            "Optional number of near->far rows/cols to sample from one-shot aggregated extrapolation. "
            "If unset, samples all available lines."
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable runtime profiling logs for major pipeline stages.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging and tables.",
    )
    parser.set_defaults(tta=True)
    args = parser.parse_args()

    if args.edge_input_rowscols < 1:
        parser.error("--edge-input-rowscols must be >= 1")
    if args.rbf_downsample_factor is not None and args.rbf_downsample_factor < 1:
        parser.error("--rbf-downsample-factor must be >= 1 when provided")
    if args.bbox_overlap_frac < 0.0 or args.bbox_overlap_frac >= 1.0:
        parser.error("--bbox-overlap-frac must be in [0, 1)")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.refine is not None and args.refine < 1:
        parser.error("--refine must be >= 1 when provided")
    if args.agg_extrap_lines is not None and args.agg_extrap_lines < 1:
        parser.error("--lines-to-keep must be >= 1 when provided")
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


@dataclass(frozen=True)
class ExtrapLookupArrays:
    uv: np.ndarray
    world: np.ndarray


def _empty_extrap_lookup_arrays():
    return ExtrapLookupArrays(
        uv=np.zeros((0, 2), dtype=np.int64),
        world=np.zeros((0, 3), dtype=np.float32),
    )


def _as_extrap_lookup_arrays(extrap_lookup):
    if isinstance(extrap_lookup, ExtrapLookupArrays):
        return extrap_lookup
    return _empty_extrap_lookup_arrays()


def _uv_struct_view(uv):
    uv_int = np.ascontiguousarray(np.asarray(uv, dtype=np.int64))
    if uv_int.ndim != 2 or uv_int.shape[1] != 2:
        return np.zeros((0,), dtype=[("r", np.int64), ("c", np.int64)])
    return uv_int.view([("r", np.int64), ("c", np.int64)]).reshape(-1)


def _dedupe_uv_first_order_last_value(uv_int, world):
    uv_view = _uv_struct_view(uv_int)
    if uv_view.shape[0] == 0:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32),
        )

    sort_idx = np.argsort(uv_view, kind="stable")
    uv_sorted = uv_view[sort_idx]
    group_start = np.flatnonzero(
        np.concatenate([np.array([True], dtype=bool), uv_sorted[1:] != uv_sorted[:-1]])
    )
    group_end = np.empty_like(group_start)
    group_end[:-1] = group_start[1:]
    group_end[-1] = uv_sorted.shape[0]

    first_idx = np.minimum.reduceat(sort_idx, group_start)
    last_idx = sort_idx[group_end - 1]
    first_order = np.argsort(first_idx, kind="stable")
    first_idx = first_idx[first_order]
    last_idx = last_idx[first_order]
    return (
        uv_int[first_idx].astype(np.int64, copy=False),
        world[last_idx].astype(np.float32, copy=False),
    )


def _build_extrap_lookup_from_uv_world(uv_query_flat, extrap_world):
    uv_int = np.asarray(uv_query_flat, dtype=np.int64)
    world = np.asarray(extrap_world, dtype=np.float32)
    if uv_int.ndim != 2 or uv_int.shape[1] != 2:
        return _empty_extrap_lookup_arrays()
    if world.ndim != 2 or world.shape[1] != 3:
        return _empty_extrap_lookup_arrays()
    if uv_int.shape[0] == 0 or world.shape[0] == 0:
        return _empty_extrap_lookup_arrays()
    if uv_int.shape[0] != world.shape[0]:
        return _empty_extrap_lookup_arrays()

    finite = np.isfinite(world).all(axis=1)
    if not finite.any():
        return _empty_extrap_lookup_arrays()
    uv_keep = uv_int[finite].astype(np.int64, copy=False)
    world_keep = world[finite].astype(np.float32, copy=False)
    uv_dedup, world_dedup = _dedupe_uv_first_order_last_value(uv_keep, world_keep)
    if uv_dedup.shape[0] == 0:
        return _empty_extrap_lookup_arrays()
    return ExtrapLookupArrays(
        uv=uv_dedup,
        world=world_dedup,
    )


def _build_extrap_lookup_arrays(edge_extrapolation):
    query_uv_grid = np.asarray(edge_extrapolation.get("query_uv_grid", np.zeros((0, 0, 2), dtype=np.int64)))
    extrapolated_world = np.asarray(edge_extrapolation.get("extrapolated_world", np.zeros((0, 3), dtype=np.float32)))
    if query_uv_grid.ndim == 3 and query_uv_grid.shape[-1] == 2:
        h, w = query_uv_grid.shape[:2]
        if h < 1 or w < 1:
            return _empty_extrap_lookup_arrays()
        if extrapolated_world.shape[0] != h * w:
            return _empty_extrap_lookup_arrays()
        uv_flat = query_uv_grid.reshape(-1, 2).astype(np.int64, copy=False)
        return _build_extrap_lookup_from_uv_world(uv_flat, extrapolated_world)

    query_uv = np.asarray(edge_extrapolation.get("query_uv", np.zeros((0, 2), dtype=np.float64)))
    return _build_extrap_lookup_from_uv_world(query_uv, extrapolated_world)


def _select_extrap_uvs_from_lookup(extrap_lookup, grow_direction, max_lines=None):
    lookup = _as_extrap_lookup_arrays(extrap_lookup)
    uv_ordered = np.asarray(lookup.uv, dtype=np.int64)
    if uv_ordered.ndim != 2 or uv_ordered.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.int64)

    if uv_ordered.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    if grow_direction in {"left", "right"}:
        axis_idx = 1
        near_to_far_desc = grow_direction == "left"
    elif grow_direction in {"up", "down"}:
        axis_idx = 0
        near_to_far_desc = grow_direction == "up"
    else:
        raise ValueError(f"Unknown grow_direction '{grow_direction}'")

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

    depth_keep = int(max_lines)
    if depth_keep < 1:
        return np.zeros((0, 2), dtype=np.int64)

    # For ragged/non-rectangular fronts, keep near->far depth per boundary line
    # (per row for left/right, per col for up/down), not global axis values.
    if axis_idx == 1:
        boundary_ids = np.unique(uv_ordered[:, 0]).astype(np.int64, copy=False)
        picked = []
        for r in boundary_ids:
            row_uv = uv_ordered[uv_ordered[:, 0] == r]
            if row_uv.shape[0] == 0:
                continue
            row_primary = -row_uv[:, 1] if near_to_far_desc else row_uv[:, 1]
            row_order = np.argsort(row_primary, kind="stable")
            picked.append(row_uv[row_order[:depth_keep]])
    else:
        boundary_ids = np.unique(uv_ordered[:, 1]).astype(np.int64, copy=False)
        picked = []
        for c in boundary_ids:
            col_uv = uv_ordered[uv_ordered[:, 1] == c]
            if col_uv.shape[0] == 0:
                continue
            col_primary = -col_uv[:, 0] if near_to_far_desc else col_uv[:, 0]
            col_order = np.argsort(col_primary, kind="stable")
            picked.append(col_uv[col_order[:depth_keep]])

    if not picked:
        return np.zeros((0, 2), dtype=np.int64)

    selected = np.concatenate(picked, axis=0).astype(np.int64, copy=False)
    if selected.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    if axis_idx == 1:
        sel_primary = -selected[:, 1] if near_to_far_desc else selected[:, 1]
        sel_secondary = selected[:, 0]
    else:
        sel_primary = -selected[:, 0] if near_to_far_desc else selected[:, 0]
        sel_secondary = selected[:, 1]
    sel_order = np.lexsort((sel_secondary, sel_primary))
    return selected[sel_order]


def _lookup_extrap_for_uv_query_flat(uv_query_flat, extrap_lookup):
    uv_int = np.asarray(uv_query_flat, dtype=np.int64)
    if uv_int.ndim != 2 or uv_int.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    lookup = _as_extrap_lookup_arrays(extrap_lookup)
    lookup_uv = np.asarray(lookup.uv, dtype=np.int64)
    lookup_world = np.asarray(lookup.world, dtype=np.float32)
    if lookup_uv.ndim != 2 or lookup_uv.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    if lookup_world.ndim != 2 or lookup_world.shape[1] != 3:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    if lookup_uv.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    lookup_view = _uv_struct_view(lookup_uv)
    query_view = _uv_struct_view(uv_int)
    lookup_sort = np.argsort(lookup_view, kind="stable")
    lookup_sorted = lookup_view[lookup_sort]
    pos = np.searchsorted(lookup_sorted, query_view, side="left")
    in_bounds = pos < lookup_sorted.shape[0]
    matched = np.zeros((uv_int.shape[0],), dtype=bool)
    if in_bounds.any():
        pos_in = pos[in_bounds]
        matched[in_bounds] = lookup_sorted[pos_in] == query_view[in_bounds]
    if not matched.any():
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    keep_idx = np.nonzero(matched)[0]
    lookup_idx = lookup_sort[pos[keep_idx]]
    extrap_uv = uv_int[keep_idx].astype(np.int64, copy=False)
    extrap_world = lookup_world[lookup_idx].astype(np.float32, copy=False)
    return extrap_uv, extrap_world


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
    extrap_lookup,
    keep_debug_points=False,
):
    cond_valid_base = np.asarray(cond_valid, dtype=bool)
    cond_zyxs32 = np.asarray(cond_zyxs, dtype=np.float32)
    uv_cond64 = np.asarray(uv_cond, dtype=np.float64)
    crop_size = tuple(int(v) for v in crop_size)
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    crop_size_arr_f32 = crop_size_arr.astype(np.float32, copy=False)
    volume_for_crops = _resolve_segment_volume(tgt_segment, volume_scale=volume_scale)

    cond_rows, cond_cols = np.where(cond_valid_base)
    if cond_rows.size == 0:
        cond_uv_all = np.zeros((0, 2), dtype=np.float64)
        cond_world_all = np.zeros((0, 3), dtype=np.float32)
    else:
        cond_uv_all = uv_cond64[cond_rows, cond_cols].astype(np.float64, copy=False)
        cond_world_all = cond_zyxs32[cond_rows, cond_cols].astype(np.float32, copy=False)

    bbox_crops = []

    for bbox_idx, bbox in enumerate(bboxes):
        min_corner, _ = _bbox_to_min_corner_and_bounds_array(bbox)
        min_corner32 = min_corner.astype(np.float32, copy=False)
        vol_crop = _crop_volume_from_min_corner(volume_for_crops, min_corner, crop_size)
        vol_crop = normalize_zscore(vol_crop)

        cond_local_all = cond_world_all - min_corner32[None, :]
        cond_in_bounds = (
            (cond_local_all[:, 0] >= 0.0) &
            (cond_local_all[:, 0] < crop_size_arr_f32[0]) &
            (cond_local_all[:, 1] >= 0.0) &
            (cond_local_all[:, 1] < crop_size_arr_f32[1]) &
            (cond_local_all[:, 2] >= 0.0) &
            (cond_local_all[:, 2] < crop_size_arr_f32[2])
        )
        cond_uv = cond_uv_all[cond_in_bounds].astype(np.float64, copy=False)
        cond_world = cond_world_all[cond_in_bounds].astype(np.float32, copy=False)
        cond_local = cond_local_all[cond_in_bounds].astype(np.float32, copy=False)
        cond_vox = _points_to_voxels(cond_local, crop_size)
        uv_query = _build_uv_query_from_cond_points(cond_uv, grow_direction, cond_pct)
        uv_query_flat = uv_query.reshape(-1, 2).astype(np.float64, copy=False)

        extrap_uv, extrap_world = _lookup_extrap_for_uv_query_flat(uv_query_flat, extrap_lookup)
        if extrap_world.shape[0] > 0:
            extrap_local = extrap_world - min_corner32[None, :]
        else:
            extrap_uv = np.zeros((0, 2), dtype=np.int64)
            extrap_world = np.zeros((0, 3), dtype=np.float32)
            extrap_local = np.zeros((0, 3), dtype=np.float32)
        extrap_vox = _points_to_voxels(extrap_local, crop_size)
        cond_world_out = cond_world if keep_debug_points else np.zeros((0, 3), dtype=np.float32)
        extrap_world_out = extrap_world if keep_debug_points else np.zeros((0, 3), dtype=np.float32)

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
                "cond_world": cond_world_out,
                "extrap_world": extrap_world_out,
                "n_cond": int(cond_uv.shape[0]),
                "n_query": int(uv_query_flat.shape[0]),
                "n_extrap": int(extrap_uv.shape[0]),
            }
        )

    return bbox_crops

def _run_inference(args, bbox_crops, model_state, verbose=True):
    expected_in_channels = int(model_state["expected_in_channels"])
    if expected_in_channels not in (2, 3):
        raise RuntimeError(
            "infer_global_extrap currently supports only 2- or 3-channel models; "
            f"checkpoint expects in_channels={expected_in_channels}."
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
        batch_inputs = []
        for crop in batch:
            if expected_in_channels == 2:
                # Dense-displacement checkpoints may be trained without extrapolation
                # conditioning (vol + cond only).
                vol_t = torch.from_numpy(crop["volume"]).float().unsqueeze(0).unsqueeze(0)
                cond_t = torch.from_numpy(crop["cond_vox"]).float().unsqueeze(0).unsqueeze(0)
                model_input = torch.cat([vol_t, cond_t], dim=1)
            else:
                model_input = _build_model_inputs(crop["volume"], crop["cond_vox"], crop["extrap_vox"])
            batch_inputs.append(model_input)
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
        del model_inputs
        del batch_inputs
        del disp_pred
        done = (batch_start // batch_size) + 1
        if verbose:
            print(f"{desc}: batch {done}/{n_batches}")

    return per_crop_fields


def _stack_displacement_results(per_crop_fields):
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

    valid = disp_count > 0
    if valid.any():
        np.divide(
            disp_sum,
            disp_count[np.newaxis, ...],
            out=disp_sum,
            where=valid[np.newaxis, ...],
        )

    bbox_inclusive = (
        int(global_min[0]),
        int(global_max_exclusive[0] - 1),
        int(global_min[1]),
        int(global_max_exclusive[1] - 1),
        int(global_min[2]),
        int(global_max_exclusive[2] - 1),
    )
    return {
        "displacement": disp_sum,
        "count": disp_count,
        "min_corner": global_min.astype(np.int64, copy=False),
        "shape": np.asarray(global_shape, dtype=np.int64),
        "bbox": bbox_inclusive,
    }


def _empty_stack_samples():
    return {
        "uv": np.zeros((0, 2), dtype=np.int64),
        "world": np.zeros((0, 3), dtype=np.float32),
        "displacement": np.zeros((0, 3), dtype=np.float32),
        "stack_count": np.zeros((0,), dtype=np.uint32),
    }


def _sample_displacement_for_extrap_uvs(
    stacked_displacement,
    extrap_lookup,
    grow_direction,
    max_lines=None,
    refine=None,
):
    if stacked_displacement is None or extrap_lookup is None:
        return _empty_stack_samples()

    sampled_uv = _select_extrap_uvs_from_lookup(extrap_lookup, grow_direction, max_lines=max_lines)
    if sampled_uv.shape[0] == 0:
        return _empty_stack_samples()

    disp = np.asarray(stacked_displacement["displacement"], dtype=np.float32)
    count = np.asarray(stacked_displacement["count"], dtype=np.uint32)
    min_corner = np.asarray(stacked_displacement["min_corner"], dtype=np.int64)
    shape = np.asarray(stacked_displacement["shape"], dtype=np.int64)

    sampled_world = _lookup_extrap_zyxs_for_uvs(sampled_uv, extrap_lookup)
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

    if refine is None:
        sampled_disp, sampled_count, valid_mask = _sample_trilinear_displacement_stack(
            disp,
            count,
            coords_local,
        )
    else:
        sampled_disp, sampled_count, valid_mask = _sample_fractional_displacement_stack(
            disp,
            count,
            coords_local,
            refine_extra_steps=int(refine),
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


def _per_crop_displacement_bbox(per_crop_fields):
    if per_crop_fields is None or len(per_crop_fields) == 0:
        return None
    min_corners = np.stack([np.asarray(item["min_corner"], dtype=np.int64) for item in per_crop_fields], axis=0)
    max_exclusive = []
    for item in per_crop_fields:
        min_corner = np.asarray(item["min_corner"], dtype=np.int64)
        disp = np.asarray(item["displacement"])
        _, d, h, w = disp.shape
        max_exclusive.append(min_corner + np.asarray([d, h, w], dtype=np.int64))
    max_exclusive = np.stack(max_exclusive, axis=0)
    global_min = min_corners.min(axis=0)
    global_max_exclusive = max_exclusive.max(axis=0)
    return (
        int(global_min[0]),
        int(global_max_exclusive[0] - 1),
        int(global_min[1]),
        int(global_max_exclusive[1] - 1),
        int(global_min[2]),
        int(global_max_exclusive[2] - 1),
    )


def _sample_displacement_for_extrap_uvs_from_crops(
    per_crop_fields,
    extrap_lookup,
    grow_direction,
    max_lines=None,
    refine=None,
):
    if per_crop_fields is None or len(per_crop_fields) == 0 or extrap_lookup is None:
        return _empty_stack_samples()

    sampled_uv = _select_extrap_uvs_from_lookup(extrap_lookup, grow_direction, max_lines=max_lines)
    if sampled_uv.shape[0] == 0:
        return _empty_stack_samples()

    sampled_world = _lookup_extrap_zyxs_for_uvs(sampled_uv, extrap_lookup)
    finite_world = np.isfinite(sampled_world).all(axis=1)
    if not finite_world.any():
        return _empty_stack_samples()

    sampled_uv = sampled_uv[finite_world]
    sampled_world = sampled_world[finite_world].astype(np.float32, copy=False)
    n_points = int(sampled_world.shape[0])

    disp_acc = np.zeros((n_points, 3), dtype=np.float32)
    count_acc = np.zeros((n_points,), dtype=np.uint32)
    ones_cache = {}

    for item in per_crop_fields:
        disp = np.asarray(item["displacement"], dtype=np.float32)
        min_corner = np.asarray(item["min_corner"], dtype=np.float32)
        _, d, h, w = disp.shape

        z_local = sampled_world[:, 0] - float(min_corner[0])
        y_local = sampled_world[:, 1] - float(min_corner[1])
        x_local = sampled_world[:, 2] - float(min_corner[2])
        in_bounds = (
            (z_local >= 0.0) & (z_local <= float(d - 1)) &
            (y_local >= 0.0) & (y_local <= float(h - 1)) &
            (x_local >= 0.0) & (x_local <= float(w - 1))
        )
        if not in_bounds.any():
            continue

        point_idx = np.nonzero(in_bounds)[0]
        coords_local = np.stack(
            [
                z_local[point_idx],
                y_local[point_idx],
                x_local[point_idx],
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

        shape_key = (int(d), int(h), int(w))
        count_field = ones_cache.get(shape_key)
        if count_field is None:
            count_field = np.ones(shape_key, dtype=np.uint8)
            ones_cache[shape_key] = count_field

        if refine is None:
            sampled_disp, _, valid_mask = _sample_trilinear_displacement_stack(
                disp,
                count_field,
                coords_local,
            )
        else:
            sampled_disp, _, valid_mask = _sample_fractional_displacement_stack(
                disp,
                count_field,
                coords_local,
                refine_extra_steps=int(refine),
            )
        if not valid_mask.any():
            continue

        keep_idx = point_idx[valid_mask]
        disp_acc[keep_idx] += sampled_disp[valid_mask]
        count_acc[keep_idx] += 1

    have_disp = count_acc > 0
    if not have_disp.any():
        return _empty_stack_samples()

    disp_mean = np.zeros_like(disp_acc, dtype=np.float32)
    disp_mean[have_disp] = disp_acc[have_disp] / count_acc[have_disp, None].astype(np.float32, copy=False)

    sampled_uv = sampled_uv[have_disp]
    sampled_world = sampled_world[have_disp]
    sampled_disp = disp_mean[have_disp]
    sampled_count = count_acc[have_disp]

    return {
        "uv": sampled_uv.astype(np.int64, copy=False),
        "world": sampled_world.astype(np.float32, copy=False),
        "displacement": sampled_disp.astype(np.float32, copy=False),
        "stack_count": sampled_count.astype(np.uint32, copy=False),
    }


def _sample_fractional_displacement_stack(disp, count, coords_local, refine_extra_steps):
    coords_local = np.asarray(coords_local, dtype=np.float32)
    if coords_local.ndim != 2 or coords_local.shape[1] != 3 or coords_local.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0,), dtype=bool),
        )

    refine_extra_steps = max(int(refine_extra_steps), 0)
    refine_parts = refine_extra_steps + 1
    refine_fraction = 1.0 / float(refine_parts)

    start_coords = coords_local.copy()
    current_coords = coords_local.copy()
    best_count = np.zeros((coords_local.shape[0],), dtype=np.uint32)
    ever_valid = np.zeros((coords_local.shape[0],), dtype=bool)

    for _ in range(refine_parts):
        stage_disp, stage_count, stage_valid = _sample_trilinear_displacement_stack(
            disp,
            count,
            current_coords,
        )
        np.maximum(best_count, stage_count, out=best_count)
        if not stage_valid.any():
            continue
        delta = stage_disp * refine_fraction
        finite_delta = np.isfinite(delta).all(axis=1)
        apply_mask = stage_valid & finite_delta
        if not apply_mask.any():
            continue
        current_coords[apply_mask] = current_coords[apply_mask] + delta[apply_mask]
        ever_valid |= apply_mask

    sampled_disp = (current_coords - start_coords).astype(np.float32, copy=False)
    finite_disp = np.isfinite(sampled_disp).all(axis=1)
    if not bool(finite_disp.all()):
        sampled_disp = np.where(finite_disp[:, None], sampled_disp, 0.0).astype(np.float32, copy=False)
        ever_valid &= finite_disp
    return sampled_disp, best_count, ever_valid


def _lookup_extrap_zyxs_for_uvs(sampled_uv, extrap_lookup):
    sampled_uv = np.asarray(sampled_uv, dtype=np.int64)
    if sampled_uv.ndim != 2 or sampled_uv.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    extrap_uv, extrap_world = _lookup_extrap_for_uv_query_flat(sampled_uv, extrap_lookup)
    if extrap_uv.shape[0] != sampled_uv.shape[0] or not np.array_equal(extrap_uv, sampled_uv):
        raise KeyError("Requested UV is not present in extrapolation lookup.")
    return extrap_world.astype(np.float32, copy=False)


def _sample_extrap_no_disp(extrap_lookup, grow_direction, max_lines=None):
    if extrap_lookup is None:
        return _empty_stack_samples()

    sampled_uv = _select_extrap_uvs_from_lookup(extrap_lookup, grow_direction, max_lines=max_lines)
    if sampled_uv.shape[0] == 0:
        return _empty_stack_samples()

    sampled_world = _lookup_extrap_zyxs_for_uvs(sampled_uv, extrap_lookup)
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


def _apply_displacement(samples, verbose=True, skip_inference=False):
    world = np.asarray(samples.get("world", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    displacement = np.asarray(samples.get("displacement", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    uv = np.asarray(samples.get("uv", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    stack_count = np.asarray(samples.get("stack_count", np.zeros((0,), dtype=np.uint32)), dtype=np.uint32)

    if skip_inference:
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


def _boundary_axis_value(valid, uv_offset, grow_direction):
    valid = np.asarray(valid, dtype=bool)
    if valid.size == 0 or not valid.any():
        return None

    _, growth_spec = _get_growth_context(grow_direction)
    r0, c0 = int(uv_offset[0]), int(uv_offset[1])

    if growth_spec["axis"] == "col":
        # For left/right growth, boundary is measured per row by extreme column.
        line_valid = np.any(valid, axis=1)
        if not line_valid.any():
            return None
        n_cols = valid.shape[1]
        if growth_spec["growth_sign"] > 0:
            boundary_rel = n_cols - 1 - np.argmax(valid[:, ::-1], axis=1)
        else:
            boundary_rel = np.argmax(valid, axis=1)
        line_ids = np.where(line_valid)[0].astype(np.int64, copy=False) + r0
        boundary_vals = boundary_rel[line_valid].astype(np.int64, copy=False) + c0
    else:
        # For up/down growth, boundary is measured per column by extreme row.
        line_valid = np.any(valid, axis=0)
        if not line_valid.any():
            return None
        n_rows = valid.shape[0]
        if growth_spec["growth_sign"] > 0:
            boundary_rel = n_rows - 1 - np.argmax(valid[::-1, :], axis=0)
        else:
            boundary_rel = np.argmax(valid, axis=0)
        line_ids = np.where(line_valid)[0].astype(np.int64, copy=False) + c0
        boundary_vals = boundary_rel[line_valid].astype(np.int64, copy=False) + r0

    return line_ids, boundary_vals


def _boundary_advanced(prev_boundary, next_boundary, grow_direction):
    if prev_boundary is None or next_boundary is None:
        return False

    prev_lines, prev_vals = prev_boundary
    next_lines, next_vals = next_boundary
    if prev_lines.size == 0 or next_lines.size == 0:
        return False

    _, prev_idx, next_idx = np.intersect1d(
        prev_lines,
        next_lines,
        assume_unique=False,
        return_indices=True,
    )
    if prev_idx.size == 0:
        return False

    delta = next_vals[next_idx] - prev_vals[prev_idx]
    _, growth_spec = _get_growth_context(grow_direction)
    if growth_spec["growth_sign"] > 0:
        return bool(np.any(delta > 0))
    return bool(np.any(delta < 0))


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


def _surface_to_stored_uv_samples_nearest(
    grid,
    valid,
    uv_offset,
    sub_r,
    sub_c,
):
    rows, cols = np.where(np.asarray(valid, dtype=bool))
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    pts = np.asarray(grid, dtype=np.float32)[rows, cols].astype(np.float32, copy=False)
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

    rows = rows[finite].astype(np.int64, copy=False)
    cols = cols[finite].astype(np.int64, copy=False)
    pts = pts[finite]

    r_abs = rows + int(uv_offset[0])
    c_abs = cols + int(uv_offset[1])

    # Project full-res UVs to nearest stored-resolution UV cell so growth that
    # lands off a single phase still contributes at save resolution.
    stored_r = np.rint(r_abs.astype(np.float64) / float(sub_r)).astype(np.int64)
    stored_c = np.rint(c_abs.astype(np.float64) / float(sub_c)).astype(np.int64)
    uv = np.stack([stored_r, stored_c], axis=-1)
    uniq_uv, inv = np.unique(uv, axis=0, return_inverse=True)

    pts_sum = np.zeros((uniq_uv.shape[0], 3), dtype=np.float64)
    pts_count = np.zeros((uniq_uv.shape[0],), dtype=np.int64)
    np.add.at(pts_sum, inv, pts.astype(np.float64, copy=False))
    np.add.at(pts_count, inv, 1)

    pts_mean = (pts_sum / np.maximum(pts_count[:, None], 1)).astype(np.float32, copy=False)
    return uniq_uv.astype(np.int64, copy=False), pts_mean


def _build_extrap_lookup_from_grid(grid, valid, offset):
    rows, cols = np.where(np.asarray(valid, dtype=bool))
    if rows.size == 0:
        return _empty_extrap_lookup_arrays()
    rows_abs = rows.astype(np.int64) + int(offset[0])
    cols_abs = cols.astype(np.int64) + int(offset[1])
    pts = np.asarray(grid, dtype=np.float32)[rows, cols]
    finite_mask = np.isfinite(pts).all(axis=1)
    if not finite_mask.any():
        return _empty_extrap_lookup_arrays()
    rows_abs = rows_abs[finite_mask]
    cols_abs = cols_abs[finite_mask]
    pts = pts[finite_mask].astype(np.float32, copy=False)
    uv = np.stack([rows_abs, cols_abs], axis=-1).astype(np.int64, copy=False)
    return ExtrapLookupArrays(
        uv=uv,
        world=pts,
    )


def main():
    args = parse_args()
    call_args = _serialize_args(args)
    crop_size = tuple(int(v) for v in args.crop_size)
    profiler = _RuntimeProfiler(enabled=args.profile, device=args.device)
    run_start = None
    if args.profile:
        profiler.sync()
        run_start = time.perf_counter()

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
        with profiler.section("load_model"):
            model_state = load_model(args)
        model_config = model_state["model_config"]
        checkpoint_path = model_state["checkpoint_path"]
        expected_in_channels = int(model_state["expected_in_channels"])
        if expected_in_channels not in (2, 3):
            raise RuntimeError(
                "infer_global_extrap currently supports only 2- or 3-channel models; "
                f"got in_channels={expected_in_channels}"
            )
    elif extrap_only_mode:
        if args.verbose:
            print("Running extrap-only iterative mode (--extrap-only set): skipping inference and stack sampling.")
        if args.checkpoint_path is not None:
            with profiler.section("load_checkpoint_config"):
                model_config, checkpoint_path = load_checkpoint_config(args.checkpoint_path)
    elif args.skip_inference:
        if args.verbose:
            print("Skipping displacement inference (--skip-inference set).")
        if args.checkpoint_path is not None:
            with profiler.section("load_checkpoint_config"):
                model_config, checkpoint_path = load_checkpoint_config(args.checkpoint_path)
    else:
        if args.verbose:
            print("Skipping displacement inference (no --checkpoint-path provided).")
        if args.checkpoint_path is not None:
            with profiler.section("load_checkpoint_config"):
                model_config, checkpoint_path = load_checkpoint_config(args.checkpoint_path)

    with profiler.section("resolve_settings"):
        extrapolation_settings = _resolve_settings(
            args,
            model_config=model_config,
            load_checkpoint_config_fn=load_checkpoint_config,
        )

    with profiler.section("setup_segment"):
        volume = zarr.open_group(args.volume_path, mode="r")
        tgt_segment, stored_zyxs, valid_s, grow_direction, h_s, w_s = setup_segment(args, volume)
    cond_direction, _ = _get_growth_context(grow_direction)

    with profiler.section("initialize_window"):
        cond_direction_init, _ = _get_growth_context(grow_direction)
        init_bboxes, _ = get_cond_edge_bboxes(
            stored_zyxs,
            cond_direction_init,
            crop_size,
            overlap_frac=args.bbox_overlap_frac,
            cond_valid=valid_s,
        )
        if len(init_bboxes) == 0:
            raise RuntimeError("No valid edge bboxes found for segment.")
        r0_s, r1_s, c0_s, c1_s = get_window_bounds_from_bboxes(
            stored_zyxs, valid_s, init_bboxes, pad=args.window_pad,
        )

        full_bounds = _stored_to_full_bounds(tgt_segment, (r0_s, r1_s, c0_s, c1_s))
        current_zyxs, current_valid, current_uv_offset = _initialize_window_state(
            tgt_segment, full_bounds,
        )

    bbox_results = []
    edge_extrapolation = {
        "edge_seed_uv": np.zeros((0, 2), dtype=np.float64),
        "edge_seed_world": np.zeros((0, 3), dtype=np.float32),
        "query_uv": np.zeros((0, 2), dtype=np.float64),
        "extrapolated_world": np.zeros((0, 3), dtype=np.float32),
    }
    extrap_lookup = _empty_extrap_lookup_arrays()
    disp_bbox = None
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

        with profiler.section("iter_get_edge_bboxes"):
            bboxes, _ = get_cond_edge_bboxes(
                cond_zyxs,
                cond_direction,
                crop_size,
                overlap_frac=args.bbox_overlap_frac,
                cond_valid=cond_valid,
            )
        if iteration_pbar is not None:
            iteration_pbar.set_postfix_str(f"bboxes={len(bboxes)}", refresh=True)
        if len(bboxes) == 0:
            if args.verbose:
                print("No edge bboxes available at current boundary; stopping iterative growth.")
            elif iteration_pbar is not None:
                iteration_pbar.set_postfix_str("bboxes=0 | stopped: no edge bboxes", refresh=True)
            break

        with profiler.section("iter_edge_extrapolation"):
            edge_extrapolation = compute_edge_one_shot_extrapolation(
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
        if edge_extrapolation is None:
            edge_extrapolation = {
                "edge_seed_uv": np.zeros((0, 2), dtype=np.float64),
                "edge_seed_world": np.zeros((0, 3), dtype=np.float32),
                "query_uv": np.zeros((0, 2), dtype=np.float64),
                "extrapolated_world": np.zeros((0, 3), dtype=np.float32),
            }

        with profiler.section("iter_aggregate_extrapolation"):
            if args.verbose or args.napari:
                query_uv, extrapolated_world = _finite_uv_world(
                    edge_extrapolation.get("query_uv"),
                    edge_extrapolation.get("extrapolated_world"),
                )
                uv_world_samples = [(query_uv, extrapolated_world)] if len(query_uv) > 0 else []
                aggregated_world_grid, aggregated_valid_mask, aggregated_uv_offset = _aggregate_pred_samples_to_uv_grid(
                    uv_world_samples
                )
                extrap_lookup = _build_extrap_lookup_from_grid(
                    aggregated_world_grid,
                    aggregated_valid_mask,
                    aggregated_uv_offset,
                )
            else:
                extrap_lookup = _build_extrap_lookup_arrays(edge_extrapolation)
            # Keep only lightweight edge_extrapolation fields after lookup construction.
            edge_extrapolation["query_uv_grid"] = np.zeros((0, 0, 2), dtype=np.int64)
            edge_extrapolation["query_uv"] = np.zeros((0, 2), dtype=np.float64)
            edge_extrapolation["extrapolated_local"] = np.zeros((0, 3), dtype=np.float32)
            edge_extrapolation["extrapolated_world"] = np.zeros((0, 3), dtype=np.float32)

        with profiler.section("iter_build_bbox_crops"):
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
                extrap_lookup=extrap_lookup,
                keep_debug_points=bool(args.napari),
            )
        _print_bbox_crop_debug_table(bbox_crops, verbose=args.verbose)
        bbox_results = bbox_crops

        disp_bbox = None
        if extrap_only_mode:
            with profiler.section("iter_sample_extrap_no_disp"):
                agg_samples = _sample_extrap_no_disp(
                    extrap_lookup,
                    grow_direction,
                    max_lines=args.agg_extrap_lines,
                )
        elif run_model_inference and model_state is not None:
            with profiler.section("iter_displacement_inference"):
                per_crop_fields = _run_inference(
                    args,
                    bbox_crops,
                    model_state,
                    verbose=args.verbose,
                )
            disp_bbox = _per_crop_displacement_bbox(per_crop_fields)
            with profiler.section("iter_sample_crop_displacement"):
                agg_samples = _sample_displacement_for_extrap_uvs_from_crops(
                    per_crop_fields,
                    extrap_lookup,
                    grow_direction,
                    max_lines=args.agg_extrap_lines,
                    refine=args.refine,
                )
            del per_crop_fields
        else:
            agg_samples = _empty_stack_samples()

        _print_iteration_summary(
            bbox_results,
            edge_extrapolation,
            extrap_lookup,
            grow_direction,
            verbose=args.verbose,
        )
        _print_agg_extrap_sampling_debug(
            agg_samples,
            extrap_lookup,
            grow_direction,
            max_lines=args.agg_extrap_lines,
            verbose=args.verbose,
        )
        if extrap_only_mode:
            with profiler.section("iter_apply_samples_direct"):
                displaced = _apply_displacement(agg_samples, verbose=args.verbose, skip_inference=True)
        else:
            with profiler.section("iter_apply_displacement"):
                displaced = _apply_displacement(agg_samples, verbose=args.verbose)

        prev_boundary = _boundary_axis_value(cond_valid, cond_uv_offset, grow_direction)
        with profiler.section("iter_merge_surface"):
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
                iteration_pbar.set_postfix_str(
                    f"bboxes={len(bboxes)} | stopped: no new valid points",
                    refresh=True,
                )
            break
        if not _boundary_advanced(prev_boundary, next_boundary, grow_direction):
            if args.verbose:
                print("Boundary did not advance this iteration; stopping iterative growth.")
            elif iteration_pbar is not None:
                iteration_pbar.set_postfix_str(
                    f"bboxes={len(bboxes)} | stopped: boundary unchanged",
                    refresh=True,
                )
            break

    if iteration_pbar is not None:
        iteration_pbar.close()

    # Merge the grown surface back onto the stored-resolution base surface
    # (already in memory from setup_segment) to avoid materializing full resolution.
    with profiler.section("final_merge_stored_surface"):
        scale_y, scale_x = tgt_segment._scale

        sub_r = _scale_to_subsample_stride(scale_y)
        sub_c = _scale_to_subsample_stride(scale_x)

        grown_uv, grown_pts = _surface_to_stored_uv_samples_nearest(
            current_zyxs,
            current_valid,
            current_uv_offset,
            sub_r,
            sub_c,
        )
        merged = _merge_displaced_points_into_full_surface(
            stored_zyxs, valid_s, (0, 0),
            {"uv": grown_uv, "world_displaced": grown_pts},
            verbose=args.verbose,
        )

    with profiler.section("save_tifxyz"):
        _save_merged_surface_tifxyz(
            args, merged, checkpoint_path, model_config, call_args,
            input_scale=tgt_segment._scale,
        )

    if args.napari:
        with profiler.section("napari"):
            _show_napari(
                current_zyxs,
                current_valid,
                bbox_results,
                edge_extrapolation,
                extrap_lookup,
                disp_bbox=disp_bbox,
                displaced=displaced,
                merged=merged,
                downsample=args.napari_downsample,
                point_size=args.napari_point_size,
            )

    if args.profile:
        profiler.sync()
        total_runtime_s = (
            time.perf_counter() - run_start
            if run_start is not None
            else None
        )
        profiler.print_summary(total_runtime_s=total_runtime_s)


if __name__ == "__main__":
    main()
