import argparse
import contextlib
import io
import time

_PROCESS_START = time.perf_counter()

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm

from vesuvius.image_proc.intensity.normalization import normalize_zscore
from vesuvius.neural_tracing.inference.extrap_lookup import ExtrapLookupArrays
from vesuvius.neural_tracing.inference.common import (
    _bbox_to_min_corner_and_bounds_array,
    _build_uv_grid,
    _build_uv_query_from_edge_band,
    _coerce_uv_int_array,
    _crop_volume_from_min_corner,
    _flat_index_dtype_for_shape,
    _get_growth_context,
    _initialize_window_state,
    _points_to_voxels,
    _print_agg_extrap_sampling_debug,
    _print_bbox_crop_debug_table,
    _print_iteration_summary,
    _resolve_segment_volume,
    _RuntimeProfiler,
    _scale_to_subsample_stride,
    _save_merged_surface_tifxyz,
    _select_extrap_uv_indices_for_sampling,
    _serialize_args,
    _stored_to_full_bounds,
    _show_napari,
    compute_edge_one_shot_extrapolation,
    get_cond_edge_bboxes,
    get_window_bounds_from_bboxes,
    resolve_extrapolation_settings,
    setup_segment,
)
from vesuvius.neural_tracing.inference.displacement_helpers import (
    load_checkpoint_config,
    load_model,
    predict_displacement,
)
from vesuvius.neural_tracing.inference.displacement_tta import (
    TTA_MERGE_METHODS,
    TTA_TRANSFORM_MODES,
)

_ALL_GROW_DIRECTION_ORDER = ("left", "right", "up", "down")
_INT32_MAX = int(np.iinfo(np.int32).max)


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run global edge extrapolation with optional displacement-model stacking."
    )
    parser.add_argument("--tifxyz-path", type=str, required=True)
    parser.add_argument("--volume-path", type=str, required=True)
    parser.add_argument("--volume-scale", type=int, default=1)
    parser.add_argument(
        "--grow-direction",
        type=str,
        required=True,
        choices=[*_ALL_GROW_DIRECTION_ORDER, "all"],
    )
    parser.add_argument("--cond-pct", type=float, default=0.50)
    parser.add_argument("--crop-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--window-pad", type=int, default=20)
    parser.add_argument("--bbox-overlap-frac", type=float, default=0.0)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--extrapolation-method", type=str, default=None)
    parser.add_argument(
        "--rbf-scale",
        type=str,
        default="stored",
        choices=("full", "stored"),
        help=(
            "RBF solve-space preset. "
            "'full' forces full-UV + float64 baseline behavior; "
            "'stored' uses stored-lattice UV + float32 and disables extra RBF downsampling "
            "(default)."
        ),
    )
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
        "--rbf-max-points",
        type=int,
        default=None,
        help=(
            "Optional cap on RBF conditioning points after downsampling. "
            "When set, takes precedence over checkpoint/config rbf_max_points."
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
        help=(
            "Output tifxyz UV step size. If unset, inferred from the stored input tifxyz scale. "
            "When provided, must match the stored input scale."
        ),
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
    parser.add_argument("--batch-size", type=int, default=4)
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
        help="Disable test-time augmentation (enabled by default).",
    )
    parser.add_argument(
        "--tta-transform",
        type=str,
        default="mirror",
        choices=TTA_TRANSFORM_MODES,
        help=(
            "TTA transform set: 'mirror' uses 8 flip variants; "
            "'rotate3' uses 3 axis-transpose variants (z-up, x-up, y-up)."
        ),
    )
    parser.add_argument(
        "--tta-merge-method",
        type=str,
        default="vector_geomedian",
        choices=TTA_MERGE_METHODS,
        help="How to merge TTA displacement predictions.",
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
        "--tta-batch-size",
        type=int,
        default=2,
        help=(
            "Number of TTA variants to evaluate per forward pass. "
            "Use 1 to keep model batch at --batch-size; use 8 (mirror) or 3 (rotate3) to fuse all variants."
        ),
    )
    parser.add_argument(
        "--edge-input-rowscols",
        type=int,
        default=40,
        help="Number of edge rows/cols from conditioning region to use in one-shot extrapolation.",
    )
    parser.add_argument(
        "--lines-to-keep",
        dest="agg_extrap_lines",
        type=int,
        default=20,
        help=(
            "Number of near->far rows/cols to sample from one-shot aggregated extrapolation. "
            "Defaults to 20."
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
    args = parser.parse_args(argv)

    if args.edge_input_rowscols < 1:
        parser.error("--edge-input-rowscols must be >= 1")
    if args.rbf_downsample_factor is not None and args.rbf_downsample_factor < 1:
        parser.error("--rbf-downsample-factor must be >= 1 when provided")
    if args.rbf_max_points is not None and args.rbf_max_points < 1:
        parser.error("--rbf-max-points must be >= 1 when provided")
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
    if args.tta_batch_size < 1:
        parser.error("--tta-batch-size must be >= 1.")
    if args.tifxyz_step_size is not None and args.tifxyz_step_size < 1:
        parser.error("--tifxyz-step-size must be >= 1 when provided.")
    if args.tifxyz_voxel_size_um is not None and args.tifxyz_voxel_size_um <= 0:
        parser.error("--tifxyz-voxel-size-um must be > 0 when provided.")
    return args


_DENSE_ARG_ALIASES = {
    "direction": "grow_direction",
    "steps": "iterations",
    "dense_checkpoint_path": "checkpoint_path",
    "dense_config_path": "config_path",
    "volume_zarr": "volume_path",
    "lines_to_keep": "agg_extrap_lines",
}

_DENSE_ARG_TO_CLI = {
    "tifxyz_path": "--tifxyz-path",
    "volume_path": "--volume-path",
    "volume_scale": "--volume-scale",
    "grow_direction": "--grow-direction",
    "cond_pct": "--cond-pct",
    "crop_size": "--crop-size",
    "window_pad": "--window-pad",
    "bbox_overlap_frac": "--bbox-overlap-frac",
    "checkpoint_path": "--checkpoint-path",
    "config_path": "--config-path",
    "extrapolation_method": "--extrapolation-method",
    "rbf_scale": "--rbf-scale",
    "rbf_downsample_factor": "--rbf-downsample-factor",
    "rbf_max_points": "--rbf-max-points",
    "tifxyz_out_dir": "--tifxyz-out-dir",
    "tifxyz_step_size": "--tifxyz-step-size",
    "tifxyz_voxel_size_um": "--tifxyz-voxel-size-um",
    "device": "--device",
    "batch_size": "--batch-size",
    "iterations": "--iterations",
    "refine": "--refine",
    "tta_transform": "--tta-transform",
    "tta_merge_method": "--tta-merge-method",
    "tta_outlier_drop_thresh": "--tta-outlier-drop-thresh",
    "tta_outlier_drop_min_keep": "--tta-outlier-drop-min-keep",
    "tta_batch_size": "--tta-batch-size",
    "edge_input_rowscols": "--edge-input-rowscols",
    "agg_extrap_lines": "--lines-to-keep",
    "napari_downsample": "--napari-downsample",
    "napari_point_size": "--napari-point-size",
}

_DENSE_BOOL_TO_CLI = {
    "overwrite_input_surface": "--overwrite-input-surface",
    "skip_inference": "--skip-inference",
    "extrap_only": "--extrap-only",
    "napari": "--napari",
    "profile": "--profile",
    "verbose": "--verbose",
}


def normalize_dense_args(dense_args):
    if not isinstance(dense_args, dict):
        raise RuntimeError(f"dense_args must be a dict, got {type(dense_args).__name__}")
    normalized = {}
    for key, value in dense_args.items():
        key_norm = str(key).replace("-", "_")
        normalized[_DENSE_ARG_ALIASES.get(key_norm, key_norm)] = value
    if normalized.get("edge_input_rowscols") is None:
        normalized["edge_input_rowscols"] = 40
    return normalized


def _append_cli_arg(argv, flag, value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        argv.append(flag)
        argv.extend(str(v) for v in value)
        return
    argv.extend([flag, str(value)])


def _dense_args_to_argv(dense_args):
    dense_args = normalize_dense_args(dense_args)
    argv = []
    for key, flag in _DENSE_ARG_TO_CLI.items():
        if key not in dense_args:
            continue
        value = dense_args.get(key)
        if key == "tta_outlier_drop_thresh" and value is None:
            _append_cli_arg(argv, flag, "none")
            continue
        if value is None:
            continue
        _append_cli_arg(argv, flag, value)

    for key, flag in _DENSE_BOOL_TO_CLI.items():
        if bool(dense_args.get(key)):
            argv.append(flag)

    if "tta" in dense_args and (dense_args.get("tta") is False):
        argv.append("--no-tta")
    return argv


def _empty_uv(dtype=np.int64):
    return np.zeros((0, 2), dtype=dtype)


def _empty_world(dtype=np.float32):
    return np.zeros((0, 3), dtype=dtype)


def _empty_counts(dtype=np.uint32):
    return np.zeros((0,), dtype=dtype)


def _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32):
    return _empty_uv(dtype=uv_dtype), _empty_world(dtype=world_dtype)


def _coerce_uv_world(uv, world, uv_dtype=np.int64, world_dtype=np.float32):
    uv_arr = np.asarray(uv, dtype=uv_dtype)
    world_arr = np.asarray(world, dtype=world_dtype)
    if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
        return _empty_uv_world(uv_dtype=uv_dtype, world_dtype=world_dtype)
    if world_arr.ndim != 2 or world_arr.shape[1] != 3:
        return _empty_uv_world(uv_dtype=uv_dtype, world_dtype=world_dtype)
    if uv_arr.shape[0] == 0 or world_arr.shape[0] == 0:
        return _empty_uv_world(uv_dtype=uv_dtype, world_dtype=world_dtype)
    if uv_arr.shape[0] != world_arr.shape[0]:
        return _empty_uv_world(uv_dtype=uv_dtype, world_dtype=world_dtype)
    return uv_arr, world_arr


def _finite_uv_world(uv, world):
    if uv is None or world is None:
        return _empty_uv_world(uv_dtype=np.float32, world_dtype=np.float32)
    uv, world = _coerce_uv_world(uv, world, uv_dtype=np.float32, world_dtype=np.float32)
    if uv.shape[0] == 0:
        return _empty_uv_world(uv_dtype=np.float32, world_dtype=np.float32)
    keep = np.isfinite(world).all(axis=1)
    if not keep.any():
        return _empty_uv_world(uv_dtype=np.float32, world_dtype=np.float32)
    return uv[keep].astype(np.float32, copy=False), world[keep].astype(np.float32, copy=False)


def _empty_edge_extrapolation():
    return {
        "edge_seed_uv": _empty_uv(dtype=np.int64),
        "edge_seed_world": _empty_world(dtype=np.float32),
        "query_uv_grid": np.zeros((0, 0, 2), dtype=np.int64),
        "extrapolated_world": _empty_world(dtype=np.float32),
    }

def _empty_extrap_lookup_arrays():
    return _make_extrap_lookup_arrays(
        _empty_uv(dtype=np.int64),
        _empty_world(dtype=np.float32),
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


def _build_lookup_index(uv_int):
    uv_view = _uv_struct_view(uv_int)
    sort_idx_dtype = np.int32 if uv_view.shape[0] <= (_INT32_MAX + 1) else np.int64
    if uv_view.shape[0] == 0:
        return np.zeros((0,), dtype=sort_idx_dtype), uv_view
    sort_idx = np.argsort(uv_view, kind="stable")
    uv_sorted = uv_view[sort_idx]
    return sort_idx.astype(sort_idx_dtype, copy=False), uv_sorted


def _make_extrap_lookup_arrays(uv, world):
    uv_int, world32 = _coerce_uv_world(
        uv,
        world,
        uv_dtype=np.int64,
        world_dtype=np.float32,
    )
    if uv_int.shape[0] == 0:
        empty_view = _uv_struct_view(_empty_uv(dtype=np.int64))
        return ExtrapLookupArrays(
            uv=_empty_uv(dtype=np.int64),
            world=_empty_world(dtype=np.float32),
            lookup_sort_idx=np.zeros((0,), dtype=np.int32),
            lookup_uv_sorted=empty_view,
        )
    sort_idx, uv_sorted = _build_lookup_index(uv_int)
    return ExtrapLookupArrays(
        uv=uv_int,
        world=world32,
        lookup_sort_idx=sort_idx,
        lookup_uv_sorted=uv_sorted,
    )


def _dedupe_uv_first_order_last_value(uv_int, world):
    uv_view = _uv_struct_view(uv_int)
    if uv_view.shape[0] == 0:
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32)

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
    uv_int, world = _coerce_uv_world(
        uv_query_flat,
        extrap_world,
        uv_dtype=np.int64,
        world_dtype=np.float32,
    )
    if uv_int.shape[0] == 0:
        return _empty_extrap_lookup_arrays()

    finite = np.isfinite(world).all(axis=1)
    if not finite.any():
        return _empty_extrap_lookup_arrays()
    uv_keep = uv_int[finite].astype(np.int64, copy=False)
    world_keep = world[finite].astype(np.float32, copy=False)
    uv_dedup, world_dedup = _dedupe_uv_first_order_last_value(uv_keep, world_keep)
    if uv_dedup.shape[0] == 0:
        return _empty_extrap_lookup_arrays()
    return _make_extrap_lookup_arrays(uv_dedup, world_dedup)


def _build_extrap_lookup_arrays(edge_extrapolation):
    query_uv_grid = np.asarray(edge_extrapolation.get("query_uv_grid", np.zeros((0, 0, 2), dtype=np.int64)))
    extrapolated_world = np.asarray(edge_extrapolation.get("extrapolated_world", _empty_world(dtype=np.float32)))
    if query_uv_grid.ndim != 3 or query_uv_grid.shape[-1] != 2:
        return _empty_extrap_lookup_arrays()
    h, w = query_uv_grid.shape[:2]
    if h < 1 or w < 1:
        return _empty_extrap_lookup_arrays()
    if extrapolated_world.shape[0] != h * w:
        return _empty_extrap_lookup_arrays()
    uv_flat = query_uv_grid.reshape(-1, 2).astype(np.int64, copy=False)
    return _build_extrap_lookup_from_uv_world(uv_flat, extrapolated_world)


def _prepare_sampled_extrap_points(extrap_lookup, grow_direction, max_lines=None):
    if extrap_lookup is None:
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32)

    lookup = _as_extrap_lookup_arrays(extrap_lookup)
    lookup_uv, lookup_world = _coerce_uv_world(
        lookup.uv,
        lookup.world,
        uv_dtype=np.int64,
        world_dtype=np.float32,
    )
    if lookup_uv.shape[0] == 0:
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32)

    finite_world = np.isfinite(lookup_world).all(axis=1)
    if not finite_world.any():
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32)
    finite_idx = np.nonzero(finite_world)[0]
    finite_uv = lookup_uv[finite_idx]
    selected_finite_idx = _select_extrap_uv_indices_for_sampling(
        finite_uv,
        grow_direction,
        max_lines=max_lines,
    )
    if selected_finite_idx.shape[0] == 0:
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32)
    selected_idx = finite_idx[selected_finite_idx]
    sampled_uv = lookup_uv[selected_idx]
    sampled_world = lookup_world[selected_idx]

    return (
        sampled_uv.astype(np.int64, copy=False),
        sampled_world.astype(np.float32, copy=False),
    )


def _lookup_extrap_for_uv_query_flat(uv_query_flat, extrap_lookup):
    uv_int = np.asarray(uv_query_flat, dtype=np.int64)
    if uv_int.ndim != 2 or uv_int.shape[1] != 2 or uv_int.shape[0] == 0:
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32)

    lookup = _as_extrap_lookup_arrays(extrap_lookup)
    lookup_uv, lookup_world = _coerce_uv_world(
        lookup.uv,
        lookup.world,
        uv_dtype=np.int64,
        world_dtype=np.float32,
    )
    if lookup_uv.shape[0] == 0:
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32)

    lookup_sorted = lookup.lookup_uv_sorted
    lookup_sort = lookup.lookup_sort_idx
    query_view = _uv_struct_view(uv_int)
    pos = np.searchsorted(lookup_sorted, query_view, side="left")
    in_bounds = pos < lookup_sorted.shape[0]
    matched = np.zeros((uv_int.shape[0],), dtype=bool)
    if in_bounds.any():
        pos_in = pos[in_bounds]
        matched[in_bounds] = lookup_sorted[pos_in] == query_view[in_bounds]
    if not matched.any():
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32)

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
    uv_cond_int = _coerce_uv_int_array(uv_cond, prefer_int32=True, default_dtype=np.int32)
    crop_size = tuple(int(v) for v in crop_size)
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    crop_size_arr_f32 = crop_size_arr.astype(np.float32, copy=False)
    volume_for_crops = _resolve_segment_volume(tgt_segment, volume_scale=volume_scale)

    cond_rows, cond_cols = np.where(cond_valid_base)
    if cond_rows.size == 0:
        cond_uv_all = np.zeros((0, 2), dtype=uv_cond_int.dtype)
        cond_world_all = np.zeros((0, 3), dtype=np.float32)
    else:
        cond_uv_all = uv_cond_int[cond_rows, cond_cols].astype(uv_cond_int.dtype, copy=False)
        cond_world_all = cond_zyxs32[cond_rows, cond_cols].astype(np.float32, copy=False)
    cond_world_z = cond_world_all[:, 0]
    cond_world_y = cond_world_all[:, 1]
    cond_world_x = cond_world_all[:, 2]

    bbox_crops = []

    for bbox_idx, bbox in enumerate(bboxes):
        min_corner, _ = _bbox_to_min_corner_and_bounds_array(bbox)
        min_corner32 = min_corner.astype(np.float32, copy=False)
        vol_crop = _crop_volume_from_min_corner(volume_for_crops, min_corner, crop_size)
        vol_crop = normalize_zscore(vol_crop)

        max_corner_exclusive32 = min_corner32 + crop_size_arr_f32
        cond_in_bounds = (
            (cond_world_z >= min_corner32[0]) &
            (cond_world_z < max_corner_exclusive32[0]) &
            (cond_world_y >= min_corner32[1]) &
            (cond_world_y < max_corner_exclusive32[1]) &
            (cond_world_x >= min_corner32[2]) &
            (cond_world_x < max_corner_exclusive32[2])
        )
        if not bool(cond_in_bounds.any()):
            cond_uv = np.zeros((0, 2), dtype=cond_uv_all.dtype)
            cond_world = np.zeros((0, 3), dtype=np.float32)
            cond_local = np.zeros((0, 3), dtype=np.float32)
        else:
            cond_uv = cond_uv_all[cond_in_bounds].astype(cond_uv_all.dtype, copy=False)
            cond_world = cond_world_all[cond_in_bounds].astype(np.float32, copy=False)
            cond_local = (cond_world - min_corner32[None, :]).astype(np.float32, copy=False)
        cond_vox = _points_to_voxels(cond_local, crop_size)
        uv_query = _build_uv_query_from_edge_band(cond_uv, grow_direction, cond_pct)
        uv_query_flat = _coerce_uv_int_array(
            uv_query.reshape(-1, 2),
            prefer_int32=True,
            default_dtype=np.int32,
        )

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

def _run_inference(args, bbox_crops, model_state, profiler, verbose=True):
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
        with profiler.section("iter_infer_batch_pack_np"):
            batch_len = len(batch)
            first_vol = np.asarray(batch[0]["volume"], dtype=np.float32)
            if first_vol.ndim != 3:
                raise RuntimeError(f"Expected crop volume shape [D, H, W], got {tuple(first_vol.shape)}")
            d, h, w = first_vol.shape
            batch_np = np.empty((batch_len, expected_in_channels, d, h, w), dtype=np.float32)

            for i, crop in enumerate(batch):
                vol_np = np.asarray(crop["volume"], dtype=np.float32)
                cond_np = np.asarray(crop["cond_vox"], dtype=np.float32)
                if vol_np.shape != (d, h, w):
                    raise RuntimeError(
                        f"All crop volumes in a batch must share shape {(d, h, w)}; got {tuple(vol_np.shape)}."
                    )
                if cond_np.shape != (d, h, w):
                    raise RuntimeError(
                        f"cond_vox shape must match volume shape {(d, h, w)}; got {tuple(cond_np.shape)}."
                    )

                batch_np[i, 0] = vol_np
                batch_np[i, 1] = cond_np

                if expected_in_channels == 3:
                    extrap_np = np.asarray(crop["extrap_vox"], dtype=np.float32)
                    if extrap_np.shape != (d, h, w):
                        raise RuntimeError(
                            f"extrap_vox shape must match volume shape {(d, h, w)}; got {tuple(extrap_np.shape)}."
                        )
                    batch_np[i, 2] = extrap_np

        with profiler.section("iter_infer_h2d"):
            model_inputs = torch.from_numpy(batch_np).to(args.device, non_blocking=True)
        with profiler.section("iter_infer_model_forward"):
            disp_pred = predict_displacement(
                args,
                model_state,
                model_inputs,
                use_tta=use_tta,
                profiler=profiler,
            )
        with profiler.section("iter_infer_d2h_convert"):
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
        with profiler.section("iter_infer_collect_outputs"):
            for i, crop in enumerate(batch):
                per_crop_fields.append(
                    {
                        "bbox_idx": int(crop["bbox_idx"]),
                        "min_corner": np.asarray(crop["min_corner"], dtype=np.int64),
                        "displacement": disp_pred[i],
                    }
                )
        del model_inputs
        del batch_np
        del disp_pred
        done = (batch_start // batch_size) + 1
        if verbose:
            print(f"{desc}: batch {done}/{n_batches}")

    return per_crop_fields


def _empty_stack_samples():
    return {
        "uv": _empty_uv(dtype=np.int64),
        "world": _empty_world(dtype=np.float32),
        "displacement": _empty_world(dtype=np.float32),
        "stack_count": _empty_counts(dtype=np.uint32),
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


def _local_coords_in_bounds(world_points, min_corner, shape_dhw):
    d, h, w = (int(shape_dhw[0]), int(shape_dhw[1]), int(shape_dhw[2]))
    min_corner = np.asarray(min_corner, dtype=np.float32)
    world_points = np.asarray(world_points, dtype=np.float32)

    z_local = world_points[:, 0] - float(min_corner[0])
    y_local = world_points[:, 1] - float(min_corner[1])
    x_local = world_points[:, 2] - float(min_corner[2])
    in_bounds = (
        (z_local >= 0.0) & (z_local <= float(d - 1)) &
        (y_local >= 0.0) & (y_local <= float(h - 1)) &
        (x_local >= 0.0) & (x_local <= float(w - 1))
    )
    point_idx = np.nonzero(in_bounds)[0]
    if point_idx.size == 0:
        return point_idx, np.zeros((0, 3), dtype=np.float32)
    coords_local = np.stack(
        [
            z_local[point_idx],
            y_local[point_idx],
            x_local[point_idx],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    return point_idx, coords_local


def _sample_displacement_for_extrap_uvs_from_crops(
    per_crop_fields,
    extrap_lookup,
    grow_direction,
    max_lines=None,
    refine=None,
):
    if per_crop_fields is None or len(per_crop_fields) == 0 or extrap_lookup is None:
        return _empty_stack_samples()

    sampled_uv, sampled_world = _prepare_sampled_extrap_points(
        extrap_lookup,
        grow_direction,
        max_lines=max_lines,
    )
    if sampled_uv.shape[0] == 0:
        return _empty_stack_samples()
    n_points = int(sampled_world.shape[0])

    disp_acc = np.zeros((n_points, 3), dtype=np.float32)
    count_acc = np.zeros((n_points,), dtype=np.uint32)
    for item in per_crop_fields:
        disp_t = item.get("_disp_t")
        if disp_t is None:
            disp_t = _as_displacement_tensor(item["displacement"])
            item["_disp_t"] = disp_t
        min_corner = np.asarray(item["min_corner"], dtype=np.float32)
        _, _, d, h, w = disp_t.shape

        point_idx, coords_local = _local_coords_in_bounds(
            sampled_world,
            min_corner,
            (d, h, w),
        )
        if point_idx.size == 0:
            continue

        sample_fn = _sample_trilinear_displacement_stack if refine is None else _sample_fractional_displacement_stack
        sample_kwargs = {} if refine is None else {"refine_extra_steps": int(refine)}
        sampled_disp, valid_mask = sample_fn(
            disp_t,
            coords_local,
            **sample_kwargs,
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


def _sample_fractional_displacement_stack(disp, coords_local, refine_extra_steps):
    coords_local = np.asarray(coords_local, dtype=np.float32)
    if coords_local.ndim != 2 or coords_local.shape[1] != 3 or coords_local.shape[0] == 0:
        return (
            _empty_world(dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    disp_t = _as_displacement_tensor(disp)
    refine_extra_steps = max(int(refine_extra_steps), 0)
    refine_parts = refine_extra_steps + 1
    refine_fraction = 1.0 / float(refine_parts)

    start_coords_t = torch.from_numpy(coords_local.copy())
    current_coords_t = start_coords_t.clone()
    ever_valid_t = torch.zeros((coords_local.shape[0],), dtype=torch.bool)

    for _ in range(refine_parts):
        stage_disp_t, stage_valid_t = _sample_trilinear_displacement_stack_tensor(
            disp_t,
            current_coords_t,
        )
        if not bool(stage_valid_t.any()):
            continue
        delta_t = stage_disp_t * refine_fraction
        finite_delta_t = torch.isfinite(delta_t).all(dim=1)
        apply_mask_t = stage_valid_t & finite_delta_t
        if not bool(apply_mask_t.any()):
            continue
        current_coords_t[apply_mask_t] = current_coords_t[apply_mask_t] + delta_t[apply_mask_t]
        ever_valid_t |= apply_mask_t

    sampled_disp_t = current_coords_t - start_coords_t
    finite_disp_t = torch.isfinite(sampled_disp_t).all(dim=1)
    if not bool(finite_disp_t.all()):
        sampled_disp_t = torch.where(
            finite_disp_t[:, None],
            sampled_disp_t,
            torch.zeros_like(sampled_disp_t),
        )
        ever_valid_t &= finite_disp_t

    return (
        sampled_disp_t.numpy().astype(np.float32, copy=False),
        ever_valid_t.numpy().astype(bool, copy=False),
    )


def _sample_extrap_no_disp(extrap_lookup, grow_direction, max_lines=None):
    sampled_uv, sampled_world = _prepare_sampled_extrap_points(
        extrap_lookup,
        grow_direction,
        max_lines=max_lines,
    )
    if sampled_uv.shape[0] == 0:
        return _empty_stack_samples()
    n = int(sampled_world.shape[0])
    return {
        "uv": sampled_uv,
        "world": sampled_world,
        "displacement": np.zeros((n, 3), dtype=np.float32),
        "stack_count": np.ones((n,), dtype=np.uint32),
    }


def _as_displacement_tensor(disp):
    if torch.is_tensor(disp):
        disp_t = disp.detach()
        if disp_t.ndim == 4:
            disp_t = disp_t.unsqueeze(0)
        if disp_t.ndim != 5 or disp_t.shape[0] != 1 or disp_t.shape[1] < 3:
            raise RuntimeError(f"Expected displacement tensor with shape [1, 3+, D, H, W], got {tuple(disp_t.shape)}")
        return disp_t[:, :3].to(dtype=torch.float32, device="cpu").contiguous()

    disp_np = np.asarray(disp, dtype=np.float32)
    if disp_np.ndim != 4 or disp_np.shape[0] < 3:
        raise RuntimeError(f"Expected displacement array with shape [3+, D, H, W], got {tuple(disp_np.shape)}")
    return torch.from_numpy(disp_np[:3]).unsqueeze(0).contiguous()


def _coords_local_to_grid(coords_t, d, h, w):
    d_denom = max(int(d) - 1, 1)
    h_denom = max(int(h) - 1, 1)
    w_denom = max(int(w) - 1, 1)
    coords_norm = coords_t.clone()
    coords_norm[:, 0] = 2.0 * coords_norm[:, 0] / float(d_denom) - 1.0
    coords_norm[:, 1] = 2.0 * coords_norm[:, 1] / float(h_denom) - 1.0
    coords_norm[:, 2] = 2.0 * coords_norm[:, 2] / float(w_denom) - 1.0
    return coords_norm[:, [2, 1, 0]].view(1, -1, 1, 1, 3)


def _sample_trilinear_displacement_stack_tensor(disp_t, coords_local):
    if coords_local is None:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.bool),
        )

    if torch.is_tensor(coords_local):
        coords_t = coords_local.to(dtype=torch.float32, device="cpu")
    else:
        coords_np = np.asarray(coords_local, dtype=np.float32)
        if coords_np.ndim != 2 or coords_np.shape[1] != 3:
            return (
                torch.zeros((0, 3), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.bool),
            )
        coords_t = torch.from_numpy(coords_np)

    if coords_t.ndim != 2 or coords_t.shape[1] != 3 or coords_t.shape[0] == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.bool),
        )

    _, _, d, h, w = disp_t.shape
    valid_mask_t = (
        (coords_t[:, 0] >= 0.0) & (coords_t[:, 0] <= float(d - 1)) &
        (coords_t[:, 1] >= 0.0) & (coords_t[:, 1] <= float(h - 1)) &
        (coords_t[:, 2] >= 0.0) & (coords_t[:, 2] <= float(w - 1))
    )

    sampled_disp_t = torch.zeros((coords_t.shape[0], 3), dtype=disp_t.dtype, device=disp_t.device)
    if bool(valid_mask_t.any()):
        valid_coords_t = coords_t[valid_mask_t]
        grid = _coords_local_to_grid(valid_coords_t, d=d, h=h, w=w)
        sampled_valid_t = F.grid_sample(
            disp_t,
            grid,
            mode="bilinear",
            align_corners=True,
        ).view(3, -1).permute(1, 0)
        sampled_disp_t[valid_mask_t] = sampled_valid_t
    return sampled_disp_t, valid_mask_t


def _sample_trilinear_displacement_stack(disp, coords_local):
    if coords_local is None or len(coords_local) == 0:
        return (
            _empty_world(dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    disp_t = _as_displacement_tensor(disp)
    sampled_disp_t, valid_mask_t = _sample_trilinear_displacement_stack_tensor(disp_t, coords_local)
    return (
        sampled_disp_t.numpy().astype(np.float32, copy=False),
        valid_mask_t.numpy().astype(bool, copy=False),
    )


def _empty_displaced_samples():
    return {
        "uv": _empty_uv(dtype=np.int64),
        "world": _empty_world(dtype=np.float32),
        "displacement": _empty_world(dtype=np.float32),
        "world_displaced": _empty_world(dtype=np.float32),
        "stack_count": _empty_counts(dtype=np.uint32),
    }


def _expand_surface_canvas_to_fit_points(
    merged_zyxs,
    merged_valid,
    uv_n,
    uv_offset,
    original_shape,
    original_offset,
    verbose=True,
):
    r0, c0 = int(uv_offset[0]), int(uv_offset[1])
    h, w = merged_valid.shape
    if uv_n.size == 0:
        return merged_zyxs, merged_valid, r0, c0, h, w

    min_r = min(r0, int(uv_n[:, 0].min()))
    max_r = max(r0 + h - 1, int(uv_n[:, 0].max()))
    min_c = min(c0, int(uv_n[:, 1].min()))
    max_c = max(c0 + w - 1, int(uv_n[:, 1].max()))
    new_h = max_r - min_r + 1
    new_w = max_c - min_c + 1
    if new_h == h and new_w == w:
        return merged_zyxs, merged_valid, r0, c0, h, w

    expanded_zyxs = np.full((new_h, new_w, 3), -1.0, dtype=np.float32)
    expanded_valid = np.zeros((new_h, new_w), dtype=bool)
    rr0 = r0 - min_r
    cc0 = c0 - min_c
    expanded_zyxs[rr0:rr0 + h, cc0:cc0 + w] = merged_zyxs
    expanded_valid[rr0:rr0 + h, cc0:cc0 + w] = merged_valid

    _print_section_header(verbose, "Merge Displaced Into Full Surface")
    if verbose:
        print(
            f"expanded UV canvas: shape ({original_shape[0]}, {original_shape[1]}) -> ({new_h}, {new_w}), "
            f"offset ({int(original_offset[0])}, {int(original_offset[1])}) -> ({min_r}, {min_c})"
        )
    return expanded_zyxs, expanded_valid, min_r, min_c, new_h, new_w


def _compute_merge_write_indices(uv_n, pts_n, r0, c0, h, w):
    idx_dtype = np.asarray(uv_n).dtype if np.asarray(uv_n).dtype.kind in "iu" else np.int64
    if uv_n.shape[0] == 0 or pts_n.shape[0] == 0:
        return (
            np.zeros((0,), dtype=idx_dtype),
            np.zeros((0,), dtype=idx_dtype),
            np.zeros((0,), dtype=idx_dtype),
            0,
            0,
        )

    finite_mask = np.isfinite(pts_n).all(axis=1)
    r0_arr = np.asarray(r0, dtype=idx_dtype)
    c0_arr = np.asarray(c0, dtype=idx_dtype)
    rr_all = uv_n[:, 0].astype(idx_dtype, copy=False) - r0_arr
    cc_all = uv_n[:, 1].astype(idx_dtype, copy=False) - c0_arr
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
    return rr_all, cc_all, write_indices, n_nonfinite, n_out_of_bounds


def _print_section_header(verbose, title):
    if verbose:
        print(f"== {title} ==")


def _apply_displacement(samples, verbose=True, skip_inference=False):
    world = np.asarray(samples.get("world", _empty_world(dtype=np.float32)), dtype=np.float32)
    displacement = np.asarray(samples.get("displacement", _empty_world(dtype=np.float32)), dtype=np.float32)
    uv = np.asarray(samples.get("uv", _empty_uv(dtype=np.int64)), dtype=np.int64)
    stack_count = np.asarray(samples.get("stack_count", _empty_counts(dtype=np.uint32)), dtype=np.uint32)

    if skip_inference:
        n = int(min(uv.shape[0], world.shape[0], stack_count.shape[0]))
        if n == 0:
            _print_section_header(verbose, "Applied Displacement")
            if verbose:
                print("Extrap-only mode: no extrapolated points selected to merge.")
            return _empty_displaced_samples()
        uv = uv[:n].astype(np.int64, copy=False)
        world = world[:n].astype(np.float32, copy=False)
        stack_count = stack_count[:n].astype(np.uint32, copy=False)
        _print_section_header(verbose, "Applied Displacement")
        if verbose:
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
        _print_section_header(verbose, "Applied Displacement")
        if verbose:
            print("No sampled points to apply displacement.")
        return _empty_displaced_samples()

    world_displaced = world + displacement
    disp_norm = np.linalg.norm(displacement.astype(np.float32, copy=False), axis=1)

    _print_section_header(verbose, "Applied Displacement")
    if verbose:
        print(f"n points: {world.shape[0]}")
        print(
            "disp_norm min/max/mean/median: "
            f"{disp_norm.min():.4f} / {disp_norm.max():.4f} / {disp_norm.mean():.4f} / {np.median(disp_norm):.4f}"
        )

        axis_names = ("z", "y", "x")
        for axis_idx, axis_name in enumerate(axis_names):
            vals = world_displaced[:, axis_idx].astype(np.float32, copy=False)
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

    uv = np.asarray(displaced.get("uv", _empty_uv(dtype=np.int64)), dtype=np.int64)
    pts = np.asarray(displaced.get("world_displaced", _empty_world(dtype=np.float32)), dtype=np.float32)

    if uv.shape[0] == 0 or pts.shape[0] == 0:
        _print_section_header(verbose, "Merge Displaced Into Full Surface")
        if verbose:
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
    uv_n = _coerce_uv_int_array(uv[:n], prefer_int32=True, default_dtype=np.int32)
    pts_n = pts[:n].astype(np.float32, copy=False)
    merged_zyxs, merged_valid, r0, c0, h, w = _expand_surface_canvas_to_fit_points(
        merged_zyxs,
        merged_valid,
        uv_n,
        (r0, c0),
        cond_valid.shape,
        cond_uv_offset,
        verbose=verbose,
    )

    n_written = 0
    n_new_valid = 0
    n_overwrite_existing = 0
    rr_all, cc_all, write_indices, n_nonfinite, n_out_of_bounds = _compute_merge_write_indices(
        uv_n,
        pts_n,
        r0,
        c0,
        h,
        w,
    )

    if write_indices.shape[0] > 0:
        flat_dtype = _flat_index_dtype_for_shape(h, w, prefer_int32=True)
        rr = rr_all[write_indices].astype(flat_dtype, copy=False)
        cc = cc_all[write_indices].astype(flat_dtype, copy=False)
        pts_write = pts_n[write_indices].astype(np.float32, copy=False)
        w_arr = np.asarray(int(w), dtype=flat_dtype)
        flat = rr * w_arr + cc

        uniq_flat, counts = np.unique(flat, return_counts=True)
        rr_u = (uniq_flat // w_arr).astype(np.int64, copy=False)
        cc_u = (uniq_flat % w_arr).astype(np.int64, copy=False)
        pre_valid = merged_valid[rr_u, cc_u]

        n_written = int(write_indices.shape[0])
        n_new_valid = int((~pre_valid).sum())
        n_overwrite_existing = int((counts - (~pre_valid).astype(np.int64)).sum())

        # Preserve loop semantics: final value at each UV is from last write.
        _, rev_first_idx = np.unique(flat[::-1], return_index=True)
        last_pos = (flat.shape[0] - 1 - rev_first_idx).astype(np.int64, copy=False)
        rr_last = rr[last_pos]
        cc_last = cc[last_pos]
        pts_last = pts_write[last_pos]

        merged_zyxs[rr_last, cc_last] = pts_last
        merged_valid[rr_last, cc_last] = True

    _print_section_header(verbose, "Merge Displaced Into Full Surface")
    if verbose:
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


def _extreme_true_index(mask, axis, toward_max):
    mask = np.asarray(mask, dtype=bool)
    if toward_max:
        flipped = np.flip(mask, axis=axis)
        return mask.shape[axis] - 1 - np.argmax(flipped, axis=axis)
    return np.argmax(mask, axis=axis)


def _boundary_axis_value(valid, uv_offset, grow_direction):
    valid = np.asarray(valid, dtype=bool)
    if valid.size == 0 or not valid.any():
        return None

    _, growth_spec = _get_growth_context(grow_direction)
    r0, c0 = int(uv_offset[0]), int(uv_offset[1])

    is_col_growth = growth_spec["axis"] == "col"
    line_axis = 0 if is_col_growth else 1
    boundary_axis = 1 - line_axis
    line_valid = np.any(valid, axis=boundary_axis)
    if not line_valid.any():
        return None

    toward_max = growth_spec["growth_sign"] > 0
    boundary_rel = _extreme_true_index(valid, axis=boundary_axis, toward_max=toward_max)
    line_base = r0 if line_axis == 0 else c0
    boundary_base = c0 if boundary_axis == 1 else r0
    line_ids = np.where(line_valid)[0].astype(np.int64, copy=False) + line_base
    boundary_vals = boundary_rel[line_valid].astype(np.int64, copy=False) + boundary_base

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


def _report_iteration_stop(verbose, iteration_pbar, message, postfix=None):
    if verbose:
        print(message)
        return
    if iteration_pbar is not None and postfix is not None:
        iteration_pbar.set_postfix_str(postfix, refresh=True)


def _finite_valid_grid_points(grid, valid):
    rows, cols = np.where(np.asarray(valid, dtype=bool))
    if rows.size == 0:
        return _empty_counts(dtype=np.int64), _empty_counts(dtype=np.int64), _empty_world(dtype=np.float32)

    pts = np.asarray(grid, dtype=np.float32)[rows, cols].astype(np.float32, copy=False)
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        return _empty_counts(dtype=np.int64), _empty_counts(dtype=np.int64), _empty_world(dtype=np.float32)
    rows = rows[finite].astype(np.int64, copy=False)
    cols = cols[finite].astype(np.int64, copy=False)
    return rows, cols, pts[finite].astype(np.float32, copy=False)


def _surface_to_stored_uv_samples_lattice(
    grid,
    valid,
    uv_offset,
    sub_r,
    sub_c,
    phase_rc=(0, 0),
):
    """Project full-resolution UV samples onto a fixed stored-resolution lattice.

    For each occupied stored UV cell, sample the full-resolution surface at the
    corresponding lattice anchor using mask-aware bilinear interpolation via
    torch.grid_sample.
    """
    grid = np.asarray(grid, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    sub_r = max(1, int(sub_r))
    sub_c = max(1, int(sub_c))
    phase_r = int(phase_rc[0])
    phase_c = int(phase_rc[1])

    if grid.ndim != 3 or grid.shape[-1] != 3:
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32) + ({
            "mode": "lattice_bilinear_torch",
            "stride_rc": [sub_r, sub_c],
            "phase_rc": [phase_r, phase_c],
            "n_full_valid": 0,
            "n_stored_valid": 0,
        },)

    h, w = grid.shape[:2]
    support = valid & np.isfinite(grid).all(axis=2)
    n_full_valid = int(support.sum())
    if n_full_valid < 1:
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32) + ({
            "mode": "lattice_bilinear_torch",
            "stride_rc": [sub_r, sub_c],
            "phase_rc": [phase_r, phase_c],
            "n_full_valid": 0,
            "n_stored_valid": 0,
        },)

    # Deterministic stored-lattice sampling: query at each stored cell center
    # in full-resolution UV so bilinear interpolation can blend neighbors.
    cell_center_r = 0.5 * float(sub_r - 1)
    cell_center_c = 0.5 * float(sub_c - 1)
    r_abs0 = int(uv_offset[0])
    c_abs0 = int(uv_offset[1])
    r_abs1 = r_abs0 + h - 1
    c_abs1 = c_abs0 + w - 1
    s_r_min = int(np.ceil((r_abs0 - (phase_r + cell_center_r)) / float(sub_r)))
    s_r_max = int(np.floor((r_abs1 - (phase_r + cell_center_r)) / float(sub_r)))
    s_c_min = int(np.ceil((c_abs0 - (phase_c + cell_center_c)) / float(sub_c)))
    s_c_max = int(np.floor((c_abs1 - (phase_c + cell_center_c)) / float(sub_c)))
    if s_r_max < s_r_min or s_c_max < s_c_min:
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32) + ({
            "mode": "lattice_bilinear_torch",
            "stride_rc": [sub_r, sub_c],
            "phase_rc": [phase_r, phase_c],
            "n_full_valid": n_full_valid,
            "n_stored_valid": 0,
        },)
    stored_rows = np.arange(s_r_min, s_r_max + 1, dtype=np.int64)
    stored_cols = np.arange(s_c_min, s_c_max + 1, dtype=np.int64)
    sr, sc = np.meshgrid(stored_rows, stored_cols, indexing="ij")
    uv_q = np.stack([sr.reshape(-1), sc.reshape(-1)], axis=-1).astype(np.int64, copy=False)

    # Stored cell centers in absolute full-resolution UV coordinates.
    q_r_abs = (
        uv_q[:, 0].astype(np.float32, copy=False) * float(sub_r) +
        float(phase_r) + float(cell_center_r)
    )
    q_c_abs = (
        uv_q[:, 1].astype(np.float32, copy=False) * float(sub_c) +
        float(phase_c) + float(cell_center_c)
    )
    q_r = (q_r_abs - float(uv_offset[0])).astype(np.float32, copy=False)
    q_c = (q_c_abs - float(uv_offset[1])).astype(np.float32, copy=False)

    in_grid = (
        (q_r >= 0.0) &
        (q_r <= float(max(h - 1, 0))) &
        (q_c >= 0.0) &
        (q_c <= float(max(w - 1, 0)))
    )
    if not bool(in_grid.any()):
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32) + ({
            "mode": "lattice_bilinear_torch",
            "stride_rc": [sub_r, sub_c],
            "phase_rc": [phase_r, phase_c],
            "n_full_valid": n_full_valid,
            "n_stored_valid": 0,
        },)

    uv_q = uv_q[in_grid].astype(np.int64, copy=False)
    q_r = q_r[in_grid].astype(np.float32, copy=False)
    q_c = q_c[in_grid].astype(np.float32, copy=False)

    denom_r = float(max(h - 1, 1))
    denom_c = float(max(w - 1, 1))
    y_norm = (2.0 * q_r / denom_r) - 1.0
    x_norm = (2.0 * q_c / denom_c) - 1.0
    query_grid = np.stack([x_norm, y_norm], axis=-1).astype(np.float32, copy=False)

    value_np = np.where(support[..., None], grid, 0.0).astype(np.float32, copy=False)
    value_t = torch.from_numpy(value_np.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    mask_t = torch.from_numpy(support.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0).contiguous()
    query_t = torch.from_numpy(query_grid.reshape(1, -1, 1, 2)).contiguous()

    with torch.no_grad():
        sampled_num_t = F.grid_sample(
            value_t,
            query_t,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_den_t = F.grid_sample(
            mask_t,
            query_t,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    sampled_num = (
        sampled_num_t[0, :, :, 0]
        .permute(1, 0)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    sampled_den = sampled_den_t[0, 0, :, 0].cpu().numpy().astype(np.float32, copy=False)
    can_sample = sampled_den > 1e-6
    if not bool(can_sample.any()):
        return _empty_uv_world(uv_dtype=np.int64, world_dtype=np.float32) + ({
            "mode": "lattice_bilinear_torch",
            "stride_rc": [sub_r, sub_c],
            "phase_rc": [phase_r, phase_c],
            "n_full_valid": n_full_valid,
            "n_stored_valid": 0,
        },)

    uv_keep = uv_q[can_sample].astype(np.int64, copy=False)
    pts_bilinear = (
        sampled_num[can_sample] / sampled_den[can_sample, None]
    ).astype(np.float32, copy=False)

    projection_meta = {
        "mode": "lattice_bilinear_torch",
        "stride_rc": [sub_r, sub_c],
        "phase_rc": [phase_r, phase_c],
        "n_full_valid": n_full_valid,
        "n_stored_valid": int(uv_keep.shape[0]),
    }
    return uv_keep, pts_bilinear, projection_meta


def _infer_lattice_phase_rc(
    stored_zyxs,
    stored_valid,
    full_zyxs,
    full_valid,
    full_uv_offset,
    sub_r,
    sub_c,
    max_compare_points=50000,
):
    stored_zyxs = np.asarray(stored_zyxs, dtype=np.float32)
    stored_valid = np.asarray(stored_valid, dtype=bool)
    full_zyxs = np.asarray(full_zyxs, dtype=np.float32)
    full_valid = np.asarray(full_valid, dtype=bool)
    sub_r = max(1, int(sub_r))
    sub_c = max(1, int(sub_c))
    r0 = int(full_uv_offset[0])
    c0 = int(full_uv_offset[1])

    if sub_r == 1 and sub_c == 1:
        return (0, 0), {"mode": "inferred", "pairs": 0, "median_err": 0.0}
    if (
        stored_zyxs.ndim != 3
        or stored_zyxs.shape[-1] != 3
        or stored_valid.shape != stored_zyxs.shape[:2]
        or full_zyxs.ndim != 3
        or full_zyxs.shape[-1] != 3
        or full_valid.shape != full_zyxs.shape[:2]
    ):
        return (0, 0), {"mode": "fallback_invalid_input", "pairs": 0, "median_err": None}

    hs, ws = stored_valid.shape
    hf, wf = full_valid.shape
    if hs < 1 or ws < 1 or hf < 1 or wf < 1:
        return (0, 0), {"mode": "fallback_empty", "pairs": 0, "median_err": None}

    stored_support = stored_valid & np.isfinite(stored_zyxs).all(axis=2)
    full_support = full_valid & np.isfinite(full_zyxs).all(axis=2)
    if not stored_support.any() or not full_support.any():
        return (0, 0), {"mode": "fallback_no_support", "pairs": 0, "median_err": None}

    row_abs = np.arange(r0, r0 + hf, dtype=np.int64)
    col_abs = np.arange(c0, c0 + wf, dtype=np.int64)

    best_phase = (0, 0)
    best_pairs = -1
    best_median = np.inf

    for phase_r in range(sub_r):
        row_local = np.where(((row_abs - phase_r) % sub_r) == 0)[0]
        if row_local.size == 0:
            continue
        row_stored = np.floor_divide(row_abs[row_local] - phase_r, sub_r).astype(np.int64, copy=False)
        row_in = (row_stored >= 0) & (row_stored < hs)
        if not row_in.any():
            continue
        row_local = row_local[row_in]
        row_stored = row_stored[row_in]

        for phase_c in range(sub_c):
            col_local = np.where(((col_abs - phase_c) % sub_c) == 0)[0]
            if col_local.size == 0:
                continue
            col_stored = np.floor_divide(col_abs[col_local] - phase_c, sub_c).astype(np.int64, copy=False)
            col_in = (col_stored >= 0) & (col_stored < ws)
            if not col_in.any():
                continue
            col_local = col_local[col_in]
            col_stored = col_stored[col_in]

            rr_local, cc_local = np.meshgrid(row_local, col_local, indexing="ij")
            rr_stored, cc_stored = np.meshgrid(row_stored, col_stored, indexing="ij")
            rr_local = rr_local.reshape(-1)
            cc_local = cc_local.reshape(-1)
            rr_stored = rr_stored.reshape(-1)
            cc_stored = cc_stored.reshape(-1)
            if rr_local.size == 0:
                continue

            pair_mask = full_support[rr_local, cc_local] & stored_support[rr_stored, cc_stored]
            n_pairs = int(pair_mask.sum())
            if n_pairs < 1:
                continue

            sel = np.nonzero(pair_mask)[0]
            if sel.size > int(max_compare_points):
                sel = sel[: int(max_compare_points)]

            full_pts = full_zyxs[rr_local[sel], cc_local[sel]].astype(np.float32, copy=False)
            stored_pts = stored_zyxs[rr_stored[sel], cc_stored[sel]].astype(np.float32, copy=False)
            errs = np.linalg.norm((full_pts - stored_pts).astype(np.float32, copy=False), axis=1)
            median_err = float(np.median(errs)) if errs.size > 0 else float("inf")

            if (n_pairs > best_pairs) or (n_pairs == best_pairs and median_err < best_median):
                best_pairs = n_pairs
                best_median = median_err
                best_phase = (int(phase_r), int(phase_c))

    if best_pairs < 1:
        return (0, 0), {"mode": "fallback_no_pairs", "pairs": 0, "median_err": None}
    return best_phase, {
        "mode": "inferred",
        "pairs": int(best_pairs),
        "median_err": float(best_median),
    }


def _stored_uv_step_spacing_stats(uv, pts):
    uv = np.asarray(uv, dtype=np.int64)
    pts = np.asarray(pts, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] < 2:
        return None

    dists = []
    # Evaluate per-step spacing along stored rows for each stored column.
    unique_cols = np.unique(uv[:, 1])
    for col in unique_cols:
        idx = np.where(uv[:, 1] == col)[0]
        if idx.size < 2:
            continue
        rr = uv[idx, 0]
        pp = pts[idx]
        order = np.argsort(rr)
        rr = rr[order]
        pp = pp[order]
        dr = np.diff(rr)
        if dr.size == 0:
            continue
        keep = dr > 0
        if not keep.any():
            continue
        dp = np.linalg.norm((pp[1:] - pp[:-1]).astype(np.float32, copy=False), axis=1)
        dists.extend((dp[keep] / dr[keep]).tolist())

    if len(dists) == 0:
        return None
    arr = np.asarray(dists, dtype=np.float32)
    return {
        "n_pairs": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90.0)),
    }


def _filter_stored_merge_samples_to_new_cells(uv, pts, base_valid):
    uv = np.asarray(uv, dtype=np.int64)
    pts = np.asarray(pts, dtype=np.float32)
    base_valid = np.asarray(base_valid, dtype=bool)
    if uv.ndim != 2 or uv.shape[1] != 2 or pts.ndim != 2 or pts.shape[1] != 3:
        return _empty_uv(dtype=np.int64), _empty_world(dtype=np.float32), {
            "input_points": 0,
            "kept_points": 0,
            "dropped_existing_valid": 0,
            "kept_outside_base": 0,
            "kept_inside_invalid": 0,
        }

    n = min(int(uv.shape[0]), int(pts.shape[0]))
    if n < 1:
        return _empty_uv(dtype=np.int64), _empty_world(dtype=np.float32), {
            "input_points": 0,
            "kept_points": 0,
            "dropped_existing_valid": 0,
            "kept_outside_base": 0,
            "kept_inside_invalid": 0,
        }

    uv_n = uv[:n].astype(np.int64, copy=False)
    pts_n = pts[:n].astype(np.float32, copy=False)
    h, w = base_valid.shape
    inside = (
        (uv_n[:, 0] >= 0) &
        (uv_n[:, 0] < h) &
        (uv_n[:, 1] >= 0) &
        (uv_n[:, 1] < w)
    )
    keep = ~inside
    if inside.any():
        rr = uv_n[inside, 0]
        cc = uv_n[inside, 1]
        inside_keep = ~base_valid[rr, cc]
        keep[inside] = inside_keep

    kept_uv = uv_n[keep].astype(np.int64, copy=False)
    kept_pts = pts_n[keep].astype(np.float32, copy=False)
    kept_inside = int((inside & keep).sum())
    kept_outside = int((~inside & keep).sum())
    dropped_existing = int((inside & ~keep).sum())
    meta = {
        "input_points": int(n),
        "kept_points": int(kept_uv.shape[0]),
        "dropped_existing_valid": dropped_existing,
        "kept_outside_base": kept_outside,
        "kept_inside_invalid": kept_inside,
    }
    return kept_uv, kept_pts, meta


def _compose_full_projection_source_original_wins(
    current_zyxs,
    current_valid,
    current_uv_offset,
    stored_zyxs,
    stored_valid,
    sub_r,
    sub_c,
    phase_rc=(0, 0),
):
    src_zyxs = np.asarray(current_zyxs, dtype=np.float32).copy()
    src_valid = np.asarray(current_valid, dtype=bool).copy()
    stored_zyxs = np.asarray(stored_zyxs, dtype=np.float32)
    stored_valid = np.asarray(stored_valid, dtype=bool)
    sub_r = max(1, int(sub_r))
    sub_c = max(1, int(sub_c))
    phase_r = int(phase_rc[0])
    phase_c = int(phase_rc[1])

    if (
        src_zyxs.ndim != 3
        or src_zyxs.shape[-1] != 3
        or src_valid.shape != src_zyxs.shape[:2]
        or stored_zyxs.ndim != 3
        or stored_zyxs.shape[-1] != 3
        or stored_valid.shape != stored_zyxs.shape[:2]
    ):
        return src_zyxs, src_valid, {"anchors_written": 0, "anchors_candidates": 0}

    h, w = src_valid.shape
    r0 = int(current_uv_offset[0])
    c0 = int(current_uv_offset[1])

    stored_support = stored_valid & np.isfinite(stored_zyxs).all(axis=2)
    rr_s, cc_s = np.where(stored_support)
    if rr_s.size == 0:
        return src_zyxs, src_valid, {"anchors_written": 0, "anchors_candidates": 0}

    rr_abs = rr_s.astype(np.int64, copy=False) * int(sub_r) + int(phase_r)
    cc_abs = cc_s.astype(np.int64, copy=False) * int(sub_c) + int(phase_c)
    rr_local = rr_abs - int(r0)
    cc_local = cc_abs - int(c0)
    inside = (
        (rr_local >= 0)
        & (rr_local < h)
        & (cc_local >= 0)
        & (cc_local < w)
    )
    if not inside.any():
        return src_zyxs, src_valid, {"anchors_written": 0, "anchors_candidates": int(rr_s.size)}

    rr_local = rr_local[inside].astype(np.int64, copy=False)
    cc_local = cc_local[inside].astype(np.int64, copy=False)
    rr_s = rr_s[inside].astype(np.int64, copy=False)
    cc_s = cc_s[inside].astype(np.int64, copy=False)

    src_zyxs[rr_local, cc_local] = stored_zyxs[rr_s, cc_s].astype(np.float32, copy=False)
    src_valid[rr_local, cc_local] = True
    return src_zyxs, src_valid, {
        "anchors_written": int(rr_local.shape[0]),
        "anchors_candidates": int(rr_abs.shape[0]),
    }


def _normalize_world_step_in_new_band_along_growth(
    uv,
    pts,
    base_zyxs,
    base_valid,
    grow_direction,
    ref_pairs=4,
):
    uv = np.asarray(uv, dtype=np.int64)
    pts = np.asarray(pts, dtype=np.float32)
    base_zyxs = np.asarray(base_zyxs, dtype=np.float32)
    base_valid = np.asarray(base_valid, dtype=bool)
    if (
        uv.ndim != 2
        or uv.shape[1] != 2
        or pts.ndim != 2
        or pts.shape[1] != 3
        or uv.shape[0] != pts.shape[0]
        or base_zyxs.ndim != 3
        or base_zyxs.shape[-1] != 3
        or base_valid.shape != base_zyxs.shape[:2]
        or uv.shape[0] == 0
    ):
        return pts.astype(np.float32, copy=False), {
            "applied": False,
            "reason": "invalid_input",
            "adjusted_points": 0,
            "adjusted_lines": 0,
        }

    _, growth_spec = _get_growth_context(grow_direction)
    if growth_spec["axis"] == "col":
        line_axis = 0
        grow_axis = 1
    else:
        line_axis = 1
        grow_axis = 0
    growth_sign = int(growth_spec["growth_sign"])
    inward_sign = -growth_sign

    finite_base = base_valid & np.isfinite(base_zyxs).all(axis=2)
    if grow_axis == 1:
        pair_mask = finite_base[:, :-1] & finite_base[:, 1:]
        if pair_mask.any():
            global_d = np.linalg.norm(
                (base_zyxs[:, 1:, :] - base_zyxs[:, :-1, :]).astype(np.float32, copy=False),
                axis=2,
            )[pair_mask]
        else:
            global_d = np.zeros((0,), dtype=np.float32)
    else:
        pair_mask = finite_base[:-1, :] & finite_base[1:, :]
        if pair_mask.any():
            global_d = np.linalg.norm(
                (base_zyxs[1:, :, :] - base_zyxs[:-1, :, :]).astype(np.float32, copy=False),
                axis=2,
            )[pair_mask]
        else:
            global_d = np.zeros((0,), dtype=np.float32)
    global_ref = float(np.median(global_d)) if global_d.size > 0 else None

    uv_key_dtype = _flat_index_dtype_for_shape(
        max(int(np.max(uv[:, 0])) + 1, 1),
        max(int(np.max(uv[:, 1])) + 1, 1),
        prefer_int32=False,
    )
    rr = uv[:, 0].astype(uv_key_dtype, copy=False)
    cc = uv[:, 1].astype(uv_key_dtype, copy=False)
    stride = np.asarray(int(np.max(cc) + 2), dtype=uv_key_dtype)
    keys = rr * stride + cc
    key_to_idx = {int(k): int(i) for i, k in enumerate(keys.tolist())}

    pts_orig = pts.astype(np.float32, copy=False)
    pts_adj = pts_orig.copy()

    hs, ws = base_valid.shape
    adjusted_points = 0
    adjusted_lines = 0
    line_ids = np.unique(uv[:, line_axis])
    eps = 1e-6

    for line_id in line_ids.tolist():
        line_id = int(line_id)
        if grow_axis == 1:
            if line_id < 0 or line_id >= hs:
                continue
            old_g = np.where(finite_base[line_id, :])[0].astype(np.int64, copy=False)
        else:
            if line_id < 0 or line_id >= ws:
                continue
            old_g = np.where(finite_base[:, line_id])[0].astype(np.int64, copy=False)
        if old_g.size < 2:
            continue

        seam_g = int(old_g.max()) if growth_sign > 0 else int(old_g.min())
        if grow_axis == 1:
            seam_pt = base_zyxs[line_id, seam_g].astype(np.float32, copy=False)
        else:
            seam_pt = base_zyxs[seam_g, line_id].astype(np.float32, copy=False)
        if not np.isfinite(seam_pt).all():
            continue

        d_local = []
        for k in range(int(max(1, ref_pairs))):
            b = seam_g + inward_sign * k
            a = b + inward_sign
            if grow_axis == 1:
                if a < 0 or b < 0 or a >= ws or b >= ws:
                    continue
                if (not finite_base[line_id, a]) or (not finite_base[line_id, b]):
                    continue
                pa = base_zyxs[line_id, a].astype(np.float32, copy=False)
                pb = base_zyxs[line_id, b].astype(np.float32, copy=False)
            else:
                if a < 0 or b < 0 or a >= hs or b >= hs:
                    continue
                if (not finite_base[a, line_id]) or (not finite_base[b, line_id]):
                    continue
                pa = base_zyxs[a, line_id].astype(np.float32, copy=False)
                pb = base_zyxs[b, line_id].astype(np.float32, copy=False)
            d_local.append(float(np.linalg.norm((pb - pa).astype(np.float32, copy=False))))

        if len(d_local) > 0:
            d_ref = float(np.median(np.asarray(d_local, dtype=np.float32)))
        else:
            d_ref = global_ref
        if d_ref is None or (not np.isfinite(d_ref)) or d_ref <= eps:
            continue

        line_mask = uv[:, line_axis] == line_id
        if not bool(line_mask.any()):
            continue
        line_idx = np.nonzero(line_mask)[0]
        grow_vals = uv[line_idx, grow_axis].astype(np.int64, copy=False)
        if growth_sign > 0:
            new_mask = grow_vals > seam_g
            order = np.argsort(grow_vals, kind="mergesort")
        else:
            new_mask = grow_vals < seam_g
            order = np.argsort(-grow_vals, kind="mergesort")
        if not bool(new_mask.any()):
            continue
        new_idx = line_idx[new_mask]
        new_g = uv[new_idx, grow_axis].astype(np.int64, copy=False)
        if growth_sign > 0:
            sort_ord = np.argsort(new_g, kind="mergesort")
        else:
            sort_ord = np.argsort(-new_g, kind="mergesort")
        new_idx = new_idx[sort_ord]
        new_g = uv[new_idx, grow_axis].astype(np.int64, copy=False)

        # Seam tangent (inward -> seam) as fallback direction.
        g_inward = seam_g + inward_sign
        seam_dir = None
        if grow_axis == 1:
            if 0 <= g_inward < ws and finite_base[line_id, g_inward]:
                p_in = base_zyxs[line_id, g_inward].astype(np.float32, copy=False)
                v = seam_pt - p_in
                nv = float(np.linalg.norm(v))
                if nv > eps:
                    seam_dir = (v / nv).astype(np.float32, copy=False)
        else:
            if 0 <= g_inward < hs and finite_base[g_inward, line_id]:
                p_in = base_zyxs[g_inward, line_id].astype(np.float32, copy=False)
                v = seam_pt - p_in
                nv = float(np.linalg.norm(v))
                if nv > eps:
                    seam_dir = (v / nv).astype(np.float32, copy=False)
        if seam_dir is None:
            seam_dir = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

        prev_g = int(seam_g)
        prev_adj = seam_pt.astype(np.float32, copy=False)
        last_dir = seam_dir.astype(np.float32, copy=False)
        adjusted_this_line = 0

        for idx, g_val in zip(new_idx.tolist(), new_g.tolist()):
            idx = int(idx)
            g_val = int(g_val)
            if grow_axis == 1:
                key_prev = int(np.asarray(line_id, dtype=uv_key_dtype) * stride + np.asarray(prev_g, dtype=uv_key_dtype))
            else:
                key_prev = int(np.asarray(prev_g, dtype=uv_key_dtype) * stride + np.asarray(line_id, dtype=uv_key_dtype))
            idx_prev = key_to_idx.get(key_prev, None)
            orig_prev = seam_pt if idx_prev is None else pts_orig[idx_prev].astype(np.float32, copy=False)
            orig_cur = pts_orig[idx].astype(np.float32, copy=False)
            vec = (orig_cur - orig_prev).astype(np.float32, copy=False)
            nvec = float(np.linalg.norm(vec))
            if nvec > eps:
                dir_unit = (vec / nvec).astype(np.float32, copy=False)
            else:
                dir_unit = last_dir

            step_count = abs(int(g_val - prev_g))
            if step_count < 1:
                continue
            target_len = float(d_ref) * float(step_count)
            adj_cur = (prev_adj + dir_unit * target_len).astype(np.float32, copy=False)
            pts_adj[idx] = adj_cur

            prev_adj = adj_cur
            prev_g = g_val
            last_dir = dir_unit
            adjusted_points += 1
            adjusted_this_line += 1

        if adjusted_this_line > 0:
            adjusted_lines += 1

    return pts_adj.astype(np.float32, copy=False), {
        "applied": True,
        "reason": "ok",
        "adjusted_points": int(adjusted_points),
        "adjusted_lines": int(adjusted_lines),
        "global_ref_step": None if global_ref is None else float(global_ref),
        "growth_axis": "col" if grow_axis == 1 else "row",
        "growth_sign": int(growth_sign),
    }


def _build_iteration_extrap_lookup(edge_extrapolation):
    return _build_extrap_lookup_arrays(edge_extrapolation)


def _prune_edge_extrapolation_after_lookup(edge_extrapolation):
    pruned = dict(edge_extrapolation) if isinstance(edge_extrapolation, dict) else {}
    # Keep only lightweight metadata after extrapolation lookup construction.
    pruned["query_uv_grid"] = np.zeros((0, 0, 2), dtype=np.int64)
    pruned["extrapolated_local"] = _empty_world(dtype=np.float32)
    pruned["extrapolated_world"] = _empty_world(dtype=np.float32)
    return pruned


def _sample_iteration_agg_samples(
    args,
    profiler,
    grow_direction,
    extrap_only_mode,
    run_model_inference,
    model_state,
    bbox_crops,
    extrap_lookup,
):
    disp_bbox = None
    if extrap_only_mode:
        with profiler.section("iter_sample_extrap_no_disp"):
            agg_samples = _sample_extrap_no_disp(
                extrap_lookup,
                grow_direction,
                max_lines=args.agg_extrap_lines,
            )
        return agg_samples, disp_bbox

    if run_model_inference and model_state is not None:
        with profiler.section("iter_displacement_inference"):
            per_crop_fields = _run_inference(
                args,
                bbox_crops,
                model_state,
                profiler=profiler,
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
        return agg_samples, disp_bbox

    return _empty_stack_samples(), disp_bbox


def _iteration_stop_reason(
    merged_iter,
    prev_boundary,
    next_boundary,
    grow_direction,
):
    if int(merged_iter.get("n_new_valid", 0)) < 1:
        return "no_new_valid"
    if not _boundary_advanced(prev_boundary, next_boundary, grow_direction):
        return "boundary_unchanged"
    return None


def _iteration_should_stop(
    merged_iter,
    prev_boundary,
    next_boundary,
    grow_direction,
    n_bboxes,
    verbose,
    iteration_pbar,
):
    stop_reason = _iteration_stop_reason(
        merged_iter,
        prev_boundary,
        next_boundary,
        grow_direction,
    )
    if stop_reason == "no_new_valid":
        _report_iteration_stop(
            verbose,
            iteration_pbar,
            "No newly added valid points this iteration; stopping iterative growth.",
            postfix=f"bboxes={n_bboxes} | stopped: no new valid points",
        )
        return True
    if stop_reason == "boundary_unchanged":
        _report_iteration_stop(
            verbose,
            iteration_pbar,
            "Boundary did not advance this iteration; stopping iterative growth.",
            postfix=f"bboxes={n_bboxes} | stopped: boundary unchanged",
        )
        return True
    return False


def _available_growth_directions(valid_s):
    valid_s = np.asarray(valid_s, dtype=bool)
    valid_rows = np.any(valid_s, axis=1)
    valid_cols = np.any(valid_s, axis=0)
    directions = []
    if valid_cols.sum() >= 2:
        directions.extend(["left", "right"])
    if valid_rows.sum() >= 2:
        directions.extend(["up", "down"])
    return directions


def _resolve_growth_directions(requested_grow_direction, valid_s):
    if requested_grow_direction != "all":
        return [requested_grow_direction]

    available = set(_available_growth_directions(valid_s))
    resolved = [direction for direction in _ALL_GROW_DIRECTION_ORDER if direction in available]
    if len(resolved) == 0:
        raise RuntimeError("Segment too small to define a split direction.")
    return resolved


def _setup_segment_with_requested_direction(args, volume):
    if args.grow_direction != "all":
        return setup_segment(args, volume)

    unavailable_exc = None
    for candidate in _ALL_GROW_DIRECTION_ORDER:
        candidate_args = argparse.Namespace(**vars(args))
        candidate_args.grow_direction = candidate
        try:
            return setup_segment(candidate_args, volume)
        except RuntimeError as exc:
            if "not available for this segment" in str(exc):
                unavailable_exc = exc
                continue
            raise

    if unavailable_exc is not None:
        raise unavailable_exc
    raise RuntimeError("Unable to initialize segment for grow-direction=all.")


def _run_growth_direction_step(
    args,
    profiler,
    extrapolation_settings,
    tgt_segment,
    crop_size,
    extrap_only_mode,
    run_model_inference,
    model_state,
    iteration_pbar,
    grow_direction,
    cond_zyxs,
    cond_valid,
    cond_uv_offset,
    lattice_phase_rc,
    stop_is_skip=False,
):
    uv_cond = _build_uv_grid(cond_uv_offset, cond_zyxs.shape[:2])
    cond_direction, _ = _get_growth_context(grow_direction)
    scale_y, scale_x = tgt_segment._scale
    rbf_lattice_stride_rc = (
        _scale_to_subsample_stride(scale_y),
        _scale_to_subsample_stride(scale_x),
    )

    with profiler.section("iter_get_edge_bboxes"):
        bboxes, _ = get_cond_edge_bboxes(
            cond_zyxs,
            cond_direction,
            crop_size,
            overlap_frac=args.bbox_overlap_frac,
            cond_valid=cond_valid,
        )
    if iteration_pbar is not None:
        iteration_pbar.set_postfix_str(
            f"dir={grow_direction} | bboxes={len(bboxes)}",
            refresh=True,
        )
    if len(bboxes) == 0:
        if stop_is_skip:
            _report_iteration_stop(
                args.verbose,
                iteration_pbar,
                f"No edge bboxes for direction '{grow_direction}'; skipping this direction.",
                postfix=f"dir={grow_direction} | bboxes=0 | skipped: no edge bboxes",
            )
        else:
            _report_iteration_stop(
                args.verbose,
                iteration_pbar,
                "No edge bboxes available at current boundary; stopping iterative growth.",
                postfix=f"dir={grow_direction} | bboxes=0 | stopped: no edge bboxes",
            )
        return {
            "merged_iter": {
                "merged_zyxs": cond_zyxs,
                "merged_valid": cond_valid,
                "uv_offset": cond_uv_offset,
            },
            "bbox_results": [],
            "edge_extrapolation": _empty_edge_extrapolation(),
            "extrap_lookup": _empty_extrap_lookup_arrays(),
            "disp_bbox": None,
            "displaced": _empty_displaced_samples(),
            "stop_requested": True,
            "progressed": False,
        }

    with profiler.section("iter_edge_extrapolation"):
        edge_extrapolation = compute_edge_one_shot_extrapolation(
            cond_zyxs=cond_zyxs,
            cond_valid=cond_valid,
            uv_cond=uv_cond,
            grow_direction=grow_direction,
            edge_input_rowscols=args.edge_input_rowscols,
            cond_pct=args.cond_pct,
            method=extrapolation_settings["method"],
            min_corner=np.zeros(3, dtype=np.float32),
            crop_size=crop_size,
            degrade_prob=extrapolation_settings["degrade_prob"],
            degrade_curvature_range=extrapolation_settings["degrade_curvature_range"],
            degrade_gradient_range=extrapolation_settings["degrade_gradient_range"],
            skip_bounds_check=True,
            profiler=profiler,
            rbf_lattice_stride_rc=rbf_lattice_stride_rc,
            rbf_lattice_phase_rc=lattice_phase_rc,
            **extrapolation_settings["method_kwargs"],
        )
    if edge_extrapolation is None:
        edge_extrapolation = _empty_edge_extrapolation()

    with profiler.section("iter_aggregate_extrapolation"):
        extrap_lookup = _build_iteration_extrap_lookup(edge_extrapolation)
        edge_extrapolation = _prune_edge_extrapolation_after_lookup(edge_extrapolation)

    with profiler.section("iter_build_bbox_crops"):
        bbox_results = _build_bbox_crops(
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
    _print_bbox_crop_debug_table(bbox_results, verbose=args.verbose)

    agg_samples, disp_bbox = _sample_iteration_agg_samples(
        args,
        profiler,
        grow_direction,
        extrap_only_mode,
        run_model_inference,
        model_state,
        bbox_results,
        extrap_lookup,
    )

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
    apply_section = "iter_apply_samples_direct" if extrap_only_mode else "iter_apply_displacement"
    with profiler.section(apply_section):
        displaced = _apply_displacement(
            agg_samples,
            verbose=args.verbose,
            skip_inference=extrap_only_mode,
        )

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

    if stop_is_skip:
        stop_reason = _iteration_stop_reason(
            merged_iter,
            prev_boundary,
            next_boundary,
            grow_direction,
        )
        stop_requested = stop_reason is not None
        if stop_reason == "no_new_valid":
            _report_iteration_stop(
                args.verbose,
                iteration_pbar,
                f"No newly added valid points for direction '{grow_direction}'; skipping this direction.",
                postfix=f"dir={grow_direction} | bboxes={len(bboxes)} | skipped: no new valid points",
            )
        elif stop_reason == "boundary_unchanged":
            _report_iteration_stop(
                args.verbose,
                iteration_pbar,
                f"Boundary did not advance for direction '{grow_direction}'; skipping this direction.",
                postfix=f"dir={grow_direction} | bboxes={len(bboxes)} | skipped: boundary unchanged",
            )
    else:
        stop_requested = _iteration_should_stop(
            merged_iter,
            prev_boundary,
            next_boundary,
            grow_direction,
            n_bboxes=len(bboxes),
            verbose=args.verbose,
            iteration_pbar=iteration_pbar,
        )

    if args.verbose:
        full_valid_n = int(np.asarray(merged_iter["merged_valid"], dtype=bool).sum())
        displaced_uv = np.asarray(displaced.get("uv", _empty_uv(dtype=np.int64)), dtype=np.int64)
        displaced_pts = np.asarray(displaced.get("world_displaced", _empty_world(dtype=np.float32)), dtype=np.float32)
        displaced_n = int(min(displaced_uv.shape[0], displaced_pts.shape[0]))
        print(
            f"Iteration merge counts: full_valid={full_valid_n} displaced_samples={displaced_n}"
        )
        if displaced_n > 1:
            iter_spacing = _stored_uv_step_spacing_stats(
                displaced_uv[:displaced_n], displaced_pts[:displaced_n]
            )
            if iter_spacing is not None:
                print(
                    "Iteration displaced spacing (per UV step): "
                    f"n={iter_spacing['n_pairs']} "
                    f"median={iter_spacing['median']:.4f} "
                    f"mean={iter_spacing['mean']:.4f} "
                    f"p90={iter_spacing['p90']:.4f}"
                )

    return {
        "merged_iter": merged_iter,
        "bbox_results": bbox_results,
        "edge_extrapolation": edge_extrapolation,
        "extrap_lookup": extrap_lookup,
        "disp_bbox": disp_bbox,
        "displaced": displaced,
        "stop_requested": bool(stop_requested),
        "progressed": bool(not stop_requested),
    }


def _run_with_args(args, parse_done):
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
    else:
        if args.verbose:
            if extrap_only_mode:
                print("Running extrap-only iterative mode (--extrap-only set): skipping inference and stack sampling.")
            elif args.skip_inference:
                print("Skipping displacement inference (--skip-inference set).")
            else:
                print("Skipping displacement inference (no --checkpoint-path provided).")
        if args.checkpoint_path is not None:
            with profiler.section("load_checkpoint_config"):
                model_config, checkpoint_path = load_checkpoint_config(args.checkpoint_path)

    with profiler.section("resolve_settings"):
        extrapolation_settings = resolve_extrapolation_settings(
            args,
            model_config=model_config,
            load_checkpoint_config_fn=load_checkpoint_config,
        )

    with profiler.section("setup_segment"):
        volume = zarr.open_group(args.volume_path, mode="r")
        tgt_segment, stored_zyxs, valid_s, _, h_s, w_s = _setup_segment_with_requested_direction(args, volume)
    grow_directions = _resolve_growth_directions(args.grow_direction, valid_s)
    multi_direction_mode = (args.grow_direction == "all")
    if args.verbose and args.grow_direction == "all":
        missing_dirs = [direction for direction in _ALL_GROW_DIRECTION_ORDER if direction not in grow_directions]
        print(f"Resolved all-direction growth order: {grow_directions}")
        if len(missing_dirs) > 0:
            print(f"Skipping unavailable directions for this segment: {missing_dirs}")

    with profiler.section("initialize_window"):
        init_bboxes = []
        for direction in grow_directions:
            cond_direction, _ = _get_growth_context(direction)
            direction_bboxes, _ = get_cond_edge_bboxes(
                stored_zyxs,
                cond_direction,
                crop_size,
                overlap_frac=args.bbox_overlap_frac,
                cond_valid=valid_s,
            )
            init_bboxes.extend(direction_bboxes)
        if len(init_bboxes) == 0:
            raise RuntimeError("No valid edge bboxes found for segment.")
        r0_s, r1_s, c0_s, c1_s = get_window_bounds_from_bboxes(
            stored_zyxs, valid_s, init_bboxes, pad=args.window_pad,
        )

        full_bounds = _stored_to_full_bounds(tgt_segment, (r0_s, r1_s, c0_s, c1_s))
        current_zyxs, current_valid, current_uv_offset = _initialize_window_state(
            tgt_segment, full_bounds,
        )
    scale_y, scale_x = tgt_segment._scale
    sub_r = _scale_to_subsample_stride(scale_y)
    sub_c = _scale_to_subsample_stride(scale_x)
    lattice_phase_rc, lattice_phase_meta = _infer_lattice_phase_rc(
        stored_zyxs,
        valid_s,
        current_zyxs,
        current_valid,
        current_uv_offset,
        sub_r,
        sub_c,
    )
    if args.verbose:
        print(
            "Inferred stored lattice phase: "
            f"phase_rc={lattice_phase_rc} stride_rc=({sub_r}, {sub_c}) "
            f"mode={lattice_phase_meta.get('mode')} "
            f"pairs={int(lattice_phase_meta.get('pairs', 0))} "
            f"median_err={lattice_phase_meta.get('median_err')}"
        )

    bbox_results = []
    edge_extrapolation = _empty_edge_extrapolation()
    extrap_lookup = _empty_extrap_lookup_arrays()
    disp_bbox = None
    displaced = _empty_displaced_samples()

    n_iterations = int(args.iterations)
    iteration_pbar = None
    if args.verbose:
        iteration_iter = range(n_iterations)
    else:
        iteration_pbar = tqdm(range(n_iterations), total=n_iterations, desc="iterations", unit="iter")
        iteration_iter = iteration_pbar

    for iteration in iteration_iter:
        with profiler.section("iter_iteration"):
            if args.verbose:
                if multi_direction_mode:
                    print(f"[iteration {iteration + 1}/{n_iterations}]")
                else:
                    print(
                        f"[iteration {iteration + 1}/{n_iterations} | dir={grow_directions[0]}]"
                    )

            iteration_progressed = False
            should_break_outer = False
            for grow_direction in grow_directions:
                with profiler.section("iter_growth_direction"):
                    if args.verbose and multi_direction_mode:
                        print(f"[iteration {iteration + 1}/{n_iterations} | dir={grow_direction}]")
                    step_result = _run_growth_direction_step(
                        args=args,
                        profiler=profiler,
                        extrapolation_settings=extrapolation_settings,
                        tgt_segment=tgt_segment,
                        crop_size=crop_size,
                        extrap_only_mode=extrap_only_mode,
                        run_model_inference=run_model_inference,
                        model_state=model_state,
                        iteration_pbar=iteration_pbar,
                        grow_direction=grow_direction,
                        cond_zyxs=current_zyxs,
                        cond_valid=current_valid,
                        cond_uv_offset=current_uv_offset,
                        lattice_phase_rc=lattice_phase_rc,
                        stop_is_skip=multi_direction_mode,
                    )

                    merged_iter = step_result["merged_iter"]
                    current_zyxs = merged_iter["merged_zyxs"]
                    current_valid = merged_iter["merged_valid"]
                    current_uv_offset = merged_iter["uv_offset"]
                    bbox_results = step_result["bbox_results"]
                    edge_extrapolation = step_result["edge_extrapolation"]
                    extrap_lookup = step_result["extrap_lookup"]
                    disp_bbox = step_result["disp_bbox"]
                    displaced = step_result["displaced"]

                    if step_result["progressed"]:
                        iteration_progressed = True
                    if (not multi_direction_mode) and step_result["stop_requested"]:
                        should_break_outer = True
                        break

            if should_break_outer:
                break
            if multi_direction_mode and (not iteration_progressed):
                _report_iteration_stop(
                    args.verbose,
                    iteration_pbar,
                    "No directional progress this iteration; stopping iterative growth.",
                    postfix="stopped: no directional progress",
                )
                break

    if iteration_pbar is not None:
        iteration_pbar.close()

    # Merge the grown surface back onto the stored-resolution base surface
    # (already in memory from setup_segment) to avoid materializing full resolution.
    with profiler.section("final_merge_stored_surface"):
        projection_src_zyxs, projection_src_valid, projection_src_meta = _compose_full_projection_source_original_wins(
            current_zyxs,
            current_valid,
            current_uv_offset,
            stored_zyxs,
            valid_s,
            sub_r,
            sub_c,
            phase_rc=lattice_phase_rc,
        )
        if args.verbose:
            print(
                "Final projection source composition: "
                f"anchors_written={projection_src_meta['anchors_written']} "
                f"anchors_candidates={projection_src_meta['anchors_candidates']}"
            )

        grown_uv, grown_pts, stored_projection = _surface_to_stored_uv_samples_lattice(
            projection_src_zyxs,
            projection_src_valid,
            current_uv_offset,
            sub_r,
            sub_c,
            phase_rc=lattice_phase_rc,
        )
        if args.grow_direction != "all":
            grown_pts, grow_spacing_meta = _normalize_world_step_in_new_band_along_growth(
                grown_uv,
                grown_pts,
                stored_zyxs,
                valid_s,
                args.grow_direction,
                ref_pairs=4,
            )
        else:
            grow_spacing_meta = {
                "applied": False,
                "reason": "grow_direction_all",
                "adjusted_points": 0,
                "adjusted_lines": 0,
            }
        if args.verbose:
            print(
                "Grow-axis spacing normalization: "
                f"applied={grow_spacing_meta.get('applied')} "
                f"reason={grow_spacing_meta.get('reason')} "
                f"adjusted_points={int(grow_spacing_meta.get('adjusted_points', 0))} "
                f"adjusted_lines={int(grow_spacing_meta.get('adjusted_lines', 0))} "
                f"global_ref_step={grow_spacing_meta.get('global_ref_step')}"
            )
        if args.verbose:
            spacing_stats = _stored_uv_step_spacing_stats(grown_uv, grown_pts)
            if spacing_stats is None:
                print("Stored projection spacing stats: no adjacent pairs")
            else:
                print(
                    "Stored projection spacing stats (per stored UV step): "
                    f"n={spacing_stats['n_pairs']} "
                    f"median={spacing_stats['median']:.4f} "
                    f"mean={spacing_stats['mean']:.4f} "
                    f"p90={spacing_stats['p90']:.4f}"
                )

        merge_uv, merge_pts, merge_filter_meta = _filter_stored_merge_samples_to_new_cells(
            grown_uv,
            grown_pts,
            valid_s,
        )
        if args.verbose:
            print(
                "Stored merge filtering: "
                f"in={merge_filter_meta['input_points']} "
                f"kept={merge_filter_meta['kept_points']} "
                f"dropped_existing_valid={merge_filter_meta['dropped_existing_valid']} "
                f"kept_outside_base={merge_filter_meta['kept_outside_base']} "
                f"kept_inside_invalid={merge_filter_meta['kept_inside_invalid']}"
            )

        merged = _merge_displaced_points_into_full_surface(
            stored_zyxs, valid_s, (0, 0),
            {"uv": merge_uv, "world_displaced": merge_pts},
            verbose=args.verbose,
        )
        merged["stored_projection"] = stored_projection
        merged["stored_merge_filter"] = merge_filter_meta
        merged["stored_lattice_phase"] = {
            "phase_rc": [int(lattice_phase_rc[0]), int(lattice_phase_rc[1])],
            "stride_rc": [int(sub_r), int(sub_c)],
            "inference": lattice_phase_meta,
            "projection_source": projection_src_meta,
        }
        merged["grow_axis_spacing_normalization"] = grow_spacing_meta

    with profiler.section("save_tifxyz"):
        output_path = _save_merged_surface_tifxyz(
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
        run_end = time.perf_counter()
        total_runtime_s = (
            run_end - run_start
            if run_start is not None
            else None
        )
        process_runtime_s = run_end - _PROCESS_START
        startup_and_argparse_s = parse_done - _PROCESS_START
        pre_profile_window_s = (
            run_start - _PROCESS_START
            if run_start is not None
            else 0.0
        )
        outside_profile_window_s = (
            max(0.0, process_runtime_s - total_runtime_s)
            if total_runtime_s is not None
            else None
        )
        sum_profiled_sections_s = profiler.total_profiled_time(inclusive=False)
        profiled_gap_s = (
            max(0.0, total_runtime_s - sum_profiled_sections_s)
            if total_runtime_s is not None
            else None
        )
        profiler.print_summary(total_runtime_s=total_runtime_s)
        if total_runtime_s is not None:
            print(f"profile_window_runtime: {float(total_runtime_s):.3f}s")
        print(f"sum_profiled_sections_exclusive: {float(sum_profiled_sections_s):.3f}s")
        if profiled_gap_s is not None:
            print(f"profiled_gap: {float(profiled_gap_s):.3f}s")
        print(f"startup_and_argparse: {float(startup_and_argparse_s):.3f}s")
        print(f"pre_profile_window: {float(pre_profile_window_s):.3f}s")
        if outside_profile_window_s is not None:
            print(f"outside_profile_window: {float(outside_profile_window_s):.3f}s")
        print(f"process_runtime: {float(process_runtime_s):.3f}s")
    return output_path


def run_global_extrap(dense_args):
    argv = _dense_args_to_argv(dense_args)
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr_buffer):
            args = parse_args(argv)
    except SystemExit as exc:
        stderr_text = stderr_buffer.getvalue().strip()
        detail = stderr_text.splitlines()[-1] if stderr_text else "argument parsing failed"
        raise RuntimeError(f"Invalid dense args for infer_global_extrap: {detail}") from exc
    parse_done = time.perf_counter()
    return _run_with_args(args, parse_done=parse_done)


def main():
    args = parse_args()
    parse_done = time.perf_counter()
    _run_with_args(args, parse_done=parse_done)


if __name__ == "__main__":
    main()
