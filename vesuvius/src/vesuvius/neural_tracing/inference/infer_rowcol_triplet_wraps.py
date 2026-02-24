import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
import torch
import zarr
from tqdm import tqdm

from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid_masked
from vesuvius.image_proc.intensity.normalization import normalize_zscore
from vesuvius.neural_tracing.inference.common import (
    _bbox_to_min_corner_and_bounds_array,
    _crop_volume_from_min_corner,
    _resolve_segment_volume,
    _scale_to_subsample_stride,
    resolve_tifxyz_params,
)
from vesuvius.neural_tracing.inference.displacement_tta import (
    TTA_MERGE_METHODS,
    TTA_TRANSFORM_MODES,
)
from vesuvius.neural_tracing.inference.displacement_helpers import load_model, predict_displacement
from vesuvius.neural_tracing.inference.generate_segment_cover_bboxes import (
    _generate_segment_cover_records,
    _serialize_bbox_record,
)
from vesuvius.neural_tracing.inference.infer_global_extrap import (
    _sample_trilinear_displacement_stack,
    _surface_to_stored_uv_samples_lattice,
)
from vesuvius.neural_tracing.tifxyz import save_tifxyz
from vesuvius.tifxyz import read_tifxyz
from vesuvius.tifxyz.upsampling import interpolate_at_points

_TTA_TRANSFORM_ALIASES = {
    "mirror+rot90": "rotate3",
}
_TTA_MERGE_ALIASES = {
    "vector_mean": "mean",
    "vector_median": "median",
}
_COPY_ARG_ALIASES = {
    "dense_checkpoint_path": "checkpoint_path",
    "volume_zarr": "volume_path",
    "tifxyz_out_dir": "out_dir",
    "_DENSE_PROJECTION_NEIGHBORHOOD_RADIUS": "dp_radius",
    "_DENSE_PROJECTION_REJECT_OUTLIER_FRACTION": "dp_reject_frac",
    "_DENSE_PROJECTION_REJECT_MIN_KEEP": "dp_reject_min_keep",
    "dense_projection_neighborhood_radius": "dp_radius",
    "dense_projection_reject_outlier_fraction": "dp_reject_frac",
    "dense_projection_reject_min_keep": "dp_reject_min_keep",
}
_COPY_ARG_TO_CLI = {
    "tifxyz_path": "--tifxyz-path",
    "volume_path": "--volume-path",
    "checkpoint_path": "--checkpoint-path",
    "device": "--device",
    "volume_scale": "--volume-scale",
    "crop_size": "--crop-size",
    "batch_size": "--batch-size",
    "crop_input_workers": "--crop-input-workers",
    "bbox_overlap": "--bbox-overlap",
    "bbox_prune_max_remove_per_band": "--bbox-prune-max-remove-per-band",
    "bbox_band_workers": "--bbox-band-workers",
    "tta_merge_method": "--tta-merge-method",
    "tta_transform": "--tta-transform",
    "tta_outlier_drop_thresh": "--tta-outlier-drop-thresh",
    "tta_outlier_drop_min_keep": "--tta-outlier-drop-min-keep",
    "tta_batch_size": "--tta-batch-size",
    "flip_check_abs_margin": "--flip-check-abs-margin",
    "flip_check_rel_margin": "--flip-check-rel-margin",
    "flip_check_min_band_points": "--flip-check-min-band-points",
    "out_dir": "--out-dir",
    "output_prefix": "--output-prefix",
    "iterations": "--iterations",
    "iter_direction": "--iter-direction",
    "tifxyz_step_size": "--tifxyz-step-size",
    "tifxyz_voxel_size_um": "--tifxyz-voxel-size-um",
    "dp_radius": "--dp-radius",
    "dp_reject_frac": "--dp-reject-frac",
    "dp_reject_min_keep": "--dp-reject-min-keep",
}
_DENSE_PROJECTION_INTERP_METHOD = "catmull_rom"
_DENSE_PROJECTION_NEIGHBORHOOD_RADIUS = 5
_DENSE_PROJECTION_REJECT_OUTLIER_FRACTION = 0.25
_DENSE_PROJECTION_REJECT_MIN_KEEP = 4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run triplet wrap displacement inference and save _front/_back tifxyz outputs."
    )
    cpu_count = int(os.cpu_count() or 1)
    max_band_workers = max(1, cpu_count // 2)
    parser.add_argument("--tifxyz-path", type=str, required=True)
    parser.add_argument("--volume-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--volume-scale", type=int, default=0)

    parser.add_argument(
        "--crop-size",
        type=int,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help="Crop size for bbox inference. Defaults to checkpoint crop_size (or 128^3).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--crop-input-workers",
        type=int,
        default=1,
        help="Number of worker threads used to prepare per-bbox crop inputs.",
    )

    parser.add_argument("--bbox-overlap", type=float, default=0.0)
    parser.add_argument("--bbox-prune", dest="bbox_prune", action="store_true")
    parser.add_argument("--no-bbox-prune", dest="bbox_prune", action="store_false")
    parser.set_defaults(bbox_prune=True)
    parser.add_argument("--bbox-prune-max-remove-per-band", type=int, default=None)
    parser.add_argument("--bbox-band-workers", type=int, default=max_band_workers)

    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument(
        "--tta-merge-method",
        type=str,
        default="vector_geomedian",
        choices=sorted(set(TTA_MERGE_METHODS).union(_TTA_MERGE_ALIASES.keys())),
    )
    parser.add_argument(
        "--tta-transform",
        type=str,
        default="mirror",
        choices=sorted(set(TTA_TRANSFORM_MODES).union(_TTA_TRANSFORM_ALIASES.keys())),
    )
    parser.add_argument("--tta-outlier-drop-thresh", type=float, default=1.25)
    parser.add_argument("--tta-outlier-drop-min-keep", type=int, default=4)
    parser.add_argument("--tta-batch-size", type=int, default=2)
    parser.add_argument("--flip-check-enabled", dest="flip_check_enabled", action="store_true")
    parser.add_argument("--no-flip-check", dest="flip_check_enabled", action="store_false")
    parser.set_defaults(flip_check_enabled=True)
    parser.add_argument("--flip-check-abs-margin", type=float, default=0.25)
    parser.add_argument("--flip-check-rel-margin", type=float, default=0.08)
    parser.add_argument("--flip-check-min-band-points", type=int, default=32)

    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--save-original-copy", dest="save_original_copy", action="store_true")
    parser.add_argument("--no-save-original-copy", dest="save_original_copy", action="store_false")
    parser.set_defaults(save_original_copy=False)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Total number of iterative passes to run. Requires --iter-direction when provided.",
    )
    parser.add_argument(
        "--iter-direction",
        type=str,
        default=None,
        choices=("front", "back"),
        help="Output direction used as the next iteration input when --iterations is provided.",
    )
    parser.add_argument("--keep-previous-wrap", dest="keep_previous_wrap", action="store_true")
    parser.add_argument("--no-keep-previous-wrap", dest="keep_previous_wrap", action="store_false")
    parser.set_defaults(keep_previous_wrap=True)

    parser.add_argument("--tifxyz-step-size", type=int, default=None)
    parser.add_argument("--tifxyz-voxel-size-um", type=float, default=None)
    parser.add_argument(
        "--dp-radius",
        type=int,
        default=_DENSE_PROJECTION_NEIGHBORHOOD_RADIUS,
        help="Dense-projection neighborhood radius in UV cells.",
    )
    parser.add_argument(
        "--dp-reject-frac",
        type=float,
        default=_DENSE_PROJECTION_REJECT_OUTLIER_FRACTION,
        help="Dense-projection max outlier rejection fraction in [0, 1].",
    )
    parser.add_argument(
        "--dp-reject-min-keep",
        type=int,
        default=_DENSE_PROJECTION_REJECT_MIN_KEEP,
        help="Dense-projection minimum kept samples after rejection.",
    )

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.crop_input_workers < 1:
        parser.error("--crop-input-workers must be >= 1")
    if args.bbox_band_workers < 1:
        parser.error("--bbox-band-workers must be >= 1")
    if args.bbox_band_workers > max_band_workers:
        parser.error(
            f"--bbox-band-workers must be <= {max_band_workers} "
            f"(half available CPUs; detected cpu_count={cpu_count})."
        )
    if args.bbox_overlap < 0.0 or args.bbox_overlap >= 1.0:
        parser.error("--bbox-overlap must satisfy 0.0 <= overlap < 1.0")
    if args.crop_size is not None and any(v < 1 for v in args.crop_size):
        parser.error("--crop-size values must be >= 1")
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be >= 1 when provided.")
    if args.iterations is not None and args.iter_direction is None:
        parser.error("--iter-direction is required when --iterations is provided.")
    if args.iterations is None and args.iter_direction is not None:
        parser.error("--iter-direction requires --iterations.")
    if args.flip_check_abs_margin < 0.0:
        parser.error("--flip-check-abs-margin must be >= 0")
    if args.flip_check_rel_margin < 0.0:
        parser.error("--flip-check-rel-margin must be >= 0")
    if args.flip_check_min_band_points < 1:
        parser.error("--flip-check-min-band-points must be >= 1")
    if args.dp_radius < 0:
        parser.error("--dp-radius must be >= 0")
    if args.dp_reject_frac < 0.0 or args.dp_reject_frac > 1.0:
        parser.error("--dp-reject-frac must satisfy 0.0 <= value <= 1.0")
    if args.dp_reject_min_keep < 1:
        parser.error("--dp-reject-min-keep must be >= 1")
    return args


def normalize_copy_args(copy_args):
    if not isinstance(copy_args, dict):
        raise RuntimeError(f"copy_args must be a dict, got {type(copy_args).__name__}")
    normalized = {}
    for key, value in copy_args.items():
        key_norm = str(key).replace("-", "_")
        normalized[_COPY_ARG_ALIASES.get(key_norm, key_norm)] = value
    return normalized


def _append_cli_arg(argv, flag, value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        argv.append(flag)
        argv.extend(str(v) for v in value)
        return
    argv.extend([flag, str(value)])


def _copy_args_to_argv(copy_args):
    copy_args = normalize_copy_args(copy_args)
    argv = []

    for key, flag in _COPY_ARG_TO_CLI.items():
        if key not in copy_args:
            continue
        value = copy_args.get(key)
        if value is None:
            continue
        _append_cli_arg(argv, flag, value)

    if "tta" in copy_args and bool(copy_args.get("tta")) is False:
        argv.append("--no-tta")
    if "bbox_prune" in copy_args and bool(copy_args.get("bbox_prune")) is False:
        argv.append("--no-bbox-prune")
    if "flip_check_enabled" in copy_args and bool(copy_args.get("flip_check_enabled")) is False:
        argv.append("--no-flip-check")
    if "save_original_copy" in copy_args and bool(copy_args.get("save_original_copy")):
        argv.append("--save-original-copy")
    if "keep_previous_wrap" in copy_args and bool(copy_args.get("keep_previous_wrap")) is False:
        argv.append("--no-keep-previous-wrap")
    if "verbose" in copy_args and bool(copy_args.get("verbose")):
        argv.append("--verbose")

    return argv


def _log(verbose, msg):
    if verbose:
        print(msg)


def _canonicalize_tta_settings(args):
    raw_transform = str(args.tta_transform).strip().lower()
    canonical_transform = _TTA_TRANSFORM_ALIASES.get(raw_transform, raw_transform)
    if canonical_transform != raw_transform:
        _log(
            args.verbose,
            f"mapped --tta-transform {raw_transform!r} -> {canonical_transform!r} for backend compatibility.",
        )
    args.tta_transform = canonical_transform

    raw_merge = str(args.tta_merge_method).strip().lower()
    canonical_merge = _TTA_MERGE_ALIASES.get(raw_merge, raw_merge)
    if canonical_merge != raw_merge:
        _log(
            args.verbose,
            f"mapped --tta-merge-method {raw_merge!r} -> {canonical_merge!r} for backend compatibility.",
        )
    args.tta_merge_method = canonical_merge
    return args


def _load_input_grid(tifxyz_path, retarget_factor=1):
    surface = read_tifxyz(tifxyz_path)
    if float(retarget_factor) != 1.0:
        surface = surface.retarget(float(retarget_factor))
    surface.use_stored_resolution()
    x_s, y_s, z_s, valid_s = surface[:]
    grid = np.stack([z_s, y_s, x_s], axis=-1).astype(np.float32, copy=False)
    valid = np.asarray(valid_s, dtype=bool) & np.isfinite(grid).all(axis=2)
    grid = grid.copy()
    grid[~valid] = -1.0
    return surface, grid, valid


def _generate_cover_bboxes_from_points(
    points_zyx,
    tifxyz_uuid,
    crop_size,
    overlap=0.0,
    prune_bboxes=False,
    prune_max_remove_per_band=None,
    band_workers=1,
):
    result = _generate_segment_cover_records(
        np.asarray(points_zyx, dtype=np.float64),
        crop_size,
        overlap=overlap,
        prune_bboxes=prune_bboxes,
        prune_max_remove_per_band=prune_max_remove_per_band,
        band_workers=band_workers,
    )
    return [
        {
            "tifxyz_uuid": str(tifxyz_uuid),
            **_serialize_bbox_record(item),
        }
        for item in result["final_records"]
    ]


def _resolve_crop_size(args, model_config):
    if args.crop_size is not None:
        return tuple(int(v) for v in args.crop_size)

    cfg_crop = model_config.get("crop_size", 128)
    if isinstance(cfg_crop, int):
        c = int(cfg_crop)
        return (c, c, c)
    if isinstance(cfg_crop, (list, tuple)) and len(cfg_crop) == 3:
        return tuple(int(v) for v in cfg_crop)
    raise RuntimeError(f"Unable to resolve crop_size from checkpoint config: {cfg_crop!r}")


def _split_triplet_displacement_channels(disp_batch):
    if disp_batch.ndim != 5:
        raise RuntimeError(f"Expected displacement batch [B, C, D, H, W], got {tuple(disp_batch.shape)}")
    if disp_batch.shape[1] < 6:
        raise RuntimeError(
            "Triplet inference requires at least 6 displacement channels "
            f"(behind+front), got C={int(disp_batch.shape[1])}."
        )
    back = disp_batch[:, 0:3]
    front = disp_batch[:, 3:6]
    return back, front


def _score_xy_distance_in_band(
    points_zyx,
    band_z_min,
    band_z_max,
    center_y,
    center_x,
    min_band_points,
):
    pts = np.asarray(points_zyx, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None
    finite = np.isfinite(pts).all(axis=1)
    if not bool(finite.any()):
        return None
    pts = pts[finite]
    z_vals = pts[:, 0]
    in_band = (z_vals >= float(band_z_min)) & (z_vals <= float(band_z_max))
    if int(np.count_nonzero(in_band)) < int(min_band_points):
        return None
    pts_band = pts[in_band]
    y_vals = pts_band[:, 1].astype(np.float64, copy=False)
    x_vals = pts_band[:, 2].astype(np.float64, copy=False)
    dy = y_vals - float(center_y)
    dx = x_vals - float(center_x)
    radial = np.sqrt((dy * dy) + (dx * dx))
    if radial.size == 0 or not bool(np.isfinite(radial).any()):
        return None
    score = float(np.median(radial))
    xy_center = np.asarray([float(np.median(y_vals)), float(np.median(x_vals))], dtype=np.float64)
    if not np.isfinite(score) or not bool(np.isfinite(xy_center).all()):
        return None
    return {
        "score": score,
        "xy_center": xy_center,
        "points_in_band": int(pts_band.shape[0]),
    }


def _z_extent(points_zyx):
    pts = np.asarray(points_zyx, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None
    finite = np.isfinite(pts).all(axis=1)
    if not bool(finite.any()):
        return None
    z_vals = pts[finite, 0].astype(np.float64, copy=False)
    return float(np.min(z_vals)), float(np.max(z_vals))


def _maybe_swap_bbox_front_back(
    world_center_zyx,
    world_back_zyx,
    world_front_zyx,
    center_y,
    center_x,
    abs_margin,
    rel_margin,
    min_band_points,
):
    abs_margin = max(0.0, float(abs_margin))
    rel_margin = max(0.0, float(rel_margin))
    min_band_points = max(1, int(min_band_points))

    center_extent = _z_extent(world_center_zyx)
    back_extent = _z_extent(world_back_zyx)
    front_extent = _z_extent(world_front_zyx)
    if center_extent is None or back_extent is None or front_extent is None:
        return False, "insufficient_band_points"

    band_z_min = min(center_extent[0], back_extent[0], front_extent[0])
    band_z_max = max(center_extent[1], back_extent[1], front_extent[1])
    back_score = _score_xy_distance_in_band(
        world_back_zyx,
        band_z_min=band_z_min,
        band_z_max=band_z_max,
        center_y=float(center_y),
        center_x=float(center_x),
        min_band_points=min_band_points,
    )
    front_score = _score_xy_distance_in_band(
        world_front_zyx,
        band_z_min=band_z_min,
        band_z_max=band_z_max,
        center_y=float(center_y),
        center_x=float(center_x),
        min_band_points=min_band_points,
    )
    if back_score is None or front_score is None:
        return False, "insufficient_band_points"

    sep_xy = float(np.linalg.norm(front_score["xy_center"] - back_score["xy_center"]))
    if not np.isfinite(sep_xy):
        sep_xy = 0.0
    margin = max(abs_margin, rel_margin * sep_xy)
    gap = abs(front_score["score"] - back_score["score"])
    if gap <= margin:
        return False, "ambiguous"

    if front_score["score"] > back_score["score"]:
        return True, "swap_confident"
    return False, "already_oriented"


def _compute_surface_tangent_axis(surface_grid, surface_valid, axis):
    grid = np.asarray(surface_grid, dtype=np.float32)
    valid = np.asarray(surface_valid, dtype=bool)
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise RuntimeError(f"Expected surface_grid shape [H,W,3], got {tuple(grid.shape)}")
    if valid.shape != grid.shape[:2]:
        raise RuntimeError(f"surface_valid shape {tuple(valid.shape)} does not match grid shape {tuple(grid.shape[:2])}")
    if axis not in (0, 1):
        raise RuntimeError(f"axis must be 0 or 1, got {axis}")

    tangent = np.zeros_like(grid, dtype=np.float32)
    tangent_valid = np.zeros(valid.shape, dtype=bool)
    h, w = valid.shape

    if axis == 0:
        if h >= 3:
            central_ok = valid[1:-1, :] & valid[:-2, :] & valid[2:, :]
            central_delta = 0.5 * (grid[2:, :, :] - grid[:-2, :, :])
            tangent[1:-1, :, :][central_ok] = central_delta[central_ok]
            tangent_valid[1:-1, :][central_ok] = True

        if h >= 2:
            diff = grid[1:, :, :] - grid[:-1, :, :]
            diff_ok = valid[1:, :] & valid[:-1, :]

            use_forward = (~tangent_valid[:-1, :]) & diff_ok
            tangent[:-1, :, :][use_forward] = diff[use_forward]
            tangent_valid[:-1, :][use_forward] = True

            use_backward = (~tangent_valid[1:, :]) & diff_ok
            tangent[1:, :, :][use_backward] = diff[use_backward]
            tangent_valid[1:, :][use_backward] = True
    else:
        if w >= 3:
            central_ok = valid[:, 1:-1] & valid[:, :-2] & valid[:, 2:]
            central_delta = 0.5 * (grid[:, 2:, :] - grid[:, :-2, :])
            tangent[:, 1:-1, :][central_ok] = central_delta[central_ok]
            tangent_valid[:, 1:-1][central_ok] = True

        if w >= 2:
            diff = grid[:, 1:, :] - grid[:, :-1, :]
            diff_ok = valid[:, 1:] & valid[:, :-1]

            use_forward = (~tangent_valid[:, :-1]) & diff_ok
            tangent[:, :-1, :][use_forward] = diff[use_forward]
            tangent_valid[:, :-1][use_forward] = True

            use_backward = (~tangent_valid[:, 1:]) & diff_ok
            tangent[:, 1:, :][use_backward] = diff[use_backward]
            tangent_valid[:, 1:][use_backward] = True

    return tangent, tangent_valid


def _compute_surface_normals_from_input_grid(input_grid, input_valid):
    grid = np.asarray(input_grid, dtype=np.float32)
    valid = np.asarray(input_valid, dtype=bool)
    row_tangent, row_tangent_valid = _compute_surface_tangent_axis(grid, valid, axis=0)
    col_tangent, col_tangent_valid = _compute_surface_tangent_axis(grid, valid, axis=1)
    normals = np.cross(col_tangent, row_tangent)
    norms = np.linalg.norm(normals, axis=2)
    finite = np.isfinite(normals).all(axis=2) & np.isfinite(norms)
    normals_valid = valid & row_tangent_valid & col_tangent_valid & finite & (norms > 1e-6)
    out = np.zeros_like(normals, dtype=np.float32)
    if bool(normals_valid.any()):
        out[normals_valid] = normals[normals_valid] / norms[normals_valid, None]
    return out, normals_valid


def _concat_selected_points(uv, world_back_all, world_front_all, from_back_mask, from_front_mask):
    uv_parts = []
    world_parts = []
    if bool(from_back_mask.any()):
        uv_parts.append(uv[from_back_mask])
        world_parts.append(world_back_all[from_back_mask])
    if bool(from_front_mask.any()):
        uv_parts.append(uv[from_front_mask])
        world_parts.append(world_front_all[from_front_mask])
    if len(uv_parts) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3), dtype=np.float32)
    uv_out = np.concatenate(uv_parts, axis=0).astype(np.int32, copy=False)
    world_out = np.concatenate(world_parts, axis=0).astype(np.float32, copy=False)
    return uv_out, world_out


def _accumulate_displaced(sum_grid, count_grid, uv_rc, world_zyx):
    if uv_rc.size == 0 or world_zyx.size == 0:
        return
    rr = uv_rc[:, 0].astype(np.int32, copy=False)
    cc = uv_rc[:, 1].astype(np.int32, copy=False)
    np.add.at(sum_grid[..., 0], (rr, cc), world_zyx[:, 0].astype(np.float32, copy=False))
    np.add.at(sum_grid[..., 1], (rr, cc), world_zyx[:, 1].astype(np.float32, copy=False))
    np.add.at(sum_grid[..., 2], (rr, cc), world_zyx[:, 2].astype(np.float32, copy=False))
    np.add.at(count_grid, (rr, cc), 1)


def _build_local_sparse_uv_grid(uv_rc, world_zyx):
    uv_arr = np.asarray(uv_rc, dtype=np.int64)
    world_arr = np.asarray(world_zyx, dtype=np.float32)
    if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
        raise RuntimeError(f"Expected uv_rc shape [N,2], got {tuple(uv_arr.shape)}")
    if world_arr.ndim != 2 or world_arr.shape[1] != 3:
        raise RuntimeError(f"Expected world_zyx shape [N,3], got {tuple(world_arr.shape)}")
    if uv_arr.shape[0] != world_arr.shape[0]:
        raise RuntimeError(
            f"uv/world size mismatch: uv N={int(uv_arr.shape[0])}, world N={int(world_arr.shape[0])}"
        )
    if uv_arr.shape[0] == 0:
        return np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0, 0), dtype=bool), (0, 0)

    r_min = int(uv_arr[:, 0].min())
    c_min = int(uv_arr[:, 1].min())
    rr = (uv_arr[:, 0] - r_min).astype(np.int64, copy=False)
    cc = (uv_arr[:, 1] - c_min).astype(np.int64, copy=False)
    h = int(rr.max()) + 1
    w = int(cc.max()) + 1

    sum_grid = np.zeros((h, w, 3), dtype=np.float64)
    count_grid = np.zeros((h, w), dtype=np.uint32)
    _accumulate_displaced(sum_grid, count_grid, np.stack([rr, cc], axis=-1), world_arr)

    sparse_grid = np.full((h, w, 3), -1.0, dtype=np.float32)
    sparse_valid = count_grid > 0
    if bool(sparse_valid.any()):
        sr, sc = np.where(sparse_valid)
        denom = count_grid[sr, sc].astype(np.float32, copy=False)[:, None]
        vals = (sum_grid[sr, sc] / denom).astype(np.float32, copy=False)
        finite = np.isfinite(vals).all(axis=1)
        if bool(finite.any()):
            sr = sr[finite]
            sc = sc[finite]
            vals = vals[finite]
            sparse_grid[sr, sc] = vals
            sparse_valid = np.zeros_like(sparse_valid, dtype=bool)
            sparse_valid[sr, sc] = True
        else:
            sparse_valid = np.zeros_like(sparse_valid, dtype=bool)
    return sparse_grid, sparse_valid, (r_min, c_min)


def _densify_local_sparse_uv_grid(
    sparse_grid,
    sparse_valid,
    subsample_stride,
    interpolation_method=_DENSE_PROJECTION_INTERP_METHOD,
):
    grid = np.asarray(sparse_grid, dtype=np.float32)
    valid = np.asarray(sparse_valid, dtype=bool)
    if grid.ndim != 3 or grid.shape[-1] != 3:
        raise RuntimeError(f"Expected sparse_grid shape [H,W,3], got {tuple(grid.shape)}")
    if valid.shape != grid.shape[:2]:
        raise RuntimeError(f"sparse_valid shape {tuple(valid.shape)} does not match grid shape {tuple(grid.shape[:2])}")

    h, w = grid.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0, 0), dtype=bool)

    stride = max(1, int(subsample_stride))
    if stride == 1:
        out = np.asarray(grid, dtype=np.float32).copy()
        out_valid = np.asarray(valid, dtype=bool).copy()
        out[~out_valid] = -1.0
        return out, out_valid

    dense_h = int((h - 1) * stride + 1)
    dense_w = int((w - 1) * stride + 1)
    if dense_h <= 0 or dense_w <= 0:
        return np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0, 0), dtype=bool)

    if not bool(valid.any()):
        return (
            np.full((dense_h, dense_w, 3), -1.0, dtype=np.float32),
            np.zeros((dense_h, dense_w), dtype=bool),
        )

    query_r = np.arange(dense_h, dtype=np.float32)
    query_c = np.arange(dense_w, dtype=np.float32)
    query_rr, query_cc = np.meshgrid(query_r, query_c, indexing="ij")
    x_src = grid[..., 2]
    y_src = grid[..., 1]
    z_src = grid[..., 0]
    x_int, y_int, z_int, int_valid = interpolate_at_points(
        x_src,
        y_src,
        z_src,
        valid,
        query_rr,
        query_cc,
        scale=(float(stride), float(stride)),
        method=str(interpolation_method),
        invalid_value=-1.0,
    )
    dense_grid = np.stack([z_int, y_int, x_int], axis=-1).astype(np.float32, copy=False)
    dense_valid = np.asarray(int_valid, dtype=bool) & np.isfinite(dense_grid).all(axis=2)
    dense_grid = dense_grid.copy()
    dense_grid[~dense_valid] = -1.0
    return dense_grid, dense_valid


def _sample_dense_neighborhood_for_uv_points(
    dense_grid,
    dense_valid,
    uv_rc,
    uv_offset_rc,
    fallback_world_zyx,
    subsample_stride,
    neighborhood_radius=_DENSE_PROJECTION_NEIGHBORHOOD_RADIUS,
    reject_outlier_fraction=_DENSE_PROJECTION_REJECT_OUTLIER_FRACTION,
    reject_min_keep=_DENSE_PROJECTION_REJECT_MIN_KEEP,
):
    dense_arr = np.asarray(dense_grid, dtype=np.float32)
    dense_mask = np.asarray(dense_valid, dtype=bool)
    uv_arr = np.asarray(uv_rc, dtype=np.int64)
    fallback = np.asarray(fallback_world_zyx, dtype=np.float32)

    if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
        raise RuntimeError(f"Expected uv_rc shape [N,2], got {tuple(uv_arr.shape)}")
    if fallback.ndim != 2 or fallback.shape[1] != 3:
        raise RuntimeError(f"Expected fallback_world_zyx shape [N,3], got {tuple(fallback.shape)}")
    if uv_arr.shape[0] != fallback.shape[0]:
        raise RuntimeError(
            f"uv/fallback size mismatch: uv N={int(uv_arr.shape[0])}, fallback N={int(fallback.shape[0])}"
        )
    if uv_arr.shape[0] == 0:
        return fallback.astype(np.float32, copy=True)
    if dense_arr.ndim != 3 or dense_arr.shape[-1] != 3:
        return fallback.astype(np.float32, copy=True)
    if dense_mask.shape != dense_arr.shape[:2]:
        return fallback.astype(np.float32, copy=True)
    if not bool(dense_mask.any()):
        return fallback.astype(np.float32, copy=True)

    h, w = dense_mask.shape
    stride = max(1, int(subsample_stride))
    radius = max(0, int(neighborhood_radius))
    reject_fraction = float(reject_outlier_fraction)
    if not np.isfinite(reject_fraction):
        reject_fraction = 0.0
    reject_fraction = min(max(0.0, reject_fraction), 1.0)
    min_keep = max(1, int(reject_min_keep))
    out = fallback.astype(np.float32, copy=True)

    local_r = uv_arr[:, 0] - int(uv_offset_rc[0])
    local_c = uv_arr[:, 1] - int(uv_offset_rc[1])
    anchor_r = local_r * int(stride)
    anchor_c = local_c * int(stride)
    in_bounds = (
        (anchor_r >= 0)
        & (anchor_r < int(h))
        & (anchor_c >= 0)
        & (anchor_c < int(w))
    )
    if not bool(in_bounds.any()):
        return out

    idx = np.nonzero(in_bounds)[0]
    ar = anchor_r[idx].astype(np.int64, copy=False)
    ac = anchor_c[idx].astype(np.int64, copy=False)
    for j, out_i in enumerate(idx):
        r = int(ar[j])
        c = int(ac[j])
        r0 = max(0, r - radius)
        r1 = min(h - 1, r + radius)
        c0 = max(0, c - radius)
        c1 = min(w - 1, c + radius)
        win_valid = dense_mask[r0:r1 + 1, c0:c1 + 1]
        if not bool(win_valid.any()):
            continue
        pts = dense_arr[r0:r1 + 1, c0:c1 + 1][win_valid]
        if pts.shape[0] == 0:
            continue
        pts = pts.astype(np.float32, copy=False)
        n_pts = int(pts.shape[0])
        max_reject = min(
            int(np.floor(reject_fraction * float(n_pts))),
            max(0, n_pts - min_keep),
        )
        if max_reject > 0 and n_pts >= 4:
            center = np.median(pts, axis=0).astype(np.float32, copy=False)
            d = np.linalg.norm(pts - center[None, :], axis=1).astype(np.float32, copy=False)
            if d.shape[0] > 0:
                med_d = float(np.median(d))
                mad = float(np.median(np.abs(d - med_d)))
                if mad <= 1e-6:
                    cand = np.where(d > (med_d + 1e-6))[0]
                else:
                    robust_z = np.abs(d - med_d) / float(1.4826 * mad)
                    cand = np.where(robust_z > 3.5)[0]
                if cand.size > 0:
                    take = min(int(max_reject), int(cand.size), int(pts.shape[0] - 1))
                    if take > 0:
                        order = np.argsort(d[cand])[::-1]
                        drop = cand[order[:take]]
                        keep = np.ones(pts.shape[0], dtype=bool)
                        keep[drop] = False
                        if bool(keep.any()):
                            pts = pts[keep]
        out[out_i] = np.mean(pts, axis=0).astype(np.float32, copy=False)
    return out


def _robustify_samples_with_dense_projection(
    uv_rc,
    world_zyx,
    subsample_stride,
    neighborhood_radius=_DENSE_PROJECTION_NEIGHBORHOOD_RADIUS,
    interpolation_method=_DENSE_PROJECTION_INTERP_METHOD,
    reject_outlier_fraction=_DENSE_PROJECTION_REJECT_OUTLIER_FRACTION,
    reject_min_keep=_DENSE_PROJECTION_REJECT_MIN_KEEP,
):
    uv_arr = np.asarray(uv_rc, dtype=np.int64)
    world_arr = np.asarray(world_zyx, dtype=np.float32)
    if uv_arr.shape[0] == 0:
        return world_arr.astype(np.float32, copy=True)

    sparse_grid, sparse_valid, uv_offset = _build_local_sparse_uv_grid(uv_arr, world_arr)
    dense_grid, dense_valid = _densify_local_sparse_uv_grid(
        sparse_grid,
        sparse_valid,
        subsample_stride=subsample_stride,
        interpolation_method=interpolation_method,
    )
    return _sample_dense_neighborhood_for_uv_points(
        dense_grid,
        dense_valid,
        uv_arr,
        uv_offset_rc=uv_offset,
        fallback_world_zyx=world_arr,
        subsample_stride=subsample_stride,
        neighborhood_radius=neighborhood_radius,
        reject_outlier_fraction=reject_outlier_fraction,
        reject_min_keep=reject_min_keep,
    )


def _finalize_sparse_prediction(sum_grid, count_grid, shape_hw):
    h, w = int(shape_hw[0]), int(shape_hw[1])
    pred_grid = np.full((h, w, 3), -1.0, dtype=np.float32)
    pred_valid = np.zeros((h, w), dtype=bool)
    rr, cc = np.where(count_grid > 0)
    if rr.size == 0:
        return pred_grid, pred_valid

    denom = count_grid[rr, cc].astype(np.float32, copy=False)[:, None]
    vals = (sum_grid[rr, cc] / denom).astype(np.float32, copy=False)
    finite = np.isfinite(vals).all(axis=1)
    if finite.any():
        rr = rr[finite]
        cc = cc[finite]
        vals = vals[finite]
        pred_grid[rr, cc] = vals
        pred_valid[rr, cc] = True
    return pred_grid, pred_valid


def _project_sparse_to_input_lattice(pred_grid, pred_valid):
    # Project through a mask-aware bilinear lattice sampler so output writing always
    # happens on an explicit lattice projection step, even though this surface is
    # already indexed in stored UV space.
    uv_keep, pts_keep, projection_meta = _surface_to_stored_uv_samples_lattice(
        pred_grid,
        pred_valid,
        uv_offset=(0, 0),
        sub_r=1,
        sub_c=1,
        phase_rc=(0, 0),
    )
    out_grid = np.full_like(pred_grid, -1.0, dtype=np.float32)
    out_valid = np.zeros(pred_valid.shape, dtype=bool)
    if uv_keep.shape[0] == 0:
        return out_grid, out_valid, projection_meta

    finite = np.isfinite(pts_keep).all(axis=1)
    uv_keep = uv_keep[finite]
    pts_keep = pts_keep[finite]
    if uv_keep.shape[0] > 0:
        rr = uv_keep[:, 0].astype(np.int32, copy=False)
        cc = uv_keep[:, 1].astype(np.int32, copy=False)
        out_grid[rr, cc] = pts_keep.astype(np.float32, copy=False)
        out_valid[rr, cc] = True
    return out_grid, out_valid, projection_meta


def _merge_with_original(original_grid, original_valid, pred_grid, pred_valid):
    original_arr = np.asarray(original_grid, dtype=np.float32)
    original_mask = np.asarray(original_valid, dtype=bool)
    pred_arr = np.asarray(pred_grid, dtype=np.float32)
    pred_mask = np.asarray(pred_valid, dtype=bool)

    merged = np.full_like(original_arr, -1.0, dtype=np.float32)
    merged_valid = np.zeros_like(original_mask, dtype=bool)

    support = pred_mask & np.isfinite(pred_arr).all(axis=2)
    if bool(support.any()):
        merged[support] = pred_arr[support]
        merged_valid[support] = True

    needs_fill = original_mask & ~support
    if bool(needs_fill.any()) and bool(support.any()):
        # Map each unresolved original-valid cell to its nearest predicted support cell in UV.
        _, nearest_idx = ndimage.distance_transform_edt(~support, return_indices=True)
        nearest_r = nearest_idx[0][needs_fill]
        nearest_c = nearest_idx[1][needs_fill]
        merged[needs_fill] = pred_arr[nearest_r, nearest_c]
        merged_valid[needs_fill] = True

    merged[~merged_valid] = -1.0
    return merged, merged_valid


def _iter_bbox_batches(records, batch_size):
    for start in range(0, len(records), batch_size):
        yield start, records[start:start + batch_size]


def _voxelize_local_surface_from_uv_points(local_points, uv_points, crop_size):
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    vox = np.zeros(tuple(crop_size_arr.tolist()), dtype=np.float32)
    if local_points is None or uv_points is None:
        return vox

    local_arr = np.asarray(local_points, dtype=np.float64)
    uv_arr = np.asarray(uv_points, dtype=np.int64)
    if local_arr.ndim != 2 or local_arr.shape[1] != 3 or uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
        return vox
    if local_arr.shape[0] == 0 or uv_arr.shape[0] == 0 or local_arr.shape[0] != uv_arr.shape[0]:
        return vox

    finite = np.isfinite(local_arr).all(axis=1)
    if not bool(finite.any()):
        return vox
    local_arr = local_arr[finite]
    uv_arr = uv_arr[finite]

    r_min = int(uv_arr[:, 0].min())
    c_min = int(uv_arr[:, 1].min())
    r_max = int(uv_arr[:, 0].max())
    c_max = int(uv_arr[:, 1].max())
    h = int(r_max - r_min + 1)
    w = int(c_max - c_min + 1)
    if h <= 0 or w <= 0:
        return vox

    grid_local = np.zeros((h, w, 3), dtype=np.float64)
    grid_valid = np.zeros((h, w), dtype=bool)
    rr = (uv_arr[:, 0] - r_min).astype(np.int64, copy=False)
    cc = (uv_arr[:, 1] - c_min).astype(np.int64, copy=False)
    grid_local[rr, cc] = local_arr
    grid_valid[rr, cc] = True
    return voxelize_surface_grid_masked(grid_local, tuple(int(v) for v in crop_size_arr.tolist()), grid_valid).astype(
        np.float32,
        copy=False,
    )


def _prepare_bbox_item(record, crop_size, world_points, uv_points, volume_arr):
    bbox = tuple(record["bbox"])
    min_corner, _ = _bbox_to_min_corner_and_bounds_array(bbox)
    crop_arr = np.asarray(crop_size, dtype=np.int32)
    max_corner = min_corner + crop_arr

    in_bounds = (
        (world_points[:, 0] >= float(min_corner[0]))
        & (world_points[:, 0] < float(max_corner[0]))
        & (world_points[:, 1] >= float(min_corner[1]))
        & (world_points[:, 1] < float(max_corner[1]))
        & (world_points[:, 2] >= float(min_corner[2]))
        & (world_points[:, 2] < float(max_corner[2]))
    )
    if not bool(in_bounds.any()):
        return None

    uv_sel = uv_points[in_bounds].astype(np.int32, copy=False)
    world_sel = world_points[in_bounds].astype(np.float32, copy=False)
    local_sel = (world_sel - min_corner[None, :].astype(np.float32, copy=False)).astype(np.float32, copy=False)

    cond_vox = _voxelize_local_surface_from_uv_points(local_sel, uv_sel, crop_size).astype(np.float32, copy=False)
    vol_crop = _crop_volume_from_min_corner(volume_arr, min_corner, crop_size)
    vol_crop = normalize_zscore(vol_crop).astype(np.float32, copy=False)

    return {
        "bbox_id": int(record.get("bbox_id", -1)),
        "min_corner": min_corner.astype(np.int32, copy=False),
        "uv": uv_sel,
        "world": world_sel,
        "local": local_sel,
        "cond_vox": cond_vox,
        "volume": vol_crop,
    }


def _gather_batch_items(
    batch_records,
    crop_size,
    world_points,
    uv_points,
    volume_arr,
    num_workers,
):
    if len(batch_records) == 0:
        return []
    if int(num_workers) <= 1 or len(batch_records) == 1:
        items = []
        for rec in batch_records:
            item = _prepare_bbox_item(rec, crop_size, world_points, uv_points, volume_arr)
            if item is not None:
                items.append(item)
        return items

    max_workers = min(int(num_workers), len(batch_records))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        prepared = executor.map(
            _prepare_bbox_item,
            batch_records,
            [crop_size] * len(batch_records),
            [world_points] * len(batch_records),
            [uv_points] * len(batch_records),
            [volume_arr] * len(batch_records),
        )
        return [item for item in prepared if item is not None]


def _run_triplet_inference(
    args,
    model_state,
    records,
    crop_size,
    world_points,
    uv_points,
    volume_arr,
    shape_hw,
    dense_subsample_stride,
    input_normals,
    input_normals_valid,
):
    h, w = int(shape_hw[0]), int(shape_hw[1])
    sum_back = np.zeros((h, w, 3), dtype=np.float32)
    sum_front = np.zeros((h, w, 3), dtype=np.float32)
    count_back = np.zeros((h, w), dtype=np.uint32)
    count_front = np.zeros((h, w), dtype=np.uint32)
    normals_arr = np.asarray(input_normals, dtype=np.float32)
    normals_valid_arr = np.asarray(input_normals_valid, dtype=bool)
    if normals_arr.shape != (h, w, 3):
        raise RuntimeError(
            f"input_normals shape {tuple(normals_arr.shape)} does not match expected {(h, w, 3)}"
        )
    if normals_valid_arr.shape != (h, w):
        raise RuntimeError(
            f"input_normals_valid shape {tuple(normals_valid_arr.shape)} does not match expected {(h, w)}"
        )
    flip_checks_total = 0
    flip_swaps_applied = 0
    flip_skipped_ambiguous = 0
    flip_skipped_insufficient_points = 0
    orientation_samples_used = 0
    orientation_sample_fallback_used = 0

    volume_shape = np.asarray(np.shape(volume_arr), dtype=np.float64)
    if volume_shape.size < 3:
        raise RuntimeError(f"Expected 3D volume for triplet inference, got shape={tuple(np.shape(volume_arr))}")

    expected_in_channels = int(model_state["expected_in_channels"])
    if expected_in_channels != 2:
        raise RuntimeError(
            "This script currently supports triplet checkpoints with in_channels=2 (volume+cond); "
            f"checkpoint expects in_channels={expected_in_channels}."
        )

    n_total = len(records)
    n_batches = (n_total + int(args.batch_size) - 1) // int(args.batch_size)
    kept_bboxes = 0
    dp_radius = int(getattr(args, "dp_radius", _DENSE_PROJECTION_NEIGHBORHOOD_RADIUS))
    dp_reject_frac = float(getattr(args, "dp_reject_frac", _DENSE_PROJECTION_REJECT_OUTLIER_FRACTION))
    dp_reject_min_keep = int(getattr(args, "dp_reject_min_keep", _DENSE_PROJECTION_REJECT_MIN_KEEP))

    batch_iter = _iter_bbox_batches(records, int(args.batch_size))
    batch_iter = tqdm(batch_iter, total=n_batches, desc="triplet_batches", unit="batch")
    for batch_idx, (_, batch_records) in enumerate(batch_iter, start=1):
        items = _gather_batch_items(
            batch_records=batch_records,
            crop_size=crop_size,
            world_points=world_points,
            uv_points=uv_points,
            volume_arr=volume_arr,
            num_workers=int(args.crop_input_workers),
        )

        if len(items) == 0:
            _log(args.verbose, f"batch {batch_idx}/{n_batches}: skipped (no points in batch bboxes)")
            continue

        kept_bboxes += len(items)
        d, h_c, w_c = crop_size
        batch_np = np.empty((len(items), 2, d, h_c, w_c), dtype=np.float32)
        for i, item in enumerate(items):
            batch_np[i, 0] = item["volume"]
            batch_np[i, 1] = item["cond_vox"]

        model_inputs = torch.from_numpy(batch_np).to(args.device, non_blocking=True)
        disp_pred = predict_displacement(args, model_state, model_inputs, use_tta=bool(args.tta), profiler=None)
        if disp_pred is None:
            raise RuntimeError("Model output did not contain 'displacement'.")
        disp_pred_np = (
            disp_pred.detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False)
        )
        back_batch, front_batch = _split_triplet_displacement_channels(disp_pred_np)

        for i, item in enumerate(items):
            local = item["local"]
            uv = item["uv"]
            world = item["world"]

            back_disp, back_valid = _sample_trilinear_displacement_stack(back_batch[i], local)
            front_disp, front_valid = _sample_trilinear_displacement_stack(front_batch[i], local)
            world_back_all = world + back_disp
            world_front_all = world + front_disp
            finite_back = np.isfinite(world_back_all).all(axis=1)
            finite_front = np.isfinite(world_front_all).all(axis=1)
            back_ok = np.asarray(back_valid, dtype=bool) & finite_back
            front_ok = np.asarray(front_valid, dtype=bool) & finite_front

            normal_uv = normals_arr[uv[:, 0], uv[:, 1]]
            normal_valid = normals_valid_arr[uv[:, 0], uv[:, 1]]
            both_ok = back_ok & front_ok
            both_with_normals = both_ok & normal_valid
            signed_delta = np.full((uv.shape[0],), np.nan, dtype=np.float32)
            if bool(both_with_normals.any()):
                signed_delta[both_with_normals] = np.einsum(
                    "ij,ij->i",
                    world_front_all[both_with_normals] - world_back_all[both_with_normals],
                    normal_uv[both_with_normals],
                ).astype(np.float32, copy=False)
            signed_delta_valid = both_with_normals & np.isfinite(signed_delta)

            should_swap_bbox = False
            use_ambiguous_fallback = False
            if bool(args.flip_check_enabled) and bool(both_ok.any()):
                flip_checks_total += 1
                n_vote_points = int(np.count_nonzero(signed_delta_valid))
                orientation_samples_used += n_vote_points
                if n_vote_points < int(args.flip_check_min_band_points):
                    flip_skipped_insufficient_points += 1
                else:
                    delta_vals = signed_delta[signed_delta_valid].astype(np.float64, copy=False)
                    median_delta = float(np.median(delta_vals))
                    separation_scale = float(np.median(np.abs(delta_vals)))
                    if not np.isfinite(separation_scale):
                        separation_scale = 0.0
                    margin = max(
                        float(args.flip_check_abs_margin),
                        float(args.flip_check_rel_margin) * separation_scale,
                    )
                    if median_delta < -margin:
                        should_swap_bbox = True
                        flip_swaps_applied += 1
                    elif median_delta <= margin:
                        use_ambiguous_fallback = True
                        flip_skipped_ambiguous += 1
                        orientation_sample_fallback_used += 1

            only_back = back_ok & (~front_ok)
            only_front = front_ok & (~back_ok)
            back_from_back = np.zeros((uv.shape[0],), dtype=bool)
            back_from_front = np.zeros((uv.shape[0],), dtype=bool)
            front_from_back = np.zeros((uv.shape[0],), dtype=bool)
            front_from_front = np.zeros((uv.shape[0],), dtype=bool)

            if should_swap_bbox:
                back_from_front = front_ok
                front_from_back = back_ok
            elif use_ambiguous_fallback:
                positive = signed_delta_valid & (signed_delta > 0.0)
                negative = signed_delta_valid & (signed_delta < 0.0)
                back_from_back = only_back | positive
                back_from_front = negative
                front_from_front = only_front | positive
                front_from_back = negative
            else:
                back_from_back = back_ok
                front_from_front = front_ok

            uv_b, world_b = _concat_selected_points(
                uv=uv,
                world_back_all=world_back_all,
                world_front_all=world_front_all,
                from_back_mask=back_from_back,
                from_front_mask=back_from_front,
            )
            uv_f, world_f = _concat_selected_points(
                uv=uv,
                world_back_all=world_back_all,
                world_front_all=world_front_all,
                from_back_mask=front_from_back,
                from_front_mask=front_from_front,
            )

            if uv_b.shape[0] > 0:
                world_b_robust = _robustify_samples_with_dense_projection(
                    uv_b,
                    world_b,
                    subsample_stride=int(dense_subsample_stride),
                    neighborhood_radius=dp_radius,
                    reject_outlier_fraction=dp_reject_frac,
                    reject_min_keep=dp_reject_min_keep,
                )
                _accumulate_displaced(sum_back, count_back, uv_b, world_b_robust)

            if uv_f.shape[0] > 0:
                world_f_robust = _robustify_samples_with_dense_projection(
                    uv_f,
                    world_f,
                    subsample_stride=int(dense_subsample_stride),
                    neighborhood_radius=dp_radius,
                    reject_outlier_fraction=dp_reject_frac,
                    reject_min_keep=dp_reject_min_keep,
                )
                _accumulate_displaced(sum_front, count_front, uv_f, world_f_robust)

        del model_inputs
        del disp_pred
        del disp_pred_np

        _log(args.verbose, f"batch {batch_idx}/{n_batches}: processed {len(items)} bbox crops")

    _log(args.verbose, f"processed bbox crops with points: {kept_bboxes}/{len(records)}")

    back_sparse_grid, back_sparse_valid = _finalize_sparse_prediction(sum_back, count_back, shape_hw=(h, w))
    front_sparse_grid, front_sparse_valid = _finalize_sparse_prediction(sum_front, count_front, shape_hw=(h, w))

    back_projected, back_projected_valid, back_proj_meta = _project_sparse_to_input_lattice(
        back_sparse_grid,
        back_sparse_valid,
    )
    front_projected, front_projected_valid, front_proj_meta = _project_sparse_to_input_lattice(
        front_sparse_grid,
        front_sparse_valid,
    )

    return {
        "back_grid": back_projected,
        "back_valid": back_projected_valid,
        "front_grid": front_projected,
        "front_valid": front_projected_valid,
        "back_projection_meta": back_proj_meta,
        "front_projection_meta": front_proj_meta,
        "n_back_cells": int(back_projected_valid.sum()),
        "n_front_cells": int(front_projected_valid.sum()),
        "flip_checks_total": int(flip_checks_total),
        "flip_swaps_applied": int(flip_swaps_applied),
        "flip_skipped_ambiguous": int(flip_skipped_ambiguous),
        "flip_skipped_insufficient_points": int(flip_skipped_insufficient_points),
        "orientation_mode": "global_normals_per_bbox_vote",
        "orientation_global_normals_points": int(np.count_nonzero(normals_valid_arr)),
        "orientation_bbox_votes_total": int(flip_checks_total),
        "orientation_bbox_flips_applied": int(flip_swaps_applied),
        "orientation_bbox_ambiguous": int(flip_skipped_ambiguous),
        "orientation_bbox_insufficient_points": int(flip_skipped_insufficient_points),
        "orientation_samples_used": int(orientation_samples_used),
        "orientation_sample_fallback_used": int(orientation_sample_fallback_used),
    }


def _save_surface(
    grid,
    valid,
    out_dir,
    uuid,
    step_size,
    voxel_size_um,
    source,
    metadata,
):
    save_grid = np.asarray(grid, dtype=np.float32).copy()
    save_valid = np.asarray(valid, dtype=bool)
    save_grid[~save_valid] = -1.0
    save_tifxyz(
        save_grid,
        out_dir,
        uuid,
        step_size=int(step_size),
        voxel_size_um=float(voxel_size_um),
        source=source,
        additional_metadata=metadata,
    )
    return str(Path(out_dir) / uuid)


def _rescale_grid_for_save(grid, valid, scale_factor):
    out = np.asarray(grid, dtype=np.float32).copy()
    valid_mask = np.asarray(valid, dtype=bool)
    out[~valid_mask] = -1.0
    s = int(scale_factor)
    if s != 1:
        out[valid_mask] = out[valid_mask] * float(s)
    return out


def _append_iteration_suffix(uuid_base, iteration_index, iterative_mode):
    if not bool(iterative_mode):
        return str(uuid_base)
    return f"{uuid_base}_iteration_{int(iteration_index)}"


def _extract_surface_points_for_iteration(input_grid, input_valid, retarget_factor, verbose):
    rows, cols = np.where(input_valid)
    world_points = input_grid[rows, cols].astype(np.float32, copy=False)
    uv_points = np.stack([rows, cols], axis=-1).astype(np.int32, copy=False)
    if world_points.shape[0] == 0:
        raise RuntimeError("Input tifxyz has no valid points.")
    if verbose:
        world_min = np.min(world_points, axis=0).astype(np.float32, copy=False)
        world_max = np.max(world_points, axis=0).astype(np.float32, copy=False)
        _log(
            verbose,
            "input lattice world bounds after retarget: "
            f"retarget_factor={retarget_factor:g} "
            f"z=[{world_min[0]:.3f},{world_max[0]:.3f}] "
            f"y=[{world_min[1]:.3f},{world_max[1]:.3f}] "
            f"x=[{world_min[2]:.3f},{world_max[2]:.3f}]",
        )
    return world_points, uv_points


def _run_single_iteration(
    args,
    model_state,
    crop_size,
    volume_arr,
    input_tifxyz_path,
    out_dir,
    out_prefix,
    retarget_factor,
    tifxyz_step_size,
    tifxyz_voxel_size_um,
    stored_scale_rc,
    dense_subsample_stride,
    save_scale_factor,
    iteration_index,
    iterations_requested,
    iterative_mode,
    iter_direction,
    keep_previous_wrap,
    preloaded_input=None,
):
    if preloaded_input is None:
        _, input_grid, input_valid = _load_input_grid(input_tifxyz_path, retarget_factor=retarget_factor)
    else:
        _, input_grid, input_valid = preloaded_input
    input_path_resolved = Path(input_tifxyz_path).resolve()
    input_uuid = input_path_resolved.name

    world_points, uv_points = _extract_surface_points_for_iteration(
        input_grid=input_grid,
        input_valid=input_valid,
        retarget_factor=retarget_factor,
        verbose=bool(args.verbose),
    )
    input_normals, input_normals_valid = _compute_surface_normals_from_input_grid(input_grid, input_valid)
    records = _generate_cover_bboxes_from_points(
        world_points,
        tifxyz_uuid=input_uuid,
        crop_size=crop_size,
        overlap=float(args.bbox_overlap),
        prune_bboxes=bool(args.bbox_prune),
        prune_max_remove_per_band=args.bbox_prune_max_remove_per_band,
        band_workers=int(args.bbox_band_workers),
    )
    _log(args.verbose, f"generated bboxes (retargeted coords): {len(records)}")

    infer_out = _run_triplet_inference(
        args=args,
        model_state=model_state,
        records=records,
        crop_size=crop_size,
        world_points=world_points,
        uv_points=uv_points,
        volume_arr=volume_arr,
        shape_hw=input_valid.shape,
        dense_subsample_stride=dense_subsample_stride,
        input_normals=input_normals,
        input_normals_valid=input_normals_valid,
    )

    back_merged, back_merged_valid = _merge_with_original(
        input_grid,
        input_valid,
        infer_out["back_grid"],
        infer_out["back_valid"],
    )
    front_merged, front_merged_valid = _merge_with_original(
        input_grid,
        input_valid,
        infer_out["front_grid"],
        infer_out["front_valid"],
    )

    run_meta = {
        "checkpoint_path": str(args.checkpoint_path),
        "crop_size": [int(v) for v in crop_size],
        "bbox_count": int(len(records)),
        "bbox_overlap": float(args.bbox_overlap),
        "bbox_prune": bool(args.bbox_prune),
        "bbox_band_workers": int(args.bbox_band_workers),
        "crop_input_workers": int(args.crop_input_workers),
        "triplet_output_channels": 6,
        "retarget_factor": float(retarget_factor),
        "tta_transform_effective": str(args.tta_transform),
        "tta_merge_method_effective": str(args.tta_merge_method),
        "n_pred_back_cells": int(infer_out["n_back_cells"]),
        "n_pred_front_cells": int(infer_out["n_front_cells"]),
        "flip_check_enabled": bool(args.flip_check_enabled),
        "flip_check_abs_margin": float(args.flip_check_abs_margin),
        "flip_check_rel_margin": float(args.flip_check_rel_margin),
        "flip_check_min_band_points": int(args.flip_check_min_band_points),
        "flip_checks_total": int(infer_out.get("flip_checks_total", 0)),
        "flip_swaps_applied": int(infer_out.get("flip_swaps_applied", 0)),
        "flip_skipped_ambiguous": int(infer_out.get("flip_skipped_ambiguous", 0)),
        "flip_skipped_insufficient_points": int(infer_out.get("flip_skipped_insufficient_points", 0)),
        "orientation_mode": str(infer_out.get("orientation_mode", "legacy")),
        "orientation_global_normals_points": int(infer_out.get("orientation_global_normals_points", 0)),
        "orientation_bbox_votes_total": int(infer_out.get("orientation_bbox_votes_total", 0)),
        "orientation_bbox_flips_applied": int(infer_out.get("orientation_bbox_flips_applied", 0)),
        "orientation_bbox_ambiguous": int(infer_out.get("orientation_bbox_ambiguous", 0)),
        "orientation_bbox_insufficient_points": int(infer_out.get("orientation_bbox_insufficient_points", 0)),
        "orientation_samples_used": int(infer_out.get("orientation_samples_used", 0)),
        "orientation_sample_fallback_used": int(infer_out.get("orientation_sample_fallback_used", 0)),
        "save_coordinate_scale_factor": int(save_scale_factor),
        "stored_scale_rc": None if stored_scale_rc is None else [float(stored_scale_rc[0]), float(stored_scale_rc[1])],
        "effective_step_size_used": int(tifxyz_step_size),
        "projection_dense_interp_method": str(_DENSE_PROJECTION_INTERP_METHOD),
        "projection_dense_uv_mode": "per_bbox",
        "projection_dense_neighborhood": int(2 * int(args.dp_radius) + 1),
        "projection_dense_reject_outlier_fraction": float(args.dp_reject_frac),
        "projection_dense_reject_min_keep": int(args.dp_reject_min_keep),
        "projection_dense_scale_factor": int(dense_subsample_stride),
        "front_projection": infer_out["front_projection_meta"],
        "back_projection": infer_out["back_projection_meta"],
        "iteration_index": int(iteration_index),
        "iterations_requested": int(iterations_requested),
        "iter_direction": None if iter_direction is None else str(iter_direction),
        "keep_previous_wrap": bool(keep_previous_wrap),
        "iterative_mode": bool(iterative_mode),
        "run_argv": list(sys.argv[1:]),
    }

    outputs = {}
    source = str(args.checkpoint_path)
    input_grid_save = _rescale_grid_for_save(input_grid, input_valid, save_scale_factor)
    back_merged_save = _rescale_grid_for_save(back_merged, back_merged_valid, save_scale_factor)
    front_merged_save = _rescale_grid_for_save(front_merged, front_merged_valid, save_scale_factor)
    original_uuid = _append_iteration_suffix(out_prefix, iteration_index, iterative_mode)
    back_uuid = _append_iteration_suffix(f"{out_prefix}_back", iteration_index, iterative_mode)
    front_uuid = _append_iteration_suffix(f"{out_prefix}_front", iteration_index, iterative_mode)

    if args.save_original_copy:
        original_target = Path(out_dir) / original_uuid
        if original_target.resolve() != input_path_resolved:
            outputs["original"] = _save_surface(
                input_grid_save,
                input_valid,
                out_dir,
                original_uuid,
                tifxyz_step_size,
                tifxyz_voxel_size_um,
                source=source,
                metadata={**run_meta, "surface_role": "original"},
            )
        else:
            outputs["original"] = str(input_path_resolved)
            _log(
                args.verbose,
                "original output path equals input tifxyz path; skipping rewrite and reusing input as unchanged original.",
            )

    save_back = True
    save_front = True
    if bool(iterative_mode) and int(iteration_index) >= 2 and (not bool(keep_previous_wrap)):
        if str(iter_direction) == "front":
            save_back = False
        elif str(iter_direction) == "back":
            save_front = False

    if save_back:
        outputs["back"] = _save_surface(
            back_merged_save,
            back_merged_valid,
            out_dir,
            back_uuid,
            tifxyz_step_size,
            tifxyz_voxel_size_um,
            source=source,
            metadata={**run_meta, "surface_role": "back"},
        )
    if save_front:
        outputs["front"] = _save_surface(
            front_merged_save,
            front_merged_valid,
            out_dir,
            front_uuid,
            tifxyz_step_size,
            tifxyz_voxel_size_um,
            source=source,
            metadata={**run_meta, "surface_role": "front"},
        )

    chain_valid_cells = None
    if iter_direction in {"front", "back"}:
        chain_valid_cells = int(infer_out[f"n_{iter_direction}_cells"])

    return {
        "input_tifxyz_path": str(input_path_resolved),
        "outputs": outputs,
        "n_pred_back_cells": int(infer_out["n_back_cells"]),
        "n_pred_front_cells": int(infer_out["n_front_cells"]),
        "chain_valid_cells": chain_valid_cells,
    }


def run(args):
    args = _canonicalize_tta_settings(args)
    retarget_factor = float(2 ** int(args.volume_scale))
    surface, input_grid, input_valid = _load_input_grid(args.tifxyz_path, retarget_factor=retarget_factor)
    input_uuid = Path(args.tifxyz_path).resolve().name
    out_prefix = args.output_prefix if args.output_prefix else input_uuid

    model_state = load_model(args)
    model_config = model_state["model_config"]
    crop_size = _resolve_crop_size(args, model_config)
    tifxyz_step_size, tifxyz_voxel_size_um, stored_scale_rc = resolve_tifxyz_params(
        args,
        model_config,
        args.volume_scale,
        input_scale=surface.get_scale_tuple(),
    )
    if stored_scale_rc is not None:
        step_y = _scale_to_subsample_stride(stored_scale_rc[0])
        step_x = _scale_to_subsample_stride(stored_scale_rc[1])
        if step_y != step_x:
            raise RuntimeError(
                "Triplet wrap inference currently requires isotropic stored scale; "
                f"got scale={stored_scale_rc!r} -> steps ({step_y}, {step_x})."
            )
    dense_subsample_stride = int(max(1, tifxyz_step_size))

    volume_root = zarr.open(args.volume_path, mode="r")

    class _Holder:
        pass

    holder = _Holder()
    holder.volume = volume_root
    holder.extra = {}
    volume_arr = _resolve_segment_volume(holder, volume_scale=args.volume_scale)
    _log(
        args.verbose,
        f"resolved volume level shape={tuple(int(v) for v in volume_arr.shape)} volume_scale={int(args.volume_scale)}",
    )

    out_dir = str(Path(args.out_dir).resolve()) if args.out_dir else str(Path(args.tifxyz_path).resolve().parent)
    os.makedirs(out_dir, exist_ok=True)
    save_scale_factor = int(2 ** int(args.volume_scale))
    iterative_mode = args.iterations is not None
    iterations_requested = int(args.iterations) if iterative_mode else 1
    iter_direction = str(args.iter_direction) if iterative_mode else None
    current_tifxyz_path = str(Path(args.tifxyz_path).resolve())
    outputs_by_iteration = {}
    iterations_completed = 0
    stop_reason = None

    for iteration_index in range(1, iterations_requested + 1):
        _log(
            args.verbose,
            f"[iteration {iteration_index}/{iterations_requested}] input={current_tifxyz_path}",
        )
        iter_result = _run_single_iteration(
            args=args,
            model_state=model_state,
            crop_size=crop_size,
            volume_arr=volume_arr,
            input_tifxyz_path=current_tifxyz_path,
            out_dir=out_dir,
            out_prefix=out_prefix,
            retarget_factor=retarget_factor,
            tifxyz_step_size=tifxyz_step_size,
            tifxyz_voxel_size_um=tifxyz_voxel_size_um,
            stored_scale_rc=stored_scale_rc,
            dense_subsample_stride=dense_subsample_stride,
            save_scale_factor=save_scale_factor,
            iteration_index=iteration_index,
            iterations_requested=iterations_requested,
            iterative_mode=iterative_mode,
            iter_direction=iter_direction,
            keep_previous_wrap=bool(args.keep_previous_wrap),
            preloaded_input=(surface, input_grid, input_valid) if iteration_index == 1 else None,
        )
        outputs_by_iteration[str(iteration_index)] = {
            "input_tifxyz_path": iter_result["input_tifxyz_path"],
            "outputs": iter_result["outputs"],
            "n_pred_back_cells": int(iter_result["n_pred_back_cells"]),
            "n_pred_front_cells": int(iter_result["n_pred_front_cells"]),
        }
        iterations_completed = int(iteration_index)

        if not iterative_mode:
            continue

        chain_valid_cells = int(iter_result["chain_valid_cells"])
        outputs_by_iteration[str(iteration_index)]["chain_valid_cells"] = chain_valid_cells
        outputs_by_iteration[str(iteration_index)]["chained_direction"] = str(iter_direction)
        if chain_valid_cells <= 0 and iteration_index < iterations_requested:
            stop_reason = f"no_valid_{iter_direction}_cells"
            _log(
                args.verbose,
                f"stopping iterative chaining early at iteration {iteration_index}: {stop_reason}",
            )
            break

        if iteration_index < iterations_requested:
            next_input_path = iter_result["outputs"].get(iter_direction, None)
            if next_input_path is None:
                raise RuntimeError(
                    f"Iteration {iteration_index} did not save chained direction output '{iter_direction}'."
                )
            current_tifxyz_path = str(next_input_path)

    if not iterative_mode:
        return outputs_by_iteration["1"]["outputs"]

    return {
        "iterations_requested": int(iterations_requested),
        "iterations_completed": int(iterations_completed),
        "iter_direction": str(iter_direction),
        "keep_previous_wrap": bool(args.keep_previous_wrap),
        "stopped_early": bool(iterations_completed < iterations_requested),
        "stop_reason": stop_reason,
        "outputs_by_iteration": outputs_by_iteration,
    }


def run_copy_displacement(copy_args):
    argv = _copy_args_to_argv(copy_args)
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        detail = str(exc)
        raise RuntimeError(f"Invalid copy args for infer_rowcol_triplet_wraps: {detail}") from exc
    return run(args)


def main(argv=None):
    args = parse_args(argv)
    outputs = run(args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
