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

try:
    import trimesh
except Exception:  # pragma: no cover - optional dependency at runtime
    trimesh = None

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency at runtime
    njit = None

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
from vesuvius.neural_tracing.heatmap_single_point.tifxyz import save_tifxyz
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
            f"(slot A + slot B), got C={int(disp_batch.shape[1])}."
        )
    branch_a = disp_batch[:, 0:3]
    branch_b = disp_batch[:, 3:6]
    return branch_a, branch_b


def _estimate_global_unit_normal(input_normals, input_normals_valid):
    normals = np.asarray(input_normals, dtype=np.float32)
    valid = np.asarray(input_normals_valid, dtype=bool)
    if normals.ndim != 3 or normals.shape[2] != 3:
        raise RuntimeError(f"Expected input_normals shape [H,W,3], got {tuple(normals.shape)}")
    if valid.shape != normals.shape[:2]:
        raise RuntimeError(
            f"input_normals_valid shape {tuple(valid.shape)} does not match normals shape {tuple(normals.shape[:2])}"
        )
    if not bool(valid.any()):
        raise RuntimeError(
            "No valid surface normals available; cannot build global triplet direction priors."
        )

    vecs = normals[valid]
    finite = np.isfinite(vecs).all(axis=1)
    vecs = vecs[finite]
    if vecs.shape[0] == 0:
        raise RuntimeError(
            "No finite surface normals available; cannot build global triplet direction priors."
        )

    mags = np.linalg.norm(vecs, axis=1)
    vecs = vecs[mags > 1e-6]
    if vecs.shape[0] == 0:
        raise RuntimeError(
            "Surface normals are degenerate; cannot build global triplet direction priors."
        )

    unit_vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-6)
    mean_vec = np.mean(unit_vecs, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    mean_norm = float(np.linalg.norm(mean_vec))
    if (not np.isfinite(mean_norm)) or mean_norm <= 1e-6:
        raise RuntimeError(
            "Global surface normal estimate is degenerate; cannot build triplet direction priors."
        )
    return (mean_vec / mean_norm).astype(np.float32, copy=False)


def _build_triplet_direction_priors_for_crop(crop_size, cond_vox, global_unit_normal, mask_mode="cond"):
    crop_size = tuple(int(v) for v in crop_size)
    if len(crop_size) != 3:
        raise RuntimeError(f"crop_size must be length 3, got {crop_size}")

    cond = np.asarray(cond_vox, dtype=np.float32)
    if cond.shape != crop_size:
        raise RuntimeError(f"cond_vox shape must match crop_size {crop_size}, got {tuple(cond.shape)}")

    n = np.asarray(global_unit_normal, dtype=np.float32).reshape(3)
    priors = np.zeros((6, *crop_size), dtype=np.float32)
    for axis in range(3):
        priors[axis, ...] = n[axis]
        priors[axis + 3, ...] = -n[axis]

    mode = str(mask_mode).lower()
    if mode == "cond":
        priors *= (cond > 0.5).astype(np.float32, copy=False)[None, ...]
    elif mode != "full":
        raise RuntimeError(f"Unknown triplet direction prior mask mode: {mask_mode!r}")
    return priors


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
    global_unit_normal = _estimate_global_unit_normal(normals_arr, normals_valid_arr)
    orientation_global_normals_points = int(np.count_nonzero(normals_valid_arr))

    volume_shape = np.asarray(np.shape(volume_arr), dtype=np.float64)
    if volume_shape.size < 3:
        raise RuntimeError(f"Expected 3D volume for triplet inference, got shape={tuple(np.shape(volume_arr))}")

    expected_in_channels = int(model_state["expected_in_channels"])
    if expected_in_channels != 8:
        raise RuntimeError(
            "Triplet wrap inference requires direction-conditioned checkpoints with in_channels=8 "
            "(volume + conditioning + 6 direction-prior channels); "
            f"checkpoint expects in_channels={expected_in_channels}."
        )
    model_config = dict(model_state.get("model_config") or {})
    triplet_direction_prior_mask = str(model_config.get("triplet_direction_prior_mask", "cond")).lower()
    if triplet_direction_prior_mask not in {"cond", "full"}:
        raise RuntimeError(
            "triplet_direction_prior_mask in checkpoint config must be 'cond' or 'full', "
            f"got {triplet_direction_prior_mask!r}."
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
        batch_np = np.empty((len(items), 8, d, h_c, w_c), dtype=np.float32)
        for i, item in enumerate(items):
            batch_np[i, 0] = item["volume"]
            batch_np[i, 1] = item["cond_vox"]
            batch_np[i, 2:8] = _build_triplet_direction_priors_for_crop(
                crop_size=crop_size,
                cond_vox=item["cond_vox"],
                global_unit_normal=global_unit_normal,
                mask_mode=triplet_direction_prior_mask,
            )

        model_inputs = torch.from_numpy(batch_np).to(args.device, non_blocking=True)
        disp_pred = predict_displacement(args, model_state, model_inputs, use_tta=bool(args.tta), profiler=None)
        if disp_pred is None:
            raise RuntimeError("Model output did not contain 'displacement'.")
        disp_pred_np = (
            disp_pred.detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False)
        )
        slot_a_batch, slot_b_batch = _split_triplet_displacement_channels(disp_pred_np)

        for i, item in enumerate(items):
            local = item["local"]
            uv = item["uv"]
            world = item["world"]

            slot_a_disp, slot_a_valid = _sample_trilinear_displacement_stack(slot_a_batch[i], local)
            slot_b_disp, slot_b_valid = _sample_trilinear_displacement_stack(slot_b_batch[i], local)

            # Deterministic branch mapping with direction priors:
            # slot A (+normal prior) -> front, slot B (-normal prior) -> back.
            world_front_all = world + slot_a_disp
            world_back_all = world + slot_b_disp
            front_ok = np.asarray(slot_a_valid, dtype=bool) & np.isfinite(world_front_all).all(axis=1)
            back_ok = np.asarray(slot_b_valid, dtype=bool) & np.isfinite(world_back_all).all(axis=1)

            if bool(back_ok.any()):
                uv_b = uv[back_ok].astype(np.int32, copy=False)
                world_b = world_back_all[back_ok].astype(np.float32, copy=False)
                world_b_robust = _robustify_samples_with_dense_projection(
                    uv_b,
                    world_b,
                    subsample_stride=int(dense_subsample_stride),
                    neighborhood_radius=dp_radius,
                    reject_outlier_fraction=dp_reject_frac,
                    reject_min_keep=dp_reject_min_keep,
                )
                _accumulate_displaced(sum_back, count_back, uv_b, world_b_robust)

            if bool(front_ok.any()):
                uv_f = uv[front_ok].astype(np.int32, copy=False)
                world_f = world_front_all[front_ok].astype(np.float32, copy=False)
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
        "orientation_mode": "global_direction_prior_fixed",
        "orientation_global_normals_points": int(orientation_global_normals_points),
        "orientation_global_normal_zyx": [float(global_unit_normal[0]), float(global_unit_normal[1]), float(global_unit_normal[2])],
        "triplet_direction_prior_mask": str(triplet_direction_prior_mask),
        "triplet_slot_to_output": {"A": "front", "B": "back"},
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
    apply_mesh_cleanup=True,
):
    save_grid = np.asarray(grid, dtype=np.float32).copy()
    save_valid = np.asarray(valid, dtype=bool)
    metadata_out = dict(metadata) if isinstance(metadata, dict) else {}

    if bool(apply_mesh_cleanup):
        save_grid, cleanup_meta = _cleanup_surface_grid_before_save(
            save_grid,
            save_valid,
            target_step_size=float(step_size),
        )
        metadata_out.update(cleanup_meta)

    save_grid[~save_valid] = -1.0
    save_tifxyz(
        save_grid,
        out_dir,
        uuid,
        step_size=int(step_size),
        voxel_size_um=float(voxel_size_um),
        source=source,
        additional_metadata=metadata_out,
    )
    return str(Path(out_dir) / uuid)


def _build_valid_grid_triangles(valid_mask):
    valid = np.asarray(valid_mask, dtype=bool)
    h, w = valid.shape
    if h < 2 or w < 2:
        return np.zeros((0, 3), dtype=np.int64)

    quad_valid = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    qr, qc = np.where(quad_valid)
    if qr.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    index_grid = -np.ones_like(valid, dtype=np.int64)
    vr, vc = np.where(valid)
    index_grid[vr, vc] = np.arange(vr.size, dtype=np.int64)

    v00 = index_grid[qr, qc]
    v10 = index_grid[qr + 1, qc]
    v01 = index_grid[qr, qc + 1]
    v11 = index_grid[qr + 1, qc + 1]

    t0 = np.stack([v00, v10, v11], axis=1)
    t1 = np.stack([v00, v11, v01], axis=1)
    faces = np.concatenate([t0, t1], axis=0)
    return faces.astype(np.int64, copy=False)


def _build_valid_grid_edges(valid_mask):
    valid = np.asarray(valid_mask, dtype=bool)
    h, w = valid.shape
    if h == 0 or w == 0:
        return np.zeros((0, 2), dtype=np.int64)

    index_grid = -np.ones_like(valid, dtype=np.int64)
    vr, vc = np.where(valid)
    if vr.size < 2:
        return np.zeros((0, 2), dtype=np.int64)
    index_grid[vr, vc] = np.arange(vr.size, dtype=np.int64)

    horiz = valid[:, :-1] & valid[:, 1:]
    hr, hc = np.where(horiz)
    h_edges = np.zeros((0, 2), dtype=np.int64)
    if hr.size > 0:
        h_edges = np.stack([index_grid[hr, hc], index_grid[hr, hc + 1]], axis=1)

    vert = valid[:-1, :] & valid[1:, :]
    vr_e, vc_e = np.where(vert)
    v_edges = np.zeros((0, 2), dtype=np.int64)
    if vr_e.size > 0:
        v_edges = np.stack([index_grid[vr_e, vc_e], index_grid[vr_e + 1, vc_e]], axis=1)

    if h_edges.size == 0 and v_edges.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if h_edges.size == 0:
        return v_edges.astype(np.int64, copy=False)
    if v_edges.size == 0:
        return h_edges.astype(np.int64, copy=False)
    return np.concatenate([h_edges, v_edges], axis=0).astype(np.int64, copy=False)


def _resolve_duplicate_vertex_positions(vertices, edges, target_step):
    verts = np.asarray(vertices, dtype=np.float64).copy()
    if verts.shape[0] < 2:
        return verts, 0

    tol = max(1e-3, float(target_step) * 1e-4)
    max_shift = max(tol, 0.75 * float(target_step))
    original = verts.copy()
    quantized = np.round(verts / tol).astype(np.int64)
    _, inverse, counts = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
    dup_groups = np.where(counts > 1)[0]
    if dup_groups.size == 0:
        return verts, 0

    edge_idx = np.asarray(edges, dtype=np.int64)
    if edge_idx.size == 0:
        return verts, 0
    src = np.concatenate([edge_idx[:, 0], edge_idx[:, 1]], axis=0)
    dst = np.concatenate([edge_idx[:, 1], edge_idx[:, 0]], axis=0)
    order = np.argsort(src, kind="stable")
    src_sorted = src[order]
    dst_sorted = dst[order]
    vertex_ids = np.arange(verts.shape[0], dtype=np.int64)
    nbr_start = np.searchsorted(src_sorted, vertex_ids, side="left")
    nbr_end = np.searchsorted(src_sorted, vertex_ids, side="right")

    fixed = 0
    for gid in dup_groups.tolist():
        members = np.where(inverse == gid)[0]
        if members.size <= 1:
            continue
        member_set = set(int(v) for v in members.tolist())
        for rank, vid in enumerate(members[1:], start=1):
            vid_i = int(vid)
            start = int(nbr_start[vid_i])
            end = int(nbr_end[vid_i])
            nbrs_all = dst_sorted[start:end]
            nbrs = [int(n) for n in nbrs_all.tolist() if int(n) not in member_set]
            if len(nbrs) > 0:
                new_pos = np.mean(verts[np.asarray(nbrs, dtype=np.int64)], axis=0)
            else:
                seed = float((vid_i + 1) * (rank + 3))
                direction = np.array(
                    [
                        np.sin(seed),
                        np.cos(2.0 * seed),
                        np.sin(3.0 * seed + 0.5),
                    ],
                    dtype=np.float64,
                )
                norm = np.linalg.norm(direction)
                if norm < 1e-8:
                    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                else:
                    direction /= norm
                new_pos = verts[vid_i] + direction * tol
            delta = new_pos - original[vid_i]
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_shift:
                new_pos = original[vid_i] + (delta / delta_norm) * max_shift
            verts[vid_i] = new_pos
            fixed += 1

    return verts, int(fixed)


if njit is not None:
    @njit(cache=True)
    def _regularize_edge_lengths_numba_kernel(
        vertices,
        edges,
        target_step,
        iterations,
        relax_step,
        anchor_weight,
        max_displacement,
    ):
        verts = vertices.copy()
        original = vertices.copy()
        n_vertices = verts.shape[0]
        n_edges = edges.shape[0]
        eps = 1e-8

        for _ in range(iterations):
            disp = np.zeros((n_vertices, 3), dtype=np.float64)
            counts = np.zeros((n_vertices,), dtype=np.float64)

            for k in range(n_edges):
                i = edges[k, 0]
                j = edges[k, 1]

                dx0 = verts[j, 0] - verts[i, 0]
                dx1 = verts[j, 1] - verts[i, 1]
                dx2 = verts[j, 2] - verts[i, 2]

                length = np.sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2)
                if length <= eps:
                    continue

                inv_len = 1.0 / length
                err = length - target_step
                move0 = 0.5 * err * dx0 * inv_len
                move1 = 0.5 * err * dx1 * inv_len
                move2 = 0.5 * err * dx2 * inv_len

                disp[i, 0] += move0
                disp[i, 1] += move1
                disp[i, 2] += move2
                disp[j, 0] -= move0
                disp[j, 1] -= move1
                disp[j, 2] -= move2
                counts[i] += 1.0
                counts[j] += 1.0

            for i in range(n_vertices):
                if counts[i] > 0.0:
                    verts[i, 0] += relax_step * (disp[i, 0] / counts[i]) + anchor_weight * (original[i, 0] - verts[i, 0])
                    verts[i, 1] += relax_step * (disp[i, 1] / counts[i]) + anchor_weight * (original[i, 1] - verts[i, 1])
                    verts[i, 2] += relax_step * (disp[i, 2] / counts[i]) + anchor_weight * (original[i, 2] - verts[i, 2])
                else:
                    verts[i, 0] += anchor_weight * (original[i, 0] - verts[i, 0])
                    verts[i, 1] += anchor_weight * (original[i, 1] - verts[i, 1])
                    verts[i, 2] += anchor_weight * (original[i, 2] - verts[i, 2])

                dx0 = verts[i, 0] - original[i, 0]
                dx1 = verts[i, 1] - original[i, 1]
                dx2 = verts[i, 2] - original[i, 2]
                dist = np.sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2)
                if dist > max_displacement and dist > eps:
                    scale = max_displacement / dist
                    verts[i, 0] = original[i, 0] + dx0 * scale
                    verts[i, 1] = original[i, 1] + dx1 * scale
                    verts[i, 2] = original[i, 2] + dx2 * scale

        return verts
else:
    _regularize_edge_lengths_numba_kernel = None


def _regularize_edge_lengths(
    vertices,
    edges,
    target_step,
    iterations=8,
    relax_step=0.35,
    anchor_weight=0.08,
    max_displacement_ratio=0.5,
):
    verts = np.asarray(vertices, dtype=np.float64).copy()
    if verts.shape[0] == 0 or np.asarray(edges).size == 0:
        return verts

    edge_idx = np.asarray(edges, dtype=np.int64)
    if edge_idx.shape[0] == 0:
        return verts

    max_displacement = max(1e-6, float(target_step) * float(max_displacement_ratio))

    if _regularize_edge_lengths_numba_kernel is not None and edge_idx.shape[0] >= 256:
        return _regularize_edge_lengths_numba_kernel(
            verts,
            edge_idx,
            float(target_step),
            int(iterations),
            float(relax_step),
            float(anchor_weight),
            float(max_displacement),
        )

    e0 = edge_idx[:, 0]
    e1 = edge_idx[:, 1]
    original = verts.copy()
    target = float(target_step)
    eps = 1e-8

    for _ in range(int(iterations)):
        delta = verts[e1] - verts[e0]
        lengths = np.linalg.norm(delta, axis=1)
        good = lengths > eps
        if not bool(good.any()):
            break

        direction = np.zeros_like(delta)
        direction[good] = delta[good] / lengths[good, None]
        length_error = lengths - target
        move = 0.5 * length_error[:, None] * direction

        disp = np.zeros_like(verts)
        counts = np.zeros((verts.shape[0], 1), dtype=np.float64)
        np.add.at(disp, e0, move)
        np.add.at(disp, e1, -move)
        np.add.at(counts, e0, 1.0)
        np.add.at(counts, e1, 1.0)

        nonzero = counts[:, 0] > 0
        update = np.zeros_like(verts)
        update[nonzero] = disp[nonzero] / counts[nonzero]

        verts = verts + float(relax_step) * update + float(anchor_weight) * (original - verts)
        delta_from_original = verts - original
        delta_norm = np.linalg.norm(delta_from_original, axis=1)
        too_far = delta_norm > max_displacement
        if bool(too_far.any()):
            scale = (max_displacement / np.maximum(delta_norm[too_far], 1e-8))[:, None]
            verts[too_far] = original[too_far] + delta_from_original[too_far] * scale

    return verts


def _cleanup_surface_grid_before_save(grid, valid, target_step_size):
    grid_arr = np.asarray(grid, dtype=np.float32)
    valid_mask = np.asarray(valid, dtype=bool)
    out = grid_arr.copy()

    cleanup_meta = {
        "mesh_cleanup_enabled": bool(trimesh is not None),
        "mesh_cleanup_target_step": float(target_step_size),
        "mesh_cleanup_duplicate_vertices_fixed": 0,
        "mesh_cleanup_unique_edges": 0,
        "mesh_cleanup_relaxation_applied": False,
    }

    if trimesh is None:
        return out, cleanup_meta

    vr, vc = np.where(valid_mask)
    if vr.size < 3:
        return out, cleanup_meta

    faces = _build_valid_grid_triangles(valid_mask)
    if faces.shape[0] == 0:
        return out, cleanup_meta

    edges_unique = _build_valid_grid_edges(valid_mask)
    if edges_unique.shape[0] == 0:
        return out, cleanup_meta

    verts = out[vr, vc].astype(np.float64, copy=True)
    if not np.isfinite(verts).all():
        return out, cleanup_meta

    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False, validate=False)
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    except ValueError:
        return out, cleanup_meta

    cleanup_meta["mesh_cleanup_unique_edges"] = int(edges_unique.shape[0])

    verts, dup_fixed = _resolve_duplicate_vertex_positions(
        verts,
        edges_unique,
        target_step=float(target_step_size),
    )
    cleanup_meta["mesh_cleanup_duplicate_vertices_fixed"] = int(dup_fixed)

    if target_step_size is not None and float(target_step_size) > 0.0:
        verts = _regularize_edge_lengths(
            verts,
            edges_unique,
            target_step=float(target_step_size),
            iterations=8,
            relax_step=0.35,
            anchor_weight=0.08,
        )
        cleanup_meta["mesh_cleanup_relaxation_applied"] = True

    out[vr, vc] = verts.astype(np.float32, copy=False)
    return out, cleanup_meta


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
        "orientation_mode": str(infer_out.get("orientation_mode", "legacy")),
        "orientation_global_normals_points": int(infer_out.get("orientation_global_normals_points", 0)),
        "orientation_global_normal_zyx": list(infer_out.get("orientation_global_normal_zyx", [])),
        "triplet_direction_prior_mask_effective": str(infer_out.get("triplet_direction_prior_mask", "cond")),
        "triplet_slot_to_output": dict(infer_out.get("triplet_slot_to_output", {"A": "front", "B": "back"})),
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
                apply_mesh_cleanup=False,
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
