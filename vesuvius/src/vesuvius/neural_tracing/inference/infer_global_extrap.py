import argparse
import time

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
    _print_stacked_displacement_debug,
    _resolve_settings,
    _RuntimeProfiler,
    _save_merged_surface_tifxyz,
    _select_extrap_uvs_for_sampling,
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
    _grid_in_bounds_mask,
    _initialize_window_state,
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
    if args.agg_extrap_lines is not None and args.agg_extrap_lines < 1:
        parser.error("--agg-extrap-lines must be >= 1 when provided")
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
    extrap_uv_to_zyx,
):
    cond_valid_base = np.asarray(cond_valid, dtype=bool)
    cond_zyxs64 = np.asarray(cond_zyxs, dtype=np.float64)
    crop_size = tuple(int(v) for v in crop_size)
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    volume_for_crops = _resolve_segment_volume(tgt_segment, volume_scale=volume_scale)
    extrap_uv_to_zyx = extrap_uv_to_zyx if isinstance(extrap_uv_to_zyx, dict) else {}
    extrap_uv_to_zyx_get = extrap_uv_to_zyx.get

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
            pt = extrap_uv_to_zyx_get(uv_key)
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
                "cond_world": cond_world,
                "extrap_world": extrap_world,
                "n_cond": int(cond_uv.shape[0]),
                "n_query": int(uv_query_flat.shape[0]),
                "n_extrap": int(extrap_uv.shape[0]),
            }
        )

    return bbox_crops

def _run_inference(args, bbox_crops, model_state, verbose=True):
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


def _empty_stack_samples():
    return {
        "uv": np.zeros((0, 2), dtype=np.int64),
        "world": np.zeros((0, 3), dtype=np.float32),
        "displacement": np.zeros((0, 3), dtype=np.float32),
        "stack_count": np.zeros((0,), dtype=np.uint32),
    }


def _sample_displacement_for_extrap_uvs(
    stacked_displacement,
    extrap_uv_to_zyx,
    grow_direction,
    max_lines=None,
):
    if stacked_displacement is None or not extrap_uv_to_zyx:
        return _empty_stack_samples()

    sampled_uv = _select_extrap_uvs_for_sampling(extrap_uv_to_zyx, grow_direction, max_lines=max_lines)
    if sampled_uv.shape[0] == 0:
        return _empty_stack_samples()

    disp = np.asarray(stacked_displacement["displacement"], dtype=np.float32)
    count = np.asarray(stacked_displacement["count"], dtype=np.uint32)
    min_corner = np.asarray(stacked_displacement["min_corner"], dtype=np.int64)
    shape = np.asarray(stacked_displacement["shape"], dtype=np.int64)

    sampled_world = _lookup_extrap_zyxs_for_uvs(sampled_uv, extrap_uv_to_zyx)
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


def _lookup_extrap_zyxs_for_uvs(sampled_uv, extrap_uv_to_zyx):
    return np.asarray(
        [extrap_uv_to_zyx[(int(uv[0]), int(uv[1]))] for uv in sampled_uv],
        dtype=np.float32,
    )


def _sample_extrap_no_disp(extrap_uv_to_zyx, grow_direction, max_lines=None):
    if not extrap_uv_to_zyx:
        return _empty_stack_samples()

    sampled_uv = _select_extrap_uvs_for_sampling(extrap_uv_to_zyx, grow_direction, max_lines=max_lines)
    if sampled_uv.shape[0] == 0:
        return _empty_stack_samples()

    sampled_world = _lookup_extrap_zyxs_for_uvs(sampled_uv, extrap_uv_to_zyx)
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

    sample_inputs = torch.cat([disp_weighted_t, valid_t, count_t], dim=1)
    sampled_all = F.grid_sample(
        sample_inputs,
        grid,
        mode="bilinear",
        align_corners=True,
    ).view(5, -1)
    sampled_disp_weighted = sampled_all[:3].permute(1, 0)
    sampled_valid_weight = sampled_all[3]
    sampled_count = sampled_all[4]

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
        if expected_in_channels != 3:
            raise RuntimeError(
                f"infer_global_extrap only supports 3-channel models; got in_channels={expected_in_channels}"
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

    if int(args.iterations) > 1:
        with profiler.section("initialize_full_surface"):
            tgt_segment.use_full_resolution()
            base_x, base_y, base_z, base_valid = tgt_segment[:]
            current_zyxs = np.stack([base_z, base_y, base_x], axis=-1).copy()
            current_valid = np.asarray(base_valid, dtype=bool).copy()
            current_uv_offset = (0, 0)
        if args.verbose:
            print("Using full input surface as initial conditioning for iterative growth.")
    else:
        with profiler.section("initialize_window"):
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
    extrap_uv_to_zyx = {}
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

        with profiler.section("iter_get_edge_bboxes"):
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

        with profiler.section("iter_one_shot_extrapolation"):
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

        with profiler.section("iter_aggregate_extrapolation"):
            one_uv, one_world = _finite_uv_world(one_shot.get("uv_query_flat"), one_shot.get("extrap_coords_world"))
            one_samples = [(one_uv, one_world)] if len(one_uv) > 0 else []
            one_grid, one_valid, one_offset = _aggregate_pred_samples_to_uv_grid(one_samples)
            extrap_uv_to_zyx = _grid_to_uv_world_dict(one_grid, one_valid, one_offset)

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
                extrap_uv_to_zyx=extrap_uv_to_zyx,
            )
        _print_bbox_crop_debug_table(bbox_crops, verbose=args.verbose)
        bbox_results = bbox_crops

        stacked_displacement = None
        if run_model_inference and model_state is not None:
            with profiler.section("iter_displacement_inference"):
                per_crop_fields = _run_inference(
                    args,
                    bbox_crops,
                    model_state,
                    verbose=args.verbose,
                )
            with profiler.section("iter_stack_displacements"):
                stacked_displacement = _stack_displacement_results(per_crop_fields)
        _print_stacked_displacement_debug(stacked_displacement, verbose=args.verbose)
        if extrap_only_mode:
            with profiler.section("iter_sample_extrap_no_disp"):
                agg_samples = _sample_extrap_no_disp(
                    extrap_uv_to_zyx,
                    grow_direction,
                    max_lines=args.agg_extrap_lines,
                )
        else:
            with profiler.section("iter_sample_stacked_displacement"):
                agg_samples = _sample_displacement_for_extrap_uvs(
                    stacked_displacement,
                    extrap_uv_to_zyx,
                    grow_direction,
                    max_lines=args.agg_extrap_lines,
                )

        _print_iteration_summary(
            bbox_results,
            one_shot,
            extrap_uv_to_zyx,
            grow_direction,
            verbose=args.verbose,
        )
        _print_agg_extrap_sampling_debug(
            agg_samples,
            extrap_uv_to_zyx,
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
    with profiler.section("final_merge_full_surface"):
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
    with profiler.section("save_tifxyz"):
        _save_merged_surface_tifxyz(args, merged, checkpoint_path, model_config, call_args)

    if args.napari:
        disp_bbox = None if stacked_displacement is None else stacked_displacement["bbox"]
        with profiler.section("napari"):
            _show_napari(
                current_zyxs,
                current_valid,
                bbox_results,
                one_shot,
                extrap_uv_to_zyx,
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
