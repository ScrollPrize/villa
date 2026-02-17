import argparse
import zarr
import tifffile

import vesuvius.tifxyz as tifxyz
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid, voxelize_surface_grid_masked
from vesuvius.neural_tracing.datasets.extrapolation import _EXTRAPOLATION_METHODS, apply_degradation
from vesuvius.image_proc.intensity.normalization import normalize_zscore
from vesuvius.neural_tracing.inference.displacement_tta import (
    TTA_MERGE_METHODS,
    run_model_tta,
)
from vesuvius.neural_tracing.inference.common import (
    _aggregate_pred_samples_to_uv_grid,
    _resolve_extrapolation_settings,
    resolve_tifxyz_params,
    save_tifxyz_output,
)
from vesuvius.neural_tracing.models import load_checkpoint, resolve_checkpoint_path
from tqdm import tqdm

VALID_DIRECTIONS = ["left", "right", "down", "up"]
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


def parse_args():
    parser = argparse.ArgumentParser(description="Row/col split inference for neural tracing")
    parser.add_argument(
        "--tifxyz-path",
        type=str,
        required=True,
        help="Path to a single tifxyz segment directory (contains x.tif/y.tif/z.tif/meta.json).",
    )
    parser.add_argument("--volume-path", type=str, required=True)
    parser.add_argument("--volume-scale", type=int, default=1)
    parser.add_argument("--cond-pct", type=float, default=0.50)
    parser.add_argument("--crop-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--window-pad", type=int, default=10)
    parser.add_argument(
        "--bbox-overlap-frac",
        type=float,
        default=0.15,
        help="Fractional overlap between consecutive edge bboxes (default 0.15 = 15%%).",
    )
    parser.add_argument("--extrapolation-method", type=str, default=None)
    parser.add_argument(
        "--rbf-downsample-factor",
        type=int,
        default=None,
        help=(
            "Override RBF downsample factor for extrapolation. "
            "When set, takes precedence over checkpoint/config rbf_downsample_factor."
        ),
    )
    parser.add_argument(
        "--grow-direction",
        type=str,
        required=True,
        choices=VALID_DIRECTIONS,
        help="Direction to grow/extrapolate toward.",
    )
    parser.add_argument("--bbox-crops-out-dir", type=str, default="/tmp/rowcol_bbox_crops")
    parser.add_argument("--tifxyz-out-dir", type=str, default=None)
    parser.add_argument("--tifxyz-step-size", type=int, default=None)
    parser.add_argument("--tifxyz-voxel-size-um", type=float, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Load checkpoint config/settings but skip model forward inference.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=1,
        help="Number of grow iterations.")
    parser.add_argument(
        "--refine",
        type=int,
        default=None,
        help=(
            "Within each outer iteration, run N+1 staged refinement forward passes. "
            "Each stage predicts full displacement but only applies 1/(N+1) of it "
            "before feeding coordinates back into the next stage."
        ),
    )
    parser.add_argument(
        "--iter-keep-voxels",
        type=int,
        default=None,
        help=(
            "For iterations>1, number of newly predicted rows/cols to keep per iteration "
            "in the growth direction. If omitted, keeps all newly predicted rows/cols "
            "outside the current boundary."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8,
        help="Number of crops to process in a single batched forward pass.")
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
        help="How to merge mirrored TTA displacement predictions (default: vector_geomedian).",
    )
    parser.add_argument(
        "--tta-outlier-drop-thresh",
        type=float,
        default=1.25,
        help=(
            "Optional robust threshold multiplier for dropping outlier TTA flip variants "
            "before merge. Uses per-variant distance to the TTA median field and drops "
            "variants above median + thresh * spread (MAD, with std fallback). "
            "(default: 1.25)"
        ),
    )
    parser.add_argument(
        "--tta-outlier-drop-min-keep",
        type=int,
        default=4,
        help=(
            "Minimum number of TTA variants to keep when --tta-outlier-drop-thresh is set "
            "(default: 4)."
        ),
    )
    parser.set_defaults(tta=True)
    args = parser.parse_args()
    if args.iter_keep_voxels is not None and args.iter_keep_voxels < 1:
        parser.error("--iter-keep-voxels must be >= 1 when provided.")
    if args.bbox_overlap_frac < 0.0 or args.bbox_overlap_frac >= 1.0:
        parser.error("--bbox-overlap-frac must be in [0, 1).")
    if args.rbf_downsample_factor is not None and args.rbf_downsample_factor < 1:
        parser.error("--rbf-downsample-factor must be >= 1 when provided.")
    if args.refine is not None and args.refine < 1:
        parser.error("--refine must be >= 1 when provided.")
    if args.tta_outlier_drop_thresh is not None and args.tta_outlier_drop_thresh <= 0:
        parser.error("--tta-outlier-drop-thresh must be > 0 when provided.")
    if args.tta_outlier_drop_min_keep < 1:
        parser.error("--tta-outlier-drop-min-keep must be >= 1.")
    return args


def _get_growth_context(grow_direction):
    # Growth semantics are encoded by the opposite conditioning side.
    cond_direction = _get_direction_spec(grow_direction)["opposite"]
    growth_spec = _get_direction_spec(cond_direction)
    return cond_direction, growth_spec

def _clamp_window(start, size, min_val, max_val):
    size = int(size)
    start = int(start)
    if size <= 0:
        return min_val, min_val
    start = max(min_val, min(start, max_val - size + 1))
    end = start + size - 1
    return start, end

def _edge_index_from_valid(valid, cond_direction):
    spec = _DIRECTION_SPECS.get(cond_direction)
    if spec is None:
        return None, None

    valid_rows = np.any(valid, axis=1)
    valid_cols = np.any(valid, axis=0)
    if not valid_rows.any() or not valid_cols.any():
        return None, None
    if spec["axis"] == "col":
        c_idx = np.where(valid_cols)[0]
        return None, int(c_idx[spec["edge_idx"]])
    r_idx = np.where(valid_rows)[0]
    return int(r_idx[spec["edge_idx"]]), None

def _place_window_on_edge(edge_idx, window_size, cond_size, cond_direction, max_idx, clamp=True):
    # Keep window size fixed; place edge at split determined by cond_size.
    spec = _DIRECTION_SPECS.get(cond_direction)
    if spec is not None and spec["growth_sign"] > 0:
        start = edge_idx - (cond_size - 1)
    elif spec is not None:
        mask_size = window_size - cond_size
        start = edge_idx - mask_size
    else:
        start = edge_idx - (window_size // 2)
    if clamp:
        return _clamp_window(start, window_size, 0, max_idx)
    end = int(start) + int(window_size) - 1
    return int(start), int(end)

def _get_cond_edge(cond_zyxs, cond_direction):
    spec = _get_direction_spec(cond_direction)
    edge_idx = spec["edge_idx"]
    # For non-rectangular inputs the fixed column/row may have no valid cells.
    # Instead, find the per-row (col axis) or per-col (row axis) frontier.
    invalid = (cond_zyxs == -1).all(axis=-1)  # (n_rows, n_cols)
    valid = ~invalid
    if spec["axis"] == "col":
        n_rows, n_cols = valid.shape
        out = np.full((n_rows, 3), -1, dtype=cond_zyxs.dtype)
        any_valid = valid.any(axis=1)
        if edge_idx == 0 or (edge_idx == -1 and n_cols == 1):
            # leftmost valid column per row (first True)
            col_indices = np.argmax(valid, axis=1)
        else:
            # rightmost valid column per row (last True)
            col_indices = n_cols - 1 - np.argmax(valid[:, ::-1], axis=1)
        out[any_valid] = cond_zyxs[np.where(any_valid)[0], col_indices[any_valid], :]
        return out
    else:
        n_rows, n_cols = valid.shape
        out = np.full((n_cols, 3), -1, dtype=cond_zyxs.dtype)
        any_valid = valid.any(axis=0)
        if edge_idx == 0 or (edge_idx == -1 and n_rows == 1):
            # topmost valid row per column (first True)
            row_indices = np.argmax(valid, axis=0)
        else:
            # bottommost valid row per column (last True)
            row_indices = n_rows - 1 - np.argmax(valid[::-1, :], axis=0)
        out[any_valid] = cond_zyxs[row_indices[any_valid], np.where(any_valid)[0], :]
        return out

def _bbox_from_center(center, crop_size):
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    # Align to voxel indices so inclusive bounds match a crop of size crop_size.
    half = (crop_size_arr - 1) / 2.0
    min_corner = np.floor(center - half).astype(np.int64)
    max_corner = min_corner + (crop_size_arr - 1)
    return (
        int(min_corner[0]), int(max_corner[0]),
        int(min_corner[1]), int(max_corner[1]),
        int(min_corner[2]), int(max_corner[2]),
    )

def get_cond_edge_bboxes(cond_zyxs, cond_direction, crop_size, overlap_frac=0.15):
    # Build center-out crop anchors along the conditioning edge. Each chunk grows
    # while its XYZ span still fits in one crop-sized bbox.
    edge = _get_cond_edge(cond_zyxs, cond_direction)

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

    def _indices_fit_single_bbox(indices):
        pts = edge[indices]
        spans = pts.max(axis=0) - pts.min(axis=0)
        return (
            spans[0] <= (crop_size_arr[0] - 1) and
            spans[1] <= (crop_size_arr[1] - 1) and
            spans[2] <= (crop_size_arr[2] - 1)
        )

    def _chunk_ordered_indices(ordered_indices):
        chunks = []
        if len(ordered_indices) == 0:
            return chunks
        start = 0
        while start < len(ordered_indices):
            end = start + 1
            while end < len(ordered_indices):
                candidate = ordered_indices[start:end + 1]
                if _indices_fit_single_bbox(candidate):
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
            bbox = _bbox_from_center(center, crop_size)
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

def _in_bounds_mask(coords, size):
    size = np.asarray(size)
    return (
        (coords[:, 0] >= 0) & (coords[:, 0] < size[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < size[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < size[2])
    )

def _points_to_voxels(points_local, crop_size):
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    vox = np.zeros(tuple(crop_size_arr.tolist()), dtype=np.float32)
    if points_local is None or len(points_local) == 0:
        return vox
    coords = np.rint(points_local).astype(np.int64)
    coords = coords[_in_bounds_mask(coords, crop_size_arr)]
    if coords.size > 0:
        vox[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return vox

def _valid_surface_mask(zyx_grid):
    return np.isfinite(zyx_grid).all(axis=-1) & ~(zyx_grid == -1).all(axis=-1)

def _grid_in_bounds_mask(zyx_grid_local, crop_size):
    flat = zyx_grid_local.reshape(-1, 3)
    return _in_bounds_mask(flat, crop_size).reshape(zyx_grid_local.shape[:2])

def _build_model_inputs(vol_crop, cond_vox, extrap_vox, other_wraps_vox=None):
    vol_t = torch.from_numpy(vol_crop).float().unsqueeze(0).unsqueeze(0)
    cond_t = torch.from_numpy(cond_vox).float().unsqueeze(0).unsqueeze(0)
    extrap_t = torch.from_numpy(extrap_vox).float().unsqueeze(0).unsqueeze(0)
    inputs = [vol_t, cond_t, extrap_t]
    if other_wraps_vox is not None:
        other_t = torch.from_numpy(other_wraps_vox).float().unsqueeze(0).unsqueeze(0)
        inputs.append(other_t)
    return torch.cat(inputs, dim=1)

def _sample_displacement_field(pred_field, coords_local):
    if coords_local is None or coords_local.numel() == 0:
        return torch.zeros((0, 3), device=pred_field.device, dtype=pred_field.dtype)

    _, _, D, H, W = pred_field.shape
    # grid_sample uses normalized coordinates in [-1, 1].
    # Ensure grid dtype matches pred_field for AMP compatibility.
    coords_norm = coords_local.to(dtype=pred_field.dtype).clone()
    d_denom = max(D - 1, 1)
    h_denom = max(H - 1, 1)
    w_denom = max(W - 1, 1)
    coords_norm[:, 0] = 2 * coords_norm[:, 0] / d_denom - 1
    coords_norm[:, 1] = 2 * coords_norm[:, 1] / h_denom - 1
    coords_norm[:, 2] = 2 * coords_norm[:, 2] / w_denom - 1

    # coords_local is (z, y, x); grid_sample expects (x, y, z).
    grid = coords_norm[:, [2, 1, 0]].view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(pred_field, grid, mode='bilinear', align_corners=True)
    sampled = sampled.view(1, 3, -1).permute(0, 2, 1)[0]
    return sampled

def get_window_bounds_from_bboxes(zyxs, valid, bboxes, pad=2):
    h, w = zyxs.shape[:2]
    r_min, r_max = h, -1
    c_min, c_max = w, -1

    z, y, x = zyxs[..., 0], zyxs[..., 1], zyxs[..., 2]

    for bbox in bboxes:
        z_min, z_max, y_min, y_max, x_min, x_max = bbox
        in_bounds = (
            (z >= z_min) & (z <= z_max) &
            (y >= y_min) & (y <= y_max) &
            (x >= x_min) & (x <= x_max) &
            valid
        )
        if not in_bounds.any():
            continue
        valid_rows = np.any(in_bounds, axis=1)
        valid_cols = np.any(in_bounds, axis=0)
        if not valid_rows.any() or not valid_cols.any():
            continue
        r0, r1 = np.where(valid_rows)[0][[0, -1]]
        c0, c1 = np.where(valid_cols)[0][[0, -1]]
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
    zyx_extrapolated = extrapolate_fn(
        uv_cond=uv_cond_flat,
        zyx_cond=zyx_cond_flat,
        uv_query=uv_query_flat,
        min_corner=min_corner,
        crop_size=crop_size,
        cond_direction=cond_direction,
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
        'extrap_coords_local': extrap_coords_local,
        'extrap_surface': extrap_surface,
    }


def _save_crop_tiff(out_dir, name, array):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_dir / name, array)



def load_model(args):
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        raise RuntimeError("checkpoint_path not set; provide a trained rowcol_cond checkpoint.")

    model, model_config = load_checkpoint(checkpoint_path)
    model.to(args.device)
    model.eval()

    expected_in_channels = int(model_config.get("in_channels", 3))
    mixed_precision = str(model_config.get("mixed_precision", "no")).lower()
    amp_enabled = False
    amp_dtype = torch.float16
    if args.device.startswith("cuda") and mixed_precision in ("bf16", "fp16", "float16"):
        amp_enabled = True
        amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16

    ckpt_name = os.path.splitext(os.path.basename(str(checkpoint_path)))[0]
    timestamp = datetime.now().strftime("%H%M%S")
    tifxyz_uuid = f"displacement_tifxyz_{ckpt_name}_{timestamp}"

    return {
        "model": model,
        "model_config": model_config,
        "checkpoint_path": checkpoint_path,
        "expected_in_channels": expected_in_channels,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
        "tifxyz_uuid": tifxyz_uuid,
    }


def load_checkpoint_config(checkpoint_path):
    if checkpoint_path is None:
        raise RuntimeError("checkpoint_path not set; provide a trained rowcol_cond checkpoint.")
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved_path, map_location='cpu', weights_only=False)
    model_config = checkpoint.get("config")
    if model_config is None:
        raise RuntimeError(f"'config' not found in checkpoint: {resolved_path}")
    return model_config, str(resolved_path)

def _load_optional_json(path):
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}

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


def compute_window_and_split(args, stored_zyxs, valid_s, grow_direction, h_s, w_s, crop_size):
    cond_direction, direction = _get_growth_context(grow_direction)
    r_edge_s, c_edge_s = _edge_index_from_valid(valid_s, cond_direction)
    if r_edge_s is None and c_edge_s is None:
        raise RuntimeError("No valid edge found for segment.")

    if direction["axis"] == "col":
        cond_edge_strip = stored_zyxs[:, c_edge_s:c_edge_s + 1]
    else:
        cond_edge_strip = stored_zyxs[r_edge_s:r_edge_s + 1, :]

    bboxes, _ = get_cond_edge_bboxes(
        cond_edge_strip, cond_direction, crop_size,
        overlap_frac=args.bbox_overlap_frac,
    )

    r0_s, r1_s, c0_s, c1_s = get_window_bounds_from_bboxes(
        stored_zyxs, valid_s, bboxes, pad=args.window_pad
    )

    win_h = r1_s - r0_s + 1
    win_w = c1_s - c0_s + 1
    if win_h < 2 or win_w < 2:
        raise RuntimeError("Window too small after edge-based bounds.")

    # Anchor to the growth side so right/down don't re-center toward the conditioning side.
    outside_dir = grow_direction
    r_edge_outside, c_edge_outside = _edge_index_from_valid(valid_s, outside_dir)

    cond_h = win_h
    cond_w = win_w
    if direction["axis"] == "col":
        c0_s, c1_s = _place_window_on_edge(
            c_edge_outside, cond_w, cond_w, outside_dir, w_s - 1
        )
        r0_s, r1_s = _clamp_window(r0_s, cond_h, 0, h_s - 1)
    else:
        r0_s, r1_s = _place_window_on_edge(
            r_edge_outside, cond_h, cond_h, outside_dir, h_s - 1
        )
        c0_s, c1_s = _clamp_window(c0_s, cond_w, 0, w_s - 1)
    return r0_s, r1_s, c0_s, c1_s


def _build_uv_query_from_cond_points(uv_cond_pts, grow_direction, cond_pct):
    if uv_cond_pts is None or len(uv_cond_pts) == 0:
        return np.zeros((0, 0, 2), dtype=np.float64)

    _, direction = _get_growth_context(grow_direction)
    uv_cond_pts = np.asarray(uv_cond_pts)
    r_min = int(np.floor(uv_cond_pts[:, 0].min()))
    r_max = int(np.ceil(uv_cond_pts[:, 0].max()))
    c_min = int(np.floor(uv_cond_pts[:, 1].min()))
    c_max = int(np.ceil(uv_cond_pts[:, 1].max()))

    def _mask_span(cond_span):
        cond_span = int(max(1, cond_span))
        # cond_pct is conditioning fraction of the combined (cond + extrap) span.
        # Rearranging gives extrap span to query beyond the current boundary.
        if cond_pct <= 0:
            return cond_span
        total_span = max(cond_span + 1, int(round(cond_span / float(cond_pct))))
        return max(1, total_span - cond_span)

    if direction["axis"] == "col":
        rows = np.arange(r_min, r_max + 1, dtype=np.int64)
        mask_w = _mask_span(c_max - c_min + 1)
        if direction["growth_sign"] > 0:
            cols = np.arange(c_max + 1, c_max + mask_w + 1, dtype=np.int64)
        else:
            cols = np.arange(c_min - mask_w, c_min, dtype=np.int64)
        return np.stack(np.meshgrid(rows, cols, indexing='ij'), axis=-1)

    cols = np.arange(c_min, c_max + 1, dtype=np.int64)
    mask_h = _mask_span(r_max - r_min + 1)
    if direction["growth_sign"] > 0:
        rows = np.arange(r_max + 1, r_max + mask_h + 1, dtype=np.int64)
    else:
        rows = np.arange(r_min - mask_h, r_min, dtype=np.int64)
    return np.stack(np.meshgrid(rows, cols, indexing='ij'), axis=-1)


def _build_edge_input_mask(cond_valid, cond_direction, edge_input_rowscols):
    cond_valid = np.asarray(cond_valid, dtype=bool)
    if cond_valid.ndim != 2:
        raise ValueError(f"cond_valid must be 2D, got shape {cond_valid.shape}")

    n_axis = int(edge_input_rowscols)
    if n_axis < 1:
        raise ValueError("edge_input_rowscols must be >= 1")

    spec = _get_direction_spec(cond_direction)
    edge_mask = np.zeros_like(cond_valid, dtype=bool)

    if spec["axis"] == "col":
        valid_axis = np.any(cond_valid, axis=0)
        axis_indices = np.where(valid_axis)[0]
        if axis_indices.size == 0:
            return edge_mask
        keep = min(n_axis, axis_indices.size)
        keep_axis = axis_indices[-keep:] if spec["edge_idx"] == -1 else axis_indices[:keep]
        edge_mask[:, keep_axis] = True
    else:
        valid_axis = np.any(cond_valid, axis=1)
        axis_indices = np.where(valid_axis)[0]
        if axis_indices.size == 0:
            return edge_mask
        keep = min(n_axis, axis_indices.size)
        keep_axis = axis_indices[-keep:] if spec["edge_idx"] == -1 else axis_indices[:keep]
        edge_mask[keep_axis, :] = True

    return cond_valid & edge_mask


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
    edge_input_mask = _build_edge_input_mask(cond_valid_base, cond_direction, edge_input_rowscols)
    if not edge_input_mask.any():
        return None

    # Build query span from the edge-input conditioning band, not the full grown
    # surface, so iterative runs do not blow up one-shot query allocations.
    uv_query_seed = uv_cond[edge_input_mask]
    uv_query = _build_uv_query_from_cond_points(uv_query_seed, grow_direction, cond_pct)
    if uv_query.size == 0:
        return None

    uv_edge = uv_cond[edge_input_mask]
    zyx_edge = cond_zyxs[edge_input_mask]

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

    extrap_result = compute_extrapolation_infer(
        uv_cond=uv_edge,
        zyx_cond=zyx_edge,
        uv_query=uv_query,
        min_corner=min_corner_arr,
        crop_size=crop_size_use,
        method=method,
        cond_direction=cond_direction,
        degrade_prob=degrade_prob,
        degrade_curvature_range=degrade_curvature_range,
        degrade_gradient_range=degrade_gradient_range,
        skip_bounds_check=skip_bounds_check,
        **method_kwargs,
    )
    if extrap_result is None:
        extrap_local = np.zeros((0, 3), dtype=np.float32)
        extrap_surface = None
    else:
        extrap_local = np.asarray(extrap_result["extrap_coords_local"], dtype=np.float32)
        extrap_surface = extrap_result["extrap_surface"]

    if extrap_local.size == 0:
        extrap_world = np.zeros((0, 3), dtype=np.float32)
    else:
        extrap_world = extrap_local + min_corner_arr[None, :].astype(np.float32, copy=False)

    return {
        "cond_direction": cond_direction,
        "edge_input_mask": edge_input_mask,
        "edge_uv": uv_edge.astype(np.float64, copy=False),
        "edge_zyx": zyx_edge.astype(np.float32, copy=False),
        "uv_query": uv_query,
        "uv_query_flat": uv_query.reshape(-1, 2),
        "min_corner": min_corner_arr,
        "extrap_coords_local": extrap_local,
        "extrap_coords_world": extrap_world,
        "extrap_surface": extrap_surface,
    }


def build_bbox_crop_data(args, bboxes, cond_zyxs, cond_valid, uv_cond, grow_direction, crop_size, tgt_segment,
                         volume_scale, extrapolation_settings):
    cond_direction, _ = _get_growth_context(grow_direction)
    volume_for_crops = _resolve_segment_volume(tgt_segment, volume_scale=volume_scale)
    crop_size_extrap = tuple(int(v) for v in crop_size)
    if cond_valid is not None and np.asarray(cond_valid).shape == cond_zyxs.shape[:2]:
        cond_valid_base = np.asarray(cond_valid, dtype=bool)
    else:
        cond_valid_base = _valid_surface_mask(cond_zyxs)

    bbox_crops = []
    for bbox_idx, bbox in enumerate(bboxes):
        min_corner, bbox_bounds_array = _bbox_to_min_corner_and_bounds_array(bbox)
        vol_crop = _crop_volume_from_min_corner(volume_for_crops, min_corner, crop_size)
        vol_crop = normalize_zscore(vol_crop)

        extrap_local = np.zeros((0, 3), dtype=np.float32)
        extrap_uv = None

        cond_grid_local = cond_zyxs.astype(np.float64, copy=False) - min_corner[None, None, :]
        # Copy so per-bbox masking does not mutate the shared conditioning mask.
        cond_grid_valid = cond_valid_base.copy()
        cond_grid_valid &= _grid_in_bounds_mask(cond_grid_local, crop_size)
        cond_vox = voxelize_surface_grid_masked(cond_grid_local, crop_size, cond_grid_valid)

        extrap_vox = None
        if cond_grid_valid.any():
            uv_for_extrap = uv_cond[cond_grid_valid]
            zyx_for_extrap = cond_zyxs[cond_grid_valid]
            uv_query = _build_uv_query_from_cond_points(uv_for_extrap, grow_direction, args.cond_pct)

            if uv_query.size > 0:
                extrap_result = compute_extrapolation_infer(
                    uv_cond=uv_for_extrap,
                    zyx_cond=zyx_for_extrap,
                    uv_query=uv_query,
                    min_corner=min_corner,
                    crop_size=crop_size_extrap,
                    method=extrapolation_settings["method"],
                    cond_direction=cond_direction,
                    degrade_prob=extrapolation_settings["degrade_prob"],
                    degrade_curvature_range=extrapolation_settings["degrade_curvature_range"],
                    degrade_gradient_range=extrapolation_settings["degrade_gradient_range"],
                    skip_bounds_check=True,
                    **extrapolation_settings["method_kwargs"],
                )
                if extrap_result is not None:
                    extrap_local_full = np.asarray(extrap_result["extrap_coords_local"], dtype=np.float64)
                    uv_query_flat = uv_query.reshape(-1, 2)
                    extrap_local = extrap_local_full.astype(np.float32, copy=False)
                    extrap_uv = uv_query_flat

                    extrap_grid_local = extrap_local_full.reshape(uv_query.shape[:2] + (3,))
                    if np.isfinite(extrap_grid_local).all():
                        extrap_vox = voxelize_surface_grid(extrap_grid_local, crop_size_extrap)
                    else:
                        extrap_grid_valid = np.isfinite(extrap_grid_local).all(axis=-1)
                        if extrap_grid_valid.any():
                            extrap_vox = voxelize_surface_grid_masked(
                                extrap_grid_local,
                                crop_size_extrap,
                                extrap_grid_valid,
                            )

        if extrap_vox is None and len(extrap_local) > 0:
            extrap_vox = _points_to_voxels(extrap_local, crop_size)

        bbox_crops.append({
            "bbox": bbox,
            "min_corner": min_corner,
            "volume": vol_crop,
            "extrap_pts_local": extrap_local,
            "extrap_uv": extrap_uv,
            "cond_vox": cond_vox,
            "extrap_vox": extrap_vox,
        })

        out_dir = Path(args.bbox_crops_out_dir) / f"bbox_{bbox_idx:04d}"
        _save_crop_tiff(out_dir, "bbox_coords.tif", bbox_bounds_array)
        _save_crop_tiff(out_dir, "volume.tif", vol_crop)
        _save_crop_tiff(out_dir, "cond.tif", cond_vox)
        if extrap_vox is not None:
            _save_crop_tiff(out_dir, "extrap.tif", extrap_vox)

    return bbox_crops

def _get_displacement_result(model, model_inputs, amp_enabled, amp_dtype):
    
    with torch.no_grad():
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                output = model(model_inputs)
        else:
            output = model(model_inputs)
    disp = output.get("displacement", None)

    return disp


def _predict_displacement(args, model_state, model_inputs, use_tta=None):
    model = model_state["model"]
    amp_enabled = model_state["amp_enabled"]
    amp_dtype = model_state["amp_dtype"]
    if use_tta is None:
        use_tta = bool(getattr(args, "tta", True))

    if use_tta:
        return run_model_tta(
            model,
            model_inputs,
            amp_enabled,
            amp_dtype,
            get_displacement_result=_get_displacement_result,
            merge_method=getattr(args, "tta_merge_method", "vector_geomedian"),
            outlier_drop_thresh=getattr(args, "tta_outlier_drop_thresh", 1.25),
            outlier_drop_min_keep=getattr(args, "tta_outlier_drop_min_keep", 4),
        )

    return _get_displacement_result(model, model_inputs, amp_enabled, amp_dtype)


def _run_refine_on_crop(args, crop, crop_size, model_state):
    refine_extra_steps = int(args.refine) if args.refine is not None else 0
    refine_parts = refine_extra_steps + 1
    refine_fraction = 1.0 / float(refine_parts)
    expected_in_channels = model_state["expected_in_channels"]

    current_coords = crop.get("extrap_pts_local", None)
    if current_coords is None or len(current_coords) == 0:
        return np.zeros((0, 3), dtype=np.float32), None

    current_coords = np.asarray(current_coords, dtype=np.float32)
    current_uv = crop.get("extrap_uv", None)
    if current_uv is not None:
        current_uv = np.asarray(current_uv, dtype=np.float64)
        if len(current_uv) != current_coords.shape[0]:
            current_uv = None

    cond_vox = crop["cond_vox"]
    other_wraps_vox = None
    if expected_in_channels > 3:
        other_wraps_vox = np.zeros(crop_size, dtype=np.float32)

    initial_extrap_vox = crop.get("extrap_vox", None)
    if initial_extrap_vox is None:
        initial_extrap_vox = _points_to_voxels(current_coords, crop_size)

    current_coords_t = torch.from_numpy(current_coords).float().to(args.device)
    n_forward = 0

    for refine_idx in range(refine_parts):
        if current_coords_t.shape[0] == 0:
            break

        if refine_idx == 0:
            extrap_vox = initial_extrap_vox
        else:
            current_coords_np = current_coords_t.detach().cpu().numpy()
            extrap_vox = _points_to_voxels(current_coords_np, crop_size)

        inputs = _build_model_inputs(
            crop["volume"], cond_vox, extrap_vox, other_wraps_vox=other_wraps_vox
        ).to(args.device)

        disp_single = _predict_displacement(args, model_state, inputs)

        n_forward += 1
        # Apply a fixed fraction of each stage's full displacement prediction.
        disp_sampled = _sample_displacement_field(disp_single, current_coords_t)
        next_coords_t = current_coords_t + disp_sampled * refine_fraction
        finite_mask = torch.isfinite(next_coords_t).all(dim=1)
        if not bool(finite_mask.all()):
            next_coords_t = torch.where(finite_mask[:, None], next_coords_t, current_coords_t)

        current_coords_t = next_coords_t

    if n_forward == 0:
        return np.zeros((0, 3), dtype=np.float32), None if current_uv is None else current_uv[:0]

    if current_coords_t is None or current_coords_t.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), None if current_uv is None else current_uv[:0]
    return current_coords_t.detach().cpu().numpy().astype(np.float32, copy=False), current_uv


def _finalize_crop_prediction(args, bbox_idx, crop, pred_local, pred_uv, crop_size):
    if torch.is_tensor(pred_local):
        pred_local_np = pred_local.detach().cpu().numpy()
    else:
        pred_local_np = np.asarray(pred_local)

    pred_world = pred_local_np + crop["min_corner"][None, :]
    pred_finite = np.isfinite(pred_world).all(axis=1)
    if not pred_finite.any():
        pred_world = np.zeros((0, 3), dtype=np.float32)
    elif not pred_finite.all():
        pred_world = pred_world[pred_finite]

    pred_sample = None
    if pred_uv is not None:
        pred_uv = np.asarray(pred_uv)
        if len(pred_uv) == pred_finite.shape[0]:
            pred_uv_keep = pred_uv[pred_finite]
            if pred_uv_keep.shape[0] == pred_world.shape[0] and pred_world.shape[0] > 0:
                pred_sample = (pred_uv_keep, pred_world)

    out_dir = Path(args.bbox_crops_out_dir) / f"bbox_{bbox_idx:04d}"
    pred_local_finite = pred_local_np[np.isfinite(pred_local_np).all(axis=1)]
    _save_crop_tiff(out_dir, "pred.tif", _points_to_voxels(pred_local_finite, crop_size))
    return pred_sample


def run_inference(args, bbox_crops, crop_size, model_state):
    expected_in_channels = model_state["expected_in_channels"]
    batch_size = args.batch_size
    refine_mode = args.refine is not None

    # Collect crops that have valid extrapolation points.
    valid_items = []
    for bbox_idx, crop in enumerate(bbox_crops):
        extrap_local = crop.get("extrap_pts_local", None)
        if extrap_local is None or len(extrap_local) == 0:
            continue

        cond_vox = crop["cond_vox"]
        extrap_vox = crop["extrap_vox"]
        if extrap_vox is None:
            extrap_vox = np.zeros(crop_size, dtype=np.float32)

        other_wraps_vox = None
        if expected_in_channels > 3:
            other_wraps_vox = np.zeros(crop_size, dtype=np.float32)

        inputs = _build_model_inputs(
            crop["volume"], cond_vox, extrap_vox, other_wraps_vox=other_wraps_vox
        )
        valid_items.append((bbox_idx, crop, inputs, extrap_local))

    pred_samples = []
    use_tta = bool(getattr(args, "tta", True))

    if refine_mode:
        desc = "inference (refine)"
        if use_tta:
            desc = "inference (refine+TTA)"

        for bbox_idx, crop, _, _ in tqdm(valid_items, desc=desc):
            pred_local, pred_uv = _run_refine_on_crop(args, crop, crop_size, model_state)
            if pred_local is None or len(pred_local) == 0:
                continue

            pred_sample = _finalize_crop_prediction(
                args, bbox_idx, crop, pred_local, pred_uv, crop_size
            )
            if pred_sample is not None:
                pred_samples.append(pred_sample)
    else:
        desc = "inference (TTA)" if use_tta else "inference"
        n_batches = (len(valid_items) + batch_size - 1) // batch_size
        for batch_start in tqdm(range(0, len(valid_items), batch_size), total=n_batches, desc=desc):
            batch = valid_items[batch_start:batch_start + batch_size]
            batch_inputs = torch.cat([item[2] for item in batch], dim=0).to(args.device)
            disp_pred = _predict_displacement(
                args, model_state, batch_inputs, use_tta=use_tta
            )

            for i, (bbox_idx, crop, _, extrap_local) in enumerate(batch):
                disp_single = disp_pred[i:i+1]  # [1, 3, D, H, W]

                extrap_coords = torch.from_numpy(extrap_local).float().to(args.device)
                extrap_uv = crop.get("extrap_uv", None)

                disp_sampled = _sample_displacement_field(disp_single, extrap_coords)
                pred_local = extrap_coords + disp_sampled
                pred_sample = _finalize_crop_prediction(
                    args, bbox_idx, crop, pred_local, extrap_uv, crop_size
                )
                if pred_sample is not None:
                    pred_samples.append(pred_sample)

    return pred_samples


def _build_uv_grid(uv_offset, shape_hw):
    r0, c0 = uv_offset
    h, w = shape_hw
    rows = np.arange(r0, r0 + h, dtype=np.int64)
    cols = np.arange(c0, c0 + w, dtype=np.int64)
    return np.stack(np.meshgrid(rows, cols, indexing='ij'), axis=-1)


def prepare_next_iteration_cond(
    full_grid_zyxs, full_valid, full_uv_offset,
    pred_grid_zyxs, pred_grid_valid, pred_uv_offset,
    grow_direction, keep_voxels=None,
):
    """Append new prediction band outside current boundary, returning next full grid state."""
    _, direction = _get_growth_context(grow_direction)
    keep_voxels = None if keep_voxels is None else int(keep_voxels)
    if keep_voxels is not None and keep_voxels < 1:
        keep_voxels = 1

    if pred_grid_zyxs.size == 0 or not pred_grid_valid.any():
        return full_grid_zyxs, full_valid, full_uv_offset, [], 0

    fh, fw = full_grid_zyxs.shape[:2]
    full_r0, full_c0 = full_uv_offset
    full_r1 = full_r0 + fh - 1
    full_c1 = full_c0 + fw - 1

    pred_r0, pred_c0 = pred_uv_offset
    pred_rows, pred_cols = np.where(pred_grid_valid)
    if pred_rows.size == 0:
        return full_grid_zyxs, full_valid, full_uv_offset, [], 0

    pred_rows_abs = pred_rows.astype(np.int64) + pred_r0
    pred_cols_abs = pred_cols.astype(np.int64) + pred_c0
    pred_pts = pred_grid_zyxs[pred_rows, pred_cols].astype(np.float32)

    # Keep only predictions outside the current boundary, in the growth direction.
    if direction["axis"] == "row":
        axis_vals = pred_rows_abs
        boundary = full_r1 if direction["growth_sign"] > 0 else full_r0
    else:
        axis_vals = pred_cols_abs
        boundary = full_c1 if direction["growth_sign"] > 0 else full_c0

    if direction["growth_sign"] > 0:
        growth_mask = axis_vals > boundary
    else:
        growth_mask = axis_vals < boundary
    ordered_axis_vals = np.sort(np.unique(axis_vals[growth_mask]))
    if direction["growth_sign"] < 0:
        ordered_axis_vals = ordered_axis_vals[::-1]

    if ordered_axis_vals.size == 0:
        return full_grid_zyxs, full_valid, full_uv_offset, [], 0

    # keep_voxels limits how many newly grown rows/cols are admitted this round.
    n_keep = ordered_axis_vals.size if keep_voxels is None else min(ordered_axis_vals.size, keep_voxels)
    kept_axis_vals = ordered_axis_vals[:n_keep]

    if direction["axis"] == "row":
        keep_mask = growth_mask & np.isin(pred_rows_abs, kept_axis_vals)
    else:
        keep_mask = growth_mask & np.isin(pred_cols_abs, kept_axis_vals)

    if not keep_mask.any():
        return full_grid_zyxs, full_valid, full_uv_offset, [], 0

    kept_rows_abs = pred_rows_abs[keep_mask]
    kept_cols_abs = pred_cols_abs[keep_mask]
    kept_pts = pred_pts[keep_mask]

    new_r0 = min(full_r0, int(kept_rows_abs.min()))
    new_c0 = min(full_c0, int(kept_cols_abs.min()))
    new_r1 = max(full_r1, int(kept_rows_abs.max()))
    new_c1 = max(full_c1, int(kept_cols_abs.max()))

    # Rebuild a contiguous UV canvas and return its new top-left offset.
    nh = new_r1 - new_r0 + 1
    nw = new_c1 - new_c0 + 1
    merged_cond = np.full((nh, nw, 3), -1.0, dtype=np.float32)
    merged_valid = np.zeros((nh, nw), dtype=bool)

    # Copy current full grid.
    full_rows, full_cols = np.where(full_valid)
    if full_rows.size > 0:
        full_rows_abs = full_rows.astype(np.int64) + full_r0
        full_cols_abs = full_cols.astype(np.int64) + full_c0
        mr = full_rows_abs - new_r0
        mc = full_cols_abs - new_c0
        merged_cond[mr, mc] = full_grid_zyxs[full_rows, full_cols]
        merged_valid[mr, mc] = True

    # Add newly selected prediction band.
    mr = kept_rows_abs - new_r0
    mc = kept_cols_abs - new_c0
    merged_cond[mr, mc] = kept_pts
    merged_valid[mr, mc] = True

    kept_uv = np.stack([kept_rows_abs, kept_cols_abs], axis=-1).astype(np.float64)
    kept_pred_samples = [(kept_uv, kept_pts)]
    return merged_cond, merged_valid, (new_r0, new_c0), kept_pred_samples, int(n_keep)


def _load_runtime_state(args):
    has_checkpoint = args.checkpoint_path is not None
    run_model_inference = has_checkpoint and not args.skip_inference

    model_state = None
    model_config = None
    tifxyz_uuid = None
    checkpoint_path = args.checkpoint_path

    if has_checkpoint:
        if run_model_inference:
            model_state = load_model(args)
            model_config = model_state["model_config"]
            checkpoint_path = model_state["checkpoint_path"]
            tifxyz_uuid = model_state["tifxyz_uuid"]
        else:
            model_config, checkpoint_path = load_checkpoint_config(args.checkpoint_path)

    file_config = _load_optional_json(args.config_path)
    runtime_config = dict(model_config) if model_config is not None else {}
    runtime_config.update(file_config)
    extrapolation_settings = _resolve_extrapolation_settings(args, runtime_config)

    return {
        "run_model_inference": run_model_inference,
        "model_state": model_state,
        "model_config": model_config,
        "checkpoint_path": checkpoint_path,
        "tifxyz_uuid": tifxyz_uuid,
        "extrapolation_settings": extrapolation_settings,
    }


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


def _run_growth_iterations(
    args,
    growth_iterations,
    crop_size,
    tgt_segment,
    grow_direction,
    cond_direction,
    growth_spec,
    current_grid,
    current_valid,
    current_uv_offset,
    run_model_inference,
    model_state,
    extrapolation_settings,
):
    if growth_iterations > 1 and args.iter_keep_voxels is None:
        print(
            "iter-keep-voxels not set; each iteration will keep all newly predicted "
            "rows/cols beyond the current boundary."
        )

    all_pred_samples = []

    # Per iteration: build UVs -> choose edge bboxes -> infer displaced points ->
    # aggregate in UV space -> keep outward band -> update conditioning state.
    for iteration in range(growth_iterations):
        print(f"[iteration {iteration + 1}/{growth_iterations}]")
        cond_zyxs = current_grid
        valid = current_valid

        # create global (row, col) coordinates for each point in the current window, offset during expansion to keep coordinates correctly aligned
        uv_cond = _build_uv_grid(current_uv_offset, cond_zyxs.shape[:2])

        # at the edge indicated by the growth direction, beginning at the center point, walk the edge in one direction and greedily add points until 
        # we have enough to create a crop_size bbox, with overlap. then, walk the edge in the opposite direction and do the same, until each point in the edge
        # has been assigned to at least one bbox. bboxes are centered on the edge points , but do not perfectly split the ratio of cond:extrap because
        # the bboxes are axis-aligned and the surface may curve in 3d despite the obvious 2d straightness
        bboxes, _ = get_cond_edge_bboxes(
            cond_zyxs, cond_direction, crop_size,
            overlap_frac=args.bbox_overlap_frac,
        )

        pred_samples = []
        # for each bbox: 
        # - crop the volume, and zscore normalize it
        # - convert the current conditioning points into crop-local coords, and voxelize them into a binary mask
        # - using the conditioning coordinates that fall within this crop, extrapolate new points beyond the current edge (extrapolation also returns new row, col indices)
        # - voxelize the extrapolated points into a binary mask
        # - pack it all into a dict for the inference call
        bbox_crops = build_bbox_crop_data(
            args, bboxes, cond_zyxs, valid, uv_cond, grow_direction, crop_size, tgt_segment,
            args.volume_scale, extrapolation_settings,
        )

        # we pass the bbox crop data to the model , and optionally apply TTA or fractional refinement
        # the model predicts a dense 3d displacement field which says at each voxel the (dz, dy, dx) displacement to reach the intended surface
        # we sample the predicted displacement field at the extrapolated points to get the correct z, y, x coordinates 
        # and return the predicted (row, col, z, y, x) for each crops extrapolated grid
        if run_model_inference and model_state is not None:
            pred_samples = run_inference(args, bbox_crops, crop_size, model_state)

        if not pred_samples:
            print("No predicted samples this iteration; stopping iterative growth.")
            break
        
        # aggregate predicted samples from all bboxes into one dense UV-indexed grid -- each sample is (uv=row,col, pts=z,y,x) 
        # we build a canvas covering all uvs and place points into uv cells, overlaps are averaged together
        # at this point we are aggregating predictions only, we have not yet merged into the full grid
        pred_grid, pred_valid, pred_offset = _aggregate_pred_samples_to_uv_grid(
            pred_samples,
        )

        # now we combine the prediction grid with the current full grid. 
        # we keep only predicted points that lie beyond the current boundary in the growth direction
        # we optionally choose to keep only the first rows or cols , determined by the --iter-keep-voxels argument
        # then we create a new (row, col, z, y, x) grid large enough to hold both the current grid and the kept prediction grid
        # we copy the current valid grid into the new grid, and copy the prediction grid (or the band determined by the --iter-keep-voxels argument / keep_mask) 
        # according to its uv offset

        merged_cond, merged_valid, merged_uv_offset, kept, n_kept_axis = prepare_next_iteration_cond(
            current_grid, current_valid, current_uv_offset,
            pred_grid, pred_valid, pred_offset,
            grow_direction, args.iter_keep_voxels,
        )
        if not kept:
            print("No new rows/cols beyond current boundary; stopping iterative growth.")
            break

        all_pred_samples.extend(kept)
        current_grid = merged_cond
        current_valid = merged_valid
        current_uv_offset = merged_uv_offset
        axis_label = "rows" if growth_spec["axis"] == "row" else "cols"
        print(f"  kept {n_kept_axis} new {axis_label}")

    return all_pred_samples


def main():
    args = parse_args()

    refine_mode = args.refine is not None
    growth_iterations = args.iterations
    if refine_mode:
        print(
            f"--refine={args.refine} enabled; running {int(args.refine) + 1} "
            "fractional refinement stages inside each outer iteration."
        )

    crop_size = tuple(args.crop_size)
    runtime_state = _load_runtime_state(args)
    run_model_inference = runtime_state["run_model_inference"]
    model_state = runtime_state["model_state"]
    model_config = runtime_state["model_config"]
    checkpoint_path = runtime_state["checkpoint_path"]
    tifxyz_uuid = runtime_state["tifxyz_uuid"]
    extrapolation_settings = runtime_state["extrapolation_settings"]

    tifxyz_step_size, tifxyz_voxel_size_um = resolve_tifxyz_params(
        args, model_config, args.volume_scale
    )

    volume = zarr.open_group(args.volume_path, mode='r')
    tgt_segment, stored_zyxs, valid_s, grow_direction, h_s, w_s = setup_segment(args, volume)
    cond_direction, growth_spec = _get_growth_context(grow_direction)

    r0_s, r1_s, c0_s, c1_s = compute_window_and_split(
        args, stored_zyxs, valid_s, grow_direction, h_s, w_s, crop_size
    )

    full_bounds = _stored_to_full_bounds(tgt_segment, (r0_s, r1_s, c0_s, c1_s))
    current_grid, current_valid, current_uv_offset = _initialize_window_state(
        tgt_segment, full_bounds
    )
    pred_samples = _run_growth_iterations(
        args,
        growth_iterations,
        crop_size,
        tgt_segment,
        grow_direction,
        cond_direction,
        growth_spec,
        current_grid,
        current_valid,
        current_uv_offset,
        run_model_inference,
        model_state,
        extrapolation_settings,
    )

    if tifxyz_uuid is not None and pred_samples:
        save_tifxyz_output(
            args, tgt_segment, pred_samples, tifxyz_uuid, tifxyz_step_size,
            tifxyz_voxel_size_um, checkpoint_path, cond_direction, grow_direction,
            args.volume_scale,
        )


if __name__ == '__main__':
    main()
