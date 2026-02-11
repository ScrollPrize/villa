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
from vesuvius.neural_tracing.models import load_checkpoint, resolve_checkpoint_path
from vesuvius.neural_tracing.tifxyz import save_tifxyz
from tqdm import tqdm

VALID_DIRECTIONS = ["left", "right", "down", "up"]

def _clamp_window(start, size, min_val, max_val):
    size = int(size)
    start = int(start)
    if size <= 0:
        return min_val, min_val
    start = max(min_val, min(start, max_val - size + 1))
    end = start + size - 1
    return start, end

def _edge_index_from_valid(valid, cond_direction):
    valid_rows = np.any(valid, axis=1)
    valid_cols = np.any(valid, axis=0)
    if not valid_rows.any() or not valid_cols.any():
        return None, None
    r_idx = np.where(valid_rows)[0]
    c_idx = np.where(valid_cols)[0]
    if cond_direction == "left":   # cond on left, extrap right -> edge is rightmost
        return None, int(c_idx[-1])
    if cond_direction == "right":  # cond on right, extrap left -> edge is leftmost
        return None, int(c_idx[0])
    if cond_direction == "up":     # cond on top, extrap down -> edge is bottommost
        return int(r_idx[-1]), None
    if cond_direction == "down":   # cond on bottom, extrap up -> edge is topmost
        return int(r_idx[0]), None
    return None, None

def _place_window_on_edge(edge_idx, window_size, cond_size, cond_direction, max_idx, clamp=True):
    # Keep window size fixed; place edge at split determined by cond_size.
    if cond_direction in ("left", "up"):
        start = edge_idx - (cond_size - 1)
    elif cond_direction in ("right", "down"):
        mask_size = window_size - cond_size
        start = edge_idx - mask_size
    else:
        start = edge_idx - (window_size // 2)
    if clamp:
        return _clamp_window(start, window_size, 0, max_idx)
    end = int(start) + int(window_size) - 1
    return int(start), int(end)

def _get_cond_edge(cond_zyxs, cond_direction, outer_edge=False):
    if cond_direction == "left":
        return cond_zyxs[:, 0, :] if outer_edge else cond_zyxs[:, -1, :]
    if cond_direction == "right":
        return cond_zyxs[:, -1, :] if outer_edge else cond_zyxs[:, 0, :]
    if cond_direction == "up":
        return cond_zyxs[0, :, :] if outer_edge else cond_zyxs[-1, :, :]
    if cond_direction == "down":
        return cond_zyxs[-1, :, :] if outer_edge else cond_zyxs[0, :, :]
    raise ValueError(f"Unknown cond_direction '{cond_direction}'")

def split_grid(zyxs, uv_offset, cond_direction, r_split, c_split):
    h, w = zyxs.shape[:2]
    r_split = int(np.clip(r_split, 0, h))
    c_split = int(np.clip(c_split, 0, w))

    uv_full = np.stack(np.meshgrid(
        np.arange(h) + uv_offset[0],
        np.arange(w) + uv_offset[1],
        indexing='ij'
    ), axis=-1)

    if cond_direction in ("left", "right"):
        a_zyxs = zyxs[:, :c_split]
        b_zyxs = zyxs[:, c_split:]
        a_uv = uv_full[:, :c_split]
        b_uv = uv_full[:, c_split:]
    else:
        a_zyxs = zyxs[:r_split, :]
        b_zyxs = zyxs[r_split:, :]
        a_uv = uv_full[:r_split, :]
        b_uv = uv_full[r_split:, :]

    if cond_direction in ("left", "up"):
        return a_zyxs, b_zyxs, a_uv, b_uv
    else:
        return b_zyxs, a_zyxs, b_uv, a_uv

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

def get_cond_edge_bboxes(cond_zyxs, cond_direction, crop_size, outer_edge=False, overlap_frac=0.15):
    edge = _get_cond_edge(cond_zyxs, cond_direction, outer_edge=outer_edge)

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

    def _chunk_ordered_indices(ordered_indices):
        chunks = []
        if len(ordered_indices) == 0:
            return chunks
        start = 0
        while start < len(ordered_indices):
            end = start + 1
            while end < len(ordered_indices):
                candidate = ordered_indices[start:end + 1]
                pts = edge[candidate]
                spans = pts.max(axis=0) - pts.min(axis=0)
                if (
                    spans[0] <= (crop_size_arr[0] - 1) and
                    spans[1] <= (crop_size_arr[1] - 1) and
                    spans[2] <= (crop_size_arr[2] - 1)
                ):
                    end += 1
                    continue
                break
            chunk = ordered_indices[start:end]
            chunks.append(chunk)
            chunk_len = len(chunk)
            overlap_count = int(round(chunk_len * overlap_frac))
            step = max(1, chunk_len - overlap_count)
            start += step
        return chunks

    center_idx = n_edge // 2
    # Center-out and two-sided: walk one side first (toward smaller indices), then the other side.
    first_side = np.arange(center_idx, -1, -1, dtype=np.int64)
    second_side = np.arange(center_idx + 1, n_edge, dtype=np.int64)

    bboxes = []
    for chunk in _chunk_ordered_indices(first_side):
        pts = edge[chunk]
        zc = (pts[:, 0].min() + pts[:, 0].max()) / 2
        yc = (pts[:, 1].min() + pts[:, 1].max()) / 2
        xc = (pts[:, 2].min() + pts[:, 2].max()) / 2
        bboxes.append(_bbox_from_center((zc, yc, xc), crop_size))
    for chunk in _chunk_ordered_indices(second_side):
        pts = edge[chunk]
        zc = (pts[:, 0].min() + pts[:, 0].max()) / 2
        yc = (pts[:, 1].min() + pts[:, 1].max()) / 2
        xc = (pts[:, 2].min() + pts[:, 2].max()) / 2
        bboxes.append(_bbox_from_center((zc, yc, xc), crop_size))

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

def _bbox_to_min_corner(bbox):
    z_min, _, y_min, _, x_min, _ = bbox
    return np.floor([z_min, y_min, x_min]).astype(np.int64)


def _bbox_to_bounds_array(bbox):
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    return np.asarray(
        [
            [z_min, y_min, x_min],
            [z_max, y_max, x_max],
        ],
        dtype=np.int32,
    )


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

def _filter_points_in_bbox_mask(points, bbox):
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    z, y, x = points[:, 0], points[:, 1], points[:, 2]
    return (
        (z >= z_min) & (z <= z_max) &
        (y >= y_min) & (y <= y_max) &
        (x >= x_min) & (x <= x_max)
    )

def _points_world_to_local(points, min_corner, crop_size):
    if points is None or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    local = points - min_corner[None, :]
    return local[_in_bounds_mask(local, crop_size)].astype(np.float32)

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
    # Ensure grid dtype matches pred_field for AMP compatibility.
    coords_norm = coords_local.to(dtype=pred_field.dtype).clone()
    d_denom = max(D - 1, 1)
    h_denom = max(H - 1, 1)
    w_denom = max(W - 1, 1)
    coords_norm[:, 0] = 2 * coords_norm[:, 0] / d_denom - 1
    coords_norm[:, 1] = 2 * coords_norm[:, 1] / h_denom - 1
    coords_norm[:, 2] = 2 * coords_norm[:, 2] / w_denom - 1

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

def _min_corner_from_edge(edge_pts, cond_bounds, crop_size, cond_pct, cond_direction):
    z_size, y_size, x_size = crop_size
    z_min, z_max, y_min, y_max, x_min, x_max = cond_bounds

    def _axis_start(axis_min, axis_max, size):
        extent = axis_max - axis_min
        if extent >= (size - 1):
            return axis_min
        center = (axis_min + axis_max) / 2
        start = center - (size - 1) / 2
        start = min(start, axis_min)
        start = max(start, axis_max - (size - 1))
        return start

    z0 = _axis_start(z_min, z_max, z_size)
    y0 = _axis_start(y_min, y_max, y_size)
    x0 = _axis_start(x_min, x_max, x_size)

    edge_center = np.median(edge_pts, axis=0)
    zc, yc, xc = edge_center

    if cond_direction in ["left", "right"]:
        cond_size = max(1, min(x_size - 1, int(round(x_size * cond_pct))))
        mask_size = x_size - cond_size
        if cond_direction == "left":
            x0 = xc - (cond_size - 1)
        else:
            x0 = xc - (mask_size - 1)
    else:
        cond_size = max(1, min(y_size - 1, int(round(y_size * cond_pct))))
        mask_size = y_size - cond_size
        if cond_direction == "up":
            y0 = yc - (cond_size - 1)
        else:
            y0 = yc - (mask_size - 1)

    return np.floor([z0, y0, x0]).astype(np.int64)

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

    z_extrap = zyx_extrapolated[:, 0]
    y_extrap = zyx_extrapolated[:, 1]
    x_extrap = zyx_extrapolated[:, 2]

    z_extrap_local = z_extrap - min_corner[0]
    y_extrap_local = y_extrap - min_corner[1]
    x_extrap_local = x_extrap - min_corner[2]

    if degrade_prob > 0.0 and cond_direction is not None:
        zyx_extrap_local_full = np.stack([z_extrap_local, y_extrap_local, x_extrap_local], axis=-1)
        uv_shape = uv_query.shape[:2]
        zyx_extrap_local_full, _ = apply_degradation(
            zyx_extrap_local_full,
            uv_shape,
            cond_direction,
            degrade_prob=degrade_prob,
            curvature_range=degrade_curvature_range,
            gradient_range=degrade_gradient_range,
        )
        z_extrap_local = zyx_extrap_local_full[:, 0]
        y_extrap_local = zyx_extrap_local_full[:, 1]
        x_extrap_local = zyx_extrap_local_full[:, 2]

    zyx_extrap_local_full = np.stack([z_extrap_local, y_extrap_local, x_extrap_local], axis=-1)
    extrap_coords_local = zyx_extrap_local_full
    extrap_surface = None
    if not skip_bounds_check:
        in_bounds = _in_bounds_mask(zyx_extrap_local_full, crop_size)
        if in_bounds.sum() == 0:
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
    parser.add_argument("--tta", action="store_true",
        help="Enable mirroring-based test-time augmentation (8 flip combos, averaged).")
    return parser.parse_args()


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
        rbf_downsample = int(cfg.get("rbf_downsample_factor", 2))
        edge_downsample_cfg = cfg.get("rbf_edge_downsample_factor", None)
        edge_downsample = rbf_downsample if edge_downsample_cfg is None else int(edge_downsample_cfg)
        method_kwargs["downsample_factor"] = edge_downsample if method == "rbf_edge_only" else rbf_downsample
        method_kwargs["rbf_max_points"] = cfg.get("rbf_max_points")

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
    _OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}
    cond_direction = _OPPOSITE[args.grow_direction]
    if cond_direction not in valid_dirs:
        raise RuntimeError(
            f"Requested grow_direction '{args.grow_direction}' (cond_direction='{cond_direction}') "
            f"not available for this segment. Valid options: {valid_dirs}"
        )

    return tgt_segment, stored_zyxs, valid_s, cond_direction, h_s, w_s


def compute_window_and_split(args, stored_zyxs, valid_s, cond_direction, h_s, w_s, crop_size):
    r_edge_s, c_edge_s = _edge_index_from_valid(valid_s, cond_direction)
    if r_edge_s is None and c_edge_s is None:
        raise RuntimeError("No valid edge found for segment.")

    if cond_direction in ["left", "right"]:
        cond_edge_strip = stored_zyxs[:, c_edge_s:c_edge_s + 1]
    else:
        cond_edge_strip = stored_zyxs[r_edge_s:r_edge_s + 1, :]

    bboxes, _ = get_cond_edge_bboxes(
        cond_edge_strip, cond_direction, crop_size, outer_edge=True,
        overlap_frac=args.bbox_overlap_frac,
    )

    r0_s, r1_s, c0_s, c1_s = get_window_bounds_from_bboxes(
        stored_zyxs, valid_s, bboxes, pad=args.window_pad
    )

    win_h = r1_s - r0_s + 1
    win_w = c1_s - c0_s + 1
    if win_h < 2 or win_w < 2:
        raise RuntimeError("Window too small after edge-based bounds.")

    outside_dir = cond_direction
    r_edge_outside, c_edge_outside = _edge_index_from_valid(valid_s, outside_dir)

    cond_h = win_h
    cond_w = win_w
    if cond_direction in ["left", "right"]:
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


def run_extrapolation(args, cond_zyxs, window_zyxs, valid, uv_cond, uv_mask, cond_direction, crop_size,
                      extrapolation_settings):
    edge_pts = _get_cond_edge(
        cond_zyxs, cond_direction, outer_edge=False
    ).reshape(-1, 3)
    edge_valid = ~(edge_pts == -1).all(axis=1)
    if edge_valid.any():
        edge_pts = edge_pts[edge_valid]
    else:
        edge_pts = cond_zyxs.reshape(-1, 3)
        edge_pts = edge_pts[~(edge_pts == -1).all(axis=1)]
        if edge_pts.size == 0:
            return None, None
    crop_size_extrap = tuple(int(v) for v in crop_size)
    if valid is not None and valid.any():
        cond_pts_bounds = window_zyxs[valid]
    else:
        cond_pts_bounds = window_zyxs.reshape(-1, 3)
    cond_bounds = (
        float(np.min(cond_pts_bounds[:, 0])),
        float(np.max(cond_pts_bounds[:, 0])),
        float(np.min(cond_pts_bounds[:, 1])),
        float(np.max(cond_pts_bounds[:, 1])),
        float(np.min(cond_pts_bounds[:, 2])),
        float(np.max(cond_pts_bounds[:, 2])),
    )
    zyx_min = _min_corner_from_edge(
        edge_pts, cond_bounds, crop_size_extrap, args.cond_pct, cond_direction
    )

    # Filter conditioning data to valid points only to avoid -1 sentinel
    # values poisoning gradient computation in linear_edge extrapolation.
    if valid is not None and valid.any():
        valid_flat = valid.ravel()
        uv_for_extrap = uv_cond.reshape(-1, 2)[valid_flat]
        zyx_for_extrap = cond_zyxs.reshape(-1, 3)[valid_flat]
    else:
        uv_for_extrap = uv_cond
        zyx_for_extrap = cond_zyxs

    extrap_result = compute_extrapolation_infer(
        uv_cond=uv_for_extrap,
        zyx_cond=zyx_for_extrap,
        uv_query=uv_mask,
        min_corner=zyx_min,
        crop_size=crop_size_extrap,
        method=extrapolation_settings["method"],
        cond_direction=cond_direction,
        degrade_prob=extrapolation_settings["degrade_prob"],
        degrade_curvature_range=extrapolation_settings["degrade_curvature_range"],
        degrade_gradient_range=extrapolation_settings["degrade_gradient_range"],
        skip_bounds_check=True,
        **extrapolation_settings["method_kwargs"],
    )

    return extrap_result, zyx_min


def _build_uv_query_from_cond_points(uv_cond_pts, cond_direction, cond_pct):
    if uv_cond_pts is None or len(uv_cond_pts) == 0:
        return np.zeros((0, 0, 2), dtype=np.float64)

    uv_cond_pts = np.asarray(uv_cond_pts)
    r_min = int(np.floor(uv_cond_pts[:, 0].min()))
    r_max = int(np.ceil(uv_cond_pts[:, 0].max()))
    c_min = int(np.floor(uv_cond_pts[:, 1].min()))
    c_max = int(np.ceil(uv_cond_pts[:, 1].max()))

    def _mask_span(cond_span):
        cond_span = int(max(1, cond_span))
        if cond_pct <= 0:
            return cond_span
        total_span = max(cond_span + 1, int(round(cond_span / float(cond_pct))))
        return max(1, total_span - cond_span)

    if cond_direction in ["left", "right"]:
        rows = np.arange(r_min, r_max + 1, dtype=np.int64)
        mask_w = _mask_span(c_max - c_min + 1)
        if cond_direction == "left":   # grow right
            cols = np.arange(c_max + 1, c_max + mask_w + 1, dtype=np.int64)
        else:                          # grow left
            cols = np.arange(c_min - mask_w, c_min, dtype=np.int64)
        return np.stack(np.meshgrid(rows, cols, indexing='ij'), axis=-1)

    cols = np.arange(c_min, c_max + 1, dtype=np.int64)
    mask_h = _mask_span(r_max - r_min + 1)
    if cond_direction == "up":         # grow down
        rows = np.arange(r_max + 1, r_max + mask_h + 1, dtype=np.int64)
    else:                              # grow up
        rows = np.arange(r_min - mask_h, r_min, dtype=np.int64)
    return np.stack(np.meshgrid(rows, cols, indexing='ij'), axis=-1)


def build_bbox_crop_data(args, bboxes, cond_zyxs, cond_valid, uv_cond, cond_direction, crop_size, tgt_segment,
                         volume_scale, extrapolation_settings):
    volume_for_crops = _resolve_segment_volume(tgt_segment, volume_scale=volume_scale)
    cond_pts_world = cond_zyxs.reshape(-1, 3)
    cond_valid_mask = ~(cond_pts_world == -1).all(axis=1)
    cond_pts_world = cond_pts_world[cond_valid_mask]
    crop_size_extrap = tuple(int(v) for v in crop_size)
    if cond_valid is not None and np.asarray(cond_valid).shape == cond_zyxs.shape[:2]:
        cond_valid_base = np.asarray(cond_valid, dtype=bool)
    else:
        cond_valid_base = _valid_surface_mask(cond_zyxs)

    bbox_crops = []
    for bbox_idx, bbox in enumerate(bboxes):
        min_corner = _bbox_to_min_corner(bbox)
        vol_crop = _crop_volume_from_min_corner(volume_for_crops, min_corner, crop_size)
        vol_crop = normalize_zscore(vol_crop)

        cond_world_in = cond_pts_world[_filter_points_in_bbox_mask(cond_pts_world, bbox)]
        cond_local = _points_world_to_local(cond_world_in, min_corner, crop_size)

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
            uv_query = _build_uv_query_from_cond_points(uv_for_extrap, cond_direction, args.cond_pct)

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
                    extrap_in_bounds = _in_bounds_mask(extrap_local_full, crop_size_extrap)

                    if extrap_in_bounds.any():
                        extrap_local = extrap_local_full[extrap_in_bounds].astype(np.float32)
                        extrap_uv = uv_query_flat[extrap_in_bounds]

                    extrap_grid_local = extrap_local_full.reshape(uv_query.shape[:2] + (3,))
                    extrap_grid_valid = _in_bounds_mask(
                        extrap_grid_local.reshape(-1, 3),
                        crop_size_extrap,
                    ).reshape(uv_query.shape[:2])
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
            "cond_pts_local": cond_local,
            "extrap_pts_local": extrap_local,
            "extrap_uv": extrap_uv,
            "cond_vox": cond_vox,
            "extrap_vox": extrap_vox,
        })

        out_dir = Path(args.bbox_crops_out_dir) / f"bbox_{bbox_idx:04d}"
        _save_crop_tiff(out_dir, "bbox_coords.tif", _bbox_to_bounds_array(bbox))
        _save_crop_tiff(out_dir, "volume.tif", vol_crop)
        _save_crop_tiff(out_dir, "cond.tif", cond_vox)
        if extrap_vox is not None:
            _save_crop_tiff(out_dir, "extrap.tif", extrap_vox)

    return bbox_crops


_TTA_FLIP_COMBOS = [
    [],
    [-1],
    [-2],
    [-3],
    [-1, -2],
    [-1, -3],
    [-2, -3],
    [-1, -2, -3],
]

# Mapping from flip dim to displacement channel that must be negated:
# dim -1 (W/X) -> channel 2, dim -2 (H/Y) -> channel 1, dim -3 (D/Z) -> channel 0
_FLIP_DIM_TO_CHANNEL = {-1: 2, -2: 1, -3: 0}


def _extract_displacement_output(output):
    if isinstance(output, dict):
        disp = output.get("displacement", None)
        if disp is None:
            raise RuntimeError("Model output missing 'displacement' head.")
        return disp
    return output


def _forward_model_displacement(model, model_inputs, amp_enabled, amp_dtype):
    with torch.no_grad():
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                output = model(model_inputs)
        else:
            output = model(model_inputs)
    return _extract_displacement_output(output)


def _run_model_tta(model, inputs, amp_enabled, amp_dtype):
    """Run mirroring-based TTA on a single sample, returning averaged displacement.

    Args:
        model: The model to run inference with.
        inputs: Input tensor of shape [1, C, D, H, W].
        amp_enabled: Whether to use automatic mixed precision.
        amp_dtype: The dtype for AMP.

    Returns:
        Averaged displacement tensor of shape [1, 3, D, H, W].
    """
    accum = None

    for flip_dims in _TTA_FLIP_COMBOS:
        # Flip input
        x = inputs
        for d in flip_dims:
            x = x.flip(d)

        # Forward pass
        disp = _forward_model_displacement(model, x, amp_enabled, amp_dtype)

        # Un-flip the displacement output
        for d in reversed(flip_dims):
            disp = disp.flip(d)

        # Negate displacement channels corresponding to flipped spatial axes
        for d in flip_dims:
            ch = _FLIP_DIM_TO_CHANNEL[d]
            disp[:, ch] = -disp[:, ch]

        if accum is None:
            accum = disp
        else:
            accum = accum + disp

    return accum / len(_TTA_FLIP_COMBOS)


def _run_refine_on_crop(args, crop, crop_size, model_state):
    refine_extra_steps = int(args.refine) if args.refine is not None else 0
    refine_parts = refine_extra_steps + 1
    refine_fraction = 1.0 / float(refine_parts)
    use_tta = bool(getattr(args, "tta", False))
    model = model_state["model"]
    amp_enabled = model_state["amp_enabled"]
    amp_dtype = model_state["amp_dtype"]
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

    n_forward = 0

    for refine_idx in range(refine_parts):
        if current_coords.shape[0] == 0:
            break

        if refine_idx == 0:
            extrap_vox = initial_extrap_vox
        else:
            extrap_vox = _points_to_voxels(current_coords, crop_size)

        inputs = _build_model_inputs(
            crop["volume"], cond_vox, extrap_vox, other_wraps_vox=other_wraps_vox
        ).to(args.device)

        if use_tta:
            disp_single = _run_model_tta(model, inputs, amp_enabled, amp_dtype)
        else:
            disp_single = _forward_model_displacement(model, inputs, amp_enabled, amp_dtype)

        n_forward += 1
        # Apply a fixed fraction of each stage's full displacement prediction.
        coords_t = torch.from_numpy(current_coords).float().to(args.device)
        disp_sampled = _sample_displacement_field(disp_single, coords_t)
        next_coords = (coords_t + disp_sampled * refine_fraction).detach().cpu().numpy().astype(np.float32)

        in_bounds = _in_bounds_mask(next_coords, crop_size)
        if not in_bounds.any():
            current_coords = np.zeros((0, 3), dtype=np.float32)
            if current_uv is not None:
                current_uv = current_uv[:0]
            break

        current_coords = next_coords[in_bounds]
        if current_uv is not None:
            current_uv = current_uv[in_bounds]

    if n_forward == 0:
        return np.zeros((0, 3), dtype=np.float32), None if current_uv is None else current_uv[:0]

    if current_coords is None or current_coords.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), None if current_uv is None else current_uv[:0]
    return current_coords.astype(np.float32, copy=False), current_uv


def run_inference(args, bbox_crops, crop_size, model_state):
    model = model_state["model"]
    amp_enabled = model_state["amp_enabled"]
    amp_dtype = model_state["amp_dtype"]
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
    use_tta = getattr(args, 'tta', False)

    if refine_mode:
        desc = "inference (refine)"
        if use_tta:
            desc = "inference (refine+TTA)"

        for bbox_idx, crop, _, _ in tqdm(valid_items, desc=desc):
            pred_local, pred_uv = _run_refine_on_crop(args, crop, crop_size, model_state)
            if pred_local is None or len(pred_local) == 0:
                continue

            pred_world = pred_local + crop["min_corner"][None, :]
            bbox_mask = _filter_points_in_bbox_mask(pred_world, crop["bbox"])
            pred_world = pred_world[bbox_mask]
            if pred_uv is not None and len(pred_uv) == bbox_mask.shape[0]:
                pred_samples.append((pred_uv[bbox_mask], pred_world))

            out_dir = Path(args.bbox_crops_out_dir) / f"bbox_{bbox_idx:04d}"
            _save_crop_tiff(out_dir, "pred.tif", _points_to_voxels(pred_local, crop_size))
    elif use_tta:
        # TTA path: process one crop at a time to avoid 8x memory blowup.
        for item_idx, (bbox_idx, crop, inputs, extrap_local) in enumerate(
            tqdm(valid_items, desc="inference (TTA)")
        ):
            inputs_dev = inputs.to(args.device)
            disp_single = _run_model_tta(model, inputs_dev, amp_enabled, amp_dtype)

            extrap_coords = torch.from_numpy(extrap_local).float().to(args.device)
            extrap_uv = crop.get("extrap_uv", None)

            disp_sampled = _sample_displacement_field(disp_single, extrap_coords)
            pred_local = extrap_coords + disp_sampled
            pred_world = pred_local.detach().cpu().numpy() + crop["min_corner"][None, :]
            bbox_mask = _filter_points_in_bbox_mask(pred_world, crop["bbox"])
            pred_world = pred_world[bbox_mask]
            if extrap_uv is not None and len(extrap_uv) == bbox_mask.shape[0]:
                pred_samples.append((extrap_uv[bbox_mask], pred_world))

            out_dir = Path(args.bbox_crops_out_dir) / f"bbox_{bbox_idx:04d}"
            _save_crop_tiff(out_dir, "pred.tif", _points_to_voxels(pred_local.detach().cpu().numpy(), crop_size))
    else:
        # Standard batched path.
        n_batches = (len(valid_items) + batch_size - 1) // batch_size
        for batch_start in tqdm(range(0, len(valid_items), batch_size), total=n_batches, desc="inference"):
            batch = valid_items[batch_start:batch_start + batch_size]

            # Stack inputs along batch dim and run a single forward pass.
            batch_inputs = torch.cat([item[2] for item in batch], dim=0).to(args.device)
            disp_pred = _forward_model_displacement(model, batch_inputs, amp_enabled, amp_dtype)

            # Sample displacement per-crop from the batched output.
            for i, (bbox_idx, crop, _, extrap_local) in enumerate(batch):
                disp_single = disp_pred[i:i+1]  # [1, 3, D, H, W]

                extrap_coords = torch.from_numpy(extrap_local).float().to(args.device)
                extrap_uv = crop.get("extrap_uv", None)

                disp_sampled = _sample_displacement_field(disp_single, extrap_coords)
                pred_local = extrap_coords + disp_sampled
                pred_world = pred_local.detach().cpu().numpy() + crop["min_corner"][None, :]
                bbox_mask = _filter_points_in_bbox_mask(pred_world, crop["bbox"])
                pred_world = pred_world[bbox_mask]
                if extrap_uv is not None and len(extrap_uv) == bbox_mask.shape[0]:
                    pred_samples.append((extrap_uv[bbox_mask], pred_world))

                out_dir = Path(args.bbox_crops_out_dir) / f"bbox_{bbox_idx:04d}"
                _save_crop_tiff(out_dir, "pred.tif", _points_to_voxels(pred_local.detach().cpu().numpy(), crop_size))

    return pred_samples


def save_tifxyz_output(args, tgt_segment, pred_samples, tifxyz_uuid, tifxyz_step_size,
                       tifxyz_voxel_size_um, checkpoint_path, cond_direction, volume_scale):
    tgt_segment.use_full_resolution()
    full_zyxs = tgt_segment.get_zyxs(stored_resolution=False)

    full_pred_zyxs = full_zyxs.copy()
    h_full, w_full = full_pred_zyxs.shape[:2]

    # Compute UV extent across original grid and all prediction UVs
    uv_r_min, uv_c_min = 0, 0
    uv_r_max, uv_c_max = h_full - 1, w_full - 1
    for uv, _ in pred_samples:
        uv_r_min = min(uv_r_min, int(uv[:, 0].min()))
        uv_c_min = min(uv_c_min, int(uv[:, 1].min()))
        uv_r_max = max(uv_r_max, int(uv[:, 0].max()))
        uv_c_max = max(uv_c_max, int(uv[:, 1].max()))

    # Allocate extended grid filled with -1.0
    ext_h = uv_r_max - uv_r_min + 1
    ext_w = uv_c_max - uv_c_min + 1
    extended = np.full((ext_h, ext_w, 3), -1.0, dtype=np.float32)

    # Place original data at offset
    r_off = -uv_r_min
    c_off = -uv_c_min
    extended[r_off:r_off + h_full, c_off:c_off + w_full] = full_pred_zyxs

    # Accumulate predictions and average overlapping points
    pred_acc = np.zeros((ext_h, ext_w, 3), dtype=np.float64)
    pred_count = np.zeros((ext_h, ext_w), dtype=np.int32)
    for uv, pred_world in pred_samples:
        rows = uv[:, 0].astype(np.int64) - uv_r_min
        cols = uv[:, 1].astype(np.int64) - uv_c_min
        np.add.at(pred_acc, (rows, cols), pred_world)
        np.add.at(pred_count, (rows, cols), 1)

    has_pred = pred_count > 0
    extended[has_pred] = (pred_acc[has_pred] / pred_count[has_pred, np.newaxis]).astype(np.float32)

    full_pred_zyxs = extended

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
            "cond_direction": cond_direction,
            "extrapolation_method": args.extrapolation_method,
            "refine_steps": None if args.refine is None else int(args.refine) + 1,
        }
    )
    print(f"Saved tifxyz to {os.path.join(args.tifxyz_out_dir, tifxyz_uuid)}")


def reassemble_predictions_to_grid(pred_samples):
    """Reassemble list of (uv, world_pts) into a dense HxWx3 grid in UV space."""
    if not pred_samples:
        return np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0, 0), dtype=bool), (0, 0)

    all_uv = np.concatenate([uv for uv, _ in pred_samples], axis=0)
    uv_r_min = int(all_uv[:, 0].min())
    uv_c_min = int(all_uv[:, 1].min())
    uv_r_max = int(all_uv[:, 0].max())
    uv_c_max = int(all_uv[:, 1].max())

    h = uv_r_max - uv_r_min + 1
    w = uv_c_max - uv_c_min + 1

    grid_acc = np.zeros((h, w, 3), dtype=np.float64)
    grid_count = np.zeros((h, w), dtype=np.int32)

    for uv, pts in pred_samples:
        rows = uv[:, 0].astype(np.int64) - uv_r_min
        cols = uv[:, 1].astype(np.int64) - uv_c_min
        np.add.at(grid_acc, (rows, cols), pts.astype(np.float64))
        np.add.at(grid_count, (rows, cols), 1)

    grid_valid = grid_count > 0
    grid_zyxs = np.full((h, w, 3), -1.0, dtype=np.float32)
    grid_zyxs[grid_valid] = (grid_acc[grid_valid] / grid_count[grid_valid, np.newaxis]).astype(np.float32)

    return grid_zyxs, grid_valid, (uv_r_min, uv_c_min)


def _build_uv_grid(uv_offset, shape_hw):
    r0, c0 = uv_offset
    h, w = shape_hw
    rows = np.arange(r0, r0 + h, dtype=np.int64)
    cols = np.arange(c0, c0 + w, dtype=np.int64)
    return np.stack(np.meshgrid(rows, cols, indexing='ij'), axis=-1)


def prepare_next_iteration_cond(
    full_grid_zyxs, full_valid, full_uv_offset,
    pred_grid_zyxs, pred_grid_valid, pred_uv_offset,
    cond_direction, keep_voxels=None,
):
    """Append new prediction band outside current boundary, returning next full grid state."""
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

    axis = None
    if cond_direction == "left":    # growing right -> keep cols right of current boundary
        growth_mask = pred_cols_abs > full_c1
        ordered_axis_vals = np.sort(np.unique(pred_cols_abs[growth_mask]))
        axis = "col"
    elif cond_direction == "right":  # growing left -> keep cols left of current boundary
        growth_mask = pred_cols_abs < full_c0
        ordered_axis_vals = np.sort(np.unique(pred_cols_abs[growth_mask]))[::-1]
        axis = "col"
    elif cond_direction == "up":     # growing down -> keep rows below current boundary
        growth_mask = pred_rows_abs > full_r1
        ordered_axis_vals = np.sort(np.unique(pred_rows_abs[growth_mask]))
        axis = "row"
    elif cond_direction == "down":   # growing up -> keep rows above current boundary
        growth_mask = pred_rows_abs < full_r0
        ordered_axis_vals = np.sort(np.unique(pred_rows_abs[growth_mask]))[::-1]
        axis = "row"
    else:
        raise ValueError(f"Unknown cond_direction '{cond_direction}'")

    if ordered_axis_vals.size == 0:
        return full_grid_zyxs, full_valid, full_uv_offset, [], 0

    n_keep = ordered_axis_vals.size if keep_voxels is None else min(ordered_axis_vals.size, keep_voxels)
    kept_axis_vals = ordered_axis_vals[:n_keep]

    if axis == "row":
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


def main():
    args = parse_args()
    if args.iter_keep_voxels is not None and args.iter_keep_voxels < 1:
        raise ValueError("--iter-keep-voxels must be >= 1 when provided.")
    if args.bbox_overlap_frac < 0.0 or args.bbox_overlap_frac >= 1.0:
        raise ValueError("--bbox-overlap-frac must be in [0, 1).")
    if args.refine is not None and args.refine < 1:
        raise ValueError("--refine must be >= 1 when provided.")

    refine_mode = args.refine is not None
    growth_iterations = args.iterations
    if refine_mode:
        print(
            f"--refine={args.refine} enabled; running {int(args.refine) + 1} "
            "fractional refinement stages inside each outer iteration."
        )

    crop_size = tuple(args.crop_size)
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

    tifxyz_step_size, tifxyz_voxel_size_um = resolve_tifxyz_params(
        args, model_config, args.volume_scale
    )

    volume = zarr.open_group(args.volume_path, mode='r')
    tgt_segment, stored_zyxs, valid_s, cond_direction, h_s, w_s = setup_segment(args, volume)

    r0_s, r1_s, c0_s, c1_s = compute_window_and_split(
        args, stored_zyxs, valid_s, cond_direction, h_s, w_s, crop_size
    )

    scale_y, scale_x = tgt_segment._scale
    full_h, full_w = tgt_segment.full_resolution_shape
    r0_full = max(0, int(np.floor(r0_s / scale_y)))
    r1_full = min(full_h, int(np.ceil((r1_s + 1) / scale_y)))
    c0_full = max(0, int(np.floor(c0_s / scale_x)))
    c1_full = min(full_w, int(np.ceil((c1_s + 1) / scale_x)))

    tgt_segment.use_full_resolution()
    x, y, z, valid = tgt_segment[r0_full:r1_full, c0_full:c1_full]

    window_zyxs = np.stack([z, y, x], axis=-1)
    current_grid = window_zyxs.copy()
    current_valid = valid.copy()
    current_uv_offset = (r0_full, c0_full)

    if growth_iterations > 1 and args.iter_keep_voxels is None:
        print(
            "iter-keep-voxels not set; each iteration will keep all newly predicted "
            "rows/cols beyond the current boundary."
        )

    all_pred_samples = []
    bboxes = []

    for iteration in range(growth_iterations):
        print(f"[iteration {iteration + 1}/{growth_iterations}]")
        cond_zyxs = current_grid
        valid = current_valid
        uv_cond = _build_uv_grid(current_uv_offset, cond_zyxs.shape[:2])

        # Outside path uses the grow-direction inner edge for bbox extraction.
        bboxes, _ = get_cond_edge_bboxes(
            cond_zyxs, cond_direction, crop_size, outer_edge=False,
            overlap_frac=args.bbox_overlap_frac,
        )

        # --- build bbox crops + run inference ---
        pred_samples = []
        bbox_crops = build_bbox_crop_data(
            args, bboxes, cond_zyxs, valid, uv_cond, cond_direction, crop_size, tgt_segment,
            args.volume_scale, extrapolation_settings,
        )

        if run_model_inference and model_state is not None:
            pred_samples = run_inference(args, bbox_crops, crop_size, model_state)

        if not pred_samples:
            print("No predicted samples this iteration; stopping iterative growth.")
            break

        pred_grid, pred_valid, pred_offset = reassemble_predictions_to_grid(pred_samples)
        merged_cond, merged_valid, merged_uv_offset, kept, n_kept_axis = prepare_next_iteration_cond(
            current_grid, current_valid, current_uv_offset,
            pred_grid, pred_valid, pred_offset,
            cond_direction, args.iter_keep_voxels,
        )
        if not kept:
            print("No new rows/cols beyond current boundary; stopping iterative growth.")
            break

        all_pred_samples.extend(kept)
        current_grid = merged_cond
        current_valid = merged_valid
        current_uv_offset = merged_uv_offset
        print(f"  kept {n_kept_axis} new {'rows' if cond_direction in ['up', 'down'] else 'cols'}")

    pred_samples = all_pred_samples

    if tifxyz_uuid is not None and pred_samples:
        save_tifxyz_output(
            args, tgt_segment, pred_samples, tifxyz_uuid, tifxyz_step_size,
            tifxyz_voxel_size_um, checkpoint_path, cond_direction, args.volume_scale,
        )


if __name__ == '__main__':
    main()
