import argparse
import zarr

import vesuvius.tifxyz as tifxyz
import numpy as np
import random
import json
import os
from datetime import datetime
import torch
import torch.nn.functional as F
from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid
from vesuvius.neural_tracing.datasets.extrapolation import _EXTRAPOLATION_METHODS, apply_degradation
from vesuvius.image_proc.intensity.normalization import normalize_zscore
from vesuvius.neural_tracing.models import load_checkpoint
from vesuvius.neural_tracing.tifxyz import save_tifxyz
import edt

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
    if cond_direction == "left":
        start = edge_idx - (cond_size - 1)
    elif cond_direction == "right":
        mask_size = window_size - cond_size
        start = edge_idx - mask_size
    elif cond_direction == "up":
        start = edge_idx - (cond_size - 1)
    elif cond_direction == "down":
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
    
    if cond_direction == "left":
        # conditioning is left, mask the right
        cond_zyxs = zyxs[:, :c_split]
        masked_zyxs = zyxs[:, c_split:]
        uv_cond = uv_full[:, :c_split]
        uv_mask = uv_full[:, c_split:]
    elif cond_direction == "right":
        # conditioning is right, mask the left
        cond_zyxs = zyxs[:, c_split:]
        masked_zyxs = zyxs[:, :c_split]
        uv_cond = uv_full[:, c_split:]
        uv_mask = uv_full[:, :c_split]
    elif cond_direction == "up":
        # conditioning is up, mask the bottom
        cond_zyxs = zyxs[:r_split, :]
        masked_zyxs = zyxs[r_split:, :]
        uv_cond = uv_full[:r_split, :]
        uv_mask = uv_full[r_split:, :]
    elif cond_direction == "down":
        # conditioning is down, mask the top
        cond_zyxs = zyxs[r_split:, :]
        masked_zyxs = zyxs[:r_split, :]
        uv_cond = uv_full[r_split:, :]
        uv_mask = uv_full[:r_split, :]
    
    return cond_zyxs, masked_zyxs, uv_cond, uv_mask
    
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

def get_cond_edge_bboxes(cond_zyxs, cond_direction, crop_size, outer_edge=False):
    edge = _get_cond_edge(cond_zyxs, cond_direction, outer_edge=outer_edge)
    
    z_edge, y_edge, x_edge = edge[:, 0], edge[:, 1], edge[:, 2]
    
    bboxes = []
    group_start = 0
    
    for i in range(1, len(z_edge) + 1):
        if i < len(z_edge):
            # Look ahead: check if adding point i still fits within crop_size (inclusive bounds)
            z_span = z_edge[group_start:i + 1].max() - z_edge[group_start:i + 1].min()
            y_span = y_edge[group_start:i + 1].max() - y_edge[group_start:i + 1].min()
            x_span = x_edge[group_start:i + 1].max() - x_edge[group_start:i + 1].min()
            if (
                z_span <= (crop_size[0] - 1) and
                y_span <= (crop_size[1] - 1) and
                x_span <= (crop_size[2] - 1)
            ):
                continue
        
        # Emit bbox for group_start:i (aligned to voxel indices)
        zc = (z_edge[group_start:i].min() + z_edge[group_start:i].max()) / 2
        yc = (y_edge[group_start:i].min() + y_edge[group_start:i].max()) / 2
        xc = (x_edge[group_start:i].min() + x_edge[group_start:i].max()) / 2
        bboxes.append(_bbox_from_center((zc, yc, xc), crop_size))
        group_start = i
        
        
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

def _filter_points_in_bbox(points, bbox):
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    z, y, x = points[:, 0], points[:, 1], points[:, 2]
    mask = (
        (z >= z_min) & (z <= z_max) &
        (y >= y_min) & (y <= y_max) &
        (x >= x_min) & (x <= x_max)
    )
    return points[mask]

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
    crop_size_arr = np.asarray(crop_size, dtype=np.float32)
    in_bounds = (
        (local[:, 0] >= 0) & (local[:, 0] < crop_size_arr[0]) &
        (local[:, 1] >= 0) & (local[:, 1] < crop_size_arr[1]) &
        (local[:, 2] >= 0) & (local[:, 2] < crop_size_arr[2])
    )
    return local[in_bounds].astype(np.float32)

def _points_to_voxels(points_local, crop_size):
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    vox = np.zeros(tuple(crop_size_arr.tolist()), dtype=np.float32)
    if points_local is None or len(points_local) == 0:
        return vox
    coords = np.rint(points_local).astype(np.int64)
    in_bounds = (
        (coords[:, 0] >= 0) & (coords[:, 0] < crop_size_arr[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < crop_size_arr[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < crop_size_arr[2])
    )
    coords = coords[in_bounds]
    if coords.size > 0:
        vox[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return vox

def _apply_edt_dilation(vox, dilation_radius):
    dist = edt.edt(1 - vox, parallel=1)
    return (dist <= dilation_radius).astype(np.float32)

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
    coords_norm[:, 0] = 2 * coords_norm[:, 0] / (D - 1) - 1
    coords_norm[:, 1] = 2 * coords_norm[:, 1] / (H - 1) - 1
    coords_norm[:, 2] = 2 * coords_norm[:, 2] / (W - 1) - 1

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
        in_bounds = (
            (z_extrap_local >= 0) & (z_extrap_local < crop_size[0]) &
            (y_extrap_local >= 0) & (y_extrap_local < crop_size[1]) &
            (x_extrap_local >= 0) & (x_extrap_local < crop_size[2])
        )
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


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Row/col split inference for neural tracing")
    parser.add_argument("--segments-path", type=str, required=True)
    parser.add_argument("--volume-path", type=str, required=True)
    parser.add_argument("--volume-scale", type=int, default=1)
    parser.add_argument("--cond-pct", type=float, default=0.50)
    parser.add_argument("--segment-idx", type=int, default=0)
    parser.add_argument("--crop-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--no-full-resolution", dest="full_resolution", action="store_false")
    parser.add_argument("--window-pad", type=int, default=4)
    parser.add_argument("--no-extrapolation", dest="use_extrapolation", action="store_false")
    parser.add_argument("--no-extrapolate-outside", dest="extrapolate_outside", action="store_false")
    parser.add_argument("--no-visualize-full-seg", dest="visualize_full_seg", action="store_false")
    parser.add_argument("--extrapolation-method", type=str, default="linear_edge")
    parser.add_argument("--extrap-degrade-prob", type=float, default=0.0)
    parser.add_argument("--no-edge-on-outer", dest="edge_on_outer", action="store_false")
    parser.add_argument("--no-debug-outside", dest="debug_outside", action="store_false")
    parser.add_argument("--no-build-bbox-crops", dest="build_bbox_crops", action="store_false")
    parser.add_argument("--no-save-bbox-crops", dest="save_bbox_crops", action="store_false")
    parser.add_argument(
        "--cond-direction",
        type=str,
        default="random",
        choices=VALID_DIRECTIONS + ["random"],
        help="Direction to condition from; use 'random' to pick a valid edge.",
    )
    parser.add_argument("--bbox-crops-out-dir", type=str, default="/tmp/rowcol_bbox_crops")
    parser.add_argument("--no-normalize-volume-crops", dest="normalize_volume_crops", action="store_false")
    parser.add_argument("--no-run-model-inference", dest="run_model_inference", action="store_false")
    parser.add_argument("--no-save-pred-crops", dest="save_pred_crops", action="store_false")
    parser.add_argument("--no-save-full-tifxyz", dest="save_full_tifxyz", action="store_false")
    parser.add_argument("--tifxyz-out-dir", type=str, default=None)
    parser.add_argument("--tifxyz-step-size", type=int, default=None)
    parser.add_argument("--tifxyz-voxel-size-um", type=float, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--napari", action="store_true", help="Launch napari viewer for visualization")
    args = parser.parse_args()

    segments_path = args.segments_path
    volume_path = args.volume_path
    volume_scale = args.volume_scale
    cond_pct = args.cond_pct
    segment_idx = args.segment_idx
    crop_size = tuple(args.crop_size)
    use_full_resolution = args.full_resolution
    window_pad = args.window_pad
    use_extrapolation = args.use_extrapolation
    extrapolate_outside = args.extrapolate_outside
    visualize_full_seg = args.visualize_full_seg
    extrapolation_method = args.extrapolation_method
    extrap_degrade_prob = args.extrap_degrade_prob
    edge_on_outer = args.edge_on_outer
    debug_outside = args.debug_outside
    build_bbox_crops = args.build_bbox_crops
    save_bbox_crops = args.save_bbox_crops
    cond_direction_arg = args.cond_direction
    bbox_crops_out_dir = args.bbox_crops_out_dir
    normalize_volume_crops = args.normalize_volume_crops
    run_model_inference = args.run_model_inference
    save_pred_crops = args.save_pred_crops
    save_full_tifxyz = args.save_full_tifxyz
    tifxyz_out_dir = args.tifxyz_out_dir
    tifxyz_step_size = args.tifxyz_step_size
    tifxyz_voxel_size_um = args.tifxyz_voxel_size_um
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    device = args.device

    if args.napari:
        import napari
        viewer = napari.Viewer()

    model = None
    model_config = None
    amp_enabled = False
    amp_dtype = torch.float16
    expected_in_channels = 3
    tifxyz_uuid = None
    use_dilation = False
    dilation_radius = 1.0

    if run_model_inference:
        if checkpoint_path is None and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            checkpoint_path = cfg.get("load_ckpt", None)

        if checkpoint_path is None:
            raise RuntimeError("checkpoint_path not set; provide a trained rowcol_cond checkpoint.")

        model, model_config = load_checkpoint(checkpoint_path)
        model.to(device)
        model.eval()

        expected_in_channels = int(model_config.get("in_channels", 3))
        use_dilation = bool(model_config.get('use_dilation', False))
        dilation_radius = float(model_config.get('dilation_radius', 1.0))
        mixed_precision = str(model_config.get("mixed_precision", "no")).lower()
        if device.startswith("cuda") and mixed_precision in ("bf16", "fp16", "float16"):
            amp_enabled = True
            amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        if save_full_tifxyz:
            ckpt_name = os.path.splitext(os.path.basename(str(checkpoint_path)))[0]
            timestamp = datetime.now().strftime("%H%M%S")
            tifxyz_uuid = f"displacement_tifxyz_{ckpt_name}_{timestamp}"

    if save_full_tifxyz:
        if tifxyz_step_size is None:
            if model_config is not None:
                tifxyz_step_size = model_config.get(
                    "step_size",
                    model_config.get("heatmap_step_size", 10),
                )
            else:
                tifxyz_step_size = 10
        if tifxyz_voxel_size_um is None:
            meta_path = os.path.join(volume_path, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "rt") as meta_fp:
                    tifxyz_voxel_size_um = json.load(meta_fp).get("voxelsize", None)
        if tifxyz_voxel_size_um is None:
            tifxyz_voxel_size_um = 8.24
        tifxyz_step_size = int(round(float(tifxyz_step_size) * (2 ** volume_scale)))
    
    volume = zarr.open_group(volume_path, mode='r')
    dataset_segments = list(tifxyz.load_folder(segments_path))
    
    retarget_factor = 2 ** volume_scale
    
    scaled_segments = []
    for i, seg in enumerate(dataset_segments):
        seg_scaled = seg.retarget(retarget_factor)
        seg_scaled.volume = volume
        scaled_segments.append(seg_scaled)
        tgt_segment = scaled_segments[segment_idx]

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
    if cond_direction_arg != "random":
        if cond_direction_arg not in valid_dirs:
            raise RuntimeError(
                f"Requested cond_direction '{cond_direction_arg}' not available for this segment. "
                f"Valid options: {valid_dirs}"
            )
        cond_direction = cond_direction_arg
    else:
        cond_direction = random.choice(valid_dirs)

    r_edge_s, c_edge_s = _edge_index_from_valid(valid_s, cond_direction)
    if r_edge_s is None and c_edge_s is None:
        raise RuntimeError("No valid edge found for segment.")

    if cond_direction in ["left", "right"]:
        cond_edge_strip = stored_zyxs[:, c_edge_s:c_edge_s + 1]
    else:
        cond_edge_strip = stored_zyxs[r_edge_s:r_edge_s + 1, :]

    bboxes, _ = get_cond_edge_bboxes(
        cond_edge_strip, cond_direction, crop_size, outer_edge=edge_on_outer
    )

    r0_s, r1_s, c0_s, c1_s = get_window_bounds_from_bboxes(
        stored_zyxs, valid_s, bboxes, pad=window_pad
    )

    win_h = r1_s - r0_s + 1
    win_w = c1_s - c0_s + 1
    if win_h < 2 or win_w < 2:
        raise RuntimeError("Window too small after edge-based bounds.")
    
    if extrapolate_outside:
        cond_h = win_h
        cond_w = win_w
        if cond_direction in ["left", "right"]:
            c0_s, c1_s = _place_window_on_edge(c_edge_s, cond_w, cond_w, cond_direction, w_s - 1)
            r0_s, r1_s = _clamp_window(r0_s, cond_h, 0, h_s - 1)
        else:
            r0_s, r1_s = _place_window_on_edge(r_edge_s, cond_h, cond_h, cond_direction, h_s - 1)
            c0_s, c1_s = _clamp_window(c0_s, cond_w, 0, w_s - 1)
        r_split_s = None
        c_split_s = None
    else:
        cond_h = int(round(win_h * cond_pct))
        cond_w = int(round(win_w * cond_pct))
        cond_h = max(1, min(win_h - 1, cond_h))
        cond_w = max(1, min(win_w - 1, cond_w))
        mask_h = win_h - cond_h
        mask_w = win_w - cond_w

        if cond_direction in ["left", "right"]:
            c0_s, c1_s = _place_window_on_edge(c_edge_s, win_w, cond_w, cond_direction, w_s - 1)
            r0_s, r1_s = _clamp_window(r0_s, win_h, 0, h_s - 1)
            r_split_rel = cond_h
            c_split_rel = cond_w if cond_direction == "left" else mask_w
        else:
            r0_s, r1_s = _place_window_on_edge(r_edge_s, win_h, cond_h, cond_direction, h_s - 1)
            c0_s, c1_s = _clamp_window(c0_s, win_w, 0, w_s - 1)
            r_split_rel = cond_h if cond_direction == "up" else mask_h
            c_split_rel = cond_w

        r_split_s = r0_s + r_split_rel
        c_split_s = c0_s + c_split_rel

    scale_y, scale_x = tgt_segment._scale
    full_h, full_w = tgt_segment.full_resolution_shape
    r0_full = max(0, int(np.floor(r0_s / scale_y)))
    r1_full = min(full_h, int(np.ceil((r1_s + 1) / scale_y)))
    c0_full = max(0, int(np.floor(c0_s / scale_x)))
    c1_full = min(full_w, int(np.ceil((c1_s + 1) / scale_x)))

    if use_full_resolution:
        if not extrapolate_outside:
            r_split_full = int(round(r_split_s / scale_y))
            c_split_full = int(round(c_split_s / scale_x))

            r0_full = max(0, min(r0_full, max(0, r_split_full - 1)))
            r1_full = min(full_h, max(r1_full, min(full_h, r_split_full + 1)))
            c0_full = max(0, min(c0_full, max(0, c_split_full - 1)))
            c1_full = min(full_w, max(c1_full, min(full_w, c_split_full + 1)))

        tgt_segment.use_full_resolution()
        x, y, z, valid = tgt_segment[r0_full:r1_full, c0_full:c1_full]
    else:
        if not extrapolate_outside:
            r_split_full = r_split_s
            c_split_full = c_split_s
        x, y, z, valid = tgt_segment[r0_s:r1_s + 1, c0_s:c1_s + 1]
        r0_full, c0_full = r0_s, c0_s

    window_zyxs = np.stack([z, y, x], axis=-1)
    if extrapolate_outside:
        cond_zyxs = window_zyxs
        masked_zyxs = None
        r0_uv = r0_full
        r1_uv = r0_full + window_zyxs.shape[0] - 1
        c0_uv = c0_full
        c1_uv = c0_full + window_zyxs.shape[1] - 1

        r_coords = np.arange(r0_uv, r1_uv + 1)
        c_coords = np.arange(c0_uv, c1_uv + 1)
        uv_cond = np.stack(np.meshgrid(r_coords, c_coords, indexing='ij'), axis=-1)

        if cond_direction in ["left", "right"]:
            cond_w = window_zyxs.shape[1]
            win_w_total = max(cond_w + 1, int(round(cond_w / cond_pct)))
            mask_w = win_w_total - cond_w
            if mask_w <= 0:
                uv_mask = np.zeros((0, 0, 2), dtype=uv_cond.dtype)
            else:
                if cond_direction == "left":
                    mask_c0 = c0_uv - mask_w
                    mask_c1 = c0_uv - 1
                else:
                    mask_c0 = c1_uv + 1
                    mask_c1 = c1_uv + mask_w
                mask_cols = np.arange(mask_c0, mask_c1 + 1)
                uv_mask = np.stack(np.meshgrid(r_coords, mask_cols, indexing='ij'), axis=-1)
        else:
            cond_h = window_zyxs.shape[0]
            win_h_total = max(cond_h + 1, int(round(cond_h / cond_pct)))
            mask_h = win_h_total - cond_h
            if mask_h <= 0:
                uv_mask = np.zeros((0, 0, 2), dtype=uv_cond.dtype)
            else:
                if cond_direction == "up":
                    mask_r0 = r0_uv - mask_h
                    mask_r1 = r0_uv - 1
                else:
                    mask_r0 = r1_uv + 1
                    mask_r1 = r1_uv + mask_h
                mask_rows = np.arange(mask_r0, mask_r1 + 1)
                uv_mask = np.stack(np.meshgrid(mask_rows, c_coords, indexing='ij'), axis=-1)
        if debug_outside:
            print(f"[outside] cond_direction={cond_direction} edge_on_outer={edge_on_outer} cond_pct={cond_pct}")
            print(f"[outside] uv_cond shape={uv_cond.shape} rows=({uv_cond[...,0].min()},{uv_cond[...,0].max()}) cols=({uv_cond[...,1].min()},{uv_cond[...,1].max()})")
            if uv_mask.size == 0:
                print("[outside] uv_mask shape=EMPTY")
            else:
                print(f"[outside] uv_mask shape={uv_mask.shape} rows=({uv_mask[...,0].min()},{uv_mask[...,0].max()}) cols=({uv_mask[...,1].min()},{uv_mask[...,1].max()})")
    else:
        r_split_rel = r_split_full - r0_full
        c_split_rel = c_split_full - c0_full
        cond_zyxs, masked_zyxs, uv_cond, uv_mask = split_grid(
            window_zyxs, (r0_full, c0_full), cond_direction, r_split_rel, c_split_rel
        )

    uv_mask_flat = None
    if uv_mask is not None and uv_mask.size > 0:
        uv_mask_flat = uv_mask.reshape(-1, 2)

    extrap_result = None
    extrap_uv_full = None
    if use_extrapolation:
        if extrapolate_outside:
            edge_pts = _get_cond_edge(
                cond_zyxs, cond_direction, outer_edge=edge_on_outer
            ).reshape(-1, 3)
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
                edge_pts, cond_bounds, crop_size_extrap, cond_pct, cond_direction
            )

            extrap_result = compute_extrapolation_infer(
                uv_cond=uv_cond,
                zyx_cond=cond_zyxs,
                uv_query=uv_mask,
                min_corner=zyx_min,
                crop_size=crop_size_extrap,
                method=extrapolation_method,
                cond_direction=cond_direction,
                degrade_prob=extrap_degrade_prob,
                skip_bounds_check=True
            )
            if debug_outside:
                if extrap_result is None:
                    print("[outside] extrap_result=None")
                else:
                    coords = extrap_result["extrap_coords_local"]
                    if coords.size == 0 or np.any(~np.isfinite(coords)):
                        print(f"[outside] extrap_coords_local invalid size={coords.shape} finite={np.all(np.isfinite(coords))}")
                    else:
                        mins = coords.min(axis=0)
                        maxs = coords.max(axis=0)
                        print(f"[outside] extrap_coords_local mins={mins} maxs={maxs}")
        else:
            zyx_min = np.floor(window_zyxs.reshape(-1, 3).min(axis=0)).astype(np.int64)
            zyx_max = np.ceil(window_zyxs.reshape(-1, 3).max(axis=0)).astype(np.int64)
            crop_size_extrap = tuple((zyx_max - zyx_min + 1).tolist())
            extrap_result = compute_extrapolation_infer(
                uv_cond=uv_cond,
                zyx_cond=cond_zyxs,
                uv_query=uv_mask,
                min_corner=zyx_min,
                crop_size=crop_size_extrap,
                method=extrapolation_method,
                cond_direction=cond_direction,
                degrade_prob=extrap_degrade_prob,
                skip_bounds_check=True
            )
        if extrap_result is not None:
            extrap_coords_world = extrap_result["extrap_coords_local"] + zyx_min
            if uv_mask_flat is not None and uv_mask_flat.shape[0] == extrap_coords_world.shape[0]:
                extrap_uv_full = uv_mask_flat

    bboxes, edge = get_cond_edge_bboxes(
        cond_zyxs, cond_direction, crop_size, outer_edge=edge_on_outer
    )

    if build_bbox_crops:
        volume_for_crops = _resolve_segment_volume(tgt_segment, volume_scale=volume_scale)
        cond_pts_world = cond_zyxs.reshape(-1, 3)
        cond_valid_mask = ~(cond_pts_world == -1).all(axis=1)
        cond_pts_world = cond_pts_world[cond_valid_mask]
        extrap_pts_world = None
        extrap_uv_world = None
        if use_extrapolation and extrap_result is not None:
            extrap_pts_world = extrap_coords_world
            extrap_uv_world = extrap_uv_full

        bbox_crops = []
        for bbox_idx, bbox in enumerate(bboxes):
            min_corner = _bbox_to_min_corner(bbox)
            vol_crop = _crop_volume_from_min_corner(volume_for_crops, min_corner, crop_size)
            if normalize_volume_crops:
                vol_crop = normalize_zscore(vol_crop)

            cond_world_in = _filter_points_in_bbox(cond_pts_world, bbox)
            cond_local = _points_world_to_local(cond_world_in, min_corner, crop_size)

            extrap_local = None
            extrap_uv = None
            if extrap_pts_world is not None:
                extrap_mask = _filter_points_in_bbox_mask(extrap_pts_world, bbox)
                extrap_world_in = extrap_pts_world[extrap_mask]
                if extrap_uv_world is not None:
                    extrap_uv = extrap_uv_world[extrap_mask]
                extrap_local = _points_world_to_local(extrap_world_in, min_corner, crop_size)

            cond_vox = _points_to_voxels(cond_local, crop_size)
            extrap_vox = _points_to_voxels(extrap_local, crop_size) if extrap_local is not None else None

            if use_dilation:
                cond_vox = _apply_edt_dilation(cond_vox, dilation_radius)
                if extrap_vox is not None:
                    extrap_vox = _apply_edt_dilation(extrap_vox, dilation_radius)

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

            if save_bbox_crops:
                import tifffile
                from pathlib import Path

                out_dir = Path(bbox_crops_out_dir) / f"bbox_{bbox_idx:04d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(out_dir / "volume.tif", vol_crop)
                tifffile.imwrite(out_dir / "cond.tif", cond_vox)
                if extrap_vox is not None:
                    tifffile.imwrite(out_dir / "extrap.tif", extrap_vox)

    pred_pts_world_all = []
    pred_samples = []
    if run_model_inference and build_bbox_crops:
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
            ).to(device)

            extrap_coords = torch.from_numpy(extrap_local).float().to(device)
            extrap_uv = crop.get("extrap_uv", None)

            with torch.no_grad():
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        output = model(inputs)
                else:
                    output = model(inputs)

            if isinstance(output, dict):
                disp_pred = output.get("displacement", None)
                if disp_pred is None:
                    raise RuntimeError("Model output missing 'displacement' head.")
            else:
                disp_pred = output

            disp_sampled = _sample_displacement_field(disp_pred, extrap_coords)
            pred_local = extrap_coords + disp_sampled
            pred_world = pred_local.detach().cpu().numpy() + crop["min_corner"][None, :]
            pred_pts_world_all.append(pred_world)
            if extrap_uv is not None and len(extrap_uv) == pred_world.shape[0]:
                pred_samples.append((extrap_uv, pred_world))

            if save_bbox_crops and save_pred_crops:
                import tifffile
                from pathlib import Path

                out_dir = Path(bbox_crops_out_dir) / f"bbox_{bbox_idx:04d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(out_dir / "pred.tif", _points_to_voxels(pred_local.detach().cpu().numpy(), crop_size))

    if save_full_tifxyz and tifxyz_uuid is not None and pred_samples:
        if use_full_resolution:
            tgt_segment.use_full_resolution()
            full_zyxs = tgt_segment.get_zyxs(stored_resolution=False)
        else:
            tgt_segment.use_stored_resolution()
            full_zyxs = tgt_segment.get_zyxs(stored_resolution=True)

        full_pred_zyxs = full_zyxs.copy()
        h_full, w_full = full_pred_zyxs.shape[:2]
        for uv, pred_world in pred_samples:
            rows = uv[:, 0].astype(np.int64)
            cols = uv[:, 1].astype(np.int64)
            in_bounds = (
                (rows >= 0) & (rows < h_full) &
                (cols >= 0) & (cols < w_full)
            )
            if in_bounds.any():
                full_pred_zyxs[rows[in_bounds], cols[in_bounds]] = pred_world[in_bounds]

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
        # "current_step" is the UV spacing (in full-res pixels) between adjacent
        # grid points in full_pred_zyxs_out.
        if use_full_resolution:
            current_step_y = int(round(2 ** volume_scale))
            current_step_x = int(round(2 ** volume_scale))
        else:
            scale_y, scale_x = tgt_segment._scale
            base_step_y = int(round(1.0 / scale_y)) if scale_y != 0 else 1
            base_step_x = int(round(1.0 / scale_x)) if scale_x != 0 else 1
            current_step_y = int(round(base_step_y * (2 ** volume_scale)))
            current_step_x = int(round(base_step_x * (2 ** volume_scale)))

        stride_y = int(round(float(tifxyz_step_size) / max(1, current_step_y)))
        stride_x = int(round(float(tifxyz_step_size) / max(1, current_step_x)))
        if stride_y > 1 or stride_x > 1:
            full_pred_zyxs_out = full_pred_zyxs_out[::max(1, stride_y), ::max(1, stride_x)]

        save_tifxyz(
            full_pred_zyxs_out,
            tifxyz_out_dir,
            tifxyz_uuid,
            step_size=tifxyz_step_size,
            voxel_size_um=tifxyz_voxel_size_um,
            source=str(checkpoint_path),
            additional_metadata={
                "cond_direction": cond_direction,
                "edge_on_outer": edge_on_outer,
                "extrapolation_method": extrapolation_method,
            }
        )
        print(f"Saved tifxyz to {os.path.join(tifxyz_out_dir, tifxyz_uuid)}")
    
    bbox_pts = []
    for z0, z1, y0, y1, x0, x1 in bboxes:
        verts = [
            [z0, y0, x0], [z0, y0, x1], [z0, y1, x1], [z0, y1, x0],
            [z1, y0, x0], [z1, y0, x1], [z1, y1, x1], [z1, y1, x0],
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]
        for a, b in edges:
            p0 = np.asarray(verts[a], dtype=np.float32)
            p1 = np.asarray(verts[b], dtype=np.float32)
            length = float(np.linalg.norm(p1 - p0))
            n = max(2, int(np.ceil(length)) + 1)
            bbox_pts.extend(np.linspace(p0, p1, n))
            
    if args.napari:
        cond_pts = cond_zyxs.reshape(-1, 3)
        mask_pts = None if extrapolate_outside else masked_zyxs.reshape(-1, 3)
        edge_pts = edge.reshape(-1, 3)
        full_seg_pts = window_zyxs[valid] if visualize_full_seg else None

        if use_full_resolution:
            pt_size = 1
        else:
            pt_size = 3

        viewer.add_points(cond_pts, name='cond_pts', size=pt_size, face_color="red")
        if mask_pts is not None:
            viewer.add_points(mask_pts, name='mask_pts', size=pt_size, face_color="green")
        viewer.add_points(edge_pts, name='edge_pts', size=pt_size, face_color="cyan")
        viewer.add_points(bbox_pts, name='bboxes', size=pt_size, face_color='yellow')
        if full_seg_pts is not None and len(full_seg_pts) > 0:
            viewer.add_points(full_seg_pts, name='full_seg_pts', size=pt_size, face_color='white')

        if use_extrapolation and extrap_result is not None:
            viewer.add_points(extrap_coords_world, name='extrap_pts', size=pt_size, face_color='magenta')
        if pred_pts_world_all:
            pred_pts_world = np.concatenate(pred_pts_world_all, axis=0)
            viewer.add_points(pred_pts_world, name='pred_pts', size=pt_size, face_color='blue')

        napari.run()
    
