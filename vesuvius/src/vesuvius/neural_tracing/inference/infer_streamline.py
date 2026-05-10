from pathlib import Path
import tempfile
from datetime import datetime

import numpy as np
import torch
import zarr
from tqdm import tqdm

from vesuvius.neural_tracing.datasets.common import (
    _read_volume_crop,
    _trim_to_world_bbox,
    _upsample_world_surface,
    open_zarr_group,
    voxelize_surface_grid_into,
)
from vesuvius.neural_tracing.datasets.growth_direction import make_growth_direction_tensor
from vesuvius.neural_tracing.inference.napari_helpers import bbox_wireframe_segments
from vesuvius.neural_tracing.nets.models import load_checkpoint
from vesuvius.tifxyz import read_tifxyz


TIFXYZ_PATH = "/home/sean/Documents/volpkgs/s1_2um.volpkg/traces/w21_0805261025_sel_20260509_110821_1"
VOLUME_PATH = "s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"
VOLUME_SCALE = 0
VOLUME_CACHE_DIR = "/tmp/vesuvius-volume-cache"
CHECKPOINT_PATH = "/home/sean/Downloads/ckpt_090000.pth"

# Keep None unless intentionally debugging shape mismatches. The default reads
# the checkpoint crop size so inference matches the model patch size.
CROP_SIZE = None
BATCH_SIZE = 1
DEVICE = "cuda"

OVERLAP_FRAC = 0.50
OUTPUT_ZARR_PATH = None
SHOW_NAPARI_GEOMETRY = True

GROW_DIRECTION = "left"

POINT_DOWNSAMPLE = 2
POINT_SIZE = 2.0
EDGE_POINT_SIZE = 4.0

TIFXYZ_VOXEL_STEP = 20.0
TIFXYZ_STEPS = 2
INTEGRATION_STEP_SIZE = 4.0
TRACE_VALIDITY_THRESHOLD = 0.5
USE_SURFACE_ATTRACT = True
SURFACE_ATTRACT_WEIGHT = 1.0
SURFACE_ATTRACT_MAX_CORRECTION = 8.0


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


def _get_growth_context(grow_direction):
    cond_direction = _get_direction_spec(grow_direction)["opposite"]
    growth_spec = _get_direction_spec(cond_direction)
    return cond_direction, growth_spec


def _valid_surface_mask(zyx_grid):
    return np.isfinite(zyx_grid).all(axis=-1) & ~(zyx_grid == -1).all(axis=-1)


def _get_cond_edge(cond_zyxs, cond_direction, cond_valid=None):
    spec = _get_direction_spec(cond_direction)
    edge_idx = spec["edge_idx"]
    if cond_valid is not None and np.asarray(cond_valid).shape == cond_zyxs.shape[:2]:
        valid = np.asarray(cond_valid, dtype=bool)
    else:
        valid = _valid_surface_mask(cond_zyxs)

    if not valid.any():
        out_len = cond_zyxs.shape[0] if spec["axis"] == "col" else cond_zyxs.shape[1]
        return np.full((out_len, 3), -1, dtype=cond_zyxs.dtype)

    if spec["axis"] == "col":
        n_rows, n_cols = valid.shape
        out = np.full((n_rows, 3), -1, dtype=cond_zyxs.dtype)
        any_valid = valid.any(axis=1)
        row_idx = np.arange(n_rows, dtype=np.int64)
        if edge_idx == 0 or (edge_idx == -1 and n_cols == 1):
            col_indices = np.argmax(valid, axis=1)
        else:
            col_indices = n_cols - 1 - np.argmax(valid[:, ::-1], axis=1)
        out[any_valid] = cond_zyxs[row_idx[any_valid], col_indices[any_valid], :]
        return out

    n_rows, n_cols = valid.shape
    out = np.full((n_cols, 3), -1, dtype=cond_zyxs.dtype)
    any_valid = valid.any(axis=0)
    col_idx = np.arange(n_cols, dtype=np.int64)
    if edge_idx == 0 or (edge_idx == -1 and n_rows == 1):
        row_indices = np.argmax(valid, axis=0)
    else:
        row_indices = n_rows - 1 - np.argmax(valid[::-1, :], axis=0)
    out[any_valid] = cond_zyxs[row_indices[any_valid], col_idx[any_valid], :]
    return out


def get_cond_edge_bboxes(cond_zyxs, cond_direction, crop_size, overlap_frac=0.15, cond_valid=None):
    # Build center-out crop anchors along the conditioning edge. Each chunk grows
    # while its XYZ span still fits in one crop-sized bbox.
    edge = _get_cond_edge(cond_zyxs, cond_direction, cond_valid=cond_valid)

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

    span_limit = crop_size_arr - 1

    def _chunk_ordered_indices(ordered_indices):
        chunks = []
        if len(ordered_indices) == 0:
            return chunks
        start = 0
        while start < len(ordered_indices):
            first_pt = edge[ordered_indices[start]]
            running_min = first_pt.copy()
            running_max = first_pt.copy()
            end = start + 1
            while end < len(ordered_indices):
                next_pt = edge[ordered_indices[end]]
                candidate_min = np.minimum(running_min, next_pt)
                candidate_max = np.maximum(running_max, next_pt)
                if np.all((candidate_max - candidate_min) <= span_limit):
                    running_min = candidate_min
                    running_max = candidate_max
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
            # Align to voxel indices so inclusive bounds match a crop of size crop_size.
            half = (crop_size_arr - 1) / 2.0
            min_corner = np.floor(center - half).astype(np.int64)
            max_corner = min_corner + (crop_size_arr - 1)
            bbox = (
                int(min_corner[0]), int(max_corner[0]),
                int(min_corner[1]), int(max_corner[1]),
                int(min_corner[2]), int(max_corner[2]),
            )
            if bbox in seen_bboxes:
                continue
            seen_bboxes.add(bbox)
            bboxes.append(bbox)

    _append_chunks(first_chunks)
    _append_chunks(second_chunks)

    return bboxes, edge


def _bbox_to_exclusive_world_bbox(bbox):
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    return (
        float(z_min),
        float(z_max) + 1.0,
        float(y_min),
        float(y_max) + 1.0,
        float(x_min),
        float(x_max) + 1.0,
    )


def _stored_grid_bbox_for_world_bbox(stored_zyxs, valid, world_bbox):
    z_min, z_max, y_min, y_max, x_min, x_max = world_bbox
    valid = np.asarray(valid, dtype=bool)
    in_bounds = (
        valid
        & (stored_zyxs[..., 0] >= z_min) & (stored_zyxs[..., 0] < z_max)
        & (stored_zyxs[..., 1] >= y_min) & (stored_zyxs[..., 1] < y_max)
        & (stored_zyxs[..., 2] >= x_min) & (stored_zyxs[..., 2] < x_max)
    )
    if not in_bounds.any():
        return None

    row_idx = np.flatnonzero(np.any(in_bounds, axis=1))
    col_idx = np.flatnonzero(np.any(in_bounds, axis=0))
    if row_idx.size == 0 or col_idx.size == 0:
        return None
    return int(row_idx[0]), int(row_idx[-1]), int(col_idx[0]), int(col_idx[-1])


def upsample_voxelize_tifxyz_surface_in_bboxes(
    surface,
    bboxes,
    crop_size,
    *,
    stored_zyxs=None,
    valid=None,
    strict_valid=True,
):
    """Voxelize a tifxyz surface inside bboxes using dataset_rowcol_cond's path.

    Each bbox is interpreted as the inclusive `(z_min, z_max, y_min, y_max,
    x_min, x_max)` tuple produced by `get_cond_edge_bboxes`. The returned
    `voxels` are crop-local `(D, H, W)` uint8 masks.
    """
    surface.use_stored_resolution()
    scale_y, scale_x = surface._scale
    crop_shape = tuple(int(v) for v in crop_size)

    if stored_zyxs is None or valid is None:
        x_s, y_s, z_s, valid_s = surface[:]
        stored_zyxs = np.stack([z_s, y_s, x_s], axis=-1)
        valid = np.asarray(valid_s, dtype=bool)
    else:
        stored_zyxs = np.asarray(stored_zyxs)
        valid = np.asarray(valid, dtype=bool)

    results = []
    for bbox in bboxes:
        world_bbox = _bbox_to_exclusive_world_bbox(bbox)
        grid_bbox = _stored_grid_bbox_for_world_bbox(stored_zyxs, valid, world_bbox)
        if grid_bbox is None:
            results.append({
                "bbox": tuple(int(v) for v in bbox),
                "world_bbox": world_bbox,
                "grid_bbox": None,
                "surface_local": None,
                "voxels": np.zeros(crop_shape, dtype=np.uint8),
            })
            continue

        r_min, r_max, c_min, c_max = grid_bbox
        x_s, y_s, z_s, valid_s = surface[r_min:r_max + 1, c_min:c_max + 1]
        if x_s.size == 0:
            trimmed = None
        elif strict_valid and valid_s is not None and not np.asarray(valid_s, dtype=bool).all():
            trimmed = None
        else:
            x_full, y_full, z_full = _upsample_world_surface(x_s, y_s, z_s, scale_y, scale_x)
            trimmed = _trim_to_world_bbox(x_full, y_full, z_full, world_bbox)

        voxels = np.zeros(crop_shape, dtype=np.uint8)
        surface_local = None
        if trimmed is not None:
            x_full, y_full, z_full = trimmed
            surface_world = np.stack([z_full, y_full, x_full], axis=-1)
            z_min, _, y_min, _, x_min, _ = world_bbox
            min_corner = np.round([z_min, y_min, x_min]).astype(np.int64)
            surface_local = (surface_world - min_corner).astype(np.float32, copy=False)
            voxelize_surface_grid_into(voxels, surface_local)

        results.append({
            "bbox": tuple(int(v) for v in bbox),
            "world_bbox": world_bbox,
            "grid_bbox": grid_bbox,
            "surface_local": surface_local,
            "voxels": voxels,
        })

    return results


def _load_surface_zyx(tifxyz_path):
    tifxyz_path = Path(tifxyz_path)
    if not tifxyz_path.exists():
        raise FileNotFoundError(f"tifxyz path not found: {tifxyz_path}")
    if not tifxyz_path.is_dir():
        raise NotADirectoryError(f"tifxyz path must be a directory: {tifxyz_path}")

    surface = read_tifxyz(tifxyz_path)
    surface.use_stored_resolution()
    x, y, z, valid = surface[:]
    stored_zyxs = np.stack([z, y, x], axis=-1)
    valid = np.asarray(valid, dtype=bool)
    return stored_zyxs, valid


def _crop_size_from_config(config):
    crop_size = CROP_SIZE if CROP_SIZE is not None else config.get("crop_size")
    if crop_size is None:
        raise ValueError("Set CROP_SIZE or use a checkpoint config with crop_size.")
    if isinstance(crop_size, (list, tuple)):
        if len(crop_size) != 3:
            raise ValueError(f"crop_size must have three values, got {crop_size!r}")
        return tuple(int(v) for v in crop_size)
    size = int(crop_size)
    return (size, size, size)


def _open_volume_array(volume_path, volume_scale):
    root = open_zarr_group(volume_path, config={"volume_cache_dir": VOLUME_CACHE_DIR})
    if isinstance(root, zarr.hierarchy.Group):
        scale_key = str(int(volume_scale))
        if scale_key not in root:
            raise KeyError(f"volume scale {scale_key!r} not found in {volume_path}")
        return root[scale_key]
    return root


def _bbox_min_corner(bbox):
    z_min, _, y_min, _, x_min, _ = bbox
    return np.asarray([z_min, y_min, x_min], dtype=np.int64)


def _output_zarr_path():
    if OUTPUT_ZARR_PATH is not None:
        return Path(OUTPUT_ZARR_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.gettempdir()) / f"infer_streamline_rowcol_{timestamp}.zarr"


def _merged_window_from_bboxes(bboxes):
    if not bboxes:
        raise ValueError("No bboxes to merge.")
    arr = np.asarray(bboxes, dtype=np.int64)
    min_corner = np.asarray([arr[:, 0].min(), arr[:, 2].min(), arr[:, 4].min()], dtype=np.int64)
    max_corner = np.asarray([arr[:, 1].max(), arr[:, 3].max(), arr[:, 5].max()], dtype=np.int64)
    shape = tuple((max_corner - min_corner + 1).astype(np.int64).tolist())
    return min_corner, shape


def _collapse_output(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _collapse_output(value[0])
    return value


def _prepare_rowcol_inputs(volume_array, voxelized_batch, crop_size, cond_direction):
    vols = []
    conds = []
    for item in voxelized_batch:
        bbox = item["bbox"]
        min_corner = _bbox_min_corner(bbox)
        max_corner = min_corner + np.asarray(crop_size, dtype=np.int64)
        vols.append(_read_volume_crop(volume_array, crop_size, min_corner, max_corner))
        conds.append(np.asarray(item["voxels"], dtype=np.float32))

    vol = torch.from_numpy(np.stack(vols, axis=0)).to(device=DEVICE, dtype=torch.float32).unsqueeze(1)
    cond = torch.from_numpy(np.stack(conds, axis=0)).to(device=DEVICE, dtype=torch.float32).unsqueeze(1)
    direction = make_growth_direction_tensor(
        [cond_direction] * len(voxelized_batch),
        crop_size,
        device=vol.device,
        dtype=vol.dtype,
    )
    return torch.cat([vol, cond, direction], dim=1)


def _ensure_output_accumulators(root, output_name, channels, window_shape):
    outputs = root.require_group("outputs")
    sums = outputs.require_group("sum")
    counts = outputs.require_group("count")
    sum_shape = (int(channels), *tuple(int(v) for v in window_shape))
    count_shape = tuple(int(v) for v in window_shape)
    chunks_3d = tuple(min(int(v), 64) for v in window_shape)
    if output_name not in sums:
        sums.create_dataset(
            output_name,
            shape=sum_shape,
            chunks=(int(channels), *chunks_3d),
            dtype="f4",
            fill_value=0.0,
        )
    if output_name not in counts:
        counts.create_dataset(
            output_name,
            shape=count_shape,
            chunks=chunks_3d,
            dtype="u4",
            fill_value=0,
        )
    return sums[output_name], counts[output_name]


def _accumulate_output(root, output_name, output_batch, voxelized_batch, window_min, window_shape, crop_size):
    output_batch = np.asarray(output_batch, dtype=np.float32)
    if output_batch.ndim != 5:
        raise ValueError(f"Output {output_name!r} must be [B, C, D, H, W], got {output_batch.shape}")
    if tuple(output_batch.shape[2:]) != tuple(crop_size):
        raise ValueError(
            f"Output {output_name!r} spatial shape {output_batch.shape[2:]} "
            f"does not match crop_size {tuple(crop_size)}"
        )

    sum_arr, count_arr = _ensure_output_accumulators(
        root,
        output_name,
        channels=output_batch.shape[1],
        window_shape=window_shape,
    )

    crop_size_arr = np.asarray(crop_size, dtype=np.int64)
    for batch_idx, item in enumerate(voxelized_batch):
        start = _bbox_min_corner(item["bbox"]) - window_min
        end = start + crop_size_arr
        z0, y0, x0 = start.astype(int).tolist()
        z1, y1, x1 = end.astype(int).tolist()
        region = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        sum_arr[(slice(None), *region)] = sum_arr[(slice(None), *region)] + output_batch[batch_idx]
        count_arr[region] = count_arr[region] + np.uint32(1)


def _iter_3d_chunks(shape, chunks):
    z_chunk, y_chunk, x_chunk = (int(v) for v in chunks)
    z_size, y_size, x_size = (int(v) for v in shape)
    for z0 in range(0, z_size, z_chunk):
        for y0 in range(0, y_size, y_chunk):
            for x0 in range(0, x_size, x_chunk):
                yield (
                    slice(z0, min(z0 + z_chunk, z_size)),
                    slice(y0, min(y0 + y_chunk, y_size)),
                    slice(x0, min(x0 + x_chunk, x_size)),
                )


def _finalize_averages(root):
    outputs = root["outputs"]
    avg_group = outputs.require_group("avg")
    avg_group.attrs["window_min_zyx"] = root.attrs.get("window_min_zyx", [0, 0, 0])
    avg_group.attrs["window_shape_zyx"] = root.attrs.get("window_shape_zyx", None)
    for output_name in outputs["sum"].array_keys():
        sum_arr = outputs["sum"][output_name]
        count_arr = outputs["count"][output_name]
        if output_name in avg_group:
            del avg_group[output_name]
        avg = avg_group.create_dataset(
            output_name,
            shape=sum_arr.shape,
            chunks=sum_arr.chunks,
            dtype="f4",
            fill_value=0.0,
        )
        for region in _iter_3d_chunks(count_arr.shape, count_arr.chunks):
            count = count_arr[region].astype(np.float32, copy=False)
            valid = count > 0
            if not bool(valid.any()):
                continue
            for channel_idx in range(sum_arr.shape[0]):
                chunk = np.zeros(count.shape, dtype=np.float32)
                values = sum_arr[(channel_idx, *region)]
                chunk[valid] = values[valid] / count[valid]
                avg[(channel_idx, *region)] = chunk
    return avg_group


def _require_avg_field(avg_group, name, channels):
    if name not in avg_group:
        raise KeyError(f"Missing required averaged output field {name!r}.")
    field = np.asarray(avg_group[name], dtype=np.float32)
    if field.ndim != 4 or field.shape[0] != int(channels):
        raise ValueError(f"Field {name!r} must have shape ({channels}, D, H, W), got {field.shape}")
    return field


def _integration_step_sizes():
    tifxyz_step = float(TIFXYZ_VOXEL_STEP)
    tifxyz_steps = int(TIFXYZ_STEPS)
    step_size = float(INTEGRATION_STEP_SIZE)
    if tifxyz_step <= 0.0:
        raise ValueError("TIFXYZ_VOXEL_STEP must be > 0.")
    if tifxyz_steps <= 0:
        raise ValueError("TIFXYZ_STEPS must be > 0.")
    if step_size <= 0.0:
        raise ValueError("INTEGRATION_STEP_SIZE must be > 0.")
    target_distance = tifxyz_step * tifxyz_steps

    steps = []
    remaining = float(target_distance)
    while remaining > 1e-6:
        this_step = min(step_size, remaining)
        steps.append(float(this_step))
        remaining -= this_step
    return steps, float(target_distance), float(target_distance)


def _sample_field_zyx(field, points_zyx):
    field = np.asarray(field, dtype=np.float32)
    points = np.asarray(points_zyx, dtype=np.float32)
    if field.ndim != 4:
        raise ValueError(f"field must be [C, D, H, W], got {field.shape}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_zyx must be [N, 3], got {points.shape}")

    channels, depth, height, width = field.shape
    out = np.zeros((points.shape[0], channels), dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    in_bounds = (
        finite
        & (points[:, 0] >= 0.0) & (points[:, 0] <= depth - 1)
        & (points[:, 1] >= 0.0) & (points[:, 1] <= height - 1)
        & (points[:, 2] >= 0.0) & (points[:, 2] <= width - 1)
    )
    if not in_bounds.any():
        return out, in_bounds

    idx = np.flatnonzero(in_bounds)
    p = points[idx]
    z0 = np.floor(p[:, 0]).astype(np.int64)
    y0 = np.floor(p[:, 1]).astype(np.int64)
    x0 = np.floor(p[:, 2]).astype(np.int64)
    z1 = np.minimum(z0 + 1, depth - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)

    dz = (p[:, 0] - z0).astype(np.float32)
    dy = (p[:, 1] - y0).astype(np.float32)
    dx = (p[:, 2] - x0).astype(np.float32)

    c000 = field[:, z0, y0, x0].T
    c001 = field[:, z0, y0, x1].T
    c010 = field[:, z0, y1, x0].T
    c011 = field[:, z0, y1, x1].T
    c100 = field[:, z1, y0, x0].T
    c101 = field[:, z1, y0, x1].T
    c110 = field[:, z1, y1, x0].T
    c111 = field[:, z1, y1, x1].T

    wx0 = (1.0 - dx)[:, None]
    wx1 = dx[:, None]
    wy0 = (1.0 - dy)[:, None]
    wy1 = dy[:, None]
    wz0 = (1.0 - dz)[:, None]
    wz1 = dz[:, None]

    c00 = c000 * wx0 + c001 * wx1
    c01 = c010 * wx0 + c011 * wx1
    c10 = c100 * wx0 + c101 * wx1
    c11 = c110 * wx0 + c111 * wx1
    c0 = c00 * wy0 + c01 * wy1
    c1 = c10 * wy0 + c11 * wy1
    out[idx] = c0 * wz0 + c1 * wz1
    return out, in_bounds


def _sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def integrate_streamlines_from_edge(avg_group, edge_zyx, window_min, zarr_root):
    velocity = _require_avg_field(avg_group, "velocity_dir", 3)
    attract = _require_avg_field(avg_group, "surface_attract", 3)
    validity = _require_avg_field(avg_group, "trace_validity", 1)

    edge = np.asarray(edge_zyx, dtype=np.float32)
    edge_valid = np.isfinite(edge).all(axis=1) & ~(edge == -1).all(axis=1)
    seeds_world = edge[edge_valid]
    if seeds_world.shape[0] == 0:
        raise RuntimeError("No valid edge points available for streamline integration.")

    step_sizes, requested_distance, target_distance = _integration_step_sizes()
    window_min = np.asarray(window_min, dtype=np.float32)
    points_local = seeds_world - window_min[None, :]

    traces_local = np.zeros((len(step_sizes) + 1, seeds_world.shape[0], 3), dtype=np.float32)
    active_mask = np.zeros((len(step_sizes) + 1, seeds_world.shape[0]), dtype=bool)
    traces_local[0] = points_local
    active = np.ones((seeds_world.shape[0],), dtype=bool)
    _, seed_in_bounds = _sample_field_zyx(validity, points_local)
    active &= seed_in_bounds
    active_mask[0] = active

    for step_idx, step_size in enumerate(step_sizes, start=1):
        next_points = points_local.copy()
        active_idx = np.flatnonzero(active)
        if active_idx.size > 0:
            velocity_samples, velocity_in_bounds = _sample_field_zyx(velocity, points_local[active_idx])
            speed = np.linalg.norm(velocity_samples, axis=1)
            usable = velocity_in_bounds & np.isfinite(speed) & (speed > 1e-6)
            if usable.any():
                candidate_idx = active_idx[usable]
                unit_velocity = velocity_samples[usable] / speed[usable, None]
                candidate = points_local[candidate_idx] + float(step_size) * unit_velocity
                if USE_SURFACE_ATTRACT:
                    attract_samples, attract_in_bounds = _sample_field_zyx(attract, candidate)
                    attract_norm = np.linalg.norm(attract_samples, axis=1)
                    if SURFACE_ATTRACT_MAX_CORRECTION is not None:
                        max_corr = float(SURFACE_ATTRACT_MAX_CORRECTION)
                        if max_corr >= 0.0:
                            scale = np.minimum(1.0, max_corr / np.maximum(attract_norm, 1e-6))
                            attract_samples = attract_samples * scale[:, None]
                    candidate = candidate + float(SURFACE_ATTRACT_WEIGHT) * attract_samples
                    usable_candidate = attract_in_bounds
                else:
                    usable_candidate = np.ones((candidate.shape[0],), dtype=bool)

                validity_logits, valid_in_bounds = _sample_field_zyx(validity, candidate)
                validity_prob = _sigmoid_np(validity_logits[:, 0])
                candidate_active = (
                    usable_candidate
                    & valid_in_bounds
                    & np.isfinite(candidate).all(axis=1)
                    & (validity_prob >= float(TRACE_VALIDITY_THRESHOLD))
                )
                next_points[candidate_idx[candidate_active]] = candidate[candidate_active]
                active[candidate_idx] = candidate_active
            active[active_idx[~usable]] = False

        points_local = next_points
        traces_local[step_idx] = points_local
        active_mask[step_idx] = active

    traces_world = traces_local + window_min[None, None, :]
    integration = zarr_root.require_group("streamline_integration")
    for name in ("seed_zyx", "points_zyx", "active_mask", "endpoint_zyx", "endpoint_active"):
        if name in integration:
            del integration[name]
    integration.create_dataset("seed_zyx", data=seeds_world.astype(np.float32), chunks=(min(seeds_world.shape[0], 4096), 3))
    integration.create_dataset("points_zyx", data=traces_world.astype(np.float32), chunks=(1, min(seeds_world.shape[0], 4096), 3))
    integration.create_dataset("active_mask", data=active_mask.astype(np.uint8), chunks=(1, min(seeds_world.shape[0], 4096)))
    integration.create_dataset("endpoint_zyx", data=traces_world[-1].astype(np.float32), chunks=(min(seeds_world.shape[0], 4096), 3))
    integration.create_dataset("endpoint_active", data=active.astype(np.uint8), chunks=(min(seeds_world.shape[0], 4096),))
    integration.attrs["tifxyz_voxel_step"] = float(TIFXYZ_VOXEL_STEP)
    integration.attrs["tifxyz_steps"] = int(TIFXYZ_STEPS)
    integration.attrs["integration_steps"] = int(len(step_sizes))
    integration.attrs["integration_step_size"] = float(INTEGRATION_STEP_SIZE)
    integration.attrs["requested_distance"] = requested_distance
    integration.attrs["target_distance"] = target_distance
    integration.attrs["actual_step_sizes"] = [float(v) for v in step_sizes]
    integration.attrs["trace_validity_threshold"] = float(TRACE_VALIDITY_THRESHOLD)
    integration.attrs["use_surface_attract"] = bool(USE_SURFACE_ATTRACT)
    integration.attrs["surface_attract_weight"] = float(SURFACE_ATTRACT_WEIGHT)
    integration.attrs["surface_attract_max_correction"] = (
        None if SURFACE_ATTRACT_MAX_CORRECTION is None else float(SURFACE_ATTRACT_MAX_CORRECTION)
    )
    integration.attrs["active_endpoints"] = int(active.sum())
    integration.attrs["seed_count"] = int(seeds_world.shape[0])
    return {
        "group": integration,
        "seed_count": int(seeds_world.shape[0]),
        "active_endpoints": int(active.sum()),
        "requested_distance": requested_distance,
        "target_distance": target_distance,
        "step_sizes": step_sizes,
    }


def run_rowcol_bbox_inference(model, model_config, volume_array, voxelized_bboxes, crop_size, cond_direction, zarr_root):
    window_min, window_shape = _merged_window_from_bboxes([item["bbox"] for item in voxelized_bboxes])
    zarr_root.attrs["window_min_zyx"] = [int(v) for v in window_min]
    zarr_root.attrs["window_shape_zyx"] = [int(v) for v in window_shape]
    zarr_root.attrs["crop_size_zyx"] = [int(v) for v in crop_size]
    zarr_root.attrs["cond_direction"] = str(cond_direction)
    zarr_root.attrs["grow_direction"] = str(GROW_DIRECTION)

    mixed_precision = str(model_config.get("mixed_precision", "no")).lower()
    amp_enabled = mixed_precision in ("bf16", "fp16", "float16")
    amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16

    model.eval()
    batch_starts = range(0, len(voxelized_bboxes), int(BATCH_SIZE))
    for start in tqdm(batch_starts, desc="Row/col bbox inference", unit="batch"):
        batch = voxelized_bboxes[start:start + int(BATCH_SIZE)]
        inputs = _prepare_rowcol_inputs(volume_array, batch, crop_size, cond_direction)
        with torch.inference_mode():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
        if not isinstance(outputs, dict):
            outputs = {"output": outputs}
        for output_name, output_value in outputs.items():
            output_value = _collapse_output(output_value)
            if output_value is None:
                continue
            _accumulate_output(
                zarr_root,
                output_name,
                output_value.detach().float().cpu().numpy(),
                batch,
                window_min,
                window_shape,
                crop_size,
            )

    return _finalize_averages(zarr_root)


def show_streamline_geometry_napari(points_zyx, edge_zyx, bboxes, voxelized_bboxes, integration_group=None):
    try:
        import napari
    except Exception as exc:
        raise RuntimeError("napari is not available.") from exc

    viewer = napari.Viewer(ndisplay=3)

    points = np.asarray(points_zyx, dtype=np.float32)
    downsample = max(1, int(POINT_DOWNSAMPLE))
    if downsample > 1:
        points = points[::downsample]
    if points.shape[0] > 0:
        viewer.add_points(
            points,
            name="tifxyz_points",
            size=float(POINT_SIZE),
            face_color=[0.0, 0.8, 0.2, 0.8],
        )

    edge = np.asarray(edge_zyx, dtype=np.float32)
    edge_valid = np.isfinite(edge).all(axis=1) & ~(edge == -1).all(axis=1)
    edge = edge[edge_valid]
    if edge.shape[0] > 0:
        viewer.add_points(
            edge,
            name="conditioning_edge",
            size=float(EDGE_POINT_SIZE),
            face_color=[1.0, 0.2, 0.0, 1.0],
        )

    voxel_points = []
    for item in voxelized_bboxes or ():
        voxels = np.asarray(item.get("voxels"))
        if voxels.ndim != 3 or not voxels.any():
            continue
        z_min, _, y_min, _, x_min, _ = item["bbox"]
        coords = np.argwhere(voxels > 0).astype(np.float32, copy=False)
        coords += np.asarray([z_min, y_min, x_min], dtype=np.float32)
        voxel_points.append(coords)
    if voxel_points:
        merged_voxel_points = np.concatenate(voxel_points, axis=0)
        merged_voxel_points = np.unique(merged_voxel_points, axis=0)
        viewer.add_points(
            merged_voxel_points,
            name="upsampled_voxelized_surface",
            size=1.0,
            face_color=[1.0, 0.85, 0.05, 0.65],
        )

    segments = []
    for bbox in bboxes:
        segments.extend(bbox_wireframe_segments(bbox))
    if segments:
        viewer.add_shapes(
            segments,
            shape_type="path",
            edge_color=[0.0, 0.45, 1.0, 0.9],
            edge_width=1,
            face_color="transparent",
            name="cond_edge_bboxes",
            opacity=0.9,
        )

    if integration_group is not None and "points_zyx" in integration_group:
        traces = np.asarray(integration_group["points_zyx"], dtype=np.float32)
        active_mask = np.asarray(integration_group["active_mask"], dtype=bool)
        path_segments = []
        if traces.ndim == 3 and traces.shape[0] > 1:
            for point_idx in range(traces.shape[1]):
                valid_steps = np.flatnonzero(active_mask[:, point_idx])
                if valid_steps.size < 2:
                    continue
                path = traces[valid_steps, point_idx, :]
                if np.isfinite(path).all():
                    path_segments.append(path)
        if path_segments:
            viewer.add_shapes(
                path_segments,
                shape_type="path",
                edge_color=[1.0, 0.0, 0.8, 0.9],
                edge_width=1,
                face_color="transparent",
                name="integrated_streamlines",
                opacity=0.9,
            )

    napari.run()


def main():
    if DEVICE != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("infer_streamline.py is configured for CUDA-only inference.")

    model, model_config = load_checkpoint(CHECKPOINT_PATH)
    model.to(DEVICE)
    model = torch.compile(model)
    crop_size = _crop_size_from_config(model_config)

    stored_zyxs, valid = _load_surface_zyx(TIFXYZ_PATH)
    points_zyx = stored_zyxs[valid]
    if points_zyx.size == 0:
        raise RuntimeError("No valid tifxyz points found.")

    cond_direction, _ = _get_growth_context(GROW_DIRECTION)
    bboxes, edge = get_cond_edge_bboxes(
        stored_zyxs,
        cond_direction,
        crop_size,
        overlap_frac=OVERLAP_FRAC,
        cond_valid=valid,
    )
    surface = read_tifxyz(TIFXYZ_PATH)
    voxelized_bboxes = upsample_voxelize_tifxyz_surface_in_bboxes(
        surface,
        bboxes,
        crop_size,
        stored_zyxs=stored_zyxs,
        valid=valid,
    )
    volume_array = _open_volume_array(VOLUME_PATH, VOLUME_SCALE)
    output_path = _output_zarr_path()
    output_root = zarr.open_group(str(output_path), mode="w")
    output_group = run_rowcol_bbox_inference(
        model,
        model_config,
        volume_array,
        voxelized_bboxes,
        crop_size,
        cond_direction,
        output_root,
    )
    integration_result = integrate_streamlines_from_edge(
        output_group,
        edge,
        np.asarray(output_root.attrs["window_min_zyx"], dtype=np.float32),
        output_root,
    )

    print(f"tifxyz_path: {TIFXYZ_PATH}")
    print(f"volume_path: {VOLUME_PATH}")
    print(f"volume_scale: {int(VOLUME_SCALE)}")
    print(f"checkpoint_path: {CHECKPOINT_PATH}")
    print(f"output_zarr_path: {output_path}")
    print(f"grow_direction: {GROW_DIRECTION}")
    print(f"cond_direction: {cond_direction}")
    print(f"crop_size: {tuple(crop_size)}")
    print(f"overlap_frac: {float(OVERLAP_FRAC)}")
    print(f"valid_points: {int(points_zyx.shape[0])}")
    print(f"bboxes: {len(bboxes)}")
    print(f"voxelized_bbox_voxels: {sum(int(item['voxels'].sum()) for item in voxelized_bboxes)}")
    print(f"tifxyz_voxel_step: {float(TIFXYZ_VOXEL_STEP)}")
    print(f"tifxyz_steps: {int(TIFXYZ_STEPS)}")
    print(f"integration_steps: {len(integration_result['step_sizes'])}")
    print(f"integration_step_size: {float(INTEGRATION_STEP_SIZE)}")
    print(f"integration_requested_distance: {integration_result['requested_distance']}")
    print(f"integration_target_distance: {integration_result['target_distance']}")
    print(f"integration_actual_steps: {integration_result['step_sizes']}")
    print(f"integration_seed_count: {integration_result['seed_count']}")
    print(f"integration_active_endpoints: {integration_result['active_endpoints']}")

    if SHOW_NAPARI_GEOMETRY:
        show_streamline_geometry_napari(
            points_zyx,
            edge,
            bboxes,
            voxelized_bboxes=voxelized_bboxes,
            integration_group=integration_result["group"],
        )


if __name__ == "__main__":
    main()
