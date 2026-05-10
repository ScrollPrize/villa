from pathlib import Path
import tempfile
from datetime import datetime
import hashlib
import re
import shutil

import numpy as np
import torch
import torch.nn.functional as F
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
from vesuvius.neural_tracing.heatmap_single_point.tifxyz import save_tifxyz
from vesuvius.neural_tracing.inference.napari_helpers import show_streamline_geometry_napari
from vesuvius.neural_tracing.nets.models import load_checkpoint
from vesuvius.tifxyz import read_tifxyz


TIFXYZ_PATH = "/home/sean/Documents/volpkgs/s1_2um.volpkg/traces/test_in_smaller"
VOLUME_PATH = "s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"
VOLUME_SCALE = 0
VOLUME_CACHE_DIR = "/tmp/vesuvius-volume-cache"
CHECKPOINT_PATH = "/home/sean/Downloads/ckpt_090000.pth"

# Keep None unless intentionally debugging shape mismatches. The default reads
# the checkpoint crop size so inference matches the model patch size.
CROP_SIZE = None
BATCH_SIZE = 2
DEVICE = "cuda"

OVERLAP_FRAC = 0.25
OUTPUT_ZARR_PATH = None
OUTPUT_TIFXYZ_DIR = "/home/sean/Documents/volpkgs/s1_2um.volpkg/traces"
OUTPUT_TIFXYZ_VOXEL_SIZE_UM = "2.24"
MERGE_OUTPUTS_CHUNK_SIZE = 256
MERGE_OUTPUTS_MMAP_DIR = "/tmp/streamline_tmp_outputs/"
GEOMEDIAN_VECTOR_OUTPUTS = ("velocity_dir", "surface_attract")
GEOMEDIAN_MAX_ITER = 8
GEOMEDIAN_EPS = 1e-6
USE_TTA = False
TTA_FLIP_AXES = ((2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4))
SHOW_NAPARI = False

GROW_DIRECTION = "left"

TIFXYZ_VOXEL_STEP = 20.0
TIFXYZ_STEPS = 4
NUM_ITERATIONS = 5
# If false, intermediate iterations are written to a temporary tifxyz directory
# only long enough to feed the next iteration; the final iteration is saved.
SAVE_EACH_ITERATION_TIFXYZ = True
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


def _output_tifxyz_dir():
    if OUTPUT_TIFXYZ_DIR is not None:
        return Path(OUTPUT_TIFXYZ_DIR)
    return Path(TIFXYZ_PATH).parent


def _output_tifxyz_uuid(timestamp=None, iteration_idx=None, total_iterations=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_name = Path(TIFXYZ_PATH).name
    suffix = ""
    if total_iterations is not None and int(total_iterations) > 1:
        suffix = f"_iter{int(iteration_idx):02d}of{int(total_iterations):02d}"
    return f"{input_name}_{timestamp}_{GROW_DIRECTION}_{int(TIFXYZ_STEPS)}steps{suffix}"


def _output_tifxyz_voxel_size_um():
    if OUTPUT_TIFXYZ_VOXEL_SIZE_UM is not None:
        return float(OUTPUT_TIFXYZ_VOXEL_SIZE_UM)
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)um", str(VOLUME_PATH))
    if match is not None:
        return float(match.group(1))
    return 8.24


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
    batch_size = len(voxelized_batch)
    crop_shape = tuple(int(v) for v in crop_size)
    inputs_np = np.empty((batch_size, 2, *crop_shape), dtype=np.float32)
    crop_size_arr = np.asarray(crop_size, dtype=np.int64)

    for batch_idx, item in enumerate(voxelized_batch):
        bbox = item["bbox"]
        min_corner = _bbox_min_corner(bbox)
        max_corner = min_corner + crop_size_arr
        inputs_np[batch_idx, 0] = _read_volume_crop(volume_array, crop_size, min_corner, max_corner)
        inputs_np[batch_idx, 1] = np.asarray(item["voxels"], dtype=np.float32)

    inputs = torch.from_numpy(inputs_np).to(device=DEVICE, dtype=torch.float32)
    direction = make_growth_direction_tensor(
        [cond_direction] * batch_size,
        crop_size,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    return torch.cat([inputs, direction], dim=1)


def _tta_variants():
    variants = [()]
    if bool(USE_TTA):
        variants.extend(tuple(int(axis) for axis in axes) for axes in TTA_FLIP_AXES)
    return variants


def _run_model_once(model, inputs, amp_enabled, amp_dtype):
    if amp_enabled:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(inputs)
    else:
        outputs = model(inputs)
    if not isinstance(outputs, dict):
        outputs = {"output": outputs}
    return outputs


def _flip_tta_inputs(inputs, flip_axes):
    if not flip_axes:
        return inputs
    flipped = torch.flip(inputs, dims=tuple(int(axis) for axis in flip_axes)).clone()
    if flipped.shape[1] >= 6:
        if 4 in flip_axes:
            flipped[:, [2, 3]] = flipped[:, [3, 2]]
        if 3 in flip_axes:
            flipped[:, [4, 5]] = flipped[:, [5, 4]]
    return flipped


def _restore_tta_output(output_name, output_value, flip_axes):
    output_value = _collapse_output(output_value)
    if output_value is None:
        return None
    if not flip_axes:
        return output_value
    restored = torch.flip(output_value, dims=tuple(int(axis) for axis in flip_axes))
    if restored.ndim == 5 and restored.shape[1] == 3 and str(output_name) in set(GEOMEDIAN_VECTOR_OUTPUTS):
        for axis in flip_axes:
            component = int(axis) - 2
            if 0 <= component < 3:
                restored[:, component] = -restored[:, component]
    return restored


def _run_model_tta(model, inputs, amp_enabled, amp_dtype):
    variants = _tta_variants()
    accum = {}
    counts = {}
    for flip_axes in variants:
        model_inputs = _flip_tta_inputs(inputs, flip_axes)
        outputs = _run_model_once(model, model_inputs, amp_enabled, amp_dtype)
        for output_name, output_value in outputs.items():
            restored = _restore_tta_output(output_name, output_value, flip_axes)
            if restored is None:
                continue
            restored = restored.float()
            if output_name in accum:
                accum[output_name] = accum[output_name] + restored
                counts[output_name] += 1
            else:
                accum[output_name] = restored
                counts[output_name] = 1
    return {
        output_name: output_sum / float(counts[output_name])
        for output_name, output_sum in accum.items()
    }


def _slice_len(s):
    return int(s.stop) - int(s.start)


def _chunk_slices_for_region(start, end, chunks):
    z0, y0, x0 = (int(v) for v in start)
    z1, y1, x1 = (int(v) for v in end)
    cz, cy, cx = (int(v) for v in chunks)
    for zz in range(z0 // cz, (z1 - 1) // cz + 1):
        zs = slice(max(z0, zz * cz), min(z1, (zz + 1) * cz))
        for yy in range(y0 // cy, (y1 - 1) // cy + 1):
            ys = slice(max(y0, yy * cy), min(y1, (yy + 1) * cy))
            for xx in range(x0 // cx, (x1 - 1) // cx + 1):
                xs = slice(max(x0, xx * cx), min(x1, (xx + 1) * cx))
                yield (zz, yy, xx), (zs, ys, xs)


def _is_geomedian_vector_output(output_name, channels):
    return int(channels) == 3 and str(output_name) in set(GEOMEDIAN_VECTOR_OUTPUTS)


def _geometric_median_vectors(samples, valid, max_iter=GEOMEDIAN_MAX_ITER, eps=GEOMEDIAN_EPS):
    samples = np.asarray(samples, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if samples.ndim != 3 or samples.shape[1] != 3:
        raise ValueError(f"samples must be [K, 3, N], got {samples.shape}")
    counts = valid.sum(axis=0)
    out = np.zeros((3, samples.shape[2]), dtype=np.float32)
    active = counts > 0
    if not bool(active.any()):
        return out

    weights0 = valid.astype(np.float32, copy=False)
    current = (samples * weights0[:, None, :]).sum(axis=0)
    current[:, active] /= counts[active].astype(np.float32, copy=False)[None, :]
    for _ in range(int(max_iter)):
        diff = samples - current[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float32), dtype=np.float32)
        weights = np.where(valid, 1.0 / np.maximum(dist, float(eps)), 0.0).astype(np.float32, copy=False)
        denom = weights.sum(axis=0)
        update = (samples * weights[:, None, :]).sum(axis=0)
        can_update = denom > 0.0
        update[:, can_update] /= denom[can_update][None, :]
        current[:, can_update] = update[:, can_update]
    out[:, active] = current[:, active]
    return out


class _SparseChunkOutputMerger:
    def __init__(self, root, window_min, window_shape, crop_size, mmap_dir=None):
        self.root = root
        self.window_min = np.asarray(window_min, dtype=np.int64)
        self.window_shape = tuple(int(v) for v in window_shape)
        self.crop_size_arr = np.asarray(crop_size, dtype=np.int64)
        self.chunks_3d = tuple(min(int(MERGE_OUTPUTS_CHUNK_SIZE), int(v)) for v in self.window_shape)
        self.current_bytes = 0
        if mmap_dir is not None:
            Path(mmap_dir).mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="infer_streamline_merge_", dir=mmap_dir)
        self._tmpdir_path = Path(self._tmpdir.name)
        self.channels = {}
        self.sum_chunks = {}
        self.count_chunks = {}
        self.counted_regions = set()
        self.vector_sample_chunks = {}
        self.vector_count_chunks = {}

    def _reserve_bytes(self, n_bytes):
        self.current_bytes += int(n_bytes)

    def _chunk_path(self, kind, key):
        output_name, zz, yy, xx = key
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(output_name))
        digest = hashlib.blake2b(str(output_name).encode("utf-8"), digest_size=6).hexdigest()
        return self._tmpdir_path / f"{safe_name}_{digest}_{int(zz)}_{int(yy)}_{int(xx)}_{kind}.npy"

    def _zeros_chunk(self, kind, key, shape, dtype):
        path = self._chunk_path(kind, key)
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        arr[...] = 0
        return arr

    def _ensure_output(self, output_name, channels):
        channels = int(channels)
        previous_channels = self.channels.get(output_name)
        if previous_channels is not None and previous_channels != channels:
            raise ValueError(
                f"Output {output_name!r} channel count changed from {previous_channels} to {channels}."
            )
        self.channels[output_name] = channels

    def _chunk_region(self, chunk_key):
        zz, yy, xx = (int(v) for v in chunk_key)
        z0 = zz * self.chunks_3d[0]
        y0 = yy * self.chunks_3d[1]
        x0 = xx * self.chunks_3d[2]
        return (
            slice(z0, min(z0 + self.chunks_3d[0], self.window_shape[0])),
            slice(y0, min(y0 + self.chunks_3d[1], self.window_shape[1])),
            slice(x0, min(x0 + self.chunks_3d[2], self.window_shape[2])),
        )

    def _ensure_chunk(self, output_name, chunk_key):
        sum_key = (output_name, *chunk_key)
        if sum_key in self.sum_chunks:
            return self.sum_chunks[sum_key], self.count_chunks[chunk_key]

        channels = self.channels[output_name]
        region = self._chunk_region(chunk_key)
        chunk_shape = tuple(_slice_len(s) for s in region)
        sum_shape = (channels, *chunk_shape)
        sum_chunk = self._zeros_chunk("sum", sum_key, sum_shape, np.float32)
        self._reserve_bytes(int(np.prod(sum_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize)
        count_chunk = self.count_chunks.get(chunk_key)
        if count_chunk is None:
            count_chunk = self._zeros_chunk("count", ("count", *chunk_key), chunk_shape, np.uint32)
            self.count_chunks[chunk_key] = count_chunk
            self._reserve_bytes(int(np.prod(chunk_shape, dtype=np.int64)) * np.dtype(np.uint32).itemsize)
        self.sum_chunks[sum_key] = sum_chunk
        return sum_chunk, count_chunk

    def _ensure_count_chunk(self, chunk_key):
        region = self._chunk_region(chunk_key)
        chunk_shape = tuple(_slice_len(s) for s in region)
        count_chunk = self.count_chunks.get(chunk_key)
        if count_chunk is None:
            count_chunk = self._zeros_chunk("count", ("count", *chunk_key), chunk_shape, np.uint32)
            self.count_chunks[chunk_key] = count_chunk
            self._reserve_bytes(int(np.prod(chunk_shape, dtype=np.int64)) * np.dtype(np.uint32).itemsize)
        return count_chunk

    def _ensure_vector_chunks(self, output_name, chunk_key):
        sample_key = (output_name, *chunk_key)
        count_chunk = self.vector_count_chunks.get(sample_key)
        if count_chunk is None:
            region = self._chunk_region(chunk_key)
            chunk_shape = tuple(_slice_len(s) for s in region)
            count_chunk = self._zeros_chunk("vector_count", sample_key, chunk_shape, np.uint16)
            self.vector_count_chunks[sample_key] = count_chunk
            self.vector_sample_chunks[sample_key] = []
            self._reserve_bytes(int(np.prod(chunk_shape, dtype=np.int64)) * np.dtype(np.uint16).itemsize)
        return self.vector_sample_chunks[sample_key], count_chunk

    def _ensure_vector_slot(self, output_name, chunk_key, slot_idx):
        sample_key = (output_name, *chunk_key)
        slots, _ = self._ensure_vector_chunks(output_name, chunk_key)
        while len(slots) <= int(slot_idx):
            channels = self.channels[output_name]
            region = self._chunk_region(chunk_key)
            chunk_shape = tuple(_slice_len(s) for s in region)
            slot_shape = (channels, *chunk_shape)
            slot = self._zeros_chunk(f"vector_sample_{len(slots)}", sample_key, slot_shape, np.float32)
            slots.append(slot)
            self._reserve_bytes(int(np.prod(slot_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize)
        return slots[int(slot_idx)]

    def accumulate(self, output_name, output_batch, voxelized_batch):
        output_batch = np.asarray(output_batch, dtype=np.float32)
        if output_batch.ndim != 5:
            raise ValueError(f"Output {output_name!r} must be [B, C, D, H, W], got {output_batch.shape}")
        if tuple(output_batch.shape[2:]) != tuple(self.crop_size_arr.tolist()):
            raise ValueError(
                f"Output {output_name!r} spatial shape {output_batch.shape[2:]} "
                f"does not match crop_size {tuple(self.crop_size_arr.tolist())}"
            )
        self._ensure_output(output_name, output_batch.shape[1])
        use_geomedian = _is_geomedian_vector_output(output_name, output_batch.shape[1])

        for batch_idx, item in enumerate(voxelized_batch):
            start = _bbox_min_corner(item["bbox"]) - self.window_min
            end = start + self.crop_size_arr
            for chunk_key, region in _chunk_slices_for_region(start, end, self.chunks_3d):
                crop_region = tuple(
                    slice(int(region_axis.start) - int(start_axis), int(region_axis.stop) - int(start_axis))
                    for region_axis, start_axis in zip(region, start)
                )
                chunk_region = self._chunk_region(chunk_key)
                local_region = tuple(
                    slice(int(region_axis.start) - int(chunk_axis.start), int(region_axis.stop) - int(chunk_axis.start))
                    for region_axis, chunk_axis in zip(region, chunk_region)
                )
                output_region = output_batch[batch_idx][(slice(None), *crop_region)]
                if use_geomedian:
                    count_chunk = self._ensure_count_chunk(chunk_key)
                    slots, vector_count_chunk = self._ensure_vector_chunks(output_name, chunk_key)
                    local_counts = np.asarray(vector_count_chunk[local_region], dtype=np.uint16)
                    for slot_idx in np.unique(local_counts):
                        slot_idx = int(slot_idx)
                        slot = self._ensure_vector_slot(output_name, chunk_key, slot_idx)
                        mask = local_counts == slot_idx
                        slot_view = slot[(slice(None), *local_region)]
                        slot_view[:, mask] = output_region[:, mask]
                    vector_count_chunk[local_region] += np.uint16(1)
                else:
                    sum_chunk, count_chunk = self._ensure_chunk(output_name, chunk_key)
                    sum_chunk[(slice(None), *local_region)] += output_region
                count_key = (tuple(int(v) for v in item["bbox"]), *chunk_key)
                if count_key not in self.counted_regions:
                    count_chunk[local_region] += np.uint32(1)
                    self.counted_regions.add(count_key)

    def _finalize_geomedian_output(self, avg, output_name):
        sample_keys = sorted(
            key for key in self.vector_sample_chunks
            if key[0] == output_name
        )
        block_voxels = 262144
        for key in tqdm(sample_keys, desc=f"Finalizing {output_name} geomedian", unit="chunk"):
            _, zz, yy, xx = key
            slots = self.vector_sample_chunks[key]
            count_chunk = self.vector_count_chunks[key]
            region = self._chunk_region((zz, yy, xx))
            counts_flat = np.asarray(count_chunk).reshape(-1)
            valid_flat = counts_flat > 0
            if not bool(valid_flat.any()):
                continue

            out_flat = np.zeros((3, counts_flat.shape[0]), dtype=np.float32)
            max_count = int(counts_flat.max())
            for block_start in range(0, counts_flat.shape[0], block_voxels):
                block_end = min(block_start + block_voxels, counts_flat.shape[0])
                block_counts = counts_flat[block_start:block_end]
                if not bool((block_counts > 0).any()):
                    continue
                block_k = min(max_count, len(slots))
                samples = np.empty((block_k, 3, block_end - block_start), dtype=np.float32)
                for slot_idx in range(block_k):
                    samples[slot_idx] = np.asarray(slots[slot_idx]).reshape(3, -1)[:, block_start:block_end]
                valid = np.arange(block_k, dtype=np.uint16)[:, None] < block_counts[None, :]
                out_flat[:, block_start:block_end] = _geometric_median_vectors(samples, valid)

            avg[(slice(None), *region)] = out_flat.reshape((3, *count_chunk.shape))

    def finalize(self):
        try:
            outputs = self.root.require_group("outputs")
            avg_group = outputs.require_group("avg")
            avg_group.attrs["window_min_zyx"] = self.root.attrs.get("window_min_zyx", [0, 0, 0])
            avg_group.attrs["window_shape_zyx"] = self.root.attrs.get("window_shape_zyx", None)
            avg_group.attrs["merge_backing"] = "mmap"
            avg_group.attrs["merge_sparse_bytes"] = int(self.current_bytes)

            for output_name, channels in self.channels.items():
                if output_name in avg_group:
                    del avg_group[output_name]

                avg = avg_group.create_dataset(
                    output_name,
                    shape=(int(channels), *self.window_shape),
                    chunks=(int(channels), *self.chunks_3d),
                    dtype="f4",
                    fill_value=0.0,
                )
                if _is_geomedian_vector_output(output_name, channels):
                    avg.attrs[f"{output_name}_merge_method"] = "vector_geometric_median"
                    self._finalize_geomedian_output(avg, output_name)
                    continue
                avg.attrs[f"{output_name}_merge_method"] = "mean"
                output_chunk_keys = sorted(
                    key for key in self.sum_chunks
                    if key[0] == output_name
                )
                for key in tqdm(output_chunk_keys, desc=f"Finalizing {output_name}", unit="chunk"):
                    _, zz, yy, xx = key
                    sum_chunk = self.sum_chunks[key]
                    count_chunk = self.count_chunks[(zz, yy, xx)]
                    region = self._chunk_region((zz, yy, xx))

                    count = count_chunk.astype(np.float32, copy=False)
                    valid = count > 0
                    if not bool(valid.any()):
                        continue
                    avg_chunk = np.empty(sum_chunk.shape, dtype=np.float32)
                    if bool(valid.all()):
                        np.divide(sum_chunk, count[None, ...], out=avg_chunk)
                    else:
                        avg_chunk.fill(0.0)
                        np.divide(sum_chunk, count[None, ...], out=avg_chunk, where=valid[None, ...])
                    avg[(slice(None), *region)] = avg_chunk
            return avg_group
        finally:
            if self._tmpdir is not None:
                self._close_memmap_chunks()
                self.sum_chunks.clear()
                self.count_chunks.clear()
                self.counted_regions.clear()
                self.vector_sample_chunks.clear()
                self.vector_count_chunks.clear()
                self._tmpdir.cleanup()
                self._tmpdir = None

    def _close_memmap_chunks(self):
        seen = set()
        vector_slots = [
            slot
            for slots in self.vector_sample_chunks.values()
            for slot in slots
        ]
        for chunk in (
            list(self.sum_chunks.values())
            + list(self.count_chunks.values())
            + list(self.vector_count_chunks.values())
            + vector_slots
        ):
            chunk_id = id(chunk)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if hasattr(chunk, "flush"):
                chunk.flush()
            mmap_obj = getattr(chunk, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()


def _require_avg_field(avg_group, name, channels):
    if name not in avg_group:
        raise KeyError(f"Missing required averaged output field {name!r}.")
    field = avg_group[name]
    if len(field.shape) != 4 or field.shape[0] != int(channels):
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


class _TorchStreamlineFieldSampler:
    def __init__(self, velocity, attract, validity, device):
        self.device = torch.device(device)
        fields = np.concatenate(
            [
                np.asarray(velocity, dtype=np.float32),
                np.asarray(attract, dtype=np.float32),
                np.asarray(validity, dtype=np.float32),
            ],
            axis=0,
        )
        self.field = torch.from_numpy(fields).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        _, channels, depth, height, width = self.field.shape
        if channels != 7:
            raise RuntimeError(f"Expected 7 streamline field channels, got {channels}")
        self.depth = int(depth)
        self.height = int(height)
        self.width = int(width)

    def sample(self, points_zyx, channel_start, channel_count):
        if points_zyx.ndim != 2 or points_zyx.shape[1] != 3:
            raise ValueError(f"points_zyx must be [N, 3], got {tuple(points_zyx.shape)}")

        points = points_zyx.to(device=self.device, dtype=torch.float32)
        finite = torch.isfinite(points).all(dim=1)
        safe_points = torch.where(finite[:, None], points, torch.zeros_like(points))
        in_bounds = (
            finite
            & (safe_points[:, 0] >= 0.0) & (safe_points[:, 0] <= self.depth - 1)
            & (safe_points[:, 1] >= 0.0) & (safe_points[:, 1] <= self.height - 1)
            & (safe_points[:, 2] >= 0.0) & (safe_points[:, 2] <= self.width - 1)
        )
        if points.shape[0] == 0:
            return points.new_zeros((0, int(channel_count))), in_bounds

        grid = points.new_empty((1, points.shape[0], 1, 1, 3))
        if self.width > 1:
            grid[0, :, 0, 0, 0] = 2.0 * safe_points[:, 2] / float(self.width - 1) - 1.0
        else:
            grid[0, :, 0, 0, 0] = 0.0
        if self.height > 1:
            grid[0, :, 0, 0, 1] = 2.0 * safe_points[:, 1] / float(self.height - 1) - 1.0
        else:
            grid[0, :, 0, 0, 1] = 0.0
        if self.depth > 1:
            grid[0, :, 0, 0, 2] = 2.0 * safe_points[:, 0] / float(self.depth - 1) - 1.0
        else:
            grid[0, :, 0, 0, 2] = 0.0

        field = self.field[:, int(channel_start):int(channel_start) + int(channel_count)]
        sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.view(int(channel_count), -1).T.contiguous()
        return sampled, in_bounds

    def sample_velocity(self, points_zyx):
        return self.sample(points_zyx, 0, 3)

    def sample_attract(self, points_zyx):
        return self.sample(points_zyx, 3, 3)

    def sample_validity(self, points_zyx):
        return self.sample(points_zyx, 6, 1)


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
    cumulative_distance = np.zeros((len(step_sizes) + 1, seeds_world.shape[0]), dtype=np.float32)
    active_mask = np.zeros((len(step_sizes) + 1, seeds_world.shape[0]), dtype=bool)
    traces_local[0] = points_local
    sampler = _TorchStreamlineFieldSampler(velocity, attract, validity, DEVICE)
    points_t = torch.from_numpy(points_local).to(device=sampler.device, dtype=torch.float32)
    distance_t = torch.zeros((seeds_world.shape[0],), device=sampler.device, dtype=torch.float32)
    active = torch.ones((seeds_world.shape[0],), device=sampler.device, dtype=torch.bool)
    _, seed_in_bounds = sampler.sample_validity(points_t)
    active &= seed_in_bounds
    active_mask[0] = active.detach().cpu().numpy()

    step_iter = tqdm(step_sizes, desc="Integrating streamlines", unit="step")
    for step_idx, step_size in enumerate(step_iter, start=1):
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        step_iter.set_postfix(active=int(active_idx.numel()))
        if active_idx.numel() > 0:
            current_points = points_t[active_idx]
            velocity_samples, velocity_in_bounds = sampler.sample_velocity(current_points)
            speed = torch.linalg.norm(velocity_samples, dim=1)
            usable = velocity_in_bounds & torch.isfinite(speed) & (speed > 1e-6)
            if bool(usable.any().item()):
                candidate_idx = active_idx[usable]
                unit_velocity = velocity_samples[usable] / speed[usable, None]
                candidate = points_t[candidate_idx] + float(step_size) * unit_velocity
                if USE_SURFACE_ATTRACT:
                    attract_samples, attract_in_bounds = sampler.sample_attract(candidate)
                    attract_norm = torch.linalg.norm(attract_samples, dim=1)
                    if SURFACE_ATTRACT_MAX_CORRECTION is not None:
                        max_corr = float(SURFACE_ATTRACT_MAX_CORRECTION)
                        if max_corr >= 0.0:
                            scale = torch.clamp(max_corr / torch.clamp(attract_norm, min=1e-6), max=1.0)
                            attract_samples = attract_samples * scale[:, None]
                    candidate = candidate + float(SURFACE_ATTRACT_WEIGHT) * attract_samples
                    usable_candidate = attract_in_bounds
                else:
                    usable_candidate = torch.ones((candidate.shape[0],), device=sampler.device, dtype=torch.bool)

                validity_logits, validity_in_bounds = sampler.sample_validity(candidate)
                validity_prob = torch.sigmoid(validity_logits[:, 0])
                candidate_active = (
                    usable_candidate
                    & validity_in_bounds
                    & torch.isfinite(candidate).all(dim=1)
                    & (validity_prob >= float(TRACE_VALIDITY_THRESHOLD))
                )
                accepted_idx = candidate_idx[candidate_active]
                accepted_candidate = candidate[candidate_active]
                segment_lengths = torch.linalg.norm(accepted_candidate - points_t[accepted_idx], dim=1)
                distance_t[accepted_idx] = distance_t[accepted_idx] + segment_lengths
                points_t[accepted_idx] = accepted_candidate
                active[candidate_idx] = candidate_active
            active[active_idx[~usable]] = False

        traces_local[step_idx] = points_t.detach().cpu().numpy()
        cumulative_distance[step_idx] = distance_t.detach().cpu().numpy()
        active_mask[step_idx] = active.detach().cpu().numpy()

    traces_world = traces_local + window_min[None, None, :]
    integration = zarr_root.require_group("streamline_integration")
    for name in ("seed_zyx", "points_zyx", "active_mask", "cumulative_distance", "endpoint_zyx", "endpoint_active"):
        if name in integration:
            del integration[name]
    integration.create_dataset("seed_zyx", data=seeds_world.astype(np.float32), chunks=(min(seeds_world.shape[0], 4096), 3))
    integration.create_dataset("points_zyx", data=traces_world.astype(np.float32), chunks=(1, min(seeds_world.shape[0], 4096), 3))
    integration.create_dataset("active_mask", data=active_mask.astype(np.uint8), chunks=(1, min(seeds_world.shape[0], 4096)))
    integration.create_dataset("cumulative_distance", data=cumulative_distance.astype(np.float32), chunks=(1, min(seeds_world.shape[0], 4096)))
    integration.create_dataset("endpoint_zyx", data=traces_world[-1].astype(np.float32), chunks=(min(seeds_world.shape[0], 4096), 3))
    integration.create_dataset("endpoint_active", data=active.detach().cpu().numpy().astype(np.uint8), chunks=(min(seeds_world.shape[0], 4096),))
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
    integration.attrs["distance_sampling"] = "actual_arclength"
    integration.attrs["active_endpoints"] = int(active.sum().item())
    integration.attrs["seed_count"] = int(seeds_world.shape[0])
    return {
        "group": integration,
        "seed_count": int(seeds_world.shape[0]),
        "active_endpoints": int(active.sum().item()),
        "requested_distance": requested_distance,
        "target_distance": target_distance,
        "step_sizes": step_sizes,
    }


def _streamline_points_at_tifxyz_steps(integration_group):
    traces = np.asarray(integration_group["points_zyx"], dtype=np.float32)
    active_mask = np.asarray(integration_group["active_mask"], dtype=bool)
    if traces.ndim != 3 or traces.shape[-1] != 3:
        raise RuntimeError(f"Unexpected streamline points shape: {traces.shape}")
    if active_mask.shape != traces.shape[:2]:
        raise RuntimeError(f"Unexpected streamline active mask shape: {active_mask.shape}")

    tifxyz_step = float(integration_group.attrs.get("tifxyz_voxel_step", TIFXYZ_VOXEL_STEP))
    tifxyz_steps = int(integration_group.attrs.get("tifxyz_steps", TIFXYZ_STEPS))
    if tifxyz_step <= 0.0:
        raise ValueError("tifxyz_voxel_step must be > 0.")
    if tifxyz_steps <= 0:
        raise ValueError("tifxyz_steps must be > 0.")
    target_distances = tifxyz_step * np.arange(1, tifxyz_steps + 1, dtype=np.float32)

    sampled_points = np.full((tifxyz_steps, traces.shape[1], 3), -1.0, dtype=np.float32)
    sampled_active = np.zeros((tifxyz_steps, traces.shape[1]), dtype=bool)

    if "cumulative_distance" not in integration_group:
        raise RuntimeError("Missing required streamline cumulative_distance dataset.")
    cumulative_distance = np.asarray(integration_group["cumulative_distance"], dtype=np.float32)
    if cumulative_distance.shape != traces.shape[:2]:
        raise RuntimeError(
            "Unexpected streamline cumulative distance shape: "
            f"{cumulative_distance.shape}, expected {traces.shape[:2]}"
        )

    for seed_idx in range(traces.shape[1]):
        active_steps = np.flatnonzero(active_mask[:, seed_idx])
        if active_steps.size == 0:
            continue
        distances = cumulative_distance[active_steps, seed_idx]
        points = traces[active_steps, seed_idx]
        positive = np.concatenate([[True], np.diff(distances) > 1e-6])
        distances = distances[positive]
        points = points[positive]
        if distances.shape[0] == 0:
            continue

        for out_idx, target_distance in enumerate(target_distances):
            if target_distance > distances[-1]:
                continue
            right = int(np.searchsorted(distances, target_distance, side="left"))
            if right >= distances.shape[0]:
                continue
            if np.isclose(distances[right], target_distance):
                sampled_points[out_idx, seed_idx] = points[right]
                sampled_active[out_idx, seed_idx] = True
                continue
            left = right - 1
            if left < 0:
                continue
            denom = float(distances[right] - distances[left])
            if denom <= 1e-6:
                continue
            frac = float((target_distance - distances[left]) / denom)
            sampled_points[out_idx, seed_idx] = points[left] + frac * (points[right] - points[left])
            sampled_active[out_idx, seed_idx] = True

    sampled_points[~sampled_active] = -1.0
    return sampled_points, sampled_active


def _interp_polyline_at_distances(points, distances):
    points = np.asarray(points, dtype=np.float32)
    distances = np.asarray(distances, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N, 3], got {points.shape}")
    if points.shape[0] == 0:
        return np.zeros((distances.shape[0], 3), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, distances.shape[0], axis=0)

    segment_vecs = points[1:] - points[:-1]
    segment_lengths = np.linalg.norm(segment_vecs, axis=1).astype(np.float32)
    cumulative = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(segment_lengths)])
    positive = segment_lengths > 1e-6
    if not bool(positive.any()):
        return np.repeat(points[:1], distances.shape[0], axis=0)

    out = np.empty((distances.shape[0], 3), dtype=np.float32)
    first_idx = int(np.flatnonzero(positive)[0])
    last_idx = int(np.flatnonzero(positive)[-1])
    for out_idx, distance in enumerate(distances):
        if distance <= cumulative[first_idx]:
            seg_idx = first_idx
        elif distance >= cumulative[last_idx + 1]:
            seg_idx = last_idx
        else:
            seg_idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
            while seg_idx < segment_lengths.shape[0] - 1 and segment_lengths[seg_idx] <= 1e-6:
                seg_idx += 1
            if segment_lengths[seg_idx] <= 1e-6:
                seg_idx = last_idx
        frac = (float(distance) - float(cumulative[seg_idx])) / float(segment_lengths[seg_idx])
        out[out_idx] = points[seg_idx] + frac * segment_vecs[seg_idx]
    return out


def _regularize_tifxyz_front_spacing(sampled_points, sampled_active, tifxyz_step):
    sampled_points = np.asarray(sampled_points, dtype=np.float32).copy()
    sampled_active = np.asarray(sampled_active, dtype=bool).copy()
    tifxyz_step = float(tifxyz_step)
    if tifxyz_step <= 0.0:
        raise ValueError("tifxyz_step must be > 0.")
    if sampled_points.ndim != 3 or sampled_points.shape[-1] != 3:
        raise ValueError(f"sampled_points must be [S, N, 3], got {sampled_points.shape}")
    if sampled_active.shape != sampled_points.shape[:2]:
        raise ValueError(f"sampled_active must match sampled_points[:2], got {sampled_active.shape}")

    for step_idx in range(sampled_points.shape[0]):
        active_indices = np.flatnonzero(sampled_active[step_idx])
        if active_indices.size <= 1:
            continue
        run_start = 0
        while run_start < active_indices.size:
            run_end = run_start + 1
            while (
                run_end < active_indices.size
                and active_indices[run_end] == active_indices[run_end - 1] + 1
            ):
                run_end += 1

            run_indices = active_indices[run_start:run_end]
            if run_indices.size > 1:
                points = sampled_points[step_idx, run_indices]
                if np.isfinite(points).all():
                    distances = tifxyz_step * np.arange(run_indices.size, dtype=np.float32)
                    sampled_points[step_idx, run_indices] = _interp_polyline_at_distances(points, distances)

            run_start = run_end

    sampled_points[~sampled_active] = -1.0
    return sampled_points, sampled_active


def _build_merged_streamline_tifxyz(stored_zyxs, valid, edge_zyx, cond_direction, integration_group):
    base = np.asarray(stored_zyxs, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=bool)
    if base.ndim != 3 or base.shape[-1] != 3:
        raise RuntimeError(f"Unexpected input tifxyz shape: {base.shape}")
    if valid.shape != base.shape[:2]:
        raise RuntimeError(f"Unexpected input tifxyz valid mask shape: {valid.shape}")
    base[~valid] = -1.0

    edge = np.asarray(edge_zyx, dtype=np.float32)
    edge_valid = np.isfinite(edge).all(axis=1) & ~(edge == -1).all(axis=1)
    sampled_points, sampled_active = _streamline_points_at_tifxyz_steps(integration_group)
    sampled_points, sampled_active = _regularize_tifxyz_front_spacing(
        sampled_points,
        sampled_active,
        float(integration_group.attrs.get("tifxyz_voxel_step", TIFXYZ_VOXEL_STEP)),
    )
    if sampled_points.shape[1] != int(edge_valid.sum()):
        raise RuntimeError(
            "Streamline seed count does not match conditioning edge valid count: "
            f"{sampled_points.shape[1]} vs {int(edge_valid.sum())}"
        )

    spec = _get_direction_spec(cond_direction)
    n_steps = int(sampled_points.shape[0])
    if spec["axis"] == "col":
        if edge.shape[0] != base.shape[0]:
            raise RuntimeError(f"Conditioning edge length {edge.shape[0]} does not match tifxyz rows {base.shape[0]}.")
        strip = np.full((base.shape[0], n_steps, 3), -1.0, dtype=np.float32)
        strip_active = np.zeros((base.shape[0], n_steps), dtype=bool)
        strip[edge_valid, :, :] = np.moveaxis(sampled_points, 0, 1)
        strip_active[edge_valid, :] = np.moveaxis(sampled_active, 0, 1)
        strip[~strip_active] = -1.0
        if spec["edge_idx"] == 0:
            strip = strip[:, ::-1, :]
            return np.concatenate([strip, base], axis=1)
        return np.concatenate([base, strip], axis=1)

    if edge.shape[0] != base.shape[1]:
        raise RuntimeError(f"Conditioning edge length {edge.shape[0]} does not match tifxyz columns {base.shape[1]}.")
    strip = np.full((n_steps, base.shape[1], 3), -1.0, dtype=np.float32)
    strip_active = np.zeros((n_steps, base.shape[1]), dtype=bool)
    strip[:, edge_valid, :] = sampled_points
    strip_active[:, edge_valid] = sampled_active
    strip[~strip_active] = -1.0
    if spec["edge_idx"] == 0:
        strip = strip[::-1, :, :]
        return np.concatenate([strip, base], axis=0)
    return np.concatenate([base, strip], axis=0)


def save_merged_streamline_tifxyz(
    stored_zyxs,
    valid,
    edge_zyx,
    cond_direction,
    integration_group,
    *,
    input_tifxyz_path=None,
    output_dir=None,
    iteration_idx=None,
    total_iterations=None,
):
    merged_zyxs = _build_merged_streamline_tifxyz(
        stored_zyxs,
        valid,
        edge_zyx,
        cond_direction,
        integration_group,
    )
    output_dir = _output_tifxyz_dir() if output_dir is None else Path(output_dir)
    output_uuid = _output_tifxyz_uuid(
        iteration_idx=iteration_idx,
        total_iterations=total_iterations,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tifxyz(
        merged_zyxs,
        str(output_dir),
        output_uuid,
        step_size=int(round(float(TIFXYZ_VOXEL_STEP))),
        voxel_size_um=_output_tifxyz_voxel_size_um(),
        source="vesuvius.neural_tracing.inference.infer_streamline",
        additional_metadata={
            "input_tifxyz_path": str(TIFXYZ_PATH if input_tifxyz_path is None else input_tifxyz_path),
            "grow_direction": str(GROW_DIRECTION),
            "cond_direction": str(cond_direction),
            "tifxyz_voxel_step": float(TIFXYZ_VOXEL_STEP),
            "tifxyz_steps": int(TIFXYZ_STEPS),
            "num_iterations": int(NUM_ITERATIONS),
            "iteration": None if iteration_idx is None else int(iteration_idx),
            "save_each_iteration_tifxyz": bool(SAVE_EACH_ITERATION_TIFXYZ),
            "integration_step_size": float(INTEGRATION_STEP_SIZE),
            "trace_validity_threshold": float(TRACE_VALIDITY_THRESHOLD),
        },
    )
    return output_dir / output_uuid


def _cleanup_temp_output_zarr(output_path):
    if OUTPUT_ZARR_PATH is not None:
        return
    output_path = Path(output_path)
    if not output_path.exists():
        return
    shutil.rmtree(output_path)


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
    merger = _SparseChunkOutputMerger(
        zarr_root,
        window_min,
        window_shape,
        crop_size,
        mmap_dir=MERGE_OUTPUTS_MMAP_DIR,
    )

    batch_starts = range(0, len(voxelized_bboxes), int(BATCH_SIZE))
    for start in tqdm(batch_starts, desc="Row/col bbox inference", unit="batch"):
        batch = voxelized_bboxes[start:start + int(BATCH_SIZE)]
        inputs = _prepare_rowcol_inputs(volume_array, batch, crop_size, cond_direction)
        with torch.inference_mode():
            outputs = _run_model_tta(model, inputs, amp_enabled, amp_dtype)
        for output_name, output_value in outputs.items():
            output_value = _collapse_output(output_value)
            if output_value is None:
                continue
            output_np = output_value.detach().float().cpu().numpy()
            merger.accumulate(output_name, output_np, batch)

    return merger.finalize()


def _run_streamline_iteration(
    model,
    model_config,
    volume_array,
    crop_size,
    input_tifxyz_path,
    iteration_idx,
    total_iterations,
    output_tifxyz_dir,
):
    stored_zyxs, valid = _load_surface_zyx(input_tifxyz_path)
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
    surface = read_tifxyz(input_tifxyz_path)
    voxelized_bboxes = upsample_voxelize_tifxyz_surface_in_bboxes(
        surface,
        bboxes,
        crop_size,
        stored_zyxs=stored_zyxs,
        valid=valid,
    )
    output_path = _output_zarr_path()
    try:
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
        output_tifxyz_path = save_merged_streamline_tifxyz(
            stored_zyxs,
            valid,
            edge,
            cond_direction,
            integration_result["group"],
            input_tifxyz_path=input_tifxyz_path,
            output_dir=output_tifxyz_dir,
            iteration_idx=iteration_idx,
            total_iterations=total_iterations,
        )

        print(f"iteration: {int(iteration_idx)}/{int(total_iterations)}")
        print(f"tifxyz_path: {input_tifxyz_path}")
        print(f"volume_path: {VOLUME_PATH}")
        print(f"volume_scale: {int(VOLUME_SCALE)}")
        print(f"checkpoint_path: {CHECKPOINT_PATH}")
        print(f"output_zarr_path: {output_path}")
        print(f"output_tifxyz_path: {output_tifxyz_path}")
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

        if SHOW_NAPARI:
            show_streamline_geometry_napari(
                points_zyx,
                edge,
                bboxes,
                voxelized_bboxes=voxelized_bboxes,
                integration_group=integration_result["group"],
            )
        return {
            "input_tifxyz_path": Path(input_tifxyz_path),
            "output_tifxyz_path": Path(output_tifxyz_path),
            "output_zarr_path": Path(output_path),
            "integration_result": integration_result,
        }
    finally:
        _cleanup_temp_output_zarr(output_path)


def main():
    if DEVICE != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("infer_streamline.py is configured for CUDA-only inference.")

    num_iterations = int(NUM_ITERATIONS)
    if num_iterations <= 0:
        raise ValueError("NUM_ITERATIONS must be >= 1.")

    model, model_config = load_checkpoint(CHECKPOINT_PATH)
    model.to(DEVICE)
    model = torch.compile(model)
    crop_size = _crop_size_from_config(model_config)
    volume_array = _open_volume_array(VOLUME_PATH, VOLUME_SCALE)

    current_tifxyz_path = Path(TIFXYZ_PATH)
    iteration_results = []
    temp_tifxyz_dir = None
    try:
        if num_iterations > 1 and not SAVE_EACH_ITERATION_TIFXYZ:
            temp_tifxyz_dir = tempfile.TemporaryDirectory(prefix="infer_streamline_tifxyz_")
        for iteration_idx in range(1, num_iterations + 1):
            is_final_iteration = iteration_idx == num_iterations
            if is_final_iteration or SAVE_EACH_ITERATION_TIFXYZ:
                output_tifxyz_dir = _output_tifxyz_dir()
            else:
                output_tifxyz_dir = Path(temp_tifxyz_dir.name)
            result = _run_streamline_iteration(
                model,
                model_config,
                volume_array,
                crop_size,
                current_tifxyz_path,
                iteration_idx,
                num_iterations,
                output_tifxyz_dir,
            )
            iteration_results.append(result)
            current_tifxyz_path = result["output_tifxyz_path"]

        print(f"num_iterations: {num_iterations}")
        print(f"save_each_iteration_tifxyz: {bool(SAVE_EACH_ITERATION_TIFXYZ)}")
        print(f"final_output_tifxyz_path: {iteration_results[-1]['output_tifxyz_path']}")
    finally:
        if temp_tifxyz_dir is not None:
            temp_tifxyz_dir.cleanup()


if __name__ == "__main__":
    main()
