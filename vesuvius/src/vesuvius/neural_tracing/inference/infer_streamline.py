import argparse
import json
import os
import pickle
from pathlib import Path
import tempfile
from datetime import datetime
import gc
import hashlib
import re
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
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


TIFXYZ_PATH = None
VOLUME_PATH = "auto"
VOLUME_SCALE = 0
VOLUME_CACHE_DIR = "/tmp/vesuvius-volume-cache"
VOLUME_CACHE_RETRY_SECONDS = 0.0
CHECKPOINT_PATH = None

# Keep None unless intentionally debugging shape mismatches. The default reads
# the checkpoint crop size so inference matches the model patch size.
CROP_SIZE = None
BATCH_SIZE = 1
DEVICE = "cuda"
COMPILE_MODEL = True

OVERLAP_FRAC = 0.5
OUTPUT_ZARR_PATH = None
OUTPUT_TIFXYZ_DIR = None
OUTPUT_TIFXYZ_VOXEL_SIZE_UM = None
RUN_OUTPUT_DIR = None
RUN_TIMESTAMP = None
MERGE_OUTPUTS_CHUNK_SIZE = 256
MERGE_OUTPUTS_MMAP_DIR = "/tmp/streamline_tmp_outputs/"
GEOMEDIAN_VECTOR_OUTPUTS = ("velocity_dir", "surface_attract")
VECTOR_MERGE_METHOD = "mean"  # "geomedian" or "mean"
GEOMEDIAN_MAX_ITER = 8
GEOMEDIAN_EPS = 1e-6
USE_TTA = True
TTA_FLIP_AXES = ((2,), (3,), (4,)) #, (2, 3), (2, 4), (3, 4), (2, 3, 4))
SHOW_NAPARI = False

GROW_DIRECTION = "left"

TIFXYZ_VOXEL_STEP = 20.0
TIFXYZ_STEPS = 2
NUM_ITERATIONS = 10
# If false, intermediate iterations are written to a temporary tifxyz directory
# only long enough to feed the next iteration; the final iteration is saved.
SAVE_EACH_ITERATION_TIFXYZ = True
INTEGRATION_STEP_SIZE = 6.0
INTEGRATION_METHOD = "huen"  # "euler", "rk2", or "heun"
TRACE_VALIDITY_THRESHOLD = 0.5
USE_SURFACE_ATTRACT = True
SURFACE_ATTRACT_MODE = "normal"  # "full" or "normal"
SURFACE_ATTRACT_WEIGHT = 1.0
SURFACE_ATTRACT_MAX_CORRECTION = 8.0
USE_EDGE_SUPPORT_SEEDS = True
EDGE_SUPPORT_OFFSETS_FULL_RES = (-1.0, 0.0, 1.0)
EDGE_SUPPORT_WEIGHTS = (0.25, 0.5, 0.25)
EDGE_SUPPORT_REQUIRE_CENTER_ACTIVE = True
STREAMLINE_DENSE_FIELD_MAX_BYTES = 2 * 1024**3


class _DistributedContext:
    def __init__(
        self,
        world_size=1,
        rank=0,
        local_rank=0,
        backend="none",
        device=None,
        initialized_here=False,
    ):
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.local_rank = int(local_rank)
        self.backend = str(backend)
        self.device = device
        self.initialized_here = bool(initialized_here)

    @property
    def is_distributed(self):
        return self.world_size > 1

    @property
    def is_rank0(self):
        return self.rank == 0


def _parse_int_env(env, name, default):
    value = env.get(name)
    if value is None or str(value).strip() == "":
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def _distributed_context_from_env(env=None):
    env = os.environ if env is None else env
    world_size = _parse_int_env(env, "WORLD_SIZE", 1)
    rank = _parse_int_env(env, "RANK", 0)
    local_rank = _parse_int_env(env, "LOCAL_RANK", rank if world_size > 1 else 0)
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE), got rank={rank}, world_size={world_size}")
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {local_rank}")
    return _DistributedContext(world_size=world_size, rank=rank, local_rank=local_rank)


def _distributed_device_for_args(args_device, context):
    requested = str(args_device)
    if context.is_distributed and requested.startswith("cuda"):
        return f"cuda:{context.local_rank}"
    return requested


def _resolve_distributed_backend(requested_backend, device):
    backend = str(requested_backend or "auto").strip().lower()
    if backend == "auto":
        return "nccl" if str(device).startswith("cuda") else "gloo"
    if backend not in ("nccl", "gloo"):
        raise ValueError(f"Unsupported distributed backend {requested_backend!r}; expected auto, nccl, or gloo.")
    return backend


def _validate_distributed_args(args, context):
    if context.is_distributed and bool(args.show_napari):
        raise ValueError("--show-napari is not supported with torchrun/distributed inference.")


def _initialize_distributed(args):
    context = _distributed_context_from_env()
    device = _distributed_device_for_args(args.device, context)
    backend = _resolve_distributed_backend(getattr(args, "distributed_backend", "auto"), device)
    context.device = device
    context.backend = backend if context.is_distributed else "none"
    _validate_distributed_args(args, context)

    if not context.is_distributed:
        return context

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available, but WORLD_SIZE > 1.")
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested distributed CUDA device {device!r}, but CUDA is not available.")
        if context.local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={context.local_rank} is outside available CUDA devices "
                f"({torch.cuda.device_count()})."
            )
        torch.cuda.set_device(torch.device(device))
    if backend == "nccl" and not str(device).startswith("cuda"):
        raise ValueError("NCCL distributed backend requires a CUDA device.")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
        context.initialized_here = True
    return context


def _destroy_distributed(context):
    if (
        context is not None
        and context.initialized_here
        and dist.is_available()
        and dist.is_initialized()
    ):
        dist.destroy_process_group()


def _should_write_outputs(context):
    return context is None or context.is_rank0


def _broadcast_object_from_rank0(value, context):
    if context is None or not context.is_distributed:
        return value
    values = [value if context.is_rank0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _wait_for_tifxyz_output(tifxyz_path, timeout_seconds=24 * 60 * 60, poll_seconds=2.0):
    tifxyz_path = Path(tifxyz_path)
    required = [tifxyz_path / name for name in ("meta.json", "x.tif", "y.tif", "z.tif")]
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        if all(path.exists() for path in required):
            return tifxyz_path
        if time.monotonic() > deadline:
            missing = [str(path) for path in required if not path.exists()]
            raise TimeoutError(f"Timed out waiting for tifxyz output {tifxyz_path}; missing: {missing}")
        time.sleep(float(poll_seconds))


def _gather_object_to_rank0(value, context):
    if context is None or not context.is_distributed:
        return [value]
    gathered = [None for _ in range(context.world_size)] if context.is_rank0 else None
    dist.gather_object(value, object_gather_list=gathered, dst=0)
    return gathered


def _distributed_payload_dir(payload_tag):
    base_dir = Path(RUN_OUTPUT_DIR) if RUN_OUTPUT_DIR is not None else Path(tempfile.gettempdir())
    tag = "default" if payload_tag is None else str(payload_tag)
    safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag)
    return base_dir / ".infer_streamline_dist_payloads" / safe_tag


def _prepare_distributed_payload_dir(context, payload_tag):
    payload_dir = _distributed_payload_dir(payload_tag)
    if context.is_rank0 and payload_dir.exists():
        shutil.rmtree(payload_dir)
    if context.is_distributed:
        dist.barrier()
    payload_dir.mkdir(parents=True, exist_ok=True)
    if context.is_distributed:
        dist.barrier()
    return payload_dir


def _write_payload_ref(payload, payload_dir, context):
    if payload is None:
        return None
    payload_path = payload_dir / (
        f"batch_{int(payload['batch_index']):08d}_rank_{int(context.rank):04d}.pkl"
    )
    with open(payload_path, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "batch_index": int(payload["batch_index"]),
        "path": str(payload_path),
    }


def _read_payload_ref(payload_ref):
    with open(payload_ref["path"], "rb") as fp:
        return pickle.load(fp)


def _cleanup_payload_ref(payload_ref):
    if payload_ref is None:
        return
    try:
        Path(payload_ref["path"]).unlink()
    except FileNotFoundError:
        pass


def _batch_specs_for_count(item_count, batch_size):
    item_count = int(item_count)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return [
        {
            "batch_index": batch_index,
            "start": start,
            "stop": min(start + batch_size, item_count),
        }
        for batch_index, start in enumerate(range(0, item_count, batch_size))
    ]


def _assigned_batch_specs(batch_specs, rank, world_size):
    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}")
    return [
        spec
        for spec in batch_specs
        if int(spec["batch_index"]) % world_size == rank
    ]


def _rank_batch_assignment_summary(total_batches, world_size):
    batch_specs = _batch_specs_for_count(total_batches, 1)
    return {
        str(rank): [
            int(spec["batch_index"])
            for spec in _assigned_batch_specs(batch_specs, rank, world_size)
        ]
        for rank in range(int(world_size))
    }


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


def _sanitize_surface_validity(zyx_grid, reader_valid=None):
    valid = _valid_surface_mask(np.asarray(zyx_grid))
    if reader_valid is not None:
        valid &= np.asarray(reader_valid, dtype=bool)
    return valid


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


def get_cond_edge_bboxes(
    cond_zyxs,
    cond_direction,
    crop_size,
    overlap_frac=0.15,
    cond_valid=None,
    integration_margin=0.0,
):
    # Build center-out crop anchors along the conditioning edge. Each chunk grows
    # while its XYZ span still fits in one crop-sized bbox.
    full_edge = _get_cond_edge(cond_zyxs, cond_direction, cond_valid=cond_valid)

    edge_valid = ~(full_edge == -1).all(axis=1)
    if not edge_valid.any():
        return [], full_edge
    edge = full_edge[edge_valid]
    n_edge = edge.shape[0]
    if n_edge == 0:
        return [], full_edge

    crop_size_arr = np.asarray(crop_size, dtype=np.int64)

    overlap_frac = float(overlap_frac)
    overlap_frac = max(0.0, min(overlap_frac, 0.99))

    integration_margin = max(0.0, float(integration_margin))
    margin_arr = np.ceil(integration_margin).astype(np.int64)
    span_limit = np.maximum(1, crop_size_arr - 1 - (2 * margin_arr))

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

    return bboxes, full_edge


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
    valid = _sanitize_surface_validity(stored_zyxs, valid)
    return stored_zyxs, valid


def _read_tifxyz_meta(tifxyz_path):
    meta_path = Path(tifxyz_path) / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "rt", encoding="utf-8") as fp:
        return json.load(fp)


def _normalize_match_token(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _path_match_text(tifxyz_path):
    tifxyz_path = Path(tifxyz_path)
    meta = _read_tifxyz_meta(tifxyz_path)
    parts = list(tifxyz_path.parts[-6:])
    parts.extend(
        str(meta.get(key, ""))
        for key in ("scroll_source", "volume", "uuid", "seed_surface_id")
    )
    return _normalize_match_token(" ".join(parts))


def _dataset_match_tokens(dataset):
    tokens = []
    segments_path = dataset.get("segments_path")
    if segments_path:
        tokens.append(Path(str(segments_path).rstrip("/")).name)
    volume_path = dataset.get("volume_path")
    if volume_path:
        for part in re.split(r"[/_:.-]+", str(volume_path).rstrip("/")):
            normalized = _normalize_match_token(part)
            if normalized.startswith("pherc") or normalized.startswith("man"):
                tokens.append(part)
    out = []
    seen = set()
    for token in tokens:
        normalized = _normalize_match_token(token)
        if len(normalized) < 4 or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def resolve_volume_path_from_config(tifxyz_path, model_config, requested_volume_path="auto", requested_volume_scale=None):
    if requested_volume_path and str(requested_volume_path).strip().lower() != "auto":
        volume_scale = VOLUME_SCALE if requested_volume_scale is None else int(requested_volume_scale)
        return str(requested_volume_path), volume_scale, None

    target_text = _path_match_text(tifxyz_path)
    datasets = list((model_config or {}).get("datasets") or [])
    matches = []
    for dataset in datasets:
        if not dataset.get("volume_path"):
            continue
        tokens = _dataset_match_tokens(dataset)
        if any(token and token in target_text for token in tokens):
            matches.append(dataset)

    if len(matches) != 1:
        candidates = [
            {
                "volume_path": dataset.get("volume_path"),
                "volume_scale": dataset.get("volume_scale"),
                "segments_path": dataset.get("segments_path"),
            }
            for dataset in datasets
        ]
        raise ValueError(
            "Unable to resolve --volume-path auto for "
            f"{tifxyz_path}. Matched {len(matches)} checkpoint datasets. "
            f"Pass --volume-path explicitly. Candidates: {candidates}"
        )

    matched = matches[0]
    volume_scale = matched.get("volume_scale", VOLUME_SCALE)
    if requested_volume_scale is not None:
        volume_scale = int(requested_volume_scale)
    return str(matched["volume_path"]), int(volume_scale), dict(matched)


def derive_tifxyz_voxel_step(tifxyz_path):
    meta = _read_tifxyz_meta(tifxyz_path)
    scale = meta.get("scale")
    if not isinstance(scale, (list, tuple)) or len(scale) < 2:
        raise ValueError(
            f"Cannot derive tifxyz voxel step from missing/invalid scale in {Path(tifxyz_path) / 'meta.json'}"
        )
    scale_x = float(scale[0])
    scale_y = float(scale[1])
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError(f"Invalid tifxyz scale {scale!r}; values must be positive.")
    step_x = int(round(1.0 / scale_x))
    step_y = int(round(1.0 / scale_y))
    if step_x != step_y:
        raise ValueError(f"Expected isotropic tifxyz scale, got {scale!r}.")
    return float(step_x)


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
    root = open_zarr_group(
        volume_path,
        config={
            "volume_cache_dir": VOLUME_CACHE_DIR,
            "volume_cache_retry_seconds": VOLUME_CACHE_RETRY_SECONDS,
        },
    )
    if isinstance(root, zarr.hierarchy.Group):
        scale_key = str(int(volume_scale))
        if scale_key not in root:
            raise KeyError(f"volume scale {scale_key!r} not found in {volume_path}")
        return root[scale_key]
    return root


def _bbox_min_corner(bbox):
    z_min, _, y_min, _, x_min, _ = bbox
    return np.asarray([z_min, y_min, x_min], dtype=np.int64)


def _output_zarr_path(iteration_idx=None):
    if OUTPUT_ZARR_PATH is not None:
        return Path(OUTPUT_ZARR_PATH)
    if RUN_OUTPUT_DIR is not None:
        iteration_idx = 1 if iteration_idx is None else int(iteration_idx)
        return Path(RUN_OUTPUT_DIR) / "zarr" / f"iter_{iteration_idx:02d}.zarr"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.gettempdir()) / f"infer_streamline_rowcol_{timestamp}.zarr"


def _output_tifxyz_dir():
    if OUTPUT_TIFXYZ_DIR is not None:
        return Path(OUTPUT_TIFXYZ_DIR)
    if RUN_OUTPUT_DIR is not None:
        return Path(RUN_OUTPUT_DIR) / "tifxyz"
    return Path(TIFXYZ_PATH).parent


def _output_tifxyz_uuid(timestamp=None, iteration_idx=None, total_iterations=None):
    if timestamp is None:
        timestamp = RUN_TIMESTAMP or datetime.now().strftime("%Y%m%d_%H%M%S")
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
        autocast_device = "cuda" if str(DEVICE).startswith("cuda") else "cpu"
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype):
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


def _model_amp_settings(model_config):
    mixed_precision = str(model_config.get("mixed_precision", "no")).lower()
    amp_enabled = mixed_precision in ("bf16", "fp16", "float16")
    amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return amp_enabled, amp_dtype


def _run_voxelized_model_batch(model, voxelized_batch, volume_array, crop_size, cond_direction, amp_enabled, amp_dtype):
    inputs = _prepare_rowcol_inputs(volume_array, voxelized_batch, crop_size, cond_direction)
    with torch.inference_mode():
        outputs = _run_model_tta(model, inputs, amp_enabled, amp_dtype)

    outputs_np = {}
    for output_name, output_value in outputs.items():
        output_value = _collapse_output(output_value)
        if output_value is None:
            continue
        outputs_np[str(output_name)] = output_value.detach().float().cpu().numpy()
    return outputs_np


def _run_bbox_batch(
    model,
    volume_array,
    surface,
    bboxes,
    crop_size,
    cond_direction,
    stored_zyxs,
    valid,
    amp_enabled,
    amp_dtype,
    batch_spec,
):
    batch_bboxes = bboxes[int(batch_spec["start"]):int(batch_spec["stop"])]
    voxelized_batch = upsample_voxelize_tifxyz_surface_in_bboxes(
        surface,
        batch_bboxes,
        crop_size,
        stored_zyxs=stored_zyxs,
        valid=valid,
    )
    outputs_np = _run_voxelized_model_batch(
        model,
        voxelized_batch,
        volume_array,
        crop_size,
        cond_direction,
        amp_enabled,
        amp_dtype,
    )
    return {
        "batch_index": int(batch_spec["batch_index"]),
        "start": int(batch_spec["start"]),
        "stop": int(batch_spec["stop"]),
        "bboxes": [tuple(int(v) for v in item["bbox"]) for item in voxelized_batch],
        "outputs": outputs_np,
        "voxelized_bbox_voxels": sum(int(item["voxels"].sum()) for item in voxelized_batch),
    }


def _accumulate_batch_payload(merger, payload):
    voxelized_metadata = [
        {"bbox": tuple(int(v) for v in bbox)}
        for bbox in payload["bboxes"]
    ]
    for output_name, output_np in payload["outputs"].items():
        merger.accumulate(output_name, output_np, voxelized_metadata)


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


def _vector_merge_method(method=None):
    method = VECTOR_MERGE_METHOD if method is None else method
    method = str(method).strip().lower()
    aliases = {
        "average": "mean",
        "avg": "mean",
        "vector_geomedian": "geomedian",
        "geometric_median": "geomedian",
        "vector_geometric_median": "geomedian",
    }
    method = aliases.get(method, method)
    if method not in ("geomedian", "mean"):
        raise ValueError(
            f"Unknown VECTOR_MERGE_METHOD {method!r}; expected 'geomedian' or 'mean'."
        )
    return method


def _use_geomedian_vector_merge(output_name, channels, merge_method):
    return (
        _vector_merge_method(merge_method) == "geomedian"
        and _is_geomedian_vector_output(output_name, channels)
    )


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
    def __init__(self, root, window_min, window_shape, crop_size, mmap_dir=None, vector_merge_method=None):
        self.root = root
        self.window_min = np.asarray(window_min, dtype=np.int64)
        self.window_shape = tuple(int(v) for v in window_shape)
        self.crop_size_arr = np.asarray(crop_size, dtype=np.int64)
        self.chunks_3d = tuple(min(int(MERGE_OUTPUTS_CHUNK_SIZE), int(v)) for v in self.window_shape)
        self.vector_merge_method = _vector_merge_method(vector_merge_method)
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
        use_geomedian = _use_geomedian_vector_merge(
            output_name,
            output_batch.shape[1],
            self.vector_merge_method,
        )

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
            for block_start in range(0, counts_flat.shape[0], block_voxels):
                block_end = min(block_start + block_voxels, counts_flat.shape[0])
                block_counts = counts_flat[block_start:block_end]
                if not bool((block_counts > 0).any()):
                    continue
                out_block = out_flat[:, block_start:block_end]
                slot0 = np.asarray(slots[0]).reshape(3, -1)[:, block_start:block_end]

                count1 = block_counts == 1
                if bool(count1.any()):
                    out_block[:, count1] = slot0[:, count1]

                count2 = block_counts == 2
                if bool(count2.any()):
                    slot1 = np.asarray(slots[1]).reshape(3, -1)[:, block_start:block_end]
                    out_block[:, count2] = (slot0[:, count2] + slot1[:, count2]) * np.float32(0.5)

                needs_iter = block_counts > 2
                if not bool(needs_iter.any()):
                    continue

                iter_counts = block_counts[needs_iter]
                block_k = min(int(iter_counts.max()), len(slots))
                iter_cols = np.flatnonzero(needs_iter)
                samples = np.empty((block_k, 3, iter_cols.size), dtype=np.float32)
                for slot_idx in range(block_k):
                    slot_flat = np.asarray(slots[slot_idx]).reshape(3, -1)[:, block_start:block_end]
                    samples[slot_idx] = slot_flat[:, iter_cols]
                valid = np.arange(block_k, dtype=np.uint16)[:, None] < iter_counts[None, :]
                out_block[:, needs_iter] = _geometric_median_vectors(samples, valid)

            avg[(slice(None), *region)] = out_flat.reshape((3, *count_chunk.shape))

    def finalize(self):
        try:
            outputs = self.root.require_group("outputs")
            avg_group = outputs.require_group("avg")
            avg_group.attrs["window_min_zyx"] = self.root.attrs.get("window_min_zyx", [0, 0, 0])
            avg_group.attrs["window_shape_zyx"] = self.root.attrs.get("window_shape_zyx", None)
            avg_group.attrs["merge_backing"] = "mmap"
            avg_group.attrs["merge_sparse_bytes"] = int(self.current_bytes)
            avg_group.attrs["vector_merge_method"] = str(self.vector_merge_method)

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
                if _use_geomedian_vector_merge(output_name, channels, self.vector_merge_method):
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


def _integration_method():
    method = str(INTEGRATION_METHOD).strip().lower()
    aliases = {
        "midpoint": "rk2",
        "rk2_midpoint": "rk2",
        "predictor_corrector": "heun",
        "huen": "heun",
    }
    method = aliases.get(method, method)
    if method not in ("euler", "rk2", "heun"):
        raise ValueError(
            f"Unknown INTEGRATION_METHOD {INTEGRATION_METHOD!r}; "
            "expected 'euler', 'rk2', or 'heun'."
        )
    return method


def _surface_attract_mode():
    mode = str(SURFACE_ATTRACT_MODE).strip().lower()
    aliases = {
        "project": "full",
        "projection": "full",
        "normal_only": "normal",
        "perpendicular": "normal",
        "perp": "normal",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("full", "normal"):
        raise ValueError(
            f"Unknown SURFACE_ATTRACT_MODE {SURFACE_ATTRACT_MODE!r}; "
            "expected 'full' or 'normal'."
        )
    return mode


def _bbox_integration_margin():
    step_sizes, _, target_distance = _integration_step_sizes()
    if not step_sizes:
        return 0.0
    return float(target_distance) + float(max(step_sizes))


def _edge_support_offsets_and_weights():
    if not bool(USE_EDGE_SUPPORT_SEEDS):
        return np.asarray([0.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32)

    offsets = np.asarray(EDGE_SUPPORT_OFFSETS_FULL_RES, dtype=np.float32).reshape(-1)
    weights = np.asarray(EDGE_SUPPORT_WEIGHTS, dtype=np.float32).reshape(-1)
    if offsets.size == 0:
        raise ValueError("EDGE_SUPPORT_OFFSETS_FULL_RES must contain at least one offset.")
    if weights.shape != offsets.shape:
        raise ValueError(
            "EDGE_SUPPORT_WEIGHTS must have the same length as "
            "EDGE_SUPPORT_OFFSETS_FULL_RES."
        )
    if not np.isfinite(offsets).all():
        raise ValueError("EDGE_SUPPORT_OFFSETS_FULL_RES must be finite.")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("EDGE_SUPPORT_WEIGHTS must be finite and non-negative.")
    if not np.any(np.isclose(offsets, 0.0)):
        raise ValueError("EDGE_SUPPORT_OFFSETS_FULL_RES must include 0.0 for the center seed.")
    if float(weights.sum()) <= 0.0:
        raise ValueError("EDGE_SUPPORT_WEIGHTS must contain at least one positive weight.")
    return offsets, weights


def _interp_edge_run_at(edge, run_start, run_end, query_index):
    query_index = float(query_index)
    if query_index < float(run_start) - 1e-6 or query_index > float(run_end) + 1e-6:
        return None
    if query_index <= float(run_start):
        return edge[run_start].astype(np.float32, copy=True)
    if query_index >= float(run_end):
        return edge[run_end].astype(np.float32, copy=True)

    left = int(np.floor(query_index))
    right = left + 1
    if right > run_end:
        return edge[run_end].astype(np.float32, copy=True)
    frac = np.float32(query_index - float(left))
    return (edge[left] + frac * (edge[right] - edge[left])).astype(np.float32, copy=False)


def _make_edge_support_seeds(edge_zyx, edge_axis_scale):
    edge = np.asarray(edge_zyx, dtype=np.float32)
    edge_valid = np.isfinite(edge).all(axis=1) & ~(edge == -1).all(axis=1)
    sparse_edge_indices = np.flatnonzero(edge_valid).astype(np.int64)
    if sparse_edge_indices.size == 0:
        raise RuntimeError("No valid edge points available for streamline integration.")

    offsets, weights = _edge_support_offsets_and_weights()
    center_offset_indices = np.flatnonzero(np.isclose(offsets, 0.0))
    center_offset = int(center_offset_indices[0])
    edge_axis_scale = float(edge_axis_scale)
    if edge_axis_scale <= 0.0:
        raise ValueError(f"edge_axis_scale must be > 0, got {edge_axis_scale!r}")

    seed_points = []
    source_sparse_indices = []
    support_offsets = []
    support_weights = []
    support_is_center = []

    valid_to_sparse = {
        int(edge_idx): int(sparse_idx)
        for sparse_idx, edge_idx in enumerate(sparse_edge_indices)
    }

    run_start_pos = 0
    while run_start_pos < sparse_edge_indices.size:
        run_end_pos = run_start_pos + 1
        while (
            run_end_pos < sparse_edge_indices.size
            and sparse_edge_indices[run_end_pos] == sparse_edge_indices[run_end_pos - 1] + 1
        ):
            run_end_pos += 1

        run_start = int(sparse_edge_indices[run_start_pos])
        run_end = int(sparse_edge_indices[run_end_pos - 1])
        for edge_idx in sparse_edge_indices[run_start_pos:run_end_pos]:
            sparse_idx = valid_to_sparse[int(edge_idx)]
            for offset_idx, offset in enumerate(offsets):
                query_index = float(edge_idx) + float(offset) * edge_axis_scale
                point = _interp_edge_run_at(edge, run_start, run_end, query_index)
                if point is None:
                    continue
                seed_points.append(point)
                source_sparse_indices.append(sparse_idx)
                support_offsets.append(float(offset))
                support_weights.append(float(weights[offset_idx]))
                support_is_center.append(offset_idx == center_offset)

        run_start_pos = run_end_pos

    if not seed_points:
        raise RuntimeError("No edge support seeds available for streamline integration.")

    return {
        "seed_zyx": np.asarray(seed_points, dtype=np.float32),
        "source_sparse_index": np.asarray(source_sparse_indices, dtype=np.int64),
        "support_offset_full_res": np.asarray(support_offsets, dtype=np.float32),
        "support_weight": np.asarray(support_weights, dtype=np.float32),
        "support_is_center": np.asarray(support_is_center, dtype=bool),
        "sparse_edge_indices": sparse_edge_indices,
        "sparse_seed_count": int(sparse_edge_indices.size),
    }


class _TorchStreamlineFieldSampler:
    def __init__(self, velocity, attract, validity, device):
        self.device = torch.device(device)
        self.velocity = velocity
        self.attract = attract
        self.validity = validity
        self.depth = int(velocity.shape[1])
        self.height = int(velocity.shape[2])
        self.width = int(velocity.shape[3])
        for name, field, channels in (
            ("velocity_dir", velocity, 3),
            ("surface_attract", attract, 3),
            ("trace_validity", validity, 1),
        ):
            if len(field.shape) != 4 or field.shape[0] != channels:
                raise RuntimeError(f"Expected {name} shape ({channels}, D, H, W), got {field.shape}")
            if tuple(field.shape[1:]) != (self.depth, self.height, self.width):
                raise RuntimeError(f"Field {name} spatial shape {field.shape[1:]} does not match velocity field.")

        dense_bytes = sum(
            int(np.prod(field.shape, dtype=np.int64)) * np.dtype(field.dtype).itemsize
            for field in (velocity, attract, validity)
        )
        self._use_dense = dense_bytes <= int(STREAMLINE_DENSE_FIELD_MAX_BYTES)
        self.field = None
        if self._use_dense:
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

    def _combined_channel(self, channel):
        channel = int(channel)
        if 0 <= channel < 3:
            return self.velocity, channel
        if 3 <= channel < 6:
            return self.attract, channel - 3
        if channel == 6:
            return self.validity, 0
        raise ValueError(f"Unexpected streamline field channel {channel}.")

    def _combined_channel_segments(self, channel_start, channel_count):
        segments = []
        out_start = 0
        while out_start < int(channel_count):
            field, source_start = self._combined_channel(int(channel_start) + out_start)
            length = 1
            while out_start + length < int(channel_count):
                next_field, next_source = self._combined_channel(int(channel_start) + out_start + length)
                if next_field is not field or next_source != source_start + length:
                    break
                length += 1
            segments.append((field, int(source_start), int(length), int(out_start)))
            out_start += length
        return segments

    def _sample_lazy_field_segment(
        self,
        field,
        source_start,
        channel_count,
        flat_point_idx,
        flat_z,
        flat_y,
        flat_x,
        flat_weight,
        out_values,
        out_start,
    ):
        chunks = getattr(field, "chunks", None)
        if chunks is None or len(chunks) != 4:
            raise RuntimeError("Lazy streamline sampling requires Zarr-style chunked fields.")
        chunk_z, chunk_y, chunk_x = (int(v) for v in chunks[1:])
        n_chunk_y = (self.height + chunk_y - 1) // chunk_y
        n_chunk_x = (self.width + chunk_x - 1) // chunk_x
        chunk_key = ((flat_z // chunk_z) * n_chunk_y + (flat_y // chunk_y)) * n_chunk_x + (flat_x // chunk_x)
        order = np.argsort(chunk_key, kind="stable")
        channel_slice = slice(int(source_start), int(source_start) + int(channel_count))

        run_start = 0
        while run_start < order.size:
            run_end = run_start + 1
            key = chunk_key[order[run_start]]
            while run_end < order.size and chunk_key[order[run_end]] == key:
                run_end += 1
            idx = order[run_start:run_end]
            z = flat_z[idx]
            y = flat_y[idx]
            x = flat_x[idx]
            z_min = int(z.min())
            y_min = int(y.min())
            x_min = int(x.min())
            z_max = int(z.max()) + 1
            y_max = int(y.max()) + 1
            x_max = int(x.max()) + 1
            block = np.asarray(
                field[channel_slice, z_min:z_max, y_min:y_max, x_min:x_max],
                dtype=np.float32,
            )
            if block.ndim != 4 or block.shape[0] != int(channel_count):
                raise RuntimeError(
                    "Unexpected lazy field block shape "
                    f"{block.shape}; expected ({int(channel_count)}, D, H, W)."
                )
            corner_values = block[:, z - z_min, y - y_min, x - x_min].T
            weighted_values = flat_weight[idx, None].astype(np.float32, copy=False) * corner_values
            local_points = flat_point_idx[idx]
            for channel_offset in range(int(channel_count)):
                np.add.at(
                    out_values[:, int(out_start) + channel_offset],
                    local_points,
                    weighted_values[:, channel_offset],
                )
            run_start = run_end

    def _sample_lazy(self, points_zyx, channel_start, channel_count):
        if points_zyx.ndim != 2 or points_zyx.shape[1] != 3:
            raise ValueError(f"points_zyx must be [N, 3], got {tuple(points_zyx.shape)}")

        points_np = points_zyx.detach().cpu().numpy().astype(np.float32, copy=False)
        n_points = int(points_np.shape[0])
        finite = np.isfinite(points_np).all(axis=1)
        safe = np.where(finite[:, None], points_np, 0.0)
        in_bounds_np = (
            finite
            & (safe[:, 0] >= 0.0) & (safe[:, 0] <= self.depth - 1)
            & (safe[:, 1] >= 0.0) & (safe[:, 1] <= self.height - 1)
            & (safe[:, 2] >= 0.0) & (safe[:, 2] <= self.width - 1)
        )
        sampled_np = np.zeros((n_points, int(channel_count)), dtype=np.float32)
        valid_idx = np.flatnonzero(in_bounds_np)
        if valid_idx.size > 0:
            points_valid = safe[valid_idx]
            z0 = np.floor(points_valid[:, 0]).astype(np.int64)
            y0 = np.floor(points_valid[:, 1]).astype(np.int64)
            x0 = np.floor(points_valid[:, 2]).astype(np.int64)
            z1 = np.minimum(z0 + 1, self.depth - 1)
            y1 = np.minimum(y0 + 1, self.height - 1)
            x1 = np.minimum(x0 + 1, self.width - 1)
            wz = (points_valid[:, 0] - z0.astype(np.float32)).astype(np.float32)
            wy = (points_valid[:, 1] - y0.astype(np.float32)).astype(np.float32)
            wx = (points_valid[:, 2] - x0.astype(np.float32)).astype(np.float32)
            z_indices = (z0, z1)
            y_indices = (y0, y1)
            x_indices = (x0, x1)
            z_weights = (1.0 - wz, wz)
            y_weights = (1.0 - wy, wy)
            x_weights = (1.0 - wx, wx)
            flat_point_idx = []
            flat_z = []
            flat_y = []
            flat_x = []
            flat_weight = []
            for zi in (0, 1):
                for yi in (0, 1):
                    for xi in (0, 1):
                        weight = z_weights[zi] * y_weights[yi] * x_weights[xi]
                        nonzero = weight != 0.0
                        if not bool(nonzero.any()):
                            continue
                        flat_point_idx.append(np.flatnonzero(nonzero).astype(np.int64, copy=False))
                        flat_z.append(z_indices[zi][nonzero])
                        flat_y.append(y_indices[yi][nonzero])
                        flat_x.append(x_indices[xi][nonzero])
                        flat_weight.append(weight[nonzero].astype(np.float32, copy=False))

            if flat_point_idx:
                flat_point_idx = np.concatenate(flat_point_idx)
                flat_z = np.concatenate(flat_z).astype(np.int64, copy=False)
                flat_y = np.concatenate(flat_y).astype(np.int64, copy=False)
                flat_x = np.concatenate(flat_x).astype(np.int64, copy=False)
                flat_weight = np.concatenate(flat_weight).astype(np.float32, copy=False)
                sampled_valid = np.zeros((valid_idx.size, int(channel_count)), dtype=np.float32)
                for field, source_start, segment_count, out_start in self._combined_channel_segments(
                    channel_start,
                    channel_count,
                ):
                    self._sample_lazy_field_segment(
                        field,
                        source_start,
                        segment_count,
                        flat_point_idx,
                        flat_z,
                        flat_y,
                        flat_x,
                        flat_weight,
                        sampled_valid,
                        out_start,
                    )
                sampled_np[valid_idx, :] = sampled_valid

        sampled = torch.from_numpy(sampled_np).to(device=self.device, dtype=torch.float32)
        in_bounds = torch.from_numpy(in_bounds_np).to(device=self.device, dtype=torch.bool)
        return sampled, in_bounds

    def sample(self, points_zyx, channel_start, channel_count):
        if points_zyx.ndim != 2 or points_zyx.shape[1] != 3:
            raise ValueError(f"points_zyx must be [N, 3], got {tuple(points_zyx.shape)}")
        if not self._use_dense:
            return self._sample_lazy(points_zyx, channel_start, channel_count)

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


def _sample_unit_velocity(sampler, points_zyx):
    velocity_samples, velocity_in_bounds = sampler.sample_velocity(points_zyx)
    speed = torch.linalg.norm(velocity_samples, dim=1)
    finite_speed = torch.isfinite(speed)
    nonzero_speed = speed > 1e-6
    usable = velocity_in_bounds & finite_speed & nonzero_speed
    unit_velocity = torch.zeros_like(velocity_samples)
    unit_velocity[usable] = velocity_samples[usable] / speed[usable, None]
    return unit_velocity, usable, velocity_in_bounds, finite_speed, nonzero_speed


def _integrate_velocity_candidate(sampler, points_zyx, step_size, method):
    step_size = float(step_size)
    unit0, usable0, in_bounds0, finite0, nonzero0 = _sample_unit_velocity(sampler, points_zyx)
    if method == "euler":
        candidate = points_zyx + step_size * unit0
        usable = usable0
        in_bounds = in_bounds0
        finite = finite0
        nonzero = nonzero0
        return candidate, usable, in_bounds, finite, nonzero

    if method == "rk2":
        midpoint = points_zyx + (0.5 * step_size) * unit0
        unit_mid, usable_mid, in_bounds_mid, finite_mid, nonzero_mid = _sample_unit_velocity(sampler, midpoint)
        candidate = points_zyx + step_size * unit_mid
        usable = usable0 & usable_mid
        in_bounds = in_bounds0 & in_bounds_mid
        finite = finite0 & finite_mid
        nonzero = nonzero0 & nonzero_mid
        return candidate, usable, in_bounds, finite, nonzero

    predictor = points_zyx + step_size * unit0
    unit1, usable1, in_bounds1, finite1, nonzero1 = _sample_unit_velocity(sampler, predictor)
    avg_unit = unit0 + unit1
    avg_norm = torch.linalg.norm(avg_unit, dim=1)
    avg_finite = torch.isfinite(avg_norm)
    avg_nonzero = avg_norm > 1e-6
    usable_avg = avg_finite & avg_nonzero
    step_unit = torch.zeros_like(avg_unit)
    step_unit[usable_avg] = avg_unit[usable_avg] / avg_norm[usable_avg, None]
    candidate = points_zyx + step_size * step_unit
    usable = usable0 & usable1 & usable_avg
    in_bounds = in_bounds0 & in_bounds1
    finite = finite0 & finite1 & avg_finite
    nonzero = nonzero0 & nonzero1 & avg_nonzero
    return candidate, usable, in_bounds, finite, nonzero


def _clamp_vector_norm(vectors, max_norm):
    if max_norm is None:
        return vectors
    max_norm = float(max_norm)
    if max_norm < 0.0:
        return vectors
    norm = torch.linalg.norm(vectors, dim=1)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1e-6), max=1.0)
    return vectors * scale[:, None]


def _apply_surface_attract_correction(sampler, candidate):
    attract_samples, attract_in_bounds = sampler.sample_attract(candidate)
    attract_samples = _clamp_vector_norm(attract_samples, SURFACE_ATTRACT_MAX_CORRECTION)
    mode = _surface_attract_mode()

    if mode == "full":
        correction = attract_samples
        usable = attract_in_bounds
    else:
        velocity_unit, velocity_usable, _, _, _ = _sample_unit_velocity(sampler, candidate)
        parallel = (attract_samples * velocity_unit).sum(dim=1, keepdim=True) * velocity_unit
        correction = attract_samples - parallel
        correction = _clamp_vector_norm(correction, SURFACE_ATTRACT_MAX_CORRECTION)
        usable = attract_in_bounds & velocity_usable

    corrected = candidate + float(SURFACE_ATTRACT_WEIGHT) * correction
    return corrected, usable


def integrate_streamlines_from_edge(avg_group, edge_zyx, edge_axis_scale, window_min, zarr_root):
    velocity = _require_avg_field(avg_group, "velocity_dir", 3)
    attract = _require_avg_field(avg_group, "surface_attract", 3)
    validity = _require_avg_field(avg_group, "trace_validity", 1)

    seed_bundle = _make_edge_support_seeds(edge_zyx, edge_axis_scale)
    seeds_world = seed_bundle["seed_zyx"]

    step_sizes, requested_distance, target_distance = _integration_step_sizes()
    window_min = np.asarray(window_min, dtype=np.float32)
    points_local = seeds_world - window_min[None, :]

    sampler = _TorchStreamlineFieldSampler(velocity, attract, validity, DEVICE)
    method = _integration_method()
    n_steps = len(step_sizes)
    n_seeds = int(seeds_world.shape[0])
    points_t = torch.from_numpy(points_local).to(device=sampler.device, dtype=torch.float32)
    traces_t = torch.zeros((n_steps + 1, n_seeds, 3), device=sampler.device, dtype=torch.float32)
    cumulative_distance_t = torch.zeros((n_steps + 1, n_seeds), device=sampler.device, dtype=torch.float32)
    active_mask_t = torch.zeros((n_steps + 1, n_seeds), device=sampler.device, dtype=torch.bool)
    traces_t[0] = points_t
    distance_t = torch.zeros((n_seeds,), device=sampler.device, dtype=torch.float32)
    active = torch.ones((n_seeds,), device=sampler.device, dtype=torch.bool)
    _, seed_in_bounds = sampler.sample_validity(points_t)
    active &= seed_in_bounds
    active_mask_t[0] = active

    diag_active_before_t = torch.zeros((n_steps,), device=sampler.device, dtype=torch.int32)
    diag_accepted_t = torch.zeros((n_steps,), device=sampler.device, dtype=torch.int32)
    diag_velocity_oob_t = torch.zeros((n_steps,), device=sampler.device, dtype=torch.int32)
    diag_velocity_nonfinite_t = torch.zeros((n_steps,), device=sampler.device, dtype=torch.int32)
    diag_velocity_zero_t = torch.zeros((n_steps,), device=sampler.device, dtype=torch.int32)
    diag_attract_oob_t = torch.zeros((n_steps,), device=sampler.device, dtype=torch.int32)
    diag_candidate_nonfinite_t = torch.zeros((n_steps,), device=sampler.device, dtype=torch.int32)
    diag_validity_oob_t = torch.zeros((n_steps,), device=sampler.device, dtype=torch.int32)
    diag_validity_nonfinite_t = torch.zeros((n_steps,), device=sampler.device, dtype=torch.int32)
    diag_low_validity_t = torch.zeros((n_steps,), device=sampler.device, dtype=torch.int32)

    step_iter = tqdm(step_sizes, desc="Integrating streamlines", unit="step")
    for step_idx, step_size in enumerate(step_iter, start=1):
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        diag_idx = step_idx - 1
        diag_active_before_t[diag_idx] = active_idx.numel()
        step_iter.set_postfix(active=int(active_idx.numel()))
        if active_idx.numel() > 0:
            current_points = points_t[active_idx]
            candidate, usable, velocity_in_bounds, finite_speed, nonzero_speed = _integrate_velocity_candidate(
                sampler,
                current_points,
                step_size,
                method,
            )
            diag_velocity_oob_t[diag_idx] = (~velocity_in_bounds).sum()
            diag_velocity_nonfinite_t[diag_idx] = (velocity_in_bounds & ~finite_speed).sum()
            diag_velocity_zero_t[diag_idx] = (velocity_in_bounds & finite_speed & ~nonzero_speed).sum()

            candidate_idx = active_idx[usable]
            candidate = candidate[usable]
            if USE_SURFACE_ATTRACT:
                candidate, usable_candidate = _apply_surface_attract_correction(sampler, candidate)
            else:
                usable_candidate = torch.ones((candidate.shape[0],), device=sampler.device, dtype=torch.bool)

            validity_logits, validity_in_bounds = sampler.sample_validity(candidate)
            validity_prob = torch.sigmoid(validity_logits[:, 0])
            candidate_finite = torch.isfinite(candidate).all(dim=1)
            validity_finite = torch.isfinite(validity_prob)
            candidate_active = (
                usable_candidate
                & validity_in_bounds
                & candidate_finite
                & validity_finite
                & (validity_prob >= float(TRACE_VALIDITY_THRESHOLD))
            )
            diag_attract_oob_t[diag_idx] = (~usable_candidate).sum()
            diag_candidate_nonfinite_t[diag_idx] = (usable_candidate & ~candidate_finite).sum()
            diag_validity_oob_t[diag_idx] = (usable_candidate & candidate_finite & ~validity_in_bounds).sum()
            diag_validity_nonfinite_t[diag_idx] = (
                usable_candidate & candidate_finite & validity_in_bounds & ~validity_finite
            ).sum()
            diag_low_validity_t[diag_idx] = (
                usable_candidate
                & candidate_finite
                & validity_in_bounds
                & validity_finite
                & (validity_prob < float(TRACE_VALIDITY_THRESHOLD))
            ).sum()
            accepted_idx = candidate_idx[candidate_active]
            accepted_candidate = candidate[candidate_active]
            diag_accepted_t[diag_idx] = accepted_idx.numel()
            segment_lengths = torch.linalg.norm(accepted_candidate - points_t[accepted_idx], dim=1)
            distance_t[accepted_idx] = distance_t[accepted_idx] + segment_lengths
            points_t[accepted_idx] = accepted_candidate
            active[candidate_idx] = candidate_active
            active[active_idx[~usable]] = False

        traces_t[step_idx] = points_t
        cumulative_distance_t[step_idx] = distance_t
        active_mask_t[step_idx] = active

    traces_world = traces_t.detach().cpu().numpy()
    traces_world += window_min[None, None, :]
    cumulative_distance = cumulative_distance_t.detach().cpu().numpy()
    active_mask = active_mask_t.detach().cpu().numpy()
    endpoint_active = active.detach().cpu().numpy()
    seed_in_bounds_np = seed_in_bounds.detach().cpu().numpy()
    diag_active_before = diag_active_before_t.detach().cpu().numpy()
    diag_accepted = diag_accepted_t.detach().cpu().numpy()
    diag_velocity_oob = diag_velocity_oob_t.detach().cpu().numpy()
    diag_velocity_nonfinite = diag_velocity_nonfinite_t.detach().cpu().numpy()
    diag_velocity_zero = diag_velocity_zero_t.detach().cpu().numpy()
    diag_attract_oob = diag_attract_oob_t.detach().cpu().numpy()
    diag_candidate_nonfinite = diag_candidate_nonfinite_t.detach().cpu().numpy()
    diag_validity_oob = diag_validity_oob_t.detach().cpu().numpy()
    diag_validity_nonfinite = diag_validity_nonfinite_t.detach().cpu().numpy()
    diag_low_validity = diag_low_validity_t.detach().cpu().numpy()
    integration = zarr_root.require_group("streamline_integration")
    for name in (
        "seed_zyx",
        "points_zyx",
        "active_mask",
        "cumulative_distance",
        "endpoint_zyx",
        "endpoint_active",
        "support_source_sparse_index",
        "support_offset_full_res",
        "support_weight",
        "support_is_center",
        "sparse_edge_index",
        "diagnostic_step_index",
        "diagnostic_active_before",
        "diagnostic_accepted",
        "diagnostic_velocity_oob",
        "diagnostic_velocity_nonfinite",
        "diagnostic_velocity_zero",
        "diagnostic_attract_oob",
        "diagnostic_candidate_nonfinite",
        "diagnostic_validity_oob",
        "diagnostic_validity_nonfinite",
        "diagnostic_low_validity",
    ):
        if name in integration:
            del integration[name]
    integration.create_dataset("seed_zyx", data=seeds_world.astype(np.float32), chunks=(min(seeds_world.shape[0], 4096), 3))
    integration.create_dataset("points_zyx", data=traces_world.astype(np.float32), chunks=(1, min(seeds_world.shape[0], 4096), 3))
    integration.create_dataset("active_mask", data=active_mask.astype(np.uint8), chunks=(1, min(seeds_world.shape[0], 4096)))
    integration.create_dataset("cumulative_distance", data=cumulative_distance.astype(np.float32), chunks=(1, min(seeds_world.shape[0], 4096)))
    integration.create_dataset("endpoint_zyx", data=traces_world[-1].astype(np.float32), chunks=(min(seeds_world.shape[0], 4096), 3))
    integration.create_dataset("endpoint_active", data=endpoint_active.astype(np.uint8), chunks=(min(seeds_world.shape[0], 4096),))
    integration.create_dataset(
        "support_source_sparse_index",
        data=seed_bundle["source_sparse_index"].astype(np.int64),
        chunks=(min(seeds_world.shape[0], 4096),),
    )
    integration.create_dataset(
        "support_offset_full_res",
        data=seed_bundle["support_offset_full_res"].astype(np.float32),
        chunks=(min(seeds_world.shape[0], 4096),),
    )
    integration.create_dataset(
        "support_weight",
        data=seed_bundle["support_weight"].astype(np.float32),
        chunks=(min(seeds_world.shape[0], 4096),),
    )
    integration.create_dataset(
        "support_is_center",
        data=seed_bundle["support_is_center"].astype(np.uint8),
        chunks=(min(seeds_world.shape[0], 4096),),
    )
    integration.create_dataset(
        "sparse_edge_index",
        data=seed_bundle["sparse_edge_indices"].astype(np.int64),
        chunks=(min(seed_bundle["sparse_seed_count"], 4096),),
    )
    integration.create_dataset("diagnostic_step_index", data=np.arange(1, n_steps + 1, dtype=np.int32), chunks=(min(n_steps, 4096),))
    integration.create_dataset("diagnostic_active_before", data=diag_active_before, chunks=(min(n_steps, 4096),))
    integration.create_dataset("diagnostic_accepted", data=diag_accepted, chunks=(min(n_steps, 4096),))
    integration.create_dataset("diagnostic_velocity_oob", data=diag_velocity_oob, chunks=(min(n_steps, 4096),))
    integration.create_dataset("diagnostic_velocity_nonfinite", data=diag_velocity_nonfinite, chunks=(min(n_steps, 4096),))
    integration.create_dataset("diagnostic_velocity_zero", data=diag_velocity_zero, chunks=(min(n_steps, 4096),))
    integration.create_dataset("diagnostic_attract_oob", data=diag_attract_oob, chunks=(min(n_steps, 4096),))
    integration.create_dataset("diagnostic_candidate_nonfinite", data=diag_candidate_nonfinite, chunks=(min(n_steps, 4096),))
    integration.create_dataset("diagnostic_validity_oob", data=diag_validity_oob, chunks=(min(n_steps, 4096),))
    integration.create_dataset("diagnostic_validity_nonfinite", data=diag_validity_nonfinite, chunks=(min(n_steps, 4096),))
    integration.create_dataset("diagnostic_low_validity", data=diag_low_validity, chunks=(min(n_steps, 4096),))
    integration.attrs["tifxyz_voxel_step"] = float(TIFXYZ_VOXEL_STEP)
    integration.attrs["tifxyz_steps"] = int(TIFXYZ_STEPS)
    integration.attrs["integration_steps"] = int(len(step_sizes))
    integration.attrs["integration_step_size"] = float(INTEGRATION_STEP_SIZE)
    integration.attrs["integration_method"] = method
    integration.attrs["requested_distance"] = requested_distance
    integration.attrs["target_distance"] = target_distance
    integration.attrs["actual_step_sizes"] = [float(v) for v in step_sizes]
    integration.attrs["trace_validity_threshold"] = float(TRACE_VALIDITY_THRESHOLD)
    integration.attrs["use_surface_attract"] = bool(USE_SURFACE_ATTRACT)
    integration.attrs["surface_attract_mode"] = _surface_attract_mode()
    integration.attrs["surface_attract_weight"] = float(SURFACE_ATTRACT_WEIGHT)
    integration.attrs["surface_attract_max_correction"] = (
        None if SURFACE_ATTRACT_MAX_CORRECTION is None else float(SURFACE_ATTRACT_MAX_CORRECTION)
    )
    integration.attrs["distance_sampling"] = "actual_arclength"
    integration.attrs["active_endpoints"] = int(endpoint_active.sum())
    integration.attrs["seed_count"] = int(seeds_world.shape[0])
    integration.attrs["dense_seed_count"] = int(seeds_world.shape[0])
    integration.attrs["sparse_seed_count"] = int(seed_bundle["sparse_seed_count"])
    integration.attrs["use_edge_support_seeds"] = bool(USE_EDGE_SUPPORT_SEEDS)
    integration.attrs["edge_axis_scale"] = float(edge_axis_scale)
    support_offsets_config, support_weights_config = _edge_support_offsets_and_weights()
    integration.attrs["edge_support_config_offsets_full_res"] = [float(v) for v in support_offsets_config]
    integration.attrs["edge_support_config_weights"] = [float(v) for v in support_weights_config]
    integration.attrs["edge_support_require_center_active"] = bool(EDGE_SUPPORT_REQUIRE_CENTER_ACTIVE)
    integration.attrs["seed_out_of_bounds"] = int((~seed_in_bounds_np).sum())
    diagnostics = {
        "seed_out_of_bounds": int((~seed_in_bounds_np).sum()),
        "velocity_oob": int(diag_velocity_oob.sum()),
        "velocity_nonfinite": int(diag_velocity_nonfinite.sum()),
        "velocity_zero": int(diag_velocity_zero.sum()),
        "attract_oob": int(diag_attract_oob.sum()),
        "candidate_nonfinite": int(diag_candidate_nonfinite.sum()),
        "validity_oob": int(diag_validity_oob.sum()),
        "validity_nonfinite": int(diag_validity_nonfinite.sum()),
        "low_validity": int(diag_low_validity.sum()),
    }
    for key, value in diagnostics.items():
        integration.attrs[f"diagnostic_total_{key}"] = int(value)
    return {
        "group": integration,
        "seed_count": int(seeds_world.shape[0]),
        "dense_seed_count": int(seeds_world.shape[0]),
        "sparse_seed_count": int(seed_bundle["sparse_seed_count"]),
        "active_endpoints": int(endpoint_active.sum()),
        "requested_distance": requested_distance,
        "target_distance": target_distance,
        "step_sizes": step_sizes,
        "diagnostics": diagnostics,
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


def _collapse_edge_support_samples(integration_group, sampled_points, sampled_active):
    if "support_source_sparse_index" not in integration_group:
        return sampled_points, sampled_active

    sampled_points = np.asarray(sampled_points, dtype=np.float32)
    sampled_active = np.asarray(sampled_active, dtype=bool)
    source_sparse_index = np.asarray(integration_group["support_source_sparse_index"], dtype=np.int64)
    support_weight = np.asarray(integration_group["support_weight"], dtype=np.float32)
    support_is_center = np.asarray(integration_group["support_is_center"], dtype=bool)
    sparse_seed_count = int(integration_group.attrs.get(
        "sparse_seed_count",
        int(source_sparse_index.max()) + 1 if source_sparse_index.size else 0,
    ))

    if sampled_points.shape[1] != source_sparse_index.shape[0]:
        raise RuntimeError(
            "Support seed metadata does not match sampled streamline count: "
            f"{source_sparse_index.shape[0]} vs {sampled_points.shape[1]}"
        )
    if support_weight.shape[0] != source_sparse_index.shape[0]:
        raise RuntimeError("support_weight length does not match support_source_sparse_index.")
    if support_is_center.shape[0] != source_sparse_index.shape[0]:
        raise RuntimeError("support_is_center length does not match support_source_sparse_index.")

    sparse_points = np.full((sampled_points.shape[0], sparse_seed_count, 3), -1.0, dtype=np.float32)
    sparse_active = np.zeros((sampled_points.shape[0], sparse_seed_count), dtype=bool)
    require_center = bool(integration_group.attrs.get(
        "edge_support_require_center_active",
        EDGE_SUPPORT_REQUIRE_CENTER_ACTIVE,
    ))

    for sparse_idx in range(sparse_seed_count):
        support_indices = np.flatnonzero(source_sparse_index == sparse_idx)
        if support_indices.size == 0:
            continue
        center_indices = support_indices[support_is_center[support_indices]]
        for step_idx in range(sampled_points.shape[0]):
            active_support = support_indices[sampled_active[step_idx, support_indices]]
            if active_support.size == 0:
                continue
            if require_center and (
                center_indices.size == 0
                or not bool(sampled_active[step_idx, center_indices].any())
            ):
                continue
            points = sampled_points[step_idx, active_support]
            finite = np.isfinite(points).all(axis=1)
            if not bool(finite.any()):
                continue
            active_support = active_support[finite]
            points = points[finite]
            weights = support_weight[active_support].astype(np.float32, copy=True)
            positive = weights > 0.0
            if not bool(positive.any()):
                continue
            points = points[positive]
            weights = weights[positive]
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue
            sparse_points[step_idx, sparse_idx] = (points * weights[:, None]).sum(axis=0) / weight_sum
            sparse_active[step_idx, sparse_idx] = True

    sparse_points[~sparse_active] = -1.0
    return sparse_points, sparse_active


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
    sampled_points, sampled_active = _collapse_edge_support_samples(
        integration_group,
        sampled_points,
        sampled_active,
    )
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
            "integration_method": _integration_method(),
            "trace_validity_threshold": float(TRACE_VALIDITY_THRESHOLD),
            "surface_attract_mode": _surface_attract_mode(),
            "vector_merge_method": _vector_merge_method(),
        },
    )
    return output_dir / output_uuid


def _cleanup_temp_output_zarr(output_path):
    if OUTPUT_ZARR_PATH is not None or RUN_OUTPUT_DIR is not None:
        return
    output_path = Path(output_path)
    if not output_path.exists():
        return
    shutil.rmtree(output_path)


def _store_iteration_geometry(output_root, bboxes, edge):
    geometry = output_root.require_group("geometry")
    bboxes = np.asarray(bboxes, dtype=np.int64).reshape(-1, 6)
    edge = np.asarray(edge, dtype=np.float32).reshape(-1, 3)
    if bboxes.shape[0] > 0:
        geometry.create_dataset(
            "bboxes",
            data=bboxes,
            chunks=(min(int(bboxes.shape[0]), 4096), 6),
        )
    if edge.shape[0] > 0:
        geometry.create_dataset(
            "edge_zyx",
            data=edge,
            chunks=(min(int(edge.shape[0]), 4096), 3),
        )


def run_rowcol_bbox_inference(
    model,
    model_config,
    volume_array,
    surface,
    bboxes,
    crop_size,
    cond_direction,
    zarr_root,
    stored_zyxs,
    valid,
    distributed_context=None,
    payload_tag=None,
):
    context = distributed_context or _DistributedContext()
    batch_specs = _batch_specs_for_count(len(bboxes), int(BATCH_SIZE))
    assignment = _rank_batch_assignment_summary(len(batch_specs), context.world_size)
    window_min, window_shape = _merged_window_from_bboxes(bboxes)

    if context.is_rank0 and zarr_root is None:
        raise ValueError("zarr_root is required on rank 0.")
    if context.is_rank0:
        zarr_root.attrs["window_min_zyx"] = [int(v) for v in window_min]
        zarr_root.attrs["window_shape_zyx"] = [int(v) for v in window_shape]
        zarr_root.attrs["crop_size_zyx"] = [int(v) for v in crop_size]
        zarr_root.attrs["cond_direction"] = str(cond_direction)
        zarr_root.attrs["grow_direction"] = str(GROW_DIRECTION)
        zarr_root.attrs["vector_merge_method"] = _vector_merge_method()
        zarr_root.attrs["distributed_world_size"] = int(context.world_size)
        zarr_root.attrs["distributed_backend"] = str(context.backend)

    amp_enabled, amp_dtype = _model_amp_settings(model_config)
    model.eval()

    merger = None
    if context.is_rank0:
        merger = _SparseChunkOutputMerger(
            zarr_root,
            window_min,
            window_shape,
            crop_size,
            mmap_dir=MERGE_OUTPUTS_MMAP_DIR,
            vector_merge_method=VECTOR_MERGE_METHOD,
        )

    total_voxelized_bbox_voxels = 0
    if not context.is_distributed:
        for batch_spec in tqdm(batch_specs, desc="Row/col bbox inference", unit="batch"):
            payload = _run_bbox_batch(
                model,
                volume_array,
                surface,
                bboxes,
                crop_size,
                cond_direction,
                stored_zyxs,
                valid,
                amp_enabled,
                amp_dtype,
                batch_spec,
            )
            _accumulate_batch_payload(merger, payload)
            total_voxelized_bbox_voxels += int(payload["voxelized_bbox_voxels"])
        return {
            "avg_group": merger.finalize(),
            "voxelized_bbox_voxels": int(total_voxelized_bbox_voxels),
            "batch_assignment": assignment,
            "rank_batch_count": len(batch_specs),
        }

    payload_dir = _prepare_distributed_payload_dir(context, payload_tag)
    try:
        rounds = (len(batch_specs) + context.world_size - 1) // context.world_size
        round_iter = tqdm(
            range(rounds),
            desc="Distributed row/col bbox inference",
            unit="round",
            disable=not context.is_rank0,
        )
        rank_batch_count = 0
        for round_idx in round_iter:
            batch_index = int(round_idx) * context.world_size + context.rank
            payload_ref = None
            if batch_index < len(batch_specs):
                rank_batch_count += 1
                payload = _run_bbox_batch(
                    model,
                    volume_array,
                    surface,
                    bboxes,
                    crop_size,
                    cond_direction,
                    stored_zyxs,
                    valid,
                    amp_enabled,
                    amp_dtype,
                    batch_specs[batch_index],
                )
                payload_ref = _write_payload_ref(payload, payload_dir, context)
                del payload

            gathered = _gather_object_to_rank0(payload_ref, context)
            if context.is_rank0:
                for item_ref in sorted((p for p in gathered if p is not None), key=lambda p: int(p["batch_index"])):
                    item = _read_payload_ref(item_ref)
                    _accumulate_batch_payload(merger, item)
                    total_voxelized_bbox_voxels += int(item["voxelized_bbox_voxels"])
                    _cleanup_payload_ref(item_ref)
                    del item

        if context.is_rank0:
            return {
                "avg_group": merger.finalize(),
                "voxelized_bbox_voxels": int(total_voxelized_bbox_voxels),
                "batch_assignment": assignment,
                "rank_batch_count": rank_batch_count,
            }
        return {
            "avg_group": None,
            "voxelized_bbox_voxels": 0,
            "batch_assignment": assignment,
            "rank_batch_count": rank_batch_count,
        }
    finally:
        if context.is_rank0:
            shutil.rmtree(payload_dir, ignore_errors=True)


def _run_streamline_iteration(
    model,
    model_config,
    volume_array,
    crop_size,
    input_tifxyz_path,
    iteration_idx,
    total_iterations,
    output_tifxyz_dir,
    distributed_context=None,
):
    context = distributed_context or _DistributedContext()
    stored_zyxs, valid = _load_surface_zyx(input_tifxyz_path)
    points_zyx = stored_zyxs[valid]
    if points_zyx.size == 0:
        raise RuntimeError("No valid tifxyz points found.")

    cond_direction, _ = _get_growth_context(GROW_DIRECTION)
    bbox_integration_margin = _bbox_integration_margin()
    bboxes, edge = get_cond_edge_bboxes(
        stored_zyxs,
        cond_direction,
        crop_size,
        overlap_frac=OVERLAP_FRAC,
        cond_valid=valid,
        integration_margin=bbox_integration_margin,
    )
    surface = read_tifxyz(input_tifxyz_path)
    edge_axis_scale = surface._scale[0] if _get_direction_spec(cond_direction)["axis"] == "col" else surface._scale[1]
    output_path = _output_zarr_path(iteration_idx)
    expected_output_tifxyz_path = None
    if context.is_rank0:
        expected_output_tifxyz_path = Path(output_tifxyz_dir) / _output_tifxyz_uuid(
            iteration_idx=iteration_idx,
            total_iterations=total_iterations,
        )
    expected_output_tifxyz_path = Path(
        _broadcast_object_from_rank0(
            None if expected_output_tifxyz_path is None else str(expected_output_tifxyz_path),
            context,
        )
    )
    result = None
    try:
        output_root = None
        if context.is_rank0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_root = zarr.open_group(str(output_path), mode="w")
            output_root.attrs["input_tifxyz_path"] = str(input_tifxyz_path)
            output_root.attrs["iteration"] = int(iteration_idx)
            output_root.attrs["total_iterations"] = int(total_iterations)
            output_root.attrs["volume_path"] = str(VOLUME_PATH)
            output_root.attrs["volume_scale"] = int(VOLUME_SCALE)
            output_root.attrs["checkpoint_path"] = str(CHECKPOINT_PATH)
            output_root.attrs["valid_points"] = int(points_zyx.shape[0])
            output_root.attrs["bboxes"] = int(len(bboxes))
            output_root.attrs["distributed_world_size"] = int(context.world_size)
            output_root.attrs["distributed_backend"] = str(context.backend)
            _store_iteration_geometry(output_root, bboxes, edge)

        inference_result = run_rowcol_bbox_inference(
            model,
            model_config,
            volume_array,
            surface,
            bboxes,
            crop_size,
            cond_direction,
            output_root,
            stored_zyxs,
            valid,
            distributed_context=context,
            payload_tag=f"iter_{int(iteration_idx):02d}",
        )

        if not context.is_rank0:
            output_tifxyz_path = _wait_for_tifxyz_output(expected_output_tifxyz_path)
            return {
                "input_tifxyz_path": Path(input_tifxyz_path),
                "output_tifxyz_path": Path(output_tifxyz_path),
                "output_zarr_path": Path(output_path),
                "rank_batch_count": int(inference_result["rank_batch_count"]),
            }

        output_root.attrs["distributed_batch_assignment"] = inference_result["batch_assignment"]
        output_root.attrs["distributed_rank0_batch_count"] = int(inference_result["rank_batch_count"])
        output_group = inference_result["avg_group"]
        integration_result = integrate_streamlines_from_edge(
            output_group,
            edge,
            edge_axis_scale,
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
        voxelized_bbox_voxels = int(inference_result["voxelized_bbox_voxels"])
        output_root.attrs["voxelized_bbox_voxels"] = int(voxelized_bbox_voxels)
        output_root.attrs["output_tifxyz_path"] = str(output_tifxyz_path)

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
        print(f"bbox_integration_margin: {float(bbox_integration_margin)}")
        print(f"vector_merge_method: {_vector_merge_method()}")
        print(f"valid_points: {int(points_zyx.shape[0])}")
        print(f"bboxes: {len(bboxes)}")
        print(f"voxelized_bbox_voxels: {voxelized_bbox_voxels}")
        print(f"tifxyz_voxel_step: {float(TIFXYZ_VOXEL_STEP)}")
        print(f"tifxyz_steps: {int(TIFXYZ_STEPS)}")
        print(f"integration_steps: {len(integration_result['step_sizes'])}")
        print(f"integration_step_size: {float(INTEGRATION_STEP_SIZE)}")
        print(f"integration_method: {_integration_method()}")
        print(f"surface_attract_mode: {_surface_attract_mode()}")
        print(f"integration_requested_distance: {integration_result['requested_distance']}")
        print(f"integration_target_distance: {integration_result['target_distance']}")
        print(f"integration_actual_steps: {integration_result['step_sizes']}")
        print(f"integration_seed_count: {integration_result['seed_count']}")
        print(f"integration_dense_seed_count: {integration_result['dense_seed_count']}")
        print(f"integration_sparse_seed_count: {integration_result['sparse_seed_count']}")
        print(f"edge_support_offsets_full_res: {list(EDGE_SUPPORT_OFFSETS_FULL_RES)}")
        print(f"edge_support_weights: {list(EDGE_SUPPORT_WEIGHTS)}")
        print(f"edge_support_require_center_active: {bool(EDGE_SUPPORT_REQUIRE_CENTER_ACTIVE)}")
        print(f"integration_active_endpoints: {integration_result['active_endpoints']}")
        print(f"integration_failure_diagnostics: {integration_result['diagnostics']}")

        if SHOW_NAPARI:
            voxelized_bboxes = upsample_voxelize_tifxyz_surface_in_bboxes(
                surface,
                bboxes,
                crop_size,
                stored_zyxs=stored_zyxs,
                valid=valid,
            )
            show_streamline_geometry_napari(
                points_zyx,
                edge,
                bboxes,
                voxelized_bboxes=voxelized_bboxes,
                integration_group=integration_result["group"],
            )
        result = {
            "input_tifxyz_path": Path(input_tifxyz_path),
            "output_tifxyz_path": Path(output_tifxyz_path),
            "output_zarr_path": Path(output_path),
            "integration_summary": {
                key: value
                for key, value in integration_result.items()
                if key != "group"
            },
            "valid_points": int(points_zyx.shape[0]),
            "bboxes": int(len(bboxes)),
            "voxelized_bbox_voxels": int(voxelized_bbox_voxels),
            "cond_direction": str(cond_direction),
            "grow_direction": str(GROW_DIRECTION),
            "distributed_world_size": int(context.world_size),
            "distributed_backend": str(context.backend),
            "distributed_batch_assignment": inference_result["batch_assignment"],
        }
        return result
    finally:
        if context.is_rank0:
            _cleanup_temp_output_zarr(output_path)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wt", encoding="utf-8") as fp:
        json.dump(_json_safe(payload), fp, indent=2, sort_keys=True)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run row/column-conditioned neural streamline inference from a tifxyz surface."
    )
    parser.add_argument("--tifxyz-path", required=True, help="Input tifxyz surface directory.")
    parser.add_argument("--checkpoint-path", required=True, help="Trained neural tracing checkpoint.")
    parser.add_argument(
        "--volume-path",
        default="auto",
        help="CT volume Zarr path, or 'auto' to resolve from checkpoint datasets.",
    )
    parser.add_argument(
        "--volume-scale",
        type=int,
        default=None,
        help="Volume pyramid scale. Defaults to the matched checkpoint dataset scale when --volume-path=auto.",
    )
    parser.add_argument("--volume-cache-dir", default="/tmp/vesuvius-volume-cache")
    parser.add_argument("--volume-cache-retry-seconds", type=float, default=60.0)
    parser.add_argument("--output-dir", required=True, help="Run output directory.")
    parser.add_argument("--grow-direction", default="left", choices=sorted(_DIRECTION_SPECS))
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--tifxyz-steps", type=int, default=2)
    parser.add_argument(
        "--tifxyz-voxel-step",
        type=float,
        default=None,
        help="Full-resolution voxels between tifxyz samples. Defaults to 1 / meta.json scale.",
    )
    parser.add_argument(
        "--tifxyz-voxel-size-um",
        type=float,
        default=None,
        help="Physical voxel size written to output tifxyz metadata. Defaults to first '<N>um' in volume path.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--distributed-backend",
        default="auto",
        choices=("auto", "nccl", "gloo"),
        help="torch.distributed backend for torchrun mode. Defaults to NCCL for CUDA, otherwise Gloo.",
    )
    parser.add_argument("--tta", dest="use_tta", action="store_true", default=True)
    parser.add_argument("--no-tta", dest="use_tta", action="store_false")
    parser.add_argument("--compile", dest="compile_model", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    parser.add_argument("--save-each-iteration", dest="save_each_iteration", action="store_true", default=True)
    parser.add_argument("--final-only", dest="save_each_iteration", action="store_false")
    parser.add_argument("--show-napari", action="store_true", help="Open the debug Napari overlay after each iteration.")
    return parser.parse_args(argv)


def _apply_runtime_config(args, resolved_volume_path, resolved_volume_scale, tifxyz_voxel_step, runtime_device=None):
    global TIFXYZ_PATH, VOLUME_PATH, VOLUME_SCALE, VOLUME_CACHE_DIR, VOLUME_CACHE_RETRY_SECONDS
    global CHECKPOINT_PATH, BATCH_SIZE, DEVICE, COMPILE_MODEL, OUTPUT_TIFXYZ_DIR
    global OUTPUT_TIFXYZ_VOXEL_SIZE_UM, RUN_OUTPUT_DIR, RUN_TIMESTAMP, USE_TTA
    global GROW_DIRECTION, TIFXYZ_VOXEL_STEP, TIFXYZ_STEPS, NUM_ITERATIONS
    global SAVE_EACH_ITERATION_TIFXYZ, SHOW_NAPARI

    TIFXYZ_PATH = str(Path(args.tifxyz_path).resolve())
    CHECKPOINT_PATH = str(Path(args.checkpoint_path).resolve())
    VOLUME_PATH = str(resolved_volume_path)
    VOLUME_SCALE = int(resolved_volume_scale)
    VOLUME_CACHE_DIR = str(args.volume_cache_dir)
    VOLUME_CACHE_RETRY_SECONDS = float(args.volume_cache_retry_seconds)
    BATCH_SIZE = int(args.batch_size)
    DEVICE = str(args.device if runtime_device is None else runtime_device)
    COMPILE_MODEL = bool(args.compile_model)
    RUN_OUTPUT_DIR = Path(args.output_dir).resolve()
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_TIFXYZ_DIR = str(RUN_OUTPUT_DIR / "tifxyz")
    OUTPUT_TIFXYZ_VOXEL_SIZE_UM = args.tifxyz_voxel_size_um
    USE_TTA = bool(args.use_tta)
    GROW_DIRECTION = str(args.grow_direction)
    TIFXYZ_VOXEL_STEP = float(tifxyz_voxel_step)
    TIFXYZ_STEPS = int(args.tifxyz_steps)
    NUM_ITERATIONS = int(args.num_iterations)
    SAVE_EACH_ITERATION_TIFXYZ = bool(args.save_each_iteration)
    SHOW_NAPARI = bool(args.show_napari)


def run_from_args(args):
    distributed_context = None
    try:
        distributed_context = _initialize_distributed(args)
        effective_device = distributed_context.device or args.device
        if str(effective_device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {effective_device!r}, but torch.cuda.is_available() is false.")

        num_iterations = int(args.num_iterations)
        if num_iterations <= 0:
            raise ValueError("--num-iterations must be >= 1.")
        if int(args.batch_size) <= 0:
            raise ValueError("--batch-size must be >= 1.")

        model, model_config = load_checkpoint(args.checkpoint_path)
        resolved_volume_path, resolved_volume_scale, matched_dataset = resolve_volume_path_from_config(
            args.tifxyz_path,
            model_config,
            requested_volume_path=args.volume_path,
            requested_volume_scale=args.volume_scale,
        )
        tifxyz_voxel_step = (
            derive_tifxyz_voxel_step(args.tifxyz_path)
            if args.tifxyz_voxel_step is None
            else float(args.tifxyz_voxel_step)
        )
        _apply_runtime_config(
            args,
            resolved_volume_path,
            resolved_volume_scale,
            tifxyz_voxel_step,
            runtime_device=effective_device,
        )

        run_output_dir = Path(RUN_OUTPUT_DIR)
        if distributed_context.is_rank0:
            run_output_dir.mkdir(parents=True, exist_ok=True)
            (run_output_dir / "zarr").mkdir(parents=True, exist_ok=True)
            (run_output_dir / "tifxyz").mkdir(parents=True, exist_ok=True)
        if distributed_context.is_distributed:
            dist.barrier()

        crop_size = _crop_size_from_config(model_config)
        summary = None
        if distributed_context.is_rank0:
            summary = {
                "started_at": RUN_TIMESTAMP,
                "input_tifxyz_path": TIFXYZ_PATH,
                "checkpoint_path": CHECKPOINT_PATH,
                "volume_path": VOLUME_PATH,
                "volume_scale": VOLUME_SCALE,
                "matched_dataset": matched_dataset,
                "output_dir": str(run_output_dir),
                "args": vars(args),
                "crop_size": list(crop_size),
                "batch_size": BATCH_SIZE,
                "device": DEVICE,
                "compile_model": COMPILE_MODEL,
                "use_tta": USE_TTA,
                "grow_direction": GROW_DIRECTION,
                "num_iterations": NUM_ITERATIONS,
                "tifxyz_steps": TIFXYZ_STEPS,
                "tifxyz_voxel_step": TIFXYZ_VOXEL_STEP,
                "tifxyz_voxel_size_um": _output_tifxyz_voxel_size_um(),
                "save_each_iteration_tifxyz": SAVE_EACH_ITERATION_TIFXYZ,
                "world_size": int(distributed_context.world_size),
                "rank_mode": "torchrun" if distributed_context.is_distributed else "single",
                "distributed_backend": str(distributed_context.backend),
                "rank": int(distributed_context.rank),
                "local_rank": int(distributed_context.local_rank),
                "iterations": [],
            }
            _write_json(run_output_dir / "resolved_config.json", model_config)
            _write_json(run_output_dir / "summary.json", summary)

        model.to(DEVICE)
        if COMPILE_MODEL:
            model = torch.compile(model)
        volume_array = _open_volume_array(VOLUME_PATH, VOLUME_SCALE)
        if distributed_context.is_rank0:
            summary["volume_shape"] = list(getattr(volume_array, "shape", ()))
            summary["volume_dtype"] = str(getattr(volume_array, "dtype", "unknown"))
            summary["volume_chunks"] = list(getattr(volume_array, "chunks", ()))
            _write_json(run_output_dir / "summary.json", summary)

        current_tifxyz_path = Path(TIFXYZ_PATH)
        final_output_tifxyz_path = None
        temp_tifxyz_dir = None
        try:
            if distributed_context.is_rank0 and num_iterations > 1 and not SAVE_EACH_ITERATION_TIFXYZ:
                temp_tifxyz_dir = tempfile.TemporaryDirectory(prefix="infer_streamline_tifxyz_")
            for iteration_idx in range(1, num_iterations + 1):
                is_final_iteration = iteration_idx == num_iterations
                if is_final_iteration or SAVE_EACH_ITERATION_TIFXYZ or not distributed_context.is_rank0:
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
                    distributed_context=distributed_context,
                )
                current_tifxyz_path = result["output_tifxyz_path"]
                final_output_tifxyz_path = current_tifxyz_path
                if distributed_context.is_rank0:
                    summary["iterations"].append(result)
                    summary["final_output_tifxyz_path"] = str(final_output_tifxyz_path)
                    summary["last_iteration_zarr_path"] = str(result["output_zarr_path"])
                    _write_json(run_output_dir / "summary.json", summary)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if distributed_context.is_rank0:
                summary["completed_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                _write_json(run_output_dir / "summary.json", summary)
                print(f"num_iterations: {num_iterations}")
                print(f"save_each_iteration_tifxyz: {bool(SAVE_EACH_ITERATION_TIFXYZ)}")
                print(f"final_output_tifxyz_path: {final_output_tifxyz_path}")
                print(f"summary_path: {run_output_dir / 'summary.json'}")
                return summary
            return None
        finally:
            if temp_tifxyz_dir is not None:
                temp_tifxyz_dir.cleanup()
    finally:
        _destroy_distributed(distributed_context)


def main(argv=None):
    return run_from_args(_parse_args(argv))


if __name__ == "__main__":
    main()
