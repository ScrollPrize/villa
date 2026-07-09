from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vesuvius.neural_tracing.fiber_trace.dataset import _SpatialChannelView


@dataclass(frozen=True)
class ZarrChunkRequest:
    store: Any
    store_identity: str
    key: str


def store_identity(store: Any) -> str:
    url = getattr(store, "_url", None)
    cache_dir = getattr(store, "_cache_dir", None)
    if url is not None:
        return f"{type(store).__module__}.{type(store).__name__}:{url}:{cache_dir}"
    path = getattr(store, "path", None)
    if path is not None:
        return f"{type(store).__module__}.{type(store).__name__}:{path}"
    return f"{type(store).__module__}.{type(store).__name__}:{repr(store)}"


def array_chunks_zyx(array: Any) -> tuple[int, int, int] | None:
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return None
    chunks_tuple = tuple(int(v) for v in chunks)
    if isinstance(array, _SpatialChannelView):
        return array.chunks
    if len(chunks_tuple) == 4:
        return chunks_tuple[1:]
    if len(chunks_tuple) == 3:
        return chunks_tuple
    return None


def is_remote_cached_store(store: Any) -> bool:
    return getattr(store, "_url", None) is not None and getattr(store, "_cache_dir", None) is not None


def chunk_key(array: Any, chunk_zyx: tuple[int, int, int]) -> str | None:
    key_fn = getattr(array, "_chunk_key", None)
    if not callable(key_fn):
        return None
    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    chunks = tuple(int(v) for v in getattr(array, "chunks", ()))
    if len(shape) == 3:
        return str(key_fn(chunk_zyx))
    if len(shape) == 4 and len(chunks) == 4:
        return str(key_fn((0,) + chunk_zyx))
    return None


def chunk_requests_for_coords(array: Any, coords_zyx: np.ndarray, valid_mask: np.ndarray) -> list[ZarrChunkRequest]:
    store = getattr(array, "store", None)
    chunks = getattr(array, "chunks", None)
    if store is None or chunks is None or not is_remote_cached_store(store):
        return []
    chunks_tuple = tuple(int(v) for v in chunks)
    if len(chunks_tuple) == 4:
        chunks_zyx = chunks_tuple[1:]
    elif len(chunks_tuple) == 3:
        chunks_zyx = chunks_tuple
    else:
        return []
    if min(chunks_zyx) <= 0:
        return []

    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(coords_zyx).all(axis=-1)
    if not bool(valid.any()):
        return []
    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    spatial_shape = shape[-3:]
    coords = np.asarray(coords_zyx, dtype=np.float64)
    valid &= (coords[..., 0] >= 0.0) & (coords[..., 0] <= float(spatial_shape[0] - 1))
    valid &= (coords[..., 1] >= 0.0) & (coords[..., 1] <= float(spatial_shape[1] - 1))
    valid &= (coords[..., 2] >= 0.0) & (coords[..., 2] <= float(spatial_shape[2] - 1))
    if not bool(valid.any()):
        return []

    base = np.floor(coords[valid]).astype(np.int64)
    high = np.minimum(base + 1, np.asarray(spatial_shape, dtype=np.int64) - 1)
    corners = []
    for dz in (0, 1):
        z = base[:, 0] if dz == 0 else high[:, 0]
        for dy in (0, 1):
            y = base[:, 1] if dy == 0 else high[:, 1]
            for dx in (0, 1):
                x = base[:, 2] if dx == 0 else high[:, 2]
                corners.append(np.stack([z, y, x], axis=1))
    corner_coords = np.concatenate(corners, axis=0)
    chunk_idx = corner_coords // np.asarray(chunks_zyx, dtype=np.int64)
    unique = np.unique(chunk_idx, axis=0)
    identity = store_identity(store)
    requests: list[ZarrChunkRequest] = []
    for chunk in unique:
        key = chunk_key(array, tuple(int(v) for v in chunk))
        if key is None:
            continue
        requests.append(ZarrChunkRequest(store=store, store_identity=identity, key=key))
    return requests


def read_array_points(array: Any, z: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    if len(shape) == 4:
        values = array[(0, z, y, x)]
    else:
        values = array[(z, y, x)]
    return np.asarray(values, dtype=np.float32)


def sample_array_trilinear(array: Any, coords_zyx: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    spatial_shape = shape[-3:]
    out = np.zeros(coords_zyx.shape[:2], dtype=np.float32)
    coords = np.asarray(coords_zyx, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool).copy()
    valid &= np.isfinite(coords).all(axis=-1)
    valid &= (coords[..., 0] >= 0.0) & (coords[..., 0] <= float(spatial_shape[0] - 1))
    valid &= (coords[..., 1] >= 0.0) & (coords[..., 1] <= float(spatial_shape[1] - 1))
    valid &= (coords[..., 2] >= 0.0) & (coords[..., 2] <= float(spatial_shape[2] - 1))
    if not bool(valid.any()):
        return out

    sample_coords = coords[valid]
    base = np.floor(sample_coords).astype(np.int64)
    high = np.minimum(base + 1, np.asarray(spatial_shape, dtype=np.int64) - 1)
    frac = (sample_coords - base.astype(np.float64)).astype(np.float32)
    fz, fy, fx = frac[:, 0], frac[:, 1], frac[:, 2]

    z0, y0, x0 = base[:, 0], base[:, 1], base[:, 2]
    z1, y1, x1 = high[:, 0], high[:, 1], high[:, 2]
    c000 = read_array_points(array, z0, y0, x0)
    c001 = read_array_points(array, z0, y0, x1)
    c010 = read_array_points(array, z0, y1, x0)
    c011 = read_array_points(array, z0, y1, x1)
    c100 = read_array_points(array, z1, y0, x0)
    c101 = read_array_points(array, z1, y0, x1)
    c110 = read_array_points(array, z1, y1, x0)
    c111 = read_array_points(array, z1, y1, x1)

    c00 = c000 * (1.0 - fx) + c001 * fx
    c01 = c010 * (1.0 - fx) + c011 * fx
    c10 = c100 * (1.0 - fx) + c101 * fx
    c11 = c110 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c01 * fy
    c1 = c10 * (1.0 - fy) + c11 * fy
    out[valid] = c0 * (1.0 - fz) + c1 * fz
    return out
