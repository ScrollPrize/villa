#!/usr/bin/env python3
"""Self-render a source TIFXYZ from its raw CT volume through rclone.

Only informative source and mapped-overlap tiles are rendered. XYZ coordinates
and normals follow vc_render_tifxyz conventions; raw CT is sampled trilinearly
in ZYX order. The source render measures its annotation-canvas offset, while
source-on-target versus target self-renders independently verify the 3D map.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import tifffile

from vesuvius.utils.cli import HyphenUnderscoreParser

from .core import (
    Surface,
    SurfaceMapper,
    apply_affine,
    bilinear_field_tile,
)
from .estimate_canvas_offset import measure_render_shift
from .io import load_surface, read_image
from .prepare_canvas_offset_evidence import (
    DEFAULT_AWS_CREDENTIALS_FILE,
    DEFAULT_OPEN_DATA_ROOT,
    ZarrLevel,
    _load_aws_credentials,
    _remote_join,
    _run_rclone,
    inspect_zarr,
)


# Raw scan volumes are not public: there is no usable default, callers
# must name their own rclone remote when provenance remapping is needed.
DEFAULT_SOURCE_RAW_ROOT: Optional[str] = None


def _sample_grid_field(
    field: np.ndarray,
    valid: np.ndarray,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample a stored TIFXYZ field at arbitrary grid positions."""

    height, width = field.shape
    y0 = np.clip(np.floor(grid_y).astype(np.int64), 0, height - 1)
    x0 = np.clip(np.floor(grid_x).astype(np.int64), 0, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = grid_y - y0
    wx = grid_x - x0
    v00 = valid[y0, x0]
    v01 = valid[y0, x1]
    v10 = valid[y1, x0]
    v11 = valid[y1, x1]
    sampled_valid = v00 & v01 & v10 & v11
    values = (
        field[y0, x0] * (1.0 - wy) * (1.0 - wx)
        + field[y0, x1] * (1.0 - wy) * wx
        + field[y1, x0] * wy * (1.0 - wx)
        + field[y1, x1] * wy * wx
    )
    sampled_valid &= np.isfinite(values)
    return values.astype(np.float64, copy=False), sampled_valid


def surface_tile_geometry(
    surface: Surface,
    output_shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate XYZ, normal, and validity for one rendered canvas tile."""

    row_start, row_end, col_start, col_end = bounds
    assert surface.valid is not None
    xyz_fields = []
    valid = None
    for field in (surface.x, surface.y, surface.z):
        values, field_valid = bilinear_field_tile(
            field,
            surface.valid,
            output_shape,
            row_start,
            row_end,
            col_start,
            col_end,
        )
        xyz_fields.append(values)
        valid = field_valid if valid is None else valid & field_valid
    xyz = np.stack(xyz_fields, axis=-1)

    stored_height, stored_width = surface.shape
    output_height, output_width = output_shape
    grid_y_1d = (
        (np.arange(row_start, row_end, dtype=np.float64) + 0.5)
        * stored_height
        / output_height
    )
    grid_x_1d = (
        (np.arange(col_start, col_end, dtype=np.float64) + 0.5)
        * stored_width
        / output_width
    )
    grid_y, grid_x = np.meshgrid(grid_y_1d, grid_x_1d, indexing="ij")
    # grid_normal clamps its evaluation location before taking +/-1 samples.
    normal_y = np.clip(grid_y, 1.0, max(float(stored_height - 3), 1.0))
    normal_x = np.clip(grid_x, 1.0, max(float(stored_width - 3), 1.0))
    plus_x = []
    minus_x = []
    plus_y = []
    minus_y = []
    normal_valid = np.ones(grid_y.shape, dtype=bool)
    for field in (surface.x, surface.y, surface.z):
        px, px_valid = _sample_grid_field(
            field, surface.valid, normal_y, normal_x + 1.0
        )
        mx, mx_valid = _sample_grid_field(
            field, surface.valid, normal_y, normal_x - 1.0
        )
        py, py_valid = _sample_grid_field(
            field, surface.valid, normal_y + 1.0, normal_x
        )
        my, my_valid = _sample_grid_field(
            field, surface.valid, normal_y - 1.0, normal_x
        )
        plus_x.append(px)
        minus_x.append(mx)
        plus_y.append(py)
        minus_y.append(my)
        normal_valid &= px_valid & mx_valid & py_valid & my_valid
    x_vector = np.stack(plus_x, axis=-1) - np.stack(minus_x, axis=-1)
    y_vector = np.stack(plus_y, axis=-1) - np.stack(minus_y, axis=-1)
    x_norm = np.linalg.norm(x_vector, axis=-1)
    y_norm = np.linalg.norm(y_vector, axis=-1)
    normal_valid &= (x_norm > 1e-9) & (y_norm > 1e-9)
    x_vector /= np.maximum(x_norm[..., None], 1e-9)
    y_vector /= np.maximum(y_norm[..., None], 1e-9)
    normals = np.cross(x_vector, y_vector)
    normal_norm = np.linalg.norm(normals, axis=-1)
    normal_valid &= normal_norm > 1e-9
    normals /= np.maximum(normal_norm[..., None], 1e-9)
    valid = valid & normal_valid & np.all(np.isfinite(xyz), axis=-1)
    return xyz, normals, valid


def surface_geometry_at_uv(
    surface: Surface,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    input_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample source XYZ and vc_render-style normals at continuous UV."""

    assert surface.valid is not None
    xyz_fields: list[np.ndarray] = []
    valid = np.asarray(input_valid, dtype=bool).copy()
    for field in (surface.x, surface.y, surface.z):
        values, field_valid = _sample_grid_field(
            field, surface.valid, grid_y, grid_x
        )
        xyz_fields.append(values)
        valid &= field_valid
    xyz = np.stack(xyz_fields, axis=-1)

    height, width = surface.shape
    normal_y = np.clip(grid_y, 1.0, max(float(height - 3), 1.0))
    normal_x = np.clip(grid_x, 1.0, max(float(width - 3), 1.0))
    plus_x: list[np.ndarray] = []
    minus_x: list[np.ndarray] = []
    plus_y: list[np.ndarray] = []
    minus_y: list[np.ndarray] = []
    for field in (surface.x, surface.y, surface.z):
        px, px_valid = _sample_grid_field(
            field, surface.valid, normal_y, normal_x + 1.0
        )
        mx, mx_valid = _sample_grid_field(
            field, surface.valid, normal_y, normal_x - 1.0
        )
        py, py_valid = _sample_grid_field(
            field, surface.valid, normal_y + 1.0, normal_x
        )
        my, my_valid = _sample_grid_field(
            field, surface.valid, normal_y - 1.0, normal_x
        )
        plus_x.append(px)
        minus_x.append(mx)
        plus_y.append(py)
        minus_y.append(my)
        valid &= px_valid & mx_valid & py_valid & my_valid
    x_vector = np.stack(plus_x, axis=-1) - np.stack(minus_x, axis=-1)
    y_vector = np.stack(plus_y, axis=-1) - np.stack(minus_y, axis=-1)
    x_norm = np.linalg.norm(x_vector, axis=-1)
    y_norm = np.linalg.norm(y_vector, axis=-1)
    valid &= (x_norm > 1e-9) & (y_norm > 1e-9)
    x_vector /= np.maximum(x_norm[..., None], 1e-9)
    y_vector /= np.maximum(y_norm[..., None], 1e-9)
    normals = np.cross(x_vector, y_vector)
    normal_norm = np.linalg.norm(normals, axis=-1)
    valid &= normal_norm > 1e-9
    normals /= np.maximum(normal_norm[..., None], 1e-9)
    valid &= np.all(np.isfinite(xyz), axis=-1)
    return xyz, normals, valid


def mapped_source_tile_geometry(
    source: Surface,
    target: Surface,
    mapped_rows: np.ndarray,
    mapped_cols: np.ndarray,
    mapped_valid: np.ndarray,
    output_shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
    affine: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return source and target geometry on one target-canvas tile."""

    row0, row1, col0, col1 = bounds
    source_rows, rows_valid = bilinear_field_tile(
        mapped_rows,
        mapped_valid,
        output_shape,
        row0,
        row1,
        col0,
        col1,
    )
    source_cols, cols_valid = bilinear_field_tile(
        mapped_cols,
        mapped_valid,
        output_shape,
        row0,
        row1,
        col0,
        col1,
    )
    source_xyz, source_normals, source_valid = surface_geometry_at_uv(
        source,
        source_rows,
        source_cols,
        rows_valid & cols_valid,
    )
    target_xyz, target_normals, target_valid = surface_tile_geometry(
        target, output_shape, bounds
    )
    transformed = apply_affine(source_xyz.reshape(-1, 3), affine).reshape(
        source_xyz.shape
    )
    distance = np.linalg.norm(transformed - target_xyz, axis=-1)
    valid = (
        source_valid
        & target_valid
        & np.isfinite(distance)
        & (distance <= float(max_distance))
    )
    return source_xyz, source_normals, target_xyz, target_normals, valid


def select_tiles(
    image: np.ndarray,
    tile_size: int,
    max_tiles: int,
    min_coverage: float,
) -> list[tuple[int, int, int, int]]:
    """Select deterministic, textured, non-overlapping source-canvas tiles."""

    height, width = image.shape
    tile = min(tile_size, height, width)
    filtered = gaussian_filter(image.astype(np.float32), 1.0) - gaussian_filter(
        image.astype(np.float32), 8.0
    )
    candidates: list[tuple[float, int, int, int, int]] = []
    for row in range(0, height - tile + 1, tile):
        for col in range(0, width - tile + 1, tile):
            block = image[row : row + tile, col : col + tile]
            if float((block > 0).mean()) < min_coverage:
                continue
            texture = float(
                np.std(filtered[row : row + tile, col : col + tile])
            )
            if texture > 1e-6:
                candidates.append(
                    (texture, row, row + tile, col, col + tile)
                )
    candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
    grid_size = 4
    per_bin = max(1, math.ceil(max_tiles / (grid_size * grid_size)))
    bins: dict[tuple[int, int], list[tuple[float, int, int, int, int]]] = {}
    for candidate in candidates:
        _, row0, row1, col0, col1 = candidate
        center_y = (row0 + row1) / 2.0
        center_x = (col0 + col1) / 2.0
        bin_y = min(int(center_y * grid_size / height), grid_size - 1)
        bin_x = min(int(center_x * grid_size / width), grid_size - 1)
        bins.setdefault((bin_y, bin_x), []).append(candidate)
    selected: list[tuple[float, int, int, int, int]] = []
    for key in sorted(bins):
        selected.extend(bins[key][:per_bin])
    if len(selected) < max_tiles:
        selected_set = set(selected)
        selected.extend(
            item
            for item in candidates
            if item not in selected_set
        )
    selected = selected[:max_tiles]
    return sorted(
        (row0, row1, col0, col1)
        for _, row0, row1, col0, col1 in selected
    )


def select_overlap_tiles(
    image: np.ndarray,
    mapped_rows: np.ndarray,
    mapped_valid: np.ndarray,
    tile_size: int,
    max_tiles: int,
    min_coverage: float,
) -> list[tuple[int, int, int, int]]:
    """Select textured target tiles covered by the source surface."""

    height, width = image.shape
    tile = min(tile_size, height, width)
    filtered = gaussian_filter(image.astype(np.float32), 1.0) - gaussian_filter(
        image.astype(np.float32), 8.0
    )
    candidates: list[tuple[float, int, int, int, int]] = []
    for row in range(0, height - tile + 1, tile):
        for col in range(0, width - tile + 1, tile):
            block = image[row : row + tile, col : col + tile]
            if float((block > 0).mean()) < min_coverage:
                continue
            _, coverage_valid = bilinear_field_tile(
                mapped_rows,
                mapped_valid,
                image.shape,
                row,
                row + tile,
                col,
                col + tile,
            )
            if float(coverage_valid.mean()) < min_coverage:
                continue
            texture = float(
                np.std(filtered[row : row + tile, col : col + tile])
            )
            if texture > 1e-6:
                candidates.append(
                    (texture, row, row + tile, col, col + tile)
                )
    candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
    grid_size = 4
    per_bin = max(1, math.ceil(max_tiles / (grid_size * grid_size)))
    bins: dict[tuple[int, int], list[tuple[float, int, int, int, int]]] = {}
    for candidate in candidates:
        _, row0, row1, col0, col1 = candidate
        key = (
            min(int(((row0 + row1) / 2) * grid_size / height), grid_size - 1),
            min(int(((col0 + col1) / 2) * grid_size / width), grid_size - 1),
        )
        bins.setdefault(key, []).append(candidate)
    selected: list[tuple[float, int, int, int, int]] = []
    for key in sorted(bins):
        selected.extend(bins[key][:per_bin])
    if len(selected) < max_tiles:
        selected_set = set(selected)
        selected.extend(item for item in candidates if item not in selected_set)
    return sorted(
        (row0, row1, col0, col1)
        for _, row0, row1, col0, col1 in selected[:max_tiles]
    )


def _relative_chunk_key(
    info: ZarrLevel, chunk_zyx: tuple[int, int, int]
) -> str:
    separator = info.metadata.get("dimension_separator", ".")
    return separator.join(str(value) for value in chunk_zyx)


def required_chunks(
    xyz: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
    offsets: Sequence[float],
    info: ZarrLevel,
) -> set[tuple[int, int, int]]:
    """Conservatively bound raw chunks needed for trilinear samples."""

    scale_xyz = np.asarray(info.scale_zyx[::-1], dtype=np.float64)
    chunk_zyx = np.asarray(info.chunks, dtype=np.int64)
    shape_zyx = np.asarray(info.shape, dtype=np.int64)
    endpoint_indices = (int(np.argmin(offsets)), int(np.argmax(offsets)))
    endpoints = []
    for index in endpoint_indices:
        offset = offsets[index]
        points_xyz = (xyz + normals * float(offset)) / scale_xyz
        points_zyx = points_xyz[..., ::-1][valid]
        if not points_zyx.size:
            continue
        floor = np.floor(points_zyx).astype(np.int64)
        floor = np.clip(floor, 0, shape_zyx - 2)
        endpoints.extend((floor, floor + 1))
    if not endpoints:
        return set()
    # Every normal-line sample lies coordinate-wise between the two endpoint
    # samples. A chunk-space box around both endpoints and their +1
    # trilinear neighbours is therefore conservative for every offset.
    all_endpoints = np.concatenate(endpoints, axis=0)
    minimum = np.min(all_endpoints, axis=0) // chunk_zyx
    maximum = np.max(all_endpoints, axis=0) // chunk_zyx
    result = {
        (z_chunk, y_chunk, x_chunk)
        for z_chunk in range(int(minimum[0]), int(maximum[0]) + 1)
        for y_chunk in range(int(minimum[1]), int(maximum[1]) + 1)
        for x_chunk in range(int(minimum[2]), int(maximum[2]) + 1)
    }
    return result


def ensure_chunks(
    info: ZarrLevel,
    chunks: set[tuple[int, int, int]],
    cache_dir: Path,
    workers: int,
) -> tuple[int, int, set[tuple[int, int, int]]]:
    """Fetch present chunks and treat omitted sparse Zarr chunks as fill."""

    missing = [
        chunk
        for chunk in sorted(chunks)
        if not (cache_dir / _relative_chunk_key(info, chunk)).is_file()
    ]
    if not missing:
        return 0, 0, set()
    if info.metadata.get("dimension_separator", ".") != "/":
        raise ValueError(
            "raw self-render currently requires '/' Zarr chunk keys"
        )

    present: set[tuple[int, int, int]] = set()
    by_z: dict[int, set[tuple[int, int, int]]] = {}
    for chunk in missing:
        by_z.setdefault(chunk[0], set()).add(chunk)
    for z_chunk, wanted in sorted(by_z.items()):
        listing = _run_rclone(
            [
                "lsf",
                _remote_join(info.remote, info.level, str(z_chunk)),
                "--recursive",
                "--files-only",
            ]
        ).decode("utf-8").splitlines()
        available = {
            tuple(int(value) for value in item.strip("/").split("/"))
            for item in listing
            if item.strip() and len(item.strip("/").split("/")) == 2
        }
        present.update(
            chunk
            for chunk in wanted
            if (chunk[1], chunk[2]) in available
        )
    sparse = set(missing) - present
    if not present:
        return 0, 0, sparse
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False
    ) as handle:
        files_path = Path(handle.name)
        for chunk in sorted(present):
            handle.write(_relative_chunk_key(info, chunk) + "\n")
    try:
        _run_rclone(
            [
                "copy",
                _remote_join(info.remote, info.level),
                str(cache_dir),
                "--files-from",
                str(files_path),
                "--no-traverse",
                "--transfers",
                str(workers),
                "--checkers",
                str(workers),
            ],
            timeout_seconds=900,
        )
    finally:
        files_path.unlink(missing_ok=True)
    still_missing = [
        chunk
        for chunk in present
        if not (cache_dir / _relative_chunk_key(info, chunk)).is_file()
    ]
    if still_missing:
        raise RuntimeError(
            f"rclone did not materialize {len(still_missing)} required raw "
            f"chunks; first missing: {still_missing[0]}"
        )
    downloaded_bytes = sum(
        (cache_dir / _relative_chunk_key(info, chunk)).stat().st_size
        for chunk in present
    )
    return len(present), downloaded_bytes, sparse


class RawChunkSampler:
    def __init__(
        self,
        info: ZarrLevel,
        cache_dir: Path,
        sparse_chunks: Optional[set[tuple[int, int, int]]] = None,
        max_cached_chunks: int = 96,
    ) -> None:
        self.info = info
        self.cache_dir = cache_dir
        self.max_cached_chunks = max_cached_chunks
        self.sparse_chunks = sparse_chunks or set()
        self._cache: OrderedDict[
            tuple[int, int, int], np.ndarray
        ] = OrderedDict()

    def _chunk(self, index: tuple[int, int, int]) -> np.ndarray:
        cached = self._cache.pop(index, None)
        if cached is not None:
            self._cache[index] = cached
            return cached
        if index in self.sparse_chunks:
            return np.full(
                self.info.chunks,
                self.info.metadata.get("fill_value") or 0,
                dtype=np.dtype(self.info.metadata["dtype"]),
            )
        path = self.cache_dir / _relative_chunk_key(self.info, index)
        payload = path.read_bytes()
        dtype = np.dtype(self.info.metadata["dtype"])
        expected = math.prod(self.info.chunks) * dtype.itemsize
        if len(payload) != expected:
            raise ValueError(
                f"raw chunk {path} has {len(payload)} bytes; expected "
                f"{expected}"
            )
        chunk = np.frombuffer(payload, dtype=dtype).reshape(self.info.chunks)
        self._cache[index] = chunk
        while len(self._cache) > self.max_cached_chunks:
            self._cache.popitem(last=False)
        return chunk

    def load_block(
        self, chunks: set[tuple[int, int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Materialize one tile's chunk box for fast C-level interpolation."""

        indices = np.asarray(sorted(chunks), dtype=np.int64)
        if indices.size == 0:
            raise ValueError("cannot load an empty raw chunk block")
        minimum = indices.min(axis=0)
        maximum = indices.max(axis=0)
        chunk_shape = np.asarray(self.info.chunks, dtype=np.int64)
        block_shape = (maximum - minimum + 1) * chunk_shape
        block = np.full(
            tuple(int(value) for value in block_shape),
            self.info.metadata.get("fill_value") or 0,
            dtype=np.dtype(self.info.metadata["dtype"]),
        )
        for chunk_index in map(tuple, indices.tolist()):
            relative = (
                np.asarray(chunk_index, dtype=np.int64) - minimum
            ) * chunk_shape
            slices = tuple(
                slice(int(start), int(start + size))
                for start, size in zip(relative, chunk_shape)
            )
            block[slices] = self._chunk(chunk_index)
        return block, minimum * chunk_shape

    def sample_block(
        self,
        points_xyz: np.ndarray,
        block: np.ndarray,
        origin_zyx: np.ndarray,
    ) -> np.ndarray:
        points_zyx = points_xyz[:, ::-1] / np.asarray(
            self.info.scale_zyx, dtype=np.float64
        )
        points_zyx = np.clip(
            points_zyx,
            0.0,
            np.asarray(self.info.shape, dtype=np.float64) - 1.0,
        )
        points_zyx -= origin_zyx
        return map_coordinates(
            block,
            points_zyx.T,
            output=np.float32,
            order=1,
            mode="nearest",
            prefilter=False,
        )

    def _lattice_values(self, indices_zyx: np.ndarray) -> np.ndarray:
        chunk_shape = np.asarray(self.info.chunks, dtype=np.int64)
        chunk_indices = indices_zyx // chunk_shape
        local = indices_zyx % chunk_shape
        values = np.zeros(indices_zyx.shape[0], dtype=np.float32)
        unique, inverse = np.unique(
            chunk_indices, axis=0, return_inverse=True
        )
        for group, chunk_index in enumerate(unique):
            selected = inverse == group
            key = tuple(int(value) for value in chunk_index)
            chunk = self._chunk(key)
            positions = local[selected]
            values[selected] = chunk[
                positions[:, 0], positions[:, 1], positions[:, 2]
            ]
        return values

    def sample(self, points_xyz: np.ndarray) -> np.ndarray:
        scale_xyz = np.asarray(
            self.info.scale_zyx[::-1], dtype=np.float64
        )
        points_zyx = points_xyz[:, ::-1] / scale_xyz[::-1]
        shape = np.asarray(self.info.shape, dtype=np.int64)
        floor = np.floor(points_zyx).astype(np.int64)
        floor = np.clip(floor, 0, shape - 2)
        fraction = np.clip(points_zyx - floor, 0.0, 1.0)
        output = np.zeros(points_zyx.shape[0], dtype=np.float32)
        for dz, dy, dx in (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ):
            delta = np.asarray((dz, dy, dx), dtype=np.int64)
            weight = (
                (fraction[:, 0] if dz else 1.0 - fraction[:, 0])
                * (fraction[:, 1] if dy else 1.0 - fraction[:, 1])
                * (fraction[:, 2] if dx else 1.0 - fraction[:, 2])
            )
            output += weight * self._lattice_values(floor + delta)
        return output


def _find_raw_volume_remote(
    selection: dict[str, Any], open_data_root: str, volume_id: str
) -> str:
    sample = str(selection["segment"]["s3_path"]).split("/", 1)[0]
    volumes_root = _remote_join(open_data_root, sample, "volumes")
    listing = _run_rclone(
        [
            "lsf",
            volumes_root,
            "--dirs-only",
            "--max-depth",
            "1",
            "--include",
            f"{volume_id}-*.zarr/",
        ]
    ).decode("utf-8").splitlines()
    matches = [item.rstrip("/") for item in listing if item.strip()]
    if len(matches) != 1:
        raise ValueError(
            f"expected one raw volume for {sample}:{volume_id}, got {matches}"
        )
    return _remote_join(volumes_root, matches[0])


def _source_raw_volume_remote(
    source_surface_info: ZarrLevel,
    source_raw_root: Optional[str],
    override: Optional[str],
) -> tuple[str, str]:
    """Resolve the raw CT from the source render's own provenance."""

    if override:
        return override.rstrip("/"), "command line"
    source_zarr = str(source_surface_info.attrs.get("source_zarr") or "")
    if not source_zarr:
        raise ValueError(
            "source surface Zarr has no source_zarr provenance; pass "
            "--source-raw-volume"
        )
    normalized = "/" + source_zarr.strip("/")
    for marker in ("/esrf/", "/dls/"):
        index = normalized.find(marker)
        if index >= 0:
            if not source_raw_root:
                raise ValueError(
                    f"source_zarr provenance {source_zarr!r} needs a raw "
                    "CT root; pass --source-raw-rclone-root (your rclone "
                    "remote holding the raw scan volumes) or an explicit "
                    "--source-raw-volume"
                )
            return (
                _remote_join(source_raw_root, normalized[index + 1 :]),
                f"source surface Zarr: {source_zarr}",
            )
    raise ValueError(
        f"cannot map source_zarr {source_zarr!r} to rclone; pass "
        "--source-raw-volume"
    )


def _resolution_from_raw_path(remote: str) -> float:
    match = re.search(r"(?:^|[_.\/-])(\d+(?:\.\d+)?)um(?=[_.\/-]|$)", remote)
    if match is None:
        raise ValueError(
            f"cannot infer voxel resolution from {remote!r}; pass the "
            "matching --source-resolution-um or --target-resolution-um"
        )
    return float(match.group(1))


def _volume_resolution(selection: dict[str, Any], volume_id: str) -> float:
    matches = [
        float(item["resolution_um"])
        for item in selection.get("surface_volumes") or []
        if str(item.get("volume_id")) == str(volume_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one resolution for volume {volume_id}, got {matches}"
        )
    return matches[0]


def matched_offsets(
    source_offsets: Sequence[float],
    source_resolution_um: float,
    target_resolution_um: float,
) -> list[float]:
    """Return target-voxel offsets covering the same physical slab."""

    physical = np.asarray(source_offsets, dtype=np.float64) * float(
        source_resolution_um
    )
    lower = float(np.min(physical) / target_resolution_um)
    upper = float(np.max(physical) / target_resolution_um)
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12):
        return [lower]
    samples = max(2, int(math.ceil(upper - lower)) + 1)
    return np.linspace(lower, upper, samples).tolist()


def _annotation_offsets(
    annotation_path: Path, source_surface_info: ZarrLevel
) -> list[float]:
    match = re.search(r"_max_(\d+)_(\d+)\.tif$", annotation_path.name)
    if match is None:
        # The prepared level image drops the original suffix. Its sibling
        # retained the original name.
        candidates = list(annotation_path.parent.glob("*_max_*_*.tif"))
        if len(candidates) == 1:
            match = re.search(
                r"_max_(\d+)_(\d+)\.tif$", candidates[0].name
            )
    if match is None:
        raise ValueError(
            "cannot infer annotation max layers; expected *_max_A_B.tif"
        )
    first, last = int(match.group(1)), int(match.group(2))
    center = int(source_surface_info.level_zero_metadata["shape"][0]) // 2
    slice_step = float(source_surface_info.attrs.get("slice_step", 1.0))
    if not math.isfinite(slice_step) or slice_step <= 0:
        raise ValueError(f"invalid source surface slice_step: {slice_step}")
    return [
        float(index - center) * slice_step
        for index in range(first, last + 1)
    ]


def _source_reference_render(
    manifest: dict[str, Any], source_center_path: Path
) -> tuple[Path, str]:
    """Choose the published raster used to audit the source canvas.

    Some ink datasets ship a dedicated annotation maximum while others only
    expose the surface-volume Zarr.  In the latter case its exact center plane
    is still a published raster on the annotation canvas; it simply provides
    no through-surface tolerance and must not be described as max evidence.
    """

    annotation_render = manifest.get("annotation_render")
    if annotation_render:
        return Path(annotation_render), "shipped-annotation-maximum"
    return source_center_path, "surface-volume-center"


def _measurement_is_translation(
    measurement: dict[str, Any], max_corner_drift_px: float
) -> bool:
    field = measurement.get("shift_field") or {}
    return bool(
        int(field.get("inlier_tiles", 0)) >= 3
        and math.isfinite(float(field.get("max_corner_drift_px", math.inf)))
        and float(field["max_corner_drift_px"]) <= max_corner_drift_px
    )


def _canvas_shape_check(
    full_shape: np.ndarray, render_shape: np.ndarray
) -> dict[str, Any]:
    scale_yx = np.asarray(full_shape, dtype=np.float64) / np.asarray(
        render_shape, dtype=np.float64
    )
    return {
        "annotation_render_shape_yx": np.asarray(render_shape)
        .astype(int)
        .tolist(),
        "tifxyz_full_resolution_shape_yx": np.asarray(full_shape)
        .astype(int)
        .tolist(),
        "assumed_scale_yx": scale_yx.tolist(),
        "scale_anisotropy": float(abs(scale_yx[0] / scale_yx[1] - 1.0)),
        "note": (
            "offsets are converted to full-resolution pixels assuming the "
            "annotation raster spans the full TIFXYZ canvas; a crop/scale "
            "disagreement between raster and canvas shows up as unequal "
            "per-axis scales but is not otherwise detected by this tool"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = HyphenUnderscoreParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--zarr-level", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-tiles", type=int, default=16)
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument(
        "--maximum-source-disagreement-full-px", type=float, default=3.0
    )
    parser.add_argument("--geometry-tolerance-px", type=float, default=0.25)
    parser.add_argument("--max-corner-drift-px", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--source-raw-volume",
        help="explicit rclone remote for the raw CT behind the source TIFXYZ",
    )
    parser.add_argument(
        "--source-surface-zarr",
        help=(
            "explicit rclone remote for source surface-volume metadata; "
            "useful when the same metadata has a public mirror"
        ),
    )
    parser.add_argument(
        "--target-raw-volume",
        help="explicit rclone remote for the raw CT behind the target TIFXYZ",
    )
    parser.add_argument(
        "--source-raw-rclone-root",
        default=DEFAULT_SOURCE_RAW_ROOT,
        help=(
            "rclone remote:path holding the raw scan volumes; required "
            "when the source provenance must be remapped and no explicit "
            "--source-raw-volume is given"
        ),
    )
    parser.add_argument("--source-resolution-um", type=float)
    parser.add_argument("--target-resolution-um", type=float)
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        help="shared root for downloaded raw Zarr chunks",
    )
    parser.add_argument(
        "--open-data-rclone-root",
        default=DEFAULT_OPEN_DATA_ROOT,
        help=(
            "rclone remote:path of the public Vesuvius open-data bucket "
            "(default: anonymous inline S3 remote, no rclone config needed)"
        ),
    )
    parser.add_argument(
        "--aws-credentials-file",
        type=Path,
        default=DEFAULT_AWS_CREDENTIALS_FILE,
        help=(
            "optional shell-format AWS exports loaded only when "
            "AWS_ACCESS_KEY_ID is absent (default: none)"
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--source-only",
        action="store_true",
        help=(
            "validate only the source annotation canvas against the raw CT "
            "behind the source TIFXYZ; does not require a completed transfer"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _render_values(
    xyz: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
    offsets: Sequence[float],
    sampler: RawChunkSampler,
    chunks: set[tuple[int, int, int]],
) -> np.ndarray:
    output = np.zeros(valid.shape, dtype=np.float32)
    selected = valid.ravel()
    if not np.any(selected):
        return output
    block, origin = sampler.load_block(chunks)
    points = xyz.reshape(-1, 3)[selected]
    directions = normals.reshape(-1, 3)[selected]
    maximum = np.zeros(points.shape[0], dtype=np.float32)
    for offset in offsets:
        values = sampler.sample_block(
            points + directions * float(offset), block, origin
        )
        maximum = np.maximum(maximum, values)
    output.ravel()[selected] = maximum
    return output


def _to_uint8(values: np.ndarray, sampler: "RawChunkSampler") -> np.ndarray:
    """Convert rendered raw-CT values to uint8 without integer wrap-around."""

    array = np.asarray(values, dtype=np.float64)
    dtype = np.dtype(sampler.info.metadata["dtype"])
    if dtype.kind in "ui":
        peak = float(np.iinfo(dtype).max)
        if peak > 255.0:
            array = array * (255.0 / peak)
    return np.clip(np.rint(array), 0.0, 255.0).astype(np.uint8)


def _open_output(
    path: Path, shape: tuple[int, int], overwrite: bool
) -> np.memmap:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        path.unlink()
    image = tifffile.memmap(path, shape=shape, dtype=np.uint8, metadata=None)
    image[:] = 0
    return image


def _source_canvas_measurements(
    center_image: np.ndarray,
    max_image: np.ndarray,
    source_center: np.ndarray,
    annotation: np.ndarray,
    full_shape: np.ndarray,
    args: argparse.Namespace,
    *,
    allow_annotation_only: bool = False,
) -> dict[str, Any]:
    """Measure and approve the source raster-to-TIFXYZ canvas transform."""

    center_error: Optional[str] = None
    try:
        center_measurement = measure_render_shift(
            center_image,
            source_center,
            tile_size=args.tile_size,
            min_coverage=args.min_coverage,
            max_tiles=args.max_tiles,
        )
    except ValueError as error:
        center_measurement = None
        center_error = f"{type(error).__name__}: {error}"
    annotation_measurement = measure_render_shift(
        max_image,
        annotation,
        tile_size=args.tile_size,
        min_coverage=args.min_coverage,
        max_tiles=args.max_tiles,
    )
    render_shape = np.asarray(annotation.shape)

    def full_offset(measurement: dict[str, Any]) -> list[float]:
        return (
            np.asarray(measurement["shift_yx"]) * full_shape / render_shape
        ).tolist()

    center_offset_full = (
        None if center_measurement is None else full_offset(center_measurement)
    )
    annotation_offset_full = full_offset(annotation_measurement)
    source_disagreement = (
        None
        if center_offset_full is None
        else float(
            np.linalg.norm(
                np.asarray(center_offset_full)
                - np.asarray(annotation_offset_full)
            )
        )
    )
    center_translation_valid = bool(
        center_measurement is not None
        and _measurement_is_translation(
            center_measurement, args.max_corner_drift_px
        )
    )
    annotation_translation_valid = _measurement_is_translation(
        annotation_measurement, args.max_corner_drift_px
    )
    both_checks_approved = bool(
        source_disagreement is not None
        and source_disagreement <= args.maximum_source_disagreement_full_px
        and center_translation_valid
        and annotation_translation_valid
    )
    annotation_only_approved = bool(
        allow_annotation_only
        and center_measurement is None
        and annotation_translation_valid
    )
    approved = both_checks_approved or annotation_only_approved
    return {
        "center_measurement": center_measurement,
        "center_error": center_error,
        "annotation_measurement": annotation_measurement,
        "center_offset_full": center_offset_full,
        "annotation_offset_full": annotation_offset_full,
        "source_disagreement": source_disagreement,
        "center_translation_valid": center_translation_valid,
        "annotation_translation_valid": annotation_translation_valid,
        "approval_basis": (
            "center-and-annotation"
            if both_checks_approved
            else (
                "annotation-maximum-only"
                if annotation_only_approved
                else "not-approved"
            )
        ),
        "approved": approved,
    }


def _run_source_only(
    args: argparse.Namespace,
    credentials_path: Optional[Path],
    case_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    output_dir: Path,
    source_center_path: Path,
    source_center: np.ndarray,
    annotation_path: Path,
    annotation: np.ndarray,
    source_surface: Surface,
    source_surface_info: ZarrLevel,
) -> int:
    """Validate label-canvas coordinates without any cross-volume mapping."""

    source_raw_remote, source_raw_provenance = _source_raw_volume_remote(
        source_surface_info,
        args.source_raw_rclone_root,
        args.source_raw_volume,
    )
    print(f"Raw source volume: {source_raw_remote}", flush=True)
    source_raw_info = inspect_zarr(source_raw_remote, args.zarr_level)
    if source_raw_info.metadata.get("compressor") is not None:
        raise ValueError("raw self-render currently requires uncompressed Zarr")
    if source_raw_info.metadata.get("filters") is not None:
        raise ValueError("raw self-render currently requires no Zarr filters")
    if source_raw_info.metadata.get("order", "C") != "C":
        raise ValueError("raw self-render currently requires C-order Zarr")

    source_tiles = select_tiles(
        annotation, args.tile_size, args.max_tiles, args.min_coverage
    )
    if not source_tiles:
        raise ValueError("no textured source tiles to self-render")
    source_reference_kind = (
        "shipped-annotation-maximum"
        if annotation_path != source_center_path
        else "surface-volume-center"
    )
    source_offsets = (
        _annotation_offsets(annotation_path, source_surface_info)
        if source_reference_kind == "shipped-annotation-maximum"
        else [0.0]
    )
    source_resolution_um = (
        float(args.source_resolution_um)
        if args.source_resolution_um is not None
        else _resolution_from_raw_path(source_raw_remote)
    )

    source_plan: list[dict[str, Any]] = []
    source_union: set[tuple[int, int, int]] = set()
    for index, bounds in enumerate(source_tiles, start=1):
        xyz, normals, valid = surface_tile_geometry(
            source_surface, annotation.shape, bounds
        )
        chunks = required_chunks(
            xyz, normals, valid, [0.0, *source_offsets], source_raw_info
        )
        source_union.update(chunks)
        source_plan.append(
            {
                "bounds_yxyx": list(bounds),
                "valid_pixels": int(valid.sum()),
                "source_chunks": len(chunks),
            }
        )
        print(f"Plan source tile {index}/{len(source_tiles)}", flush=True)

    plan_document = {
        "mode": "source-only",
        "source_raw_volume": source_raw_remote,
        "source_raw_provenance": source_raw_provenance,
        "source_resolution_um": source_resolution_um,
        "raw_level": int(source_raw_info.level),
        "source_canvas_shape_yx": list(annotation.shape),
        "source_offsets_full_voxels": source_offsets,
        "source_tiles": source_plan,
        "source_unique_required_chunks": len(source_union),
    }
    (output_dir / "plan.json").write_text(
        json.dumps(plan_document, indent=2) + "\n", encoding="utf-8"
    )
    if args.plan_only:
        print(json.dumps(plan_document, indent=2))
        return 0

    cache_root = (
        args.raw_cache_dir.resolve()
        if args.raw_cache_dir
        else output_dir / "raw-chunks"
    )
    source_cache = (
        cache_root
        / Path(source_raw_remote).name
        / source_raw_info.level
    )
    downloaded, downloaded_bytes, sparse_chunks = ensure_chunks(
        source_raw_info, source_union, source_cache, args.workers
    )
    print(
        f"Raw cache ready: {downloaded} new chunks, "
        f"{downloaded_bytes / 1024**3:.2f} GiB",
        flush=True,
    )

    center_output = output_dir / "source-self-center.tif"
    max_output = output_dir / "source-self-annotation-max.tif"
    valid_output = output_dir / "source-self.valid.tif"
    output_paths = (center_output, max_output, valid_output)
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"{existing_outputs[0]} exists; pass --overwrite"
        )
    if args.overwrite:
        for path in output_paths:
            path.unlink(missing_ok=True)
    center_image = _open_output(center_output, annotation.shape, args.overwrite)
    max_image = _open_output(max_output, annotation.shape, args.overwrite)
    valid_image = _open_output(valid_output, annotation.shape, args.overwrite)
    source_sampler = RawChunkSampler(
        source_raw_info, source_cache, sparse_chunks
    )
    for index, bounds in enumerate(source_tiles, start=1):
        row0, row1, col0, col1 = bounds
        xyz, normals, valid = surface_tile_geometry(
            source_surface, annotation.shape, bounds
        )
        chunks = required_chunks(
            xyz, normals, valid, [0.0, *source_offsets], source_raw_info
        )
        center_image[row0:row1, col0:col1] = _to_uint8(
            _render_values(
                xyz, normals, valid, [0.0], source_sampler, chunks
            ),
            source_sampler,
        )
        max_image[row0:row1, col0:col1] = _to_uint8(
            _render_values(
                xyz, normals, valid, source_offsets, source_sampler, chunks
            ),
            source_sampler,
        )
        valid_image[row0:row1, col0:col1] = valid.astype(np.uint8) * 255
        print(f"Rendered source tile {index}/{len(source_tiles)}", flush=True)
    for image in (center_image, max_image, valid_image):
        image.flush()

    full_shape = np.asarray(source_surface.full_resolution_shape)
    measurements = _source_canvas_measurements(
        np.asarray(center_image),
        np.asarray(max_image),
        source_center,
        annotation,
        full_shape,
        args,
        allow_annotation_only=(
            source_reference_kind == "shipped-annotation-maximum"
        ),
    )
    report = {
        "tool": "self_render_tifxyz.py",
        "mode": "source-only",
        "transport": "rclone",
        "aws_credentials_file": (
            str(credentials_path) if credentials_path else None
        ),
        "source_tifxyz": str(case_dir / "hf" / "source.tifxyz"),
        "source_raw_volume": source_raw_remote,
        "source_raw_provenance": source_raw_provenance,
        "source_resolution_um": source_resolution_um,
        "raw_level": int(source_raw_info.level),
        "annotation_render": str(annotation_path),
        "source_reference_kind": source_reference_kind,
        "source_center_render": str(source_center_path),
        "self_center_render": str(center_output),
        "self_annotation_max_render": str(max_output),
        "source_validity": str(valid_output),
        "source_offsets_full_voxels": source_offsets,
        "downloaded_chunks": downloaded,
        "downloaded_bytes": downloaded_bytes,
        "canvas_shape_check": _canvas_shape_check(
            full_shape, np.asarray(annotation.shape)
        ),
        "center_canvas_check": {
            "canvas_offset_yx_full_resolution_px": measurements[
                "center_offset_full"
            ],
            "measurement": measurements["center_measurement"],
            "error": measurements["center_error"],
        },
        "annotation_canvas_offset": {
            "canvas_offset_yx_full_resolution_px": measurements[
                "annotation_offset_full"
            ],
            "measurement": measurements["annotation_measurement"],
            "authoritative": measurements["approved"],
        },
        "two_sided_geometry_check": None,
        "approval": {
            "approved": measurements["approved"],
            "scope": "source-canvas-only",
            "source_canvas_approved": measurements["approved"],
            "geometry_approved": None,
            "basis": measurements["approval_basis"],
            "source_center_max_disagreement_full_px": measurements[
                "source_disagreement"
            ],
            "maximum_source_disagreement_full_px": (
                args.maximum_source_disagreement_full_px
            ),
            "source_center_translation_valid": measurements[
                "center_translation_valid"
            ],
            "source_annotation_translation_valid": measurements[
                "annotation_translation_valid"
            ],
            "max_corner_drift_px": args.max_corner_drift_px,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    manifest["source_canvas_self_render_report"] = str(report_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["annotation_canvas_offset"], indent=2))
    print(json.dumps(report["approval"], indent=2))
    print(f"Wrote {report_path}")
    return 0 if measurements["approved"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    credentials_path = _load_aws_credentials(
        args.aws_credentials_file.expanduser()
        if args.aws_credentials_file
        else None
    )
    case_dir = args.case_dir.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else case_dir / "renders" / "offset-evidence" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(
        (case_dir / "selection.json").read_text(encoding="utf-8")
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else case_dir / "renders" / "self-render"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_center = [
        item
        for item in manifest["comparisons"]
        if item["name"] == "exact-center"
    ]
    if len(exact_center) != 1:
        reason = manifest.get("source_center", {}).get(
            "reason", "the evidence manifest has no exact-center comparison"
        )
        raise ValueError(
            f"self-render requires a Z-preserving exact-center source: {reason}"
        )
    source_center_path = Path(exact_center[0]["source_render"])
    source_center = read_image(source_center_path)
    annotation_path, source_reference_kind = _source_reference_render(
        manifest, source_center_path
    )
    annotation = read_image(annotation_path)
    if annotation.shape != source_center.shape:
        raise ValueError(
            f"annotation and source center shapes differ: "
            f"{annotation.shape} != {source_center.shape}"
        )
    source_tifxyz = case_dir / "hf" / "source.tifxyz"
    target_candidates = sorted(
        (case_dir / "open-data").glob("*um-*.tifxyz"),
        key=lambda path: float(path.name.split("um-", 1)[0]),
    )
    if not target_candidates:
        raise ValueError(f"{case_dir} contains no target TIFXYZ")
    target_tifxyz = target_candidates[0]
    target_volume_id = target_tifxyz.name.split("um-", 1)[1].split(".", 1)[0]
    source_surface = load_surface(source_tifxyz)
    target_surface = load_surface(target_tifxyz)
    source_surface_info = inspect_zarr(
        args.source_surface_zarr or manifest["source_zarr_remote"],
        args.zarr_level,
    )
    if args.source_only:
        return _run_source_only(
            args,
            credentials_path,
            case_dir,
            manifest_path,
            manifest,
            output_dir,
            source_center_path,
            source_center,
            annotation_path,
            annotation,
            source_surface,
            source_surface_info,
        )
    target_center_path = Path(
        next(
            item["target_render"]
            for item in manifest["comparisons"]
            if item["name"] == "exact-center"
        )
    )
    target_center = read_image(target_center_path)
    stage = f"{float(target_tifxyz.name.split('um-', 1)[0]):g}um"
    stage_report_path = case_dir / "results" / f"inklabels-{stage}.report.json"
    stage_report = json.loads(stage_report_path.read_text(encoding="utf-8"))
    affine = np.asarray(stage_report["affine"]["matrix"], dtype=np.float64)
    mapper = SurfaceMapper(
        source_surface,
        affine=affine,
        nearest_vertices=int(stage_report["nearest_vertices"]),
        index_max_distance=float(stage_report["max_distance"]),
    )
    print("Building source→target surface correspondence...", flush=True)
    mapped_rows, mapped_cols, _, mapped_valid = mapper.build_target_uv_map(
        target_surface,
        max_distance=float(stage_report["max_distance"]),
        query_batch_size=int(stage_report["query_batch_size"]),
    )

    source_raw_remote, source_raw_provenance = _source_raw_volume_remote(
        source_surface_info,
        args.source_raw_rclone_root,
        args.source_raw_volume,
    )
    target_raw_remote = (
        args.target_raw_volume.rstrip("/")
        if args.target_raw_volume
        else _find_raw_volume_remote(
            selection, args.open_data_rclone_root, target_volume_id
        )
    )
    print(f"Raw source volume: {source_raw_remote}", flush=True)
    print(f"Raw target volume: {target_raw_remote}", flush=True)
    source_raw_info = inspect_zarr(source_raw_remote, args.zarr_level)
    target_raw_info = (
        source_raw_info
        if target_raw_remote == source_raw_remote
        else inspect_zarr(target_raw_remote, args.zarr_level)
    )
    for raw_info in (source_raw_info, target_raw_info):
        if raw_info.metadata.get("compressor") is not None:
            raise ValueError(
                "raw self-render currently requires uncompressed Zarr"
            )
        if raw_info.metadata.get("filters") is not None:
            raise ValueError("raw self-render currently requires no Zarr filters")
        if raw_info.metadata.get("order", "C") != "C":
            raise ValueError("raw self-render currently requires C-order Zarr")

    source_tiles = select_tiles(
        annotation, args.tile_size, args.max_tiles, args.min_coverage
    )
    target_tiles = select_overlap_tiles(
        target_center,
        mapped_rows,
        mapped_valid,
        args.tile_size,
        args.max_tiles,
        args.min_coverage,
    )
    if not source_tiles or not target_tiles:
        raise ValueError("no textured source/target overlap tiles to self-render")
    source_offsets = (
        _annotation_offsets(annotation_path, source_surface_info)
        if source_reference_kind == "shipped-annotation-maximum"
        else [0.0]
    )
    source_resolution_um = (
        float(args.source_resolution_um)
        if args.source_resolution_um is not None
        else _resolution_from_raw_path(source_raw_remote)
    )
    target_resolution_um = (
        float(args.target_resolution_um)
        if args.target_resolution_um is not None
        else _volume_resolution(selection, target_volume_id)
    )
    target_offsets = matched_offsets(
        source_offsets,
        source_resolution_um,
        target_resolution_um,
    )

    source_plan: list[dict[str, Any]] = []
    target_plan: list[dict[str, Any]] = []
    source_union: set[tuple[int, int, int]] = set()
    target_union: set[tuple[int, int, int]] = set()
    for index, bounds in enumerate(source_tiles, start=1):
        xyz, normals, valid = surface_tile_geometry(
            source_surface, annotation.shape, bounds
        )
        chunks = required_chunks(
            xyz, normals, valid, [0.0, *source_offsets], source_raw_info
        )
        source_union.update(chunks)
        source_plan.append(
            {
                "bounds_yxyx": list(bounds),
                "valid_pixels": int(valid.sum()),
                "source_chunks": len(chunks),
            }
        )
        print(f"Plan source tile {index}/{len(source_tiles)}", flush=True)
    target_geometry: list[tuple[Any, ...]] = []
    for index, bounds in enumerate(target_tiles, start=1):
        geometry = mapped_source_tile_geometry(
            source_surface,
            target_surface,
            mapped_rows,
            mapped_cols,
            mapped_valid,
            target_center.shape,
            bounds,
            affine,
            float(stage_report["max_distance"]),
        )
        source_xyz, source_normals, target_xyz, target_normals, valid = geometry
        source_chunks = required_chunks(
            source_xyz,
            source_normals,
            valid,
            [0.0, *source_offsets],
            source_raw_info,
        )
        target_chunks = required_chunks(
            target_xyz,
            target_normals,
            valid,
            [0.0, *target_offsets],
            target_raw_info,
        )
        source_union.update(source_chunks)
        target_union.update(target_chunks)
        target_geometry.append(
            (*geometry, source_chunks, target_chunks)
        )
        target_plan.append(
            {
                "bounds_yxyx": list(bounds),
                "overlap_pixels": int(valid.sum()),
                "source_chunks": len(source_chunks),
                "target_chunks": len(target_chunks),
            }
        )
        print(f"Plan overlap tile {index}/{len(target_tiles)}", flush=True)
    plan_document = {
        "source_raw_volume": source_raw_remote,
        "source_raw_provenance": source_raw_provenance,
        "target_raw_volume": target_raw_remote,
        "source_resolution_um": source_resolution_um,
        "target_resolution_um": target_resolution_um,
        "raw_level": int(source_raw_info.level),
        "source_canvas_shape_yx": list(annotation.shape),
        "target_canvas_shape_yx": list(target_center.shape),
        "source_offsets_full_voxels": source_offsets,
        "target_matched_offsets_full_voxels": target_offsets,
        "source_tiles": source_plan,
        "target_overlap_tiles": target_plan,
        "source_unique_required_chunks": len(source_union),
        "target_unique_required_chunks": len(target_union),
    }
    (output_dir / "plan.json").write_text(
        json.dumps(plan_document, indent=2) + "\n", encoding="utf-8"
    )
    if args.plan_only:
        print(json.dumps(plan_document, indent=2))
        return 0

    cache_root = (
        args.raw_cache_dir.resolve()
        if args.raw_cache_dir
        else output_dir / "raw-chunks"
    )
    source_cache = (
        cache_root / Path(source_raw_remote).name / source_raw_info.level
    )
    target_cache = (
        cache_root / Path(target_raw_remote).name / target_raw_info.level
    )
    source_downloaded, source_bytes, source_sparse = ensure_chunks(
        source_raw_info, source_union, source_cache, args.workers
    )
    if target_raw_remote == source_raw_remote:
        combined = source_union | target_union
        more_downloaded, more_bytes, combined_sparse = ensure_chunks(
            source_raw_info, combined, source_cache, args.workers
        )
        source_downloaded += more_downloaded
        source_bytes += more_bytes
        source_sparse |= combined_sparse
        target_downloaded, target_bytes = 0, 0
        target_sparse = source_sparse
    else:
        target_downloaded, target_bytes, target_sparse = ensure_chunks(
            target_raw_info, target_union, target_cache, args.workers
        )
    print(
        f"Raw caches ready: {source_downloaded + target_downloaded} new "
        f"chunks, {(source_bytes + target_bytes) / 1024**3:.2f} GiB",
        flush=True,
    )

    center_output = output_dir / "source-self-center.tif"
    max_output = output_dir / "source-self-annotation-max.tif"
    source_valid_output = output_dir / "source-self.valid.tif"
    projected_center_output = output_dir / "source-self-on-target-center.tif"
    target_center_output = output_dir / "target-self-center.tif"
    projected_max_output = output_dir / "source-self-on-target-matched-max.tif"
    target_max_output = output_dir / "target-self-matched-max.tif"
    target_valid_output = output_dir / "target-overlap.valid.tif"
    output_paths = (
        center_output,
        max_output,
        source_valid_output,
        projected_center_output,
        target_center_output,
        projected_max_output,
        target_max_output,
        target_valid_output,
    )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"{existing_outputs[0]} exists; pass --overwrite"
        )
    if args.overwrite:
        for path in output_paths:
            path.unlink(missing_ok=True)
    center_image = _open_output(center_output, annotation.shape, args.overwrite)
    max_image = _open_output(max_output, annotation.shape, args.overwrite)
    source_valid_image = _open_output(source_valid_output, annotation.shape, args.overwrite)
    projected_center = _open_output(projected_center_output, target_center.shape, args.overwrite)
    target_center_image = _open_output(target_center_output, target_center.shape, args.overwrite)
    projected_max = _open_output(projected_max_output, target_center.shape, args.overwrite)
    target_max = _open_output(target_max_output, target_center.shape, args.overwrite)
    target_valid_image = _open_output(target_valid_output, target_center.shape, args.overwrite)
    source_sampler = RawChunkSampler(source_raw_info, source_cache, source_sparse)
    target_sampler = (
        source_sampler
        if target_raw_remote == source_raw_remote
        else RawChunkSampler(target_raw_info, target_cache, target_sparse)
    )
    for index, bounds in enumerate(source_tiles, start=1):
        row0, row1, col0, col1 = bounds
        xyz, normals, valid = surface_tile_geometry(
            source_surface, annotation.shape, bounds
        )
        chunks = required_chunks(xyz, normals, valid, [0.0, *source_offsets], source_raw_info)
        center_image[row0:row1, col0:col1] = _to_uint8(
            _render_values(xyz, normals, valid, [0.0], source_sampler, chunks),
            source_sampler,
        )
        max_image[row0:row1, col0:col1] = _to_uint8(
            _render_values(xyz, normals, valid, source_offsets, source_sampler, chunks),
            source_sampler,
        )
        source_valid_image[row0:row1, col0:col1] = valid.astype(np.uint8) * 255
        print(f"Rendered source tile {index}/{len(source_tiles)}", flush=True)
    for index, (bounds, geometry) in enumerate(zip(target_tiles, target_geometry), start=1):
        row0, row1, col0, col1 = bounds
        source_xyz, source_normals, target_xyz, target_normals, valid, source_chunks, target_chunks = geometry
        projected_center[row0:row1, col0:col1] = _to_uint8(
            _render_values(source_xyz, source_normals, valid, [0.0], source_sampler, source_chunks),
            source_sampler,
        )
        projected_max[row0:row1, col0:col1] = _to_uint8(
            _render_values(source_xyz, source_normals, valid, source_offsets, source_sampler, source_chunks),
            source_sampler,
        )
        target_center_image[row0:row1, col0:col1] = _to_uint8(
            _render_values(target_xyz, target_normals, valid, [0.0], target_sampler, target_chunks),
            target_sampler,
        )
        target_max[row0:row1, col0:col1] = _to_uint8(
            _render_values(target_xyz, target_normals, valid, target_offsets, target_sampler, target_chunks),
            target_sampler,
        )
        target_valid_image[row0:row1, col0:col1] = valid.astype(np.uint8) * 255
        print(f"Rendered overlap tile {index}/{len(target_tiles)}", flush=True)
    for image in (center_image, max_image, source_valid_image, projected_center, target_center_image, projected_max, target_max, target_valid_image):
        image.flush()

    center_measurement = measure_render_shift(
        np.asarray(center_image),
        source_center,
        tile_size=args.tile_size,
        min_coverage=args.min_coverage,
        max_tiles=args.max_tiles,
    )
    annotation_measurement = measure_render_shift(
        np.asarray(max_image),
        annotation,
        tile_size=args.tile_size,
        min_coverage=args.min_coverage,
        max_tiles=args.max_tiles,
    )
    geometry_center_measurement = measure_render_shift(
        np.asarray(target_center_image),
        np.asarray(projected_center),
        tile_size=args.tile_size,
        min_coverage=args.min_coverage,
        max_tiles=args.max_tiles,
    )
    geometry_max_measurement = measure_render_shift(
        np.asarray(target_max),
        np.asarray(projected_max),
        tile_size=args.tile_size,
        min_coverage=args.min_coverage,
        max_tiles=args.max_tiles,
    )
    full_shape = np.asarray(source_surface.full_resolution_shape)
    render_shape = np.asarray(annotation.shape)

    def full_offset(measurement: dict[str, Any]) -> list[float]:
        return (
            np.asarray(measurement["shift_yx"])
            * full_shape
            / render_shape
        ).tolist()

    center_offset_full = full_offset(center_measurement)
    annotation_offset_full = full_offset(annotation_measurement)
    source_disagreement = float(
        np.linalg.norm(
            np.asarray(center_offset_full)
            - np.asarray(annotation_offset_full)
        )
    )
    geometry_center_residual = float(
        np.linalg.norm(geometry_center_measurement["shift_yx"])
    )
    geometry_max_residual = float(
        np.linalg.norm(geometry_max_measurement["shift_yx"])
    )
    source_center_translation_valid = _measurement_is_translation(
        center_measurement, args.max_corner_drift_px
    )
    source_annotation_translation_valid = _measurement_is_translation(
        annotation_measurement, args.max_corner_drift_px
    )
    geometry_valid = all(
        (
            geometry_center_residual <= args.geometry_tolerance_px,
            geometry_max_residual <= args.geometry_tolerance_px,
            geometry_center_measurement["shift_field"]["max_corner_drift_px"]
            <= args.max_corner_drift_px,
            geometry_max_measurement["shift_field"]["max_corner_drift_px"]
            <= args.max_corner_drift_px,
        )
    )
    source_canvas_approved = (
        source_disagreement <= args.maximum_source_disagreement_full_px
        and source_center_translation_valid
        and source_annotation_translation_valid
    )
    approved = source_canvas_approved and geometry_valid
    cross_evidence_path = (
        case_dir / "affines" / "hf-render-canvas-offset-evidence.json"
    )
    cross_comparison = None
    if cross_evidence_path.is_file():
        cross_evidence = json.loads(
            cross_evidence_path.read_text(encoding="utf-8")
        )
        if cross_evidence.get("approved"):
            cross_offset = np.asarray(
                cross_evidence["canvas_offset_yx_full_resolution_px"],
                dtype=np.float64,
            )
            cross_comparison = {
                "report": str(cross_evidence_path),
                "canvas_offset_yx_full_resolution_px": cross_offset.tolist(),
                "disagreement_full_resolution_px": float(
                    np.linalg.norm(
                        cross_offset - np.asarray(annotation_offset_full)
                    )
                ),
            }

    report = {
        "tool": "self_render_tifxyz.py",
        "transport": "rclone",
        "aws_credentials_file": (
            str(credentials_path) if credentials_path else None
        ),
        "source_tifxyz": str(case_dir / "hf" / "source.tifxyz"),
        "source_raw_volume": source_raw_remote,
        "source_raw_provenance": source_raw_provenance,
        "target_raw_volume": target_raw_remote,
        "source_resolution_um": source_resolution_um,
        "target_resolution_um": target_resolution_um,
        "raw_level": int(source_raw_info.level),
        "annotation_render": str(annotation_path),
        "source_reference_kind": source_reference_kind,
        "source_center_render": str(source_center_path),
        "self_center_render": str(center_output),
        "self_annotation_max_render": str(max_output),
        "source_validity": str(source_valid_output),
        "source_on_target_center": str(projected_center_output),
        "target_self_center": str(target_center_output),
        "source_on_target_matched_max": str(projected_max_output),
        "target_self_matched_max": str(target_max_output),
        "target_overlap_validity": str(target_valid_output),
        "source_offsets_full_voxels": source_offsets,
        "target_matched_offsets_full_voxels": target_offsets,
        "downloaded_chunks": source_downloaded + target_downloaded,
        "downloaded_bytes": source_bytes + target_bytes,
        "canvas_shape_check": _canvas_shape_check(full_shape, render_shape),
        "center_canvas_check": {
            "canvas_offset_yx_full_resolution_px": center_offset_full,
            "measurement": center_measurement,
        },
        "annotation_canvas_offset": {
            "canvas_offset_yx_full_resolution_px": annotation_offset_full,
            "measurement": annotation_measurement,
            "authoritative": source_canvas_approved,
        },
        "two_sided_geometry_check": {
            "center": geometry_center_measurement,
            "matched_max": geometry_max_measurement,
        },
        "approval": {
            "approved": approved,
            "source_canvas_approved": source_canvas_approved,
            "geometry_approved": geometry_valid,
            "source_center_max_disagreement_full_px": source_disagreement,
            "maximum_source_disagreement_full_px": (
                args.maximum_source_disagreement_full_px
            ),
            "source_center_translation_valid": (
                source_center_translation_valid
            ),
            "source_annotation_translation_valid": (
                source_annotation_translation_valid
            ),
            "geometry_center_residual_px": geometry_center_residual,
            "geometry_matched_max_residual_px": geometry_max_residual,
            "geometry_tolerance_px": args.geometry_tolerance_px,
            "max_corner_drift_px": args.max_corner_drift_px,
        },
        "cross_version_comparison": cross_comparison,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    manifest["self_render_report"] = str(report_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["annotation_canvas_offset"], indent=2))
    print(json.dumps(report["approval"], indent=2))
    print(f"Wrote {report_path}")
    return 0 if approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
