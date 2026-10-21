"""Embed a volume region with tiles whose blended outputs do not depend on the region.

Tiles sit on one global lattice per axis. Tile ``k`` starts at ``k*stride - margin`` and
contributes only on its support ``[k*stride, k*stride + stride + ramp)``: weights rise
linearly over the first ``ramp`` voxels, are one in between, and fall over the last
``ramp`` voxels. Adjacent supports overlap by exactly ``ramp``, where the two ramps sum
to one, so each output is a fixed convex blend of the tiles covering its global
position; requesting a different region (or slab) reproduces the same values. The
``margin`` voxels at every tile face are context only and never contribute.

Outputs, all channel-last in a zarr v2 group, at global resolution steps:

- ``tokens``: backbone ``x_norm_patchtokens`` (one per patch), blended weighted mean.
- ``tokens_unit``: the same tokens L2-normalized, so dot products are cosine similarities.
- ``dense`` / ``ink``: ``PointDenseHead`` object and ink embeddings at ``dense_stride``
  voxel steps, blended then re-normalized to unit length.

Outputs whose patch/stride block contains only zero (masked-out) voxels are not
embedded and stay zero; tiles without any such valid output are skipped. Intensity
statistics for normalization come from nonzero voxels of a fixed sample of the whole
volume, so every tile and region is normalized identically.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numcodecs import Zstd
from tqdm import tqdm
from vc_reader import transforms as A

from dinovol_2.dataset.cuda_pipeline import NORMALIZATION_SCHEMES, normalization_transforms
from dinovol_2.dataset.volume_reader import open_volume
from dinovol_2.eval.napari_visualizer import (
    LoadedBackbone, _configure_backbone_for_spatial_shape, load_backbone_from_checkpoint,
)
from dinovol_2.ops.point_embeddings import sample_grid_points

OUTPUT_NAMES = ("tokens", "tokens_unit", "dense", "ink")
OUTPUT_VALUES = {
    "tokens": "blended x_norm_patchtokens",
    "tokens_unit": "L2-normalized blended x_norm_patchtokens (dot product = cosine)",
    "dense": "unit-norm blended embedding",
    "ink": "unit-norm blended embedding",
}
DTYPES = ("int8", "float16")
# Per-output storage: int8 with a per-vector scale array, int8 at a fixed 1/127 scale (unit vectors only;
# keeps trilinear interpolation of the stored values linear), or plain floats.
STORAGE = {"int8": (np.int8, True, None), "unit8": (np.int8, False, 1 / 127), "float16": (np.float16, False, None),
           "float32": (np.float32, False, None)}
FORMAT = "dinovol-embeddings/1"
DENSE_POINT_CHUNK_SIZE = 262_144
CHUNK_BYTES = 16 << 20
NEED_MARGIN = 16  # voxels of need() context around a column: covers the token dilation


# ---------------------------------------------------------------------------
# Lattice


@dataclass(frozen=True)
class AxisLattice:
    tile: int
    margin: int
    ramp: int

    @property
    def stride(self) -> int:
        return self.tile - 2 * self.margin - self.ramp

    @property
    def support(self) -> int:
        return self.stride + self.ramp

    def validate(self, *, step: int, name: str) -> None:
        if self.margin < 0 or self.ramp <= 0 or self.stride < self.ramp:
            raise ValueError(f"{name}: need margin >= 0, ramp > 0 and tile - 2*margin >= 2*ramp, got {self}")
        for label, value in (("tile", self.tile), ("margin", self.margin), ("ramp", self.ramp)):
            if value % step:
                raise ValueError(f"{name}: {label}={value} must be a multiple of {step}")

    def origin(self, k: int) -> int:
        return k * self.stride - self.margin

    def tiles(self, lo: int, hi: int) -> range:
        """Tiles whose support intersects [lo, hi)."""
        return range((lo - self.support) // self.stride + 1, (hi - 1) // self.stride + 1)

    def weights(self, u: torch.Tensor) -> torch.Tensor:
        """Blend weight at continuous positions u relative to the tile origin."""
        rise = ((u - self.margin) / self.ramp).clamp(0, 1)
        fall = ((self.tile - self.margin - u) / self.ramp).clamp(0, 1)
        return rise * fall


# ---------------------------------------------------------------------------
# Normalization with volume-wide statistics


def _sorted_percentile(get, n: int, percent: float) -> float:
    # vc_reader's (NumPy "linear") percentile, transcribed for bitwise parity.
    q = percent / 100
    v = n * q + (1 + q * -1.0) - 1
    if v >= n - 1:
        return get(n - 1)
    if v < 0:
        return get(0)
    j = math.floor(v)
    a, b, t = get(j), get(j + 1), v - j
    d = b - a
    return b - d * (1 - t) if t >= 0.5 else a + d * t


def _histogram_percentile(values: np.ndarray, counts: np.ndarray, percent: float) -> float:
    """Percentile of a multiset given as sorted distinct values and their counts."""
    ends = np.cumsum(counts)
    return _sorted_percentile(lambda k: float(values[np.searchsorted(ends, k, side="right")]),
                              int(ends[-1]), percent)


def statistics_from_histogram(histogram: np.ndarray) -> dict[str, float]:
    """Intensity statistics of the voxels counted by ``histogram`` (index = integer value).

    Robust statistics reproduce vc_reader's ``Normalize("robust")`` on those voxels,
    including its float32 clipping and scale fallbacks.
    """
    values = np.nonzero(histogram)[0].astype(np.float64)
    counts = histogram[values.astype(np.int64)].astype(np.int64)
    n = int(counts.sum())
    if n == 0:
        raise ValueError("no voxels to compute normalization statistics from")
    mean = float((values * counts).sum() / n)
    std = float(np.sqrt((((values - mean) ** 2) * counts).sum() / n))
    lower = _histogram_percentile(values, counts, 1)
    upper = _histogram_percentile(values, counts, 99)
    # Clipping is monotone, so the clipped multiset keeps the value order.
    clipped = np.clip(values.astype(np.float32), np.float32(lower), np.float32(upper)).astype(np.float64)
    median = _histogram_percentile(clipped, counts, 50)
    deviations = np.abs(clipped - median)
    order = np.argsort(deviations, kind="stable")
    mad = _histogram_percentile(deviations[order], counts[order], 50)
    scale = 1.4826 * mad
    if not (np.isfinite(scale) and scale >= 1e-6):
        span = float(np.float32(upper)) - float(np.float32(lower))
        scale = abs(std) if abs(std) >= 1e-6 else abs(span) / 2 if abs(span) / 2 >= 1e-6 else 1.0
    return {"count": n, "min": float(values[0]), "max": float(values[-1]), "mean": mean, "std": std,
            "percentile_01": lower, "percentile_99": upper, "median_clipped": median,
            "mad_clipped": mad, "robust_scale": scale}


def sample_volume_histogram(reader, *, chunk_stride: int = 8, batch: int = 32,
                            skip_missing: bool = False) -> tuple[np.ndarray, int]:
    """Histogram of nonzero values over every ``chunk_stride``-th chunk along each axis.

    skip_missing: leave out chunks an offline reader has not cached (CacheMissError); the count
    returned is then the number of chunks actually read.
    """
    from vc_reader.reader import CacheMissError
    dtype = np.dtype(reader.dtype)
    if dtype.kind not in "ui" or dtype.itemsize > 2:
        raise ValueError(f"volume-wide statistics need 8/16-bit integer data, got {dtype}; "
                         "pass --normalization-stats instead")
    chunks = tuple(int(c) for c in reader.chunks)
    grid = [range(0, int(size), chunk * chunk_stride) for size, chunk in zip(reader.shape, chunks)]
    starts = [(z, y, x) for z in grid[0] for y in grid[1] for x in grid[2]]
    histogram = np.zeros(1 << (8 * dtype.itemsize), dtype=np.int64)
    read = 0
    for i in tqdm(range(0, len(starts), batch), desc="normalization sample", unit="batch"):
        regions = [(start, chunks) for start in starts[i:i + batch]]
        try:
            blocks = reader.read_regions(regions)
        except CacheMissError:
            if not skip_missing:
                raise
            blocks = []
            for region in regions:
                try:
                    blocks.append(reader.read_region(*region))
                except CacheMissError:
                    pass
        for block in blocks:
            read += 1
            values = block[block != 0]
            if values.size:
                histogram += np.bincount(values.ravel().astype(np.int64), minlength=histogram.size)
    return histogram, read


def global_normalization(scheme: str, statistics: dict[str, float] | None,
                         intensity_properties: dict[str, Any] | None = None, *, clip_lower: bool = False) -> list:
    """vc_reader transforms applying ``scheme`` with fixed, volume-wide statistics.

    Robust normalization clips only the upper percentile by default, so masked zeros keep
    their own value instead of being raised to the lower percentile like scroll air.
    """
    if scheme in ("none", "ct"):
        return normalization_transforms(scheme, intensity_properties)
    if statistics is None:
        raise ValueError(f"{scheme!r} normalization needs volume statistics")
    if scheme == "robust":
        return [A.Normalize("fixed", center=statistics["median_clipped"], scale=statistics["robust_scale"],
                            bounds=(statistics["percentile_01"] if clip_lower else -math.inf,
                                    statistics["percentile_99"]))]
    if scheme == "zscore":
        return [A.Normalize("fixed", center=statistics["mean"], scale=max(statistics["std"], 1e-8))]
    if scheme in ("minmax", "rescale_to_01"):
        span = statistics["max"] - statistics["min"]
        return [A.Normalize("fixed", center=statistics["min"], scale=span if span >= 1e-6 else 1.0)]
    raise ValueError(f"Unknown normalization scheme: {scheme}")


# ---------------------------------------------------------------------------
# Output storage


def _chunk_shape(shape: Sequence[int], cell: int, itemsize: int, *, z_cell: bool = False) -> tuple[int, ...]:
    """Chunks tile every lattice cell exactly, so each cell write touches whole chunks.

    z_cell: also align z chunks to cells (arrays spanning the volume, written slab by slab).
    """
    z = cell if z_cell else min(int(shape[0]), 64)
    yx = cell
    channels = int(shape[3]) if len(shape) == 4 else 1
    while z * yx * yx * channels * itemsize > CHUNK_BYTES and yx % 2 == 0:
        yx //= 2
        if z_cell and z % 2 == 0:
            z //= 2
    return (z, yx, yx, *shape[3:])


class ZarrGroupWriter:
    """Minimal zarr v2 writer (Zstd, '/' separator) that never writes all-zero chunks."""

    def __init__(self, path: str | Path, *, attrs: dict[str, Any] | None, overwrite: bool = False,
                 workers: int = 8):
        """attrs=None opens an existing group to add arrays; otherwise a new group is started."""
        self.path = Path(path)
        self.arrays: dict[str, tuple[tuple[int, ...], tuple[int, ...], np.dtype]] = {}
        if attrs is None:
            for meta_path in self.path.glob("*/.zarray"):
                meta = json.loads(meta_path.read_text())
                self.arrays[meta_path.parent.name] = (tuple(meta["shape"]), tuple(meta["chunks"]),
                                                      np.dtype(meta["dtype"]))
        else:
            if self.path.exists() and any(self.path.iterdir()) and not overwrite:
                raise FileExistsError(f"{self.path} exists; pass --overwrite to replace chunks")
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
            (self.path / ".zattrs").write_text(json.dumps(attrs, indent=2))
        self.codec = Zstd(level=3)
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)
        self.pending: list[concurrent.futures.Future] = []

    def create(self, name: str, shape: Sequence[int], chunks: Sequence[int], dtype,
               attrs: dict[str, Any] | None = None) -> None:
        dtype = np.dtype(dtype)
        root = self.path / name
        root.mkdir(exist_ok=True)
        if attrs:
            (root / ".zattrs").write_text(json.dumps(attrs, indent=2))
        meta = {"zarr_format": 2, "shape": [int(v) for v in shape], "chunks": [int(v) for v in chunks],
                "dtype": dtype.str, "fill_value": 0, "compressor": self.codec.get_config(),
                "filters": None, "order": "C", "dimension_separator": "/"}
        (root / ".zarray").write_text(json.dumps(meta, indent=2))
        self.arrays[name] = (tuple(meta["shape"]), tuple(meta["chunks"]), dtype)

    def write(self, name: str, start: Sequence[int], block: np.ndarray) -> None:
        """Write a block whose start is chunk-aligned and which ends on chunks or the array edge."""
        shape, chunks, dtype = self.arrays[name]
        start = tuple(int(v) for v in start) + (0,) * (len(shape) - len(start))
        for axis, (s, c, n, size) in enumerate(zip(start, chunks, block.shape, shape)):
            if s % c or ((s + n) % c and s + n != size):
                raise ValueError(f"{name}: block {start}+{block.shape} is not chunk-aligned on axis {axis}")
        counts = [-(-n // c) for n, c in zip(block.shape, chunks)]
        for index in np.ndindex(*counts):
            lo = [i * c for i, c in zip(index, chunks)]
            part = block[tuple(slice(l, l + c) for l, c in zip(lo, chunks))]
            if not part.any():
                continue
            chunk_index = [(s + l) // c for s, l, c in zip(start, lo, chunks)]
            if part.shape != chunks:
                padded = np.zeros(chunks, dtype=dtype)
                padded[tuple(slice(0, n) for n in part.shape)] = part
                part = padded
            self._submit(self.path / name / "/".join(map(str, chunk_index)), np.ascontiguousarray(part, dtype))

    def _submit(self, path: Path, data: np.ndarray) -> None:
        for future in [f for f in self.pending if f.done()]:
            future.result()
        self.pending = [f for f in self.pending if not f.done()]
        while len(self.pending) > 64:
            self.pending.pop(0).result()
        self.pending.append(self.pool.submit(self._store, path, data))

    def _store(self, path: Path, data: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(self.codec.encode(data))
        os.replace(tmp, path)

    def close(self) -> None:
        for future in self.pending:
            future.result()
        self.pool.shutdown()


def read_embeddings(path: str | Path, name: str, selection: Any = ...) -> np.ndarray:
    """Read (and dequantize) one output of an embedding group as float32 (z, y, x, C).

    ``selection`` is a NumPy index over the stored (z, y, x) grid. Masked outputs are zero.
    """
    from vc_reader.reader import open_zarr

    values = open_zarr(str(Path(path) / name)).read(selection).astype(np.float32)
    attrs_path = Path(path) / name / ".zattrs"
    value_scale = json.loads(attrs_path.read_text()).get("value_scale") if attrs_path.exists() else None
    if value_scale is not None:
        values *= np.float32(value_scale)
    if not (Path(path) / f"{name}_scale").exists():
        return values
    return values * open_zarr(str(Path(path) / f"{name}_scale")).read(selection).astype(np.float32)[..., None]


# ---------------------------------------------------------------------------
# Tiled inference


@dataclass(frozen=True)
class OutputSpec:
    name: str
    step: tuple[int, int, int]
    channels: int


def quantize(values: torch.Tensor, valid: torch.Tensor, dtype: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Zero invalid vectors; int8 uses a symmetric per-vector absmax scale stored as float16.

    unit8 stores unit vectors as round(127 * value) with no scale array.
    """
    values = torch.where(valid[..., None], values, 0)
    if dtype == "float16":
        return values.half(), None
    if dtype == "float32":
        return values.float(), None
    if dtype == "unit8":
        return (values * 127).round().clamp(-127, 127).to(torch.int8), None
    scale = (values.abs().amax(-1) / 127).half()
    divisor = torch.where(scale > 0, scale.float(), 1)[..., None]
    return (values / divisor).round().clamp(-127, 127).to(torch.int8), scale


class RegionEmbedder:
    def __init__(self, reader, loaded: LoadedBackbone, *, transforms: list, outputs: Sequence[str],
                 dense_stride: int, lattice: Sequence[AxisLattice], dtype: str = "int8",
                 dtypes: dict[str, str] | None = None,
                 amp_dtype: torch.dtype = torch.bfloat16, point_chunk_size: int = DENSE_POINT_CHUNK_SIZE):
        """dtypes overrides the storage (a STORAGE key) of individual outputs."""
        self.reader, self.loaded, self.device = reader, loaded, loaded.device
        self.backbone, self.dense_head = loaded.backbone, loaded.dense_head
        self.lattice, self.dtype, self.amp_dtype = tuple(lattice), dtype, amp_dtype
        self.point_chunk_size = int(point_chunk_size)
        patch = tuple(int(p) for p in loaded.patch_size)
        self.patch = patch
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        unknown = set(outputs) - set(OUTPUT_NAMES)
        if unknown or not outputs:
            raise ValueError(f"outputs must be a non-empty subset of {OUTPUT_NAMES}, got {outputs}")
        if loaded.embedding_type != "deeper" and set(outputs) & {"dense", "ink"}:
            raise ValueError("dense/ink outputs need the deeper CNN embedding")
        if set(outputs) & {"dense", "ink"} and self.dense_head is None:
            raise ValueError("checkpoint has no dense head; request only 'tokens'")
        if any(p % dense_stride for p in patch):
            raise ValueError(f"dense_stride {dense_stride} must divide patch_size {patch}")
        for axis, (lat, p) in enumerate(zip(self.lattice, patch)):
            lat.validate(step=p, name=f"axis {axis}")
        if tuple(lat.tile for lat in self.lattice) != tuple(loaded.global_input_size):
            print(f"warning: tile {tuple(lat.tile for lat in self.lattice)} differs from the training "
                  f"global crop {tuple(loaded.global_input_size)}; embeddings depend on the tile size, so "
                  "only compare outputs made with the same tile")
        channels = {"tokens": int(self.backbone.embed_dim)}
        if self.dense_head is not None:
            channels.update(dense=int(self.dense_head.out_dim), ink=int(self.dense_head.ink_dim))
        self.outputs = tuple(name for name in OUTPUT_NAMES if name in outputs)
        self.dtypes = {name: (dtypes or {}).get(name, dtype) for name in self.outputs}
        for name, storage in self.dtypes.items():
            if storage not in STORAGE or (storage == "unit8" and name == "tokens"):
                raise ValueError(f"{name}: unsupported storage {storage!r}")
        # Accumulated grids; tokens_unit is written from the tokens accumulator.
        accumulated = {"tokens" if name == "tokens_unit" else name for name in self.outputs}
        self.specs = [OutputSpec(name, patch if name == "tokens" else (dense_stride,) * 3, channels[name])
                      for name in OUTPUT_NAMES if name in accumulated]
        self.pipeline = A.Pipeline(transforms, targets={"image": A.Image(valid=None)}, backend="cuda",
                                   compact=False)
        _configure_backbone_for_spatial_shape(self.backbone, tuple(lat.tile for lat in self.lattice))

    # -- geometry ---------------------------------------------------------

    def output_shape(self, spec: OutputSpec, z_range: tuple[int, int]) -> tuple[int, ...]:
        shape = self.reader.shape
        return (-(-(z_range[1] - z_range[0]) // spec.step[0]), -(-int(shape[1]) // spec.step[1]),
                -(-int(shape[2]) // spec.step[2]), spec.channels)

    def chunk_shape(self, spec: OutputSpec, z_range: tuple[int, int], itemsize: int,
                    z_cell: bool = False) -> tuple[int, ...]:
        cell = self.lattice[1].stride // spec.step[1]
        return _chunk_shape(self.output_shape(spec, z_range), cell, itemsize, z_cell=z_cell)

    def arrays(self, z_range: tuple[int, int], *, z_cell: bool = False):
        """(name, spec, shape, chunks) of every written output, excluding int8 scales."""
        specs = {spec.name: spec for spec in self.specs}
        result = []
        for name in self.outputs:
            spec = specs["tokens" if name == "tokens_unit" else name]
            itemsize = np.dtype(STORAGE[self.dtypes[name]][0]).itemsize
            result.append((name, spec, self.output_shape(spec, z_range),
                           self.chunk_shape(spec, z_range, itemsize, z_cell)))
        return result

    def create_arrays(self, sink, z_range: tuple[int, int], *, z_cell: bool = False) -> None:
        """Create every output (and int8 scale) array in sink for outputs covering z_range."""
        for name, spec, shape, chunks in self.arrays(z_range, z_cell=z_cell):
            dtype, scaled, value_scale = STORAGE[self.dtypes[name]]
            sink.create(name, shape, chunks, dtype, attrs={"value_scale": value_scale} if value_scale else None)
            if scaled:
                sink.create(f"{name}_scale", shape[:3], chunks[:3], np.float16)

    def _axis_weights(self, axis: int, lo: int, n: int, step: int, origin: int) -> torch.Tensor:
        # Weights at output block centers [lo + i*step + step/2] relative to the tile origin.
        centers = lo + step * torch.arange(n, device=self.device, dtype=torch.float32) + step / 2 - origin
        return self.lattice[axis].weights(centers)

    # -- main loop ------------------------------------------------------------

    @torch.inference_mode()
    def run(self, sink, *, z_range: tuple[int, int], y_range: tuple[int, int] | None = None,
            x_range: tuple[int, int] | None = None, progress: bool = True, need=None,
            array_z_range: tuple[int, int] | None = None, read: str = "band") -> dict[str, int]:
        """Embed z_range (cells overlapping y/x_range) into sink.

        need(z0, z1, y0, x0, size, margin) -> bool tensor or None: the voxels to embed over
        [z0-margin, z1+margin) x [y0-margin, y0+size+margin) x (same for x), or None when there are
        none. It must be a fixed function of global position so overlapping columns agree.
        array_z_range: global z extent of the output arrays (default z_range); writes past its end
        are clipped, so slabs may overhang the volume.
        read: "band" reads a full-width row of columns at once, "column" one column at a time.
        """
        z0, z1 = (int(v) for v in z_range)
        array_z_range = tuple(int(v) for v in (array_z_range or z_range))
        steps = [spec.step for spec in self.specs]
        if z1 <= z0 or any(z0 % s[0] or z1 % s[0] for s in steps):
            raise ValueError(f"z range {z_range} must be non-empty and aligned to {[s[0] for s in steps]}")
        shape = tuple(int(v) for v in self.reader.shape)
        lz, ly, lx = self.lattice
        y_range = y_range or (0, shape[1])
        x_range = x_range or (0, shape[2])
        cells_y = range(y_range[0] // ly.stride, (min(y_range[1], shape[1]) - 1) // ly.stride + 1)
        cells_x = range(x_range[0] // lx.stride, (min(x_range[1], shape[2]) - 1) // lx.stride + 1)
        tiles_z = lz.tiles(z0, z1)
        rows = range(cells_y.start - 1, cells_y.stop)
        cols = range(cells_x.start - 1, cells_x.stop)
        band_z0 = lz.origin(tiles_z.start)
        band_shape = (lz.origin(tiles_z.stop - 1) + lz.tile - band_z0, ly.tile,
                      lx.origin(cols.stop - 1) + lx.tile - lx.origin(cols.start))

        def read_band(ky):
            if read == "column":
                return None
            return self.reader.read_region((band_z0, ly.origin(ky), lx.origin(cols.start)), band_shape)

        stats = {"columns": 0, "tiles": 0, "skipped_tiles": 0, "skipped_columns": 0, "cells_written": 0}
        top: dict[int, dict[str, torch.Tensor]] = {}
        bar = tqdm(total=len(rows) * len(cols), desc="columns", unit="col", disable=not progress)
        with concurrent.futures.ThreadPoolExecutor(1) as prefetch:
            future = prefetch.submit(read_band, rows.start)
            for ky in rows:
                band = future.result()
                if ky + 1 < rows.stop:
                    future = prefetch.submit(read_band, ky + 1)
                next_top: dict[int, dict[str, torch.Tensor]] = {}
                left: dict[str, torch.Tensor] | None = None
                for kx in cols:
                    need_mask = None
                    if need is not None:
                        need_mask = need(z0, z1, ky * ly.stride, kx * lx.stride, ly.support, NEED_MARGIN)
                        if need_mask is None:
                            stats["skipped_columns"] += 1
                            left = None
                            bar.update(1)
                            continue
                    if read == "column":
                        column = self.reader.read_region((band_z0, ly.origin(ky), lx.origin(kx)),
                                                         (band_shape[0], ly.tile, lx.tile))
                    else:
                        offset = (kx - cols.start) * lx.stride
                        column = band[:, :, offset:offset + lx.tile]
                    box = (ly.stride if ky < cells_y.start else 0, ly.stride if ky == cells_y.stop - 1 else ly.support,
                           lx.stride if kx < cells_x.start else 0, lx.stride if kx == cells_x.stop - 1 else lx.support)
                    acc, valid = self._embed_column(column, band_z0, (z0, z1), ky, kx, box, tiles_z, stats,
                                                    need_mask)
                    if acc is not None:
                        self._finish_column(sink, acc, valid, (z0, z1), ky, kx, cells_y, cells_x,
                                            top.get(kx), left, next_top, stats, array_z_range)
                        left = {spec.name: acc[spec.name][:, :ly.stride // spec.step[1], lx.stride // spec.step[2]:]
                                for spec in self.specs}
                    else:
                        # No valid output here: neighbors' strips into this column are dropped too.
                        stats["skipped_columns"] += 1
                        left = None
                    bar.update(1)
                top = next_top
        bar.close()
        return stats

    def _embed_column(self, column: np.ndarray, band_z0: int, z_range, ky: int, kx: int, box, tiles_z, stats,
                      need_mask=None):
        """Blend all z tiles of one (ky, kx) column; accumulators cover ROI z x the yx support.

        need_mask covers that region plus NEED_MARGIN voxels on every side: dense outputs are embedded
        where it is set, tokens within one token of it (their trilinear support), always only where
        the volume is nonzero.
        """
        z0, z1 = z_range
        lz, ly, lx = self.lattice
        y_lo, y_hi, x_lo, x_hi = box
        roi = column[z0 - band_z0:z1 - band_z0, ly.margin + y_lo:ly.margin + y_hi,
                     lx.margin + x_lo:lx.margin + x_hi]
        if not roi.any():
            return None, None
        raw = torch.from_numpy(np.ascontiguousarray(column)).to(self.device, non_blocking=True)
        nonzero = raw[z0 - band_z0:z1 - band_z0, ly.margin:ly.margin + ly.support,
                      lx.margin:lx.margin + lx.support] != 0
        needed = torch.zeros_like(nonzero)
        needed[:, y_lo:y_hi, x_lo:x_hi] = True
        valid = {}
        m = NEED_MARGIN
        for spec in self.specs:
            base = nonzero & needed
            if need_mask is None:
                valid[spec.name] = F.max_pool3d(base[None].to(torch.float16), spec.step)[0] > 0
                continue
            inner = need_mask[m:m + nonzero.shape[0], m:m + nonzero.shape[1], m:m + nonzero.shape[2]]
            if spec.name == "tokens":
                # Tokens whose patch, or a neighboring token's, holds needed voxels: the margin makes
                # this dilation agree between columns that share an overlap.
                pooled = F.max_pool3d(need_mask[None].to(torch.float16), spec.step)
                near = F.max_pool3d(pooled, 3, stride=1, padding=1)[0] > 0
                t = [m // s for s in spec.step]
                near = near[t[0]:t[0] + nonzero.shape[0] // spec.step[0], t[1]:t[1] + nonzero.shape[1] // spec.step[1],
                            t[2]:t[2] + nonzero.shape[2] // spec.step[2]]
                valid[spec.name] = (F.max_pool3d(base[None].to(torch.float16), spec.step)[0] > 0) & near
            else:
                valid[spec.name] = F.max_pool3d((base & inner)[None].to(torch.float16), spec.step)[0] > 0
        if not any(v.any() for v in valid.values()):
            return None, None
        stats["columns"] += 1
        image = self.pipeline.batch({"image": raw[None, None]}, seeds=[0], device=self.device)["image"][0, 0]
        acc = {}
        for spec in self.specs:
            grid = tuple(n // s for n, s in zip(nonzero.shape, spec.step))
            acc[spec.name] = torch.zeros((*grid, spec.channels + (spec.name == "tokens")), device=self.device)
        coarse = max(self.specs, key=lambda s: s.step[0])
        point_specs = [spec for spec in self.specs if spec.name != "tokens"]
        for kz in tiles_z:
            a_z = lz.origin(kz)
            lo, hi = max(kz * lz.stride, z0), min(kz * lz.stride + lz.support, z1)
            if not valid[coarse.name][(lo - z0) // coarse.step[0]:(hi - z0) // coarse.step[0]].any():
                stats["skipped_tiles"] += 1
                continue
            stats["tiles"] += 1
            tile = image[a_z - band_z0:a_z - band_z0 + lz.tile][None, None].contiguous()
            with torch.autocast(self.device.type, dtype=self.amp_dtype):
                self.backbone.return_embed_skips = bool(point_specs)
                out = self.backbone.forward_features(tile, masks=None, view_kind="global")
            tokens = out["x_norm_patchtokens"][0]
            token_grid = tokens.reshape(*(lat.tile // p for lat, p in zip(self.lattice, self.patch)), -1)
            for spec in self.specs:
                if spec.name == "tokens":
                    self._add_tokens(acc["tokens"], token_grid, (lo, hi), z0, a_z)
            if point_specs:
                self._add_points(acc, valid, point_specs, token_grid, out["embed_skips"], (lo, hi), z0, a_z)
        return acc, valid

    def _add_tokens(self, acc, token_grid, z_bounds, z0, a_z):
        lz, ly, lx = self.lattice
        pz, py, px = self.patch
        lo, hi = z_bounds
        nz, ny, nx = (hi - lo) // pz, ly.support // py, lx.support // px
        weight = (self._axis_weights(0, lo, nz, pz, a_z)[:, None, None]
                  * self._axis_weights(1, ly.margin, ny, py, 0)[None, :, None]
                  * self._axis_weights(2, lx.margin, nx, px, 0)[None, None, :])
        block = token_grid[(lo - a_z) // pz:(hi - a_z) // pz, ly.margin // py:ly.margin // py + ny,
                           lx.margin // px:lx.margin // px + nx].float()
        target = acc[(lo - z0) // pz:(hi - z0) // pz]
        target[..., :-1] += block * weight[..., None]
        target[..., -1] += weight

    def _add_points(self, acc, valid, specs, token_grid, skips, z_bounds, z0, a_z):
        lz, ly, lx = self.lattice
        step = specs[0].step
        lo, hi = z_bounds
        grid_shape = valid[specs[0].name].shape
        sub = valid[specs[0].name][(lo - z0) // step[0]:(hi - z0) // step[0]]
        index = sub.nonzero()
        if not index.numel():
            return
        index[:, 0] += (lo - z0) // step[0]
        flat = (index[:, 0] * grid_shape[1] + index[:, 1]) * grid_shape[2] + index[:, 2]
        # Voxel-index coordinates of block centers in the tile; their continuous position is +0.5.
        start = torch.tensor((z0 - a_z, ly.margin, lx.margin), device=self.device, dtype=torch.float32)
        st = torch.tensor(step, device=self.device, dtype=torch.float32)
        coords = start + index.float() * st + (st - 1) / 2
        weight = (lz.weights(coords[:, 0] + 0.5) * ly.weights(coords[:, 1] + 0.5) * lx.weights(coords[:, 2] + 0.5))
        input_shape = tuple(lat.tile for lat in self.lattice)
        head = self.dense_head
        with torch.autocast(self.device.type, dtype=self.amp_dtype):
            grid = token_grid[None]
            fpn_grid = head.fpn_grid(grid, skips)
            skip_grids = [level.permute(0, 2, 3, 4, 1) for level in skips]
            for begin in range(0, coords.shape[0], self.point_chunk_size):
                points = coords[None, begin:begin + self.point_chunk_size]
                fpn = head.fpn_norm(sample_grid_points(fpn_grid, points, input_shape).to(grid.dtype))
                readout = head.readout(sample_grid_points(grid, points, input_shape),
                                       [sample_grid_points(g, points, input_shape) for g in skip_grids],
                                       fpn, dtype=grid.dtype)
                w = weight[begin:begin + self.point_chunk_size, None]
                rows = flat[begin:begin + self.point_chunk_size]
                for spec in specs:
                    target = acc[spec.name].view(-1, spec.channels)
                    target.index_add_(0, rows, readout[spec.name][0].float() * w)

    def _finish_column(self, sink, acc, valid, z_range, ky, kx, cells_y, cells_x, top, left, next_top, stats,
                       array_z_range):
        """Finalize cell (ky, kx) and hand the overlap strips to the cells below and right."""
        ly, lx = self.lattice[1:]
        for spec in self.specs:
            a = acc[spec.name]
            sy, sx = ly.stride // spec.step[1], lx.stride // spec.step[2]
            ry, rx = ly.ramp // spec.step[1], lx.ramp // spec.step[2]
            # Strip y in [stride, support) belongs to the next row: x < stride to cell kx, the rest
            # (the corner) to cell kx + 1, which this column reaches first.
            below = next_top.setdefault(kx, {})
            if spec.name in below:
                below[spec.name] += a[:, sy:, :sx].cpu()
            else:
                below[spec.name] = a[:, sy:, :sx].cpu()
            corner = next_top.setdefault(kx + 1, {})
            corner[spec.name] = torch.zeros((a.shape[0], ry, sx, a.shape[3]))
            corner[spec.name][:, :, :rx] = a[:, sy:, sx:].cpu()
            if ky not in cells_y or kx not in cells_x:
                continue
            cell = a[:, :sy, :sx].clone()
            if top is not None and spec.name in top:
                cell[:, :ry] += top[spec.name].to(self.device)
            if left is not None and spec.name in left:
                cell[:, :, :rx] += left[spec.name]
            ok = valid[spec.name][:, :sy, :sx]
            if spec.name == "tokens":
                values = cell[..., :-1] / cell[..., -1:].clamp_min(torch.finfo(torch.float32).tiny)
                written = {"tokens": values, "tokens_unit": F.normalize(values, dim=-1)}
            else:
                written = {spec.name: F.normalize(cell, dim=-1)}
            shape = self.output_shape(spec, z_range)
            y0, x0 = ky * sy, kx * sx
            ny, nx = min(sy, shape[1] - y0), min(sx, shape[2] - x0)
            zi = (z_range[0] - array_z_range[0]) // spec.step[0]
            nz = min((z_range[1] - z_range[0]) // spec.step[0],
                     -(-(array_z_range[1] - array_z_range[0]) // spec.step[0]) - zi)
            for name, values in written.items():
                if name not in self.outputs:
                    continue
                data, scale = quantize(values, ok, self.dtypes[name])
                sink.write(name, (zi, y0, x0), data[:nz, :ny, :nx].cpu().numpy())
                if scale is not None:
                    sink.write(f"{name}_scale", (zi, y0, x0), scale[:nz, :ny, :nx].cpu().numpy())
        if ky in cells_y and kx in cells_x:
            stats["cells_written"] += 1


# ---------------------------------------------------------------------------
# CLI


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--add-unit-tokens", metavar="GROUP", default=None,
                        help="only add tokens_unit to an existing output group from its stored tokens")
    parser.add_argument("--checkpoint")
    parser.add_argument("--volume", help="ome-zarr URL or path (s3:// uses --cache-dir)")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--connections", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16, help="vc_reader read threads")
    parser.add_argument("--z-start", type=int)
    parser.add_argument("--depth", type=int, default=192)
    parser.add_argument("--y-range", type=int, nargs=2, default=None, help="restrict to lattice cells overlapping [lo, hi)")
    parser.add_argument("--x-range", type=int, nargs=2, default=None, help="restrict to lattice cells overlapping [lo, hi)")
    parser.add_argument("--outputs", default=",".join(OUTPUT_NAMES))
    parser.add_argument("--dense-stride", type=int, default=1)
    parser.add_argument("--dtype", choices=DTYPES, default="int8")
    parser.add_argument("--tile", type=int, default=None, help="tile edge in voxels (default: training global crop)")
    parser.add_argument("--margin", type=int, default=None,
                        help="context-only voxels per tile face (default: tile/6, patch-aligned)")
    parser.add_argument("--ramp", type=int, default=None, help="blend ramp in voxels (default: tile/6, patch-aligned)")
    parser.add_argument("--normalization-stats", default=None,
                        help="statistics JSON or an earlier output's .zattrs to reuse (skips sampling)")
    parser.add_argument("--stats-chunk-stride", type=int, default=8)
    parser.add_argument("--branch", default="teacher")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    missing = [f"--{name.replace('_', '-')}" for name in ("checkpoint", "volume", "z_start", "output")
               if getattr(args, name) is None]
    if args.add_unit_tokens is None and missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def add_unit_tokens(path: str | Path, *, name: str = "tokens_unit", overwrite: bool = False) -> int:
    """Add L2-normalized tokens to an existing group from its stored tokens, chunk by chunk.

    A positive per-vector scale does not change direction, so this matches computing them
    during the run up to the int8 rounding already in ``tokens``. Returns chunks written.
    """
    from vc_reader.reader import open_zarr

    path = Path(path)
    attrs = json.loads((path / ".zattrs").read_text())
    if (path / name).exists() and not overwrite:
        raise FileExistsError(f"{path / name} exists; pass overwrite=True to replace it")
    sink = ZarrGroupWriter(path, attrs=None)
    shape, chunks, dtype = sink.arrays["tokens"]
    quantized = (path / "tokens_scale").exists()
    sink.create(name, shape, chunks, dtype)
    if quantized:
        sink.create(f"{name}_scale", shape[:3], chunks[:3], np.float16)
    tokens = open_zarr(str(path / "tokens"))
    scales = open_zarr(str(path / "tokens_scale")) if quantized else None
    stored = sorted(tuple(int(v) for v in p.relative_to(path / "tokens").parts[:3])
                    for p in (path / "tokens").glob("*/*/*/0"))
    for index in tqdm(stored, desc=name, unit="chunk"):
        start = [i * c for i, c in zip(index, chunks)]
        selection = tuple(slice(s, min(s + c, n)) for s, c, n in zip(start, chunks, shape))
        values = torch.from_numpy(tokens.read(selection).astype(np.float32))
        if scales is not None:
            values *= torch.from_numpy(scales.read(selection).astype(np.float32))[..., None]
        data, scale = quantize(F.normalize(values, dim=-1), values.abs().amax(-1) > 0,
                               "int8" if quantized else "float16")
        sink.write(name, start, data.numpy())
        if scale is not None:
            sink.write(f"{name}_scale", start, scale.numpy())
    sink.close()
    attrs["outputs"][name] = {**attrs["outputs"]["tokens"], "values": OUTPUT_VALUES["tokens_unit"],
                              "derived_from": "tokens"}
    (path / ".zattrs").write_text(json.dumps(attrs, indent=2))
    return len(stored)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.add_unit_tokens is not None:
        print(f"wrote {add_unit_tokens(args.add_unit_tokens, overwrite=args.overwrite)} tokens_unit chunks")
        return
    torch.backends.cudnn.benchmark = False
    s3 = {"anon": True} if str(args.volume).startswith("s3://") else None
    reader = open_volume(args.volume, level=args.level, s3_storage_options=s3, cache_dir=args.cache_dir,
                         connections=args.connections, threads=args.threads)
    loaded = load_backbone_from_checkpoint(args.checkpoint, preferred_branch=args.branch)
    checkpoint_step = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True).get("step")

    scheme = loaded.normalization_scheme
    statistics = None
    if scheme not in ("none", "ct"):
        if args.normalization_stats:
            # A statistics dict, or the .zattrs of an earlier output (reuse keeps slabs consistent).
            saved = json.loads(Path(args.normalization_stats).read_text())
            statistics = saved.get("normalization", saved).get("statistics", saved)
        else:
            histogram, sampled = sample_volume_histogram(reader, chunk_stride=args.stats_chunk_stride)
            statistics = {**statistics_from_histogram(histogram), "sampled_chunks": sampled,
                          "chunk_stride": args.stats_chunk_stride, "selection": "nonzero voxels"}
        print("normalization", scheme, json.dumps(statistics))
    transforms = global_normalization(scheme, statistics, loaded.intensity_properties)

    tile = args.tile or int(loaded.global_input_size[0])
    # Token agreement with a centered view rises from 0.88 at a tile face to 0.98 six tokens in
    # (640k checkpoint, 192^3 tiles); tile/6 keeps blend midpoints that far from every face.
    patch = int(loaded.patch_size[0])
    overlap = max(patch, tile // 6 // patch * patch)
    margin = overlap if args.margin is None else args.margin
    ramp = overlap if args.ramp is None else args.ramp
    lattice = [AxisLattice(tile, margin, ramp)] * 3
    outputs = [name.strip() for name in args.outputs.split(",") if name.strip()]
    embedder = RegionEmbedder(reader, loaded, transforms=transforms, outputs=outputs,
                              dense_stride=args.dense_stride, lattice=lattice, dtype=args.dtype)
    z_range = (args.z_start, args.z_start + args.depth)
    attrs = {
        "format": FORMAT,
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_step": checkpoint_step,
        "branch": loaded.source_branch,
        "volume": str(args.volume), "level": args.level, "volume_shape": list(reader.shape),
        "z_range": list(z_range), "y_range": args.y_range, "x_range": args.x_range,
        "lattice": {"tile": tile, "margin": margin, "ramp": ramp, "stride": lattice[0].stride,
                    "tile_origin": "k*stride - margin", "support": "[k*stride, k*stride + stride + ramp)"},
        "normalization": {"scheme": scheme, "statistics": statistics,
                          "applied": "fixed volume-wide statistics" + (
                              "; robust clips only the upper percentile" if scheme == "robust" else "")},
        "amp_dtype": "bfloat16", "masked": "outputs whose block has no nonzero voxel are zero vectors",
        "outputs": {},
    }
    sink = ZarrGroupWriter(args.output, attrs=attrs, overwrite=args.overwrite)
    embedder.create_arrays(sink, z_range)
    for name, spec, shape, chunks in embedder.arrays(z_range):
        attrs["outputs"][name] = {
            "shape_zyxc": list(shape), "step_zyx": list(spec.step),
            "origin_zyx": [z_range[0], 0, 0],
            "index_to_voxel": "voxel block [origin + i*step, origin + (i+1)*step)",
            "dtype": args.dtype,
            "quantization": ("symmetric int8 per vector: value = q * scale; scale in <name>_scale"
                             if args.dtype == "int8" else None),
            "values": OUTPUT_VALUES[name],
        }
    (Path(args.output) / ".zattrs").write_text(json.dumps(attrs, indent=2))
    try:
        stats = embedder.run(sink, z_range=z_range, y_range=args.y_range, x_range=args.x_range)
    finally:
        sink.close()
    attrs["run"] = stats
    (Path(args.output) / ".zattrs").write_text(json.dumps(attrs, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
