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

Without --z-start, the CLI streams the entire volume into one group. --slab-depth
sets the bounded processing depth (default: two lattice strides). Re-run the same
command with --resume to skip committed slabs and retry an interrupted slab.
Local vc_reader cache paths are resolved to their remote source so uncached chunks
can be fetched. --codec vc-delta3d compresses every int8 embedding array losslessly;
the float16 per-vector scale arrays use Zstd.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import threading
from contextlib import contextmanager
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
from dinovol_2.ops.point_embeddings import grid_sample_geometry, sample_grid_points

OUTPUT_NAMES = ("tokens", "tokens_unit", "dense", "ink")
OUTPUT_VALUES = {
    "tokens": "blended x_norm_patchtokens",
    "tokens_unit": "L2-normalized blended x_norm_patchtokens (dot product = cosine)",
    "dense": "unit-norm blended embedding",
    "ink": "unit-norm blended embedding",
}
DTYPES = ("int8", "float16")
# Per-output storage: (dtype, per-vector scale array, fixed value scale, codec). int8 with a per-vector
# scale array, int8 at a fixed 1/127 scale (unit vectors only; keeps trilinear interpolation of the stored
# values linear), or plain floats. Fixed-scale fields vary smoothly in space, so vc-delta3d (lossless,
# channels last) stores them ~5x smaller than Zstd; per-vector scales break that smoothness.
STORAGE = {"int8": (np.int8, True, None, "zstd"), "unit8": (np.int8, False, 1 / 127, "vc-delta3d"),
           "float16": (np.float16, False, None, "zstd"), "float32": (np.float32, False, None, "zstd")}
# --dtype int8 stores the unit-norm head outputs at the fixed scale; tokens keep per-vector scales.
CLI_DTYPES = {"int8": {"dense": "unit8", "ink": "unit8"}, "float16": {}}
CODECS = ("zstd", "vc-delta3d")
QUANTIZATION = {"int8": "symmetric int8 per vector: value = q * scale; scale in <name>_scale",
                "unit8": "fixed-scale int8: value = q * value_scale (1/127, in the array's .zattrs)",
                "float16": None, "float32": None}
FORMAT = "dinovol-embeddings/1"
DENSE_POINT_CHUNK_SIZE = 32_768  # every dense-head call is padded to exactly this many points
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
    """Minimal zarr v2 writer (Zstd or vc-delta3d, '/' separator) omitting all-fill chunks."""

    def __init__(self, path: str | Path, *, attrs: dict[str, Any] | None, overwrite: bool = False,
                 workers: int = 8, async_transfers: bool = True):
        """attrs=None opens an existing group to add arrays; otherwise a new group is started."""
        self.path = Path(path)
        self.arrays: dict[str, tuple[tuple[int, ...], tuple[int, ...], np.dtype]] = {}
        self.fill_values: dict[str, int | float] = {}
        self.codecs: dict[str, str] = {}
        self.channel_axes: dict[str, str] = {}
        if attrs is None:
            for meta_path in self.path.glob("*/.zarray"):
                meta = json.loads(meta_path.read_text())
                self.arrays[meta_path.parent.name] = (tuple(meta["shape"]), tuple(meta["chunks"]),
                                                      np.dtype(meta["dtype"]))
                self.fill_values[meta_path.parent.name] = meta.get("fill_value", 0)
                self.codecs[meta_path.parent.name] = (meta.get("compressor") or {}).get("id", "zstd")
                self.channel_axes[meta_path.parent.name] = (meta.get("compressor") or {}).get("channel_axis", "last")
        else:
            if self.path.exists() and any(self.path.iterdir()):
                if not overwrite:
                    raise FileExistsError(f"{self.path} exists; pass --resume or --overwrite")
                if not (self.path / ".zgroup").is_file():
                    raise ValueError(f"refusing to overwrite a directory without .zgroup: {self.path}")
                shutil.rmtree(self.path)
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
            (self.path / ".zattrs").write_text(json.dumps(attrs, indent=2))
        self.zstd = Zstd(level=3)
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)
        self.pending: list[concurrent.futures.Future] = []
        self.async_transfers = async_transfers
        self._write_lock = threading.Lock()
        self._transfer_pool = None
        self._transfers = []
        self._copy_streams = {}

    def create(self, name: str, shape: Sequence[int], chunks: Sequence[int], dtype,
               attrs: dict[str, Any] | None = None, *, fill_value: int | float = 0, codec: str = "zstd",
               channel_axis: str = "last") -> None:
        """vc-delta3d needs integer 4D chunks holding all channels, first or last."""
        dtype = np.dtype(dtype)
        if channel_axis not in ("first", "last"):
            raise ValueError("channel_axis must be first or last")
        if codec not in CODECS:
            raise ValueError(f"codec must be one of {CODECS}")
        if codec == "vc-delta3d":
            axis = 0 if channel_axis == "first" else 3
            if len(shape) != 4 or len(chunks) != 4 or int(chunks[axis]) != int(shape[axis]) or dtype.kind not in "iu":
                raise ValueError(f"{name}: vc-delta3d needs integer 4D chunks with all channels")
            compressor = {"id": "vc-delta3d", "quant": 1, "channel_axis": channel_axis}
        else:
            compressor = self.zstd.get_config()
        root = self.path / name
        root.mkdir(exist_ok=True)
        if attrs:
            (root / ".zattrs").write_text(json.dumps(attrs, indent=2))
        meta = {"zarr_format": 2, "shape": [int(v) for v in shape], "chunks": [int(v) for v in chunks],
                "dtype": dtype.str, "fill_value": fill_value, "compressor": compressor,
                "filters": None, "order": "C", "dimension_separator": "/"}
        (root / ".zarray").write_text(json.dumps(meta, indent=2))
        self.arrays[name] = (tuple(meta["shape"]), tuple(meta["chunks"]), dtype)
        self.fill_values[name] = fill_value
        self.codecs[name] = codec
        self.channel_axes[name] = channel_axis

    def write(self, name: str, start: Sequence[int], block: np.ndarray) -> None:
        """Write a block whose start is chunk-aligned and which ends on chunks or the array edge."""
        with self._write_lock:
            self._write(name, start, block)

    def write_tensor(self, name: str, start: Sequence[int], block: torch.Tensor) -> None:
        """Stage an immutable CUDA block to pinned memory while inference continues.

        A single submission worker waits for the copy event before reading host data.
        At most four blocks can be in flight; flush also waits for their chunk writes.
        """
        if not block.is_cuda or not self.async_transfers:
            self.write(name, start, block.cpu().numpy())
            return
        while self._transfers and (self._transfers[0].done() or len(self._transfers) >= 4):
            self._transfers.pop(0).result()
        if self._transfer_pool is None:
            self._transfer_pool = concurrent.futures.ThreadPoolExecutor(1)
        if block.device not in self._copy_streams:
            self._copy_streams[block.device] = torch.cuda.Stream(device=block.device)
        stream = self._copy_streams[block.device]
        block = block.contiguous()
        host = torch.empty(block.shape, dtype=block.dtype, device="cpu", pin_memory=True)
        stream.wait_stream(torch.cuda.current_stream(block.device))
        with torch.cuda.stream(stream):
            host.copy_(block, non_blocking=True)
            block.record_stream(stream)
            ready = torch.cuda.Event()
            ready.record(stream)

        def submit():
            ready.synchronize()
            self.write(name, start, host.numpy())

        self._transfers.append(self._transfer_pool.submit(submit))

    def _write(self, name: str, start: Sequence[int], block: np.ndarray) -> None:
        shape, chunks, dtype = self.arrays[name]
        fill_value = self.fill_values[name]
        start = tuple(int(v) for v in start) + (0,) * (len(shape) - len(start))
        for axis, (s, c, n, size) in enumerate(zip(start, chunks, block.shape, shape)):
            if s % c or ((s + n) % c and s + n != size):
                raise ValueError(f"{name}: block {start}+{block.shape} is not chunk-aligned on axis {axis}")
        counts = [-(-n // c) for n, c in zip(block.shape, chunks)]
        for index in np.ndindex(*counts):
            lo = [i * c for i, c in zip(index, chunks)]
            part = block[tuple(slice(l, l + c) for l, c in zip(lo, chunks))]
            if np.all(part == fill_value):
                continue
            chunk_index = [(s + l) // c for s, l, c in zip(start, lo, chunks)]
            if part.shape != chunks:
                padded = np.full(chunks, fill_value, dtype=dtype)
                padded[tuple(slice(0, n) for n in part.shape)] = part
                part = padded
            self._submit(self.path / name / "/".join(map(str, chunk_index)), np.ascontiguousarray(part, dtype),
                         self.codecs[name], self.channel_axes[name])

    def _submit(self, path: Path, data: np.ndarray, codec: str = "zstd", channel_axis: str = "last") -> None:
        pending = []
        for future in self.pending:
            if future.done():
                future.result()
            else:
                pending.append(future)
        self.pending = pending
        while len(self.pending) > 64:
            self.pending.pop(0).result()
        self.pending.append(self.pool.submit(self._store, path, data, codec, channel_axis))

    def _store(self, path: Path, data: np.ndarray, codec: str = "zstd", channel_axis: str = "last") -> None:
        if codec == "vc-delta3d":
            from vc_reader import _reader

            # Decoded and compared with the input before it is returned.
            last = channel_axis == "last"
            encoded = _reader._encode_delta3d_channels_verified(
                data.tobytes(), list(data.shape[:3] if last else data.shape[1:]),
                int(data.shape[3] if last else data.shape[0]), last, data.itemsize)
        else:
            encoded = self.zstd.encode(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(encoded)
        os.replace(tmp, path)

    def flush(self) -> None:
        """Wait for all chunk writes before recording a completed slab."""
        for future in self._transfers:
            future.result()
        self._transfers.clear()
        for future in self.pending:
            future.result()
        self.pending.clear()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self._transfer_pool is not None:
                self._transfer_pool.shutdown(wait=True)
            self.pool.shutdown(wait=True)

    def clear_z_range(self, z_range: tuple[int, int], *, array_z_start: int,
                      steps: dict[str, int]) -> None:
        """Remove an unfinished slab's chunks before retrying (including now-empty chunks)."""
        self.flush()
        for name, (shape, chunks, _) in self.arrays.items():
            step = steps[name.removesuffix("_scale")]
            lo, hi = ((z - array_z_start) // step for z in z_range)
            hi = min(hi, shape[0])
            if lo % chunks[0] or (hi % chunks[0] and hi != shape[0]):
                raise ValueError(f"{name}: retry slab is not chunk aligned")
            for z in range(lo // chunks[0], -(-hi // chunks[0])):
                root = self.path / name / str(z)
                if root.exists():
                    shutil.rmtree(root)


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
                 codec: str | None = None,
                 compile_model: bool = False,
                 amp_dtype: torch.dtype = torch.bfloat16, point_chunk_size: int = DENSE_POINT_CHUNK_SIZE,
                 strip_device: str | torch.device = "cpu", point_sampler: str = "auto",
                 transformer_precision: str = "bf16", column_block_size: int = 16):
        """dtypes overrides the storage (a STORAGE key) of individual outputs. strip_device holds the
        overlap strips pending for the next row (one per column; the GPU is faster when it has room)."""
        self.reader, self.loaded, self.device = reader, loaded, loaded.device
        self.strip_device = torch.device(strip_device)
        self.column_block_size = int(column_block_size)
        if self.column_block_size < 0:
            raise ValueError("column_block_size must be nonnegative")
        self.backbone, self.dense_head = loaded.backbone, loaded.dense_head
        self.backbone.eval().requires_grad_(False)
        if self.dense_head is not None:
            self.dense_head.eval().requires_grad_(False)
        self.lattice, self.dtype, self.amp_dtype = tuple(lattice), dtype, amp_dtype
        self.point_chunk_size = int(point_chunk_size)
        if transformer_precision not in ("bf16", "fp8"):
            raise ValueError("transformer_precision must be 'bf16' or 'fp8'")
        if transformer_precision == "fp8" and (not compile_model or amp_dtype != torch.bfloat16):
            raise ValueError("FP8 transformer inference requires compilation and BF16 autocast")
        self.transformer_precision = transformer_precision
        self.compile_mode = "max-autotune-no-cudagraphs" if transformer_precision == "fp8" else "default"
        if point_sampler not in ("auto", "torch"):
            raise ValueError("point_sampler must be 'auto' or 'torch'")
        self.point_sampler = point_sampler
        self.sampling_backend = "torch"
        if point_sampler == "auto" and self.device.type == "cuda":
            from dinovol_2.ops.selected_interpolation import available
            if available():
                self.sampling_backend = "triton"
        self._sampling_geometry = {}
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
        if codec is not None and codec not in CODECS:
            raise ValueError(f"codec must be one of {CODECS}")
        self.codecs = {name: codec or STORAGE[storage][3] for name, storage in self.dtypes.items()}
        if any(self.codecs[name] == "vc-delta3d" and np.dtype(STORAGE[storage][0]).kind not in "iu"
               for name, storage in self.dtypes.items()):
            raise ValueError("vc-delta3d requires integer outputs; use --dtype int8")
        # Accumulated grids; tokens_unit is written from the tokens accumulator.
        accumulated = {"tokens" if name == "tokens_unit" else name for name in self.outputs}
        self.specs = [OutputSpec(name, patch if name == "tokens" else (dense_stride,) * 3, channels[name])
                      for name in OUTPUT_NAMES if name in accumulated]
        self.pipeline = A.Pipeline(transforms, targets={"image": A.Image(valid=None)}, backend="cuda",
                                   compact=False)
        _configure_backbone_for_spatial_shape(self.backbone, tuple(lat.tile for lat in self.lattice))
        self.backbone.return_embed_skips = any(spec.name != "tokens" for spec in self.specs)
        if transformer_precision == "fp8":
            from dinovol_2.ops.fp8_inference import enable_fp8_transformer
            enable_fp8_transformer(self.backbone)
        # Compile the methods actually called below: compiling Module.forward would not cover
        # forward_features / fpn_grid / readout. Compilation is lazy, on the first inference call.
        self._forward_features = self.backbone.forward_features
        if self.backbone.return_embed_skips:
            self._fpn_grid = self.dense_head.fpn_grid
            self._readout = self.dense_head.readout
        if compile_model:
            options = dict(backend="inductor", mode=self.compile_mode, dynamic=False)
            self._forward_features = torch.compile(self._forward_features, **options)
            if self.backbone.return_embed_skips:
                self._fpn_grid = torch.compile(self._fpn_grid, **options)
                self._readout = torch.compile(self._readout, **options)

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
            dtype, scaled, value_scale, _ = STORAGE[self.dtypes[name]]
            sink.create(name, shape, chunks, dtype, attrs={"value_scale": value_scale} if value_scale else None,
                        codec=self.codecs[name])
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
        """Embed a region in bounded-width column blocks, retaining overlaps on their device.

        Blocks own disjoint output cells and each recompute one left halo column.
        This bounds row-strip memory independently of the full volume width.
        """
        stride = self.lattice[2].stride
        lo, hi = x_range or (0, int(self.reader.shape[2]))
        first, stop = lo // stride, (min(hi, int(self.reader.shape[2])) - 1) // stride + 1
        width = self.column_block_size or max(1, stop - first)
        totals = {}
        for begin in range(first, stop, width):
            end = min(begin + width, stop)
            if progress and stop - first > width:
                print(f"column block: x={max(lo, begin * stride)}:{min(hi, end * stride)}", flush=True)
            stats = self._run_columns(
                sink, z_range=z_range, y_range=y_range,
                x_range=(max(lo, begin * stride), min(hi, end * stride)),
                progress=progress, need=need, array_z_range=array_z_range, read=read)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
        return totals

    def _run_columns(self, sink, *, z_range: tuple[int, int], y_range: tuple[int, int] | None = None,
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
                            top.pop(kx, None)
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
                        # This strip has no further consumers. Releasing it here avoids holding
                        # both complete rows at once, especially when strips live on the GPU.
                        self._finish_column(sink, acc, valid, (z0, z1), ky, kx, cells_y, cells_x,
                                            top.pop(kx, None), left, next_top, stats, array_z_range)
                        # A view would retain the entire multi-GiB accumulator during
                        # the next column's model/transform allocations.
                        left = {spec.name: acc[spec.name][:, :ly.stride // spec.step[1], lx.stride // spec.step[2]:].clone()
                                for spec in self.specs}
                    else:
                        # No valid output here: neighbors' strips into this column are dropped too.
                        stats["skipped_columns"] += 1
                        top.pop(kx, None)
                        left = None
                    del acc, valid
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
        image = self._normalize_column(raw)
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
                out = self._forward_features(tile, masks=None, view_kind="global")
            tokens = out["x_norm_patchtokens"][0]
            token_grid = tokens.reshape(*(lat.tile // p for lat, p in zip(self.lattice, self.patch)), -1)
            for spec in self.specs:
                if spec.name == "tokens":
                    self._add_tokens(acc["tokens"], token_grid, (lo, hi), z0, a_z)
            if point_specs:
                self._add_points(acc, valid, point_specs, token_grid, out["embed_skips"], (lo, hi), z0, a_z)
        return acc, valid

    def _normalize_column(self, raw):
        batch = {"image": raw[None, None]}
        try:
            return self.pipeline.batch(batch, seeds=[0], device=self.device)["image"][0, 0]
        except RuntimeError as error:
            if self.device.type != "cuda" or str(error) != "CUDA transform: out of memory":
                raise
        # vc_reader's scratch allocations use CUDA's pool, outside PyTorch's
        # allocator. They cannot reclaim idle PyTorch blocks automatically.
        # Leave the except block first to release the failed batch's traceback
        # and tensors, then return cached blocks to CUDA and retry only once.
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        return self.pipeline.batch(batch, seeds=[0], device=self.device)["image"][0, 0]

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

    def _sample_points(self, grid, points, input_shape):
        key = (tuple(grid.shape[1:4]), tuple(input_shape), points.device, points.dtype)
        if key not in self._sampling_geometry:
            self._sampling_geometry[key] = grid_sample_geometry(grid.shape[1:4], input_shape, points)
        return sample_grid_points(grid, points, input_shape, geometry=self._sampling_geometry[key],
                                  use_triton=self.point_sampler == "auto")

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
        # Fixed-size calls keep GEMM shapes, hence kernels, independent of how many points a tile has,
        # so each point's embedding does not depend on the requested region (bf16 kernels otherwise
        # vary with the batch size on some GPUs).
        n = coords.shape[0]
        chunk = self.point_chunk_size
        padded = torch.cat([coords, coords[-1:].expand(-n % chunk, 3)]) if n % chunk else coords
        with torch.autocast(self.device.type, dtype=self.amp_dtype):
            grid = token_grid[None]
            fpn_grid = self._fpn_grid(grid, skips)
            skip_grids = [level.permute(0, 2, 3, 4, 1) for level in skips]
            for begin in range(0, n, chunk):
                points = padded[None, begin:begin + chunk]
                fpn = head.fpn_norm(self._sample_points(fpn_grid, points, input_shape).to(grid.dtype))
                readout = self._readout(self._sample_points(grid, points, input_shape),
                                       [self._sample_points(g, points, input_shape) for g in skip_grids],
                                       fpn, dtype=grid.dtype)
                w = weight[begin:begin + chunk, None]
                rows = flat[begin:begin + chunk]
                for spec in specs:
                    target = acc[spec.name].view(-1, spec.channels)
                    target.index_add_(0, rows, readout[spec.name][0, :len(rows)].float() * w)

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
                below[spec.name] += a[:, sy:, :sx].to(self.strip_device)
            else:
                below[spec.name] = a[:, sy:, :sx].to(self.strip_device, copy=True)
            corner = next_top.setdefault(kx + 1, {})
            corner[spec.name] = torch.zeros((a.shape[0], ry, sx, a.shape[3]), device=self.strip_device)
            corner[spec.name][:, :, :rx] = a[:, sy:, sx:].to(self.strip_device)
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
                self._write_output(sink, name, (zi, y0, x0), data[:nz, :ny, :nx])
                if scale is not None:
                    self._write_output(sink, f"{name}_scale", (zi, y0, x0), scale[:nz, :ny, :nx])
        if ky in cells_y and kx in cells_x:
            stats["cells_written"] += 1

    @staticmethod
    def _write_output(sink, name, start, block):
        if hasattr(sink, "write_tensor"):
            sink.write_tensor(name, start, block)
        else:
            sink.write(name, start, block.cpu().numpy())


# ---------------------------------------------------------------------------
# CLI


def plan_slabs(z_range: tuple[int, int], slab_depth: int) -> list[tuple[int, int]]:
    """Fixed-depth processing slabs; the last may overhang the output array."""
    lo, hi = z_range
    if lo < 0 or hi <= lo or slab_depth <= 0:
        raise ValueError("need a nonempty, nonnegative z range and positive slab depth")
    return [(z, z + slab_depth) for z in range(lo, hi, slab_depth)]


def resolve_volume_source(volume: str, level: int, cache_dir: str | None) -> tuple[str, str | None]:
    """Treat a local vc_reader cache as a cache of its remote source, allowing misses to fetch."""
    if "://" in volume:
        return volume.rstrip("/"), cache_dir
    path = Path(volume).resolve()
    meta_path = path / str(level) / ".zattrs"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    source = meta.get("vc_reader", {}).get("source_url")
    if not source:
        return str(path), cache_dir
    suffix = f"/{level}"
    if not source.endswith(suffix):
        raise ValueError(f"cached source {source!r} does not match --level {level}")
    if cache_dir is None:
        relative = Path(source.replace("://", "/", 1)).parts
        array_path = meta_path.parent
        if tuple(array_path.parts[-len(relative):]) != relative:
            raise ValueError("cannot infer the cache root from this cache path; pass --cache-dir")
        cache_dir = str(array_path.parents[len(relative) - 1])
    return source[:-len(suffix)], cache_dir


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


@contextmanager
def output_lock(path: Path):
    """One CLI writer per group; the OS releases the lock even after a killed process."""
    import fcntl  # supported on both Linux and macOS

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Leave the lock inode in place: unlinking would let concurrent writers lock different files.
    with (path.parent / f".{path.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another embedding process is writing {path}") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def validate_resume(saved: dict, expected: dict, path: Path) -> None:
    """Check configuration and on-disk schemas before modifying an existing output."""
    for key, value in expected.items():
        # Earlier manifests imply BF16 and the original fixed 32768-point batch.
        default = {"point_batch_size": DENSE_POINT_CHUNK_SIZE, "transformer_precision": "bf16"}.get(key)
        if saved.get(key, default) != value:
            raise ValueError(f"cannot resume: {key} differs from the saved run")
    if not (path / ".zgroup").is_file():
        raise ValueError("cannot resume: missing .zgroup")
    wanted = set()
    for name, info in expected["outputs"].items():
        dtype, scaled, value_scale, _ = STORAGE[info["dtype"]]
        arrays = [(name, info["shape_zyxc"], info["chunks_zyxc"], dtype, info["codec"])]
        if scaled:
            arrays.append((name + "_scale", info["shape_zyxc"][:3], info["chunks_zyxc"][:3],
                           np.float16, "zstd"))
        for array, shape, chunks, dt, codec in arrays:
            wanted.add(array)
            meta = json.loads((path / array / ".zarray").read_text())
            compressor = ({"id": "vc-delta3d", "quant": 1, "channel_axis": "last"}
                          if codec == "vc-delta3d" else Zstd(level=3).get_config())
            fields = dict(zarr_format=2, shape=shape, chunks=chunks, dtype=np.dtype(dt).str,
                          fill_value=0, order="C", filters=None, dimension_separator="/", compressor=compressor)
            if any(meta.get(k) != v for k, v in fields.items()):
                raise ValueError(f"cannot resume: {array} storage metadata differs")
        attrs_path = path / name / ".zattrs"
        attrs = json.loads(attrs_path.read_text()) if attrs_path.exists() else {}
        if attrs.get("value_scale") != value_scale:
            raise ValueError(f"cannot resume: {name} value_scale differs")
    if {p.parent.name for p in path.glob("*/.zarray")} != wanted:
        raise ValueError("cannot resume: output arrays differ")
    completed = saved.get("completed_slabs")
    if not isinstance(completed, dict) or any(k not in {str(i) for i in range(len(expected["slabs"]))}
                                               for k in completed):
        raise ValueError("cannot resume: invalid completed_slabs manifest")


def pending_slab_groups(slabs, completed, group_size):
    """Consecutive unfinished slabs, never spanning a committed slab or a gap."""
    if group_size < 1:
        raise ValueError("slab group size must be positive")
    group = []
    for i, slab in enumerate(slabs):
        if group and (str(i) in completed or slab[0] != slabs[group[-1]][1] or len(group) == group_size):
            yield group
            group = []
        if str(i) not in completed:
            group.append(i)
    if group:
        yield group


def run_slabs(embedder, sink, attrs: dict, *, resume: bool, read: str = "band",
              slab_group_size: int = 1) -> dict[str, int]:
    """Commit each slab only after its async chunk writes succeed; retry partial slabs cleanly."""
    z_range = tuple(attrs["z_range"])
    completed = attrs.setdefault("completed_slabs", {})
    steps = {name: info["step_zyx"][0] for name, info in attrs["outputs"].items()}
    for group in pending_slab_groups(attrs["slabs"], completed, slab_group_size):
        slab = (attrs["slabs"][group[0]][0], attrs["slabs"][group[-1]][1])
        print(f"slabs {group[0] + 1}-{group[-1] + 1}/{len(attrs['slabs'])}: "
              f"z={slab[0]}:{min(slab[1], z_range[1])}", flush=True)
        if resume:
            sink.clear_z_range(slab, array_z_start=z_range[0], steps=steps)
        stats = embedder.run(sink, z_range=slab, array_z_range=z_range,
                             y_range=attrs["y_range"], x_range=attrs["x_range"], read=read)
        sink.flush()
        # One atomic manifest update commits the entire group after all its writes.
        # Counts describe actual work (including reused boundary tiles), so store
        # them once on the group's first slab; the other completion entries are empty.
        for i in group:
            completed[str(i)] = stats if i == group[0] else {}
            # Performance backends may change between slabs (small rounding differences
            # are allowed). Record them without invalidating old manifests.
            attrs.setdefault("slab_runtime", {})[str(i)] = {
                "point_sampler": embedder.sampling_backend,
                "async_output": sink.async_transfers,
                "group_slabs": group,
                "column_block_size": embedder.column_block_size,
            }
        atomic_json(sink.path / ".zattrs", attrs)
    totals: dict[str, int] = {}
    for stats in completed.values():
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value
    attrs["run"] = totals
    attrs["complete"] = True
    atomic_json(sink.path / ".zattrs", attrs)
    return totals


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--add-unit-tokens", metavar="GROUP", default=None,
                        help="only add tokens_unit to an existing output group from its stored tokens")
    parser.add_argument("--checkpoint")
    parser.add_argument("--volume", help="ome-zarr URL or path (s3:// uses --cache-dir)")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true", help="fail on uncached chunks instead of fetching")
    parser.add_argument("--no-cache-writes", action="store_true", help="fetch misses without caching them")
    parser.add_argument("--connections", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16, help="vc_reader read threads")
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="PyTorch CPU threads (default: PyTorch's environment setting)")
    parser.add_argument("--writer-workers", type=int, default=8, help="compression/write worker threads")
    parser.add_argument("--strip-device", choices=("cpu", "cuda"), default="cpu",
                        help="store row-overlap buffers on CPU or GPU (cuda uses more VRAM)")
    parser.add_argument("--column-block-size", type=int, default=16,
                        help="maximum output columns per pass, bounding row-strip memory "
                             "(default: 16; 0 processes the full width)")
    parser.add_argument("--point-batch-size", type=int, default=DENSE_POINT_CHUNK_SIZE,
                        help="fixed dense-head point batch; must stay unchanged on resume")
    parser.add_argument("--point-sampler", choices=("auto", "torch"), default="auto",
                        help="auto uses fused Triton interpolation on CUDA (small rounding differences); "
                             "torch uses the original sampler")
    parser.add_argument("--async-output", action=argparse.BooleanOptionalAction, default=True,
                        help="overlap pinned-memory output copies and writes with GPU inference (default: on)")
    parser.add_argument("--z-start", type=int)
    parser.add_argument("--depth", type=int, help="region depth (default: 192 with --z-start)")
    parser.add_argument("--full-volume", action="store_true", help="embed all z (also the default without --z-start)")
    parser.add_argument("--slab-depth", type=int,
                        help="processing depth, a multiple of lattice stride (default: two strides)")
    parser.add_argument("--slab-group-size", type=int, default=1,
                        help="process adjacent unfinished slabs together to reuse boundary tiles; "
                             "increases working memory, preserves the saved slab plan (default: 1)")
    parser.add_argument("--read", choices=("band", "column"), default="band",
                        help="band prefetches a row within the column block; column uses less host memory")
    parser.add_argument("--y-range", type=int, nargs=2, default=None, help="restrict to lattice cells overlapping [lo, hi)")
    parser.add_argument("--x-range", type=int, nargs=2, default=None, help="restrict to lattice cells overlapping [lo, hi)")
    parser.add_argument("--outputs", default=",".join(OUTPUT_NAMES))
    parser.add_argument("--dense-stride", type=int, default=1)
    parser.add_argument("--dtype", choices=DTYPES, default="int8",
                        help="int8: tokens with per-vector scales, dense/ink at a fixed 1/127 scale (vc-delta3d)")
    parser.add_argument("--codec", choices=CODECS,
                        help="override embedding compression; vc-delta3d requires int8 (float scale arrays use Zstd)")
    parser.add_argument("--tile", type=int, default=None, help="tile edge in voxels (default: training global crop)")
    parser.add_argument("--margin", type=int, default=None,
                        help="context-only voxels per tile face (default: tile/6, patch-aligned)")
    parser.add_argument("--ramp", type=int, default=None, help="blend ramp in voxels (default: tile/6, patch-aligned)")
    parser.add_argument("--normalization-stats", default=None,
                        help="statistics JSON or an earlier output's .zattrs to reuse (skips sampling)")
    parser.add_argument("--stats-chunk-stride", type=int, default=8)
    parser.add_argument("--branch", default="teacher")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="compile backbone and dense-head inference methods with torch.compile (default: on)")
    parser.add_argument("--transformer-precision", choices=("bf16", "fp8"), default="bf16",
                        help="FP8 is lossy, opt-in transformer compute on NVIDIA GPUs >= SM89; "
                             "requires --compile and a separate output from an existing BF16 run")
    parser.add_argument("--output")
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--overwrite", action="store_true")
    existing.add_argument("--resume", action="store_true", help="continue an existing run with the same configuration")
    args = parser.parse_args(argv)
    missing = [f"--{name.replace('_', '-')}" for name in ("checkpoint", "volume", "output")
               if getattr(args, name) is None]
    if args.add_unit_tokens is None and missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    if args.full_volume and (args.z_start is not None or args.depth is not None):
        parser.error("--full-volume cannot be combined with --z-start or --depth")
    if args.depth is not None and args.z_start is None:
        parser.error("--depth requires --z-start")
    for name in ("depth", "slab_depth", "dense_stride", "stats_chunk_stride", "connections", "threads", "tile",
                 "torch_threads", "writer_workers", "point_batch_size", "slab_group_size"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.z_start is not None and args.z_start < 0:
        parser.error("--z-start must be nonnegative")
    if args.column_block_size < 0:
        parser.error("--column-block-size must be nonnegative")
    if args.codec == "vc-delta3d" and args.dtype != "int8":
        parser.error("--codec vc-delta3d requires --dtype int8")
    if args.transformer_precision == "fp8" and not args.compile:
        parser.error("--transformer-precision fp8 requires --compile")
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
        with output_lock(Path(args.add_unit_tokens)):
            print(f"wrote {add_unit_tokens(args.add_unit_tokens, overwrite=args.overwrite)} tokens_unit chunks")
        return
    with output_lock(Path(args.output)):
        embed_volume(args)


def embed_volume(args) -> None:
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    path = Path(args.output)
    saved = None
    if args.resume:
        saved = json.loads((path / ".zattrs").read_text())
        if saved.get("scheduler_version") != 1:
            raise ValueError("cannot resume: this output has no supported slab manifest")
    elif path.exists() and any(path.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{path} exists; pass --resume or --overwrite")
    volume, cache_dir = resolve_volume_source(args.volume, args.level, args.cache_dir)
    if volume != args.volume:
        print(f"volume: {volume}\ncache: {cache_dir}", flush=True)
    torch.backends.cudnn.benchmark = False
    s3 = {"anon": True} if volume.startswith("s3://") else None
    with open_volume(volume, level=args.level, s3_storage_options=s3, cache_dir=cache_dir,
                     offline=args.offline, cache_writes=not (args.offline or args.no_cache_writes),
                     connections=args.connections, threads=args.threads) as reader:
        _embed_volume(args, reader, volume, saved)


def _embed_volume(args, reader, volume: str, saved: dict | None) -> None:
    loaded = load_backbone_from_checkpoint(args.checkpoint, preferred_branch=args.branch)
    checkpoint_step = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True).get("step")
    with Path(args.checkpoint).open("rb") as checkpoint:
        checkpoint_sha256 = hashlib.file_digest(checkpoint, "sha256").hexdigest()

    scheme = loaded.normalization_scheme
    statistics = None
    if scheme not in ("none", "ct"):
        if args.normalization_stats:
            # A statistics dict, or the .zattrs of an earlier output (reuse keeps slabs consistent).
            supplied = json.loads(Path(args.normalization_stats).read_text())
            statistics = supplied.get("normalization", supplied).get("statistics", supplied)
        elif saved is not None:
            statistics = saved["normalization"]["statistics"]
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
                              dense_stride=args.dense_stride, lattice=lattice, dtype=args.dtype,
                              dtypes=CLI_DTYPES[args.dtype], codec=args.codec, compile_model=args.compile,
                              strip_device=args.strip_device, point_chunk_size=args.point_batch_size,
                              point_sampler=args.point_sampler, transformer_precision=args.transformer_precision,
                              column_block_size=args.column_block_size)
    z_range = ((0, int(reader.shape[0])) if args.z_start is None else
               (args.z_start, args.z_start + (args.depth if args.depth is not None else 192)))
    if any(z_range[0] % spec.step[0] for spec in embedder.specs):
        raise ValueError("--z-start must align to every output's grid step")
    if z_range[0] >= reader.shape[0] or z_range[1] > reader.shape[0]:
        raise ValueError(f"requested z range {z_range} is outside volume depth {reader.shape[0]}")
    for name, bounds, size in (("y", args.y_range, reader.shape[1]), ("x", args.x_range, reader.shape[2])):
        if bounds is not None and not (0 <= bounds[0] < bounds[1] <= size):
            raise ValueError(f"invalid {name} range {bounds} for size {size}")
    slab_depth = args.slab_depth if args.slab_depth is not None else 2 * lattice[0].stride
    if slab_depth <= 0 or slab_depth % lattice[0].stride:
        raise ValueError(f"--slab-depth must be a positive multiple of lattice stride {lattice[0].stride}")
    slabs = plan_slabs(z_range, slab_depth)
    attrs = {
        "format": FORMAT,
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha256,
        "branch": loaded.source_branch,
        "volume": volume, "level": args.level, "volume_shape": list(reader.shape),
        "z_range": list(z_range), "y_range": args.y_range, "x_range": args.x_range,
        "scheduler_version": 1, "slabs": [list(slab) for slab in slabs],
        "point_batch_size": args.point_batch_size,
        "transformer_precision": args.transformer_precision,
        "execution": {"compile": args.compile, "backend": "inductor" if args.compile else None,
                      "mode": embedder.compile_mode if args.compile else None, "dynamic": False,
                      "torch_version": str(torch.__version__)},
        "lattice": {"tile": tile, "margin": margin, "ramp": ramp, "stride": lattice[0].stride,
                    "tile_origin": "k*stride - margin", "support": "[k*stride, k*stride + stride + ramp)"},
        "normalization": {"scheme": scheme, "statistics": statistics,
                          "intensity_properties": loaded.intensity_properties,
                          "applied": "fixed volume-wide statistics" + (
                              "; robust clips only the upper percentile" if scheme == "robust" else "")},
        "amp_dtype": "bfloat16", "masked": "outputs whose block has no nonzero voxel are zero vectors",
        "outputs": {},
    }
    for name, spec, shape, chunks in embedder.arrays(z_range, z_cell=True):
        attrs["outputs"][name] = {
            "shape_zyxc": list(shape), "step_zyx": list(spec.step),
            "chunks_zyxc": list(chunks),
            "origin_zyx": [z_range[0], 0, 0],
            "index_to_voxel": "voxel block [origin + i*step, origin + (i+1)*step)",
            "dtype": embedder.dtypes[name],
            "quantization": QUANTIZATION[embedder.dtypes[name]],
            "codec": embedder.codecs[name],
            "values": OUTPUT_VALUES[name],
        }
    # JSON round-trip canonicalizes tuples and dictionary values before resume comparison.
    attrs = json.loads(json.dumps(attrs))
    if saved is not None:
        validate_resume(saved, attrs, Path(args.output))
        attrs = saved
        sink = ZarrGroupWriter(args.output, attrs=None, workers=args.writer_workers,
                               async_transfers=args.async_output)
    else:
        attrs.update(completed_slabs={}, complete=False)
        sink = ZarrGroupWriter(args.output, attrs=attrs, overwrite=args.overwrite, workers=args.writer_workers,
                               async_transfers=args.async_output)
    try:
        if saved is None:
            embedder.create_arrays(sink, z_range, z_cell=True)
            atomic_json(Path(args.output) / ".zattrs", attrs)
        stats = run_slabs(embedder, sink, attrs, resume=args.resume, read=args.read,
                          slab_group_size=args.slab_group_size)
    finally:
        sink.close()
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
