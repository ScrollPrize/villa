"""Sparse native-volume inference planned from tifxyz occupancy."""

from __future__ import annotations

import itertools
import logging
import math
import shutil
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import numpy as np
from numcodecs import Blosc
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import vesuvius.tifxyz as tifxyz

from vesuvius.ink_detection.data.geometry import (
    SURFACE_MASK_MAX_DISTANCE_LEVEL0_VOXELS,
    native_tifxyz_pyramid_params,
    native_volume_downsample_factor,
    project_surface_distance,
    select_flat_pixels_via_stored_resolution,
)
from vesuvius.ink_detection.data.normalization import normalize_image
from vesuvius.ink_detection.inference.inference_runtime import (
    TargetModel,
    flip_spatial,
    iter_mirror_axes,
    parse_gpu_ids,
    prepare_model_for_inference,
    resolve_amp_dtype,
)
from vesuvius.ink_detection.models.checkpoint import (
    config_from_checkpoint,
    load_checkpoint,
    load_model_state,
)
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.volume_io import (
    open_volume,
    read_bbox_with_padding,
)
from vesuvius.label_zarr import create_v2_array, open_v2_group
from vesuvius.utils.cli import HyphenUnderscoreParser


LOGGER = logging.getLogger("vesuvius.ink_detection.inference.infer_full3d_tifxyz")
DEFAULT_LEVELS = 6
DEFAULT_OVERLAP = 0.5
DEFAULT_PREFETCH_FACTOR = 2
ARRAY_DIMENSIONS = ["z", "y", "x"]
AXES = [
    {"name": "z", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]


@dataclass(frozen=True, order=True)
class ChunkId:
    """Integer ZYX identity of one storage chunk."""

    z: int
    y: int
    x: int


@dataclass(frozen=True, order=True)
class PatchSpec:
    """Inclusive top-left-front ZYX start of one inference patch."""

    z0: int
    y0: int
    x0: int


@dataclass(frozen=True)
class NativePlan:
    """Immutable occupied-chunk plan in ZYX array coordinates."""

    volume_path: str
    array_shape_zyx: tuple[int, int, int]
    chunk_shape_zyx: tuple[int, int, int]
    occupied_chunks: frozenset[ChunkId]
    expanded_chunks: frozenset[ChunkId]
    target_chunks: frozenset[ChunkId]
    patches: tuple[PatchSpec, ...]
    contribution_counts: dict[ChunkId, int]
    patch_size_zyx: tuple[int, int, int]
    stride_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class NativeModelBundle:
    """Prepared native model and its execution-device metadata."""

    model: Any
    device: Any
    amp_dtype: Any


def parse_args(argv: Sequence[str] | None = None):
    """Parse the native inference command line."""

    parser = HyphenUnderscoreParser(
        description=(
            "Run native full-3D ink inference over tifxyz-occupied chunks and "
            "write a sparse six-level uint8 OME-Zarr."
        )
    )
    parser.add_argument("tifxyz_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_zarr", type=Path)
    parser.add_argument("--resolution", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-workers", "--workers", dest="num_workers", type=int, default=4
    )
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--downsample-workers", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--chunk-halo", type=int, default=1)
    parser.add_argument(
        "--write-region", choices=("expanded", "occupied"), default="expanded"
    )
    parser.add_argument(
        "--blend-mode", choices=("gaussian", "constant"), default="gaussian"
    )
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-batch-size", type=int, default=None)
    parser.add_argument(
        "--amp-dtype", choices=("auto", "default", "fp16", "bf16"), default="auto"
    )
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=True)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-target-chunks", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cache-max-gb", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    checks = (
        (args.batch_size > 0, "--batch-size must be positive"),
        (args.num_workers >= 0, "--num-workers must be >= 0"),
        (args.prefetch_factor > 0, "--prefetch-factor must be positive"),
        (args.downsample_workers > 0, "--downsample-workers must be positive"),
        (
            args.tta_batch_size is None or args.tta_batch_size > 0,
            "--tta-batch-size must be positive",
        ),
    )
    for valid, message in checks:
        if not valid:
            parser.error(message)
    try:
        args.gpu_ids = parse_gpu_ids(args.gpus)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def is_url_like(path: str | Path) -> bool:
    """Return whether a source has both a URL scheme and network location."""

    parsed = urlparse(str(path))
    return bool(parsed.scheme and parsed.netloc)


def read_volume_source(tifxyz_dir: Path) -> str:
    """Resolve a non-URL volume source relative only to the tifxyz directory."""

    source_path = Path(tifxyz_dir) / "volume_source.txt"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing volume_source.txt in {tifxyz_dir}")
    source = source_path.read_text(encoding="utf-8").strip()
    if not source:
        raise ValueError(f"volume_source.txt is empty: {source_path}")
    expanded = Path(source).expanduser()
    if not is_url_like(source) and not expanded.is_absolute():
        source = str((Path(tifxyz_dir) / expanded).resolve())
    return source


def read_tifxyz_points(
    tifxyz_dir: Path, *, native_coordinate_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read level-0 XYZ TIFF coordinates and return scaled arrays plus validity."""

    arrays = []
    for name in ("x.tif", "y.tif", "z.tif"):
        path = Path(tifxyz_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing tifxyz coordinate file: {path}")
        arrays.append(np.asarray(tifffile.imread(path), dtype=np.float32))
    x, y, z = arrays
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(f"x/y/z tif shapes must match, got {x.shape}, {y.shape}, {z.shape}")
    scale = float(native_coordinate_scale)
    if scale <= 0.0:
        raise ValueError(f"native_coordinate_scale must be positive, got {scale}")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    valid &= (x >= 0.0) & (y >= 0.0) & (z >= 0.0)
    if scale != 1.0:
        x, y, z = x * scale, y * scale, z * scale
    return x, y, z, valid


def _validate_zyx_shape(name: str, values: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in values)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain three positive integers, got {values}")
    return values


def chunk_grid_shape(
    array_shape_zyx: Sequence[int], chunk_shape_zyx: Sequence[int]
) -> tuple[int, int, int]:
    """Return ceil-divided ZYX chunk-grid dimensions."""

    array_shape_zyx = _validate_zyx_shape("array_shape_zyx", array_shape_zyx)
    chunk_shape_zyx = _validate_zyx_shape("chunk_shape_zyx", chunk_shape_zyx)
    return tuple(
        int(math.ceil(int(size) / int(chunk)))
        for size, chunk in zip(array_shape_zyx, chunk_shape_zyx)
    )


def tifxyz_occupied_chunks(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    array_shape_zyx: tuple[int, int, int],
    chunk_shape_zyx: tuple[int, int, int],
) -> set[ChunkId]:
    """Map finite, in-bounds tifxyz positions to occupied ZYX chunks."""

    if not np.any(valid):
        raise ValueError("No finite non-negative tifxyz points were found")
    indices_zyx = np.stack(
        [np.floor(z[valid]), np.floor(y[valid]), np.floor(x[valid])], axis=1
    ).astype(np.int64, copy=False)
    shape = np.asarray(array_shape_zyx, dtype=np.int64)
    in_bounds = ((indices_zyx >= 0) & (indices_zyx < shape)).all(axis=1)
    if not np.any(in_bounds):
        raise ValueError(f"No tifxyz points fall inside input volume shape {array_shape_zyx}")
    chunk_indices = indices_zyx[in_bounds] // np.asarray(chunk_shape_zyx)
    return {ChunkId(*(int(value) for value in row)) for row in np.unique(chunk_indices, axis=0)}


def expand_chunks(
    chunk_ids: Iterable[ChunkId],
    *,
    radius: int,
    grid_shape_zyx: tuple[int, int, int],
) -> set[ChunkId]:
    """Expand chunk identities by a clipped Chebyshev-radius halo."""

    source = set(chunk_ids)
    if int(radius) <= 0:
        return source
    expanded = set()
    for chunk in source:
        for offsets in itertools.product(range(-radius, radius + 1), repeat=3):
            values = tuple(
                value + offset
                for value, offset in zip((chunk.z, chunk.y, chunk.x), offsets)
            )
            if all(0 <= values[axis] < grid_shape_zyx[axis] for axis in range(3)):
                expanded.add(ChunkId(*values))
    return expanded


def chunk_bounds(
    chunk: ChunkId,
    *,
    chunk_shape_zyx: tuple[int, int, int],
    array_shape_zyx: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return half-open ZYX array bounds for one chunk."""

    coordinates = (chunk.z, chunk.y, chunk.x)
    starts = tuple(coordinates[axis] * chunk_shape_zyx[axis] for axis in range(3))
    stops = tuple(
        min(array_shape_zyx[axis], starts[axis] + chunk_shape_zyx[axis])
        for axis in range(3)
    )
    return starts, stops


def chunks_overlapping_box(
    starts: tuple[int, int, int],
    stops: tuple[int, int, int],
    *,
    chunk_shape_zyx: tuple[int, int, int],
    array_shape_zyx: tuple[int, int, int],
):
    """Yield chunks intersecting a half-open ZYX box after volume clipping."""

    clipped_starts = tuple(max(0, int(value)) for value in starts)
    clipped_stops = tuple(
        min(array_shape_zyx[axis], int(stops[axis])) for axis in range(3)
    )
    if any(stop <= start for start, stop in zip(clipped_starts, clipped_stops)):
        return
    ranges = [
        range(
            clipped_starts[axis] // chunk_shape_zyx[axis],
            (clipped_stops[axis] - 1) // chunk_shape_zyx[axis] + 1,
        )
        for axis in range(3)
    ]
    for values in itertools.product(*ranges):
        yield ChunkId(*values)


def _global_patch_starts_for_interval(
    *,
    interval_start: int,
    interval_stop: int,
    volume_length: int,
    patch_size: int,
    stride: int,
) -> list[int]:
    max_start = max(0, int(volume_length) - int(patch_size))
    first = max(0, ((int(interval_start) - int(patch_size) + 1) // int(stride)) * int(stride))
    starts = []
    start = first
    while start <= max_start and start < interval_stop:
        if start + patch_size > interval_start:
            starts.append(start)
        start += stride
    if interval_stop > max_start and max_start + patch_size > interval_start:
        starts.append(max_start)
    return sorted(set(starts))


def build_patch_plan_for_chunks(
    *,
    target_chunks: Iterable[ChunkId],
    array_shape_zyx: tuple[int, int, int],
    chunk_shape_zyx: tuple[int, int, int],
    patch_size_zyx: tuple[int, int, int],
    stride_zyx: tuple[int, int, int],
) -> list[PatchSpec]:
    """Return sorted global-grid patches covering every target chunk."""

    patches = set()
    for chunk in target_chunks:
        starts, stops = chunk_bounds(
            chunk,
            chunk_shape_zyx=chunk_shape_zyx,
            array_shape_zyx=array_shape_zyx,
        )
        per_axis = [
            _global_patch_starts_for_interval(
                interval_start=starts[axis],
                interval_stop=stops[axis],
                volume_length=array_shape_zyx[axis],
                patch_size=patch_size_zyx[axis],
                stride=stride_zyx[axis],
            )
            for axis in range(3)
        ]
        patches.update(PatchSpec(*values) for values in itertools.product(*per_axis))
    return sorted(patches)


def compute_chunk_contribution_counts(
    patches: Sequence[PatchSpec],
    *,
    target_chunks: set[ChunkId],
    array_shape_zyx: tuple[int, int, int],
    chunk_shape_zyx: tuple[int, int, int],
    patch_size_zyx: tuple[int, int, int],
) -> dict[ChunkId, int]:
    """Count planned patch intersections for each written chunk."""

    counts = {chunk: 0 for chunk in target_chunks}
    for patch in patches:
        starts = (patch.z0, patch.y0, patch.x0)
        stops = tuple(starts[axis] + patch_size_zyx[axis] for axis in range(3))
        for chunk in chunks_overlapping_box(
            starts,
            stops,
            chunk_shape_zyx=chunk_shape_zyx,
            array_shape_zyx=array_shape_zyx,
        ):
            if chunk in counts:
                counts[chunk] += 1
    return {chunk: count for chunk, count in counts.items() if count > 0}


def build_native_plan(
    *,
    volume_path: str,
    array_shape_zyx: tuple[int, int, int],
    chunk_shape_zyx: tuple[int, int, int],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    patch_size_zyx: tuple[int, int, int],
    overlap: float,
    chunk_halo: int,
    write_region: str,
    max_target_chunks: int | None,
) -> NativePlan:
    """Build a complete occupied-chunk and overlapping-patch plan in ZYX."""

    array_shape_zyx = _validate_zyx_shape("array_shape_zyx", array_shape_zyx)
    chunk_shape_zyx = _validate_zyx_shape("chunk_shape_zyx", chunk_shape_zyx)
    patch_size_zyx = _validate_zyx_shape("patch_size_zyx", patch_size_zyx)
    if not 0.0 <= float(overlap) < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if int(chunk_halo) < 0:
        raise ValueError(f"chunk_halo must be non-negative, got {chunk_halo}")
    if max_target_chunks is not None and int(max_target_chunks) <= 0:
        raise ValueError(
            f"max_target_chunks must be positive, got {max_target_chunks}"
        )
    occupied = tifxyz_occupied_chunks(
        x=x,
        y=y,
        z=z,
        valid=valid,
        array_shape_zyx=array_shape_zyx,
        chunk_shape_zyx=chunk_shape_zyx,
    )
    expanded = expand_chunks(
        occupied,
        radius=chunk_halo,
        grid_shape_zyx=chunk_grid_shape(array_shape_zyx, chunk_shape_zyx),
    )
    if write_region not in {"expanded", "occupied"}:
        raise ValueError(f"write_region must be 'expanded' or 'occupied', got {write_region!r}")
    target = set(expanded if write_region == "expanded" else occupied)
    if max_target_chunks is not None and len(target) > max_target_chunks:
        target = set(sorted(target)[:max_target_chunks])
    stride = tuple(
        max(1, int(round(size * (1.0 - float(overlap)))))
        for size in patch_size_zyx
    )
    patches = build_patch_plan_for_chunks(
        target_chunks=target,
        array_shape_zyx=array_shape_zyx,
        chunk_shape_zyx=chunk_shape_zyx,
        patch_size_zyx=patch_size_zyx,
        stride_zyx=stride,
    )
    counts = compute_chunk_contribution_counts(
        patches,
        target_chunks=target,
        array_shape_zyx=array_shape_zyx,
        chunk_shape_zyx=chunk_shape_zyx,
        patch_size_zyx=patch_size_zyx,
    )
    missing = target - set(counts)
    if missing:
        raise RuntimeError(f"Patch plan failed to cover {len(missing)} target chunks")
    return NativePlan(
        volume_path=str(volume_path),
        array_shape_zyx=array_shape_zyx,
        chunk_shape_zyx=chunk_shape_zyx,
        occupied_chunks=frozenset(occupied),
        expanded_chunks=frozenset(expanded),
        target_chunks=frozenset(target),
        patches=tuple(patches),
        contribution_counts=counts,
        patch_size_zyx=patch_size_zyx,
        stride_zyx=stride,
    )


def create_importance_map(
    patch_size_zyx: tuple[int, int, int], *, mode: str, sigma_scale: float = 0.125
) -> np.ndarray:
    """Return positive float32 constant or Gaussian ZYX blend weights."""

    if mode == "constant":
        return np.ones(patch_size_zyx, dtype=np.float32)
    if mode != "gaussian":
        raise ValueError(f"Unsupported blend mode {mode!r}")
    axes = []
    for size in patch_size_zyx:
        sigma = max(float(size) * float(sigma_scale), 1e-6)
        coordinate = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
        axes.append(np.exp(-0.5 * (coordinate / sigma) ** 2).astype(np.float32))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    weight /= max(float(weight.max()), np.finfo(np.float32).eps)
    return np.clip(weight, np.finfo(np.float32).eps, None).astype(np.float32)


def tta_variants(enabled: bool) -> list[tuple[int, ...]]:
    """Return the identity or all eight ZYX mirror combinations."""

    return [()] if not enabled else iter_mirror_axes((0, 1, 2))


def logits_to_probabilities(
    logits,
    *,
    patch_size_zyx: tuple[int, int, int],
):
    """Resize one-channel BCZYX logits and return sigmoid probabilities."""

    if logits.ndim != 5:
        raise ValueError(f"Expected logits [B,C,Z,Y,X], got {tuple(logits.shape)}")
    if tuple(logits.shape[-3:]) != patch_size_zyx:
        logits = F.interpolate(
            logits.float(), size=patch_size_zyx, mode="trilinear", align_corners=False
        )
    return logits.float().sigmoid()


def predict_batch(
    model,
    images,
    *,
    variants: Sequence[tuple[int, ...]],
    tta_batch_size: int | None,
    patch_size_zyx: tuple[int, int, int],
):
    """Average restored mirror predictions for one BCZYX image batch."""

    variants = list(variants)
    if not variants:
        raise RuntimeError("Mirror TTA produced no variants")
    if len(variants) == 1:
        return logits_to_probabilities(
            model(images),
            patch_size_zyx=patch_size_zyx,
        )
    maximum = len(variants) if tta_batch_size is None else min(tta_batch_size, len(variants))
    probability_sum = None
    batch_size = int(images.shape[0])
    for start in range(0, len(variants), maximum):
        group = variants[start : start + maximum]
        augmented = torch.cat([flip_spatial(images, axes) for axes in group], dim=0)
        probabilities = logits_to_probabilities(
            model(augmented),
            patch_size_zyx=patch_size_zyx,
        ).reshape(len(group), batch_size, -1, *patch_size_zyx)
        for index, axes in enumerate(group):
            restored = flip_spatial(probabilities[index], axes)
            probability_sum = restored if probability_sum is None else probability_sum + restored
    return probability_sum / float(len(variants))


class ChunkAccumulator3D:
    """Flush each target chunk after its final overlapping patch contribution."""

    def __init__(
        self,
        *,
        output,
        target_chunks: set[ChunkId],
        contribution_counts: dict[ChunkId, int],
        chunk_shape_zyx: tuple[int, int, int],
    ) -> None:
        self.output = output
        self.target_chunks = set(target_chunks)
        self.contribution_counts = dict(contribution_counts)
        self.chunk_shape_zyx = chunk_shape_zyx
        self.array_shape_zyx = tuple(int(value) for value in output.shape)
        self.seen_counts: dict[ChunkId, int] = {}
        self.buffers: dict[ChunkId, tuple[np.ndarray, np.ndarray]] = {}

    def _bounds(self, chunk: ChunkId):
        return chunk_bounds(
            chunk,
            chunk_shape_zyx=self.chunk_shape_zyx,
            array_shape_zyx=self.array_shape_zyx,
        )

    def _buffers(self, chunk: ChunkId):
        if chunk not in self.buffers:
            starts, stops = self._bounds(chunk)
            shape = tuple(stops[axis] - starts[axis] for axis in range(3))
            self.buffers[chunk] = (
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
            )
        return self.buffers[chunk]

    def add_patch(
        self,
        *,
        patch_start_zyx: tuple[int, int, int],
        probabilities: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        patch_stop = tuple(
            patch_start_zyx[axis] + probabilities.shape[axis] for axis in range(3)
        )
        for chunk in chunks_overlapping_box(
            patch_start_zyx,
            patch_stop,
            chunk_shape_zyx=self.chunk_shape_zyx,
            array_shape_zyx=self.array_shape_zyx,
        ):
            if chunk not in self.target_chunks:
                continue
            chunk_start, chunk_stop = self._bounds(chunk)
            intersection_start = tuple(
                max(patch_start_zyx[axis], chunk_start[axis]) for axis in range(3)
            )
            intersection_stop = tuple(
                min(patch_stop[axis], chunk_stop[axis]) for axis in range(3)
            )
            patch_slices = tuple(
                slice(
                    intersection_start[axis] - patch_start_zyx[axis],
                    intersection_stop[axis] - patch_start_zyx[axis],
                )
                for axis in range(3)
            )
            chunk_slices = tuple(
                slice(
                    intersection_start[axis] - chunk_start[axis],
                    intersection_stop[axis] - chunk_start[axis],
                )
                for axis in range(3)
            )
            probability_buffer, weight_buffer = self._buffers(chunk)
            weight_view = weights[patch_slices]
            probability_buffer[chunk_slices] += probabilities[patch_slices] * weight_view
            weight_buffer[chunk_slices] += weight_view
            seen = self.seen_counts.get(chunk, 0) + 1
            if seen >= self.contribution_counts[chunk]:
                self._flush(chunk)
            else:
                self.seen_counts[chunk] = seen

    def _flush(self, chunk: ChunkId) -> None:
        probability, weight = self.buffers.pop(chunk)
        self.seen_counts.pop(chunk, None)
        np.divide(probability, weight, out=probability, where=weight > 1e-6)
        np.clip(probability, 0.0, 1.0, out=probability)
        encoded = np.rint(probability * 255.0).astype(np.uint8)
        starts, stops = self._bounds(chunk)
        self.output[
            starts[0] : stops[0], starts[1] : stops[1], starts[2] : stops[2]
        ] = encoded

    def flush_remaining(self) -> None:
        for chunk in list(self.buffers):
            self._flush(chunk)


def multiscales_metadata(name: str, levels: int) -> dict[str, Any]:
    """Build OME-NGFF metadata for ZYX arrays downsampled by powers of two."""

    return {
        "multiscales": [
            {
                "name": name,
                "version": "0.4",
                "axes": AXES,
                "datasets": [
                    {
                        "path": str(level),
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [float(2**level)] * 3,
                            }
                        ],
                    }
                    for level in range(levels)
                ],
            }
        ]
    }


def pyramid_shapes(
    shape_zyx: tuple[int, int, int], levels: int = DEFAULT_LEVELS
) -> list[tuple[int, int, int]]:
    """Return ceil-halved ZYX shapes for each pyramid level."""

    shapes = []
    current = shape_zyx
    for _ in range(levels):
        shapes.append(current)
        current = tuple((value + 1) // 2 for value in current)
    return shapes


def create_output_zarr(
    output_path: Path,
    *,
    shape_zyx: tuple[int, int, int],
    chunks_zyx: tuple[int, int, int],
    levels: int = DEFAULT_LEVELS,
    overwrite: bool = False,
):
    """Create a six-level local Zarr-v2 OME prediction pyramid."""

    output_path = Path(output_path)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    group = open_v2_group(output_path)
    group.attrs.update(multiscales_metadata(output_path.stem, levels))
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    arrays = []
    for level, shape in enumerate(pyramid_shapes(shape_zyx, levels)):
        chunks = tuple(min(chunks_zyx[axis], shape[axis]) for axis in range(3))
        array = create_v2_array(
            group,
            str(level),
            shape=shape,
            chunks=chunks,
            dtype=np.uint8,
            compressor=compressor,
            fill_value=0,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ARRAY_DIMENSIONS
        arrays.append(array)
    return arrays


def downsample_mean_3d(block: np.ndarray) -> np.ndarray:
    """Mean-pool a ZYX block by two and round to its input dtype."""

    block = np.asarray(block)
    output_shape = tuple((value + 1) // 2 for value in block.shape)
    total = np.zeros(output_shape, dtype=np.float64)
    count = np.zeros(output_shape, dtype=np.float64)
    for offsets in itertools.product((0, 1), repeat=3):
        sample = block[
            offsets[0] :: 2, offsets[1] :: 2, offsets[2] :: 2
        ]
        if sample.size:
            slices = tuple(slice(0, value) for value in sample.shape)
            total[slices] += sample
            count[slices] += 1.0
    return np.ascontiguousarray(np.rint(total / count).astype(block.dtype))


def scale_chunks_to_next_level(chunks, *, source_array, target_array) -> set[ChunkId]:
    """Map written source chunks to overlapping chunks at the next level."""

    result = set()
    source_shape = tuple(source_array.shape)
    source_chunks = tuple(source_array.chunks)
    target_shape = tuple(target_array.shape)
    target_chunks = tuple(target_array.chunks)
    for chunk in chunks:
        starts, stops = chunk_bounds(
            chunk,
            chunk_shape_zyx=source_chunks,
            array_shape_zyx=source_shape,
        )
        result.update(
            chunks_overlapping_box(
                tuple(value // 2 for value in starts),
                tuple((value + 1) // 2 for value in stops),
                chunk_shape_zyx=target_chunks,
                array_shape_zyx=target_shape,
            )
        )
    return result


def downsample_level_chunk(source, target, chunk: ChunkId) -> None:
    """Populate one target ZYX chunk from its corresponding source extent."""

    target_starts, target_stops = chunk_bounds(
        chunk,
        chunk_shape_zyx=tuple(target.chunks),
        array_shape_zyx=tuple(target.shape),
    )
    source_starts = tuple(value * 2 for value in target_starts)
    source_stops = tuple(
        min(source.shape[axis], target_stops[axis] * 2) for axis in range(3)
    )
    source_block = np.asarray(
        source[
            source_starts[0] : source_stops[0],
            source_starts[1] : source_stops[1],
            source_starts[2] : source_stops[2],
        ]
    )
    output = downsample_mean_3d(source_block)
    target[
        target_starts[0] : target_starts[0] + output.shape[0],
        target_starts[1] : target_starts[1] + output.shape[1],
        target_starts[2] : target_starts[2] + output.shape[2],
    ] = output


def build_downsample_levels(
    arrays, *, level0_written_chunks: set[ChunkId], workers: int = 1
) -> None:
    """Populate all lower pyramid levels reachable from written level-0 chunks."""

    current_chunks = set(level0_written_chunks)
    for level in range(1, len(arrays)):
        source, target = arrays[level - 1], arrays[level]
        target_chunks = sorted(
            scale_chunks_to_next_level(
                current_chunks, source_array=source, target_array=target
            )
        )
        if workers <= 1 or len(target_chunks) <= 1:
            for chunk in target_chunks:
                downsample_level_chunk(source, target, chunk)
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(target_chunks))) as executor:
                list(
                    executor.map(
                        lambda chunk: downsample_level_chunk(source, target, chunk),
                        target_chunks,
                    )
                )
        current_chunks = set(target_chunks)


class NativePatchDataset(Dataset):
    """Lazy, spawn-safe reader for planned native ZYX patches."""

    def __init__(
        self,
        *,
        tifxyz_dir: Path,
        volume_path: str,
        resolution: str,
        patches: Sequence[PatchSpec],
        patch_size_zyx: tuple[int, int, int],
        config,
        cache_dir=None,
        cache_max_gb=None,
    ) -> None:
        self.tifxyz_dir = Path(tifxyz_dir)
        self.volume_path = str(volume_path)
        self.resolution = str(resolution)
        self.patches = tuple(patches)
        self.patch_size_zyx = tuple(int(value) for value in patch_size_zyx)
        self.mode = str(config.data.mode)
        self.normalization = config.data.normalization
        self.cache_dir = cache_dir
        self.cache_max_gb = cache_max_gb
        (
            self.flat_grid_stride,
            self.native_coordinate_scale,
            self.coarse_native_pad,
        ) = native_tifxyz_pyramid_params(int(self.resolution))
        self._volume = None
        self._tifxyz = None
        self._coarse_positions = None
        self._coarse_valid = None

    def __len__(self) -> int:
        return len(self.patches)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_volume"] = None
        state["_tifxyz"] = None
        state["_coarse_positions"] = None
        state["_coarse_valid"] = None
        return state

    def _ensure_volume(self):
        if self._volume is None:
            self._volume = _open_shared_volume(
                self.volume_path,
                self.resolution,
                cache_dir=self.cache_dir,
                cache_max_gb=self.cache_max_gb,
            )
        return self._volume

    def _ensure_tifxyz(self):
        if self._tifxyz is None:
            self._tifxyz = tifxyz.read_tifxyz(
                self.tifxyz_dir,
                load_mask=False,
                validate=False,
            )
            self._tifxyz.use_full_resolution()
        return self._tifxyz

    def _ensure_coarse_positions(self):
        if self._coarse_positions is None or self._coarse_valid is None:
            positions = np.asarray(
                self._ensure_tifxyz().get_zyxs(stored_resolution=True),
                dtype=np.float32,
            )
            valid = np.isfinite(positions).all(axis=-1)
            valid &= (positions >= 0).all(axis=-1)
            self._coarse_positions = positions
            self._coarse_valid = valid
        return self._coarse_positions, self._coarse_valid

    def _surface_mask(self, bbox_zyx) -> np.ndarray:
        coarse_positions, coarse_valid = self._ensure_coarse_positions()
        selection = select_flat_pixels_via_stored_resolution(
            self._ensure_tifxyz(),
            bbox_zyx,
            coarse_native_pad=self.coarse_native_pad,
            coarse_positions_zyx=coarse_positions,
            coarse_valid=coarse_valid,
            native_coordinate_scale=self.native_coordinate_scale,
            flat_grid_stride=self.flat_grid_stride,
            required=False,
        )
        if selection is None:
            return np.zeros(self.patch_size_zyx, dtype=np.float32)
        _, positions, valid = selection
        return project_surface_distance(
            positions,
            valid,
            bbox_zyx,
            max_distance_voxels=(
                SURFACE_MASK_MAX_DISTANCE_LEVEL0_VOXELS
                / float(self.flat_grid_stride)
            ),
        )

    def __getitem__(self, index: int):
        patch = self.patches[int(index)]
        starts = (int(patch.z0), int(patch.y0), int(patch.x0))
        stops = tuple(
            starts[axis] + self.patch_size_zyx[axis] for axis in range(3)
        )
        bbox = (*starts, *stops)
        crop, valid_slices = read_bbox_with_padding(
            self._ensure_volume(), bbox, fill_value=0
        )
        crop = crop.astype(np.float32, copy=False)
        if valid_slices is not None:
            crop[valid_slices] = normalize_image(
                crop[valid_slices], self.normalization
            )
        channels = [crop]
        if self.mode == "full_3d_single_wrap":
            channels.append(self._surface_mask(bbox))
        image = torch.from_numpy(
            np.ascontiguousarray(np.stack(channels, axis=0))
        ).float()
        return image, torch.tensor(starts, dtype=torch.int64)


def select_native_inference_weights(
    payload: Mapping[str, Any], *, source: str | Path = "<memory>"
) -> tuple[str, Mapping[str, Any]]:
    """Choose native inference weights using EMA-then-model priority."""

    for name in ("ema_model", "model"):
        state = payload.get(name)
        if isinstance(state, Mapping):
            return name, state
    raise ValueError(f"Checkpoint {str(source)!r} is missing model weights")


def _load_native_model(payload, config, args) -> NativeModelBundle:
    """Build a native target model from checkpoint configuration and weights."""

    _, state = select_native_inference_weights(
        payload, source=args.checkpoint
    )
    base_model = make_model(config)
    load_model_state(base_model, state)
    model = TargetModel(base_model.eval()).eval()
    model, device = prepare_model_for_inference(
        model,
        gpu_ids=args.gpu_ids,
        compile_model=bool(args.compile_model),
        compile_mode=str(args.compile_mode),
    )
    return NativeModelBundle(
        model=model,
        device=device,
        amp_dtype=resolve_amp_dtype(args.amp_dtype, payload, args.checkpoint),
    )


def run_native_inference(
    *,
    args,
    bundle: NativeModelBundle,
    dataset,
    plan: NativePlan,
    output,
    importance_map: np.ndarray,
) -> None:
    """Execute native patch loading, prediction, and chunk accumulation."""

    effective_batch_size = int(args.batch_size) * max(1, len(args.gpu_ids))
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": effective_batch_size,
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": bundle.device.type == "cuda",
        "drop_last": False,
    }
    if int(args.num_workers) > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=int(args.prefetch_factor),
            multiprocessing_context="spawn",
        )
    loader = DataLoader(**loader_kwargs)
    accumulator = ChunkAccumulator3D(
        output=output,
        target_chunks=set(plan.target_chunks),
        contribution_counts=plan.contribution_counts,
        chunk_shape_zyx=plan.chunk_shape_zyx,
    )
    autocast = (
        torch.autocast(
            device_type="cuda", enabled=True, dtype=bundle.amp_dtype
        )
        if bundle.device.type == "cuda"
        else nullcontext()
    )
    variants = tta_variants(bool(args.tta))
    with torch.inference_mode(), autocast:
        for images, metadata in loader:
            images = images.to(bundle.device, non_blocking=True)
            probabilities = predict_batch(
                bundle.model,
                images,
                variants=variants,
                tta_batch_size=args.tta_batch_size,
                patch_size_zyx=plan.patch_size_zyx,
            )
            probability_values = probabilities[:, 0].cpu().numpy()
            metadata_values = metadata.cpu().numpy()
            for index in range(probability_values.shape[0]):
                accumulator.add_patch(
                    patch_start_zyx=tuple(
                        int(value) for value in metadata_values[index]
                    ),
                    probabilities=probability_values[index],
                    weights=importance_map,
                )
    accumulator.flush_remaining()


def _open_shared_volume(
    path: str,
    resolution: str,
    *,
    cache_dir,
    cache_max_gb,
):
    return open_volume(
        path,
        int(resolution),
        cache_dir=cache_dir,
        cache_max_gb=cache_max_gb,
        root_array_is_requested_level=True,
    )


def _load_checkpoint_config(checkpoint_path: Path):
    payload = load_checkpoint(checkpoint_path)
    return payload, config_from_checkpoint(payload, source=checkpoint_path)


def _plan_from_args(args):
    payload, config = _load_checkpoint_config(args.checkpoint)
    if config.data.mode not in {"full_3d", "full_3d_single_wrap"}:
        raise ValueError(
            "infer_full3d_tifxyz requires mode 'full_3d' or "
            f"'full_3d_single_wrap', got {config.data.mode!r}"
        )
    downsample_factor = native_volume_downsample_factor(int(args.resolution))
    volume_path = read_volume_source(args.tifxyz_dir)
    volume = _open_shared_volume(
        volume_path,
        args.resolution,
        cache_dir=args.cache_dir,
        cache_max_gb=args.cache_max_gb,
    )
    shape = tuple(int(value) for value in volume.shape)
    chunks = tuple(int(value) for value in (volume.chunks or volume.shape))
    if len(shape) != 3 or len(chunks) != 3:
        raise ValueError(f"Expected a 3D ZYX input array, got shape={shape}, chunks={chunks}")
    scale = 1.0 / float(downsample_factor)
    x, y, z, valid = read_tifxyz_points(
        args.tifxyz_dir, native_coordinate_scale=scale
    )
    plan = build_native_plan(
        volume_path=volume_path,
        array_shape_zyx=shape,
        chunk_shape_zyx=chunks,
        x=x,
        y=y,
        z=z,
        valid=valid,
        patch_size_zyx=config.model.crop_size,
        overlap=args.overlap,
        chunk_halo=args.chunk_halo,
        write_region=args.write_region,
        max_target_chunks=args.max_target_chunks,
    )
    return plan, payload, config


def summarize_plan(plan: NativePlan) -> None:
    """Log the volume, chunk, stride, and inference-patch plan counts."""

    LOGGER.info("Input volume: %s", plan.volume_path)
    LOGGER.info("Input shape=%s chunks=%s", plan.array_shape_zyx, plan.chunk_shape_zyx)
    LOGGER.info(
        "Patch size=%s stride=%s occupied_chunks=%d expanded_chunks=%d "
        "target_chunks=%d inference_patches=%d",
        plan.patch_size_zyx,
        plan.stride_zyx,
        len(plan.occupied_chunks),
        len(plan.expanded_chunks),
        len(plan.target_chunks),
        len(plan.patches),
    )


def run_command(args) -> int:
    """Plan occupied work, then execute model and output lifecycles."""

    plan, payload, config = _plan_from_args(args)
    summarize_plan(plan)
    if args.plan_only:
        return 0
    bundle = _load_native_model(payload, config, args)
    arrays = create_output_zarr(
        args.output_zarr,
        shape_zyx=plan.array_shape_zyx,
        chunks_zyx=plan.chunk_shape_zyx,
        overwrite=args.overwrite,
    )
    dataset = NativePatchDataset(
        tifxyz_dir=args.tifxyz_dir,
        volume_path=plan.volume_path,
        resolution=args.resolution,
        patches=plan.patches,
        patch_size_zyx=plan.patch_size_zyx,
        config=config,
        cache_dir=args.cache_dir,
        cache_max_gb=args.cache_max_gb,
    )
    run_native_inference(
        args=args,
        bundle=bundle,
        dataset=dataset,
        plan=plan,
        output=arrays[0],
        importance_map=create_importance_map(
            plan.patch_size_zyx, mode=args.blend_mode
        ),
    )
    build_downsample_levels(
        arrays,
        level0_written_chunks=set(plan.target_chunks),
        workers=args.downsample_workers,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the native full-3D inference command."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    return run_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
