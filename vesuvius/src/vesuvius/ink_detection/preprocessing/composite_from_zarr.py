#!/usr/bin/env python3
"""Project Zarr volumes to uint8 TIFF composites without loading full arrays.

Two input modes are supported:

* ``--zarr PATH``: project one Zarr array or OME-Zarr group and write the TIFF
  beside it (or to ``--output``).
* ``--input-root DIR``: recursively scan for folders that contain exactly one
  ``.zarr`` directory and write one TIFF beside each source volume.

The selected array (OME-Zarr level ``0`` by default) is optionally filtered in
3D (currently a median; see ``VolumeFilter``), reduced along Z with ``min``,
``max`` or ``mean``, optionally resized, and optionally filtered in 2D
(``ImageFilter``, currently a median). Work is partitioned over a chunk-aligned Y/X tile
grid; each worker reads one tile's Z slab plus the halo the filters and any
resize need (in Z as well as Y/X, so tiles are seamless), filters, projects,
resizes, and the tiles are streamed into a tiled BigTIFF, so only a bounded
number of tiles is resident at a time.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import threading
from collections import deque
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import tifffile
import zarr
from tqdm import tqdm

from vesuvius.ink_detection.preprocessing.staged_write import (
    create_staged_output,
    discard_staged_output,
    publish_staged_output,
)
from vesuvius.utils.cli import HyphenUnderscoreParser
from vesuvius.utils.median3d import median_filter_3d
from vesuvius.utils.zarr_v2_direct import DirectZarrV2Reader, UnsupportedZarrArray

DEFAULT_TILE_SIZE = 1024
METHODS = ("min", "max", "mean")
READERS = ("auto", "direct", "zarr")
INTERPOLATIONS = ("auto", "area", "linear", "nearest", "cubic", "lanczos4")

_THREAD_LOCAL = threading.local()
_WORKER_CONFIG: "WorkerConfig | None" = None


@dataclass(frozen=True, slots=True)
class VolumeFilter:
    """A 3D filter applied to the Z slab before projection.

    ``halo`` is the number of extra voxels needed on every side for the output to be
    exact at tile borders; filters are applied in order and their halos add up.
    Add new stages by subclassing (keep them picklable for process workers) and
    appending to ``_build_volume_filters``.
    """

    halo: int

    @property
    def suffix(self) -> str:
        raise NotImplementedError

    def apply(self, block: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Median3D(VolumeFilter):
    """Exact (2*radius+1)^3 median with replicated borders at the volume edge."""

    radius: int

    def __init__(self, radius: int) -> None:
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "halo", radius)

    @property
    def suffix(self) -> str:
        return f"median{self.radius}"

    def apply(self, block: np.ndarray) -> np.ndarray:
        return median_filter_3d(block, self.radius)


def _build_volume_filters(*, median_radius: int = 0) -> tuple[VolumeFilter, ...]:
    """Pre-projection filter chain in application order."""

    filters: list[VolumeFilter] = []
    if median_radius > 0:
        filters.append(Median3D(median_radius))
    return tuple(filters)


def _apply_volume_filters(block: np.ndarray, filters: Sequence[VolumeFilter]) -> np.ndarray:
    for stage in filters:
        block = stage.apply(block)
    return block


@dataclass(frozen=True, slots=True)
class ImageFilter:
    """A 2D filter applied to the uint8 composite after projection and resizing.

    ``halo`` is in output pixels. Stages run in order; add new ones by subclassing
    (picklable) and appending in ``_build_image_filters``.
    """

    halo: int

    @property
    def suffix(self) -> str:
        raise NotImplementedError

    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Median2D(ImageFilter):
    """OpenCV (2*radius+1)^2 median with replicated borders at the image edge."""

    radius: int

    def __init__(self, radius: int) -> None:
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "halo", radius)

    @property
    def suffix(self) -> str:
        return f"median2d{self.radius}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.medianBlur(image, 2 * self.radius + 1)


def _build_image_filters(*, median_2d_radius: int = 0) -> tuple[ImageFilter, ...]:
    """Post-composite 2D filter chain in application order."""

    filters: list[ImageFilter] = []
    if median_2d_radius > 0:
        filters.append(Median2D(median_2d_radius))
    return tuple(filters)


def _apply_image_filters(image: np.ndarray, filters: Sequence[ImageFilter]) -> np.ndarray:
    for stage in filters:
        image = stage.apply(image)
    return image


@dataclass(frozen=True, slots=True)
class ArraySpec:
    zarr_path: Path
    array_key: str | None
    shape: tuple[int, int, int]
    chunks: tuple[int, int, int]
    leading_axis_index: int | None = None


@dataclass(frozen=True, slots=True)
class DatasetJob:
    folder: Path
    zarr_path: Path
    output_path: Path
    array_key: str | None
    shape: tuple[int, int, int]
    chunks: tuple[int, int, int]
    leading_axis_index: int | None
    start_z: int
    end_z: int
    geometry: "TileGeometry"
    method: str
    parallelism: str
    workers: int
    no_progress: bool
    compression: str | None
    filters: tuple[VolumeFilter, ...] = ()
    reader: str = "auto"
    interpolation: str = "auto"
    image_filters: tuple[ImageFilter, ...] = ()


@dataclass(frozen=True, slots=True)
class TileGeometry:
    """Input/output tiling for one job.

    ``in_tile`` is the input-space edge of a full tile and ``out_tile`` its edge
    after scaling; both are integers because ``in_tile`` is a multiple of the
    scale's denominator (and of the chunk size, so tiles read whole chunks).
    ``tiff_tile`` is ``out_tile`` rounded up to the TIFF-required multiple of 16
    (only differs when a single tile covers the whole axis).
    """

    scale: Fraction
    in_shape: tuple[int, int]
    out_shape: tuple[int, int]
    in_tile: tuple[int, int]
    out_tile: tuple[int, int]
    tiff_tile: tuple[int, int]


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    zarr_path: str
    array_key: str | None
    leading_axis_index: int | None
    shape: tuple[int, int, int]
    start_z: int
    end_z: int
    method: str
    filters: tuple[VolumeFilter, ...]
    reader: str
    scale: Fraction
    halo: int
    interpolation: str
    image_filters: tuple[ImageFilter, ...]
    out_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ChunkTask:
    """One output tile ``[out_y_start, out_y_stop) x [out_x_start, out_x_stop)``.

    ``y_start``/``x_start`` are the exact input coordinates of the tile origin;
    ``y_stop``/``x_stop`` are the nominal input extent (``ceil(out_stop / scale)``),
    which can exceed the array for the last tile when the scaled size is not an
    integer and is clipped and edge-padded when read.
    """

    y_start: int
    y_stop: int
    x_start: int
    x_stop: int
    out_y_start: int
    out_y_stop: int
    out_x_start: int
    out_x_stop: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = HyphenUnderscoreParser(
        description="Create chunked uint8 TIFF projections from Zarr volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Declared as plain options (not a mutually exclusive group) so the parser's
    # hyphen/underscore aliasing applies; exclusivity is checked below.
    parser.add_argument(
        "--input-root",
        default=None,
        help="Root directory to scan recursively for folders containing a .zarr directory. "
        "Exactly one of --input-root or --zarr is required.",
    )
    parser.add_argument(
        "--zarr",
        default=None,
        help="Path to a single Zarr array or OME-Zarr group to project.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output TIFF path (only with --zarr). Defaults to a name derived from the "
        "Zarr, method, median radius, level and Z range, written beside the Zarr.",
    )
    parser.add_argument(
        "--start_z",
        type=int,
        default=None,
        help="Start Z index (inclusive, default 0). Negative values follow Python slice semantics.",
    )
    parser.add_argument(
        "--end_z",
        type=int,
        default=None,
        help="End Z index (exclusive, default: depth). Negative values follow Python slice semantics.",
    )
    parser.add_argument(
        "--center-layers",
        type=int,
        default=None,
        help="Project only the middle N layers of the selected level (an odd remainder leaves one "
        "extra layer below the centre). Cannot be combined with --start_z/--end_z.",
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        required=True,
        help="Projection method to apply along the Z axis.",
    )
    parser.add_argument(
        "--median-radius",
        type=int,
        default=0,
        help="Radius of a 3D median filter applied to the volume before projection "
        "(kernel (2*r+1)^3, replicated borders at the volume edge). 0 disables it.",
    )
    parser.add_argument(
        "--median-2d-radius",
        type=int,
        default=0,
        help="Radius of a 2D median filter applied to the composite after projection and "
        "any --output-scale resize (kernel (2*r+1)^2). 0 disables it.",
    )
    parser.add_argument(
        "--resolution",
        default="0",
        help="OME-Zarr resolution level key to read when the input is a group.",
    )
    parser.add_argument(
        "--output-scale",
        type=_parse_scale,
        default=Fraction(1),
        help="Resize factor for the output image, e.g. 0.25 for quarter size or 2 for double. "
        "Integers, decimals and fractions like 1/3 are accepted.",
    )
    parser.add_argument(
        "--resize-interpolation",
        choices=INTERPOLATIONS,
        default="auto",
        help="OpenCV interpolation for --output-scale. 'auto' uses area for downscaling and "
        "linear for upscaling.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Requested tile edge in input pixels (the TIFF tile is this times --output-scale). "
        "Rounded up so every tile reads whole Zarr chunks and scales to a multiple of 16; "
        "larger tiles amortise per-tile overhead and the median/resize halo.",
    )
    parser.add_argument(
        "--reader",
        choices=READERS,
        default="auto",
        help="How workers read chunks. 'direct' decodes local Zarr v2 chunk files in the "
        "worker (scales with --workers; zarr-python 3 serialises reads on one event "
        "loop); 'zarr' uses zarr-python; 'auto' picks direct when supported.",
    )
    parser.add_argument(
        "--parallelism",
        choices=("threads", "processes"),
        default="threads",
        help="Execution backend for chunk workers. Threads are usually better for compressed Zarr I/O.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
        help="Number of chunk workers to use per volume.",
    )
    parser.add_argument(
        "--compression",
        default="zlib",
        help="Compression passed to tifffile. Use 'none' to disable compression.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output TIFFs.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args(argv)

    if (args.input_root is None) == (args.zarr is None):
        parser.error("Exactly one of --input-root or --zarr is required.")
    if args.center_layers is not None:
        if args.center_layers <= 0:
            parser.error("--center-layers must be at least 1.")
        if args.start_z is not None or args.end_z is not None:
            parser.error("--center-layers cannot be combined with --start_z or --end_z.")
    if args.workers <= 0:
        parser.error("--workers must be at least 1.")
    if args.median_radius < 0:
        parser.error("--median-radius must be non-negative.")
    if args.median_2d_radius < 0:
        parser.error("--median-2d-radius must be non-negative.")
    if args.tile_size <= 0:
        parser.error("--tile-size must be at least 1.")
    if args.output is not None and args.zarr is None:
        parser.error("--output requires --zarr.")

    if isinstance(args.compression, str) and args.compression.lower() == "none":
        args.compression = None

    return args


def _resolve_interpolation(scale: Fraction, interpolation: str) -> str:
    """Map 'auto' to the OpenCV mode actually used: area to shrink, linear to enlarge."""

    if interpolation == "auto":
        return "area" if scale < 1 else "linear"
    return interpolation


def _parse_scale(text: str) -> Fraction:
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(f"invalid scale {text!r}: {exc}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"scale must be positive, got {text!r}")
    return value


def _discover_zarr_dirs(root: Path) -> list[Path]:
    stack = [root]
    zarr_dirs: list[Path] = []

    while stack:
        current = stack.pop()
        if current.suffix.lower() == ".zarr" and current.is_dir():
            zarr_dirs.append(current)
            continue
        if not current.is_dir():
            continue
        children = sorted(current.iterdir(), reverse=True)
        stack.extend(children)

    zarr_dirs.sort()
    return zarr_dirs


def _resolve_array_spec(zarr_path: Path, resolution: str) -> ArraySpec:
    root = zarr.open(str(zarr_path), mode="r")

    if isinstance(root, zarr.Array):
        array = root
        array_key: str | None = None
    elif isinstance(root, zarr.Group):
        if resolution not in root:
            available = sorted(str(key) for key in root.array_keys())
            raise ValueError(
                f"Resolution '{resolution}' not found in {zarr_path}. Available arrays: {available}"
            )
        array = root[resolution]
        array_key = resolution
    else:
        raise ValueError(f"Unsupported Zarr root type at {zarr_path}: {type(root)!r}")

    if array.ndim == 3:
        shape = tuple(int(v) for v in array.shape)
        chunks = tuple(int(v) for v in array.chunks)
        return ArraySpec(
            zarr_path=zarr_path,
            array_key=array_key,
            shape=shape,
            chunks=chunks,
        )

    if array.ndim == 4 and array.shape[0] == 1:
        shape = tuple(int(v) for v in array.shape[1:])
        chunks = tuple(int(v) for v in array.chunks[1:])
        return ArraySpec(
            zarr_path=zarr_path,
            array_key=array_key,
            shape=shape,
            chunks=chunks,
            leading_axis_index=0,
        )

    raise ValueError(
        f"Expected a 3D array or a 4D array with a leading singleton axis in {zarr_path}, "
        f"got shape {array.shape}."
    )


def _resolve_z_range(
    total: int,
    start_z: int | None,
    end_z: int | None,
    center_layers: int | None = None,
) -> tuple[int, int]:
    if center_layers is not None:
        if center_layers > total:
            raise ValueError(f"--center-layers {center_layers} exceeds the depth {total} of the selected level")
        start_idx = (total - center_layers) // 2
        return start_idx, start_idx + center_layers

    start_idx = 0 if start_z is None else start_z
    end_idx = total if end_z is None else end_z

    if start_idx < 0:
        start_idx += total
    if end_idx < 0:
        end_idx += total

    start_idx = max(0, min(total, start_idx))
    end_idx = max(0, min(total, end_idx))

    if start_idx >= end_idx:
        raise ValueError(f"Empty Z range after applying start/end to depth {total}: [{start_idx}, {end_idx})")

    return start_idx, end_idx


def _resolve_tile_geometry(
    shape: tuple[int, int, int],
    chunks: tuple[int, int, int],
    tile_size: int,
    scale: Fraction = Fraction(1),
) -> TileGeometry:
    """Pick tiles that read whole Zarr chunks and scale to TIFF tiles (multiples of 16).

    Per axis the input tile is the smallest multiple of ``lcm(chunk, M)`` at or above
    ``tile_size`` where ``M`` makes ``tile * scale`` an integer multiple of 16. A tile
    that would exceed the image covers the whole axis instead.
    """

    num, den = scale.numerator, scale.denominator
    in_shape = (shape[1], shape[2])
    out_dims: list[int] = []
    in_tiles: list[int] = []
    out_tiles: list[int] = []
    tiff_tiles: list[int] = []
    for image_dim, chunk_dim in zip(in_shape, chunks[1:], strict=True):
        out_dim = max(1, -(-image_dim * num // den))
        align = 16 * den // math.gcd(num, 16 * den)
        step = math.lcm(max(1, chunk_dim), align)
        in_tile = -(-tile_size // step) * step
        if in_tile >= image_dim:
            in_tile = image_dim
            out_tile = out_dim
            tiff_tile = -(-out_dim // 16) * 16
        else:
            out_tile = in_tile * num // den
            tiff_tile = out_tile
        out_dims.append(out_dim)
        in_tiles.append(in_tile)
        out_tiles.append(out_tile)
        tiff_tiles.append(tiff_tile)
    return TileGeometry(
        scale=scale,
        in_shape=in_shape,
        out_shape=(out_dims[0], out_dims[1]),
        in_tile=(in_tiles[0], in_tiles[1]),
        out_tile=(out_tiles[0], out_tiles[1]),
        tiff_tile=(tiff_tiles[0], tiff_tiles[1]),
    )


def _filter_halo(filters: Sequence[VolumeFilter]) -> int:
    return sum(stage.halo for stage in filters)


def _image_filter_halo(filters: Sequence[ImageFilter]) -> int:
    return sum(stage.halo for stage in filters)


def _resolve_halo(
    filters: Sequence[VolumeFilter],
    scale: Fraction,
    interpolation: str,
    image_filters: Sequence[ImageFilter] = (),
) -> int:
    """Input-space Y/X halo (multiple of the scale denominator) for all stages.

    A 2D filter needs ``halo`` correct output pixels beyond the tile, each of which
    needs ``1/scale`` input pixels plus the resize halo, plus the 3D filter halo.
    """

    image_halo = _image_filter_halo(image_filters)
    median_radius = _filter_halo(filters) + -(-image_halo * scale.denominator // scale.numerator)

    resize_halo = 0
    if scale != 1:
        # area/linear need ceil(1/scale)+1 source pixels beyond the tile; cubic and
        # lanczos4 reach 2 and 4 further. Add slack rather than special-case.
        resize_halo = -(-scale.denominator // scale.numerator) + 6
    den = scale.denominator
    return -(-(median_radius + resize_halo) // den) * den


def _output_name(
    stem: str,
    *,
    method: str,
    filters: Sequence[VolumeFilter],
    array_key: str | None,
    start_z: int,
    end_z: int,
    scale: Fraction = Fraction(1),
    image_filters: Sequence[ImageFilter] = (),
    interpolation: str = "auto",
) -> str:
    parts = [stem, method]
    parts.extend(stage.suffix for stage in filters)
    if scale != 1:
        parts.append(f"scale{float(scale):g}")
        parts.append(_resolve_interpolation(scale, interpolation))
    parts.extend(stage.suffix for stage in image_filters)
    if array_key is not None and array_key != "0":
        parts.append(f"level{array_key}")
    parts.extend((str(start_z), str(end_z)))
    return "_".join(parts) + ".tif"


def _array_dir(zarr_path: Path, array_key: str | None) -> Path:
    return zarr_path if array_key is None else zarr_path / array_key


def _resolve_reader(spec: ArraySpec, reader: str) -> str:
    """Map 'auto' to 'direct' when the array is a supported local Zarr v2 array."""

    if reader == "zarr":
        return "zarr"
    try:
        DirectZarrV2Reader.open(_array_dir(spec.zarr_path, spec.array_key))
    except UnsupportedZarrArray:
        if reader == "direct":
            raise
        return "zarr"
    return "direct"


def _make_job(
    *,
    folder: Path,
    zarr_path: Path,
    output_path: Path | None,
    name_stem: str,
    start_z: int | None,
    end_z: int | None,
    method: str,
    resolution: str,
    overwrite: bool,
    parallelism: str,
    workers: int,
    no_progress: bool,
    compression: str | None,
    median_radius: int,
    tile_size: int,
    reader: str,
    center_layers: int | None = None,
    output_scale: Fraction = Fraction(1),
    interpolation: str = "auto",
    median_2d_radius: int = 0,
) -> DatasetJob:
    spec = _resolve_array_spec(zarr_path, resolution)
    reader = _resolve_reader(spec, reader)
    z_start, z_end = _resolve_z_range(spec.shape[0], start_z, end_z, center_layers)
    filters = _build_volume_filters(median_radius=median_radius)
    image_filters = _build_image_filters(median_2d_radius=median_2d_radius)
    if output_path is None:
        output_path = folder / _output_name(
            name_stem,
            method=method,
            filters=filters,
            array_key=spec.array_key,
            start_z=z_start,
            end_z=z_end,
            scale=output_scale,
            image_filters=image_filters,
            interpolation=interpolation,
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    return DatasetJob(
        folder=folder,
        zarr_path=zarr_path,
        output_path=output_path,
        array_key=spec.array_key,
        shape=spec.shape,
        chunks=spec.chunks,
        leading_axis_index=spec.leading_axis_index,
        start_z=z_start,
        end_z=z_end,
        geometry=_resolve_tile_geometry(spec.shape, spec.chunks, tile_size, output_scale),
        method=method,
        parallelism=parallelism,
        workers=workers,
        no_progress=no_progress,
        compression=compression,
        filters=filters,
        reader=reader,
        interpolation=interpolation,
        image_filters=image_filters,
    )


def _build_jobs(
    *,
    input_root: Path,
    start_z: int | None,
    end_z: int | None,
    method: str,
    resolution: str,
    overwrite: bool,
    parallelism: str,
    workers: int,
    no_progress: bool,
    compression: str | None,
    median_radius: int = 0,
    tile_size: int = DEFAULT_TILE_SIZE,
    reader: str = "auto",
    center_layers: int | None = None,
    output_scale: Fraction = Fraction(1),
    interpolation: str = "auto",
    median_2d_radius: int = 0,
) -> list[DatasetJob]:
    zarr_dirs = _discover_zarr_dirs(input_root)
    if not zarr_dirs:
        raise FileNotFoundError(f"No .zarr directories found under {input_root}")

    by_parent: dict[Path, list[Path]] = {}
    for zarr_dir in zarr_dirs:
        by_parent.setdefault(zarr_dir.parent, []).append(zarr_dir)

    jobs: list[DatasetJob] = []
    for folder in sorted(by_parent):
        zarr_paths = sorted(by_parent[folder])
        if len(zarr_paths) != 1:
            names = ", ".join(path.name for path in zarr_paths)
            raise ValueError(
                f"Folder {folder} contains multiple .zarr directories: {names}. "
                "Use --zarr to project one of them."
            )

        jobs.append(
            _make_job(
                folder=folder,
                zarr_path=zarr_paths[0],
                output_path=None,
                name_stem=folder.name,
                start_z=start_z,
                end_z=end_z,
                method=method,
                resolution=resolution,
                overwrite=overwrite,
                parallelism=parallelism,
                workers=workers,
                no_progress=no_progress,
                compression=compression,
                median_radius=median_radius,
                tile_size=tile_size,
                reader=reader,
                center_layers=center_layers,
                output_scale=output_scale,
                interpolation=interpolation,
                median_2d_radius=median_2d_radius,
            )
        )

    return jobs


def _build_single_job(
    *,
    zarr_path: Path,
    output: Path | None,
    start_z: int | None,
    end_z: int | None,
    method: str,
    resolution: str,
    overwrite: bool,
    parallelism: str,
    workers: int,
    no_progress: bool,
    compression: str | None,
    median_radius: int = 0,
    tile_size: int = DEFAULT_TILE_SIZE,
    reader: str = "auto",
    center_layers: int | None = None,
    output_scale: Fraction = Fraction(1),
    interpolation: str = "auto",
    median_2d_radius: int = 0,
) -> DatasetJob:
    if not zarr_path.is_dir():
        raise NotADirectoryError(f"Zarr path is not a directory: {zarr_path}")
    return _make_job(
        folder=zarr_path.parent,
        zarr_path=zarr_path,
        output_path=output,
        name_stem=zarr_path.name.removesuffix(".zarr"),
        start_z=start_z,
        end_z=end_z,
        method=method,
        resolution=resolution,
        overwrite=overwrite,
        parallelism=parallelism,
        workers=workers,
        no_progress=no_progress,
        compression=compression,
        median_radius=median_radius,
        tile_size=tile_size,
        reader=reader,
        center_layers=center_layers,
        output_scale=output_scale,
        interpolation=interpolation,
        median_2d_radius=median_2d_radius,
    )


def _iter_chunk_tasks(geometry: TileGeometry) -> list[ChunkTask]:
    num, den = geometry.scale.numerator, geometry.scale.denominator
    out_h, out_w = geometry.out_shape
    tile_y, tile_x = geometry.out_tile
    tasks: list[ChunkTask] = []

    def to_input(out_start: int, out_stop: int) -> tuple[int, int]:
        # out_start is a multiple of out_tile = in_tile * scale, so this is exact.
        return out_start * den // num, -(-out_stop * den // num)

    for oy0 in range(0, out_h, tile_y):
        oy1 = min(out_h, oy0 + tile_y)
        iy0, iy1 = to_input(oy0, oy1)
        for ox0 in range(0, out_w, tile_x):
            ox1 = min(out_w, ox0 + tile_x)
            ix0, ix1 = to_input(ox0, ox1)
            tasks.append(
                ChunkTask(
                    y_start=iy0,
                    y_stop=iy1,
                    x_start=ix0,
                    x_stop=ix1,
                    out_y_start=oy0,
                    out_y_stop=oy1,
                    out_x_start=ox0,
                    out_x_stop=ox1,
                )
            )

    return tasks


def _set_worker_config(config: WorkerConfig) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    _THREAD_LOCAL.array = None
    _THREAD_LOCAL.key = None
    if config.scale != 1 or config.image_filters:
        # Tiles are already processed in parallel; keep OpenCV single-threaded per call.
        import cv2

        cv2.setNumThreads(1)


def _get_worker_array():
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker configuration has not been initialized.")

    cache_key = (
        _WORKER_CONFIG.zarr_path,
        _WORKER_CONFIG.array_key,
    )
    if getattr(_THREAD_LOCAL, "key", None) != cache_key:
        if _WORKER_CONFIG.reader == "direct":
            array = DirectZarrV2Reader.open(
                _array_dir(Path(_WORKER_CONFIG.zarr_path), _WORKER_CONFIG.array_key)
            )
        else:
            root = zarr.open(_WORKER_CONFIG.zarr_path, mode="r")
            array = root if isinstance(root, zarr.Array) else root[_WORKER_CONFIG.array_key]
        _THREAD_LOCAL.array = array
        _THREAD_LOCAL.key = cache_key

    return _THREAD_LOCAL.array


def _read_block(array, ranges: Sequence[tuple[int, int]]) -> np.ndarray:
    """Read ``array[r0, r1, ...]`` for (start, stop) ranges via either reader."""

    if isinstance(array, DirectZarrV2Reader):
        return array.read(ranges)
    return np.asarray(array[tuple(slice(start, stop) for start, stop in ranges)])


def _project_block(block: np.ndarray, method: str) -> np.ndarray:
    if method == "max":
        return np.max(block, axis=0)

    if method == "min":
        return np.min(block, axis=0)

    if method == "mean":
        if np.issubdtype(block.dtype, np.integer) or np.issubdtype(block.dtype, np.bool_):
            # Exact integer sums; a uint32 accumulator is exact for uint8/bool inputs of
            # fewer than 2**24 slices and avoids the wider reduction.
            depth = block.shape[0]
            small = block.dtype == np.uint8 or block.dtype == np.bool_
            acc_dtype = np.uint32 if small and depth * 255 < 2**32 else np.uint64
            sums = np.add.reduce(block, axis=0, dtype=acc_dtype)
            return sums.astype(np.float32) / float(depth)
        return np.mean(block, axis=0, dtype=np.float32)

    raise ValueError(f"Unsupported method: {method}")


def _to_uint8(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        return np.ascontiguousarray(data)

    if np.issubdtype(data.dtype, np.floating):
        rounded = np.rint(data)
        np.clip(rounded, 0, 255, out=rounded)
        return np.ascontiguousarray(rounded.astype(np.uint8, copy=False))

    clipped = np.clip(data, 0, 255)
    return np.ascontiguousarray(clipped.astype(np.uint8, copy=False))


def _resize(image: np.ndarray, scale: Fraction, interpolation: str) -> np.ndarray:
    """Resize by exactly ``scale``; the image extent times ``scale`` must be integral."""

    import cv2

    height, width = image.shape
    out_h = height * scale
    out_w = width * scale
    assert out_h.denominator == 1 and out_w.denominator == 1, (height, width, scale)
    interpolation = _resolve_interpolation(scale, interpolation)
    flag = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }[interpolation]
    return cv2.resize(image, (int(out_w), int(out_h)), interpolation=flag)


def _project_chunk(task: ChunkTask) -> np.ndarray:
    array = _get_worker_array()
    config = _WORKER_CONFIG
    if config is None:
        raise RuntimeError("Worker configuration has not been initialized.")

    scale = config.scale
    den = scale.denominator
    depth, height, width = config.shape
    # Read a halo so the 3D filters (and resize) see true neighbours across tile
    # borders and, in Z, real layers beyond the selected range; at the volume edge
    # the halo is clipped and the filters replicate the border. With scaling the
    # Y/X region is kept a multiple of the scale denominator so the resized region
    # is an exact integer size.
    halo = config.halo
    z_halo = _filter_halo(config.filters)
    z0 = max(0, config.start_z - z_halo)
    z1 = min(depth, config.end_z + z_halo)
    ry0 = task.y_start - halo
    ry1 = -(-task.y_stop // den) * den + halo
    rx0 = task.x_start - halo
    rx1 = -(-task.x_stop // den) * den + halo
    y0, y1 = max(0, ry0), min(height, ry1)
    x0, x1 = max(0, rx0), min(width, rx1)

    ranges: list[tuple[int, int]] = [(z0, z1), (y0, y1), (x0, x1)]
    if config.leading_axis_index is not None:
        ranges.insert(0, (config.leading_axis_index, config.leading_axis_index + 1))
    block = _read_block(array, ranges)
    if config.leading_axis_index is not None:
        block = block[0]

    if config.filters:
        block = _apply_volume_filters(block, config.filters)
        if z0 != config.start_z or z1 != config.end_z:
            block = block[config.start_z - z0 : config.end_z - z0]

    projected = _to_uint8(_project_block(block, config.method))
    del block

    if scale != 1:
        # Replicate-pad the clipped halo back to the nominal region so the resize sees
        # the same border values a whole-image resize would clamp to.
        pad = ((y0 - ry0, ry1 - y1), (x0 - rx0, rx1 - x1))
        if any(v for pair in pad for v in pair):
            projected = np.pad(projected, pad, mode="edge")
        projected = _resize(projected, scale, config.interpolation)
        # Clip the resized region to the output image so the 2D filters see the true
        # image border (replicated by OpenCV) rather than the padded halo.
        region_y0, region_x0 = int(ry0 * scale), int(rx0 * scale)
        cy0 = max(0, region_y0)
        cx0 = max(0, region_x0)
        cy1 = min(config.out_shape[0], int(ry1 * scale))
        cx1 = min(config.out_shape[1], int(rx1 * scale))
        if (cy0, cx0) != (region_y0, region_x0) or projected.shape != (cy1 - region_y0, cx1 - region_x0):
            projected = np.ascontiguousarray(
                projected[cy0 - region_y0 : cy1 - region_y0, cx0 - region_x0 : cx1 - region_x0]
            )
        oy = task.out_y_start - cy0
        ox = task.out_x_start - cx0
    else:
        oy = task.y_start - y0
        ox = task.x_start - x0

    if config.image_filters:
        projected = _apply_image_filters(projected, config.image_filters)

    out_h = task.out_y_stop - task.out_y_start
    out_w = task.out_x_stop - task.out_x_start
    if projected.shape != (out_h, out_w):
        projected = np.ascontiguousarray(projected[oy : oy + out_h, ox : ox + out_w])

    # Edge tiles are smaller than the tile shape; tifffile zero-pads them.
    return projected


def _iter_projected_tiles(
    executor: Executor,
    tasks: Sequence[ChunkTask],
    *,
    max_in_flight: int,
    progress: tqdm,
) -> Iterator[np.ndarray]:
    """Yield projected tiles in task order with at most ``max_in_flight`` outstanding.

    ``Executor.map`` submits every task up front and buffers completed results
    without bound; this sliding window keeps memory proportional to the worker
    count while keeping the pool busy ahead of the TIFF writer.
    """

    pending: deque[Future[np.ndarray]] = deque()
    task_iter = iter(tasks)
    try:
        for task in itertools.islice(task_iter, max_in_flight):
            pending.append(executor.submit(_project_chunk, task))
        while pending:
            future = pending.popleft()
            tile = future.result()
            for task in itertools.islice(task_iter, 1):
                pending.append(executor.submit(_project_chunk, task))
            progress.update(1)
            yield tile
    finally:
        for future in pending:
            future.cancel()
        progress.close()


def _write_projection(job: DatasetJob) -> None:
    worker_config = WorkerConfig(
        zarr_path=str(job.zarr_path),
        array_key=job.array_key,
        leading_axis_index=job.leading_axis_index,
        shape=job.shape,
        start_z=job.start_z,
        end_z=job.end_z,
        method=job.method,
        filters=job.filters,
        reader=job.reader,
        scale=job.geometry.scale,
        halo=_resolve_halo(job.filters, job.geometry.scale, job.interpolation, job.image_filters),
        image_filters=job.image_filters,
        out_shape=job.geometry.out_shape,
        interpolation=job.interpolation,
    )
    tasks = _iter_chunk_tasks(job.geometry)

    if job.parallelism == "threads":
        _set_worker_config(worker_config)
        executor_cls = ThreadPoolExecutor
        executor_kwargs = {"max_workers": job.workers}
    else:
        executor_cls = ProcessPoolExecutor
        executor_kwargs = {
            "max_workers": job.workers,
            "initializer": _set_worker_config,
            "initargs": (worker_config,),
        }

    staged_output = create_staged_output(job.output_path)
    try:
        with executor_cls(**executor_kwargs) as executor:
            progress = tqdm(
                total=len(tasks),
                desc=job.output_path.name,
                unit="tile",
                disable=job.no_progress,
                leave=False,
            )
            tiles = _iter_projected_tiles(
                executor,
                tasks,
                max_in_flight=job.workers * 4,
                progress=progress,
            )
            try:
                with tifffile.TiffWriter(str(staged_output), bigtiff=True) as writer:
                    writer.write(
                        data=tiles,
                        shape=job.geometry.out_shape,
                        dtype=np.uint8,
                        tile=job.geometry.tiff_tile,
                        photometric="minisblack",
                        compression=job.compression,
                        # Let tifffile compress tiles in parallel instead of serially
                        # on the writer thread.
                        maxworkers=job.workers if job.compression else 1,
                    )
            finally:
                tiles.close()
        publish_staged_output(staged_output, job.output_path)
    finally:
        discard_staged_output(staged_output)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    common = {
        "start_z": args.start_z,
        "end_z": args.end_z,
        "method": args.method,
        "resolution": args.resolution,
        "overwrite": args.overwrite,
        "parallelism": args.parallelism,
        "workers": args.workers,
        "no_progress": args.no_progress,
        "compression": args.compression,
        "median_radius": args.median_radius,
        "tile_size": args.tile_size,
        "reader": args.reader,
        "center_layers": args.center_layers,
        "output_scale": args.output_scale,
        "interpolation": args.resize_interpolation,
        "median_2d_radius": args.median_2d_radius,
    }

    if args.zarr is not None:
        output = None if args.output is None else Path(args.output)
        jobs = [_build_single_job(zarr_path=Path(args.zarr), output=output, **common)]
    else:
        input_root = Path(args.input_root)
        if not input_root.is_dir():
            raise NotADirectoryError(f"Input root is not a directory: {input_root}")
        jobs = _build_jobs(input_root=input_root, **common)

    job_iter: Sequence[DatasetJob] | tqdm[DatasetJob]
    job_iter = jobs
    if not args.no_progress and len(jobs) > 1:
        job_iter = tqdm(jobs, desc="Volumes")

    wrote = 0
    for job in job_iter:
        _write_projection(job)
        wrote += 1

    print(f"Wrote {wrote} TIFF projection(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
