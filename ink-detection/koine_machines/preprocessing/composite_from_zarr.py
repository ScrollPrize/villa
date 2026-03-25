#!/usr/bin/env python3
"""Project Zarr volumes to uint8 TIFF composites without loading full arrays.

The script recursively scans an input root for folders that contain exactly one
`.zarr` directory, resolves the selected Zarr array (OME-Zarr level `0` by
default), and writes a single 2D TIFF projection beside each source volume.

Projection work is partitioned over the Y/X chunk grid, so only one projected
tile per worker is resident in memory at a time.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import tifffile
import zarr
from tqdm import tqdm

# Allow direct execution via `python path/to/composite_zarr_projections.py`
# without requiring an editable install or preconfigured PYTHONPATH.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


_THREAD_LOCAL = threading.local()
_WORKER_CONFIG: "WorkerConfig | None" = None


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
    tile_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    zarr_path: str
    array_key: str | None
    leading_axis_index: int | None
    start_z: int
    end_z: int
    method: str
    tile_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ChunkTask:
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create chunked uint8 TIFF projections from Zarr volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory to scan recursively for folders containing a .zarr directory.",
    )
    parser.add_argument(
        "--start_z",
        type=int,
        default=0,
        help="Start Z index (inclusive). Negative values follow Python slice semantics.",
    )
    parser.add_argument(
        "--end_z",
        type=int,
        default=None,
        help="End Z index (exclusive). Negative values follow Python slice semantics.",
    )
    parser.add_argument(
        "--method",
        choices=("max", "mean"),
        required=True,
        help="Projection method to apply along the Z axis.",
    )
    parser.add_argument(
        "--resolution",
        default="0",
        help="OME-Zarr resolution level key to read when the input is a group.",
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
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be at least 1.")

    if isinstance(args.compression, str) and args.compression.lower() == "none":
        args.compression = None

    return args


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


def _resolve_z_range(total: int, start_z: int | None, end_z: int | None) -> tuple[int, int]:
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


def _validate_tile_shape(chunks: tuple[int, int, int]) -> tuple[int, int]:
    tile_shape = (chunks[1], chunks[2])
    if tile_shape[0] < 16 or tile_shape[1] < 16:
        raise ValueError(f"Chunk-derived tile shape {tile_shape} is too small for tiled TIFF output.")
    if tile_shape[0] % 16 != 0 or tile_shape[1] % 16 != 0:
        raise ValueError(
            f"Chunk-derived tile shape {tile_shape} is not valid for TIFF tiling. "
            "Chunk sizes must be multiples of 16."
        )
    return tile_shape


def _build_jobs(
    *,
    input_root: Path,
    start_z: int | None,
    end_z: int | None,
    method: str,
    resolution: str,
    overwrite: bool,
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
            raise ValueError(f"Folder {folder} contains multiple .zarr directories: {names}")

        zarr_path = zarr_paths[0]
        spec = _resolve_array_spec(zarr_path, resolution)
        z_start, z_end = _resolve_z_range(spec.shape[0], start_z, end_z)
        output_path = folder / f"{folder.name}_{method}_{z_start}_{z_end}.tif"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")

        jobs.append(
            DatasetJob(
                folder=folder,
                zarr_path=zarr_path,
                output_path=output_path,
                array_key=spec.array_key,
                shape=spec.shape,
                chunks=spec.chunks,
                leading_axis_index=spec.leading_axis_index,
                start_z=z_start,
                end_z=z_end,
                tile_shape=_validate_tile_shape(spec.chunks),
            )
        )

    return jobs


def _iter_chunk_tasks(shape: tuple[int, int, int], chunks: tuple[int, int, int]) -> list[ChunkTask]:
    _, height, width = shape
    _, chunk_y, chunk_x = chunks
    tasks: list[ChunkTask] = []

    for y_start in range(0, height, chunk_y):
        y_stop = min(height, y_start + chunk_y)
        for x_start in range(0, width, chunk_x):
            x_stop = min(width, x_start + chunk_x)
            tasks.append(
                ChunkTask(
                    y_start=y_start,
                    y_stop=y_stop,
                    x_start=x_start,
                    x_stop=x_stop,
                )
            )

    return tasks


def _set_worker_config(config: WorkerConfig) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    _THREAD_LOCAL.array = None
    _THREAD_LOCAL.key = None


def _get_worker_array():
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker configuration has not been initialized.")

    cache_key = (
        _WORKER_CONFIG.zarr_path,
        _WORKER_CONFIG.array_key,
    )
    if getattr(_THREAD_LOCAL, "key", None) != cache_key:
        root = zarr.open(_WORKER_CONFIG.zarr_path, mode="r")
        array = root if isinstance(root, zarr.Array) else root[_WORKER_CONFIG.array_key]
        _THREAD_LOCAL.array = array
        _THREAD_LOCAL.key = cache_key

    return _THREAD_LOCAL.array


def _project_block(block: np.ndarray, method: str) -> np.ndarray:
    if method == "max":
        return np.max(block, axis=0)

    if method == "mean":
        if np.issubdtype(block.dtype, np.integer) or np.issubdtype(block.dtype, np.bool_):
            sums = np.add.reduce(block, axis=0, dtype=np.uint64)
            return sums.astype(np.float32) / float(block.shape[0])
        return np.mean(block, axis=0, dtype=np.float32)

    raise ValueError(f"Unsupported method: {method}")


def _to_uint8(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        return np.ascontiguousarray(data)

    if np.issubdtype(data.dtype, np.floating):
        clipped = np.clip(np.rint(data), 0, 255)
        return np.ascontiguousarray(clipped.astype(np.uint8, copy=False))

    clipped = np.clip(data, 0, 255)
    return np.ascontiguousarray(clipped.astype(np.uint8, copy=False))


def _project_chunk(task: ChunkTask) -> np.ndarray:
    array = _get_worker_array()
    config = _WORKER_CONFIG
    assert config is not None

    if config.leading_axis_index is None:
        block = np.asarray(
            array[
                config.start_z : config.end_z,
                task.y_start : task.y_stop,
                task.x_start : task.x_stop,
            ]
        )
    else:
        block = np.asarray(
            array[
                config.leading_axis_index,
                config.start_z : config.end_z,
                task.y_start : task.y_stop,
                task.x_start : task.x_stop,
            ]
        )

    projected = _to_uint8(_project_block(block, config.method))
    tile = np.zeros(config.tile_shape, dtype=np.uint8)
    tile[: projected.shape[0], : projected.shape[1]] = projected
    return tile


def _write_projection(job: DatasetJob, args: argparse.Namespace) -> None:
    worker_config = WorkerConfig(
        zarr_path=str(job.zarr_path),
        array_key=job.array_key,
        leading_axis_index=job.leading_axis_index,
        start_z=job.start_z,
        end_z=job.end_z,
        method=args.method,
        tile_shape=job.tile_shape,
    )
    tasks = _iter_chunk_tasks(job.shape, job.chunks)

    if args.parallelism == "threads":
        _set_worker_config(worker_config)
        executor_cls = ThreadPoolExecutor
        executor_kwargs = {"max_workers": args.workers}
    else:
        executor_cls = ProcessPoolExecutor
        executor_kwargs = {
            "max_workers": args.workers,
            "initializer": _set_worker_config,
            "initargs": (worker_config,),
        }

    with executor_cls(**executor_kwargs) as executor:
        tile_iter = executor.map(_project_chunk, tasks, chunksize=1)

        progress = tqdm(
            total=len(tasks),
            desc=job.folder.name,
            disable=args.no_progress,
            leave=False,
        )

        def _yield_tiles() -> Iterator[np.ndarray]:
            try:
                for tile in tile_iter:
                    progress.update(1)
                    yield tile
            finally:
                progress.close()

        with tifffile.TiffWriter(str(job.output_path), bigtiff=True) as writer:
            writer.write(
                data=_yield_tiles(),
                shape=(job.shape[1], job.shape[2]),
                dtype=np.uint8,
                tile=job.tile_shape,
                photometric="minisblack",
                compression=args.compression,
                maxworkers=1,
            )


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    if not input_root.is_dir():
        print(f"Input root is not a directory: {input_root}", file=sys.stderr)
        return 1

    try:
        jobs = _build_jobs(
            input_root=input_root,
            start_z=args.start_z,
            end_z=args.end_z,
            method=args.method,
            resolution=args.resolution,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    job_iter: Sequence[DatasetJob] | tqdm[DatasetJob]
    job_iter = jobs
    if not args.no_progress:
        job_iter = tqdm(jobs, desc="Volumes")

    wrote = 0
    for job in job_iter:
        _write_projection(job, args)
        wrote += 1

    print(f"Wrote {wrote} TIFF projection(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
