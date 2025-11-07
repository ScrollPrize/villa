#!/usr/bin/env python3
"""Stack or composite a range of TIFF slices into large 3D or 2D outputs.

The script streams slices into either a NumPy memmap file or a temporary Zarr
store so we can handle volumes that do not fit in memory. After the streaming
load finishes, the data are written out either as a 3D stack or a projection
computed over the Z dimension.
"""

from __future__ import annotations

import argparse
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import tifffile
import zarr

from vesuvius.image_proc.run.workflow import gather_inputs, run_workflow

_STORAGE: np.memmap | zarr.Array | None = None
_EXPECTED_SHAPE: tuple[int, int] | None = None
_EXPECTED_DTYPE: np.dtype | None = None


@dataclass(slots=True)
class StorageSpec:
    mode: str
    path: Path
    shape: tuple[int, int, int]
    dtype: np.dtype
    array_path: str | None = None
    sync_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack or composite a subset of TIFF slices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Directory containing TIFF slices.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination TIFF file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive) within the sorted slice list.",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop index (exclusive) within the sorted slice list.",
    )
    parser.add_argument(
        "--mode",
        choices=("stack", "composite"),
        default="stack",
        help="Whether to write a 3D stack or a composite projection.",
    )
    parser.add_argument(
        "--projection",
        choices=("min", "max", "mean", "median"),
        default="mean",
        help="Projection to compute when --mode is composite.",
    )
    parser.add_argument(
        "--storage",
        choices=("memmap", "zarr"),
        default="memmap",
        help="Backing storage used while streaming slices from disk.",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Directory for temporary storage (defaults to the output directory).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes used while loading slices.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Chunk size forwarded to the worker pool.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=512,
        help="Row chunk size used when computing median/mean projections.",
    )
    parser.add_argument(
        "--chunk-cols",
        type=int,
        default=512,
        help="Column chunk size used when computing median/mean projections.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zlib",
        help="Compression argument forwarded to tifffile.imwrite ('' disables).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for TIFFs inside the input folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )

    args = parser.parse_args()

    if args.num_workers <= 0:
        parser.error("--num-workers must be at least 1.")

    if args.chunksize <= 0:
        parser.error("--chunksize must be at least 1.")

    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be at least 1.")

    if args.chunk_cols <= 0:
        parser.error("--chunk-cols must be at least 1.")

    if args.compression == "":
        args.compression = None

    if args.mode == "stack" and args.projection:
        # Projection is ignored for stack mode but we keep the argument optional.
        args.projection = None

    return args


def _resolve_slice_range(
    total: int,
    start: int | None,
    stop: int | None,
) -> tuple[int, int]:
    start_idx = 0 if start is None else start
    stop_idx = total if stop is None else stop

    if start_idx < 0:
        start_idx = total + start_idx
    if stop_idx < 0:
        stop_idx = total + stop_idx

    start_idx = max(0, start_idx)
    stop_idx = min(total, stop_idx)

    if start_idx >= stop_idx:
        raise ValueError("Slice range is empty after applying start/stop.")

    return start_idx, stop_idx


def _inspect_slice_metadata(path: Path) -> tuple[tuple[int, int], np.dtype]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.ndim != 2:
            raise ValueError(f"Slice {path} expected to be 2D, got shape {series.shape}.")
        return tuple(series.shape), np.dtype(series.dtype)


def _prepare_storage(
    *,
    storage_mode: str,
    shape: tuple[int, int, int],
    dtype: np.dtype,
    base_dir: Path,
) -> StorageSpec:
    base_dir.mkdir(parents=True, exist_ok=True)

    if storage_mode == "memmap":
        temporary_path = base_dir / f"stack_{uuid.uuid4().hex}.dat"
        memmap = np.memmap(temporary_path, dtype=dtype, mode="w+", shape=shape)
        memmap.flush()
        del memmap
        return StorageSpec(
            mode="memmap",
            path=temporary_path,
            shape=shape,
            dtype=dtype,
        )

    if storage_mode == "zarr":
        store_dir = base_dir / f"stack_{uuid.uuid4().hex}.zarr"
        store_dir.mkdir(parents=False, exist_ok=False)
        sync_path = store_dir / ".sync"
        store = zarr.DirectoryStore(str(store_dir))
        synchronizer = zarr.ProcessSynchronizer(str(sync_path))
        group = zarr.open_group(store=store, mode="w", synchronizer=synchronizer)
        group.create_dataset(
            "stack",
            shape=shape,
            dtype=dtype,
            chunks=(1,) + shape[1:],
            overwrite=True,
        )
        return StorageSpec(
            mode="zarr",
            path=store_dir,
            shape=shape,
            dtype=dtype,
            array_path="stack",
            sync_path=sync_path,
        )

    raise ValueError(f"Unsupported storage mode: {storage_mode}")


def _initialize_storage(spec: StorageSpec) -> None:
    global _STORAGE, _EXPECTED_SHAPE, _EXPECTED_DTYPE
    _EXPECTED_SHAPE = spec.shape[1:]
    _EXPECTED_DTYPE = np.dtype(spec.dtype)

    if spec.mode == "memmap":
        _STORAGE = np.memmap(spec.path, dtype=_EXPECTED_DTYPE, mode="r+", shape=spec.shape)
        return

    if spec.mode == "zarr":
        assert spec.array_path is not None and spec.sync_path is not None
        store = zarr.DirectoryStore(str(spec.path))
        synchronizer = zarr.ProcessSynchronizer(str(spec.sync_path))
        _STORAGE = zarr.open(store=store, path=spec.array_path, mode="r+", synchronizer=synchronizer)
        return

    raise ValueError(f"Unsupported storage mode: {spec.mode}")


def _stack_worker(item: tuple[int, Path]) -> Path:
    if _STORAGE is None or _EXPECTED_SHAPE is None or _EXPECTED_DTYPE is None:
        raise RuntimeError("Worker storage not initialized.")

    index, slice_path = item
    data = tifffile.imread(slice_path)
    if data.shape != _EXPECTED_SHAPE:
        raise ValueError(f"Slice {slice_path} shape {data.shape} does not match expected {_EXPECTED_SHAPE}.")
    if data.dtype != _EXPECTED_DTYPE:
        data = data.astype(_EXPECTED_DTYPE, copy=False)

    _STORAGE[index, ...] = data
    return slice_path


def _open_storage_for_read(spec: StorageSpec):
    if spec.mode == "memmap":
        array = np.memmap(spec.path, dtype=spec.dtype, mode="r", shape=spec.shape)

        def cleanup() -> None:
            try:
                array._mmap.close()
            except AttributeError:
                pass
            finally:
                try:
                    Path(spec.path).unlink()
                except FileNotFoundError:
                    pass

        return array, cleanup

    if spec.mode == "zarr":
        assert spec.array_path is not None and spec.sync_path is not None
        store = zarr.DirectoryStore(str(spec.path))
        synchronizer = zarr.ProcessSynchronizer(str(spec.sync_path))
        array = zarr.open(store=store, path=spec.array_path, mode="r", synchronizer=synchronizer)

        def cleanup() -> None:
            try:
                array.store.flush()
            except AttributeError:
                pass
            shutil.rmtree(spec.path, ignore_errors=True)

        return array, cleanup

    raise ValueError(f"Unsupported storage mode: {spec.mode}")


def _determine_projection_dtype(projection: str, input_dtype: np.dtype) -> np.dtype:
    if projection in ("min", "max"):
        return input_dtype

    if np.issubdtype(input_dtype, np.floating):
        return input_dtype

    return np.float32


def _composite_min(array: Iterable[np.ndarray], dtype: np.dtype) -> np.ndarray:
    result = None
    for data in array:
        if result is None:
            result = np.array(data, copy=True)
            continue
        np.minimum(result, data, out=result)
    if result is None:
        raise RuntimeError("No slices provided for min projection.")
    return result.astype(dtype, copy=False)


def _composite_max(array: Iterable[np.ndarray], dtype: np.dtype) -> np.ndarray:
    result = None
    for data in array:
        if result is None:
            result = np.array(data, copy=True)
            continue
        np.maximum(result, data, out=result)
    if result is None:
        raise RuntimeError("No slices provided for max projection.")
    return result.astype(dtype, copy=False)


def _composite_mean(
    array: Iterable[np.ndarray],
    spatial_shape: tuple[int, int],
    count: int,
    dtype: np.dtype,
) -> np.ndarray:
    accumulator = np.zeros(spatial_shape, dtype=np.float64)
    for data in array:
        accumulator += data.astype(np.float64, copy=False)
    mean = accumulator / float(count)
    if np.issubdtype(dtype, np.floating):
        return mean.astype(dtype, copy=False)
    return np.rint(mean, out=mean).astype(dtype, copy=False)


def _iter_slices(array: np.ndarray | zarr.Array) -> Iterator[np.ndarray]:
    for index in range(array.shape[0]):
        yield np.asarray(array[index])


def _composite_median(
    array: np.ndarray | zarr.Array,
    spatial_shape: tuple[int, int],
    chunk_rows: int,
    chunk_cols: int,
    dtype: np.dtype,
) -> np.ndarray:
    height, width = spatial_shape
    result = np.empty(spatial_shape, dtype=dtype)

    row_step = min(chunk_rows, height)
    col_step = min(chunk_cols, width)

    for row_start in range(0, height, row_step):
        row_stop = min(height, row_start + row_step)
        for col_start in range(0, width, col_step):
            col_stop = min(width, col_start + col_step)
            block = np.asarray(array[:, row_start:row_stop, col_start:col_stop])
            median_block = np.median(block, axis=0)
            result[row_start:row_stop, col_start:col_stop] = median_block.astype(dtype, copy=False)

    return result


def _write_output_stack(
    array: np.ndarray | zarr.Array,
    output_path: Path,
    *,
    compression: str | None,
) -> None:
    tifffile.imwrite(output_path, np.asarray(array), compression=compression)


def _write_output_projection(
    array: np.ndarray | zarr.Array,
    *,
    projection: str,
    output_path: Path,
    compression: str | None,
    chunk_rows: int,
    chunk_cols: int,
) -> None:
    spatial_shape = array.shape[1:]
    input_dtype = np.dtype(array.dtype)

    if projection == "min":
        data = _composite_min(_iter_slices(array), input_dtype)
    elif projection == "max":
        data = _composite_max(_iter_slices(array), input_dtype)
    elif projection == "mean":
        output_dtype = _determine_projection_dtype("mean", input_dtype)
        data = _composite_mean(
            _iter_slices(array),
            spatial_shape,
            array.shape[0],
            output_dtype,
        )
    elif projection == "median":
        output_dtype = _determine_projection_dtype("median", input_dtype)
        data = _composite_median(
            array,
            spatial_shape,
            chunk_rows,
            chunk_cols,
            output_dtype,
        )
    else:
        raise ValueError(f"Unsupported projection: {projection}")

    tifffile.imwrite(output_path, data, compression=compression)


def main() -> int:
    args = parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"Refusing to overwrite existing output: {output_path}")
        return 1

    input_folder = Path(args.input_folder)
    slices = gather_inputs(
        input_folder,
        image_extensions=(".tif", ".tiff"),
        include_zarr=False,
        recursive=args.recursive,
    )

    if not slices:
        print(f"No TIFF files found in {input_folder}")
        return 1

    try:
        start_idx, stop_idx = _resolve_slice_range(len(slices), args.start, args.stop)
    except ValueError as error:
        print(error)
        return 1

    selected = slices[start_idx:stop_idx]
    if not selected:
        print("No slices selected after applying start/stop.")
        return 1

    spatial_shape, dtype = _inspect_slice_metadata(selected[0])
    total_slices = len(selected)
    storage_shape = (total_slices, *spatial_shape)

    temp_dir = Path(args.temp_dir) if args.temp_dir else output_path.parent
    storage_spec = _prepare_storage(
        storage_mode=args.storage,
        shape=storage_shape,
        dtype=dtype,
        base_dir=temp_dir,
    )

    items = list(enumerate(selected))
    run_workflow(
        worker_fn=_stack_worker,
        inputs=items,
        num_workers=args.num_workers,
        progress_desc="Loading slices",
        chunksize=args.chunksize,
        initializer=_initialize_storage,
        initargs=(storage_spec,),
        show_progress=not args.no_progress,
    )

    storage_array, cleanup = _open_storage_for_read(storage_spec)
    try:
        if args.mode == "stack":
            _write_output_stack(storage_array, output_path, compression=args.compression)
        else:
            assert args.projection is not None
            _write_output_projection(
                storage_array,
                projection=args.projection,
                output_path=output_path,
                compression=args.compression,
                chunk_rows=args.chunk_rows,
                chunk_cols=args.chunk_cols,
            )
    finally:
        cleanup()

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
