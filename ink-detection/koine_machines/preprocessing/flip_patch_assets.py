#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from math import ceil, isclose, prod
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import numcodecs.blosc as blosc
from scipy import ndimage
import tifffile
import zarr
from numcodecs import Blosc
from tqdm.auto import tqdm


_DEFAULT_WORKERS = max(1, min(32, os.cpu_count() or 1))
_DEFAULT_BLOSC_THREADS = 1
_CACHE_SIZE = 16
DEFAULT_ZARR_COMPRESSOR = Blosc(cname="zstd", clevel=2)
_BACKUP_DIRNAME = "backup"

_THREAD_LOCAL = threading.local()


@dataclass(frozen=True, slots=True)
class TransformConfig:
    horizontal: bool
    vertical: bool
    rotation_degrees_ccw: float
    label: str

    @property
    def has_rotation(self) -> bool:
        return not isclose(self.rotation_degrees_ccw % 360.0, 0.0, abs_tol=1e-9)

    @property
    def quarter_turns(self) -> int | None:
        if not self.has_rotation:
            return None
        normalized = self.rotation_degrees_ccw % 360.0
        turns = normalized / 90.0
        rounded_turns = round(turns)
        if isclose(turns, rounded_turns, abs_tol=1e-9):
            return rounded_turns % 4
        return None


@dataclass(frozen=True, slots=True)
class Task:
    src: Path
    dst: Path
    kind: str
    is_label: bool


@dataclass(frozen=True, slots=True)
class TaskPlan:
    task: Task
    write_dst: Path
    final_dst: Path
    backup_dst: Path | None


@dataclass(slots=True)
class _TiffContext:
    path: Path
    tif: tifffile.TiffFile
    page: tifffile.TiffPage
    tile_cache: OrderedDict[int, np.ndarray]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flip or rotate matched .zarr and _max_22_42 TIFF assets inside a "
            "patch folder using chunk/tile streaming where possible."
        )
    )
    parser.add_argument(
        "patch_dir",
        type=Path,
        help="Patch directory to scan recursively.",
    )
    transform_group = parser.add_mutually_exclusive_group(required=True)
    transform_group.add_argument(
        "--flip",
        choices=("horizontal", "vertical", "both"),
        help="Flip direction.",
    )
    transform_group.add_argument(
        "--rotate",
        type=float,
        help="Rotate by this many degrees, from 0 through 360 inclusive.",
    )
    parser.add_argument(
        "--rotate-dir",
        choices=("clockwise", "counterclockwise"),
        help="Rotation direction to use with --rotate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without writing outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help=f"Thread workers for chunk/tile jobs. Default: {_DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--blosc-threads",
        type=int,
        default=_DEFAULT_BLOSC_THREADS,
        help=(
            "Process-wide Blosc codec threads to use during Zarr reads/writes. "
            f"Default: {_DEFAULT_BLOSC_THREADS}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output, temp, or backup path.",
    )
    parser.add_argument(
        "--flip-labels",
        action="store_true",
        help="Also transform *_inklabels and *_supervision_mask TIFF/Zarr assets.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help=(
            "Replace each original asset after writing a temporary transformed copy. "
            "Original label assets are moved under patch_dir/backup."
        ),
    )
    args = parser.parse_args(argv)

    if args.workers <= 0:
        parser.error("--workers must be at least 1")
    if args.blosc_threads <= 0:
        parser.error("--blosc-threads must be at least 1")

    if args.rotate is None and args.rotate_dir is not None:
        parser.error("--rotate-dir requires --rotate")
    if args.rotate is not None:
        if args.rotate_dir is None:
            parser.error("--rotate requires --rotate-dir")
        if not (0.0 <= args.rotate <= 360.0):
            parser.error("--rotate must be between 0 and 360 inclusive")

    return args


def _format_rotation_value(value: float) -> str:
    text = format(value, "g")
    return text.replace(".", "p")


def build_transform_config(args: argparse.Namespace) -> TransformConfig:
    if args.flip is not None:
        return build_flip_config(args.flip)

    assert args.rotate is not None
    assert args.rotate_dir is not None

    rotation_degrees_ccw = (
        args.rotate if args.rotate_dir == "counterclockwise" else -args.rotate
    )
    return TransformConfig(
        horizontal=False,
        vertical=False,
        rotation_degrees_ccw=rotation_degrees_ccw,
        label=f"rotate_{_format_rotation_value(args.rotate)}_{args.rotate_dir}",
    )


def build_flip_config(value: str) -> TransformConfig:
    if value == "horizontal":
        return TransformConfig(
            horizontal=True,
            vertical=False,
            rotation_degrees_ccw=0.0,
            label=value,
        )
    if value == "vertical":
        return TransformConfig(
            horizontal=False,
            vertical=True,
            rotation_degrees_ccw=0.0,
            label=value,
        )
    return TransformConfig(
        horizontal=True,
        vertical=True,
        rotation_degrees_ccw=0.0,
        label=value,
    )


def _is_label_stem(stem_lower: str) -> bool:
    return stem_lower.endswith("_inklabels") or stem_lower.endswith("_supervision_mask")


def _is_selected_zarr(
    dirname: str,
    parent_dir_name: str,
    *,
    include_labels: bool,
) -> bool:
    if not dirname.lower().endswith(".zarr"):
        return False
    stem = Path(dirname).stem
    stem_lower = stem.lower()
    parent_lower = parent_dir_name.lower()
    if stem_lower == parent_lower or stem_lower.endswith("_max_22_42"):
        return True
    if include_labels and _is_label_stem(stem_lower):
        return True
    return False


def _is_selected_tiff(filename: str, *, include_labels: bool) -> bool:
    low = filename.lower()
    if not (low.endswith(".tif") or low.endswith(".tiff")):
        return False
    stem = Path(filename).stem.lower()
    if stem.endswith("_max_22_42") or stem in {"x", "y", "z"}:
        return True
    if include_labels and _is_label_stem(stem):
        return True
    return False


def discover_tasks(patch_dir: Path, flip_label: str, *, include_labels: bool) -> list[Task]:
    tasks: list[Task] = []
    for current_root, dirnames, filenames in os.walk(patch_dir):
        current = Path(current_root)
        child_dirnames = sorted(
            dirname
            for dirname in dirnames
            if not (current == patch_dir and dirname == _BACKUP_DIRNAME)
        )

        selected_zarr_dirs = [
            dirname
            for dirname in child_dirnames
            if _is_selected_zarr(dirname, current.name, include_labels=include_labels)
        ]
        for dirname in selected_zarr_dirs:
            src = current / dirname
            dst = src.with_name(f"{src.stem}_flip_{flip_label}.zarr")
            tasks.append(
                Task(
                    src=src,
                    dst=dst,
                    kind="zarr",
                    is_label=_is_label_stem(src.stem.lower()),
                )
            )

        for filename in sorted(filenames):
            if _is_selected_tiff(filename, include_labels=include_labels):
                src = current / filename
                dst = src.with_name(f"{src.stem}_flip_{flip_label}{src.suffix}")
                tasks.append(
                    Task(
                        src=src,
                        dst=dst,
                        kind="tiff",
                        is_label=_is_label_stem(src.stem.lower()),
                    )
                )

        dirnames[:] = sorted(
            dirname for dirname in child_dirnames if not dirname.lower().endswith(".zarr")
        )

    return tasks


def _inplace_temp_path(src: Path, flip_label: str, kind: str) -> Path:
    if kind == "zarr":
        return src.with_name(f"{src.stem}_flip_tmp_{flip_label}.zarr")
    return src.with_name(f"{src.stem}_flip_tmp_{flip_label}{src.suffix}")


def _label_backup_path(src: Path, patch_dir: Path) -> Path:
    return patch_dir / _BACKUP_DIRNAME / src.relative_to(patch_dir)


def build_task_plans(
    tasks: Sequence[Task],
    patch_dir: Path,
    flip_label: str,
    *,
    inplace: bool,
) -> list[TaskPlan]:
    plans: list[TaskPlan] = []
    for task in tasks:
        if not inplace:
            plans.append(
                TaskPlan(task=task, write_dst=task.dst, final_dst=task.dst, backup_dst=None)
            )
            continue

        plans.append(
            TaskPlan(
                task=task,
                write_dst=_inplace_temp_path(task.src, flip_label, task.kind),
                final_dst=task.src,
                backup_dst=_label_backup_path(task.src, patch_dir) if task.is_label else None,
            )
        )
    return plans


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _prepare_output_path(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    _remove_path(path)


def _complete_inplace_replace(plan: TaskPlan) -> None:
    src = plan.task.src
    if plan.backup_dst is not None:
        plan.backup_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(plan.backup_dst))
    else:
        _remove_path(src)
    plan.write_dst.rename(src)


def _describe_plan(plan: TaskPlan, *, inplace: bool) -> str:
    if not inplace:
        return f"{plan.task.src} -> {plan.final_dst}"
    if plan.backup_dst is None:
        return f"{plan.task.src} -> {plan.final_dst} (in-place via {plan.write_dst.name})"
    return (
        f"{plan.task.src} -> {plan.final_dst} "
        f"(in-place via {plan.write_dst.name}; backup {plan.backup_dst})"
    )


def _chunk_grid(shape: Sequence[int], chunks: Sequence[int]) -> Iterator[tuple[slice, ...]]:
    starts_per_dim = [
        list(range(0, int(size), int(chunk))) for size, chunk in zip(shape, chunks)
    ]
    for starts in product(*starts_per_dim):
        slices: list[slice] = []
        for axis, start in enumerate(starts):
            stop = min(int(shape[axis]), start + int(chunks[axis]))
            slices.append(slice(start, stop))
        yield tuple(slices)


def _count_chunks(shape: Sequence[int], chunks: Sequence[int]) -> int:
    total = 1
    for size, chunk in zip(shape, chunks):
        total *= ceil(int(size) / int(chunk))
    return total


def _copy_group_structure(
    src_group: zarr.Group,
    dst_group: zarr.Group,
    *,
    compressor: Blosc,
    transform: TransformConfig,
) -> None:
    dst_group.attrs.update(dict(src_group.attrs))

    for subgroup_name in sorted(src_group.group_keys()):
        src_subgroup = src_group[subgroup_name]
        dst_subgroup = dst_group.create_group(subgroup_name, overwrite=False)
        _copy_group_structure(
            src_subgroup,
            dst_subgroup,
            compressor=compressor,
            transform=transform,
        )

    for array_name in sorted(src_group.array_keys()):
        src_arr = src_group[array_name]
        dst_arr = dst_group.create_dataset(
            array_name,
            shape=_transformed_shape(src_arr.shape, transform),
            chunks=_transformed_chunks(src_arr.chunks, transform),
            dtype=src_arr.dtype,
            compressor=compressor,
            filters=src_arr.filters,
            fill_value=src_arr.fill_value,
            order=src_arr.order,
            write_empty_chunks=False,
            overwrite=False,
        )
        dst_arr.attrs.update(dict(src_arr.attrs))


def _iter_arrays(group: zarr.Group) -> Iterator[zarr.Array]:
    for subgroup_name in sorted(group.group_keys()):
        yield from _iter_arrays(group[subgroup_name])
    for array_name in sorted(group.array_keys()):
        yield group[array_name]


def _get_thread_zarr_cache() -> dict[tuple[str, str, str], zarr.Array]:
    cache: dict[tuple[str, str, str], zarr.Array] | None = getattr(
        _THREAD_LOCAL,
        "zarr_arrays",
        None,
    )
    if cache is None:
        cache = {}
        _THREAD_LOCAL.zarr_arrays = cache
    return cache


def _get_thread_zarr_array(store_path: Path, *, mode: str, array_path: str) -> zarr.Array:
    cache = _get_thread_zarr_cache()
    key = (str(store_path), mode, array_path)
    cached = cache.get(key)
    if cached is not None:
        return cached

    root = zarr.open(str(store_path), mode=mode)
    if isinstance(root, zarr.Array):
        arr = root
    else:
        arr = root[array_path]

    cache[key] = arr
    return arr


def _is_corrupt_zarr_chunk_error(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and "error during blosc decompression" in str(exc)


def _flip_axes_for_ndim(ndim: int, config: TransformConfig) -> tuple[int, ...]:
    axes: list[int] = []
    if config.vertical:
        axes.append(ndim - 2)
    if config.horizontal:
        axes.append(ndim - 1)
    return tuple(axes)


def _rotation_fill_value(dtype: np.dtype) -> int | float | bool:
    if np.issubdtype(dtype, np.bool_):
        return False
    if np.issubdtype(dtype, np.integer):
        return 0
    return 0.0


def _rotated_2d_shape(shape: Sequence[int], transform: TransformConfig) -> tuple[int, int]:
    height, width = int(shape[0]), int(shape[1])
    if not transform.has_rotation:
        return height, width

    quarter_turns = transform.quarter_turns
    if quarter_turns is not None:
        if quarter_turns % 2 == 1:
            return width, height
        return height, width

    sample = np.zeros((height, width), dtype=np.uint8)
    rotated = ndimage.rotate(
        sample,
        angle=transform.rotation_degrees_ccw,
        reshape=True,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return int(rotated.shape[0]), int(rotated.shape[1])


def _transformed_shape(shape: Sequence[int], transform: TransformConfig) -> tuple[int, ...]:
    if len(shape) < 2:
        return tuple(int(dim) for dim in shape)
    height, width = _rotated_2d_shape(shape[-2:], transform)
    return tuple(int(dim) for dim in shape[:-2]) + (height, width)


def _transformed_chunks(chunks: Sequence[int], transform: TransformConfig) -> tuple[int, ...]:
    if len(chunks) < 2:
        return tuple(int(chunk) for chunk in chunks)
    if transform.has_rotation and transform.quarter_turns is not None and transform.quarter_turns % 2 == 1:
        return tuple(int(chunk) for chunk in chunks[:-2]) + (
            int(chunks[-1]),
            int(chunks[-2]),
        )
    return tuple(int(chunk) for chunk in chunks)


def _transform_array(arr: np.ndarray, transform: TransformConfig) -> np.ndarray:
    out = np.asarray(arr)
    axes_to_flip = _flip_axes_for_ndim(out.ndim, transform)
    if axes_to_flip:
        out = np.flip(out, axis=axes_to_flip)

    if not transform.has_rotation:
        return np.ascontiguousarray(out)

    quarter_turns = transform.quarter_turns
    if quarter_turns is not None:
        if quarter_turns:
            out = np.rot90(out, k=quarter_turns, axes=(-2, -1))
        return np.ascontiguousarray(out)

    out = ndimage.rotate(
        out,
        angle=transform.rotation_degrees_ccw,
        axes=(-2, -1),
        reshape=True,
        order=0,
        mode="constant",
        cval=_rotation_fill_value(np.dtype(out.dtype)),
        prefilter=False,
    )
    return np.ascontiguousarray(out)


def _iter_plane_indices(shape: Sequence[int]) -> Iterator[tuple[int, ...]]:
    if not shape:
        yield ()
        return

    ranges = [range(int(dim)) for dim in shape]
    yield from product(*ranges)


def _quarter_turn_source_slices_2d(
    out_slices: tuple[slice, slice],
    shape: Sequence[int],
    quarter_turns: int,
) -> tuple[slice, slice]:
    height, width = int(shape[0]), int(shape[1])
    out_y, out_x = out_slices

    if quarter_turns == 0:
        return out_y, out_x
    if quarter_turns == 1:
        return out_x, slice(width - out_y.stop, width - out_y.start)
    if quarter_turns == 2:
        return (
            slice(height - out_y.stop, height - out_y.start),
            slice(width - out_x.stop, width - out_x.start),
        )
    if quarter_turns == 3:
        return slice(height - out_x.stop, height - out_x.start), out_y
    raise ValueError(f"Unsupported quarter_turns={quarter_turns}")


def _flipped_source_slices(
    out_slices: tuple[slice, ...],
    shape: Sequence[int],
    axes_to_flip: Iterable[int],
) -> tuple[slice, ...]:
    src_slices = list(out_slices)
    for axis in axes_to_flip:
        out_slice = out_slices[axis]
        src_slices[axis] = slice(int(shape[axis]) - out_slice.stop, int(shape[axis]) - out_slice.start)
    return tuple(src_slices)


@contextmanager
def _blosc_thread_limit(thread_count: int) -> Iterator[None]:
    previous = blosc.get_nthreads()
    if previous != thread_count:
        blosc.set_nthreads(thread_count)
    try:
        yield
    finally:
        if previous != thread_count:
            blosc.set_nthreads(previous)


def transform_zarr(
    src_path: Path,
    dst_path: Path,
    transform: TransformConfig,
    workers: int,
) -> None:
    src_obj = zarr.open(str(src_path), mode="r")
    target_compressor = DEFAULT_ZARR_COMPRESSOR

    if isinstance(src_obj, zarr.Array):
        dst_arr = zarr.open_array(
            store=zarr.DirectoryStore(str(dst_path)),
            mode="w",
            shape=_transformed_shape(src_obj.shape, transform),
            chunks=_transformed_chunks(src_obj.chunks, transform),
            dtype=src_obj.dtype,
            compressor=target_compressor,
            filters=src_obj.filters,
            fill_value=src_obj.fill_value,
            order=src_obj.order,
            write_empty_chunks=False,
        )
        dst_arr.attrs.update(dict(src_obj.attrs))
        array_pairs = [(src_obj, dst_arr, "", src_path.name)]
    else:
        dst_root = zarr.group(store=zarr.DirectoryStore(str(dst_path)), overwrite=False)
        _copy_group_structure(
            src_obj,
            dst_root,
            compressor=target_compressor,
            transform=transform,
        )
        array_pairs = []
        src_arrays = {array.path: array for array in _iter_arrays(src_obj)}
        dst_arrays = {array.path: array for array in _iter_arrays(dst_root)}
        for key in sorted(src_arrays):
            array_pairs.append((src_arrays[key], dst_arrays[key], key, key))

    for src_arr, dst_arr, array_path, label in array_pairs:
        if src_arr.ndim < 2:
            raise ValueError(
                f"{src_path}:{label} has ndim={src_arr.ndim}; expected at least 2D for flips/rotations."
            )

        if transform.has_rotation and transform.quarter_turns is None:
            plane_shape = tuple(int(dim) for dim in src_arr.shape[:-2])
            total_planes = prod(plane_shape) if plane_shape else 1
            for plane_index in tqdm(
                _iter_plane_indices(plane_shape),
                total=total_planes,
                desc=f"zarr:{src_path.name}:{label}",
                unit="slice",
            ):
                src_key = plane_index + (slice(None), slice(None))
                dst_arr[src_key] = _transform_array(np.asarray(src_arr[src_key]), transform)
            continue

        axes_to_flip = _flip_axes_for_ndim(src_arr.ndim, transform)
        total_chunks = _count_chunks(dst_arr.shape, dst_arr.chunks)
        skipped_chunks: list[str] = []
        skipped_lock = threading.Lock()
        quarter_turns = transform.quarter_turns

        def process_one(out_slices: tuple[slice, ...]) -> None:
            thread_src_arr = _get_thread_zarr_array(src_path, mode="r", array_path=array_path)
            thread_dst_arr = _get_thread_zarr_array(dst_path, mode="a", array_path=array_path)
            if quarter_turns is not None:
                src_slices = out_slices[:-2] + _quarter_turn_source_slices_2d(
                    (out_slices[-2], out_slices[-1]),
                    src_arr.shape[-2:],
                    quarter_turns,
                )
            else:
                src_slices = _flipped_source_slices(out_slices, src_arr.shape, axes_to_flip)
            try:
                data = np.asarray(thread_src_arr[src_slices])
            except Exception as exc:
                if _is_corrupt_zarr_chunk_error(exc):
                    with skipped_lock:
                        skipped_chunks.append(
                            f"{src_path}:{label} source={src_slices} output={out_slices}"
                        )
                    return
                raise RuntimeError(
                    f"Failed reading Zarr chunk {src_path}:{label} at source slices "
                    f"{src_slices}."
                ) from exc
            if quarter_turns is not None:
                data = _transform_array(data, transform)
            elif axes_to_flip:
                data = np.flip(data, axis=axes_to_flip)
            thread_dst_arr[out_slices] = data

        with ThreadPoolExecutor(max_workers=workers) as executor:
            chunk_iter = _chunk_grid(dst_arr.shape, dst_arr.chunks)
            for _ in tqdm(
                executor.map(process_one, chunk_iter),
                total=total_chunks,
                desc=f"zarr:{src_path.name}:{label}",
                unit="chunk",
            ):
                pass
        if skipped_chunks:
            print(
                f"Warning: skipped {len(skipped_chunks)} corrupt Zarr chunk(s) in "
                f"{src_path}:{label}; output left at fill_value for those regions."
            )
            for item in skipped_chunks:
                print(f"  - {item}")


def _get_thread_tiff_context(path: Path) -> _TiffContext:
    context: _TiffContext | None = getattr(_THREAD_LOCAL, "tiff_context", None)
    if context is not None and context.path == path:
        return context

    if context is not None:
        context.tif.close()

    tif = tifffile.TiffFile(path)
    page = tif.pages[0]
    context = _TiffContext(
        path=path,
        tif=tif,
        page=page,
        tile_cache=OrderedDict(),
    )
    _THREAD_LOCAL.tiff_context = context
    return context


def _decode_tiff_tile(
    path: Path,
    tile_index: int,
    tile_length: int,
    tile_width: int,
) -> np.ndarray:
    context = _get_thread_tiff_context(path)
    cache = context.tile_cache
    cached = cache.get(tile_index)
    if cached is not None:
        cache.move_to_end(tile_index)
        return cached

    page = context.page
    file_handle = context.tif.filehandle
    offset = int(page.dataoffsets[tile_index])
    bytecount = int(page.databytecounts[tile_index])

    file_handle.seek(offset)
    encoded = file_handle.read(bytecount)
    decoded, _, _ = page.decode(encoded, tile_index, jpegtables=page.jpegtables)
    if decoded is None:
        tile = np.zeros((tile_length, tile_width), dtype=page.dtype)
    else:
        tile_array = np.asarray(decoded)
        if tile_array.ndim == 4:
            # For 2D grayscale tiles, tifffile returns shape (1, tile_h, tile_w, 1).
            if tile_array.shape[-1] == 2 and _is_grayscale_alpha_tiff(page):
                tile = tile_array[0, :, :, 0]
            else:
                tile = tile_array[0, :, :, 0]
        elif tile_array.ndim == 3:
            if tile_array.shape[0] == 1:
                tile = tile_array[0]
            elif tile_array.shape[-1] == 1:
                tile = tile_array[:, :, 0]
            elif tile_array.shape[-1] == 2 and _is_grayscale_alpha_tiff(page):
                tile = tile_array[:, :, 0]
            else:
                raise ValueError(
                    f"Unsupported decoded tile shape {tile_array.shape} for {path}"
                )
        elif tile_array.ndim == 2:
            tile = tile_array
        else:
            raise ValueError(
                f"Unsupported decoded tile ndim={tile_array.ndim} for {path}"
            )
        tile = np.asarray(tile, dtype=page.dtype)

    cache[tile_index] = tile
    if len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)
    return tile


def _extract_source_region(
    src_path: Path,
    src_y0: int,
    src_y1: int,
    src_x0: int,
    src_x1: int,
    *,
    image_width: int,
    tile_length: int,
    tile_width: int,
    dtype: np.dtype,
) -> np.ndarray:
    out_h = src_y1 - src_y0
    out_w = src_x1 - src_x0
    out = np.empty((out_h, out_w), dtype=dtype)

    tile_x_count = ceil(image_width / tile_width)
    tile_y_start = src_y0 // tile_length
    tile_y_end = (src_y1 - 1) // tile_length
    tile_x_start = src_x0 // tile_width
    tile_x_end = (src_x1 - 1) // tile_width

    for tile_y in range(tile_y_start, tile_y_end + 1):
        tile_y0 = tile_y * tile_length
        tile_y1 = tile_y0 + tile_length
        copy_y0 = max(src_y0, tile_y0)
        copy_y1 = min(src_y1, tile_y1)

        for tile_x in range(tile_x_start, tile_x_end + 1):
            tile_x0 = tile_x * tile_width
            tile_x1 = tile_x0 + tile_width
            copy_x0 = max(src_x0, tile_x0)
            copy_x1 = min(src_x1, tile_x1)

            source_tile_index = tile_y * tile_x_count + tile_x
            tile = _decode_tiff_tile(src_path, source_tile_index, tile_length, tile_width)

            out[
                (copy_y0 - src_y0) : (copy_y1 - src_y0),
                (copy_x0 - src_x0) : (copy_x1 - src_x0),
            ] = tile[
                (copy_y0 - tile_y0) : (copy_y1 - tile_y0),
                (copy_x0 - tile_x0) : (copy_x1 - tile_x0),
            ]

    return out


def _normalized_tiff_predictor(raw_predictor: int | None, dtype: np.dtype) -> int | None:
    if raw_predictor is None:
        return None

    if np.issubdtype(dtype, np.floating):
        # Floating-point TIFFs should use predictor=3. Some source files
        # incorrectly report predictor=2, which tifffile rejects on write.
        if raw_predictor == 2:
            return 3
        return raw_predictor

    # Integer TIFFs should use predictor=2 if source metadata is mismatched.
    if raw_predictor == 3:
        return 2
    return raw_predictor


def _drop_alpha_tiff_if_present(page: tifffile.TiffPage, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 2:
        return out

    extrasamples = tuple(page.extrasamples) if page.extrasamples else ()
    if out.ndim == 3 and out.shape[-1] == 2 and len(extrasamples) == 1:
        return np.asarray(out[..., 0])

    raise ValueError(
        f"{page.parent.filehandle.path} has shape {page.shape}; "
        "only 2D TIFFs or grayscale+alpha TIFFs are supported."
    )


def _is_grayscale_alpha_tiff(page: tifffile.TiffPage) -> bool:
    extrasamples = tuple(page.extrasamples) if page.extrasamples else ()
    return page.ndim == 3 and int(page.shape[-1]) == 2 and len(extrasamples) == 1


def flip_tiff(src_path: Path, dst_path: Path, flip: TransformConfig, workers: int) -> None:
    with tifffile.TiffFile(src_path) as src_tif:
        page = src_tif.pages[0]
        if page.ndim == 2:
            height, width = int(page.shape[0]), int(page.shape[1])
        elif _is_grayscale_alpha_tiff(page):
            height, width = int(page.shape[0]), int(page.shape[1])
        else:
            raise ValueError(
                f"{src_path} has shape {page.shape}; "
                "only 2D TIFFs or grayscale+alpha TIFFs are supported."
            )
        dtype = np.dtype(page.dtype)

        raw_predictor = int(page.predictor) if getattr(page, "predictor", None) else None
        predictor = _normalized_tiff_predictor(raw_predictor, dtype)
        description = page.description if page.description else None
        resolution = page.resolution if page.resolution else None
        resolutionunit = page.resolutionunit if page.resolutionunit else None
        with tifffile.TiffWriter(
            dst_path,
            bigtiff=src_tif.is_bigtiff,
            byteorder=src_tif.byteorder,
        ) as writer:
            if flip.has_rotation and flip.quarter_turns is not None and page.is_tiled:
                tile_length = int(page.tilelength)
                tile_width = int(page.tilewidth)
                out_height, out_width = _rotated_2d_shape((height, width), flip)
                tile_y_count = ceil(out_height / tile_length)
                tile_x_count = ceil(out_width / tile_width)
                tile_count = tile_y_count * tile_x_count

                def process_output_tile(tile_index: int) -> np.ndarray:
                    tile_y = tile_index // tile_x_count
                    tile_x = tile_index % tile_x_count

                    out_y0 = tile_y * tile_length
                    out_y1 = min(out_height, out_y0 + tile_length)
                    out_x0 = tile_x * tile_width
                    out_x1 = min(out_width, out_x0 + tile_width)
                    src_y_slice, src_x_slice = _quarter_turn_source_slices_2d(
                        (slice(out_y0, out_y1), slice(out_x0, out_x1)),
                        (height, width),
                        flip.quarter_turns,
                    )
                    region = _extract_source_region(
                        src_path,
                        int(src_y_slice.start),
                        int(src_y_slice.stop),
                        int(src_x_slice.start),
                        int(src_x_slice.stop),
                        image_width=width,
                        tile_length=tile_length,
                        tile_width=tile_width,
                        dtype=dtype,
                    )
                    region = _transform_array(region, flip)

                    out_tile = np.zeros((tile_length, tile_width), dtype=dtype)
                    out_tile[: (out_y1 - out_y0), : (out_x1 - out_x0)] = region
                    return out_tile

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    mapped_tiles = executor.map(process_output_tile, range(tile_count))

                    def tile_iterator_with_progress() -> Iterator[np.ndarray]:
                        for tile in tqdm(
                            mapped_tiles,
                            total=tile_count,
                            desc=f"tiff:{src_path.name}",
                            unit="tile",
                        ):
                            yield tile

                    writer.write(
                        tile_iterator_with_progress(),
                        shape=(out_height, out_width),
                        dtype=dtype,
                        tile=(tile_length, tile_width),
                        compression=page.compression,
                        predictor=predictor,
                        photometric=page.photometric,
                        planarconfig=page.planarconfig,
                        description=description,
                        resolution=resolution,
                        resolutionunit=resolutionunit,
                        metadata=None,
                    )
                return

            if flip.has_rotation:
                arr = _transform_array(_drop_alpha_tiff_if_present(page, page.asarray()), flip)
                out_height, out_width = int(arr.shape[0]), int(arr.shape[1])
                if page.is_tiled:
                    writer.write(
                        np.ascontiguousarray(arr, dtype=dtype),
                        shape=(out_height, out_width),
                        dtype=dtype,
                        tile=(int(page.tilelength), int(page.tilewidth)),
                        compression=page.compression,
                        predictor=predictor,
                        photometric=page.photometric,
                        planarconfig=page.planarconfig,
                        description=description,
                        resolution=resolution,
                        resolutionunit=resolutionunit,
                        metadata=None,
                    )
                else:
                    rowsperstrip = int(page.rowsperstrip) if page.rowsperstrip else None
                    writer.write(
                        np.ascontiguousarray(arr, dtype=dtype),
                        shape=(out_height, out_width),
                        dtype=dtype,
                        rowsperstrip=rowsperstrip,
                        compression=page.compression,
                        predictor=predictor,
                        photometric=page.photometric,
                        planarconfig=page.planarconfig,
                        description=description,
                        resolution=resolution,
                        resolutionunit=resolutionunit,
                        metadata=None,
                    )
                return

            if page.is_tiled:
                tile_length = int(page.tilelength)
                tile_width = int(page.tilewidth)
                tile_y_count = ceil(height / tile_length)
                tile_x_count = ceil(width / tile_width)
                tile_count = tile_y_count * tile_x_count

                def process_output_tile(tile_index: int) -> np.ndarray:
                    tile_y = tile_index // tile_x_count
                    tile_x = tile_index % tile_x_count

                    out_y0 = tile_y * tile_length
                    out_y1 = min(height, out_y0 + tile_length)
                    out_x0 = tile_x * tile_width
                    out_x1 = min(width, out_x0 + tile_width)

                    src_y0, src_y1 = out_y0, out_y1
                    src_x0, src_x1 = out_x0, out_x1
                    if flip.vertical:
                        src_y0, src_y1 = height - out_y1, height - out_y0
                    if flip.horizontal:
                        src_x0, src_x1 = width - out_x1, width - out_x0

                    region = _extract_source_region(
                        src_path,
                        src_y0,
                        src_y1,
                        src_x0,
                        src_x1,
                        image_width=width,
                        tile_length=tile_length,
                        tile_width=tile_width,
                        dtype=dtype,
                    )

                    if flip.vertical:
                        region = region[::-1, :]
                    if flip.horizontal:
                        region = region[:, ::-1]

                    out_tile = np.zeros((tile_length, tile_width), dtype=dtype)
                    out_tile[: (out_y1 - out_y0), : (out_x1 - out_x0)] = region
                    return out_tile

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    mapped_tiles = executor.map(process_output_tile, range(tile_count))

                    def tile_iterator_with_progress() -> Iterator[np.ndarray]:
                        for tile in tqdm(
                            mapped_tiles,
                            total=tile_count,
                            desc=f"tiff:{src_path.name}",
                            unit="tile",
                        ):
                            yield tile

                    writer.write(
                        tile_iterator_with_progress(),
                        shape=(height, width),
                        dtype=dtype,
                        tile=(tile_length, tile_width),
                        compression=page.compression,
                        predictor=predictor,
                        photometric=page.photometric,
                        planarconfig=page.planarconfig,
                        description=description,
                        resolution=resolution,
                        resolutionunit=resolutionunit,
                        metadata=None,
                    )
            else:
                arr = _drop_alpha_tiff_if_present(page, page.asarray())
                if flip.vertical:
                    arr = arr[::-1, :]
                if flip.horizontal:
                    arr = arr[:, ::-1]
                rowsperstrip = int(page.rowsperstrip) if page.rowsperstrip else None
                writer.write(
                    np.ascontiguousarray(arr, dtype=dtype),
                    shape=(height, width),
                    dtype=dtype,
                    rowsperstrip=rowsperstrip,
                    compression=page.compression,
                    predictor=predictor,
                    photometric=page.photometric,
                    planarconfig=page.planarconfig,
                    description=description,
                    resolution=resolution,
                    resolutionunit=resolutionunit,
                    metadata=None,
                )


def _preview_tiff(src_path: Path) -> str:
    with tifffile.TiffFile(src_path) as tif:
        page = tif.pages[0]
        tile_repr = (
            f"{int(page.tilelength)}x{int(page.tilewidth)}"
            if page.is_tiled
            else "not-tiled"
        )
        return (
            f"shape={tuple(page.shape)} dtype={page.dtype} "
            f"compression={int(page.compression)} tiles={tile_repr}"
        )


def _preview_zarr(src_path: Path) -> str:
    obj = zarr.open(str(src_path), mode="r")
    if isinstance(obj, zarr.Array):
        return (
            f"array shape={tuple(obj.shape)} dtype={obj.dtype} "
            f"chunks={tuple(obj.chunks)} compressor={obj.compressor}"
        )

    arrays = list(_iter_arrays(obj))
    if not arrays:
        return "group (no arrays)"
    first = arrays[0]
    return (
        f"group arrays={len(arrays)} first={first.path or '/'} "
        f"shape={tuple(first.shape)} dtype={first.dtype} chunks={tuple(first.chunks)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    patch_dir = args.patch_dir.resolve()
    if not patch_dir.is_dir():
        raise FileNotFoundError(f"Patch directory not found: {patch_dir}")

    transform = build_transform_config(args)
    tasks = discover_tasks(
        patch_dir,
        transform.label,
        include_labels=bool(args.flip_labels),
    )
    if not tasks:
        print("No matching assets found.")
        return 0
    plans = build_task_plans(tasks, patch_dir, transform.label, inplace=bool(args.inplace))

    print(f"Discovered {len(tasks)} matching assets under {patch_dir}")
    for plan in plans:
        task = plan.task
        preview = _preview_zarr(task.src) if task.kind == "zarr" else _preview_tiff(task.src)
        print(f"- [{task.kind}] {_describe_plan(plan, inplace=bool(args.inplace))} | {preview}")

    if args.dry_run:
        print("Dry-run only; no outputs were written.")
        return 0

    for plan in plans:
        _prepare_output_path(plan.write_dst, overwrite=bool(args.overwrite))
        if plan.backup_dst is not None:
            _prepare_output_path(plan.backup_dst, overwrite=bool(args.overwrite))

    with _blosc_thread_limit(args.blosc_threads):
        for plan in tqdm(plans, desc="assets", unit="file"):
            task = plan.task
            plan.write_dst.parent.mkdir(parents=True, exist_ok=True)
            if task.kind == "zarr":
                transform_zarr(task.src, plan.write_dst, transform, workers=args.workers)
            else:
                flip_tiff(task.src, plan.write_dst, transform, workers=args.workers)
            if args.inplace:
                _complete_inplace_replace(plan)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
