#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from math import ceil
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import tifffile
import zarr
from numcodecs import Blosc
from tqdm.auto import tqdm


_DEFAULT_WORKERS = max(1, min(32, os.cpu_count() or 1))
_CACHE_SIZE = 16
DEFAULT_ZARR_COMPRESSOR = Blosc(cname="zstd", clevel=2)

_THREAD_LOCAL = threading.local()


@dataclass(frozen=True, slots=True)
class FlipConfig:
    horizontal: bool
    vertical: bool
    label: str


@dataclass(frozen=True, slots=True)
class Task:
    src: Path
    dst: Path
    kind: str


@dataclass(slots=True)
class _TiffContext:
    path: Path
    tif: tifffile.TiffFile
    page: tifffile.TiffPage
    tile_cache: OrderedDict[int, np.ndarray]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flip matched .zarr and _max_22_42 TIFF assets inside a patch folder "
            "using chunk/tile streaming."
        )
    )
    parser.add_argument(
        "patch_dir",
        type=Path,
        help="Patch directory to scan recursively.",
    )
    parser.add_argument(
        "--flip",
        choices=("horizontal", "vertical", "both"),
        required=True,
        help="Flip direction.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output path.",
    )
    parser.add_argument(
        "--flip-labels",
        action="store_true",
        help="Also flip *_inklabels and *_supervision_mask TIFF/Zarr assets.",
    )
    args = parser.parse_args(argv)

    if args.workers <= 0:
        parser.error("--workers must be at least 1")

    return args


def build_flip_config(value: str) -> FlipConfig:
    if value == "horizontal":
        return FlipConfig(horizontal=True, vertical=False, label=value)
    if value == "vertical":
        return FlipConfig(horizontal=False, vertical=True, label=value)
    return FlipConfig(horizontal=True, vertical=True, label=value)


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

        selected_zarr_dirs = [
            dirname
            for dirname in sorted(dirnames)
            if _is_selected_zarr(dirname, current.name, include_labels=include_labels)
        ]
        for dirname in selected_zarr_dirs:
            src = current / dirname
            dst = src.with_name(f"{src.stem}_flip_{flip_label}.zarr")
            tasks.append(Task(src=src, dst=dst, kind="zarr"))

        for filename in sorted(filenames):
            if _is_selected_tiff(filename, include_labels=include_labels):
                src = current / filename
                dst = src.with_name(f"{src.stem}_flip_{flip_label}{src.suffix}")
                tasks.append(Task(src=src, dst=dst, kind="tiff"))

        dirnames[:] = sorted(
            dirname for dirname in dirnames if not dirname.lower().endswith(".zarr")
        )

    return tasks


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
) -> None:
    dst_group.attrs.update(dict(src_group.attrs))

    for subgroup_name in sorted(src_group.group_keys()):
        src_subgroup = src_group[subgroup_name]
        dst_subgroup = dst_group.create_group(subgroup_name, overwrite=False)
        _copy_group_structure(src_subgroup, dst_subgroup, compressor=compressor)

    for array_name in sorted(src_group.array_keys()):
        src_arr = src_group[array_name]
        dst_arr = dst_group.create_dataset(
            array_name,
            shape=src_arr.shape,
            chunks=src_arr.chunks,
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


def _flip_axes_for_ndim(ndim: int, config: FlipConfig) -> tuple[int, ...]:
    axes: list[int] = []
    if config.vertical:
        axes.append(ndim - 2)
    if config.horizontal:
        axes.append(ndim - 1)
    return tuple(axes)


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


def flip_zarr(src_path: Path, dst_path: Path, flip: FlipConfig, workers: int) -> None:
    src_obj = zarr.open(str(src_path), mode="r")
    target_compressor = DEFAULT_ZARR_COMPRESSOR

    if isinstance(src_obj, zarr.Array):
        dst_arr = zarr.open_array(
            store=zarr.DirectoryStore(str(dst_path)),
            mode="w",
            shape=src_obj.shape,
            chunks=src_obj.chunks,
            dtype=src_obj.dtype,
            compressor=target_compressor,
            filters=src_obj.filters,
            fill_value=src_obj.fill_value,
            order=src_obj.order,
            write_empty_chunks=False,
        )
        dst_arr.attrs.update(dict(src_obj.attrs))
        array_pairs = [(src_obj, dst_arr, src_path.name)]
    else:
        dst_root = zarr.group(store=zarr.DirectoryStore(str(dst_path)), overwrite=False)
        _copy_group_structure(src_obj, dst_root, compressor=target_compressor)
        array_pairs = []
        src_arrays = {array.path: array for array in _iter_arrays(src_obj)}
        dst_arrays = {array.path: array for array in _iter_arrays(dst_root)}
        for key in sorted(src_arrays):
            array_pairs.append((src_arrays[key], dst_arrays[key], key))

    for src_arr, dst_arr, label in array_pairs:
        if src_arr.ndim < 2:
            raise ValueError(
                f"{src_path}:{label} has ndim={src_arr.ndim}; expected at least 2D for horizontal/vertical flips."
            )

        axes_to_flip = _flip_axes_for_ndim(src_arr.ndim, flip)
        total_chunks = _count_chunks(src_arr.shape, src_arr.chunks)

        def process_one(out_slices: tuple[slice, ...]) -> None:
            src_slices = _flipped_source_slices(out_slices, src_arr.shape, axes_to_flip)
            data = np.asarray(src_arr[src_slices])
            if axes_to_flip:
                data = np.flip(data, axis=axes_to_flip)
            dst_arr[out_slices] = data

        with ThreadPoolExecutor(max_workers=workers) as executor:
            chunk_iter = _chunk_grid(src_arr.shape, src_arr.chunks)
            for _ in tqdm(
                executor.map(process_one, chunk_iter),
                total=total_chunks,
                desc=f"zarr:{src_path.name}:{label}",
                unit="chunk",
            ):
                pass


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
            tile = tile_array[0, :, :, 0]
        elif tile_array.ndim == 3:
            if tile_array.shape[0] == 1:
                tile = tile_array[0]
            elif tile_array.shape[-1] == 1:
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


def flip_tiff(src_path: Path, dst_path: Path, flip: FlipConfig, workers: int) -> None:
    with tifffile.TiffFile(src_path) as src_tif:
        page = src_tif.pages[0]
        if page.ndim != 2:
            raise ValueError(f"{src_path} has shape {page.shape}; only 2D TIFFs are supported.")
        height, width = int(page.shape[0]), int(page.shape[1])
        dtype = np.dtype(page.dtype)

        raw_predictor = int(page.predictor) if getattr(page, "predictor", None) else None
        predictor = _normalized_tiff_predictor(raw_predictor, dtype)
        extrasamples = tuple(page.extrasamples) if page.extrasamples else None
        description = page.description if page.description else None
        resolution = page.resolution if page.resolution else None
        resolutionunit = page.resolutionunit if page.resolutionunit else None
        with tifffile.TiffWriter(
            dst_path,
            bigtiff=src_tif.is_bigtiff,
            byteorder=src_tif.byteorder,
        ) as writer:
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
                        extrasamples=extrasamples,
                        description=description,
                        resolution=resolution,
                        resolutionunit=resolutionunit,
                        metadata=None,
                    )
            else:
                arr = np.asarray(page.asarray())
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
                    extrasamples=extrasamples,
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

    flip = build_flip_config(args.flip)
    tasks = discover_tasks(patch_dir, flip.label, include_labels=bool(args.flip_labels))
    if not tasks:
        print("No matching assets found.")
        return 0

    print(f"Discovered {len(tasks)} matching assets under {patch_dir}")
    for task in tasks:
        preview = _preview_zarr(task.src) if task.kind == "zarr" else _preview_tiff(task.src)
        print(f"- [{task.kind}] {task.src} -> {task.dst} | {preview}")

    if args.dry_run:
        print("Dry-run only; no outputs were written.")
        return 0

    for task in tasks:
        if task.dst.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {task.dst}. Use --overwrite to replace it."
                )
            if task.dst.is_dir():
                shutil.rmtree(task.dst)
            else:
                task.dst.unlink()

    for task in tqdm(tasks, desc="assets", unit="file"):
        task.dst.parent.mkdir(parents=True, exist_ok=True)
        if task.kind == "zarr":
            flip_zarr(task.src, task.dst, flip, workers=args.workers)
        else:
            flip_tiff(task.src, task.dst, flip, workers=args.workers)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
