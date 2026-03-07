#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
from pathlib import Path
from multiprocessing.process import BaseProcess
from typing import Dict, Iterable, List, Literal, Sequence

import cv2
import numpy as np
import tifffile
import zarr
from tqdm.auto import tqdm


TARGET_SUFFIXES = (
    "_supervision_mask.tif",
    "_supervision_mask.tiff",
    "_supervision_mask.png",
    "_inklabels.tif",
    "_inklabels.tiff",
    "_inklabels.png",
)
AXES = [
    {"name": "z", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]
ARRAY_DIMENSIONS = ["z", "y", "x"]
DEFAULT_LEVELS = 6
DEFAULT_CHUNKS = (1, 128, 128)
STREAM_BLOCK_SIZE = 1024
SKIP_DIR_NAMES = {".git", "__pycache__"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert supervision-mask and inklabel images into "
            "six-level OME-Zarr pyramids."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root folder to scan recursively.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes to use. Defaults to min(CPU count, number of files, 8).",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=DEFAULT_LEVELS,
        help=f"Number of pyramid levels to write. Default: {DEFAULT_LEVELS}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output .zarr directory if present.",
    )
    return parser.parse_args(argv)


def is_target_image(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(TARGET_SUFFIXES)


def is_composite_image(path: Path) -> bool:
    if not path.is_file():
        return False

    suffix = path.suffix.lower()
    stem = path.stem.lower()
    folder_name = path.parent.name.lower()
    return (
        suffix in {".tif", ".tiff"}
        and any(token in stem for token in ("max", "composite"))
        and stem.startswith(folder_name)
    )


def find_target_images(root: Path) -> List[Path]:
    matches: List[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if dirname not in SKIP_DIR_NAMES and not dirname.lower().endswith(".zarr")
        )
        current_dir = Path(current_root)
        target_images: List[Path] = []
        composite_candidates: List[Path] = []

        for filename in sorted(filenames):
            candidate = current_dir / filename
            if is_target_image(candidate):
                target_images.append(candidate)
            elif is_composite_image(candidate):
                composite_candidates.append(candidate)

        matches.extend(target_images)
        if target_images and composite_candidates:
            matches.append(composite_candidates[0])
    return matches


def _normalize_to_2d(image: np.ndarray, source_path: Path) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)

    if image.ndim == 3:
        image = image[..., 0]

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image at {source_path}, but got shape={tuple(image.shape)}"
        )

    return np.ascontiguousarray(image)


def _normalized_2d_shape(shape: Sequence[int], source_path: Path) -> tuple[int, int]:
    squeezed = tuple(dimension for dimension in shape if dimension != 1)

    if len(squeezed) == 3:
        squeezed = squeezed[:-1]

    if len(squeezed) != 2:
        raise ValueError(f"Expected a 2D image at {source_path}, but got shape={tuple(shape)}")

    return int(squeezed[0]), int(squeezed[1])


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        image = tifffile.imread(path)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError(f"Failed to read image data from {path}")

    return _normalize_to_2d(image, path)


def build_pyramid(image_2d: np.ndarray, levels: int = DEFAULT_LEVELS) -> List[np.ndarray]:
    return build_pyramid_with_mode(image_2d, levels=levels, downsample_mode="nearest")


def _downsample_mean(current: np.ndarray) -> np.ndarray:
    out_y = (current.shape[1] + 1) // 2
    out_x = (current.shape[2] + 1) // 2

    accum = np.zeros((current.shape[0], out_y, out_x), dtype=np.float64)
    counts = np.zeros((out_y, out_x), dtype=np.float64)

    for y_offset in (0, 1):
        for x_offset in (0, 1):
            block = current[:, y_offset::2, x_offset::2]
            if block.size == 0:
                continue
            accum[:, : block.shape[1], : block.shape[2]] += block
            counts[: block.shape[1], : block.shape[2]] += 1.0

    mean = accum / counts[np.newaxis, :, :]
    if np.issubdtype(current.dtype, np.integer):
        mean = np.rint(mean).astype(current.dtype, copy=False)
    else:
        mean = mean.astype(current.dtype, copy=False)
    return np.ascontiguousarray(mean)


def build_pyramid_with_mode(
    image_2d: np.ndarray,
    *,
    levels: int = DEFAULT_LEVELS,
    downsample_mode: Literal["nearest", "mean"] = "nearest",
) -> List[np.ndarray]:
    if levels < 1:
        raise ValueError("levels must be at least 1")
    if downsample_mode not in {"nearest", "mean"}:
        raise ValueError(f"Unsupported downsample_mode: {downsample_mode}")

    current = np.ascontiguousarray(image_2d[np.newaxis, :, :])
    pyramid = [current]

    for _ in range(1, levels):
        if downsample_mode == "mean":
            current = _downsample_mean(current)
        else:
            current = np.ascontiguousarray(current[:, ::2, ::2])
        pyramid.append(current)

    return pyramid


def _downsample_chunk(
    current: np.ndarray,
    *,
    downsample_mode: Literal["nearest", "mean"],
) -> np.ndarray:
    if downsample_mode == "mean":
        return _downsample_mean(current)
    return np.ascontiguousarray(current[:, ::2, ::2])


def _iter_block_slices(height: int, width: int, *, block_size: int = STREAM_BLOCK_SIZE) -> Iterable[tuple[int, int, int, int]]:
    for y_start in range(0, height, block_size):
        block_height = min(block_size, height - y_start)
        for x_start in range(0, width, block_size):
            block_width = min(block_size, width - x_start)
            yield y_start, x_start, block_height, block_width


def _pyramid_shapes(image_shape: tuple[int, int], levels: int) -> List[tuple[int, int, int]]:
    if levels < 1:
        raise ValueError("levels must be at least 1")

    height, width = image_shape
    shapes: List[tuple[int, int, int]] = []
    for _ in range(levels):
        shapes.append((1, height, width))
        height = (height + 1) // 2
        width = (width + 1) // 2
    return shapes


def _multiscales_metadata(name: str, levels: int) -> Dict[str, object]:
    datasets = []
    for level in range(levels):
        scale_factor = 2 ** level
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, float(scale_factor), float(scale_factor)]}
                ],
            }
        )

    return {
        "multiscales": [
            {
                "name": name,
                "version": "0.4",
                "axes": AXES,
                "datasets": datasets,
            }
        ]
    }


def write_ome_zarr(
    pyramid: Sequence[np.ndarray],
    output_path: Path,
    *,
    chunk_shape: Sequence[int] = DEFAULT_CHUNKS,
    overwrite: bool = False,
) -> None:
    if not pyramid:
        raise ValueError("pyramid must contain at least one level")

    datasets = _create_ome_zarr_datasets(
        output_path,
        image_shape=tuple(int(value) for value in pyramid[0].shape[1:]),
        dtype=pyramid[0].dtype,
        levels=len(pyramid),
        chunk_shape=chunk_shape,
        overwrite=overwrite,
    )
    for dataset, array in zip(datasets, pyramid):
        dataset[:] = array


def _create_ome_zarr_datasets(
    output_path: Path,
    *,
    image_shape: tuple[int, int],
    dtype: np.dtype,
    levels: int,
    chunk_shape: Sequence[int] = DEFAULT_CHUNKS,
    overwrite: bool = False,
) -> List[zarr.Array]:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        shutil.rmtree(output_path)

    shapes = _pyramid_shapes(image_shape, levels)
    group = zarr.open_group(str(output_path), mode="w")
    group.attrs.update(_multiscales_metadata(output_path.stem, len(shapes)))

    datasets: List[zarr.Array] = []
    for level, shape in enumerate(shapes):
        dataset = group.create_dataset(
            str(level),
            shape=shape,
            chunks=tuple(chunk_shape),
            dtype=dtype,
            overwrite=True,
            dimension_separator="/",
        )
        dataset.attrs["_ARRAY_DIMENSIONS"] = ARRAY_DIMENSIONS
        datasets.append(dataset)

    return datasets


def _get_tiled_tiff_metadata(path: Path) -> tuple[tuple[int, int], np.dtype] | None:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if not page.is_tiled:
            return None
        return _normalized_2d_shape(page.shape, path), np.dtype(page.dtype)


def _read_decoded_tile(
    tif: tifffile.TiffFile,
    page: tifffile.TiffPage,
    tile_index: int,
) -> tuple[np.ndarray | None, tuple[int, int, int, int, int], tuple[int, int, int, int]]:
    offset = page.dataoffsets[tile_index]
    bytecount = page.databytecounts[tile_index]
    tif.filehandle.seek(offset)
    data = tif.filehandle.read(bytecount)
    return page.decode(data, tile_index, jpegtables=page.jpegtables)


def _write_tiled_tiff_level_zero(
    input_path: Path,
    dataset: zarr.Array,
) -> None:
    with tifffile.TiffFile(input_path) as tif:
        page = tif.pages[0]
        if not page.is_tiled:
            raise ValueError(f"Expected tiled TIFF input for streaming path: {input_path}")

        image_height, image_width = _normalized_2d_shape(page.shape, input_path)
        tile_height, tile_width = page.chunks
        _, tiles_across = page.chunked

        for block_y, block_x, block_height, block_width in _iter_block_slices(image_height, image_width):
            block = np.zeros((block_height, block_width), dtype=page.dtype)

            tile_row_start = block_y // tile_height
            tile_row_stop = (block_y + block_height + tile_height - 1) // tile_height
            tile_col_start = block_x // tile_width
            tile_col_stop = (block_x + block_width + tile_width - 1) // tile_width

            for tile_row in range(tile_row_start, tile_row_stop):
                for tile_col in range(tile_col_start, tile_col_stop):
                    tile_index = tile_row * tiles_across + tile_col
                    if tile_index >= len(page.dataoffsets):
                        continue

                    decoded, position, _ = _read_decoded_tile(tif, page, tile_index)
                    if decoded is None:
                        continue

                    tile = _normalize_to_2d(decoded, input_path)
                    tile_y = position[2]
                    tile_x = position[3]
                    tile_bottom = tile_y + tile.shape[0]
                    tile_right = tile_x + tile.shape[1]

                    overlap_y0 = max(block_y, tile_y)
                    overlap_y1 = min(block_y + block_height, tile_bottom)
                    overlap_x0 = max(block_x, tile_x)
                    overlap_x1 = min(block_x + block_width, tile_right)
                    if overlap_y0 >= overlap_y1 or overlap_x0 >= overlap_x1:
                        continue

                    block[
                        overlap_y0 - block_y : overlap_y1 - block_y,
                        overlap_x0 - block_x : overlap_x1 - block_x,
                    ] = tile[
                        overlap_y0 - tile_y : overlap_y1 - tile_y,
                        overlap_x0 - tile_x : overlap_x1 - tile_x,
                    ]

            dataset[0, block_y : block_y + block_height, block_x : block_x + block_width] = block


def _write_downsample_block(
    source_dataset: zarr.Array,
    target_dataset: zarr.Array,
    *,
    block_y: int,
    block_x: int,
    block_height: int,
    block_width: int,
    downsample_mode: Literal["nearest", "mean"],
) -> None:
    source_y0 = block_y * 2
    source_x0 = block_x * 2
    source_y1 = min(source_dataset.shape[1], (block_y + block_height) * 2)
    source_x1 = min(source_dataset.shape[2], (block_x + block_width) * 2)

    source_block = np.asarray(source_dataset[:, source_y0:source_y1, source_x0:source_x1])
    downsampled = _downsample_chunk(source_block, downsample_mode=downsample_mode)
    target_dataset[:, block_y : block_y + downsampled.shape[1], block_x : block_x + downsampled.shape[2]] = downsampled


def _build_downsample_levels_from_zarr(
    datasets: Sequence[zarr.Array],
    *,
    downsample_mode: Literal["nearest", "mean"],
    chunk_workers: int,
) -> None:
    for level in range(1, len(datasets)):
        source_dataset = datasets[level - 1]
        target_dataset = datasets[level]
        blocks = list(_iter_block_slices(target_dataset.shape[1], target_dataset.shape[2]))

        if chunk_workers <= 1 or len(blocks) <= 1:
            for block_y, block_x, block_height, block_width in blocks:
                _write_downsample_block(
                    source_dataset,
                    target_dataset,
                    block_y=block_y,
                    block_x=block_x,
                    block_height=block_height,
                    block_width=block_width,
                    downsample_mode=downsample_mode,
                )
            continue

        max_workers = min(chunk_workers, len(blocks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _write_downsample_block,
                    source_dataset,
                    target_dataset,
                    block_y=block_y,
                    block_x=block_x,
                    block_height=block_height,
                    block_width=block_width,
                    downsample_mode=downsample_mode,
                )
                for block_y, block_x, block_height, block_width in blocks
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()


def _convert_tiled_tiff(
    input_path: Path,
    output_path: Path,
    *,
    levels: int,
    overwrite: bool,
    downsample_mode: Literal["nearest", "mean"],
    chunk_workers: int,
) -> None:
    metadata = _get_tiled_tiff_metadata(input_path)
    if metadata is None:
        raise ValueError(f"Expected tiled TIFF metadata for {input_path}")

    image_shape, dtype = metadata
    datasets = _create_ome_zarr_datasets(
        output_path,
        image_shape=image_shape,
        dtype=dtype,
        levels=levels,
        overwrite=overwrite,
    )
    _write_tiled_tiff_level_zero(input_path, datasets[0])
    _build_downsample_levels_from_zarr(
        datasets,
        downsample_mode=downsample_mode,
        chunk_workers=chunk_workers,
    )


def convert_image(
    input_path: Path,
    *,
    levels: int = DEFAULT_LEVELS,
    overwrite: bool = False,
    chunk_workers: int = 1,
) -> Dict[str, str]:
    output_path = input_path.with_suffix(".zarr")

    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "input": str(input_path),
            "output": str(output_path),
        }

    downsample_mode: Literal["nearest", "mean"] = "mean" if is_composite_image(input_path) else "nearest"
    tiled_metadata = _get_tiled_tiff_metadata(input_path)
    if tiled_metadata is not None:
        _convert_tiled_tiff(
            input_path,
            output_path,
            levels=levels,
            overwrite=overwrite,
            downsample_mode=downsample_mode,
            chunk_workers=chunk_workers,
        )
    else:
        image = load_image(input_path)
        pyramid = build_pyramid_with_mode(
            image,
            levels=levels,
            downsample_mode=downsample_mode,
        )
        write_ome_zarr(pyramid, output_path, overwrite=overwrite)

    return {
        "status": "written",
        "input": str(input_path),
        "output": str(output_path),
        "downsample_mode": downsample_mode,
        "streamed_tiled_tiff": str(tiled_metadata is not None).lower(),
    }


def _convert_image_worker(
    input_path: str,
    levels: int,
    overwrite: bool,
    chunk_workers: int,
) -> Dict[str, str]:
    return convert_image(Path(input_path), levels=levels, overwrite=overwrite, chunk_workers=chunk_workers)


def _terminate_process_pool(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    # Python 3.11 lacks a public hard-stop API for ProcessPoolExecutor, so on
    # Ctrl-C we explicitly terminate child workers to avoid orphaned processes.
    processes = [
        process
        for process in getattr(executor, "_processes", {}).values()
        if isinstance(process, BaseProcess)
    ]
    executor.shutdown(wait=False, cancel_futures=True)

    for process in processes:
        if process.is_alive():
            process.terminate()

    for process in processes:
        process.join(timeout=0.2)

    for process in processes:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()

    for process in processes:
        process.join(timeout=0.2)


def run_conversion(
    image_paths: Sequence[Path],
    *,
    workers: int,
    levels: int,
    overwrite: bool,
) -> Dict[str, Iterable[Dict[str, str]]]:
    results: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    if workers <= 1:
        chunk_workers = min(os.cpu_count() or 1, 8)
        iterator = tqdm(image_paths, total=len(image_paths), desc="Converting", unit="file")
        for image_path in iterator:
            try:
                results.append(
                    convert_image(
                        image_path,
                        levels=levels,
                        overwrite=overwrite,
                        chunk_workers=chunk_workers,
                    )
                )
            except Exception as exc:
                errors.append({"input": str(image_path), "error": str(exc)})
        return {"results": results, "errors": errors}

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_convert_image_worker, str(image_path), levels, overwrite, 1): image_path
            for image_path in image_paths
        }
        try:
            for future in tqdm(
                concurrent.futures.as_completed(future_map),
                total=len(future_map),
                desc="Converting",
                unit="file",
            ):
                image_path = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append({"input": str(image_path), "error": str(exc)})
        except KeyboardInterrupt:
            _terminate_process_pool(executor)
            raise

    return {"results": results, "errors": errors}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    image_paths = find_target_images(root)
    if not image_paths:
        print(f"No matching images found under {root}")
        return 0

    max_workers = args.workers
    if max_workers is None:
        max_workers = min(len(image_paths), os.cpu_count() or 1, 8)
    if max_workers < 1:
        raise ValueError("--workers must be at least 1")

    outcome = run_conversion(
        image_paths,
        workers=max_workers,
        levels=args.levels,
        overwrite=args.overwrite,
    )

    results = list(outcome["results"])
    errors = list(outcome["errors"])
    written = sum(1 for result in results if result["status"] == "written")
    skipped = sum(1 for result in results if result["status"] == "skipped")

    print(
        f"Processed {len(image_paths)} image(s): "
        f"{written} written, {skipped} skipped, {len(errors)} failed."
    )
    if errors:
        for error in errors:
            print(f"ERROR {error['input']}: {error['error']}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
