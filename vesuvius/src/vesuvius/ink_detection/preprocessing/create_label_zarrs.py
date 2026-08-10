"""Convert flat label images into multiscale ZYX OME-Zarr label volumes."""

from __future__ import annotations

import concurrent.futures
from multiprocessing.process import BaseProcess
import os
from pathlib import Path
import shutil
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np
from numcodecs import Blosc
import tifffile
from tqdm.auto import tqdm
import zarr

from vesuvius.ink_detection.segment import parse_label_asset_path
from vesuvius.utils.cli import HyphenUnderscoreParser


AXES = [
    {"name": "z", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]
ARRAY_DIMENSIONS = ["z", "y", "x"]
DEFAULT_LEVELS = 6
DEFAULT_DEPTH = 65
DEFAULT_LABEL_SLICE = 32
DEFAULT_CHUNKS = (65, 128, 128)
STREAM_BLOCK_SIZE = 1024
SKIP_DIR_NAMES = {".git", "__pycache__"}
LABEL_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def parse_target_image(path: Path) -> dict[str, object] | None:
    """Parse a supported label image name, preserving its source spelling."""
    if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff", ".png"}:
        return None

    parser_suffix = ".tif" if path.suffix.lower() == ".png" else path.suffix.lower()
    normalized = path.with_name(path.stem.lower() + parser_suffix)
    parsed = parse_label_asset_path(normalized)
    if parsed is None:
        return None
    version_num = int(parsed["version_num"])
    final_stem_part = path.stem.rsplit("_", 1)[-1].lower()
    explicit_version = final_stem_part.startswith("v") and final_stem_part[1:].isdigit()
    prefix_length = len(str(parsed["prefix"]))
    return {
        "prefix": path.stem[:prefix_length],
        "label_kind": parsed["label_kind"],
        "version_num": version_num if explicit_version else None,
        "extension": path.suffix,
    }


def is_target_image(path: Path) -> bool:
    """Return whether a file is a supported flat label image."""
    return parse_target_image(path) is not None


def is_composite_image(path: Path) -> bool:
    """Return whether a TIFF is the folder's max/composite context image."""
    if not path.is_file():
        return False
    stem = path.stem.lower()
    return (
        path.suffix.lower() in {".tif", ".tiff"}
        and any(token in stem for token in ("max", "composite"))
        and stem.startswith(path.parent.name.lower())
    )


def find_target_images(root: Path) -> list[Path]:
    """Find labels and one context composite per labeled directory in stable order."""
    matches: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if dirname not in SKIP_DIR_NAMES
            and not dirname.lower().endswith(".zarr")
        )
        current_dir = Path(current_root)
        target_images: list[Path] = []
        composite_candidates: list[Path] = []
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
    image = np.squeeze(np.asarray(image))
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image at {source_path}, but got shape={tuple(image.shape)}"
        )
    return np.ascontiguousarray(image)


def _normalized_2d_shape(
    shape: Sequence[int], source_path: Path
) -> tuple[int, int]:
    squeezed = tuple(dimension for dimension in shape if dimension != 1)
    if len(squeezed) == 3:
        squeezed = squeezed[:-1]
    if len(squeezed) != 2:
        raise ValueError(
            f"Expected a 2D image at {source_path}, but got shape={tuple(shape)}"
        )
    return int(squeezed[0]), int(squeezed[1])


def load_image(path: Path) -> np.ndarray:
    """Read a TIFF or PNG as one contiguous two-dimensional array."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(path)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image data from {path}")
    return _normalize_to_2d(image, path)


def _embed_label_volume(
    image_2d: np.ndarray,
    *,
    depth: int = DEFAULT_DEPTH,
    label_slice: int = DEFAULT_LABEL_SLICE,
) -> np.ndarray:
    if not 0 <= label_slice < depth:
        raise ValueError(f"label_slice must be within [0, {depth}), got {label_slice}")
    volume_ZYX = np.zeros(
        (depth, image_2d.shape[0], image_2d.shape[1]), dtype=image_2d.dtype
    )
    volume_ZYX[label_slice] = image_2d
    return volume_ZYX


def _downsample_mean(current_ZYX: np.ndarray) -> np.ndarray:
    out_y = (current_ZYX.shape[1] + 1) // 2
    out_x = (current_ZYX.shape[2] + 1) // 2
    accum = np.zeros((current_ZYX.shape[0], out_y, out_x), dtype=np.float64)
    counts = np.zeros((out_y, out_x), dtype=np.float64)
    for y_offset in (0, 1):
        for x_offset in (0, 1):
            block_ZYX = current_ZYX[:, y_offset::2, x_offset::2]
            if block_ZYX.size == 0:
                continue
            accum[:, : block_ZYX.shape[1], : block_ZYX.shape[2]] += block_ZYX
            counts[: block_ZYX.shape[1], : block_ZYX.shape[2]] += 1.0
    mean_ZYX = accum / counts[np.newaxis, :, :]
    if np.issubdtype(current_ZYX.dtype, np.integer):
        mean_ZYX = np.rint(mean_ZYX).astype(current_ZYX.dtype, copy=False)
    else:
        mean_ZYX = mean_ZYX.astype(current_ZYX.dtype, copy=False)
    return np.ascontiguousarray(mean_ZYX)


def build_pyramid_with_mode(
    image_2d: np.ndarray,
    *,
    levels: int = DEFAULT_LEVELS,
    downsample_mode: Literal["nearest", "mean"] = "nearest",
) -> list[np.ndarray]:
    """Embed a label at Z=32 and derive XY pyramid levels without Z pooling."""
    if levels < 1:
        raise ValueError("levels must be at least 1")
    if downsample_mode not in {"nearest", "mean"}:
        raise ValueError(f"Unsupported downsample_mode: {downsample_mode}")
    current_ZYX = np.ascontiguousarray(_embed_label_volume(image_2d))
    pyramid = [current_ZYX]
    for _ in range(1, levels):
        if downsample_mode == "mean":
            current_ZYX = _downsample_mean(current_ZYX)
        else:
            current_ZYX = np.ascontiguousarray(current_ZYX[:, ::2, ::2])
        pyramid.append(current_ZYX)
    return pyramid


def build_pyramid(
    image_2d: np.ndarray, levels: int = DEFAULT_LEVELS
) -> list[np.ndarray]:
    """Build the nearest-neighbor label pyramid used for label assets."""
    return build_pyramid_with_mode(image_2d, levels=levels)


def _iter_block_slices(
    height: int, width: int, *, block_size: int = STREAM_BLOCK_SIZE
) -> Iterable[tuple[int, int, int, int]]:
    for y_start in range(0, height, block_size):
        block_height = min(block_size, height - y_start)
        for x_start in range(0, width, block_size):
            block_width = min(block_size, width - x_start)
            yield y_start, x_start, block_height, block_width


def _pyramid_shapes(
    image_shape: tuple[int, int], levels: int
) -> list[tuple[int, int, int]]:
    if levels < 1:
        raise ValueError("levels must be at least 1")
    height, width = image_shape
    shapes = []
    for _ in range(levels):
        shapes.append((DEFAULT_DEPTH, height, width))
        height = (height + 1) // 2
        width = (width + 1) // 2
    return shapes


def _multiscales_metadata(name: str, levels: int) -> dict[str, object]:
    datasets = []
    for level in range(levels):
        scale_factor = 2**level
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1.0, float(scale_factor), float(scale_factor)],
                    }
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


def _create_dataset(
    group: zarr.Group,
    name: str,
    *,
    shape: tuple[int, int, int],
    chunks: tuple[int, int, int],
    dtype: np.dtype,
    compressor: Blosc | None,
) -> zarr.Array:
    kwargs = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "compressor": compressor,
        "fill_value": 0,
        "overwrite": True,
    }
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        return group.create_array(
            name,
            **kwargs,
            chunk_key_encoding={"name": "v2", "separator": "/"},
            config={"write_empty_chunks": False},
        )
    return group.create_dataset(
        name,
        **kwargs,
        dimension_separator="/",
        write_empty_chunks=False,
    )


def _create_ome_zarr_datasets(
    output_path: Path,
    *,
    image_shape: tuple[int, int],
    dtype: np.dtype,
    levels: int,
    chunk_shape: Sequence[int] = DEFAULT_CHUNKS,
    overwrite: bool = False,
    use_compression: bool = True,
) -> list[zarr.Array]:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    group_kwargs: dict[str, object] = {"mode": "w"}
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        group_kwargs["zarr_format"] = 2
    group = zarr.open_group(str(output_path), **group_kwargs)
    shapes = _pyramid_shapes(image_shape, levels)
    group.attrs.update(_multiscales_metadata(output_path.stem, len(shapes)))
    compressor = LABEL_COMPRESSOR if use_compression else None
    chunks = tuple(int(value) for value in chunk_shape)
    datasets = []
    for level, shape in enumerate(shapes):
        dataset = _create_dataset(
            group,
            str(level),
            shape=shape,
            chunks=chunks,
            dtype=np.dtype(dtype),
            compressor=compressor,
        )
        dataset.attrs["_ARRAY_DIMENSIONS"] = ARRAY_DIMENSIONS
        datasets.append(dataset)
    return datasets


def write_ome_zarr(
    pyramid: Sequence[np.ndarray],
    output_path: Path,
    *,
    chunk_shape: Sequence[int] = DEFAULT_CHUNKS,
    overwrite: bool = False,
    use_compression: bool = True,
) -> None:
    """Write an in-memory ZYX pyramid as an explicit Zarr-v2 OME group."""
    if not pyramid:
        raise ValueError("pyramid must contain at least one level")
    datasets = _create_ome_zarr_datasets(
        output_path,
        image_shape=tuple(int(value) for value in pyramid[0].shape[1:]),
        dtype=pyramid[0].dtype,
        levels=len(pyramid),
        chunk_shape=chunk_shape,
        overwrite=overwrite,
        use_compression=use_compression,
    )
    for dataset, array_ZYX in zip(datasets, pyramid):
        dataset[:] = array_ZYX


def _get_tiled_tiff_metadata(
    path: Path,
) -> tuple[tuple[int, int], np.dtype] | None:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if not page.is_tiled:
            return None
        return _normalized_2d_shape(page.shape, path), np.dtype(page.dtype)


def _write_tiled_tiff_level_zero(input_path: Path, dataset: zarr.Array) -> None:
    with tifffile.TiffFile(input_path) as tif:
        page = tif.pages[0]
        if not page.is_tiled:
            raise ValueError(f"Expected tiled TIFF input for streaming path: {input_path}")
        image_height, image_width = _normalized_2d_shape(page.shape, input_path)
        tile_height, tile_width = page.chunks
        _, tiles_across = page.chunked
        for block_y, block_x, block_height, block_width in _iter_block_slices(
            image_height, image_width
        ):
            block_YX = np.zeros((block_height, block_width), dtype=page.dtype)
            tile_row_start = block_y // tile_height
            tile_row_stop = (block_y + block_height + tile_height - 1) // tile_height
            tile_col_start = block_x // tile_width
            tile_col_stop = (block_x + block_width + tile_width - 1) // tile_width
            for tile_row in range(tile_row_start, tile_row_stop):
                for tile_col in range(tile_col_start, tile_col_stop):
                    tile_index = tile_row * tiles_across + tile_col
                    if tile_index >= len(page.dataoffsets):
                        continue
                    offset = page.dataoffsets[tile_index]
                    bytecount = page.databytecounts[tile_index]
                    tif.filehandle.seek(offset)
                    data = tif.filehandle.read(bytecount)
                    decoded, position, _ = page.decode(
                        data, tile_index, jpegtables=page.jpegtables
                    )
                    if decoded is None:
                        continue
                    tile_YX = _normalize_to_2d(decoded, input_path)
                    tile_y, tile_x = position[2], position[3]
                    overlap_y0 = max(block_y, tile_y)
                    overlap_y1 = min(block_y + block_height, tile_y + tile_YX.shape[0])
                    overlap_x0 = max(block_x, tile_x)
                    overlap_x1 = min(block_x + block_width, tile_x + tile_YX.shape[1])
                    if overlap_y0 >= overlap_y1 or overlap_x0 >= overlap_x1:
                        continue
                    block_YX[
                        overlap_y0 - block_y : overlap_y1 - block_y,
                        overlap_x0 - block_x : overlap_x1 - block_x,
                    ] = tile_YX[
                        overlap_y0 - tile_y : overlap_y1 - tile_y,
                        overlap_x0 - tile_x : overlap_x1 - tile_x,
                    ]
            dataset[
                DEFAULT_LABEL_SLICE,
                block_y : block_y + block_height,
                block_x : block_x + block_width,
            ] = block_YX


def _write_downsample_block(
    source: zarr.Array,
    target: zarr.Array,
    *,
    block_y: int,
    block_x: int,
    block_height: int,
    block_width: int,
    downsample_mode: Literal["nearest", "mean"],
) -> None:
    source_y1 = min(source.shape[1], (block_y + block_height) * 2)
    source_x1 = min(source.shape[2], (block_x + block_width) * 2)
    source_ZYX = np.asarray(
        source[:, block_y * 2 : source_y1, block_x * 2 : source_x1]
    )
    if downsample_mode == "mean":
        downsampled_ZYX = _downsample_mean(source_ZYX)
    else:
        downsampled_ZYX = np.ascontiguousarray(source_ZYX[:, ::2, ::2])
    target[
        :,
        block_y : block_y + downsampled_ZYX.shape[1],
        block_x : block_x + downsampled_ZYX.shape[2],
    ] = downsampled_ZYX


def _build_downsample_levels_from_zarr(
    datasets: Sequence[zarr.Array],
    *,
    downsample_mode: Literal["nearest", "mean"],
    chunk_workers: int,
) -> None:
    for level in range(1, len(datasets)):
        source, target = datasets[level - 1], datasets[level]
        blocks = list(_iter_block_slices(target.shape[1], target.shape[2]))
        if chunk_workers <= 1 or len(blocks) <= 1:
            for block_y, block_x, block_height, block_width in blocks:
                _write_downsample_block(
                    source,
                    target,
                    block_y=block_y,
                    block_x=block_x,
                    block_height=block_height,
                    block_width=block_width,
                    downsample_mode=downsample_mode,
                )
            continue
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(chunk_workers, len(blocks))
        ) as executor:
            futures = [
                executor.submit(
                    _write_downsample_block,
                    source,
                    target,
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


def convert_image(
    input_path: Path,
    *,
    levels: int = DEFAULT_LEVELS,
    overwrite: bool = False,
    chunk_workers: int = 1,
) -> dict[str, str]:
    """Convert one label or context image, skipping an existing output by default."""
    output_path = input_path.with_suffix(".zarr")
    if not overwrite and output_path.exists():
        return {
            "status": "skipped",
            "input": str(input_path),
            "output": str(output_path),
            "additional_outputs": "",
        }
    downsample_mode: Literal["nearest", "mean"] = (
        "mean" if is_composite_image(input_path) else "nearest"
    )
    tiled_metadata = _get_tiled_tiff_metadata(input_path)
    if tiled_metadata is not None:
        image_shape, dtype = tiled_metadata
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
    else:
        image_YX = load_image(input_path)
        pyramid = build_pyramid_with_mode(
            image_YX, levels=levels, downsample_mode=downsample_mode
        )
        write_ome_zarr(pyramid, output_path, overwrite=overwrite)
    return {
        "status": "written",
        "input": str(input_path),
        "output": str(output_path),
        "additional_outputs": "",
        "downsample_mode": downsample_mode,
        "streamed_tiled_tiff": str(tiled_metadata is not None).lower(),
    }


def _convert_image_worker(
    input_path: str, levels: int, overwrite: bool, chunk_workers: int
) -> dict[str, str]:
    return convert_image(
        Path(input_path),
        levels=levels,
        overwrite=overwrite,
        chunk_workers=chunk_workers,
    )


def _terminate_process_pool(
    executor: concurrent.futures.ProcessPoolExecutor,
) -> None:
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
) -> dict[str, Iterable[dict[str, str]]]:
    """Convert discovered inputs, collecting per-file failures for command reporting."""
    results: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    if workers <= 1:
        chunk_workers = min(os.cpu_count() or 1, 8)
        for image_path in tqdm(
            image_paths, total=len(image_paths), desc="Converting", unit="file"
        ):
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
            executor.submit(
                _convert_image_worker, str(image_path), levels, overwrite, 1
            ): image_path
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


def parse_args(argv: Sequence[str] | None = None):
    """Parse the recursive label-conversion command line."""
    parser = HyphenUnderscoreParser(
        description=(
            "Recursively convert supervision-mask, validation-mask, and inklabel "
            "images into six-level OME-Zarr pyramids."
        )
    )
    parser.add_argument("root", type=Path, help="Root folder to scan recursively.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes to use. Defaults to min(CPU count, files, 8).",
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run recursive label conversion and return one when any input fails."""
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
    workers = args.workers
    if workers is None:
        workers = min(len(image_paths), os.cpu_count() or 1, 8)
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    outcome = run_conversion(
        image_paths,
        workers=workers,
        levels=args.levels,
        overwrite=args.overwrite,
    )
    results = list(outcome["results"])
    errors = list(outcome["errors"])
    written = sum(result["status"] == "written" for result in results)
    skipped = sum(result["status"] == "skipped" for result in results)
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
