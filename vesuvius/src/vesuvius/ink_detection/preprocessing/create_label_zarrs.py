"""Convert flat label images into multiscale ZYX OME-Zarr label volumes."""

from __future__ import annotations

import concurrent.futures
import logging
from multiprocessing.process import BaseProcess
import os
from pathlib import Path
import shutil
from typing import Iterable, Literal, Sequence

import numpy as np
from vc_reader import images
from tqdm.auto import tqdm
from vc_reader.reader import ZarrWriter, write_pyramid_level

from vesuvius.ink_detection.data.segment import parse_label_asset_path
from vesuvius.label_zarr import (
    DEFAULT_LEVELS,
    LABEL_SLICE as DEFAULT_LABEL_SLICE,
    VOLUME_DEPTH as DEFAULT_DEPTH,
    create_label_array,
    create_label_group,
    pyramid_shapes,
)
from vesuvius.utils.cli import HyphenUnderscoreParser

LOGGER = logging.getLogger(__name__)


STREAM_BLOCK_SIZE = 1024
STREAM_DECODED_BYTES = 1 << 30
SKIP_DIR_NAMES = {".git", "__pycache__"}


def parse_target_image(path: Path) -> dict[str, object] | None:
    """Parse a supported label image name, preserving its source spelling."""
    if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff", ".png"}:
        return None

    parser_suffix = ".tif" if path.suffix.lower() == ".png" else path.suffix.lower()
    normalized = path.with_name(path.stem.lower() + parser_suffix)
    parsed = parse_label_asset_path(normalized)
    if parsed is None:
        return None
    unversioned_stem = f"{parsed['prefix']}_{parsed['label_kind']}"
    version_num = (
        None
        if normalized.stem == unversioned_stem
        else int(parsed["version_num"])
    )
    prefix_length = len(str(parsed["prefix"]))
    return {
        "prefix": path.stem[:prefix_length],
        "label_kind": parsed["label_kind"],
        "version_num": version_num,
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


def _image_page_count(reader) -> int:
    """Number of pages in the file's first series."""
    return int(reader.shape[0]) if reader.axes[0] == "z" else 1


def _warn_extra_pages(path: Path, page_count: int) -> None:
    """Say so when only the first page of a multi-page TIFF is converted."""
    if page_count > 1:
        ignored = "page 2 is" if page_count == 2 else f"pages 2-{page_count} are"
        LOGGER.warning(
            "%s has %d pages; converting page 1 only (%s ignored). "
            "Export a single-page image if that is not what you meant.",
            path,
            page_count,
            ignored,
        )


def load_image(path: Path) -> np.ndarray:
    """Read a TIFF or PNG as one contiguous two-dimensional array.

    A multi-page TIFF is read as its first page (#1738).
    """
    with images.open(path) as reader:
        _warn_extra_pages(path, _image_page_count(reader))
    return _normalize_to_2d(images.read(path, page=0), path)


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


def _create_ome_zarr_datasets(
    output_path: Path,
    *,
    image_shape: tuple[int, int],
    dtype: np.dtype,
    levels: int,
    reduction: str,
    overwrite: bool = False,
) -> list[ZarrWriter]:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    shapes = pyramid_shapes(image_shape, levels)
    group = create_label_group(output_path, levels=len(shapes), reduction=reduction)
    return [
        create_label_array(group, str(level), shape=shape, dtype=np.dtype(dtype))
        for level, shape in enumerate(shapes)
    ]


def write_label_pyramid(
    image_2d: np.ndarray,
    output_path: Path,
    *,
    levels: int = DEFAULT_LEVELS,
    downsample_mode: Literal["nearest", "mean"] = "nearest",
    overwrite: bool = False,
    chunk_workers: int = 1,
) -> None:
    """Embed a label at Z=32 and write it with XY pyramid levels, without Z pooling."""
    if levels < 1:
        raise ValueError("levels must be at least 1")
    level_zero = _embed_label_volume(image_2d)
    datasets = _create_ome_zarr_datasets(
        output_path,
        image_shape=tuple(int(value) for value in level_zero.shape[1:]),
        dtype=level_zero.dtype,
        levels=levels,
        reduction=downsample_mode,
        overwrite=overwrite,
    )
    datasets[0][...] = level_zero
    _write_levels_above_zero(output_path, levels, downsample_mode, chunk_workers)


def _write_levels_above_zero(
    output_path: Path, levels: int, downsample_mode: str, chunk_workers: int
) -> None:
    for level in range(1, levels):
        write_pyramid_level(
            output_path, level, factors=(1, 2, 2), reduction=downsample_mode, threads=chunk_workers
        )


def _get_streamable_tiff_metadata(
    path: Path,
) -> tuple[tuple[int, int], np.dtype] | None:
    """Shape and dtype of a TIFF, which is converted without loading the whole image."""
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    with images.open(path) as reader:
        pages = _image_page_count(reader)
        _warn_extra_pages(path, pages)
        shape = reader.shape[1:] if reader.axes[0] == "z" else reader.shape
        return _normalized_2d_shape(shape, path), np.dtype(reader.dtype)


def _write_streamed_tiff_level_zero(input_path: Path, dataset: ZarrWriter) -> None:
    """Copy a TIFF's first page into ``dataset`` block by block."""
    # Retained strips and tiles keep a block's neighbours from decoding one twice.
    with images.open(input_path, decoded_cache_bytes=STREAM_DECODED_BYTES) as reader:
        stack = reader.axes[0] == "z"
        height, width = _normalized_2d_shape(reader.shape[1:] if stack else reader.shape, input_path)
        channels = reader.axes[-1] == "c"
        for by in range(0, height, STREAM_BLOCK_SIZE):
            for bx in range(0, width, STREAM_BLOCK_SIZE):
                window = (*((0,) if stack else ()), slice(by, by + STREAM_BLOCK_SIZE), slice(bx, bx + STREAM_BLOCK_SIZE), *((0,) if channels else ()))
                block_YX = reader[window]
                dataset[
                    DEFAULT_LABEL_SLICE,
                    by : by + block_YX.shape[0],
                    bx : bx + block_YX.shape[1],
                ] = block_YX


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
        }
    downsample_mode: Literal["nearest", "mean"] = (
        "mean" if is_composite_image(input_path) else "nearest"
    )
    streamable_metadata = _get_streamable_tiff_metadata(input_path)
    if streamable_metadata is not None:
        image_shape, dtype = streamable_metadata
        datasets = _create_ome_zarr_datasets(
            output_path,
            image_shape=image_shape,
            dtype=dtype,
            levels=levels,
            reduction=downsample_mode,
            overwrite=overwrite,
        )
        _write_streamed_tiff_level_zero(input_path, datasets[0])
        _write_levels_above_zero(output_path, levels, downsample_mode, chunk_workers)
    else:
        write_label_pyramid(
            load_image(input_path),
            output_path,
            levels=levels,
            downsample_mode=downsample_mode,
            overwrite=overwrite,
            chunk_workers=chunk_workers,
        )
    return {
        "status": "written",
        "input": str(input_path),
        "output": str(output_path),
        "downsample_mode": downsample_mode,
        "streamed_tiled_tiff": str(streamable_metadata is not None).lower(),
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
