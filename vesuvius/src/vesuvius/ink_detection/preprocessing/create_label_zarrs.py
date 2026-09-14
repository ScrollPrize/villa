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
import tifffile
from tqdm.auto import tqdm
import zarr

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


STREAM_BLOCK_SIZE = 1024
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


def _decoded_block_to_2d(decoded: np.ndarray, source_path: Path) -> np.ndarray:
    """Return a decoded tile or strip as 2D without losing a length-1 row axis.

    ``page.decode`` returns ``(depth, rows, columns, samples)`` for a strip
    exactly as it does for a tile. ``_normalize_to_2d`` squeezes, which is
    right for a whole image but drops the row axis of a one-row strip --
    ``(1, 1, width, 1)`` becomes ``(width,)`` -- so index the axes explicitly.
    One-row strips arise when ``rowsperstrip == 1``, or when
    ``height % rowsperstrip == 1`` leaves a single row in the final strip.
    Tiles cannot hit this: TIFF tile heights are multiples of 16.
    """
    block = np.asarray(decoded)
    if block.ndim == 4:
        return np.ascontiguousarray(block[0, :, :, 0])
    return _normalize_to_2d(block, source_path)


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


def _iter_block_slices(
    height: int, width: int, *, block_size: int = STREAM_BLOCK_SIZE
) -> Iterable[tuple[int, int, int, int]]:
    for y_start in range(0, height, block_size):
        block_height = min(block_size, height - y_start)
        for x_start in range(0, width, block_size):
            block_width = min(block_size, width - x_start)
            yield y_start, x_start, block_height, block_width


def _create_ome_zarr_datasets(
    output_path: Path,
    *,
    image_shape: tuple[int, int],
    dtype: np.dtype,
    levels: int,
    overwrite: bool = False,
) -> list[zarr.Array]:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    shapes = pyramid_shapes(image_shape, levels)
    group = create_label_group(output_path, levels=len(shapes))
    datasets = []
    for level, shape in enumerate(shapes):
        dataset = create_label_array(
            group,
            str(level),
            shape=shape,
            dtype=np.dtype(dtype),
        )
        datasets.append(dataset)
    return datasets


def write_ome_zarr(
    pyramid: Sequence[np.ndarray],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write an in-memory ZYX pyramid as an explicit Zarr-v2 OME group."""
    if not pyramid:
        raise ValueError("pyramid must contain at least one level")
    datasets = _create_ome_zarr_datasets(
        output_path,
        image_shape=tuple(int(value) for value in pyramid[0].shape[1:]),
        dtype=pyramid[0].dtype,
        levels=len(pyramid),
        overwrite=overwrite,
    )
    for dataset, array_ZYX in zip(datasets, pyramid):
        dataset[:] = array_ZYX


def _get_streamable_tiff_metadata(
    path: Path,
) -> tuple[tuple[int, int], np.dtype] | None:
    """Whether this TIFF can be converted without loading the whole image.

    tifffile exposes tiled and striped TIFFs through the same block API --
    ``page.chunks``/``page.chunked``/``page.dataoffsets``/``page.databytecounts``
    describe a strip identically to how they describe a tile, just with a
    chunk shape of ``(rows_per_strip, full_width)`` and a chunk grid that is
    ``N`` blocks tall by 1 wide instead of a 2D grid. ``page.decode()`` accepts
    either. Verified directly against tifffile 2026.3.3: for a striped page,
    ``position[2]`` from ``decode()`` is the strip's y-offset, exactly as it is
    the tile's y-offset for a tiled page -- the same indexing in
    ``_write_streamed_tiff_level_zero`` below is correct for both.

    Gating only on ``is_tiled`` (an earlier version of this function) meant
    the only published label images that stream are the ones that happen to
    be tiled; every other TIFF -- including any written with a default
    ``tifffile.imwrite``, which produces strips -- silently falls through to
    ``build_pyramid_with_mode``, which loads the entire volume into memory.
    That is invisible until someone converts a large enough image (a full
    z-stack label volume, not a single flattened image) and hits an OOM with
    no indication that a streamed path exists or why it wasn't used.
    """
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if page.is_tiled:
            # Tiled input streamed before this change, unconditionally. Leave
            # that exactly as it was: every extra condition here is a tiled
            # file that regresses to the in-memory path it was exempt from.
            return _normalized_2d_shape(page.shape, path), np.dtype(page.dtype)

        # Striped input is what this change adds. The two conditions below
        # apply only to it.
        if len(tif.pages) != 1:
            # The rest of this module -- _normalize_to_2d,
            # _normalized_2d_shape, _create_ome_zarr_datasets(image_shape:
            # tuple[int, int]) -- is built for a single flat 2D label image.
            # A genuine multi-page file already produces silently wrong output
            # on the in-memory path today, independent of this change; that is
            # a separate bug and this PR does not touch it.
            return None
        if page.compression not in _STREAMABLE_COMPRESSIONS:
            # Some codecs (notably old-style JPEG, compression 6) need
            # cross-block state tifffile does not expose per-block, so a block
            # cannot be decoded in isolation. Fall through to the in-memory
            # path rather than decode blocks independently for a codec that
            # does not support it.
            return None
        return _normalized_2d_shape(page.shape, path), np.dtype(page.dtype)


# Codecs verified to support independent per-block decode via page.decode():
# none (1), LZW (5), old-style Deflate (32946) and Deflate/zlib (8), PackBits
# (32773). Excludes old-JPEG (6) and other codecs with cross-block state.
_STREAMABLE_COMPRESSIONS = frozenset({1, 5, 8, 32773, 32946})


def _write_streamed_tiff_level_zero(input_path: Path, dataset: zarr.Array) -> None:
    """Stream a tiled OR striped TIFF's level-0 data into ``dataset``.

    Reads ``page.chunks``/``page.chunked`` generically -- for a tiled page
    these describe the tile grid; for a striped page they describe strips as
    a chunk grid one column wide (verified: ``page.chunks == (rows_per_strip,
    full_width)``, ``page.chunked == (n_strips, 1)``). The block-addressing
    math below never distinguishes the two cases, because tifffile does not
    either at this level of its API.
    """
    with tifffile.TiffFile(input_path) as tif:
        page = tif.pages[0]
        if not page.is_tiled and page.compression not in _STREAMABLE_COMPRESSIONS:
            raise ValueError(
                f"Expected a streamable TIFF (tiled or striped, compression in "
                f"{sorted(_STREAMABLE_COMPRESSIONS)}) for the streaming path: "
                f"{input_path} has compression {page.compression}"
            )
        image_height, image_width = _normalized_2d_shape(page.shape, input_path)
        tile_height, tile_width = page.chunks
        tiles_down, tiles_across = page.chunked

        # Destination blocks, keyed by (block_y, block_x). Each entry holds the
        # partially-filled array and a count of source chunks still to come. A
        # block is held only while chunks that can contribute to it remain
        # unread; since chunks are visited in row-major order, at most one row
        # of destination blocks is resident at a time.
        blocks: dict[tuple[int, int], np.ndarray] = {}
        pending: dict[tuple[int, int], int] = {}

        def _block_span(start: int, length: int, extent: int) -> range:
            """Destination block origins covering [start, start+length)."""
            first = (start // STREAM_BLOCK_SIZE) * STREAM_BLOCK_SIZE
            stop = min(start + length, extent)
            return range(first, stop, STREAM_BLOCK_SIZE)

        # Count contributions per destination block up front, from the chunk
        # grid alone -- no decoding required. A block is complete, and can be
        # written and released, once its count reaches zero.
        for tile_row in range(tiles_down):
            tile_y = tile_row * tile_height
            for tile_col in range(tiles_across):
                tile_x = tile_col * tile_width
                for by in _block_span(tile_y, tile_height, image_height):
                    for bx in _block_span(tile_x, tile_width, image_width):
                        pending[(by, bx)] = pending.get((by, bx), 0) + 1

        itemsize = np.dtype(page.dtype).itemsize

        def _flush(key: tuple[int, int]) -> None:
            block_YX = blocks.pop(key)
            by, bx = key
            dataset[
                DEFAULT_LABEL_SLICE,
                by : by + block_YX.shape[0],
                bx : bx + block_YX.shape[1],
            ] = block_YX

        # Source-chunk-major: decode each chunk exactly once.
        for tile_index in range(tiles_down * tiles_across):
            if tile_index >= len(page.dataoffsets):
                continue
            bytecount = page.databytecounts[tile_index]
            if bytecount == 0:
                # Sparse TIFFs may record empty chunks; nothing to scatter, but
                # the pending counts still have to be settled.
                decoded = None
            else:
                tif.filehandle.seek(page.dataoffsets[tile_index])
                data = tif.filehandle.read(bytecount)
                decoded, position, _ = page.decode(
                    data, tile_index, jpegtables=page.jpegtables
                )

            if decoded is None:
                tile_row, tile_col = divmod(tile_index, tiles_across)
                tile_y, tile_x = tile_row * tile_height, tile_col * tile_width
                tile_h, tile_w = tile_height, tile_width
                tile_YX = None
            else:
                tile_YX = _decoded_block_to_2d(decoded, input_path)
                tile_y, tile_x = position[2], position[3]
                tile_h, tile_w = tile_YX.shape

            for by in _block_span(tile_y, tile_h, image_height):
                bh = min(STREAM_BLOCK_SIZE, image_height - by)
                for bx in _block_span(tile_x, tile_w, image_width):
                    bw = min(STREAM_BLOCK_SIZE, image_width - bx)
                    key = (by, bx)
                    if key not in blocks:
                        blocks[key] = np.zeros((bh, bw), dtype=page.dtype)
                    if tile_YX is not None:
                        y0 = max(by, tile_y)
                        y1 = min(by + bh, tile_y + tile_h)
                        x0 = max(bx, tile_x)
                        x1 = min(bx + bw, tile_x + tile_w)
                        if y0 < y1 and x0 < x1:
                            blocks[key][
                                y0 - by : y1 - by, x0 - bx : x1 - bx
                            ] = tile_YX[
                                y0 - tile_y : y1 - tile_y, x0 - tile_x : x1 - tile_x
                            ]
                    pending[key] -= 1
                    if pending[key] == 0:
                        _flush(key)
                        del pending[key]

        # Any block whose contributions were all skipped (missing offsets in a
        # truncated file) is still zero-filled and must be written.
        for key in sorted(blocks):
            _flush(key)


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
            overwrite=overwrite,
        )
        _write_streamed_tiff_level_zero(input_path, datasets[0])
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
