#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import tifffile
from tqdm.auto import tqdm


MATCH_TERMS = [
    "betti",
    "ema",
    "640"
]
MERGED_FILE_PREFIX = "merged_"
DEFAULT_MERGE_METHOD = "confidence_vote"
MERGE_METHOD_CHOICES = ("confidence_vote", "soft_mean", "soft_median")
SUPPORTED_SUFFIXES = {".png", ".tif", ".tiff"}
DEFAULT_MAX_WORKERS = 2
DEFAULT_TARGET_CHUNK_MB = 96
FOREGROUND_THRESHOLD_U8 = 128


@dataclass(frozen=True)
class PreparedPrediction:
    path: Path
    array: np.ndarray
    cleanup_path: Path | None


@dataclass(frozen=True)
class DirectionResult:
    direction: str
    matched_files: int
    output_path: str | None
    skipped_reason: str | None = None


@dataclass(frozen=True)
class FolderResult:
    preds_dir: str
    directions: tuple[DirectionResult, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find preds folders, merge matching prediction files with a "
            "selected aggregation method, and write merged outputs back into each preds folder."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="A folder that is either a preds directory or contains preds directories recursively.",
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "reverse", "both"),
        default="both",
        help="Which prediction directions to merge. Default: both.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Worker processes across preds folders. "
            f"Default: min(CPU count, preds folders, {DEFAULT_MAX_WORKERS})."
        ),
    )
    parser.add_argument(
        "--method",
        choices=MERGE_METHOD_CHOICES,
        default=DEFAULT_MERGE_METHOD,
        help=f"Merge method to use. Default: {DEFAULT_MERGE_METHOD}.",
    )
    return parser.parse_args(argv)


def normalize_terms(terms: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = str(term).strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return tuple(normalized)


def find_preds_folders(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")
    if root.name == "preds":
        return [root]

    matches: list[Path] = []
    for current_root, dirnames, _ in os.walk(root):
        dirnames[:] = sorted(dirnames)
        current_path = Path(current_root)
        if current_path.name == "preds":
            matches.append(current_path)
            dirnames[:] = []
    return sorted(matches)


def parse_prediction_direction(path: Path) -> str | None:
    lower_name = path.stem.lower()
    if "_forward_" in lower_name or lower_name.endswith("_forward"):
        return "forward"
    if "_reverse_" in lower_name or lower_name.endswith("_reverse"):
        return "reverse"
    return None


def is_supported_prediction_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES


def is_merged_output(path: Path) -> bool:
    return path.stem.lower().startswith(MERGED_FILE_PREFIX)


def matches_prediction_terms(path: Path, normalized_terms: Sequence[str]) -> bool:
    if is_merged_output(path):
        return False
    if not normalized_terms:
        return True
    lower_name = path.stem.lower()
    return any(term in lower_name for term in normalized_terms)


def collect_matching_prediction_files(
    preds_dir: Path,
    *,
    normalized_terms: Sequence[str],
    direction: str,
) -> list[Path]:
    matches: list[Path] = []
    for candidate in sorted(preds_dir.iterdir()):
        if not is_supported_prediction_file(candidate):
            continue
        if parse_prediction_direction(candidate) != direction:
            continue
        if not matches_prediction_terms(candidate, normalized_terms):
            continue
        matches.append(candidate)
    return matches


def choose_output_suffix(paths: Sequence[Path]) -> str:
    suffixes = {path.suffix.lower() for path in paths}
    if suffixes == {".png"}:
        return ".png"
    return ".tif"


def build_terms_slug(normalized_terms: Sequence[str]) -> str:
    if not normalized_terms:
        return "all"
    return "_".join(normalized_terms)


def build_output_path(
    preds_dir: Path,
    *,
    normalized_terms: Sequence[str],
    direction: str,
    merge_method: str,
    suffix: str,
) -> Path:
    terms_slug = build_terms_slug(normalized_terms)
    return preds_dir / f"{MERGED_FILE_PREFIX}{merge_method}_{terms_slug}_{direction}{suffix}"


def normalize_prediction_array(array: np.ndarray, *, path: Path) -> np.ndarray:
    normalized = np.asarray(array)
    normalized = np.squeeze(normalized)
    if normalized.ndim == 3 and normalized.shape[-1] in {2, 3, 4}:
        normalized = normalized[..., 0]
    if normalized.ndim != 2:
        raise ValueError(f"Expected a 2D prediction image in {path}, got shape={tuple(normalized.shape)}")
    return normalized


def write_array_to_temp_memmap(array: np.ndarray) -> tuple[np.memmap, Path]:
    handle = tempfile.NamedTemporaryFile(prefix="merge_predictions_", suffix=".memmap", delete=False)
    handle.close()
    temp_path = Path(handle.name)
    memmap_array = np.memmap(temp_path, mode="w+", dtype=array.dtype, shape=array.shape)
    memmap_array[...] = array
    memmap_array.flush()
    return memmap_array, temp_path


def prepare_prediction(path: Path) -> PreparedPrediction:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) == 0:
                raise RuntimeError(f"No TIFF pages found in {path}")
            memmap_array = tif.pages[0].asarray(out="memmap")
        normalized = normalize_prediction_array(memmap_array, path=path)
        cleanup_path = Path(memmap_array.filename)
        return PreparedPrediction(path=path, array=normalized, cleanup_path=cleanup_path)

    with Image.open(path) as image:
        loaded = normalize_prediction_array(np.asarray(image), path=path)
    memmap_array, cleanup_path = write_array_to_temp_memmap(loaded)
    return PreparedPrediction(path=path, array=memmap_array, cleanup_path=cleanup_path)


def cleanup_prepared_predictions(predictions: Sequence[PreparedPrediction]) -> None:
    for prepared in predictions:
        try:
            if isinstance(prepared.array, np.memmap):
                prepared.array.flush()
        except Exception:
            pass
        if prepared.cleanup_path is None:
            continue
        try:
            prepared.cleanup_path.unlink(missing_ok=True)
        except Exception:
            pass


def normalize_chunk_to_u8(chunk: np.ndarray) -> np.ndarray:
    array = np.asarray(chunk)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array))
        min_value = float(np.nanmin(array))
        if min_value >= 0.0 and max_value <= 1.0:
            return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
        return np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def rounded_integer_divide(sum_array: np.ndarray, count_array: np.ndarray | int) -> np.ndarray:
    count_u32 = np.asarray(count_array, dtype=np.uint32)
    result = np.zeros_like(sum_array, dtype=np.uint32)
    np.floor_divide(sum_array + (count_u32 // 2), count_u32, out=result, where=count_u32 != 0)
    return result


def merge_confidence_vote_chunk(chunks: Sequence[np.ndarray]) -> np.ndarray:
    if not chunks:
        raise ValueError("Expected at least one chunk to merge")

    first_chunk = normalize_chunk_to_u8(chunks[0])
    chunk_shape = first_chunk.shape
    sum_all = np.zeros(chunk_shape, dtype=np.uint32)
    positive_weight = np.zeros(chunk_shape, dtype=np.uint32)
    negative_weight = np.zeros(chunk_shape, dtype=np.uint32)
    positive_count = np.zeros(chunk_shape, dtype=np.uint16)

    for chunk in chunks:
        chunk_u8 = normalize_chunk_to_u8(chunk)
        if chunk_u8.shape != chunk_shape:
            raise ValueError(f"Prediction chunk shape mismatch: {chunk_u8.shape} != {chunk_shape}")
        sum_all += chunk_u8.astype(np.uint32, copy=False)
        positive_mask = chunk_u8 >= FOREGROUND_THRESHOLD_U8
        positive_weight += np.where(positive_mask, chunk_u8, 0).astype(np.uint32, copy=False)
        negative_weight += np.where(positive_mask, 0, 255 - chunk_u8).astype(np.uint32, copy=False)
        positive_count += positive_mask.astype(np.uint16, copy=False)

    input_count = len(chunks)
    negative_count = np.uint16(input_count) - positive_count
    tie_mean = rounded_integer_divide(sum_all, input_count).astype(np.uint8, copy=False)
    output = tie_mean.copy()

    positive_wins = positive_weight > negative_weight
    negative_wins = negative_weight > positive_weight

    if np.any(positive_wins):
        positive_mean = rounded_integer_divide(positive_weight, positive_count)
        output[positive_wins] = positive_mean[positive_wins].astype(np.uint8, copy=False)

    if np.any(negative_wins):
        negative_mean = 255 - rounded_integer_divide(negative_weight, negative_count)
        output[negative_wins] = negative_mean[negative_wins].astype(np.uint8, copy=False)

    return output


def merge_soft_mean_chunk(chunks: Sequence[np.ndarray]) -> np.ndarray:
    if not chunks:
        raise ValueError("Expected at least one chunk to merge")

    first_chunk = normalize_chunk_to_u8(chunks[0])
    chunk_shape = first_chunk.shape
    sum_all = np.zeros(chunk_shape, dtype=np.uint32)

    for chunk in chunks:
        chunk_u8 = normalize_chunk_to_u8(chunk)
        if chunk_u8.shape != chunk_shape:
            raise ValueError(f"Prediction chunk shape mismatch: {chunk_u8.shape} != {chunk_shape}")
        sum_all += chunk_u8.astype(np.uint32, copy=False)

    return rounded_integer_divide(sum_all, len(chunks)).astype(np.uint8, copy=False)


def merge_soft_median_chunk(chunks: Sequence[np.ndarray]) -> np.ndarray:
    if not chunks:
        raise ValueError("Expected at least one chunk to merge")

    normalized_chunks: list[np.ndarray] = []
    first_chunk = normalize_chunk_to_u8(chunks[0])
    chunk_shape = first_chunk.shape
    normalized_chunks.append(first_chunk)

    for chunk in chunks[1:]:
        chunk_u8 = normalize_chunk_to_u8(chunk)
        if chunk_u8.shape != chunk_shape:
            raise ValueError(f"Prediction chunk shape mismatch: {chunk_u8.shape} != {chunk_shape}")
        normalized_chunks.append(chunk_u8)

    stacked = np.stack(normalized_chunks, axis=0)
    return np.clip(np.rint(np.median(stacked, axis=0)), 0, 255).astype(np.uint8)


def merge_chunk(chunks: Sequence[np.ndarray], *, merge_method: str) -> np.ndarray:
    if merge_method == "confidence_vote":
        return merge_confidence_vote_chunk(chunks)
    if merge_method == "soft_mean":
        return merge_soft_mean_chunk(chunks)
    if merge_method == "soft_median":
        return merge_soft_median_chunk(chunks)
    raise ValueError(f"Unsupported merge method: {merge_method}")


def choose_row_chunk_size(width: int, input_count: int) -> int:
    bytes_per_pixel = 4 + 4 + 4 + 2 + max(1, input_count) + 1
    target_bytes = DEFAULT_TARGET_CHUNK_MB * 1024 * 1024
    rows = max(64, target_bytes // max(1, width * bytes_per_pixel))
    return int(min(2048, rows))


def validate_prediction_shapes(predictions: Sequence[PreparedPrediction]) -> tuple[int, int]:
    if not predictions:
        raise ValueError("Expected at least one prepared prediction")
    reference_shape = tuple(int(value) for value in predictions[0].array.shape)
    for prepared in predictions[1:]:
        current_shape = tuple(int(value) for value in prepared.array.shape)
        if current_shape != reference_shape:
            raise ValueError(
                f"Prediction shape mismatch in {prepared.path}: {current_shape} != {reference_shape}"
            )
    return reference_shape


def write_prediction_image(output_path: Path, output_array: np.ndarray) -> None:
    if output_path.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(output_path, output_array, compression="lzw")
        return
    Image.fromarray(np.asarray(output_array), mode="L").save(output_path)


def merge_prediction_files(files: Sequence[Path], output_path: Path, *, merge_method: str) -> None:
    prepared_predictions = [prepare_prediction(path) for path in files]
    output_memmap_path: Path | None = None
    output_array: np.memmap | None = None
    try:
        height, width = validate_prediction_shapes(prepared_predictions)
        row_chunk_size = choose_row_chunk_size(width, len(prepared_predictions))
        output_handle = tempfile.NamedTemporaryFile(prefix="merge_predictions_out_", suffix=".memmap", delete=False)
        output_handle.close()
        output_memmap_path = Path(output_handle.name)
        output_array = np.memmap(output_memmap_path, mode="w+", dtype=np.uint8, shape=(height, width))

        for row_start in range(0, height, row_chunk_size):
            row_end = min(height, row_start + row_chunk_size)
            chunks = [prepared.array[row_start:row_end, :] for prepared in prepared_predictions]
            output_array[row_start:row_end, :] = merge_chunk(chunks, merge_method=merge_method)

        output_array.flush()
        write_prediction_image(output_path, output_array)
    finally:
        if output_array is not None:
            try:
                output_array.flush()
            except Exception:
                pass
        cleanup_prepared_predictions(prepared_predictions)
        if output_memmap_path is not None:
            try:
                output_memmap_path.unlink(missing_ok=True)
            except Exception:
                pass


def directions_to_process(direction: str) -> tuple[str, ...]:
    if direction == "both":
        return ("forward", "reverse")
    return (direction,)


def process_preds_folder(
    preds_dir: Path,
    *,
    normalized_terms: Sequence[str],
    requested_direction: str,
    merge_method: str,
) -> FolderResult:
    direction_results: list[DirectionResult] = []
    for direction in directions_to_process(requested_direction):
        matched_files = collect_matching_prediction_files(
            preds_dir,
            normalized_terms=normalized_terms,
            direction=direction,
        )
        if not matched_files:
            direction_results.append(
                DirectionResult(
                    direction=direction,
                    matched_files=0,
                    output_path=None,
                    skipped_reason="no_matches",
                )
            )
            continue

        output_suffix = choose_output_suffix(matched_files)
        output_path = build_output_path(
            preds_dir,
            normalized_terms=normalized_terms,
            direction=direction,
            merge_method=merge_method,
            suffix=output_suffix,
        )
        merge_prediction_files(matched_files, output_path, merge_method=merge_method)
        direction_results.append(
            DirectionResult(
                direction=direction,
                matched_files=len(matched_files),
                output_path=str(output_path),
            )
        )

    return FolderResult(preds_dir=str(preds_dir), directions=tuple(direction_results))


def process_preds_folder_task(task: tuple[str, tuple[str, ...], str, str]) -> FolderResult:
    preds_dir_str, normalized_terms, requested_direction, merge_method = task
    return process_preds_folder(
        Path(preds_dir_str),
        normalized_terms=normalized_terms,
        requested_direction=requested_direction,
        merge_method=merge_method,
    )


def default_worker_count(folder_count: int) -> int:
    if folder_count <= 0:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, folder_count, DEFAULT_MAX_WORKERS))


def summarize_results(results: Sequence[FolderResult]) -> str:
    merged_outputs = 0
    skipped_directions = 0
    for folder_result in results:
        for direction_result in folder_result.directions:
            if direction_result.output_path is not None:
                merged_outputs += 1
            else:
                skipped_directions += 1
    return f"merged_outputs={merged_outputs} skipped_directions={skipped_directions}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    normalized_terms = normalize_terms(MATCH_TERMS)
    preds_folders = find_preds_folders(args.root)
    if not preds_folders:
        raise RuntimeError(f"No preds folders found under {args.root}")

    worker_count = args.workers if args.workers is not None else default_worker_count(len(preds_folders))
    if worker_count < 1:
        raise ValueError("--workers must be at least 1")

    tasks = [(str(preds_dir), normalized_terms, args.direction, args.method) for preds_dir in preds_folders]
    results: list[FolderResult] = []

    if worker_count == 1:
        iterator = (process_preds_folder_task(task) for task in tasks)
        for result in tqdm(iterator, total=len(tasks), desc="Merging preds folders"):
            results.append(result)
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=worker_count) as pool:
            iterator = pool.imap_unordered(process_preds_folder_task, tasks)
            for result in tqdm(iterator, total=len(tasks), desc="Merging preds folders"):
                results.append(result)

    print(summarize_results(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
