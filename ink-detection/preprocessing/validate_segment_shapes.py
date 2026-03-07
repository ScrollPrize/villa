#!/usr/bin/env python3
"""Validate segment TIFF shapes against the spatial shape of each segment Zarr."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
from tqdm.auto import tqdm
import zarr


TIFF_SUFFIXES = {".tif", ".tiff"}
LABEL_NAME_TOKENS = ("_inklabels", "_supervision_mask")
ALLOWED_BINARY_LABEL_VALUES = frozenset({0, 1, 255})


@dataclass(frozen=True, slots=True)
class ShapeIssue:
    file_path: Path
    message: str


@dataclass(frozen=True, slots=True)
class SegmentResult:
    segment_dir: Path
    expected_shape: tuple[int, int] | None
    issues: tuple[ShapeIssue, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan scroll/segment folders and verify that each segment's .zarr spatial "
            "shape matches its relevant TIFF files, and that label TIFFs are binary "
            "without alpha channels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Parent folder containing scroll directories, each with segment subdirectories.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
        help="Number of worker threads to use while reading metadata.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress reporting.",
    )
    return parser.parse_args(argv)


def discover_segment_dirs(root: Path) -> list[Path]:
    segments: list[Path] = []
    for scroll_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for segment_dir in sorted(path for path in scroll_dir.iterdir() if path.is_dir()):
            if segment_dir.suffix.lower() == ".zarr":
                continue
            segments.append(segment_dir)
    return segments


def _is_target_tiff(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in TIFF_SUFFIXES:
        return False

    name = path.name.lower()
    return (
        "_inklabels" in name
        or "_supervision_mask" in name
        or "max" in name
        or "composite" in name
    )


def iter_target_tiffs(segment_dir: Path) -> Iterable[Path]:
    for path in sorted(segment_dir.iterdir()):
        if _is_target_tiff(path):
            yield path


def _is_label_tiff(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in LABEL_NAME_TOKENS)


def _normalize_zarr_shape(shape: Sequence[int], source_path: Path) -> tuple[int, int]:
    dims = tuple(int(dimension) for dimension in shape if int(dimension) != 1)
    if len(dims) < 2:
        raise ValueError(f"Expected at least 2 non-singleton dimensions in {source_path}, got {tuple(shape)}")
    return dims[-2], dims[-1]


def _normalize_tiff_shape(shape: Sequence[int], source_path: Path) -> tuple[int, int]:
    dims = tuple(int(dimension) for dimension in shape if int(dimension) != 1)

    if len(dims) == 2:
        return dims[0], dims[1]

    if len(dims) == 3:
        if dims[-1] <= 4 and dims[0] > 4 and dims[1] > 4:
            return dims[0], dims[1]
        if dims[0] <= 4 and dims[1] > 4 and dims[2] > 4:
            return dims[1], dims[2]
        return dims[-2], dims[-1]

    raise ValueError(f"Expected a 2D TIFF shape in {source_path}, got {tuple(shape)}")


def _read_tiff_payload(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    with tifffile.TiffFile(path) as tif:
        if tif.series:
            series = tif.series[0]
            return tuple(int(dimension) for dimension in series.shape), np.asarray(series.asarray())
        if tif.pages:
            page = tif.pages[0]
            return tuple(int(dimension) for dimension in page.shape), np.asarray(page.asarray())
        raise RuntimeError(f"No TIFF image data found in {path}")


def read_tiff_shape(path: Path) -> tuple[int, int]:
    shape, _ = _read_tiff_payload(path)
    return _normalize_tiff_shape(shape, path)


def _label_channel_axis(shape: Sequence[int]) -> int | None:
    dims = tuple(int(dimension) for dimension in shape)
    if len(dims) != 3:
        return None
    if dims[-1] <= 4 and dims[0] > 4 and dims[1] > 4:
        return 2
    if dims[0] <= 4 and dims[1] > 4 and dims[2] > 4:
        return 0
    return None


def _label_has_alpha_channel(image: np.ndarray) -> bool:
    squeezed = np.squeeze(np.asarray(image))
    channel_axis = _label_channel_axis(squeezed.shape)
    if channel_axis is None:
        return False
    return int(squeezed.shape[channel_axis]) in {2, 4}


def _normalize_label_image(image: np.ndarray, source_path: Path) -> np.ndarray:
    squeezed = np.squeeze(np.asarray(image))

    if squeezed.ndim == 2:
        return np.ascontiguousarray(squeezed)
    if squeezed.ndim != 3:
        raise ValueError(f"Expected a 2D or channelized label TIFF in {source_path}, got shape={tuple(squeezed.shape)}")

    channel_axis = _label_channel_axis(squeezed.shape)
    if channel_axis is None:
        raise ValueError(
            f"Could not determine the channel axis for label TIFF {source_path} with shape={tuple(squeezed.shape)}"
        )

    channel_count = int(squeezed.shape[channel_axis])
    if channel_axis == 2:
        color = squeezed[..., :-1] if channel_count in {2, 4} else squeezed
        return np.ascontiguousarray(np.max(color, axis=2))

    color = squeezed[:-1, ...] if channel_count in {2, 4} else squeezed
    return np.ascontiguousarray(np.max(color, axis=0))


def validate_label_tiff(path: Path) -> tuple[str, ...]:
    issues: list[str] = []
    _, image = _read_tiff_payload(path)

    if _label_has_alpha_channel(image):
        issues.append("label image has an alpha channel")

    normalized = _normalize_label_image(image, path)
    unique_values = np.unique(normalized)
    invalid_values = [int(value) for value in unique_values.tolist() if int(value) not in ALLOWED_BINARY_LABEL_VALUES]
    if invalid_values:
        preview = ", ".join(str(value) for value in invalid_values[:8])
        if len(invalid_values) > 8:
            preview += ", ..."
        issues.append(f"label values are not binary; found non-binary values: {preview}")

    return tuple(issues)


def _resolve_zarr_array(zarr_path: Path) -> zarr.Array:
    root = zarr.open(str(zarr_path), mode="r")

    if isinstance(root, zarr.Array):
        return root

    if isinstance(root, zarr.Group):
        if "0" in root:
            array = root["0"]
            if isinstance(array, zarr.Array):
                return array

        array_keys = sorted(str(key) for key in root.array_keys())
        if len(array_keys) == 1:
            array = root[array_keys[0]]
            if isinstance(array, zarr.Array):
                return array

        raise ValueError(
            f"Could not determine which array to validate in {zarr_path}. "
            f"Available array keys: {array_keys}"
        )

    raise ValueError(f"Unsupported Zarr object at {zarr_path}: {type(root)!r}")


def read_zarr_shape(path: Path) -> tuple[int, int]:
    array = _resolve_zarr_array(path)
    return _normalize_zarr_shape(array.shape, path)


def check_segment(segment_dir: Path) -> SegmentResult:
    issues: list[ShapeIssue] = []
    zarr_path = segment_dir / f"{segment_dir.name}.zarr"
    if not zarr_path.is_dir():
        issues.append(
            ShapeIssue(
                zarr_path,
                f"Expected segment zarr directory is missing: {zarr_path.name}",
            )
        )
        return SegmentResult(segment_dir=segment_dir, expected_shape=None, issues=tuple(issues))
    try:
        expected_shape = read_zarr_shape(zarr_path)
    except Exception as exc:
        issues.append(ShapeIssue(zarr_path, f"Failed to read Zarr shape: {exc}"))
        return SegmentResult(segment_dir=segment_dir, expected_shape=None, issues=tuple(issues))

    for tiff_path in iter_target_tiffs(segment_dir):
        try:
            tiff_shape = read_tiff_shape(tiff_path)
        except Exception as exc:
            issues.append(ShapeIssue(tiff_path, f"Failed to read TIFF shape: {exc}"))
            continue

        if tiff_shape != expected_shape:
            issues.append(
                ShapeIssue(
                    tiff_path,
                    f"shape {tiff_shape[0]}x{tiff_shape[1]} does not match zarr {expected_shape[0]}x{expected_shape[1]}",
                )
            )

        if _is_label_tiff(tiff_path):
            try:
                for message in validate_label_tiff(tiff_path):
                    issues.append(ShapeIssue(tiff_path, message))
            except Exception as exc:
                issues.append(ShapeIssue(tiff_path, f"Failed to validate label contents: {exc}"))

    return SegmentResult(segment_dir=segment_dir, expected_shape=expected_shape, issues=tuple(issues))


def validate_root(root: Path, *, workers: int, show_progress: bool = True) -> list[SegmentResult]:
    segment_dirs = discover_segment_dirs(root)
    if not segment_dirs:
        raise FileNotFoundError(f"No segment directories found under {root}")

    worker_count = max(1, min(workers, len(segment_dirs)))
    results: list[SegmentResult] = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_segment = {executor.submit(check_segment, segment_dir): segment_dir for segment_dir in segment_dirs}
        progress = tqdm(
            total=len(segment_dirs),
            desc="Validating segments",
            unit="segment",
            disable=not show_progress,
        )
        try:
            for future in as_completed(future_to_segment):
                segment_dir = future_to_segment[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = SegmentResult(
                        segment_dir=segment_dir,
                        expected_shape=None,
                        issues=(ShapeIssue(segment_dir, f"Unhandled worker failure: {exc}"),),
                    )
                results.append(result)
                progress.update(1)
        finally:
            progress.close()

    results.sort(key=lambda result: result.segment_dir.as_posix())
    return results


def _format_result(root: Path, result: SegmentResult) -> str:
    segment_label = result.segment_dir.relative_to(root).as_posix()
    lines = [f"[{segment_label}]"]

    if result.expected_shape is not None:
        lines.append(f"  zarr shape: {result.expected_shape[0]}x{result.expected_shape[1]}")

    for issue in result.issues:
        target = issue.file_path.relative_to(result.segment_dir) if issue.file_path != result.segment_dir else Path(".")
        lines.append(f"  {target.as_posix()}: {issue.message}")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()

    if args.workers <= 0:
        raise SystemExit("--workers must be at least 1")
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")

    results = validate_root(root, workers=args.workers, show_progress=not args.no_progress)
    bad_results = [result for result in results if result.issues]
    total_issues = sum(len(result.issues) for result in bad_results)

    print(f"Checked {len(results)} segment(s) under {root}")

    if not bad_results:
        print(
            "All relevant TIFF files match their segment .zarr spatial shape, and label TIFFs are binary without alpha channels."
        )
        return 0

    print(f"Found {total_issues} issue(s) across {len(bad_results)} segment(s):")
    for result in bad_results:
        print()
        print(_format_result(root, result))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
