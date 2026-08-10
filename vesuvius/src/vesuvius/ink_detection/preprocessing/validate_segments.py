#!/usr/bin/env python3
"""Validate segment TIFF shapes against the spatial shape of each segment Zarr."""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import tifffile
from tqdm.auto import tqdm
import zarr


TIFF_SUFFIXES = {".tif", ".tiff"}
LABEL_TIFF_NAME_RE = re.compile(
    r"^(?P<prefix>.*)_(?P<label_kind>inklabels|supervision_mask|validation_mask)"
    r"(?:_v(?P<version_num>\d+))?(?P<extension>\.tiff?)$",
    re.IGNORECASE,
)
REQUIRED_LABEL_KINDS = ("inklabels", "supervision_mask")
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


@dataclass(slots=True)
class SegmentState:
    segment_dir: Path
    zarr_path: Path
    expected_shape: tuple[int, int] | None = None
    target_tiffs: tuple[Path, ...] = ()
    issues: list[ShapeIssue] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TiffMetadata:
    file_path: Path
    raw_shape: tuple[int, ...]
    spatial_shape: tuple[int, int]


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
        help="Worker count for metadata checks and per-image tiled label validation.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress reporting.",
    )
    parser.add_argument(
        "--version-id",
        type=str,
        default=None,
        help=(
            "Restrict label TIFF validation to a specific label version, such as 'v2' or 'base'. "
            "Non-label TIFFs like max/composite are still checked."
        ),
    )
    return parser.parse_args(argv)


def _normalize_version_id(version_id: str | int | None) -> int | None:
    if version_id in (None, ""):
        return None
    if isinstance(version_id, str):
        value = version_id.strip().lower()
        if value in {"", "all", "auto", "latest"}:
            return None
        if value in {"base", "unversioned", "v1"}:
            return 1
        if value.startswith("v") and value[1:].isdigit():
            value = value[1:]
        if value.isdigit():
            version_num = int(value)
            if version_num >= 1:
                return version_num
    raise ValueError(
        f"--version-id must be one of None/'all', 'base', or 'vN' with N >= 1; got {version_id!r}"
    )


def _format_version_id(version_num: int) -> str:
    return "base" if int(version_num) <= 1 else f"v{int(version_num)}"


def _parse_label_tiff(path: Path) -> tuple[str, int] | None:
    match = LABEL_TIFF_NAME_RE.match(path.name)
    if match is None:
        return None
    version_num_raw = match.group("version_num")
    version_num = 1 if version_num_raw is None else int(version_num_raw)
    return match.group("label_kind").lower(), version_num


def discover_segment_dirs(root: Path) -> list[Path]:
    segments: list[Path] = []
    for scroll_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for segment_dir in sorted(path for path in scroll_dir.iterdir() if path.is_dir()):
            if segment_dir.suffix.lower() == ".zarr":
                continue
            segments.append(segment_dir)
    return segments


def _is_target_tiff(path: Path, *, label_version: int | None = None) -> bool:
    if not path.is_file() or path.suffix.lower() not in TIFF_SUFFIXES:
        return False

    parsed_label = _parse_label_tiff(path)
    if parsed_label is not None:
        _, version_num = parsed_label
        return label_version is None or version_num == label_version

    name = path.name.lower()
    return "max" in name or "composite" in name


def iter_target_tiffs(segment_dir: Path, *, label_version: int | None = None) -> Iterable[Path]:
    for path in sorted(segment_dir.iterdir()):
        if _is_target_tiff(path, label_version=label_version):
            yield path


def _is_label_tiff(path: Path) -> bool:
    return _parse_label_tiff(path) is not None


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


def _read_tiff_series_shape(path: Path) -> tuple[int, ...]:
    with tifffile.TiffFile(path) as tif:
        if tif.series:
            return tuple(int(dimension) for dimension in tif.series[0].shape)
        if tif.pages:
            return tuple(int(dimension) for dimension in tif.pages[0].shape)
        raise RuntimeError(f"No TIFF image data found in {path}")


def read_tiff_metadata(path: Path) -> TiffMetadata:
    raw_shape = _read_tiff_series_shape(path)
    return TiffMetadata(
        file_path=path,
        raw_shape=raw_shape,
        spatial_shape=_normalize_tiff_shape(raw_shape, path),
    )


def read_tiff_shape(path: Path) -> tuple[int, int]:
    return read_tiff_metadata(path).spatial_shape


def _label_channel_axis(shape: Sequence[int]) -> int | None:
    dims = tuple(int(dimension) for dimension in shape)
    if len(dims) != 3:
        return None
    if dims[-1] <= 4 and dims[0] > 4 and dims[1] > 4:
        return 2
    if dims[0] <= 4 and dims[1] > 4 and dims[2] > 4:
        return 0
    return None


def _label_has_alpha_channel(shape: Sequence[int]) -> bool:
    squeezed = tuple(int(dimension) for dimension in shape if int(dimension) != 1)
    channel_axis = _label_channel_axis(squeezed)
    if channel_axis is None:
        return False
    return int(squeezed[channel_axis]) in {2, 4}


def _normalize_label_tile(tile: np.ndarray, source_path: Path) -> np.ndarray:
    squeezed = np.squeeze(np.asarray(tile))

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


def _preview_invalid_values(values: set[int]) -> str:
    ordered = sorted(values)
    preview = ", ".join(str(value) for value in ordered[:8])
    if len(ordered) > 8:
        preview += ", ..."
    return preview


def _iter_label_tiles(path: Path, *, workers: int) -> Iterator[np.ndarray]:
    with tifffile.TiffFile(path) as tif:
        if tif.series:
            page = tif.series[0].pages[0]
        elif tif.pages:
            page = tif.pages[0]
        else:
            raise RuntimeError(f"No TIFF image data found in {path}")

        for tile, _, _ in page.segments(maxworkers=workers):
            if tile is None:
                continue
            yield tile


def validate_label_tiff(path: Path, *, workers: int, raw_shape: Sequence[int] | None = None) -> tuple[str, ...]:
    issues: list[str] = []

    if raw_shape is None:
        raw_shape = _read_tiff_series_shape(path)
    if _label_has_alpha_channel(raw_shape):
        issues.append("label image has an alpha channel")

    invalid_values: set[int] = set()
    for tile in _iter_label_tiles(path, workers=workers):
        normalized = _normalize_label_tile(tile, path)
        tile_values = np.unique(normalized)
        invalid_values.update(
            int(value)
            for value in tile_values.tolist()
            if int(value) not in ALLOWED_BINARY_LABEL_VALUES
        )
        if len(invalid_values) > 8:
            break

    if invalid_values:
        issues.append(
            "label values are not binary; found non-binary values: "
            f"{_preview_invalid_values(invalid_values)}"
        )

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


def _run_parallel_map(
    items: Sequence[object],
    *,
    workers: int,
    fn,
    desc: str,
    unit: str,
    show_progress: bool,
) -> Iterator[tuple[object, object | Exception]]:
    if not items:
        return

    worker_count = max(1, min(workers, len(items)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_item = {executor.submit(fn, item): item for item in items}
        progress = tqdm(total=len(items), desc=desc, unit=unit, disable=not show_progress)
        try:
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    yield item, future.result()
                except Exception as exc:
                    yield item, exc
                finally:
                    progress.update(1)
        finally:
            progress.close()


def validate_root(
    root: Path,
    *,
    workers: int,
    show_progress: bool = True,
    label_version: int | None = None,
) -> list[SegmentResult]:
    segment_dirs = discover_segment_dirs(root)
    if not segment_dirs:
        raise FileNotFoundError(f"No segment directories found under {root}")

    states: dict[Path, SegmentState] = {}
    for segment_dir in segment_dirs:
        zarr_path = segment_dir / f"{segment_dir.name}.zarr"
        state = SegmentState(segment_dir=segment_dir, zarr_path=zarr_path)
        states[segment_dir] = state
        if not zarr_path.is_dir():
            state.issues.append(
                ShapeIssue(
                    zarr_path,
                    f"Expected segment zarr directory is missing: {zarr_path.name}",
                )
            )

    zarr_ready = [state for state in states.values() if state.zarr_path.is_dir()]
    for item, result in _run_parallel_map(
        zarr_ready,
        workers=workers,
        fn=lambda state: read_zarr_shape(state.zarr_path),
        desc="Zarr metadata",
        unit="segment",
        show_progress=show_progress,
    ):
        state = item
        if isinstance(result, Exception):
            state.issues.append(ShapeIssue(state.zarr_path, f"Failed to read Zarr shape: {result}"))
            continue
        state.expected_shape = result

    tiff_items: list[tuple[SegmentState, Path]] = []
    label_stage_items: list[tuple[SegmentState, Path, tuple[int, ...]]] = []
    version_suffix = f" for version {_format_version_id(label_version)}" if label_version is not None else ""
    for state in states.values():
        state.target_tiffs = tuple(iter_target_tiffs(state.segment_dir, label_version=label_version))
        present_label_kinds = {
            parsed_label[0]
            for tiff_path in state.target_tiffs
            for parsed_label in [_parse_label_tiff(tiff_path)]
            if parsed_label is not None
        }
        for label_kind in REQUIRED_LABEL_KINDS:
            if label_kind not in present_label_kinds:
                state.issues.append(
                    ShapeIssue(
                        state.segment_dir,
                        f"missing required label TIFF{version_suffix} containing '{label_kind}'",
                    )
                )
        for tiff_path in state.target_tiffs:
            tiff_items.append((state, tiff_path))

    for item, result in _run_parallel_map(
        tiff_items,
        workers=workers,
        fn=lambda item: read_tiff_metadata(item[1]),
        desc="TIFF metadata",
        unit="file",
        show_progress=show_progress,
    ):
        state, tiff_path = item
        if isinstance(result, Exception):
            state.issues.append(ShapeIssue(tiff_path, f"Failed to read TIFF shape: {result}"))
            continue
        metadata = result
        if state.expected_shape is None:
            continue
        if metadata.spatial_shape != state.expected_shape:
            state.issues.append(
                ShapeIssue(
                    tiff_path,
                    "shape "
                    f"{metadata.spatial_shape[0]}x{metadata.spatial_shape[1]} does not match zarr "
                    f"{state.expected_shape[0]}x{state.expected_shape[1]}",
                )
            )
        if _is_label_tiff(tiff_path) and _label_has_alpha_channel(metadata.raw_shape):
            state.issues.append(ShapeIssue(tiff_path, "label image has an alpha channel"))
        if _is_label_tiff(tiff_path):
            label_stage_items.append((state, tiff_path, metadata.raw_shape))

    if any(state.issues for state in states.values()):
        return [
            SegmentResult(
                segment_dir=state.segment_dir,
                expected_shape=state.expected_shape,
                issues=tuple(state.issues),
            )
            for state in sorted(states.values(), key=lambda value: value.segment_dir.as_posix())
        ]

    progress = tqdm(total=len(label_stage_items), desc="Label contents", unit="file", disable=not show_progress)
    try:
        for state, label_path, raw_shape in label_stage_items:
            try:
                messages = [
                    message
                    for message in validate_label_tiff(label_path, workers=workers, raw_shape=raw_shape)
                    if message != "label image has an alpha channel"
                ]
                for message in messages:
                    state.issues.append(ShapeIssue(label_path, message))
            except Exception as exc:
                state.issues.append(ShapeIssue(label_path, f"Failed to validate label contents: {exc}"))
            finally:
                progress.update(1)
    finally:
        progress.close()

    results = [
        SegmentResult(
            segment_dir=state.segment_dir,
            expected_shape=state.expected_shape,
            issues=tuple(state.issues),
        )
        for state in sorted(states.values(), key=lambda value: value.segment_dir.as_posix())
    ]
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

    try:
        label_version = _normalize_version_id(args.version_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    results = validate_root(
        root,
        workers=args.workers,
        show_progress=not args.no_progress,
        label_version=label_version,
    )
    bad_results = [result for result in results if result.issues]
    total_issues = sum(len(result.issues) for result in bad_results)

    version_suffix = f" using label version {_format_version_id(label_version)}" if label_version is not None else ""
    print(f"Checked {len(results)} segment(s) under {root}{version_suffix}")

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
