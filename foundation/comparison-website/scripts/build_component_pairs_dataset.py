#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count, get_context
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


@dataclass(frozen=True)
class Component:
    index: int
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float


@dataclass(frozen=True)
class ImageAnalysis:
    path: Path
    shape: tuple[int, int]
    components: list[Component]


def parse_folder_name(name: str) -> tuple[str, str]:
    """Map source directory names like ``l_2_line_04`` to `(fold, sample)`."""
    if "_line_" in name:
        fold, sample = name.split("_line_", 1)
        return fold, f"line_{sample}"
    if "_sample_" in name:
        fold, sample = name.split("_sample_", 1)
        return fold, f"sample_{sample}"

    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Cannot parse fold/sample from source folder name: {name}")


def collect_image_paths(sample_dir: Path) -> list[Path]:
    return [
        file
        for file in sorted(sample_dir.iterdir())
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")

    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise RuntimeError(f"Unsupported image shape for {path}: {image.shape}")


def extract_components(mask: np.ndarray, min_area: int) -> list[Component]:
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components: list[Component] = []
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue
        cx, cy = centroids[idx]
        components.append(
            Component(
                index=int(idx),
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=int(area),
                cx=float(cx),
                cy=float(cy),
            )
        )
    return components


def analyze_source_image(path: Path, min_area: int) -> ImageAnalysis:
    gray = read_grayscale(path)
    mask = (gray > 0).astype(np.uint8)
    components = extract_components(mask, min_area)
    return ImageAnalysis(path=path, shape=gray.shape[:2], components=components)


def is_scroll4_source(path: Path) -> bool:
    normalized = path.as_posix().lower()
    return (
        "s4_archive_images" in normalized
        or "/scroll4/" in normalized
        or normalized.endswith("/scroll4")
    )


def _worker_init() -> None:
    # Avoid nested threading (OpenCV defaults can oversubscribe workers in
    # process-parallel workloads).
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def bbox_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    if inter_w == 0 or inter_h == 0:
        return 0.0

    inter_area = inter_w * inter_h
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def component_similarity(
    left: Component,
    right: Component,
    left_shape: tuple[int, int],
    right_shape: tuple[int, int],
) -> float:
    left_bbox = (left.x, left.y, left.w, left.h)
    right_bbox = (right.x, right.y, right.w, right.h)

    overlap = bbox_overlap_ratio(left_bbox, right_bbox)
    area_ratio = min(left.area, right.area) / max(left.area, right.area)

    # Use a normalized center-distance term for comparable scales.
    scale_x = max(1.0, float(max(left_shape[1], right_shape[1])))
    scale_y = max(1.0, float(max(left_shape[0], right_shape[0])))
    dx = abs(left.cx - right.cx) / scale_x
    dy = abs(left.cy - right.cy) / scale_y
    center_distance = math.hypot(dx, dy)
    center_score = max(0.0, 1.0 - center_distance)

    return 0.45 * overlap + 0.35 * center_score + 0.20 * area_ratio


def match_components(
    left_components: list[Component],
    right_components: list[Component],
    left_shape: tuple[int, int],
    right_shape: tuple[int, int],
    threshold: float,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []

    for li, left_comp in enumerate(left_components):
        for ri, right_comp in enumerate(right_components):
            score = component_similarity(left_comp, right_comp, left_shape, right_shape)
            if score >= threshold:
                candidates.append((score, li, ri))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for score, li, ri in candidates:
        if li in used_left or ri in used_right:
            continue
        used_left.add(li)
        used_right.add(ri)
        matches.append((li, ri, score))

    return matches


def split_image_horiz_half(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    if width < 2:
        return np.zeros((height, 0), dtype=image.dtype), np.zeros((height, 0), dtype=image.dtype)

    mid = max(1, width // 2)
    return image[:, :mid], image[:, mid:]


def split_image_four_parts(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    if height < 1 or width < 4:
        return (
            np.zeros((0, 0), dtype=image.dtype),
            np.zeros((0, 0), dtype=image.dtype),
            np.zeros((0, 0), dtype=image.dtype),
            np.zeros((0, 0), dtype=image.dtype),
        )

    base_width, remainder = divmod(width, 4)
    left_edge = 0
    right_edge = 0
    parts = []

    for part_idx in range(4):
        part_width = base_width + (1 if part_idx < remainder else 0)

        if part_idx == 3:
            right_edge = width
        else:
            right_edge = left_edge + part_width

        parts.append(image[:, left_edge:right_edge])
        left_edge = right_edge

    # guard against any pathological tiny-width input where rounding may under-fill
    if left_edge < width:
        parts[-1] = image[:, left_edge:width]

    return (
        parts[0],
        parts[1],
        parts[2],
        parts[3],
    )


def expand_bbox(component: Component, shape: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    h, w = shape
    x1 = max(0, component.x - margin)
    y1 = max(0, component.y - margin)
    x2 = min(w, component.x + component.w + margin)
    y2 = min(h, component.y + component.h + margin)
    return x1, y1, x2, y2


def canonical_pair_id(fold: str, sample: str, left_rel: str, right_rel: str) -> str:
    parts = sorted([left_rel, right_rel])
    token = f"{fold}|{sample}|{parts[0]}|{parts[1]}"
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]


def load_existing_manifest(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()

    rows: list[dict[str, Any]] = []
    pair_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            rows.append(payload)
            pair_id = payload.get("pair_id")
            if isinstance(pair_id, str):
                pair_ids.add(pair_id)

    return rows, pair_ids


def process_source_folder(task: tuple) -> tuple[dict[str, Any], ...]:
    (
        source_dir_s,
        output_root_s,
        fold,
        sample,
        min_area,
        margin,
        threshold,
        existing_pair_ids,
        adjacency_only,
        crop_mode,
        overwrite,
        dry_run,
    ) = task

    source_dir = Path(source_dir_s)
    output_root = Path(output_root_s)
    source_images = collect_image_paths(source_dir)
    if len(source_images) < 2:
        return (fold, sample, [], 0, 0)

    analyses: list[ImageAnalysis] = []
    if crop_mode == "components":
        analyses = [analyze_source_image(image_path, min_area) for image_path in source_images]

    # For this implementation, adjacency pairing is deterministic and default behavior.
    pair_indices = list(range(len(source_images) - 1))
    if not adjacency_only:
        # Kept for forward compatibility: same as adjacency default in current dataset.
        pair_indices = list(range(len(source_images) - 1))

    sample_dir = output_root / fold / sample
    existing_set = set(existing_pair_ids)

    new_entries: list[dict[str, Any]] = []
    skipped = 0
    generated = 0

    for image_index in pair_indices:
        left_source = source_images[image_index]
        right_source = source_images[image_index + 1]
        if crop_mode == "half":
            left_arr = read_grayscale(left_source)
            right_arr = read_grayscale(right_source)
            if left_arr.size == 0 or right_arr.size == 0:
                continue

            left_halves = split_image_horiz_half(left_arr)
            right_halves = split_image_horiz_half(right_arr)
            for side_idx, (left_half, right_half) in enumerate(
                zip(left_halves, right_halves)
            ):
                if left_half.size == 0 or right_half.size == 0:
                    continue

                side_name = "left" if side_idx == 0 else "right"
                left_temp = f"{left_source.stem}_{right_source.stem}_{side_name}.png"
                right_temp = f"{left_source.stem}_{right_source.stem}_{side_name}_pair.png"

                pair_id = canonical_pair_id(
                    fold,
                    sample,
                    f"{fold}/{sample}/{left_temp}",
                    f"{fold}/{sample}/{right_temp}",
                )

                if pair_id in existing_set and not overwrite:
                    skipped += 1
                    continue

                left_filename = f"{pair_id}_left.png"
                right_filename = f"{pair_id}_right.png"
                left_box = (0, 0, left_half.shape[1], left_half.shape[0])
                right_box = (0, 0, right_half.shape[1], right_half.shape[0])

                if not dry_run:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    left_target = sample_dir / left_filename
                    right_target = sample_dir / right_filename
                    if overwrite and left_target.exists():
                        left_target.unlink()
                    if overwrite and right_target.exists():
                        right_target.unlink()

                    if not cv2.imwrite(str(left_target), left_half):
                        raise RuntimeError(f"Failed writing {left_target}")
                    if not cv2.imwrite(str(right_target), right_half):
                        raise RuntimeError(f"Failed writing {right_target}")

                generated += 1
                new_entries.append(
                    {
                        "pair_id": pair_id,
                        "fold": fold,
                        "sample": sample,
                        "left_image": f"{fold}/{sample}/{left_filename}",
                        "right_image": f"{fold}/{sample}/{right_filename}",
                        "left_source": left_source.name,
                        "right_source": right_source.name,
                        "left_bbox": [*left_box],
                        "right_bbox": [*right_box],
                        "left_area": int(left_half.size),
                        "right_area": int(right_half.size),
                        "score": 1.0,
                    }
                )
        elif crop_mode == "four":
            left_arr = read_grayscale(left_source)
            right_arr = read_grayscale(right_source)
            if left_arr.size == 0 or right_arr.size == 0:
                continue

            left_parts = split_image_four_parts(left_arr)
            right_parts = split_image_four_parts(right_arr)
            part_names = ("q1", "q2", "q3", "q4")

            for idx, (left_part, right_part) in enumerate(zip(left_parts, right_parts)):
                if left_part.size == 0 or right_part.size == 0:
                    continue

                left_temp = f"{left_source.stem}_{right_source.stem}_{part_names[idx]}_left.png"
                right_temp = f"{left_source.stem}_{right_source.stem}_{part_names[idx]}_right.png"

                pair_id = canonical_pair_id(
                    fold,
                    sample,
                    f"{fold}/{sample}/{left_temp}",
                    f"{fold}/{sample}/{right_temp}",
                )

                if pair_id in existing_set and not overwrite:
                    skipped += 1
                    continue

                left_filename = f"{pair_id}_left.png"
                right_filename = f"{pair_id}_right.png"
                left_box = (0, 0, left_part.shape[1], left_part.shape[0])
                right_box = (0, 0, right_part.shape[1], right_part.shape[0])

                if not dry_run:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    left_target = sample_dir / left_filename
                    right_target = sample_dir / right_filename
                    if overwrite and left_target.exists():
                        left_target.unlink()
                    if overwrite and right_target.exists():
                        right_target.unlink()

                    if not cv2.imwrite(str(left_target), left_part):
                        raise RuntimeError(f"Failed writing {left_target}")
                    if not cv2.imwrite(str(right_target), right_part):
                        raise RuntimeError(f"Failed writing {right_target}")

                generated += 1
                new_entries.append(
                    {
                        "pair_id": pair_id,
                        "fold": fold,
                        "sample": sample,
                        "left_image": f"{fold}/{sample}/{left_filename}",
                        "right_image": f"{fold}/{sample}/{right_filename}",
                        "left_source": left_source.name,
                        "right_source": right_source.name,
                        "left_bbox": [*left_box],
                        "right_bbox": [*right_box],
                        "left_area": int(left_part.size),
                        "right_area": int(right_part.size),
                        "score": 1.0,
                    }
                )
        else:
            left_analysis = analyses[image_index]
            right_analysis = analyses[image_index + 1]

            left_components = left_analysis.components
            right_components = right_analysis.components
            matches = match_components(
                left_components,
                right_components,
                left_analysis.shape,
                right_analysis.shape,
                threshold,
            )

            left_arr: np.ndarray | None = None
            right_arr: np.ndarray | None = None
            if not dry_run:
                left_arr = read_grayscale(left_analysis.path)
                right_arr = read_grayscale(right_analysis.path)
                if left_arr.size == 0 or right_arr.size == 0:
                    continue

            for left_idx, right_idx, score in matches:
                left_component = left_components[left_idx]
                right_component = right_components[right_idx]

                left_temp = f"{left_source.stem}_{right_source.stem}_c{left_component.index}_to_{right_component.index}.png"
                right_temp = f"{left_source.stem}_{right_source.stem}_c{right_component.index}_to_{left_component.index}.png"

                pair_id = canonical_pair_id(
                    fold,
                    sample,
                    f"{fold}/{sample}/{left_temp}",
                    f"{fold}/{sample}/{right_temp}",
                )

                if pair_id in existing_set and not overwrite:
                    skipped += 1
                    continue

                left_filename = f"{pair_id}_left.png"
                right_filename = f"{pair_id}_right.png"

                left_box = expand_bbox(left_component, left_analysis.shape, margin)
                right_box = expand_bbox(right_component, right_analysis.shape, margin)

                if not dry_run:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    left_target = sample_dir / left_filename
                    right_target = sample_dir / right_filename

                    if overwrite and left_target.exists():
                        left_target.unlink()
                    if overwrite and right_target.exists():
                        right_target.unlink()

                    l_x1, l_y1, l_x2, l_y2 = left_box
                    r_x1, r_y1, r_x2, r_y2 = right_box

                    assert left_arr is not None and right_arr is not None
                    left_crop = left_arr[l_y1:l_y2, l_x1:l_x2]
                    right_crop = right_arr[r_y1:r_y2, r_x1:r_x2]

                    if left_crop.size == 0 or right_crop.size == 0:
                        continue

                    if not cv2.imwrite(str(left_target), left_crop):
                        raise RuntimeError(f"Failed writing {left_target}")
                    if not cv2.imwrite(str(right_target), right_crop):
                        raise RuntimeError(f"Failed writing {right_target}")

                generated += 1
                new_entries.append(
                    {
                        "pair_id": pair_id,
                        "fold": fold,
                        "sample": sample,
                        "left_image": f"{fold}/{sample}/{left_filename}",
                        "right_image": f"{fold}/{sample}/{right_filename}",
                        "left_source": left_source.name,
                        "right_source": right_source.name,
                        "left_bbox": [*left_box],
                        "right_bbox": [*right_box],
                        "left_area": left_component.area,
                        "right_area": right_component.area,
                        "score": score,
                    }
                )

    return fold, sample, new_entries, generated, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a component-paired dataset.")
    parser.add_argument("--source", required=True, type=Path, help="Input root directory of source images")
    parser.add_argument("--output", required=True, type=Path, help="Output images root")
    parser.add_argument("--min-area", type=int, default=64, help="Minimum component area")
    parser.add_argument("--margin", type=int, default=8, help="Pixel margin around component crops")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Minimum component similarity")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--dry-run", action="store_true", help="Enumerate matches without writing files")
    parser.add_argument("--pairing", choices=["adjacent"], default="adjacent", help="Pairing strategy")
    parser.add_argument(
        "--crop-mode",
        choices=["components", "half", "four"],
        default="components",
        help=(
            "How to derive pair crops from adjacent image pairs: "
            "`components` connected components, `half` horizontal split, "
            "`four` split into four vertical stripes."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(32, max(1, cpu_count() or 1)),
        help="Number of workers (32 max)",
    )
    parser.add_argument(
        "--allow-scroll4",
        action="store_true",
        help="Allow source roots that look like scroll-4 datasets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_root = args.source.resolve()
    output_root = args.output.resolve()

    if args.min_area <= 0:
        raise SystemExit("--min-area must be greater than zero")
    if args.margin < 0:
        raise SystemExit("--margin must be zero or positive")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise SystemExit("--score-threshold must be between 0 and 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    if not source_root.exists():
        raise SystemExit(f"Source root does not exist: {source_root}")
    if is_scroll4_source(source_root) and not args.allow_scroll4:
        raise SystemExit(
            "Refusing to run on scroll-4 source by default. "
            f"Use --allow-scroll4 if you explicitly want this path: {source_root}"
        )

    source_folders = [
        p
        for p in sorted(source_root.iterdir())
        if p.is_dir() and not p.name.startswith(".")
    ]

    output_root.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Any, ...]] = []
    for source_dir in source_folders:
        fold, sample = parse_folder_name(source_dir.name)

        sample_dir = output_root / fold / sample
        manifest_path = sample_dir / "pairs.jsonl"

        existing_rows, existing_pair_ids = ([], set())
        if manifest_path.exists():
            existing_rows, existing_pair_ids = load_existing_manifest(manifest_path)

        tasks.append(
            (
                str(source_dir),
                str(output_root),
                fold,
                sample,
                args.min_area,
                args.margin,
                args.score_threshold,
                sorted(existing_pair_ids),
                True,
                args.crop_mode,
                bool(args.overwrite),
                bool(args.dry_run),
            )
        )

    # Preload existing manifests to merge new rows with previous outputs
    existing_manifests: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source_dir in source_folders:
        fold, sample = parse_folder_name(source_dir.name)
        sample_dir = output_root / fold / sample
        manifest_path = sample_dir / "pairs.jsonl"
        rows, _ = load_existing_manifest(manifest_path)
        existing_manifests[(fold, sample)] = rows

    if args.jobs == 1:
        results = [process_source_folder(task) for task in tasks]
    else:
        try:
            # use multiprocessing if supported; keep cv2 thread count low inside workers
            try:
                context = get_context("fork")
            except ValueError:
                context = get_context()
            with context.Pool(
                processes=args.jobs,
                initializer=_worker_init,
            ) as pool:
                results = pool.map(process_source_folder, tasks)
        except (PermissionError, OSError) as exc:
            # Some restricted environments disallow process pool creation. Use threads
            # so OpenCV/NumPy heavy work can still execute in parallel.
            print(f"Multiprocessing unavailable ({exc}). Falling back to thread pool with {args.jobs} workers.")
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                results = list(executor.map(process_source_folder, tasks))

    new_rows: list[dict[str, Any]] = []
    total_generated = 0
    total_skipped = 0

    for fold, sample, pairs, generated, skipped in results:
        total_generated += generated
        total_skipped += skipped

        sample_manifest_path = output_root / fold / sample / "pairs.jsonl"
        if args.dry_run:
            continue

        existing_rows = existing_manifests.get((fold, sample), [])
        if args.overwrite:
            existing_rows = []

        merged: dict[str, dict[str, Any]] = {
            row.get("pair_id", ""): row
            for row in existing_rows
            if isinstance(row.get("pair_id"), str)
        }
        for row in pairs:
            merged[row["pair_id"]] = row

        merged_rows = sorted(merged.values(), key=lambda item: str(item.get("pair_id", "")))

        sample_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_manifest_path.open("w", encoding="utf-8") as handle:
            for row in merged_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        new_rows.extend(merged_rows)

    if args.dry_run:
        print(f"Dry-run: generated={total_generated}, skipped={total_skipped}")
        return

    global_manifest_path = output_root.parent / "data" / "component_pairs_index.jsonl"
    global_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep deterministic order for easier diffs and reproducible tests.
    seen_global: dict[str, dict[str, Any]] = {}
    for row in new_rows:
        pair_id = row.get("pair_id")
        if isinstance(pair_id, str):
            seen_global[pair_id] = row

    with global_manifest_path.open("w", encoding="utf-8") as handle:
        for row in sorted(seen_global.values(), key=lambda item: str(item.get("pair_id", ""))):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated pairs: {total_generated}")
    print(f"Skipped (already present): {total_skipped}")
    print(f"Saved dataset to: {output_root}")


if __name__ == "__main__":
    main()
