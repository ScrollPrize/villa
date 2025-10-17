#!/usr/bin/env python3
"""
Move label volumes that contain skeleton branch/junction points into a target folder.

For each .tif/.tiff label in the input directory the script:
1. Binarizes the data (non-zero -> foreground)
2. Runs 2D skeletonization slice-wise
3. Counts branch/junction pixels (>=3 neighbors in 8-connectivity)
4. Moves any label with at least one branch point into <output>/labels/<relative_path>
5. Optionally moves a matching image directory (same relative stem) into <output>/images

Supports multiprocessing, tqdm progress, and restart-friendly behavior by skipping
labels that already exist at the destination.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from tifffile import imread
from tqdm import tqdm

from vesuvius.image_proc.features.skeletonization import skeletonize_stack_2d


@dataclass(frozen=True)
class TaskConfig:
    """Configuration passed to worker processes."""

    label_path: Path


@dataclass
class TaskResult:
    """Result returned from worker processes."""

    label_path: Path
    branch_points: int
    error: Optional[str] = None

    @property
    def has_branch(self) -> bool:
        return self.branch_points > 0


def find_label_files(input_dir: Path, recursive: bool) -> List[Path]:
    """Collect .tif/.tiff files from input_dir."""
    exts = {".tif", ".tiff"}
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def _load_mask(label_path: Path) -> np.ndarray:
    """Load a TIFF label and return a squeezed binary numpy array."""
    arr = imread(label_path)
    arr = np.squeeze(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D label, got shape {arr.shape}")
    return arr != 0


def _count_branch_points(mask: np.ndarray) -> int:
    """Return total number of branch pixels across slices."""
    _, branch_counts = skeletonize_stack_2d(mask, return_branch_counts=True)
    return int(np.asarray(branch_counts).sum())


def _worker(task: TaskConfig) -> TaskResult:
    """Worker entry-point run in a separate process."""
    try:
        mask = _load_mask(task.label_path)
        branch_points = _count_branch_points(mask)
        return TaskResult(label_path=task.label_path, branch_points=branch_points)
    except Exception as exc:  # pragma: no cover - defensive
        return TaskResult(label_path=task.label_path, branch_points=0, error=str(exc))


def _resolve_image_path(
    image_root: Path,
    label_rel: Path,
) -> Tuple[Optional[Path], Optional[Path], Tuple[Path, ...]]:
    """
    Try to locate a matching image for the label by probing common extensions.

    Returns (src_path, relative_dest_path, tried_candidates).
    """
    candidates: List[Path] = []
    label_suffix = label_rel.suffix.lower()
    suffixes = [label_suffix] if label_suffix else []

    # Normalise to .tif/.tiff if necessary
    if label_suffix in {".tif", ".tiff"}:
        suffixes = [".tif", ".tiff"]
    elif not suffixes:
        suffixes = [".tif", ".tiff"]

    for suffix in suffixes:
        if suffix:
            if label_rel.suffix:
                candidate_rel = label_rel.with_suffix(suffix)
            else:
                candidate_rel = label_rel.parent / f"{label_rel.name}{suffix}"
        else:
            candidate_rel = label_rel
        candidate_abs = image_root / candidate_rel
        if candidate_abs not in candidates:
            candidates.append(candidate_abs)

    tried = tuple(candidates)
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidate.relative_to(image_root), tried
    return None, None, tried


def run(
    input_dir: Path,
    output_dir: Path,
    image_dir: Optional[Path],
    workers: int,
    recursive: bool,
) -> None:
    files = find_label_files(input_dir, recursive=recursive)
    if not files:
        raise SystemExit(f"No .tif/.tiff files found under {input_dir}")

    labels_out_root = output_dir / "labels"
    images_out_root = output_dir / "images" if image_dir else None
    labels_out_root.mkdir(parents=True, exist_ok=True)
    if images_out_root:
        images_out_root.mkdir(parents=True, exist_ok=True)

    to_process: List[TaskConfig] = []
    dest_map: dict[Path, Path] = {}
    image_map: dict[Path, Tuple[Path, Path]] = {}
    missing_image_candidates: dict[Path, Tuple[Path, ...]] = {}
    skipped_existing = 0

    for label_path in files:
        rel_path = label_path.relative_to(input_dir)
        dest_label_path = labels_out_root / rel_path
        if dest_label_path.exists():
            skipped_existing += 1
            continue

        dest_map[label_path] = dest_label_path
        if image_dir is not None and images_out_root is not None:
            src_img, rel_dest, tried = _resolve_image_path(image_dir, rel_path)
            if src_img is not None and rel_dest is not None:
                dest_image_path = images_out_root / rel_dest
                image_map[label_path] = (src_img, dest_image_path)
            else:
                missing_image_candidates[label_path] = tried

        to_process.append(TaskConfig(label_path=label_path))

    if not to_process:
        msg = "All label outputs already exist; nothing to do." if skipped_existing else "Nothing to process."
        print(msg)
        return

    moved_labels = 0
    moved_images = 0
    errors: List[TaskResult] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker, task) for task in to_process]
        with tqdm(total=len(futures), desc="Scanning labels") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                pbar.update(1)

                if result.error:
                    errors.append(result)
                    continue

                if result.has_branch:
                    src_label = result.label_path
                    dest_label = dest_map[src_label]
                    dest_label.parent.mkdir(parents=True, exist_ok=True)

                    if not src_label.exists():
                        errors.append(
                            TaskResult(
                                label_path=src_label,
                                branch_points=result.branch_points,
                                error="Source label missing before move (already processed?)",
                            )
                        )
                        continue

                    if dest_label.exists():
                        errors.append(
                            TaskResult(
                                label_path=src_label,
                                branch_points=result.branch_points,
                                error=f"Destination label already exists: {dest_label}",
                            )
                        )
                        continue

                    shutil.move(str(src_label), str(dest_label))
                    moved_labels += 1

                    if image_dir is not None and images_out_root is not None:
                        mapping = image_map.get(src_label)
                        if mapping is None:
                            tried = missing_image_candidates.get(src_label, ())
                            tried_str = ", ".join(str(p) for p in tried) if tried else "no candidates"
                            errors.append(
                                TaskResult(
                                    label_path=src_label,
                                    branch_points=result.branch_points,
                                    error=f"Associated image not found (tried: {tried_str})",
                                )
                            )
                        else:
                            src_img, dest_img = mapping
                            if not src_img.exists():
                                errors.append(
                                    TaskResult(
                                        label_path=src_label,
                                        branch_points=result.branch_points,
                                        error=f"Associated image missing before move: {src_img}",
                                    )
                                )
                                continue
                            dest_img.parent.mkdir(parents=True, exist_ok=True)
                            if dest_img.exists():
                                errors.append(
                                    TaskResult(
                                        label_path=src_label,
                                        branch_points=result.branch_points,
                                        error=f"Destination image path already exists: {dest_img}",
                                    )
                                )
                                continue
                            shutil.move(str(src_img), str(dest_img))
                            moved_images += 1

    if errors:
        print("Completed with errors:")
        for err in errors:
            msg = f"- {err.label_path}: {err.error}"
            if err.branch_points:
                msg += f" (branch_points={err.branch_points})"
            print(msg)

    moved_msg = f"Moved {moved_labels} labels with branch points."
    if image_dir:
        moved_msg += f" Moved {moved_images} image{'s' if moved_images == 1 else 's'}."
    print(moved_msg)
    if skipped_existing:
        print(f"Skipped {skipped_existing} labels because outputs already exist.")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move labels containing skeleton branch/junction points into a separate folder.",
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing label .tif/.tiff files.")
    parser.add_argument("output_dir", type=Path, help="Destination root directory (expects labels/ subfolder).")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional directory containing image folders matching label stems.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes (default: CPU cores - 1).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search; only look at files directly under input_dir.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.input_dir.is_dir():
        parser.error(f"input_dir is not a directory: {args.input_dir}")

    if args.image_dir is not None and not args.image_dir.is_dir():
        parser.error(f"image_dir is not a directory: {args.image_dir}")

    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        workers=args.workers,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
