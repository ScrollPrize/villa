#!/usr/bin/env python3
"""
Move overlapping image/label tiles from two dataset roots into *_skip folders.

Given two dataset directories that each contain matching `images/` and `labels/`
subdirectories, this script detects tiles that cover the same spatial region
(matching scene, origin z/y/x, and array shape) and moves them into
`images_skip/` and `labels_skip/` under their respective roots. This is useful
for isolating duplicated spatial coverage between dataset splits.

The script:
    * Parses tile filenames to extract scene and origin coordinates
    * Reads TIFF metadata (without loading the full array) to determine shapes
    * Compares tiles between the two roots to find overlapping spatial regions
    * Moves the smaller of the two overlapping tiles (image + label) into *_skip folders
      when the fractional overlap exceeds a user-defined threshold
    * Supports tqdm progress bars and optional threaded metadata loading
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

from tifffile import TiffFile
from tqdm import tqdm


COORD_PATTERN = re.compile(r"z(?P<z>\d+)_y(?P<y>\d+)_x(?P<x>\d+)")
TIFF_EXTENSIONS = {".tif", ".tiff"}


@dataclass(frozen=True)
class VolumeRecord:
    """Metadata describing a single tile."""

    scene: Optional[str]
    origin: Tuple[int, int, int]  # (z, y, x)
    shape: Tuple[int, int, int]   # (z, y, x)
    bbox: Tuple[int, int, int, int, int, int]  # (z0, z1, y0, y1, x0, x1)
    volume: int
    image_path: Path
    image_rel: Path
    label_path: Optional[Path]
    label_rel: Optional[Path]

    @property
    def key(self) -> Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]:
        scene_key = self.scene if self.scene is not None else ""
        return (scene_key, self.origin, self.shape)


def parse_filename_metadata(path: Path) -> Tuple[Optional[str], Tuple[int, int, int]]:
    """Extract optional scene and origin coordinates from a tile filename."""
    name = path.stem  # includes suffixes like "_surface"
    match = COORD_PATTERN.search(name)
    if not match:
        raise ValueError(f"Could not parse z/y/x coordinates from: {path.name}")

    z = int(match.group("z"))
    y = int(match.group("y"))
    x = int(match.group("x"))

    prefix = name[:match.start()].rstrip("_")
    scene = prefix or None
    return scene, (z, y, x)


def read_tiff_shape(path: Path) -> Tuple[int, int, int]:
    """Read TIFF metadata to obtain array shape (Z, Y, X) without full load."""
    with TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(dim) for dim in series.shape)
        axes = series.axes

    z = 1
    y: Optional[int] = None
    x: Optional[int] = None

    for axis, dim in zip(axes, shape):
        if axis == "Z":
            z = dim
        elif axis == "Y":
            y = dim
        elif axis == "X":
            x = dim
        elif axis in {"S", "C"}:  # samples / channels
            continue
        else:
            # Treat other axes (e.g., time) as additional depth
            z *= dim

    if y is None or x is None:
        # Fall back to interpreting the final two dimensions as Y/X
        if len(shape) < 2:
            raise ValueError(f"Unable to infer Y/X dimensions for TIFF: {path}")
        y = shape[-2]
        x = shape[-1]

    return (int(z), int(y), int(x))


def find_label_path(labels_root: Path, image_rel: Path) -> Optional[Path]:
    """Locate the corresponding label file, allowing .tif/.tiff suffix swaps."""
    candidates: List[Path] = []
    suffix = image_rel.suffix.lower()
    if suffix in TIFF_EXTENSIONS:
        # Prioritise exact match
        candidates.append(labels_root / image_rel)
        alt = ".tiff" if suffix == ".tif" else ".tif"
        candidates.append((labels_root / image_rel).with_suffix(alt))
    else:
        candidates.append(labels_root / image_rel.with_suffix(".tif"))
        candidates.append(labels_root / image_rel.with_suffix(".tiff"))

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def process_image_file(
    image_path: Path,
    images_root: Path,
    labels_root: Path,
) -> Tuple[Optional[VolumeRecord], Optional[str]]:
    """Collect metadata for a single image tile."""
    image_rel = image_path.relative_to(images_root)

    try:
        scene, origin = parse_filename_metadata(image_path)
        shape = read_tiff_shape(image_path)
        bbox = (
            origin[0],
            origin[0] + shape[0],
            origin[1],
            origin[1] + shape[1],
            origin[2],
            origin[2] + shape[2],
        )
        volume = shape[0] * shape[1] * shape[2]
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{image_path}: {exc}"

    label_path = find_label_path(labels_root, image_rel)
    label_rel = label_path.relative_to(labels_root) if label_path else None

    record = VolumeRecord(
        scene=scene,
        origin=origin,
        shape=shape,
        bbox=bbox,
        volume=volume,
        image_path=image_path,
        image_rel=image_rel,
        label_path=label_path,
        label_rel=label_rel,
    )
    return record, None


def gather_records(
    root: Path,
    images_subdir: str,
    labels_subdir: str,
    workers: int,
) -> Tuple[List[VolumeRecord], List[str]]:
    """Load metadata for all tiles under a dataset root."""
    images_root = root / images_subdir
    labels_root = root / labels_subdir
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    if not labels_root.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_root}")

    image_files = sorted(
        p for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in TIFF_EXTENSIONS
    )

    records: List[VolumeRecord] = []
    errors: List[str] = []
    duplicates: List[str] = []
    seen_keys: Dict[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]], Path] = {}

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_image_file, path, images_root, labels_root): path
                for path in image_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Scanning {root.name}"):
                record, err = future.result()
                if err:
                    errors.append(err)
                    continue
                if record is None:
                    continue
                key = record.key
                if key in seen_keys:
                    duplicates.append(f"{record.image_path} (dupe of {seen_keys[key]})")
                    continue
                seen_keys[key] = record.image_path
                records.append(record)
    else:
        for path in tqdm(image_files, desc=f"Scanning {root.name}"):
            record, err = process_image_file(path, images_root, labels_root)
            if err:
                errors.append(err)
                continue
            if record is None:
                continue
            key = record.key
            if key in seen_keys:
                duplicates.append(f"{record.image_path} (dupe of {seen_keys[key]})")
                continue
            seen_keys[key] = record.image_path
            records.append(record)

    if duplicates:
        errors.append(
            f"Encountered {len(duplicates)} duplicate spatial keys under {root}: "
            f"{duplicates[:5]}{'...' if len(duplicates) > 5 else ''}"
        )

    return records, errors


def move_record(
    record: VolumeRecord,
    overlap_fraction: Optional[float],
    images_skip_root: Path,
    labels_skip_root: Path,
    dry_run: bool,
) -> Tuple[int, int, List[str]]:
    """Move the image (and label if present) for a single record."""
    errors: List[str] = []
    moved_images = 0
    moved_labels = 0

    # Move image
    src_img = record.image_path
    dest_img = images_skip_root / record.image_rel
    if src_img.exists():
        if dest_img.exists():
            errors.append(f"Destination image already exists: {dest_img}")
        else:
            if dry_run:
                if overlap_fraction is not None:
                    print(
                        f"[dry-run] {src_img} -> {dest_img} "
                        f"(overlap={overlap_fraction:.3f})"
                    )
                else:
                    print(f"[dry-run] {src_img} -> {dest_img}")
            else:
                dest_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_img), str(dest_img))
            moved_images += 1
    else:
        if not dest_img.exists():
            errors.append(f"Image missing and no destination file: {src_img}")

    # Move label
    if record.label_path is not None and record.label_rel is not None:
        src_lbl = record.label_path
        dest_lbl = labels_skip_root / record.label_rel
        if src_lbl.exists():
            if dest_lbl.exists():
                errors.append(f"Destination label already exists: {dest_lbl}")
            else:
                if dry_run:
                    if overlap_fraction is not None:
                        print(
                            f"[dry-run] {src_lbl} -> {dest_lbl} "
                            f"(overlap={overlap_fraction:.3f})"
                        )
                    else:
                        print(f"[dry-run] {src_lbl} -> {dest_lbl}")
                else:
                    dest_lbl.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_lbl), str(dest_lbl))
                moved_labels += 1
        else:
            if not dest_lbl.exists():
                errors.append(f"Label missing and no destination file: {src_lbl}")
    else:
        errors.append(f"No label found for image {record.image_path}")

    return moved_images, moved_labels, errors


def execute_moves(
    records: Iterable[Tuple[VolumeRecord, float]],
    root: Path,
    images_skip_subdir: str,
    labels_skip_subdir: str,
    dry_run: bool,
) -> Tuple[int, int, List[str]]:
    """Move a batch of records into skip folders."""
    images_skip_root = root / images_skip_subdir
    labels_skip_root = root / labels_skip_subdir

    total_images = 0
    total_labels = 0
    errors: List[str] = []

    records_list = list(records)
    for record, overlap_fraction in tqdm(records_list, desc=f"Moving under {root.name}"):
        moved_i, moved_l, errs = move_record(
            record,
            overlap_fraction,
            images_skip_root,
            labels_skip_root,
            dry_run,
        )
        total_images += moved_i
        total_labels += moved_l
        errors.extend(errs)

    return total_images, total_labels, errors


def _boxes_overlap(
    bbox_a: Tuple[int, int, int, int, int, int],
    bbox_b: Tuple[int, int, int, int, int, int],
) -> bool:
    """Return True if two axis-aligned bounding boxes overlap."""
    az0, az1, ay0, ay1, ax0, ax1 = bbox_a
    bz0, bz1, by0, by1, bx0, bx1 = bbox_b
    return (
        az0 < bz1 and bz0 < az1
        and ay0 < by1 and by0 < ay1
        and ax0 < bx1 and bx0 < ax1
    )


def _overlap_volume(
    bbox_a: Tuple[int, int, int, int, int, int],
    bbox_b: Tuple[int, int, int, int, int, int],
) -> int:
    """Compute the overlapping volume between two bounding boxes."""
    az0, az1, ay0, ay1, ax0, ax1 = bbox_a
    bz0, bz1, by0, by1, bx0, bx1 = bbox_b
    dz = max(0, min(az1, bz1) - max(az0, bz0))
    dy = max(0, min(ay1, by1) - max(ay0, by0))
    dx = max(0, min(ax1, bx1) - max(ax0, bx0))
    return dz * dy * dx


def find_overlaps(
    records_a: List[VolumeRecord],
    records_b: List[VolumeRecord],
    overlap_threshold: float,
) -> Tuple[
    List[VolumeRecord],
    List[VolumeRecord],
    List[Tuple[VolumeRecord, VolumeRecord, int, float]],
]:
    """
    Return overlapping records between two datasets.

    Parameters
    ----------
    overlap_threshold:
        Maximum allowed fractional overlap (relative to the smaller tile's volume).
        Pairs exceeding this threshold trigger a move of the smaller tile.
    """
    overlaps: List[Tuple[VolumeRecord, VolumeRecord, int, float]] = []
    move_a: Dict[Path, Tuple[VolumeRecord, float]] = {}
    move_b: Dict[Path, Tuple[VolumeRecord, float]] = {}

    for rec_a in records_a:
        for rec_b in records_b:
            if not _boxes_overlap(rec_a.bbox, rec_b.bbox):
                continue
            vol = _overlap_volume(rec_a.bbox, rec_b.bbox)
            if vol <= 0:
                continue
            frac = vol / min(rec_a.volume, rec_b.volume)
            overlaps.append((rec_a, rec_b, vol, frac))

            if frac <= overlap_threshold:
                continue

            if rec_a.volume < rec_b.volume:
                move_a.setdefault(rec_a.image_path, (rec_a, frac))
            elif rec_b.volume < rec_a.volume:
                move_b.setdefault(rec_b.image_path, (rec_b, frac))
            else:
                # Equal volumes: prefer moving dataset B to keep dataset A intact
                move_b.setdefault(rec_b.image_path, (rec_b, frac))

    return list(move_a.values()), list(move_b.values()), overlaps


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move overlapping image/label tiles into *_skip directories.",
    )
    parser.add_argument("dir_a", type=Path, help="First dataset root (contains images/ and labels/)")
    parser.add_argument("dir_b", type=Path, help="Second dataset root (contains images/ and labels/)")
    parser.add_argument("--images-subdir", default="images", help="Name of the images subdirectory (default: images)")
    parser.add_argument("--labels-subdir", default="labels", help="Name of the labels subdirectory (default: labels)")
    parser.add_argument("--images-skip", default="images_skip", help="Destination subdirectory for skipped images")
    parser.add_argument("--labels-skip", default="labels_skip", help="Destination subdirectory for skipped labels")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker threads for metadata loading (default: 0 -> sequential)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without moving files")
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.0,
        help="Maximum allowed fractional overlap (relative to the smaller tile). "
        "Pairs exceeding this threshold trigger moving the smaller tile. Default: 0.0",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    records_a, errors_a = gather_records(args.dir_a, args.images_subdir, args.labels_subdir, args.workers)
    records_b, errors_b = gather_records(args.dir_b, args.images_subdir, args.labels_subdir, args.workers)

    all_errors = errors_a + errors_b

    move_a_records, move_b_records, overlap_pairs = find_overlaps(
        records_a,
        records_b,
        args.overlap_threshold,
    )
    if not overlap_pairs:
        if all_errors:
            print("No overlapping tiles found. Encountered the following issues:")
            for err in all_errors:
                print(f"- {err}")
        else:
            print("No overlapping tiles found.")
        return

    pairs_over_thresh = [p for p in overlap_pairs if p[3] > args.overlap_threshold]

    print(
        f"Found {len(overlap_pairs)} overlapping tile pair{'s' if len(overlap_pairs) != 1 else ''}. "
        f"{len(pairs_over_thresh)} exceed threshold {args.overlap_threshold:.3f}."
    )

    if not pairs_over_thresh:
        print("All overlaps are within the tolerable threshold; no tiles will be moved.")
        if all_errors:
            print("Warnings encountered during scanning:")
            for err in all_errors:
                print(f"- {err}")
        return

    print(
        f"{len(move_a_records)} tile{'s' if len(move_a_records) != 1 else ''} from {args.dir_a.name} "
        f"and {len(move_b_records)} from {args.dir_b.name} will be moved (smaller tiles)."
    )

    moved_a_img, moved_a_lbl, errors_a_move = execute_moves(
        move_a_records,
        args.dir_a,
        args.images_skip,
        args.labels_skip,
        args.dry_run,
    )
    moved_b_img, moved_b_lbl, errors_b_move = execute_moves(
        move_b_records,
        args.dir_b,
        args.images_skip,
        args.labels_skip,
        args.dry_run,
    )

    print(
        f"{'Would move' if args.dry_run else 'Moved'} "
        f"{moved_a_img} images / {moved_a_lbl} labels from {args.dir_a}"
    )
    print(
        f"{'Would move' if args.dry_run else 'Moved'} "
        f"{moved_b_img} images / {moved_b_lbl} labels from {args.dir_b}"
    )

    all_errors.extend(errors_a_move)
    all_errors.extend(errors_b_move)

    if all_errors:
        print("Completed with warnings:")
        for err in all_errors:
            print(f"- {err}")


if __name__ == "__main__":  # pragma: no cover
    main()
