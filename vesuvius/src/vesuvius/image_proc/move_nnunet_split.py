#!/usr/bin/env python3
"""
Move image/label pairs in the first validation set from a dataset with imagesTr/labelsTr.

Usage:
    python move_nnunet_split.py <splits_json> <dataset_root_in> <dataset_root_out> [--overwrite]

Behavior:
    - Reads the splits JSON (list of folds or single dict) and selects the first fold.
    - Extracts the `val` list from that fold.
    - For each id in the val list, moves:
        - Label: `<labelsTr>/<id>.*` -> `<out>/labelsTr/`
        - Image: `<imagesTr>/<id>_0000.*` -> `<out>/imagesTr/`
    - Prefers `.tif` if present; otherwise accepts any single matching extension.
    - Skips existing files in `<out>` unless `--overwrite` is provided.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Set, Optional


def load_first_val_set(json_path: Path) -> Set[str]:
    with json_path.open("r") as f:
        data = json.load(f)

    # If the JSON is a list of folds, take the first
    if isinstance(data, list):
        if not data:
            raise ValueError("JSON list is empty; no folds found")
        first = data[0]
    elif isinstance(data, dict):
        first = data
    else:
        raise ValueError("Unsupported JSON structure; expected list or dict")

    if not isinstance(first, dict):
        raise ValueError("First fold is not an object with keys")

    # Prefer 'val' key; allow a couple common variants
    for key in ("val", "valid", "validation"):
        if key in first:
            val = first[key]
            break
    else:
        raise KeyError("No 'val' (or 'valid'/'validation') key in first fold")

    if not isinstance(val, Iterable):
        raise ValueError("'val' must be a list/iterable of identifiers")

    # Cast to set of strings
    val_ids = {str(x) for x in val}
    return val_ids


def find_file_by_stem(dir_path: Path, stem: str, preferred_exts: Optional[Iterable[str]] = None) -> Optional[Path]:
    """Find a file in dir with the exact stem, preferring given extensions.

    Returns the Path if found, else None. If multiple matches exist, prefer
    the first present in preferred_exts; else the lexicographically first match.
    """
    if preferred_exts:
        for ext in preferred_exts:
            cand = dir_path / f"{stem}{ext}"
            if cand.exists() and cand.is_file():
                return cand
    # Fallback: any extension
    matches = sorted(dir_path.glob(f"{stem}.*"))
    if not matches:
        return None
    if preferred_exts:
        for ext in preferred_exts:
            for m in matches:
                if m.suffix.lower() == ext.lower():
                    return m
    return matches[0]


def move_pairs(dataset_in: Path, dataset_out: Path, val_ids: Set[str], overwrite: bool) -> dict:
    images_in = dataset_in / "imagesTr"
    labels_in = dataset_in / "labelsTr"
    images_out = dataset_out / "imagesTr"
    labels_out = dataset_out / "labelsTr"

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    preferred_exts = [".tif", ".tiff", ".png"]

    stats = {"moved_images": 0, "moved_labels": 0, "missing_images": 0, "missing_labels": 0}

    for vid in sorted(val_ids):
        # Label: <id>.*
        label_path = find_file_by_stem(labels_in, vid, preferred_exts)
        if label_path is None:
            print(f"Missing label for id '{vid}' in {labels_in}")
            stats["missing_labels"] += 1
        else:
            dest = labels_out / label_path.name
            if dest.exists():
                if overwrite:
                    dest.unlink()
                else:
                    print(f"Skip label (exists): {dest}")
                # If not overwriting, still try image
            else:
                shutil.move(str(label_path), str(dest))
                stats["moved_labels"] += 1

        # Image: <id>_0000.*
        image_stem = f"{vid}_0000"
        image_path = find_file_by_stem(images_in, image_stem, preferred_exts)
        if image_path is None:
            print(f"Missing image for id '{vid}' in {images_in}")
            stats["missing_images"] += 1
        else:
            dest = images_out / image_path.name
            if dest.exists():
                if overwrite:
                    dest.unlink()
                else:
                    print(f"Skip image (exists): {dest}")
                    continue
            shutil.move(str(image_path), str(dest))
            stats["moved_images"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Move image/label pairs from first 'val' set to a new dataset root")
    parser.add_argument("splits_json", type=Path, help="Path to splits JSON (e.g., splits_final.json)")
    parser.add_argument("dataset_in", type=Path, help="Source dataset root containing 'imagesTr' and 'labelsTr'")
    parser.add_argument("dataset_out", type=Path, help="Destination dataset root (will create 'imagesTr'/'labelsTr')")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in output folders")

    args = parser.parse_args()

    if not args.splits_json.exists():
        parser.error(f"splits_json not found: {args.splits_json}")
    if not args.dataset_in.exists() or not args.dataset_in.is_dir():
        parser.error(f"dataset_in not found or not a directory: {args.dataset_in}")
    if not (args.dataset_in / "imagesTr").exists() or not (args.dataset_in / "labelsTr").exists():
        parser.error("dataset_in must contain 'imagesTr' and 'labelsTr' subdirectories")

    val_ids = load_first_val_set(args.splits_json)
    print(f"Loaded {len(val_ids)} validation identifiers from first fold")

    stats = move_pairs(args.dataset_in, args.dataset_out, val_ids, args.overwrite)
    print("Summary:")
    print(f"  Moved labels: {stats['moved_labels']}")
    print(f"  Moved images: {stats['moved_images']}")
    print(f"  Missing labels: {stats['missing_labels']}")
    print(f"  Missing images: {stats['missing_images']}")
    print(f"From: {args.dataset_in}")
    print(f"To:   {args.dataset_out}")


if __name__ == "__main__":
    main()
