#!/usr/bin/env python3
"""
Copy files from folder A to folder C only when a file with the same
name exists in both folder A and folder B.

Usage:
  python scripts/copy_matching_files.py /path/to/folder_a /path/to/folder_b /path/to/folder_c

Notes:
- Comparison is non-recursive and based on filenames in the top level of each folder.
- Only regular files are considered (directories are ignored).
- Existing files in folder C are not overwritten unless --overwrite is provided.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import shutil


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy files from A to C if the same filename exists in B."
        )
    )
    parser.add_argument("folder_a", type=Path, help="Source folder A")
    parser.add_argument("folder_b", type=Path, help="Comparison folder B")
    parser.add_argument("folder_c", type=Path, help="Destination folder C")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in folder C (default: skip existing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without making changes",
    )
    return parser.parse_args(argv)


def validate_dirs(a: Path, b: Path, c: Path) -> None:
    for label, p in ("A", a), ("B", b):
        if not p.exists():
            raise SystemExit(f"Folder {label} does not exist: {p}")
        if not p.is_dir():
            raise SystemExit(f"Folder {label} is not a directory: {p}")

    # Create C if needed
    if not c.exists():
        c.mkdir(parents=True, exist_ok=True)
    elif not c.is_dir():
        raise SystemExit(f"Folder C is not a directory: {c}")


def list_top_level_files(directory: Path) -> set[str]:
    return {
        p.name
        for p in directory.iterdir()
        if p.is_file()
    }


def copy_intersection(a: Path, b: Path, c: Path, overwrite: bool, dry_run: bool) -> int:
    files_a = list_top_level_files(a)
    files_b = list_top_level_files(b)

    common = sorted(files_a & files_b)

    copied = 0
    for name in common:
        src = a / name
        dst = c / name

        if dst.exists() and not overwrite:
            print(f"skip (exists): {dst}")
            continue

        if dry_run:
            action = "overwrite" if dst.exists() else "copy"
            print(f"dry-run {action}: {src} -> {dst}")
            copied += 1
            continue

        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"copied: {src} -> {dst}")
        copied += 1

    print(f"Done. {copied} file(s) {'would be ' if dry_run else ''}processed.")
    return copied


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    a: Path = args.folder_a
    b: Path = args.folder_b
    c: Path = args.folder_c

    validate_dirs(a, b, c)
    copy_intersection(a, b, c, overwrite=args.overwrite, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

