#!/usr/bin/env python3
"""
Add scroll_source metadata to all segmentation meta.json files in a volpkg.

This script reads the "name" field from the volpkg's config.json and writes it
as "scroll_source" to each meta.json in paths/, traces/, and export/.
"""

import argparse
import json
import sys
from pathlib import Path


def add_scroll_source(volpkg_root: Path, dry_run: bool = False) -> int:
    """
    Add scroll_source field to all meta.json files under paths/, traces/, export/.

    Args:
        volpkg_root: Path to the volpkg root directory
        dry_run: If True, only print what would be done without modifying files

    Returns:
        Number of files updated
    """
    config_path = volpkg_root / "config.json"

    if not config_path.exists():
        print(f"Error: config.json not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)

    if "name" not in config:
        print(f"Error: 'name' field not found in {config_path}", file=sys.stderr)
        sys.exit(1)

    scroll_name = config["name"]
    print(f"Scroll source name: {scroll_name}")

    updated_count = 0
    directories = ["paths", "traces", "export"]

    for dir_name in directories:
        dir_path = volpkg_root / dir_name
        if not dir_path.exists():
            continue

        for segment_dir in dir_path.iterdir():
            if not segment_dir.is_dir():
                continue

            meta_path = segment_dir / "meta.json"
            if not meta_path.exists():
                continue

            with open(meta_path, 'r') as f:
                meta = json.load(f)

            if meta.get("scroll_source") == scroll_name:
                print(f"  [skip] {meta_path} (already set)")
                continue

            meta["scroll_source"] = scroll_name

            if dry_run:
                print(f"  [dry-run] Would update: {meta_path}")
            else:
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                    f.write('\n')
                print(f"  [updated] {meta_path}")

            updated_count += 1

    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Add scroll_source field to segmentation meta.json files"
    )
    parser.add_argument(
        "volpkg",
        type=Path,
        help="Path to the volpkg root directory"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print what would be done without modifying files"
    )

    args = parser.parse_args()

    volpkg_root = args.volpkg.resolve()

    if not volpkg_root.exists():
        print(f"Error: {volpkg_root} does not exist", file=sys.stderr)
        sys.exit(1)

    if not volpkg_root.is_dir():
        print(f"Error: {volpkg_root} is not a directory", file=sys.stderr)
        sys.exit(1)

    count = add_scroll_source(volpkg_root, args.dry_run)

    action = "Would update" if args.dry_run else "Updated"
    print(f"\n{action} {count} meta.json file(s)")


if __name__ == "__main__":
    main()
