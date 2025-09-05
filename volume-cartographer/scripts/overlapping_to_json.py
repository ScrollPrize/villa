#!/usr/bin/env python3
"""
Script to convert overlapping directories to overlapping.json files
Usage: python overlapping_to_json.py ~/PHerc332.volpkg/paths/
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Set

def get_overlapping_names(overlap_dir: Path) -> List[str]:
    """Get all filenames from the overlapping directory."""
    names = []
    if overlap_dir.exists() and overlap_dir.is_dir():
        for entry in overlap_dir.iterdir():
            if entry.is_file() or entry.is_dir():
                names.append(entry.name)
    return sorted(names)  # Sort for consistent output

def write_overlapping_json(segment_dir: Path, names: List[str]) -> None:
    """Write overlapping names to JSON file."""
    json_path = segment_dir / "overlapping.json"
    data = {"overlapping": names}

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"  Created {json_path} with {len(names)} entries")

def migrate_segment(segment_dir: Path, keep_old: bool = False) -> tuple[bool, bool]:
    """Migrate a single segment directory. Returns (migrated, removed_dir)."""
    overlap_dir = segment_dir / "overlapping"
    json_path = segment_dir / "overlapping.json"
    migrated = False
    removed_dir = False

    # Skip if no overlapping directory exists
    if not overlap_dir.exists():
        return False, False

    # If JSON doesn't exist, create it
    if not json_path.exists():
        # Get overlapping names
        names = get_overlapping_names(overlap_dir)

        if not names:
            print(f"  {segment_dir.name}: empty overlapping directory, skipping...")
        else:
            # Write JSON file
            print(f"  {segment_dir.name}:")
            write_overlapping_json(segment_dir, names)
            migrated = True
    else:
        print(f"  {segment_dir.name}: overlapping.json already exists")

    # Remove old directory by default (unless --keep-old is specified)
    if overlap_dir.exists() and not keep_old:
        try:
            shutil.rmtree(overlap_dir)
            print(f"  Removed old directory: {overlap_dir}")
            removed_dir = True
        except Exception as e:
            print(f"  Warning: Could not remove {overlap_dir}: {e}")

    return migrated, removed_dir

def main():
    parser = argparse.ArgumentParser(
        description='Convert overlapping directories to overlapping.json files'
    )
    parser.add_argument(
        'path',
        type=str,
        help='Base path containing segment directories (e.g., ~/PHerc332.volpkg/paths/)'
    )
    parser.add_argument(
        '--keep-old',
        action='store_true',
        help='Keep old overlapping directories after conversion (by default they are removed)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='Update existing overlapping.json files'
    )

    args = parser.parse_args()

    # Expand ~ in path
    base_path = Path(args.path).expanduser()

    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        return 1

    if not base_path.is_dir():
        print(f"Error: Path is not a directory: {base_path}")
        return 1

    print(f"Scanning for segment directories in: {base_path}")

    migrated_count = 0
    skipped_count = 0
    error_count = 0
    removed_count = 0
    cleanup_only_count = 0

    # Find all subdirectories
    segment_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    print(f"Found {len(segment_dirs)} subdirectories")

    for segment_dir in sorted(segment_dirs):
        overlap_dir = segment_dir / "overlapping"
        json_path = segment_dir / "overlapping.json"

        # Check if this looks like a segment directory with overlapping data
        if not overlap_dir.exists() and not json_path.exists():
            continue

        # Also process if we just need to remove a directory
        if json_path.exists() and overlap_dir.exists() and not args.update:
            # Just need to clean up the directory
            if not args.keep_old:
                try:
                    if args.dry_run:
                        print(f"{segment_dir.name}: has both JSON and directory")
                        print(f"  Would remove directory: {overlap_dir}")
                        cleanup_only_count += 1
                    else:
                        shutil.rmtree(overlap_dir)
                        print(f"{segment_dir.name}: Removed old directory: {overlap_dir}")
                        removed_count += 1
                        cleanup_only_count += 1
                except Exception as e:
                    print(f"  Warning: Could not remove {overlap_dir}: {e}")
                    error_count += 1
            continue

        try:
            if args.dry_run:
                if overlap_dir.exists():
                    names = get_overlapping_names(overlap_dir)
                    if not json_path.exists() and names:
                        print(f"Would migrate {segment_dir.name} ({len(names)} overlaps)")
                        migrated_count += 1
                    elif json_path.exists():
                        print(f"{segment_dir.name}: has both JSON and directory")
                    else:
                        print(f"Would skip {segment_dir.name} (empty overlapping dir)")
                        skipped_count += 1

                    if not args.keep_old:
                        print(f"  Would remove directory: {overlap_dir}")
                elif json_path.exists() and args.update:
                    print(f"Would update {segment_dir.name}")
                    migrated_count += 1
                else:
                    skipped_count += 1
            else:
                # Handle updates
                if json_path.exists() and args.update and overlap_dir.exists():
                    names = get_overlapping_names(overlap_dir)
                    if names:
                        print(f"  Updating {segment_dir.name}:")
                        write_overlapping_json(segment_dir, names)
                        migrated_count += 1
                    else:
                        skipped_count += 1

                    # Always try to remove directory if not keeping old
                    if overlap_dir.exists() and not args.keep_old:
                        try:
                            shutil.rmtree(overlap_dir)
                            print(f"  Removed old directory: {overlap_dir}")
                            removed_count += 1
                        except Exception as e:
                            print(f"  Warning: Could not remove {overlap_dir}: {e}")
                else:
                    # Normal migration and/or directory removal
                    was_migrated, was_removed = migrate_segment(segment_dir, args.keep_old)
                    if was_migrated:
                        migrated_count += 1
                    else:
                        skipped_count += 1
                    if was_removed:
                        removed_count += 1

        except Exception as e:
            print(f"  Error processing {segment_dir.name}: {e}")
            error_count += 1

    # Summary
    print("\nSummary:")
    print(f"  Migrated: {migrated_count}")
    if cleanup_only_count > 0:
        print(f"  Cleanup only: {cleanup_only_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    if removed_count > 0:
        print(f"  Total directories removed: {removed_count}")

    if args.dry_run:
        print("\nThis was a dry run. No changes were made.")
        print("Run without --dry-run to perform the migration.")
    elif args.keep_old:
        print("\nOld overlapping directories were kept. Run without --keep-old to remove them.")

    return 0

if __name__ == "__main__":
    exit(main())