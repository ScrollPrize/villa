#!/usr/bin/env python3
"""
Move backup directories from paths/traces to centralized backups directory

Old: scroll.volpkg/{paths,traces}/my_segment/backups/0/
New: scroll.volpkg/backups/my_segment/0/

Usage:
    python move_backups.py <volpkg_path> [--dry-run]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def is_numeric_backup_dir(name):
    """Check if directory name is a numeric backup (0, 1, 2, etc.)"""
    return name.isdigit()


def find_backups(source_dir):
    """Find all numeric backup directories in source directory"""
    backups = []

    if not os.path.exists(source_dir):
        return backups

    for segment_name in os.listdir(source_dir):
        segment_path = os.path.join(source_dir, segment_name)

        if not os.path.isdir(segment_path):
            continue

        # Look for backups subdirectory within each segment
        backups_subdir = os.path.join(segment_path, 'backups')

        if not os.path.exists(backups_subdir) or not os.path.isdir(backups_subdir):
            continue

        # Look for numeric subdirectories within backups/
        for item in os.listdir(backups_subdir):
            item_path = os.path.join(backups_subdir, item)

            if os.path.isdir(item_path) and is_numeric_backup_dir(item):
                backups.append({
                    'segment_name': segment_name,
                    'backup_number': item,
                    'source_path': item_path,
                    'source_type': os.path.basename(source_dir)  # 'paths' or 'traces'
                })

    return backups


def move_backup(backup_info, backups_dir, dry_run=False):
    """Move a single backup directory"""
    segment_name = backup_info['segment_name']
    backup_number = backup_info['backup_number']
    source_path = backup_info['source_path']
    source_type = backup_info['source_type']

    # Create destination path
    dest_segment_dir = os.path.join(backups_dir, segment_name)
    dest_backup_dir = os.path.join(dest_segment_dir, backup_number)

    print(f"{'[DRY RUN] ' if dry_run else ''}Moving: {source_type}/{segment_name}/backups/{backup_number}/")
    print(f"  From: {source_path}")
    print(f"  To:   {dest_backup_dir}")

    if dry_run:
        return True

    # Create destination segment directory
    os.makedirs(dest_segment_dir, exist_ok=True)

    # Check if destination already exists
    if os.path.exists(dest_backup_dir):
        print(f"  ⚠️  Warning: Destination already exists, skipping")
        return False

    # Move the directory
    try:
        shutil.move(source_path, dest_backup_dir)
        print(f"  ✓ Moved successfully")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def cleanup_empty_dirs(volpkg_dir, dry_run=False):
    """Remove empty backups subdirectories and segment directories after moving backups"""
    for source_type in ['paths', 'traces']:
        source_dir = os.path.join(volpkg_dir, source_type)

        if not os.path.exists(source_dir):
            continue

        for segment_name in os.listdir(source_dir):
            segment_path = os.path.join(source_dir, segment_name)

            if not os.path.isdir(segment_path):
                continue

            # Check for empty backups subdirectory
            backups_subdir = os.path.join(segment_path, 'backups')
            if os.path.exists(backups_subdir) and os.path.isdir(backups_subdir):
                try:
                    if not os.listdir(backups_subdir):
                        print(f"{'[DRY RUN] ' if dry_run else ''}Removing empty directory: {source_type}/{segment_name}/backups/")
                        if not dry_run:
                            os.rmdir(backups_subdir)
                            print(f"  ✓ Removed")
                except OSError:
                    pass


def main():
    parser = argparse.ArgumentParser(
        description='Move backup directories to centralized backups folder'
    )
    parser.add_argument('volpkg', help='Path to .volpkg directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be moved without actually moving')

    args = parser.parse_args()

    volpkg_dir = os.path.abspath(args.volpkg)

    # Verify volpkg directory exists
    if not os.path.exists(volpkg_dir):
        print(f"Error: Directory not found: {volpkg_dir}")
        sys.exit(1)

    # Define directories
    paths_dir = os.path.join(volpkg_dir, 'paths')
    traces_dir = os.path.join(volpkg_dir, 'traces')
    backups_dir = os.path.join(volpkg_dir, 'backups')

    print(f"Volpkg directory: {volpkg_dir}")
    print(f"Searching for backups in paths/*/backups/ and traces/*/backups/...\n")

    # Find all backups
    all_backups = []
    all_backups.extend(find_backups(paths_dir))
    all_backups.extend(find_backups(traces_dir))

    if not all_backups:
        print("✓ No numeric backup directories found")
        return

    # Group by segment for summary
    by_segment = {}
    for backup in all_backups:
        key = (backup['segment_name'], backup['source_type'])
        if key not in by_segment:
            by_segment[key] = []
        by_segment[key].append(backup['backup_number'])

    # Print summary
    print(f"Found {len(all_backups)} backup directories:\n")
    for (segment_name, source_type), backup_nums in sorted(by_segment.items()):
        backup_nums_sorted = sorted(backup_nums, key=int)
        print(f"  {source_type}/{segment_name}/backups/: {', '.join(backup_nums_sorted)}")

    print()

    if args.dry_run:
        print("=== DRY RUN MODE ===\n")
    else:
        response = input("Proceed with moving these backups? [y/N]: ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
        print()

    # Create backups directory if needed
    if not args.dry_run and not os.path.exists(backups_dir):
        print(f"Creating backups directory: {backups_dir}\n")
        os.makedirs(backups_dir)

    # Move all backups
    success_count = 0
    for backup in all_backups:
        if move_backup(backup, backups_dir, args.dry_run):
            success_count += 1
        print()

    # Clean up empty directories
    print("Cleaning up empty directories...\n")
    cleanup_empty_dirs(volpkg_dir, args.dry_run)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Complete!")
    print(f"Successfully moved: {success_count}/{len(all_backups)} backups")


if __name__ == '__main__':
    main()