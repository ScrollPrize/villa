#!/usr/bin/env python3
"""
Generate masks for all segments in a directory using multiprocessing.
"""

import argparse
import subprocess
import multiprocessing
from pathlib import Path
from typing import Tuple, Optional
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def find_segment_dirs(segments_dir: Path) -> list[Path]:
    """Find all segment directories (those containing meta.json)."""
    segment_dirs = []

    for item in segments_dir.iterdir():
        if item.is_dir():
            meta_file = item / "meta.json"
            if meta_file.exists():
                segment_dirs.append(item)

    return sorted(segment_dirs)


def process_segment(args: Tuple[Path, Path, str, bool, bool]) -> Tuple[Path, bool, str]:
    """
    Process a single segment.

    Returns: (segment_path, success, message)
    """
    segment_dir, volume_dir, executable, overwrite, verbose = args

    try:
        # Check if mask already exists
        mask_file = segment_dir / "mask.tif"
        if mask_file.exists() and not overwrite:
            return (segment_dir, True, "Skipped (mask exists)")

        # Build command
        cmd = [executable, str(segment_dir), str(volume_dir)]
        if overwrite:
            cmd.append("--overwrite")

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        if verbose:
            print(f"[{segment_dir.name}] {result.stdout.strip()}")
        print(segment_dir)
        return (segment_dir, True, "Success")

    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed: {e.stderr.strip() if e.stderr else e.stdout.strip()}"
        return (segment_dir, False, error_msg)
    except Exception as e:
        return (segment_dir, False, f"Error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate masks for all segments in a directory"
    )
    parser.add_argument(
        "volume_dir",
        type=Path,
        help="Path to the zarr volume directory"
    )
    parser.add_argument(
        "segments_dir",
        type=Path,
        help="Path to directory containing segment folders"
    )
    parser.add_argument(
        "--executable",
        default="vc_create_segment_mask",
        help="Path to vc_create_segment_mask executable (default: vc_create_segment_mask)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing masks"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each segment"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List segments that would be processed without actually processing them"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.volume_dir.exists():
        print(f"Error: Volume directory not found: {args.volume_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.segments_dir.exists():
        print(f"Error: Segments directory not found: {args.segments_dir}", file=sys.stderr)
        sys.exit(1)

    # Check if executable exists
    if not Path(args.executable).exists():
        # Try to find it in PATH
        result = subprocess.run(["which", args.executable], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: Executable not found: {args.executable}", file=sys.stderr)
            print("Make sure vc_create_segment_mask is built and in your PATH", file=sys.stderr)
            sys.exit(1)

    # Find all segment directories
    segment_dirs = find_segment_dirs(args.segments_dir)

    if not segment_dirs:
        print(f"No segment directories found in {args.segments_dir}")
        sys.exit(0)

    print(f"Found {len(segment_dirs)} segment directories")

    if args.dry_run:
        print("\nSegments to process:")
        for seg_dir in segment_dirs:
            mask_exists = (seg_dir / "mask.tif").exists()
            status = " [has mask]" if mask_exists else ""
            print(f"  - {seg_dir.name}{status}")
        sys.exit(0)

    # Determine number of workers
    num_workers = args.workers or multiprocessing.cpu_count()
    print(f"Using {num_workers} parallel workers")

    # Prepare arguments for each segment
    process_args = [
        (seg_dir, args.volume_dir, args.executable, args.overwrite, args.verbose)
        for seg_dir in segment_dirs
    ]

    # Process segments in parallel
    successful = 0
    failed = 0
    skipped = 0
    failures = []

    print(f"\nProcessing segments...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_segment, args): args[0]
            for args in process_args
        }

        # Process results as they complete
        for future in as_completed(futures):
            segment_dir, success, message = future.result()

            if success:
                if "Skipped" in message:
                    skipped += 1
                    if not args.verbose:
                        print(f".", end="", flush=True)
                else:
                    successful += 1
                    if not args.verbose:
                        print(f"+", end="", flush=True)
            else:
                failed += 1
                failures.append((segment_dir.name, message))
                if not args.verbose:
                    print(f"x", end="", flush=True)

    elapsed_time = time.time() - start_time

    # Print summary
    print(f"\n\nCompleted in {elapsed_time:.1f} seconds")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

    if failures:
        print("\nFailed segments:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()