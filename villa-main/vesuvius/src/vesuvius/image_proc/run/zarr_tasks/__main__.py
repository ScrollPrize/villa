#!/usr/bin/env python3
"""CLI entry point for zarr processing tasks.

This module provides a unified command-line interface for various zarr
processing operations. Tasks are registered via the zarr_tasks package
and can be extended by adding new task modules.

Usage:
    python -m vesuvius.image_proc.run.zarr_tasks input.zarr output.zarr --task threshold -t 127
    python -m vesuvius.image_proc.run.zarr_tasks input.zarr output.zarr --task scale --scale-factor 2.0
    python -m vesuvius.image_proc.run.zarr_tasks --list-tasks

Available tasks:
    threshold   - Threshold values to create binary mask
    scale       - Scale array by a uniform factor
    transpose   - Transpose array axes
    recompress  - Recompress with new compressor
    edt-dilate  - Dilate mask using EDT
    merge       - Merge two zarrs (max)
    remap       - Remap values via dictionary
    resize      - Resize to reference shape
    export-tifs - Export chunks as TIFFs
"""

import sys
import warnings
from pathlib import Path

# Suppress deprecation warning for NestedDirectoryStore
warnings.filterwarnings("ignore", message="The NestedDirectoryStore.*")

from . import get_base_parser, get_task, list_tasks


def _get_task_from_args(argv):
    """Extract task name from command line arguments."""
    for i, arg in enumerate(argv):
        if arg == "--task" and i + 1 < len(argv):
            return argv[i + 1]
    return "threshold"  # default


def main():
    """Main entry point for zarr tasks CLI."""
    # Quick check for --list-tasks before parsing
    if "--list-tasks" in sys.argv:
        available_tasks = list_tasks()
        print("Available tasks:")
        for task_name in available_tasks:
            print(f"  {task_name}")
        return

    # Create base parser with common options
    parser = get_base_parser()

    # Get task name from args to add task-specific arguments before parsing
    task_name = _get_task_from_args(sys.argv)

    # Get the task class and add its arguments
    try:
        task_cls = get_task(task_name)
    except KeyError as e:
        # If task not found, parse normally to show error
        parser.parse_args()
        parser.error(str(e))

    # Add task-specific arguments before parsing
    task_cls.add_arguments(parser)

    # Now parse all arguments (this handles --help correctly)
    args = parser.parse_args()

    # Validate --inplace usage
    if args.inplace:
        if args.output_zarr is not None:
            raise SystemExit("Error: --inplace cannot be used with output_zarr")
        if args.task not in ("recompress", "zero-range"):
            raise SystemExit(
                f"Error: --inplace only supported for recompress, zero-range tasks, not {args.task}"
            )
    else:
        # Check if task requires output_zarr
        if args.task not in ("export-tifs", "zero-range"):
            if args.output_zarr is None:
                raise SystemExit(
                    "Error: output_zarr is required unless --inplace is specified"
                )

    # Validate task-specific arguments
    task_cls.validate_args(args)

    # Confirm overwrite for output
    if not args.inplace and args.output_zarr:
        output_path = Path(args.output_zarr)
        if output_path.exists():
            response = input(
                f"Output path {output_path} already exists. Overwrite? (y/n): "
            )
            if response.lower() != "y":
                print("Aborted.")
                return

    # Create and run task
    task = task_cls.from_args(args)
    print(f"Using {task.config.num_workers} worker processes")
    task.run()


if __name__ == "__main__":
    main()
