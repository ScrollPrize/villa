"""Extensible zarr task system.

This package provides a registry-based system for zarr processing tasks.
New tasks can be added by creating a module in the `tasks/` subdirectory
and using the `@register_task` decorator.

Example:
    from vesuvius.image_proc.run.zarr_tasks import register_task, ZarrTask

    @register_task("my-task")
    class MyTask(ZarrTask):
        ...

Usage:
    python -m vesuvius.image_proc.run.zarr_tasks input.zarr output.zarr --task threshold
    python -m vesuvius.image_proc.run.zarr_tasks --list-tasks
"""

from .base import ZarrTask, TaskConfig, get_base_parser, make_task_config
from .registry import register_task, get_task, list_tasks, get_all_tasks
from .utils import (
    get_chunk_coords,
    get_chunk_slices,
    create_level_dataset,
    build_pyramid,
    write_multiscales_metadata,
    get_default_compressor,
)

# Import tasks to trigger registration
from . import tasks

__all__ = [
    # Base classes
    "ZarrTask",
    "TaskConfig",
    # Registry functions
    "register_task",
    "get_task",
    "list_tasks",
    "get_all_tasks",
    # Utility functions
    "get_chunk_coords",
    "get_chunk_slices",
    "create_level_dataset",
    "build_pyramid",
    "write_multiscales_metadata",
    "get_default_compressor",
    # CLI helpers
    "get_base_parser",
    "make_task_config",
]
