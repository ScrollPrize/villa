"""Task registry for zarr processing tasks.

Provides a decorator-based registration system for zarr tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Type

if TYPE_CHECKING:
    from .base import ZarrTask

_TASK_REGISTRY: Dict[str, Type["ZarrTask"]] = {}


def register_task(name: str):
    """Decorator to register a zarr task.

    Args:
        name: The CLI name for the task (e.g., "threshold", "edt-dilate")

    Returns:
        Decorator function that registers the class

    Example:
        @register_task("threshold")
        class ThresholdTask(ZarrTask):
            ...
    """

    def decorator(cls: Type["ZarrTask"]) -> Type["ZarrTask"]:
        cls.name = name
        _TASK_REGISTRY[name] = cls
        return cls

    return decorator


def get_task(name: str) -> Type["ZarrTask"]:
    """Get a registered task by name.

    Args:
        name: The CLI name of the task

    Returns:
        The task class

    Raises:
        KeyError: If no task with that name is registered
    """
    if name not in _TASK_REGISTRY:
        available = ", ".join(sorted(_TASK_REGISTRY.keys()))
        raise KeyError(f"Unknown task '{name}'. Available tasks: {available}")
    return _TASK_REGISTRY[name]


def list_tasks() -> List[str]:
    """List all registered task names.

    Returns:
        Sorted list of task names
    """
    return sorted(_TASK_REGISTRY.keys())


def get_all_tasks() -> Dict[str, Type["ZarrTask"]]:
    """Get all registered tasks.

    Returns:
        Dictionary mapping task names to task classes
    """
    return _TASK_REGISTRY.copy()
