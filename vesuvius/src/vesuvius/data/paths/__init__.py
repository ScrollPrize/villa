"""Utilities for managing packaged and local dataset path metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .local import update_local_list

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from vesuvius.utils.catalog import (
        is_aws_ec2_instance,
        list_cubes,
        list_files,
        update_list,
    )


__all__ = [
    "is_aws_ec2_instance",
    "list_cubes",
    "list_files",
    "update_list",
    "update_local_list",
]


def __getattr__(name: str) -> Any:
    if name in {"update_list", "list_files", "list_cubes", "is_aws_ec2_instance"}:
        from vesuvius.utils import catalog

        return getattr(catalog, name)
    raise AttributeError(f"module 'vesuvius.data.paths' has no attribute '{name}'")
