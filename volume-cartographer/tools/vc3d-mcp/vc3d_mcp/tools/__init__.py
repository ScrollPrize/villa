"""Domain tool modules for vc3d-mcp.

Importing this package imports every domain submodule, whose module-level
``@mcp.tool()`` decorators register each tool on the single shared ``mcp``
instance in :mod:`vc3d_mcp.core`. The split is purely organizational: the set of
registered tools is identical to the original monolithic ``server.py``.
"""

from __future__ import annotations

from . import (  # noqa: F401  (imported for their registration side effects)
    atlas,
    catalog_volume,
    fiber,
    flatten,
    lasagna,
    manual_add,
    misc,
    points,
    seeding,
    segmentation,
    viewer,
    wrap,
)

__all__ = [
    "atlas",
    "catalog_volume",
    "fiber",
    "flatten",
    "lasagna",
    "manual_add",
    "misc",
    "points",
    "seeding",
    "segmentation",
    "viewer",
    "wrap",
]
