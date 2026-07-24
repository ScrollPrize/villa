"""Domain tool modules for vc3d-mcp.

Importing this package imports every submodule, whose ``@mcp.tool()``
decorators register each tool on the shared ``mcp`` instance in
:mod:`vc3d_mcp.core`.
"""

from __future__ import annotations

from . import (  # noqa: F401  (imported for their registration side effects)
    atlas,
    catalog_volume,
    fiber,
    flatten,
    jobs,
    lasagna,
    manual_add,
    points,
    review,
    seeding,
    segmentation,
    session,
    surface_ops,
    viewer,
    wrap,
)
