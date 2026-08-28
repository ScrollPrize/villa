"""Native incremental winding-constraint graph.

Only checkpoint reconstruction and batched model inference live in Python;
topology, attachment, input parsing, graph mutation, rollback, and persistence
are implemented by :mod:`spiral_graph._spiral_graph`.
"""

from ._spiral_graph import (
    AddResult,
    Conflict,
    ConflictKind,
    Constraint,
    CycleEdge,
    GraphOptions,
    GraphStats,
    LayoutOptions,
    LayoutResult,
    Holonomy,
    HolonomyAudit,
    InputRole,
    LiftedWinding,
    Provenance,
    TrackInfo,
    TrackIndexInfo,
    WindingGraph,
    fit_rigid_registration,
    layout_largest_fiber_component,
    refine_patch_pose_graph,
)
from .checkpoint import SpiralThetaProvider

__all__ = [
    "AddResult",
    "Conflict",
    "ConflictKind",
    "Constraint",
    "CycleEdge",
    "GraphOptions",
    "GraphStats",
    "LayoutOptions",
    "LayoutResult",
    "Holonomy",
    "HolonomyAudit",
    "InputRole",
    "LiftedWinding",
    "Provenance",
    "SpiralThetaProvider",
    "TrackInfo",
    "TrackIndexInfo",
    "WindingGraph",
    "fit_rigid_registration",
    "layout_largest_fiber_component",
    "refine_patch_pose_graph",
]
