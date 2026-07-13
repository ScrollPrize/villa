"""M2 decimation: pymeshlab texture-aware quadric collapse (in-memory only) with
geometry + perceptual (SSIM) QA gates. All file IO via scrollkit.io."""

from .core import (
    DECIMATION_FILTER,
    DEFAULT_PARAMS,
    GROUP_C_EXTENDED_LADDER,
    NO_SAFE_DECIMATION_RATIONALE,
    MeshDecimator,
    ladder_for_group,
    redecimate_at,
    run_ladder,
    ship_undecimated_copy,
)
from .metrics import (
    edge_stats,
    isolated_vertex_count,
    mesh_stats,
    repair_uv_flips,
    uv_chart_count,
    uv_orientation_counts,
    uv_signed_areas,
    wedge_to_vertex_uv_exact,
)

__all__ = [
    "DECIMATION_FILTER",
    "DEFAULT_PARAMS",
    "GROUP_C_EXTENDED_LADDER",
    "NO_SAFE_DECIMATION_RATIONALE",
    "MeshDecimator",
    "edge_stats",
    "isolated_vertex_count",
    "ladder_for_group",
    "mesh_stats",
    "redecimate_at",
    "repair_uv_flips",
    "run_ladder",
    "ship_undecimated_copy",
    "uv_chart_count",
    "uv_orientation_counts",
    "uv_signed_areas",
    "wedge_to_vertex_uv_exact",
]
