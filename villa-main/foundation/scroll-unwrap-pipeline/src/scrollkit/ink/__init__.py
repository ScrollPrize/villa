"""scrollkit.ink — M2.5 ink-overlay baking (documentary grade, see docs/RENDER-STYLE.md)."""

from .bake import (
    DEFAULT_PARAMS,
    bake_one,
    block_mean,
    build_jobs,
    choose_polarity,
    compose_overlay,
    grade_overlay,
    load_audit,
    load_params,
    normalize_ink,
    registration_check,
    resample_ink,
    smoothstep,
)

__all__ = [
    "DEFAULT_PARAMS",
    "bake_one",
    "block_mean",
    "build_jobs",
    "choose_polarity",
    "compose_overlay",
    "grade_overlay",
    "load_audit",
    "load_params",
    "normalize_ink",
    "registration_check",
    "resample_ink",
    "smoothstep",
]
