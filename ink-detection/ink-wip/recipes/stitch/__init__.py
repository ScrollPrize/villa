"""Standalone stitching helpers."""

from ink.recipes.stitch.data import (
    EvalStitchConfig,
    StitchComponentSpec,
    StitchData,
    StitchLayout,
    StitchSegmentSpec,
    TrainStitchConfig,
    TrainStitchLossConfig,
    TrainStitchVizConfig,
    normalize_component_key,
)
from ink.recipes.stitch.ops import (
    accumulate_to_buffers,
    allocate_segment_buffers,
    build_segment_roi_meta,
    compose_segment_from_roi_buffers,
    gaussian_weights,
    resolve_buffer_crop,
    stitch_prob_map,
)
from ink.recipes.stitch.runtime import (
    EvalStitchRuntime,
    StitchExecutionContext,
    StitchRuntime,
    StitchRuntimeRecipe,
    TrainStitchRuntime,
    compute_stitched_component_loss,
    compute_train_stitch_loss,
    run_train_stitch_pass,
)
from ink.recipes.stitch.store import ZarrStitchStore

__all__ = [
    "EvalStitchConfig",
    "EvalStitchRuntime",
    "StitchComponentSpec",
    "StitchData",
    "StitchExecutionContext",
    "StitchLayout",
    "StitchRuntime",
    "StitchRuntimeRecipe",
    "StitchSegmentSpec",
    "TrainStitchConfig",
    "TrainStitchLossConfig",
    "TrainStitchRuntime",
    "TrainStitchVizConfig",
    "ZarrStitchStore",
    "accumulate_to_buffers",
    "allocate_segment_buffers",
    "build_segment_roi_meta",
    "compose_segment_from_roi_buffers",
    "compute_stitched_component_loss",
    "compute_train_stitch_loss",
    "gaussian_weights",
    "normalize_component_key",
    "resolve_buffer_crop",
    "run_train_stitch_pass",
    "stitch_prob_map",
]
