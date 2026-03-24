"""Standalone stitching helpers."""

from ink.recipes.stitch.config import (
    EvalStitchConfig,
    LogOnlyStitchConfig,
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
from ink.recipes.stitch.inference import StitchInference, StitchInferenceRecipe
from ink.recipes.stitch.runtime import (
    SegmentLayout,
    StitchRuntime,
    StitchRuntimeRecipe,
    compute_stitched_loss_components,
)
from ink.recipes.stitch.terms import StitchLossBatch
from ink.recipes.stitch.train_runtime import TrainStitchRuntime
from ink.recipes.stitch.store import ZarrStitchStore

__all__ = [
    "EvalStitchConfig",
    "LogOnlyStitchConfig",
    "SegmentLayout",
    "StitchComponentSpec",
    "StitchData",
    "StitchInference",
    "StitchInferenceRecipe",
    "StitchLayout",
    "StitchLossBatch",
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
    "compute_stitched_loss_components",
    "gaussian_weights",
    "normalize_component_key",
    "resolve_buffer_crop",
    "stitch_prob_map",
]
