from __future__ import annotations

from dataclasses import dataclass, field

from ink.core.types import ModelOutputBatch
from ink.recipes.stitch.runtime import StitchRuntime, StitchRuntimeRecipe
from ink.recipes.stitch.store import ZarrStitchStore


@dataclass(kw_only=True)
class StitchInference:
    store: ZarrStitchStore
    segment_shapes: dict[str, tuple[int, int]] = field(repr=False)
    stitch_runtime: StitchRuntime = field(repr=False)

    def begin_epoch(self) -> None:
        self.store.reset()

    def observe_batch(self, batch: ModelOutputBatch) -> None:
        if not isinstance(batch, ModelOutputBatch):
            raise TypeError("StitchInference requires ModelOutputBatch")
        self.store.add_batch(
            logits=batch.logits,
            xyxys=batch.require_patch_xyxy(),
            segment_ids=tuple(batch.segment_ids),
        )


@dataclass(frozen=True, kw_only=True)
class StitchInferenceRecipe:
    stitch_runtime: StitchRuntime | StitchRuntimeRecipe
    store: ZarrStitchStore = field(default_factory=ZarrStitchStore)

    def build(self, *, data=None, runtime=None, logger=None, patch_loss=None) -> StitchInference:
        stitch_runtime = self.stitch_runtime
        if isinstance(stitch_runtime, StitchRuntimeRecipe):
            stitch_runtime = stitch_runtime.build(
                data,
                runtime=runtime,
                logger=logger,
                patch_loss=patch_loss,
            )
        if not isinstance(stitch_runtime, StitchRuntime):
            raise TypeError("StitchInferenceRecipe stitch_runtime must build to StitchRuntime")
        if not stitch_runtime.data.eval.segments:
            raise ValueError("stitched evaluation requires stitch.eval.segments when StitchInferenceRecipe is configured")
        layout = stitch_runtime.eval_segment_layout()
        bound_store = self.store.build(
            segment_shapes=layout.segment_shapes,
            downsample=layout.downsample,
            segment_rois=layout.segment_rois,
        )
        return StitchInference(
            store=bound_store,
            segment_shapes=layout.segment_shapes,
            stitch_runtime=stitch_runtime,
        )
