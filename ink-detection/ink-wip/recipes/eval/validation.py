from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field, replace

import torch

from ink.core.device import move_batch_to_device
from ink.core.types import Batch, EvalReport, ModelOutputBatch
from ink.recipes.eval.patch_stage import PatchEval
from ink.recipes.eval.stitch_stage import StitchEval
from ink.recipes.stitch import StitchInference, StitchInferenceRecipe
from ink.recipes.stitch.artifacts import export_store_preview_artifacts


def _write_stitched_segment_probs(stitch_inference: StitchInference | None) -> None:
    if stitch_inference is None:
        return
    store = getattr(stitch_inference, "store", None)
    if store is None:
        return
    if not callable(getattr(store, "segment_ids", None)):
        return
    if not callable(getattr(store, "write_full_segment_probs", None)):
        return
    for segment_id in tuple(store.segment_ids()):
        store.write_full_segment_probs(segment_id=str(segment_id))


@dataclass(frozen=True, kw_only=True)
class ValidationEvaluator:
    patch: PatchEval | None = None
    stitch_inference: StitchInference | StitchInferenceRecipe | None = None
    stitch: StitchEval | None = None
    _precision_context: object = field(default=None, repr=False, compare=False)
    stage_prefix: str = field(default="val", init=False)

    def build(self, *, data, runtime=None, logger=None) -> ValidationEvaluator:
        precision_context = getattr(runtime, "precision_context", None)
        if not callable(precision_context):
            precision_context = None

        bound_patch = self.patch
        if bound_patch is not None:
            bound_patch = bound_patch.build(
                data=data,
                runtime=runtime,
                logger=logger,
            )

        if self.stitch is not None and self.stitch_inference is None:
            raise ValueError("ValidationEvaluator requires stitch_inference when stitch evaluation is configured")

        bound_stitch_inference = self.stitch_inference
        if isinstance(bound_stitch_inference, StitchInferenceRecipe):
            bound_stitch_inference = bound_stitch_inference.build(data=data, runtime=runtime, logger=logger)
        elif bound_stitch_inference is not None and not isinstance(bound_stitch_inference, StitchInference):
            raise TypeError("ValidationEvaluator stitch_inference must be StitchInference or StitchInferenceRecipe")

        bound_stitch = self.stitch
        if bound_stitch is not None:
            if bound_stitch_inference is None:
                raise ValueError("ValidationEvaluator requires stitch_inference when stitch evaluation is configured")
            bound_stitch = bound_stitch.build(
                data=data,
                runtime=runtime,
                logger=logger,
                inference=bound_stitch_inference,
            )

        return replace(
            self,
            patch=bound_patch,
            stitch_inference=bound_stitch_inference,
            stitch=bound_stitch,
            _precision_context=precision_context,
        )

    def prepare_run_artifacts(self, *, run_fs=None) -> None:
        if run_fs is None:
            return
        stitch_inference = self.stitch_inference
        if stitch_inference is not None:
            stitch_inference.store.root_dir = run_fs.artifacts_dir / "stitch_eval"

    def export_logged_images(self, *, media_downsample: int) -> dict[str, dict[str, object]]:
        _preview_paths, logged_images = export_store_preview_artifacts(
            store=getattr(self.stitch_inference, "store", None),
            media_downsample=int(media_downsample),
            split_name="stitch_eval",
        )
        return logged_images

    def evaluate(self, model, eval_loader, *, device=None) -> EvalReport:
        if not callable(model):
            raise TypeError("evaluation model must be callable")
        patch = self.patch
        stitch_inference = self.stitch_inference
        if self.stitch is not None and stitch_inference is None:
            raise ValueError("ValidationEvaluator requires build(...) before stitched validation")
        if patch is None and stitch_inference is None and self.stitch is None:
            return EvalReport(summary={})

        if patch is not None:
            patch.begin_epoch()
        if stitch_inference is not None:
            stitch_inference.begin_epoch()
        if device is not None and hasattr(model, "to"):
            model.to(device)

        was_training = bool(getattr(model, "training", False))
        if callable(getattr(model, "eval", None)):
            model.eval()

        try:
            with torch.inference_mode():
                for batch in eval_loader:
                    if not isinstance(batch, Batch):
                        raise TypeError("validation batch must be Batch")
                    batch = move_batch_to_device(batch, device=device)
                    if patch is not None and batch.y is None:
                        raise ValueError("patch evaluation requires batch.y")

                    context = (
                        self._precision_context(device=batch.x.device)
                        if callable(self._precision_context)
                        else nullcontext()
                    )
                    with context:
                        output_batch = ModelOutputBatch.from_batch_and_logits(batch, model(batch.x))
                    if patch is not None:
                        patch.observe_batch(output_batch)
                    if stitch_inference is not None:
                        stitch_inference.observe_batch(output_batch)
        finally:
            if was_training and callable(getattr(model, "train", None)):
                model.train()

        stages = {}
        if patch is not None:
            stages["patch"] = patch.finalize_epoch()
        if self.stitch is not None:
            stages["stitch"] = self.stitch.finalize_epoch()
        _write_stitched_segment_probs(stitch_inference)
        return EvalReport(summary={}, stages=stages)
