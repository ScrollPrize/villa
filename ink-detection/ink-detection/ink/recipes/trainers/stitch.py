from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Any

import torch

from ink.core.device import move_batch_to_device, resolve_runtime_device
from ink.core.types import Batch, DataBundle, ModelOutputBatch
from ink.recipes.stitch.artifacts import export_store_artifacts, resolve_media_downsample
from ink.recipes.stitch.loaders import build_stitch_inference_loaders
from ink.recipes.stitch import StitchInference, StitchInferenceRecipe
from ink.recipes.trainers.support.logging import init_wandb_session
from ink.recipes.trainers.support.run_state import apply_init_checkpoint, resolve_checkpoint_path


def _log(logger, message: str) -> None:
    if callable(logger):
        logger(str(message))


_PROGRESS_LOG_INTERVAL_BATCHES = 100


@dataclass(frozen=True)
class StitchInferenceResult:
    batches: int
    store_root_dir: str | None = None
    segment_prob_paths: dict[str, str] | None = None
    segment_preview_paths: dict[str, str] | None = None


@dataclass
class StitchInferenceRun:
    experiment: object
    inference_loaders: tuple[Any, ...]
    stitch_inference: StitchInference
    log: object = None
    init_ckpt_path: str | None = None
    resume_ckpt_path: str | None = None

    def run(self, *, device=None, run_fs=None) -> StitchInferenceResult:
        device = resolve_runtime_device(device)
        model = self.experiment.model
        if not callable(model):
            raise TypeError("stitch inference model must be callable")
        runtime = self.experiment.runtime
        init_ckpt_path = resolve_checkpoint_path(self.init_ckpt_path)
        resume_ckpt_path = resolve_checkpoint_path(self.resume_ckpt_path)
        if init_ckpt_path is not None and resume_ckpt_path is not None:
            raise ValueError("init_ckpt_path and resume_ckpt_path are mutually exclusive")
        ckpt_path = init_ckpt_path if init_ckpt_path is not None else resume_ckpt_path
        if ckpt_path is not None:
            apply_init_checkpoint(model, ckpt_path=ckpt_path)

        stitch_inference = self.stitch_inference
        if run_fs is not None:
            stitch_inference.store.root_dir = run_fs.artifacts_dir / "stitch_eval"
        wandb_session = init_wandb_session(self.experiment, run_fs=run_fs)
        stitch_inference.begin_epoch()
        if device is not None and hasattr(model, "to"):
            model.to(device)

        batch_count = 0
        batch_total = sum(int(len(loader)) for loader in self.inference_loaders)
        _log(self.log, f"[stitch_inference] start loaders={len(self.inference_loaders)} batches={batch_total}")
        run_started_at = time.perf_counter()
        progress_window_started_at = run_started_at
        progress_window_started_batch = 0
        was_training = bool(getattr(model, "training", False))
        if callable(getattr(model, "eval", None)):
            model.eval()
        try:
            with torch.inference_mode():
                for loader in self.inference_loaders:
                    for batch in loader:
                        if batch is None:
                            continue
                        if not isinstance(batch, Batch):
                            raise TypeError("stitch inference batch must be Batch")
                        batch = move_batch_to_device(batch, device=device)
                        precision_context = getattr(runtime, "precision_context", None)
                        context = precision_context(device=batch.x.device) if callable(precision_context) else nullcontext()
                        with context:
                            output_batch = ModelOutputBatch.from_batch_and_logits(batch, model(batch.x))
                        stitch_inference.observe_batch(output_batch)
                        batch_count += 1
                        if batch_total > 0 and (
                            batch_count == batch_total
                            or (batch_count % _PROGRESS_LOG_INTERVAL_BATCHES) == 0
                        ):
                            now = time.perf_counter()
                            elapsed = max(0.0, now - run_started_at)
                            window_batches = batch_count - progress_window_started_batch
                            window_elapsed = max(1e-12, now - progress_window_started_at)
                            rate = float(window_batches) / float(window_elapsed)
                            remaining_batches = max(0, batch_total - batch_count)
                            eta_seconds = float(remaining_batches) / float(rate) if rate > 0.0 else 0.0
                            progress_pct = (100.0 * float(batch_count)) / float(batch_total)
                            _log(
                                self.log,
                                f"[stitch_inference] batches={batch_count}/{batch_total} "
                                f"({progress_pct:.1f}%) elapsed={elapsed:.1f}s "
                                f"rate={rate:.2f} batches/s eta={eta_seconds:.1f}s",
                            )
                            progress_window_started_at = now
                            progress_window_started_batch = batch_count
        finally:
            if was_training and callable(getattr(model, "train", None)):
                model.train()
        try:
            flush_store = getattr(stitch_inference.store, "flush", None)
            if callable(flush_store):
                flush_store()
            store_root_dir, segment_prob_paths, segment_preview_paths, logged_images = export_store_artifacts(
                store=self.stitch_inference.store,
                media_downsample=resolve_media_downsample(runtime),
                split_name="stitch_eval",
            )
            _log(
                self.log,
                f"[stitch_inference] done batches={batch_count} "
                f"segments={0 if segment_prob_paths is None else len(segment_prob_paths)}",
            )
            if wandb_session is not None and logged_images:
                wandb_session.log_images(0, logged_images)
            return StitchInferenceResult(
                batches=batch_count,
                store_root_dir=store_root_dir,
                segment_prob_paths=segment_prob_paths,
                segment_preview_paths=segment_preview_paths,
            )
        finally:
            if wandb_session is not None:
                wandb_session.finish()


@dataclass(frozen=True, kw_only=True)
class StitchInferenceTrainer:
    stitch_inference: StitchInference | StitchInferenceRecipe
    init_ckpt_path: str | None = None
    resume_ckpt_path: str | None = None

    def build(self, *, experiment, data: DataBundle, logger=None) -> StitchInferenceRun:
        stitch_inference = self.stitch_inference
        if isinstance(stitch_inference, StitchInferenceRecipe):
            stitch_inference = stitch_inference.build(
                data=data,
                runtime=experiment.runtime,
                logger=logger,
                patch_loss=experiment.loss,
            )
        if not isinstance(stitch_inference, StitchInference):
            raise TypeError("StitchInferenceTrainer stitch_inference must build to StitchInference")
        inference_loaders = build_stitch_inference_loaders(
            stitch_data=stitch_inference.stitch_runtime.data,
            eval_loader=data.eval_loader,
        )
        return StitchInferenceRun(
            experiment=experiment,
            inference_loaders=tuple(inference_loaders or [data.eval_loader]),
            stitch_inference=stitch_inference,
            log=logger,
            init_ckpt_path=None if self.init_ckpt_path is None else str(self.init_ckpt_path),
            resume_ckpt_path=None if self.resume_ckpt_path is None else str(self.resume_ckpt_path),
        )
