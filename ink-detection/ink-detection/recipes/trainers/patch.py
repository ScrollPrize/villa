from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping

import torch

from ink.core.device import move_batch_to_device, resolve_runtime_device
from ink.recipes.trainers.support.logging import init_wandb_session
from ink.recipes.trainers.support.run_state import (
    apply_init_checkpoint,
    apply_resume_checkpoint,
    maybe_save_best_checkpoint,
    resolve_checkpoint_path,
    save_epoch_checkpoint,
    save_last_checkpoint,
)
from ink.core.types import Batch, DataBundle, EvalReport
from ink.recipes.losses.reporting import resolve_train_output as resolve_train_loss_output
from ink.recipes.metrics import flatten_eval_report
from ink.recipes.stitch.artifacts import (
    resolve_media_downsample,
    stitch_source_downsample,
    write_segment_viz_artifacts,
)
from ink.recipes.stitch.loaders import build_stitch_runtime_loaders
from ink.recipes.stitch.runtime import StitchRuntime, StitchRuntimeRecipe


@dataclass(frozen=True)
class TrainStepOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    primary_loss: torch.Tensor
    components: dict[str, torch.Tensor]


@dataclass(frozen=True)
class TrainEpochResult:
    batches: int
    optimizer_steps: int
    scheduler_steps: int
    components: dict[str, float]


@dataclass(frozen=True)
class EvalEpochResult:
    epoch: int
    report: EvalReport


@dataclass(frozen=True)
class TrainingRunResult:
    train_epochs: tuple[TrainEpochResult, ...]
    eval_epochs: tuple[EvalEpochResult, ...]


def _log(logger, message: str) -> None:
    if callable(logger):
        logger(str(message))


def _detached_scalar(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach()
    if tensor.ndim > 0:
        tensor = tensor.reshape(-1).mean()
    return tensor.to(dtype=torch.float32)


def _accumulate_component_sum(component_sums: dict[str, torch.Tensor], key: str, value: torch.Tensor) -> None:
    scalar = _detached_scalar(value)
    previous = component_sums.get(str(key))
    if previous is None:
        component_sums[str(key)] = scalar.clone()
        return
    component_sums[str(key)] = previous + scalar


@dataclass
class PatchTraining:
    experiment: Any
    train_loader: Any
    eval_loader: Any
    stitch_runtime: Any = None
    log: Any = None
    epochs: int = 30
    eval_every: int = 1
    log_every_n_steps: int | None = 100
    save_every_n_epochs: int | None = None
    save_best_higher_is_better: bool = True
    init_ckpt_path: str | None = None
    resume_ckpt_path: str | None = None

    def _logged_eval_report(self, report: EvalReport) -> EvalReport:
        evaluator = getattr(self.experiment, "evaluator", None)
        stage_prefix = getattr(evaluator, "stage_prefix", "")
        return flatten_eval_report(report, stage_prefix=stage_prefix)

    def _run_stitch_epoch_artifacts(self, *, epoch: int, run_fs=None) -> dict[str, dict[str, object]]:
        stitch_train = getattr(self.stitch_runtime, "train", None)
        if stitch_train is None:
            return {}

        source_downsample = stitch_source_downsample(self.stitch_runtime, stitch_train)
        media_downsample = resolve_media_downsample(self.experiment.runtime)
        logged_images = write_segment_viz_artifacts(
            root_dir=(run_fs.artifacts_dir if run_fs is not None else None),
            split_name="stitch_train",
            epoch=int(epoch),
            segment_viz=stitch_train.run_viz_pass(self.experiment.model, epoch=int(epoch)),
            source_downsample=source_downsample,
            media_downsample=media_downsample,
        )
        if callable(getattr(stitch_train, "run_log_only_viz_pass", None)):
            logged_images.update(
                write_segment_viz_artifacts(
                    root_dir=(run_fs.artifacts_dir if run_fs is not None else None),
                    split_name="stitch_log_only",
                    epoch=int(epoch),
                    segment_viz=stitch_train.run_log_only_viz_pass(self.experiment.model, epoch=int(epoch)),
                    source_downsample=source_downsample,
                    media_downsample=media_downsample,
                )
            )
        return logged_images

    def _should_save_epoch_checkpoint(self, *, epoch: int) -> bool:
        cadence = self.save_every_n_epochs
        if cadence is None:
            return False
        return ((int(epoch) + 1) % cadence) == 0

    def run_step(self, batch) -> TrainStepOutput:
        if not isinstance(batch, Batch):
            raise TypeError("training batch must be Batch")
        if batch.y is None:
            raise ValueError("run_step requires batch.y")

        runtime = self.experiment.runtime
        with runtime.precision_context(device=batch.x.device):
            logits = self.experiment.model(batch.x)
            resolved = resolve_train_loss_output(
                self.experiment.loss,
                logits,
                batch.y,
                valid_mask=batch.meta.valid_mask,
            )
            primary_loss = resolved.loss
            objective_loss = self.experiment.objective(primary_loss, meta=batch.meta)
        if not isinstance(primary_loss, torch.Tensor):
            raise TypeError("resolved loss output must include a torch.Tensor under 'loss'")

        reduced_primary = primary_loss.mean() if getattr(primary_loss, "ndim", 0) > 0 else primary_loss
        loss_name = getattr(self.experiment.loss, "__name__", None)
        if not isinstance(loss_name, str) or not loss_name:
            loss_name = getattr(type(self.experiment.loss), "__name__", "") or "loss"

        components = {f"train/{loss_name}": reduced_primary.detach()}
        for key, value in dict(resolved.components or {}).items():
            key = str(key)
            if "/" not in key:
                key = f"train/{key}"
            if isinstance(value, torch.Tensor):
                tensor = value.to(device=primary_loss.device)
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32, device=primary_loss.device)
            if tensor.ndim > 0:
                tensor = tensor.reshape(-1).mean()
            components[key] = tensor.detach()

        components["train/loss"] = objective_loss.detach()
        return TrainStepOutput(
            loss=objective_loss,
            logits=logits,
            primary_loss=primary_loss,
            components=components,
        )

    def run_epoch(
        self,
        *,
        optimizer_setup=None,
        device=None,
        epoch: int | None = None,
        end_epoch: int | None = None,
        global_step_offset: int = 0,
        wandb_session=None,
    ) -> TrainEpochResult:
        model = self.experiment.model
        runtime = self.experiment.runtime
        if optimizer_setup is None:
            optimizer_setup = runtime.build_optimizer_setup(model, epochs=int(self.epochs))

        grad_accum = max(1, int(getattr(runtime, "grad_accum", 1)))
        grad_clip_norm = getattr(runtime, "grad_clip_norm", None)
        scheduler_interval = str(getattr(optimizer_setup, "scheduler_interval", "step")).strip().lower()
        if scheduler_interval not in {"step", "epoch"}:
            raise ValueError(f"optimizer scheduler interval must be 'step' or 'epoch', got {scheduler_interval!r}")

        train = getattr(model, "train", None)
        if callable(train):
            train()
        if device is not None and hasattr(model, "to"):
            model.to(device)

        optimizer = optimizer_setup.optimizer
        scheduler = optimizer_setup.scheduler
        optimizer.zero_grad(set_to_none=True)

        component_sums: dict[str, torch.Tensor] = {}
        batch_count = 0
        optimizer_steps = 0
        scheduler_steps = 0
        pending_batches = 0
        total_batches = len(self.train_loader)
        window_batches = 0
        window_loss_sum: torch.Tensor | None = None
        window_started_at = time.perf_counter()
        log_every_n_steps = None if self.log_every_n_steps is None else max(1, int(self.log_every_n_steps))
        step_component_sums: dict[str, torch.Tensor] = {}
        step_component_batches = 0

        def flush_optimizer_step() -> None:
            nonlocal optimizer_steps, scheduler_steps, pending_batches, step_component_batches
            if pending_batches <= 0:
                return
            if grad_clip_norm is not None:
                import torch.nn.utils

                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
            optimizer_steps += 1
            if scheduler_interval == "step":
                scheduler.step()
                scheduler_steps += 1
            if wandb_session is not None:
                wandb_session.log_averaged_train_step(
                    global_step=int(global_step_offset) + optimizer_steps,
                    epoch=0 if epoch is None else int(epoch),
                    component_sums=step_component_sums,
                    component_batches=step_component_batches,
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
            optimizer.zero_grad(set_to_none=True)
            pending_batches = 0
            step_component_sums.clear()
            step_component_batches = 0

        def maybe_log_progress(*, force: bool = False) -> None:
            nonlocal window_batches, window_loss_sum, window_started_at
            if not callable(self.log):
                return
            if log_every_n_steps is None:
                return
            if epoch is None or end_epoch is None:
                return
            if window_batches <= 0:
                return
            if not force and window_batches < log_every_n_steps:
                return
            elapsed = max(1e-9, time.perf_counter() - window_started_at)
            avg_loss = 0.0
            if window_loss_sum is not None:
                avg_loss = float((window_loss_sum / float(window_batches)).detach().cpu().item())
            message = (
                f"[train] epoch={epoch + 1}/{end_epoch} "
                f"step={batch_count}/{total_batches} "
                f"loss={avg_loss:.4f} "
                f"it_s={window_batches / elapsed:.2f}"
            )
            _log(self.log, message)
            window_batches = 0
            window_loss_sum = None
            window_started_at = time.perf_counter()

        for batch in self.train_loader:
            batch = move_batch_to_device(batch, device=device)
            step_output = self.run_step(batch)
            (step_output.loss / grad_accum).backward()

            batch_count += 1
            pending_batches += 1
            for key, value in step_output.components.items():
                _accumulate_component_sum(component_sums, str(key), value)
                _accumulate_component_sum(step_component_sums, str(key), value)
            step_component_batches += 1
            train_loss = step_output.components.get("train/loss")
            if train_loss is not None:
                scalar_loss = _detached_scalar(train_loss)
                if window_loss_sum is None:
                    window_loss_sum = scalar_loss.clone()
                else:
                    window_loss_sum = window_loss_sum + scalar_loss
            window_batches += 1

            if pending_batches == grad_accum:
                flush_optimizer_step()
            if log_every_n_steps is not None and (batch_count % log_every_n_steps) == 0:
                maybe_log_progress()

        flush_optimizer_step()
        maybe_log_progress(force=True)
        if batch_count > 0 and scheduler_interval == "epoch":
            scheduler.step()
            scheduler_steps += 1

        components = {}
        if batch_count > 0:
            components = {
                key: float((value / float(batch_count)).detach().cpu().item())
                for key, value in component_sums.items()
            }
        return TrainEpochResult(
            batches=batch_count,
            optimizer_steps=optimizer_steps,
            scheduler_steps=scheduler_steps,
            components=components,
        )

    def run(
        self,
        *,
        optimizer_setup=None,
        device=None,
        start_epoch: int = 0,
        run_fs=None,
        save_best_higher_is_better: bool | None = None,
        checkpoint_extra_state: Mapping[str, Any] | Callable[[int], Mapping[str, Any] | None] | None = None,
    ) -> TrainingRunResult:
        device = resolve_runtime_device(device)
        runtime = self.experiment.runtime
        model = self.experiment.model
        if optimizer_setup is None:
            optimizer_setup = runtime.build_optimizer_setup(model, epochs=int(self.epochs))

        total_epochs = int(self.epochs)
        eval_every = int(self.eval_every)
        evaluator = self.experiment.evaluator
        grad_accum = max(1, int(getattr(runtime, "grad_accum", 1)))
        steps_per_epoch = int(math.ceil(len(self.train_loader) / float(grad_accum))) if len(self.train_loader) > 0 else 0
        if evaluator is not None and not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("evaluator must define evaluate(model, eval_loader, *, device=None)")

        init_ckpt_path = resolve_checkpoint_path(self.init_ckpt_path)
        resume_ckpt_path = resolve_checkpoint_path(self.resume_ckpt_path)
        if init_ckpt_path is not None and resume_ckpt_path is not None:
            raise ValueError("init_ckpt_path and resume_ckpt_path are mutually exclusive")
        if resume_ckpt_path is not None and int(start_epoch) != 0:
            raise ValueError("start_epoch must be 0 when resume_ckpt_path is set")

        start_epoch = int(start_epoch)
        if resume_ckpt_path is not None:
            start_epoch = apply_resume_checkpoint(
                model,
                optimizer_setup.optimizer,
                ckpt_path=resume_ckpt_path,
            )
        elif init_ckpt_path is not None:
            apply_init_checkpoint(model, ckpt_path=init_ckpt_path)

        selected_best_key: str | None = None
        selected_best_higher_is_better = (
            bool(self.save_best_higher_is_better)
            if save_best_higher_is_better is None
            else bool(save_best_higher_is_better)
        )

        wandb_session = init_wandb_session(self.experiment, run_fs=run_fs)
        try:
            train_epochs = []
            eval_epochs = []
            _log(
                self.log,
                "[train] start "
                f"epochs={total_epochs} "
                f"train_batches={len(self.train_loader)} "
                f"eval_batches={len(self.eval_loader)} "
                f"eval_every={eval_every}",
            )
            for epoch in range(start_epoch, start_epoch + total_epochs):
                global_step_offset = int(epoch) * int(steps_per_epoch)
                train_epoch = self.run_epoch(
                    optimizer_setup=optimizer_setup,
                    device=device,
                    epoch=epoch,
                    end_epoch=start_epoch + total_epochs,
                    global_step_offset=global_step_offset,
                    wandb_session=wandb_session,
                )
                train_epochs.append(train_epoch)
                train_loss = train_epoch.components.get("train/loss")
                train_loss_text = ""
                if train_loss is not None:
                    train_loss_text = f" loss={float(train_loss):.4f}"
                _log(
                    self.log,
                    f"[train] epoch={epoch + 1}/{start_epoch + total_epochs} "
                    f"batches={train_epoch.batches} "
                    f"optimizer_steps={train_epoch.optimizer_steps}"
                    f"{train_loss_text}",
                )
                if run_fs is not None:
                    run_fs.log_train_epoch(epoch, train_epoch.components)
                if wandb_session is not None:
                    wandb_session.log_train_epoch(epoch, train_epoch.components)

                if self.stitch_runtime is not None:
                    logged_images = self._run_stitch_epoch_artifacts(epoch=epoch, run_fs=run_fs)
                    if wandb_session is not None and logged_images:
                        wandb_session.log_images(epoch, logged_images)

                if evaluator is not None and ((epoch + 1) % eval_every) == 0:
                    prepare_artifacts = getattr(evaluator, "prepare_run_artifacts", None)
                    if callable(prepare_artifacts):
                        prepare_artifacts(run_fs=run_fs, epoch=epoch)
                    raw_report = evaluator.evaluate(model, self.eval_loader, device=device)
                    if not isinstance(raw_report, EvalReport):
                        raise TypeError("evaluator must return EvalReport")
                    export_logged_images = getattr(evaluator, "export_logged_images", None)
                    if callable(export_logged_images):
                        logged_images = export_logged_images(
                            media_downsample=resolve_media_downsample(runtime),
                        )
                        if wandb_session is not None and logged_images:
                            wandb_session.log_images(epoch, logged_images)
                    report = self._logged_eval_report(raw_report)

                    eval_epoch = EvalEpochResult(epoch=epoch, report=report)
                    eval_epochs.append(eval_epoch)
                    patch_dice = report.summary.get("val/patch/Dice")
                    stitch_dice = report.summary.get("val/stitch/Dice")
                    eval_parts = []
                    if patch_dice is not None:
                        eval_parts.append(f"patch_dice={float(patch_dice):.4f}")
                    if stitch_dice is not None:
                        eval_parts.append(f"stitch_dice={float(stitch_dice):.4f}")
                    _log(
                        self.log,
                        f"[eval] epoch={epoch + 1}/{start_epoch + total_epochs} "
                        f"{' '.join(eval_parts)}".rstrip(),
                    )
                    if run_fs is not None:
                        run_fs.log_eval_epoch(epoch, report)
                        if selected_best_key is None:
                            for key in dict(report.summary).keys():
                                selected_best_key = str(key)
                                break
                        if selected_best_key is not None:
                            maybe_save_best_checkpoint(
                                run_fs,
                                selected_best_key,
                                report,
                                model=model,
                                optimizer=optimizer_setup.optimizer,
                                epoch=epoch,
                                checkpoint_extra_state=checkpoint_extra_state,
                                higher_is_better=selected_best_higher_is_better,
                            )
                    if wandb_session is not None:
                        wandb_session.log_eval_epoch(epoch, report)

                if run_fs is not None:
                    if self._should_save_epoch_checkpoint(epoch=epoch):
                        save_epoch_checkpoint(
                            run_fs,
                            model=model,
                            optimizer=optimizer_setup.optimizer,
                            epoch=epoch,
                            checkpoint_extra_state=checkpoint_extra_state,
                        )
                    save_last_checkpoint(
                        run_fs,
                        model=model,
                        optimizer=optimizer_setup.optimizer,
                        epoch=epoch,
                        checkpoint_extra_state=checkpoint_extra_state,
                    )

            return TrainingRunResult(
                train_epochs=tuple(train_epochs),
                eval_epochs=tuple(eval_epochs),
            )
        finally:
            if wandb_session is not None:
                wandb_session.finish()


@dataclass(frozen=True, kw_only=True)
class PatchTrainer:
    stitch_runtime: StitchRuntime | StitchRuntimeRecipe | None = None
    epochs: int = 30
    eval_every: int = 1
    log_every_n_steps: int | None = 100
    save_every_n_epochs: int | None = None
    save_best_higher_is_better: bool = True
    init_ckpt_path: str | None = None
    resume_ckpt_path: str | None = None

    def build(self, *, experiment, data: DataBundle, logger=None) -> PatchTraining:
        stitch_runtime = self.stitch_runtime
        if isinstance(stitch_runtime, StitchRuntimeRecipe):
            stitch_runtime = stitch_runtime.build(
                data,
                runtime=experiment.runtime,
                logger=logger,
                patch_loss=experiment.loss,
            )
        if stitch_runtime is not None and not isinstance(stitch_runtime, StitchRuntime):
            raise TypeError("PatchTrainer stitch_runtime must build to StitchRuntime")
        if stitch_runtime is not None:
            precision_context = getattr(experiment.runtime, "precision_context", None)
            if callable(precision_context):
                stitch_runtime.train.precision_context = precision_context
            train_viz_loaders, log_only_loaders = build_stitch_runtime_loaders(
                stitch_data=stitch_runtime.data,
                train_loader=data.train_loader,
                eval_loader=data.eval_loader,
            )
            if train_viz_loaders:
                stitch_runtime.train.set_loaders(train_viz_loaders)
            if log_only_loaders:
                stitch_runtime.train.set_log_only_loaders(log_only_loaders)
            _log(
                logger,
                "[stitch] "
                f"train_segments={len(getattr(getattr(stitch_runtime.data, 'train', None), 'segments', ()))} "
                f"eval_segments={len(getattr(getattr(stitch_runtime.data, 'eval', None), 'segments', ()))} "
                f"log_only_segments={len(getattr(getattr(stitch_runtime.data, 'log_only', None), 'segments', ()))} "
                f"train_loaders={len(getattr(getattr(stitch_runtime, 'train', None), 'loaders', ()))} "
                f"log_only_loaders={len(getattr(getattr(stitch_runtime, 'train', None), 'log_only_loaders', ()))}",
            )
        return PatchTraining(
            experiment=experiment,
            train_loader=data.train_loader,
            eval_loader=data.eval_loader,
            stitch_runtime=stitch_runtime,
            log=logger,
            epochs=int(self.epochs),
            eval_every=int(self.eval_every),
            log_every_n_steps=(None if self.log_every_n_steps is None else int(self.log_every_n_steps)),
            save_every_n_epochs=(
                None if self.save_every_n_epochs is None else int(self.save_every_n_epochs)
            ),
            save_best_higher_is_better=bool(self.save_best_higher_is_better),
            init_ckpt_path=None if self.init_ckpt_path is None else str(self.init_ckpt_path),
            resume_ckpt_path=None if self.resume_ckpt_path is None else str(self.resume_ckpt_path),
        )
