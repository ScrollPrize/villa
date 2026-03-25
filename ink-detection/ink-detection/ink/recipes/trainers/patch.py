from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from ink.core.device import move_batch_to_device, resolve_runtime_device
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
from ink.recipes.trainers.support.logging import init_wandb_session
from ink.recipes.trainers.support.run_state import (
    apply_init_checkpoint,
    apply_resume_checkpoint,
    maybe_save_best_checkpoint,
    resolve_checkpoint_path,
    save_epoch_checkpoint,
    save_last_checkpoint,
)


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
            log_every_n_steps=None if self.log_every_n_steps is None else int(self.log_every_n_steps),
            save_every_n_epochs=None if self.save_every_n_epochs is None else int(self.save_every_n_epochs),
            save_best_higher_is_better=bool(self.save_best_higher_is_better),
            init_ckpt_path=None if self.init_ckpt_path is None else str(self.init_ckpt_path),
            resume_ckpt_path=None if self.resume_ckpt_path is None else str(self.resume_ckpt_path),
        )


@dataclass
class PatchTraining:
    experiment: Any
    train_loader: Any
    eval_loader: Any
    stitch_runtime: StitchRuntime | None = None
    log: Any = None
    epochs: int = 30
    eval_every: int = 1
    log_every_n_steps: int | None = 100
    save_every_n_epochs: int | None = None
    save_best_higher_is_better: bool = True
    init_ckpt_path: str | None = None
    resume_ckpt_path: str | None = None

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

        evaluator = self.experiment.evaluator
        if evaluator is not None and not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("evaluator must define evaluate(model, eval_loader, *, device=None)")

        start_epoch = self._restore_start_epoch(
            model=model,
            optimizer=optimizer_setup.optimizer,
            start_epoch=start_epoch,
        )
        total_epochs = int(self.epochs)
        end_epoch = start_epoch + total_epochs
        eval_every = int(self.eval_every)
        grad_accum = max(1, int(getattr(runtime, "grad_accum", 1)))
        steps_per_epoch = int(math.ceil(len(self.train_loader) / float(grad_accum))) if len(self.train_loader) > 0 else 0
        selected_best_key: str | None = None
        selected_best_higher_is_better = (
            bool(self.save_best_higher_is_better)
            if save_best_higher_is_better is None
            else bool(save_best_higher_is_better)
        )

        wandb_session = init_wandb_session(self.experiment, run_fs=run_fs)
        try:
            train_epochs: list[TrainEpochResult] = []
            eval_epochs: list[EvalEpochResult] = []
            _log(
                self.log,
                "[train] start "
                f"epochs={total_epochs} "
                f"train_batches={len(self.train_loader)} "
                f"eval_batches={len(self.eval_loader)} "
                f"eval_every={eval_every}",
            )

            for epoch in range(start_epoch, end_epoch):
                global_step_offset = int(epoch) * int(steps_per_epoch)
                train_epoch = self.run_epoch(
                    optimizer_setup=optimizer_setup,
                    device=device,
                    epoch=epoch,
                    end_epoch=end_epoch,
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
                    f"[train] epoch={epoch + 1}/{end_epoch} "
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
                    eval_epoch, selected_best_key = self._evaluate_epoch(
                        epoch=epoch,
                        end_epoch=end_epoch,
                        evaluator=evaluator,
                        device=device,
                        run_fs=run_fs,
                        wandb_session=wandb_session,
                        optimizer=optimizer_setup.optimizer,
                        checkpoint_extra_state=checkpoint_extra_state,
                        selected_best_key=selected_best_key,
                        save_best_higher_is_better=selected_best_higher_is_better,
                    )
                    eval_epochs.append(eval_epoch)

                if run_fs is not None:
                    save_every_n_epochs = self.save_every_n_epochs
                    if save_every_n_epochs is not None and ((int(epoch) + 1) % int(save_every_n_epochs)) == 0:
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

        optimizer_setup.optimizer.zero_grad(set_to_none=True)
        log_every_n_steps = None if self.log_every_n_steps is None else max(1, int(self.log_every_n_steps))
        state = _TrainEpochState(total_batches=len(self.train_loader))

        for batch in self.train_loader:
            batch = move_batch_to_device(batch, device=device)
            step_output = self.run_step(batch)
            (step_output.loss / grad_accum).backward()
            state.add_batch(components=step_output.components)

            if state.pending_batches == grad_accum:
                self._flush_pending_optimizer_step(
                    model=model,
                    optimizer_setup=optimizer_setup,
                    state=state,
                    grad_clip_norm=grad_clip_norm,
                    scheduler_interval=scheduler_interval,
                    wandb_session=wandb_session,
                    epoch=epoch,
                    global_step_offset=global_step_offset,
                )

            if log_every_n_steps is not None and (state.batch_count % log_every_n_steps) == 0:
                self._log_train_progress_window(
                    state=state,
                    epoch=epoch,
                    end_epoch=end_epoch,
                    log_every_n_steps=log_every_n_steps,
                )

        self._flush_pending_optimizer_step(
            model=model,
            optimizer_setup=optimizer_setup,
            state=state,
            grad_clip_norm=grad_clip_norm,
            scheduler_interval=scheduler_interval,
            wandb_session=wandb_session,
            epoch=epoch,
            global_step_offset=global_step_offset,
        )
        self._log_train_progress_window(
            state=state,
            epoch=epoch,
            end_epoch=end_epoch,
            log_every_n_steps=log_every_n_steps,
            force=True,
        )
        if state.batch_count > 0 and scheduler_interval == "epoch":
            optimizer_setup.scheduler.step()
            state.scheduler_steps += 1

        return TrainEpochResult(
            batches=state.batch_count,
            optimizer_steps=state.optimizer_steps,
            scheduler_steps=state.scheduler_steps,
            components=state.averaged_components(),
        )

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

    def _restore_start_epoch(self, *, model, optimizer, start_epoch: int) -> int:
        init_ckpt_path = resolve_checkpoint_path(self.init_ckpt_path)
        resume_ckpt_path = resolve_checkpoint_path(self.resume_ckpt_path)
        if init_ckpt_path is not None and resume_ckpt_path is not None:
            raise ValueError("init_ckpt_path and resume_ckpt_path are mutually exclusive")

        requested_start_epoch = int(start_epoch)
        if resume_ckpt_path is not None and requested_start_epoch != 0:
            raise ValueError("start_epoch must be 0 when resume_ckpt_path is set")
        if resume_ckpt_path is not None:
            return apply_resume_checkpoint(
                model,
                optimizer,
                ckpt_path=resume_ckpt_path,
            )
        if init_ckpt_path is not None:
            apply_init_checkpoint(model, ckpt_path=init_ckpt_path)
        return requested_start_epoch

    def _evaluate_epoch(
        self,
        *,
        epoch: int,
        end_epoch: int,
        evaluator,
        device,
        run_fs,
        wandb_session,
        optimizer,
        checkpoint_extra_state,
        selected_best_key: str | None,
        save_best_higher_is_better: bool,
    ) -> tuple[EvalEpochResult, str | None]:
        prepare_artifacts = getattr(evaluator, "prepare_run_artifacts", None)
        if callable(prepare_artifacts):
            prepare_artifacts(run_fs=run_fs, epoch=epoch)

        raw_report = evaluator.evaluate(self.experiment.model, self.eval_loader, device=device)
        if not isinstance(raw_report, EvalReport):
            raise TypeError("evaluator must return EvalReport")

        export_logged_images = getattr(evaluator, "export_logged_images", None)
        if callable(export_logged_images):
            logged_images = export_logged_images(
                media_downsample=resolve_media_downsample(self.experiment.runtime),
            )
            if wandb_session is not None and logged_images:
                wandb_session.log_images(epoch, logged_images)

        report = flatten_eval_report(
            raw_report,
            stage_prefix=getattr(evaluator, "stage_prefix", ""),
        )

        patch_dice = report.summary.get("val/patch/Dice")
        stitch_dice = report.summary.get("val/stitch/Dice")
        eval_parts = []
        if patch_dice is not None:
            eval_parts.append(f"patch_dice={float(patch_dice):.4f}")
        if stitch_dice is not None:
            eval_parts.append(f"stitch_dice={float(stitch_dice):.4f}")
        _log(
            self.log,
            f"[eval] epoch={epoch + 1}/{end_epoch} "
            f"{' '.join(eval_parts)}".rstrip(),
        )

        if run_fs is not None:
            run_fs.log_eval_epoch(epoch, report)
            if selected_best_key is None:
                selected_best_key = next((str(key) for key in dict(report.summary).keys()), None)
            if selected_best_key is not None:
                maybe_save_best_checkpoint(
                    run_fs,
                    selected_best_key,
                    report,
                    model=self.experiment.model,
                    optimizer=optimizer,
                    epoch=epoch,
                    checkpoint_extra_state=checkpoint_extra_state,
                    higher_is_better=save_best_higher_is_better,
                )

        if wandb_session is not None:
            wandb_session.log_eval_epoch(epoch, report)

        return EvalEpochResult(epoch=epoch, report=report), selected_best_key

    def _run_stitch_epoch_artifacts(self, *, epoch: int, run_fs=None) -> dict[str, dict[str, object]]:
        stitch_train = getattr(self.stitch_runtime, "train", None)
        if stitch_train is None:
            return {}

        source_downsample = stitch_source_downsample(self.stitch_runtime, stitch_train)
        media_downsample = resolve_media_downsample(self.experiment.runtime)
        logged_images = write_segment_viz_artifacts(
            root_dir=run_fs.artifacts_dir if run_fs is not None else None,
            split_name="stitch_train",
            epoch=int(epoch),
            segment_viz=stitch_train.run_viz_pass(self.experiment.model, epoch=int(epoch)),
            source_downsample=source_downsample,
            media_downsample=media_downsample,
        )
        if callable(getattr(stitch_train, "run_log_only_viz_pass", None)):
            logged_images.update(
                write_segment_viz_artifacts(
                    root_dir=run_fs.artifacts_dir if run_fs is not None else None,
                    split_name="stitch_log_only",
                    epoch=int(epoch),
                    segment_viz=stitch_train.run_log_only_viz_pass(self.experiment.model, epoch=int(epoch)),
                    source_downsample=source_downsample,
                    media_downsample=media_downsample,
                )
            )
        return logged_images

    def _flush_pending_optimizer_step(
        self,
        *,
        model,
        optimizer_setup,
        state: _TrainEpochState,
        grad_clip_norm,
        scheduler_interval: str,
        wandb_session,
        epoch: int | None,
        global_step_offset: int,
    ) -> None:
        if state.pending_batches <= 0:
            return

        if grad_clip_norm is not None:
            import torch.nn.utils

            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

        optimizer = optimizer_setup.optimizer
        optimizer.step()
        state.optimizer_steps += 1

        if scheduler_interval == "step":
            optimizer_setup.scheduler.step()
            state.scheduler_steps += 1

        if wandb_session is not None:
            wandb_session.log_averaged_train_step(
                global_step=int(global_step_offset) + state.optimizer_steps,
                epoch=0 if epoch is None else int(epoch),
                component_sums=state.step_component_sums,
                component_batches=state.step_component_batches,
                lr=float(optimizer.param_groups[0]["lr"]),
            )

        optimizer.zero_grad(set_to_none=True)
        state.pending_batches = 0
        state.step_component_sums.clear()
        state.step_component_batches = 0

    def _log_train_progress_window(
        self,
        *,
        state: _TrainEpochState,
        epoch: int | None,
        end_epoch: int | None,
        log_every_n_steps: int | None,
        force: bool = False,
    ) -> None:
        if not callable(self.log):
            return
        if log_every_n_steps is None:
            return
        if epoch is None or end_epoch is None:
            return
        if state.window_batches <= 0:
            return
        if not force and state.window_batches < log_every_n_steps:
            return

        elapsed = max(1e-9, time.perf_counter() - state.window_started_at)
        avg_loss = 0.0
        if state.window_loss_sum is not None:
            avg_loss = float((state.window_loss_sum / float(state.window_batches)).detach().cpu().item())
        _log(
            self.log,
            f"[train] epoch={epoch + 1}/{end_epoch} "
            f"step={state.batch_count}/{state.total_batches} "
            f"loss={avg_loss:.4f} "
            f"it_s={state.window_batches / elapsed:.2f}",
        )
        state.window_batches = 0
        state.window_loss_sum = None
        state.window_started_at = time.perf_counter()


@dataclass
class _TrainEpochState:
    total_batches: int
    batch_count: int = 0
    optimizer_steps: int = 0
    scheduler_steps: int = 0
    pending_batches: int = 0
    step_component_batches: int = 0
    component_sums: dict[str, torch.Tensor] = field(default_factory=dict)
    step_component_sums: dict[str, torch.Tensor] = field(default_factory=dict)
    window_loss_sum: torch.Tensor | None = None
    window_batches: int = 0
    window_started_at: float = field(default_factory=time.perf_counter)

    def add_batch(self, *, components: Mapping[str, torch.Tensor]) -> None:
        self.batch_count += 1
        self.pending_batches += 1
        self.step_component_batches += 1
        self.window_batches += 1

        for key, value in components.items():
            _accumulate_component_sum(self.component_sums, str(key), value)
            _accumulate_component_sum(self.step_component_sums, str(key), value)

        train_loss = components.get("train/loss")
        if train_loss is None:
            return
        scalar_loss = _detached_scalar(train_loss)
        if self.window_loss_sum is None:
            self.window_loss_sum = scalar_loss.clone()
            return
        self.window_loss_sum = self.window_loss_sum + scalar_loss

    def averaged_components(self) -> dict[str, float]:
        if self.batch_count <= 0:
            return {}
        return {
            key: float((value / float(self.batch_count)).detach().cpu().item())
            for key, value in self.component_sums.items()
        }


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
