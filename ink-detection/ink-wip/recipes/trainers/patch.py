from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from ink.core.device import move_batch_to_device
from ink.recipes.trainers.support.logging import init_wandb_session
from ink.recipes.trainers.support.run_state import (
    apply_init_checkpoint,
    apply_resume_checkpoint,
    maybe_save_best_checkpoint,
    resolve_checkpoint_path,
    save_last_checkpoint,
)
from ink.core.types import Batch, DataBundle, EvalReport
from ink.recipes.losses.reporting import resolve_train_output as resolve_train_loss_output


@dataclass(frozen=True)
class TrainStepOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    primary_loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class TrainEpochResult:
    batches: int
    optimizer_steps: int
    scheduler_steps: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class EvalEpochResult:
    epoch: int
    report: EvalReport


@dataclass(frozen=True)
class TrainingRunResult:
    train_epochs: tuple[TrainEpochResult, ...]
    eval_epochs: tuple[EvalEpochResult, ...]


@dataclass
class PatchTraining:
    experiment: Any
    train_loader: Any
    val_loader: Any

    def run_step(self, batch) -> TrainStepOutput:
        if self.experiment.stitch is not None:
            raise TypeError("PatchTraining requires experiment.stitch is None")
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

        metrics = {f"train/{loss_name}": reduced_primary.detach()}
        for key, value in dict(resolved.metrics or {}).items():
            key = str(key)
            if "/" not in key:
                key = f"train/{key}"
            if isinstance(value, torch.Tensor):
                tensor = value.to(device=primary_loss.device)
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32, device=primary_loss.device)
            if tensor.ndim > 0:
                tensor = tensor.reshape(-1).mean()
            metrics[key] = tensor.detach()

        metrics["train/loss"] = objective_loss.detach()
        return TrainStepOutput(
            loss=objective_loss,
            logits=logits,
            primary_loss=primary_loss,
            metrics=metrics,
        )

    def run_epoch(self, *, optimizer_setup=None, device=None) -> TrainEpochResult:
        if self.experiment.stitch is not None:
            raise TypeError("PatchTraining requires experiment.stitch is None")

        model = self.experiment.model
        runtime = self.experiment.runtime
        if optimizer_setup is None:
            optimizer_setup = runtime.build_optimizer_setup(model)

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

        metric_sums: dict[str, float] = defaultdict(float)
        batch_count = 0
        optimizer_steps = 0
        scheduler_steps = 0
        pending_batches = 0

        def flush_optimizer_step() -> None:
            nonlocal optimizer_steps, scheduler_steps, pending_batches
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
            optimizer.zero_grad(set_to_none=True)
            pending_batches = 0

        for batch in self.train_loader:
            batch = move_batch_to_device(batch, device=device)
            step_output = self.run_step(batch)
            (step_output.loss / grad_accum).backward()

            batch_count += 1
            pending_batches += 1
            for key, value in step_output.metrics.items():
                metric_sums[str(key)] += float(value.detach().item())

            if pending_batches == grad_accum:
                flush_optimizer_step()

        flush_optimizer_step()
        if batch_count > 0 and scheduler_interval == "epoch":
            scheduler.step()
            scheduler_steps += 1

        metrics = {}
        if batch_count > 0:
            metrics = {key: value / batch_count for key, value in metric_sums.items()}
        return TrainEpochResult(
            batches=batch_count,
            optimizer_steps=optimizer_steps,
            scheduler_steps=scheduler_steps,
            metrics=metrics,
        )

    def run(
        self,
        *,
        optimizer_setup=None,
        device=None,
        epochs: int | None = None,
        start_epoch: int = 0,
        run_fs=None,
        save_best_on: str | None = None,
        save_best_higher_is_better: bool = True,
        checkpoint_extra_state: Mapping[str, Any] | Callable[[int], Mapping[str, Any] | None] | None = None,
    ) -> TrainingRunResult:
        if self.experiment.stitch is not None:
            raise TypeError("PatchTraining requires experiment.stitch is None")

        runtime = self.experiment.runtime
        model = self.experiment.model
        if optimizer_setup is None:
            optimizer_setup = runtime.build_optimizer_setup(model)

        total_epochs = int(getattr(runtime, "epochs", 1) if epochs is None else epochs)
        eval_every = max(1, int(getattr(runtime, "eval_every", 1)))
        evaluator = self.experiment.evaluator
        if evaluator is not None and not callable(getattr(evaluator, "evaluate", None)):
            raise TypeError("evaluator must define evaluate(model, val_loader, *, device=None)")

        init_ckpt_path = resolve_checkpoint_path(getattr(runtime, "init_ckpt_path", None))
        resume_ckpt_path = resolve_checkpoint_path(getattr(runtime, "resume_ckpt_path", None))
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

        wandb_session = init_wandb_session(self.experiment, run_fs=run_fs)
        try:
            train_epochs = []
            eval_epochs = []
            for epoch in range(start_epoch, start_epoch + total_epochs):
                train_epoch = self.run_epoch(
                    optimizer_setup=optimizer_setup,
                    device=device,
                )
                train_epochs.append(train_epoch)
                if run_fs is not None:
                    run_fs.log_train_epoch(epoch, train_epoch.metrics)
                if wandb_session is not None:
                    wandb_session.log_train_epoch(epoch, train_epoch.metrics)

                if evaluator is not None and ((epoch + 1) % eval_every) == 0:
                    report = evaluator.evaluate(model, self.val_loader, device=device)
                    if not isinstance(report, EvalReport):
                        raise TypeError("evaluator must return EvalReport")

                    eval_epoch = EvalEpochResult(epoch=epoch, report=report)
                    eval_epochs.append(eval_epoch)
                    if run_fs is not None:
                        run_fs.log_eval_epoch(epoch, report)
                        if save_best_on is not None:
                            maybe_save_best_checkpoint(
                                run_fs,
                                save_best_on,
                                report,
                                model=model,
                                optimizer=optimizer_setup.optimizer,
                                epoch=epoch,
                                checkpoint_extra_state=checkpoint_extra_state,
                                higher_is_better=bool(save_best_higher_is_better),
                            )
                    if wandb_session is not None:
                        wandb_session.log_eval_epoch(epoch, report.summary)

                if run_fs is not None:
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


@dataclass(frozen=True)
class PatchTrainer:
    def build(self, *, experiment, data: DataBundle, logger=None) -> PatchTraining:
        del logger
        return PatchTraining(
            experiment=experiment,
            train_loader=data.train_loader,
            val_loader=data.val_loader,
        )
