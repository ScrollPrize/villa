"""Accelerate-based ink model training and checkpoint artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from vesuvius.ink_detection.models.checkpoint import (
    load_checkpoint,
    resolve_checkpoint_path,
    restore_training_state,
)
from vesuvius.ink_detection.config import (
    InkDataConfig,
    TrainingConfig,
    resolve_training_mapping,
)
from vesuvius.ink_detection.training.deep_supervision import (
    DeepSupervisionWrapper,
    concatenate_deep_supervision_ignore,
    deep_supervision_weights,
)
from vesuvius.ink_detection.training.dilation import (
    apply_label_dilation,
    resolve_dilation_distances,
)
from vesuvius.ink_detection.models.input_padding import center_pad_input_depth
from vesuvius.ink_detection.training.metrics import BalancedAccuracy, Confusion
from vesuvius.ink_detection.training.stitching import run_model_forward
from vesuvius.ink_detection.types import ConfusionCounts, MetricBatch
from vesuvius.ink_detection.training.visualization import (
    PreviewAccumulator,
    build_validation_preview_log,
    central_full_3d_preview,
)
from vesuvius.utils.cli import HyphenUnderscoreParser


@dataclass(frozen=True)
class TrainingRequest:
    """Command inputs after checkpoint selection and config resolution."""

    config: TrainingConfig
    checkpoint_path: Path | None
    checkpoint: Any | None


def stage_training_request(
    config_path: str | Path,
) -> TrainingRequest:
    """Create the frozen config views and load any selected checkpoint."""

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        authored = json.load(stream)
    config = TrainingConfig.from_mapping(resolve_training_mapping(authored))
    checkpoint_path = resolve_checkpoint_path(
        config.ink.checkpoint.path, config_path
    )
    return TrainingRequest(
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint=(
            None if checkpoint_path is None else load_checkpoint(checkpoint_path)
        ),
    )


def training_dataset_config(config: TrainingConfig) -> InkDataConfig:
    """Derive the loader crop and defer any positive native dilation."""

    mapping = config.to_mapping()
    mapping["patch_size"] = list(config.loader_patch_size)
    label_distance, supervision_distance = resolve_dilation_distances(config)
    if label_distance > 0.0 or supervision_distance > 0.0:
        full_3d = dict(mapping.get("full_3d") or {})
        full_3d["label_dilation_distance"] = 0.0
        full_3d["supervision_dilation_distance"] = 0.0
        mapping["full_3d"] = full_3d
    return InkDataConfig.from_mapping(mapping)


def prepare_model_input(
    batch: Mapping[str, torch.Tensor], config: TrainingConfig
) -> torch.Tensor:
    """Build the one- or two-channel BCZYX input and center-pad depth."""

    image_BCZYX = batch["image"].float()
    if config.surface_mask_channel:
        image_BCZYX = torch.cat(
            (image_BCZYX, batch["surface_mask"].float()), dim=1
        )
    return center_pad_input_depth(
        image_BCZYX, config.ink.model.input_pad_depth_to
    )


def prepare_loss_inputs(predictions, batch, *, mode: str):
    """Return float-workflow predictions, binary targets, and ignore masks."""

    if isinstance(predictions, (list, tuple)):
        prepared_predictions = []
        targets = None
        ignore_mask = None
        for index, prediction_level in enumerate(predictions):
            prepared, current_targets, current_ignore = prepare_loss_inputs(
                prediction_level, batch, mode=mode
            )
            prepared_predictions.append(prepared)
            if index == 0:
                targets = current_targets
                ignore_mask = current_ignore
        return type(predictions)(prepared_predictions), targets, ignore_mask

    if mode in {"full_3d", "full_3d_single_wrap"}:
        crop_shape = tuple(int(value) for value in batch["image"].shape[-3:])
        if tuple(int(value) for value in predictions.shape[-3:]) != crop_shape:
            predictions = F.interpolate(
                predictions,
                size=crop_shape,
                mode="trilinear",
                align_corners=True,
            )
        targets = batch["inklabels"]
        ignore_mask = (batch["supervision_mask"] <= 0).to(
            dtype=targets.dtype
        )
        return predictions, targets, ignore_mask

    targets = (torch.amax(batch["inklabels"], dim=2) > 0).to(
        dtype=batch["inklabels"].dtype
    )
    supervision = torch.amax(batch["supervision_mask"], dim=2)
    ignore_mask = (supervision <= 0).to(dtype=targets.dtype)
    output_size = tuple(int(value) for value in predictions.shape[-2:])
    if tuple(int(value) for value in targets.shape[-2:]) != output_size:
        targets = F.interpolate(
            targets.float(), size=output_size, mode="nearest"
        ).to(dtype=batch["inklabels"].dtype)
        ignore_mask = F.interpolate(
            ignore_mask.float(), size=output_size, mode="nearest"
        ).to(dtype=targets.dtype)
    return predictions, targets, ignore_mask


def masked_unsmoothed_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute plain BCE over valid pixels, independent of label smoothing."""

    valid_mask = (ignore_mask <= 0).to(dtype=torch.float32)
    elements = F.binary_cross_entropy_with_logits(
        logits.detach().float(), targets.detach().float(), reduction="none"
    )
    return (elements * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def append_validation_metrics(path: str | Path, record: Mapping) -> None:
    """Append one sorted JSON object without truncation or deduplication."""

    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), sort_keys=True) + "\n")


def benchmark_summary(
    *,
    elapsed_seconds: float,
    measured_steps: int,
    batch_size: int,
    world_size: int,
    data_wait_seconds: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> dict[str, float | int]:
    """Build the machine-readable short production benchmark summary."""

    elapsed_seconds = float(elapsed_seconds)
    measured_steps = int(measured_steps)
    if elapsed_seconds <= 0 or measured_steps <= 0:
        raise ValueError("benchmark duration and measured steps must be positive")
    examples = measured_steps * int(batch_size) * int(world_size)
    return {
        "elapsed_seconds": elapsed_seconds,
        "measured_steps": measured_steps,
        "batch_size_per_process": int(batch_size),
        "world_size": int(world_size),
        "examples": examples,
        "steps_per_second": measured_steps / elapsed_seconds,
        "examples_per_second": examples / elapsed_seconds,
        "data_wait_seconds": float(data_wait_seconds),
        "data_wait_seconds_per_step": (
            float(data_wait_seconds) / measured_steps
        ),
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "peak_reserved_bytes": int(peak_reserved_bytes),
    }


def create_training_scheduler(optimizer, config: TrainingConfig):
    """Build one of the three active schedulers without Accelerate wrapping."""

    scheduler = config.scheduler
    if scheduler.name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler.t_max, eta_min=scheduler.eta_min
        )
    if scheduler.name == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler.max_lr,
            total_steps=scheduler.total_steps,
            pct_start=scheduler.pct_start,
            final_div_factor=scheduler.final_div_factor,
        )
    from vesuvius.models.training.lr_schedulers import get_scheduler

    return get_scheduler(
        "diffusers_cosine_warmup",
        optimizer,
        initial_lr=config.optimizer.learning_rate,
        max_steps=config.max_steps,
        warmup_steps=scheduler.warmup_steps,
    )


def initialize_training_model(model, config: TrainingConfig) -> None:
    """Apply He initialization only when no pretrained backbone is configured."""

    if config.ink.model.pretrained_backbone is not None:
        return
    from vesuvius.models.utils import InitWeights_He

    model.apply(InitWeights_He(neg_slope=0.2))


def _distributed_mean_scalar(accelerator, value) -> torch.Tensor:
    tensor = (
        value.detach()
        if isinstance(value, torch.Tensor)
        else torch.tensor(float(value), device=accelerator.device)
    )
    tensor = tensor.to(
        device=accelerator.device, dtype=torch.float32
    ).reshape(())
    return accelerator.reduce(tensor, reduction="mean")


def _distributed_mean_metrics(accelerator, metrics) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    return {
        str(name): float(
            _distributed_mean_scalar(accelerator, value).item()
        )
        for name, value in metrics.items()
    }


def build_training_checkpoint_payload(
    *,
    accelerator,
    model,
    optimizer,
    scheduler,
    config: TrainingConfig,
    step: int,
    optimizer_step: int,
    ema_model,
    wandb_module,
    validation_metrics: Mapping | None = None,
) -> dict:
    canonical = config.to_mapping()
    run = None if wandb_module is None else wandb_module.run
    payload = {
        "model": accelerator.get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "config": canonical,
        "step": step,
        "wandb_run_id": (
            run.id if run is not None else config.wandb_run_id
        ),
    }
    if validation_metrics is not None:
        payload["validation_metrics"] = dict(validation_metrics)
    if ema_model is not None and config.ema.save_in_checkpoint:
        payload["ema_model"] = ema_model.state_dict()
        payload["ema_optimizer_step"] = optimizer_step
    return payload


def _write_json_replace(path: Path, value: Mapping) -> None:
    temporary = Path(str(path) + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _run_training(request: TrainingRequest) -> int:
    try:
        import accelerate
        from accelerate.utils import (
            DistributedDataParallelKwargs,
            GradientAccumulationPlugin,
            set_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "ink training requires the models extra with accelerate installed"
        ) from exc

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from vesuvius.ink_detection.data.dataset import InkDataset
    from vesuvius.ink_detection.training.losses import create_loss
    from vesuvius.ink_detection.models.model import make_model
    from vesuvius.ink_detection.training.optimizers import create_training_optimizer
    from vesuvius.ink_detection.training.samplers import build_sampling_policy

    config = request.config
    canonical = config.to_mapping()
    data_loader_configuration = accelerate.DataLoaderConfiguration(
        non_blocking=True
    )
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=config.ddp_find_unused_parameters,
        broadcast_buffers=config.ddp_broadcast_buffers,
    )
    accumulation = GradientAccumulationPlugin(
        num_steps=config.grad_acc_steps, sync_with_dataloader=False
    )
    # The training loop reuses the dataloader indefinitely, so accumulation
    # boundaries must stay independent of dataloader exhaustion.
    accelerator = accelerate.Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_plugin=accumulation,
        dataloader_config=data_loader_configuration,
        kwargs_handlers=[ddp_kwargs],
    )

    wandb_module = None
    if "wandb_project" in canonical and accelerator.is_main_process:
        import wandb as wandb_module

        wandb_kwargs = {
            "project": canonical["wandb_project"],
            "entity": canonical["wandb_entity"],
            "config": canonical,
        }
        if config.wandb_resume:
            run_id = config.wandb_run_id
            if not run_id and request.checkpoint is not None:
                run_id = request.checkpoint.get("wandb_run_id")
            if not run_id:
                raise ValueError(
                    "wandb_resume=true requires wandb_run_id in config or checkpoint"
                )
            wandb_kwargs["id"] = run_id
            wandb_kwargs["resume"] = "must"
        wandb_module.init(**wandb_kwargs)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    train_preview_dir = config.out_dir / "train_previews"
    val_preview_dir = config.out_dir / "val_previews"
    train_preview_dir.mkdir(parents=True, exist_ok=True)
    val_preview_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    dataset_config = training_dataset_config(config)
    label_distance, supervision_distance = resolve_dilation_distances(config)
    shared_dataset = InkDataset(dataset_config, do_augmentations=False)
    if not shared_dataset.training_patches:
        raise ValueError(
            "InkDataset produced no training patches after applying supervision masking"
        )
    train_dataset = InkDataset(
        dataset_config,
        do_augmentations=True,
        patches=shared_dataset.training_patches,
        segments=shared_dataset.segments,
    )
    val_dataset = InkDataset(
        dataset_config,
        do_augmentations=False,
        patches=shared_dataset.validation_patches,
        segments=shared_dataset.segments,
    )

    dataloader_kwargs = {
        "pin_memory": (
            accelerator.device.type == "cuda"
            if config.pin_memory is None
            else config.pin_memory
        )
    }
    if config.dataloader_workers > 0:
        dataloader_kwargs.update(
            {
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
                "prefetch_factor": config.prefetch_factor,
            }
        )
    policy = build_sampling_policy(
        shared_dataset.training_patches,
        dataset_config,
        batch_size=config.batch_size,
    )
    accelerator.print(
        "sampling_audit=" + json.dumps(policy.audit, sort_keys=True), flush=True
    )
    if policy.batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=policy.batch_sampler,
            num_workers=config.dataloader_workers,
            **dataloader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=policy.shuffle,
            sampler=policy.sampler,
            generator=policy.generator,
            num_workers=config.dataloader_workers,
            **dataloader_kwargs,
        )
    # Validation consumes only val_steps batches, so shuffle to sample a
    # different deterministic subset on each pass.
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=len(val_dataset) > 0,
        generator=torch.Generator().manual_seed(config.seed + 1),
        num_workers=config.dataloader_workers,
        **dataloader_kwargs,
    )

    model = make_model(config.ink)
    optimizer = create_training_optimizer(model, config)
    scheduler = create_training_scheduler(optimizer, config)
    initialize_training_model(model, config)
    loss = create_loss(config.ink)
    if config.ink.model.enable_deep_supervision:
        task_decoders = getattr(model, "task_decoders", None)
        if task_decoders is None and hasattr(model, "network"):
            task_decoders = getattr(model.network, "task_decoders", None)
        if not task_decoders:
            raise ValueError(
                "enable_deep_supervision requires per-task decoders, but the "
                "current model did not build any"
            )
        first_decoder = next(iter(task_decoders.values()))
        weights = deep_supervision_weights(
            len(getattr(first_decoder, "stages", ()))
        )
        if weights is None:
            raise ValueError(
                "enable_deep_supervision requires at least one decoder supervision stage"
            )
        loss = DeepSupervisionWrapper(loss, weights)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    # NOTE: we intentionally do NOT prepare lr_scheduler with Accelerate.
    # AcceleratedScheduler calls scheduler.step() num_processes times per
    # optimizer step (when split_batches=False), which makes the LR schedule
    # run num_processes-x faster than intended. Instead we step the raw
    # scheduler ourselves exactly once per optimizer step inside the
    # sync_gradients guard.
    unwrapped_model = accelerator.unwrap_model(model)
    freeze_encoder = config.ink.model.freeze_encoder
    frozen_encoder = unwrapped_model.shared_encoder if freeze_encoder else None
    ema_model = deepcopy(unwrapped_model) if config.ema.enabled else None
    if ema_model is not None:
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

    start_step = 0
    optimizer_step = 0
    if request.checkpoint is not None:
        weights_only = config.ink.checkpoint.weights_only
        start_step, optimizer_step = restore_training_state(
            unwrapped_model,
            optimizer,
            scheduler,
            request.checkpoint,
            request.checkpoint_path,
            load_weights_only=weights_only,
            ema_model=ema_model,
        )
        suffix = (
            f" and resuming from step {start_step}"
            if not weights_only
            else " (weights only)"
        )
        accelerator.print(
            f"Loaded checkpoint '{request.checkpoint_path}'{suffix}"
        )

    fixed_batch_sampler = policy.batch_sampler
    gradient_health = {
        "checked_steps": 0,
        "nonfinite_events": [],
        "max_amp_overflow_events": config.max_amp_overflow_events,
    }

    def write_gradient_health() -> None:
        if (
            not accelerator.is_main_process
            or config.verify_finite_gradients_steps <= 0
        ):
            return
        _write_json_replace(
            config.out_dir / "gradient_health.json", gradient_health
        )

    def write_sampling_audit(training_step: int) -> None:
        if fixed_batch_sampler is None or not accelerator.is_main_process:
            return
        observed = fixed_batch_sampler.observed_audit()
        observed["training_step_completed"] = int(training_step)
        observed["target"] = policy.audit
        _write_json_replace(
            config.out_dir / "sampling_observed.json", observed
        )
        accelerator.print(
            "sampling_observed=" + json.dumps(observed, sort_keys=True)
        )

    def save_checkpoint(
        step: int,
        *,
        force: bool = False,
        filename: str | None = None,
        validation_metrics: Mapping | None = None,
    ) -> None:
        if not accelerator.is_main_process or (
            not force and (step + 1) % config.save_every != 0
        ):
            return
        payload = build_training_checkpoint_payload(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=step,
            optimizer_step=optimizer_step,
            ema_model=ema_model,
            wandb_module=wandb_module,
            validation_metrics=validation_metrics,
        )
        torch.save(
            payload,
            config.out_dir / (filename or f"ckpt_{step + 1:06}.pth"),
        )

    write_gradient_health()
    write_sampling_audit(start_step)
    train_iterator = iter(train_loader)
    latest_val_loss = None
    latest_ema_val_loss = None
    best_checkpoint_value = None
    confusion_metric = Confusion()
    progress = tqdm(
        range(start_step, config.num_iterations),
        total=config.num_iterations,
        initial=start_step,
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )
    benchmark_started_at = None
    benchmark_data_wait_seconds = 0.0
    benchmark_measured_steps = 0
    if config.benchmark.enabled and config.benchmark.warmup_steps == 0:
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)
            torch.cuda.reset_peak_memory_stats(accelerator.device)
        benchmark_started_at = time.perf_counter()

    def get_model_input(batch):
        return prepare_model_input(batch, config)

    for step in progress:
        model.train()
        if frozen_encoder is not None:
            frozen_encoder.eval()
        fetch_started_at = time.perf_counter()
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        fetch_seconds = time.perf_counter() - fetch_started_at
        if label_distance > 0.0 or supervision_distance > 0.0:
            batch = apply_label_dilation(
                batch, label_distance, supervision_distance
            )

        with accelerator.accumulate(model):
            with accelerator.autocast():
                predictions = run_model_forward(
                    model,
                    get_model_input(batch),
                    config.model_crop_size,
                    stitched=config.use_stitched_forward,
                    use_gradient_checkpointing=(
                        config.stitched_gradient_checkpointing
                    ),
                )
            loss_predictions, targets, ignore_mask = prepare_loss_inputs(
                predictions, batch, mode=config.ink.data.mode
            )
            primary_predictions = (
                loss_predictions[0]
                if isinstance(loss_predictions, (list, tuple))
                else loss_predictions
            )
            targets_with_ignore = concatenate_deep_supervision_ignore(
                targets, ignore_mask, loss_predictions
            )
            scalar_loss = loss(
                type(loss_predictions)(
                    value.float() for value in loss_predictions
                )
                if isinstance(loss_predictions, (list, tuple))
                else loss_predictions.float(),
                type(targets_with_ignore)(
                    value.float() for value in targets_with_ignore
                )
                if isinstance(targets_with_ignore, (list, tuple))
                else targets_with_ignore.float(),
            )
            if not torch.isfinite(scalar_loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            accelerator.backward(scalar_loss)
            if (
                config.grad_clip is not None
                and config.grad_clip > 0
                and accelerator.sync_gradients
            ):
                accelerator.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
            gradient_event = None
            if (
                step < config.verify_finite_gradients_steps
                and accelerator.sync_gradients
            ):
                gradient_health["checked_steps"] += 1
                nonfinite = [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all().item())
                ]
                if nonfinite:
                    gradient_event = {
                        "step": int(step),
                        "parameters": nonfinite[:10],
                    }
                    gradient_health["nonfinite_events"].append(gradient_event)
                    accelerator.print(
                        f"AMP gradient overflow at step {step}: "
                        + ", ".join(nonfinite[:10])
                    )
                    if len(gradient_health["nonfinite_events"]) > (
                        config.max_amp_overflow_events
                    ):
                        write_gradient_health()
                        raise RuntimeError(
                            "Exceeded permitted AMP gradient overflow events: "
                            f"{len(gradient_health['nonfinite_events'])} > "
                            f"{config.max_amp_overflow_events}"
                        )
            optimizer.step()
            if gradient_event is not None:
                gradient_event["optimizer_step_was_skipped"] = bool(
                    accelerator.optimizer_step_was_skipped
                )
            if (
                step < config.verify_finite_gradients_steps
                and accelerator.sync_gradients
            ):
                write_gradient_health()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                scheduler.step()
                optimizer_step += 1
                ema = config.ema
                if (
                    ema_model is not None
                    and optimizer_step >= ema.start_step
                    and not (
                        (optimizer_step - ema.start_step)
                        % ema.update_every_steps
                    )
                ):
                    ema_state = ema_model.state_dict()
                    for name, model_value in unwrapped_model.state_dict().items():
                        ema_value = ema_state[name]
                        model_value = model_value.detach()
                        if torch.is_floating_point(ema_value):
                            ema_value.lerp_(
                                model_value.to(dtype=ema_value.dtype),
                                1.0 - ema.decay,
                            )
                        else:
                            ema_value.copy_(model_value)

        if config.benchmark.enabled:
            if step >= config.benchmark.warmup_steps:
                benchmark_data_wait_seconds += fetch_seconds
                benchmark_measured_steps += 1
            if step + 1 == config.benchmark.warmup_steps:
                if accelerator.device.type == "cuda":
                    torch.cuda.synchronize(accelerator.device)
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
                benchmark_started_at = time.perf_counter()

        should_log = step % config.log_every == 0
        train_loss = float(
            _distributed_mean_scalar(accelerator, scalar_loss).item()
            if should_log
            else scalar_loss.item()
        )
        if accelerator.is_main_process:
            postfix = {
                "loss": f"{train_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            if latest_val_loss is not None:
                postfix["val_loss"] = f"{latest_val_loss:.4f}"
            if latest_ema_val_loss is not None:
                postfix["ema_val_loss"] = f"{latest_ema_val_loss:.4f}"
            progress.set_postfix(postfix, refresh=False)
            progress.update(0)
        reduced_metrics = None
        if should_log:
            reduced_metrics = _distributed_mean_metrics(
                accelerator, getattr(loss, "latest_metrics", None)
            )
        if accelerator.is_main_process and should_log:
            log_values = {
                "train/loss": train_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "step": step,
                **reduced_metrics,
            }
            if wandb_module is not None and wandb_module.run is not None:
                wandb_module.log(log_values, step=step)

        if step > 0 and step % config.val_every == 0:
            train_preview = PreviewAccumulator(
                accelerator=accelerator, get_model_input=get_model_input
            )
            preview_batch = batch
            preview_predictions = primary_predictions.detach()
            preview_targets = targets.detach()
            preview_ignore = ignore_mask.detach()
            if config.ink.data.mode in {"full_3d", "full_3d_single_wrap"}:
                (
                    preview_batch,
                    preview_predictions,
                    preview_targets,
                    preview_ignore,
                ) = central_full_3d_preview(
                    batch,
                    preview_predictions,
                    preview_targets,
                    preview_ignore,
                )
            train_preview.add_batch(
                preview_batch,
                preview_predictions,
                preview_targets,
                preview_ignore,
            )
            model.eval()
            val_loss_total = torch.zeros(
                (), device=accelerator.device, dtype=torch.float32
            )
            val_bce_total = torch.zeros_like(val_loss_total)
            val_batches = torch.zeros_like(val_loss_total)
            ema_loss_total = torch.zeros_like(val_loss_total)
            ema_batches = torch.zeros_like(val_loss_total)
            counts = Confusion.zero_counts(device=accelerator.device)
            val_preview = PreviewAccumulator(
                accelerator=accelerator, get_model_input=get_model_input
            )
            num_val_batches = min(len(val_loader), config.val_steps)
            if num_val_batches == 0:
                if accelerator.is_main_process:
                    latest_val_loss = None
                    latest_ema_val_loss = None
                save_checkpoint(step)
                continue
            preview_indices = set(
                random.sample(
                    range(num_val_batches),
                    k=min(config.val_preview_batches, num_val_batches),
                )
            )
            val_iterator = iter(val_loader)
            with torch.no_grad():
                for val_index in range(num_val_batches):
                    val_batch = next(val_iterator)
                    if label_distance > 0.0 or supervision_distance > 0.0:
                        val_batch = apply_label_dilation(
                            val_batch, label_distance, supervision_distance
                        )
                    with accelerator.autocast():
                        val_predictions = run_model_forward(
                            model,
                            get_model_input(val_batch),
                            config.model_crop_size,
                            stitched=config.use_stitched_forward,
                            use_gradient_checkpointing=(
                                config.stitched_gradient_checkpointing
                            ),
                        )
                    (
                        val_loss_predictions,
                        val_targets,
                        val_ignore,
                    ) = prepare_loss_inputs(
                        val_predictions,
                        val_batch,
                        mode=config.ink.data.mode,
                    )
                    primary_val_predictions = (
                        val_loss_predictions[0]
                        if isinstance(val_loss_predictions, (list, tuple))
                        else val_loss_predictions
                    )
                    val_targets_with_ignore = (
                        concatenate_deep_supervision_ignore(
                            val_targets, val_ignore, val_loss_predictions
                        )
                    )
                    val_loss = loss(
                        type(val_loss_predictions)(
                            value.float() for value in val_loss_predictions
                        )
                        if isinstance(val_loss_predictions, (list, tuple))
                        else val_loss_predictions.float(),
                        type(val_targets_with_ignore)(
                            value.float() for value in val_targets_with_ignore
                        )
                        if isinstance(val_targets_with_ignore, (list, tuple))
                        else val_targets_with_ignore.float(),
                    )
                    val_loss_total += _distributed_mean_scalar(
                        accelerator, val_loss
                    )
                    val_bce_total += _distributed_mean_scalar(
                        accelerator,
                        masked_unsmoothed_bce_with_logits(
                            primary_val_predictions, val_targets, val_ignore
                        ),
                    )
                    val_batches += 1.0
                    batch_counts = confusion_metric.compute_batch(
                        MetricBatch(
                            logits=primary_val_predictions.detach(),
                            targets=val_targets.detach(),
                            valid_mask=(val_ignore <= 0).detach(),
                        )
                    )
                    gathered = accelerator.gather_for_metrics(
                        torch.stack(
                            (
                                batch_counts.tp,
                                batch_counts.fp,
                                batch_counts.fn,
                                batch_counts.tn,
                            )
                        ).unsqueeze(0)
                    )
                    counts = Confusion.add_counts(
                        counts,
                        ConfusionCounts(
                            tp=gathered[:, 0].sum(),
                            fp=gathered[:, 1].sum(),
                            fn=gathered[:, 2].sum(),
                            tn=gathered[:, 3].sum(),
                        ),
                    )
                    displayed_predictions = primary_val_predictions
                    if ema_model is not None and config.ema.validate:
                        with accelerator.autocast():
                            ema_predictions = run_model_forward(
                                ema_model,
                                get_model_input(val_batch),
                                config.model_crop_size,
                                stitched=config.use_stitched_forward,
                                use_gradient_checkpointing=(
                                    config.stitched_gradient_checkpointing
                                ),
                            )
                        ema_loss_predictions, _, _ = prepare_loss_inputs(
                            ema_predictions,
                            val_batch,
                            mode=config.ink.data.mode,
                        )
                        ema_targets_with_ignore = (
                            concatenate_deep_supervision_ignore(
                                val_targets,
                                val_ignore,
                                ema_loss_predictions,
                            )
                        )
                        ema_val_loss = loss(
                            type(ema_loss_predictions)(
                                value.float() for value in ema_loss_predictions
                            )
                            if isinstance(ema_loss_predictions, (list, tuple))
                            else ema_loss_predictions.float(),
                            type(ema_targets_with_ignore)(
                                value.float()
                                for value in ema_targets_with_ignore
                            )
                            if isinstance(ema_targets_with_ignore, (list, tuple))
                            else ema_targets_with_ignore.float(),
                        )
                        ema_loss_total += _distributed_mean_scalar(
                            accelerator, ema_val_loss
                        )
                        ema_batches += 1.0
                        displayed_predictions = (
                            ema_loss_predictions[0]
                            if isinstance(ema_loss_predictions, (list, tuple))
                            else ema_loss_predictions
                        )
                    if val_index in preview_indices:
                        preview_batch = val_batch
                        preview_predictions = displayed_predictions.detach()
                        preview_targets = val_targets.detach()
                        preview_ignore = val_ignore.detach()
                        if config.ink.data.mode in {
                            "full_3d",
                            "full_3d_single_wrap",
                        }:
                            (
                                preview_batch,
                                preview_predictions,
                                preview_targets,
                                preview_ignore,
                            ) = central_full_3d_preview(
                                val_batch,
                                preview_predictions,
                                preview_targets,
                                preview_ignore,
                            )
                        val_preview.add_batch(
                            preview_batch,
                            preview_predictions,
                            preview_targets,
                            preview_ignore,
                        )

            mean_val_loss = float((val_loss_total / val_batches).item())
            mean_val_bce = float((val_bce_total / val_batches).item())
            mean_ema_loss = (
                None
                if float(ema_batches.item()) == 0.0
                else float((ema_loss_total / ema_batches).item())
            )
            if accelerator.is_main_process:
                latest_val_loss = mean_val_loss
                latest_ema_val_loss = mean_ema_loss
                balanced_accuracy = float(
                    BalancedAccuracy.from_counts(counts).item()
                )
                log_values = build_validation_preview_log(
                    step=step,
                    train_preview=train_preview,
                    val_preview=val_preview,
                    train_preview_dir=train_preview_dir,
                    val_preview_dir=val_preview_dir,
                    mean_val_loss=mean_val_loss,
                    mean_ema_val_loss=mean_ema_loss,
                    include_wandb_images=(
                        wandb_module is not None
                        and wandb_module.run is not None
                    ),
                )
                log_values.update(
                    {
                        "val/bce_unsmoothed": mean_val_bce,
                        "val/balanced_accuracy": balanced_accuracy,
                        "val/tp": float(counts.tp.item()),
                        "val/fp": float(counts.fp.item()),
                        "val/fn": float(counts.fn.item()),
                        "val/tn": float(counts.tn.item()),
                    }
                )
                record = {
                    "step": int(step),
                    "val_loss": mean_val_loss,
                    "val_bce_unsmoothed": mean_val_bce,
                    "val_balanced_accuracy": balanced_accuracy,
                    "val_tp": float(counts.tp.item()),
                    "val_fp": float(counts.fp.item()),
                    "val_fn": float(counts.fn.item()),
                    "val_tn": float(counts.tn.item()),
                    "val_batches": int(num_val_batches),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                append_validation_metrics(
                    config.out_dir / "validation_metrics.jsonl", record
                )
                if config.best_checkpoint_metric is not None:
                    candidate = record[config.best_checkpoint_metric]
                    improved = (
                        best_checkpoint_value is None
                        or (
                            config.best_checkpoint_metric == "val_loss"
                            and candidate < best_checkpoint_value
                        )
                        or (
                            config.best_checkpoint_metric
                            == "val_balanced_accuracy"
                            and candidate > best_checkpoint_value
                        )
                    )
                    if improved:
                        best_checkpoint_value = float(candidate)
                        best_name = (
                            f"best_{config.best_checkpoint_metric}.pth"
                        )
                        save_checkpoint(
                            step,
                            force=True,
                            filename=best_name,
                            validation_metrics=record,
                        )
                        with (config.out_dir / "best_checkpoint.json").open(
                            "w", encoding="utf-8"
                        ) as stream:
                            json.dump(
                                {
                                    "checkpoint": best_name,
                                    "metric": config.best_checkpoint_metric,
                                    "value": best_checkpoint_value,
                                    "step": int(step),
                                },
                                stream,
                                indent=2,
                                sort_keys=True,
                            )
                            stream.write("\n")
                if wandb_module is not None and wandb_module.run is not None:
                    wandb_module.log(log_values, step=step)

        save_checkpoint(step)
        if fixed_batch_sampler is not None and (
            (step + 1) % config.sampling_audit_every == 0
            or step + 1 == config.num_iterations
        ):
            write_sampling_audit(step + 1)

    accelerator.wait_for_everyone()
    if config.benchmark.enabled:
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)
        if benchmark_started_at is None:
            raise RuntimeError("benchmark timer never started")
        summary = benchmark_summary(
            elapsed_seconds=time.perf_counter() - benchmark_started_at,
            measured_steps=benchmark_measured_steps,
            batch_size=config.batch_size,
            world_size=accelerator.num_processes,
            data_wait_seconds=benchmark_data_wait_seconds,
            peak_allocated_bytes=(
                torch.cuda.max_memory_allocated(accelerator.device)
                if accelerator.device.type == "cuda"
                else 0
            ),
            peak_reserved_bytes=(
                torch.cuda.max_memory_reserved(accelerator.device)
                if accelerator.device.type == "cuda"
                else 0
            ),
        )
        summary.update(
            {
                "warmup_steps": config.benchmark.warmup_steps,
                "device": str(accelerator.device),
            }
        )
        if accelerator.is_main_process:
            output_path = (
                config.benchmark.output_path
                or config.out_dir / "benchmark_summary.json"
            )
            with Path(output_path).open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, sort_keys=True)
                stream.write("\n")
            accelerator.print(f"benchmark_summary={output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = HyphenUnderscoreParser(
        description="Train an ink-detection model from a JSON configuration."
    )
    parser.add_argument("config_path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run training through an import-pure argparse command boundary."""

    arguments = build_parser().parse_args(argv)
    request = stage_training_request(arguments.config_path)
    return _run_training(request)


if __name__ == "__main__":
    raise SystemExit(main())
