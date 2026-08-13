#!/usr/bin/env python
"""Trainer for the winding phase model.

From a ray-aligned 3-D slab, the model predicts — for every supervised
column at its transverse column stride — a monotone relative winding phase
(whose softplus increments are per-segment winding-density integrals) and,
when the optional heads are configured, a per-sample log-variance for those
increments and a crossing heatmap: the dense observations consumed by
fit_spiral's winding losses
(volume-cartographer/scripts/spiral/fit_spiral.py). Phase is supervised
shift-invariantly with a single free offset per slab (winding indices are
globally consistent across a slab's columns) and canonicalized to increase
along the ray axis; the consumer applies each ray's known winding
direction, registers the free offset, and — with the variance head —
precision-weights increments by exp(-log_variance). Without the crossing
head, crossings decode as the phase's integer passages; without the
variance head, increments are treated as homoscedastic.
"""

import copy
import json
import math
import os
import random
from pathlib import Path

# Every dataloader worker gets its own vc render-sampler pool; the default
# (8 threads per process) oversubscribes the host badly at ~200 workers.
# Set before vc loads so worker forks inherit it; export the variable to
# override.
os.environ.setdefault("VC_RENDER_SAMPLER_THREADS", "1")
# Near-full GPUs make the caching allocator synchronize-and-retry on
# fragmentation, which shows up as random ranks stalling the DDP collective.
# Must be set before torch initializes CUDA; export the variable to override.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import accelerate
import click
import numpy as np
import torch
import wandb
from tqdm import tqdm

from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.neural_tracing.nets.models import strip_state
from vesuvius.neural_tracing.winding_models import winding_targets
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel
from vesuvius.neural_tracing.winding_models.winding_model_dataset import (
    WindingModelDataset,
)
from vesuvius.neural_tracing.winding_models.winding_visualization import (
    make_winding_visualization,
)


def seed_worker(worker_id):
    """Seed worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_losses(output, batch, config, pair_output=None):
    phase = winding_targets.phase_loss(
        output["phase"],
        batch["phase_target"],
        batch["phase_valid"],
        huber_delta=float(config.get("phase_huber_delta", 0.25)),
    )
    # Heteroscedastic beta-NLL on the model's per-segment phase increments,
    # compared against the rendered per-segment label integrals; also trains
    # the log-variance the consumer uses to precision-weight registered
    # observations.
    density = winding_targets.density_loss(
        output["phase_increments"],
        output.get("density_log_variance"),
        batch["density_target"],
        batch["density_gap_wv"],
        batch["phase_valid"],
        min_gap_wv=float(config.get("density_min_gap_wv", 4.0)),
        beta=float(config.get("density_nll_beta", 0.5)),
        huber_delta=float(config.get("density_huber_delta", 0.05)),
    )
    # Winding-count supervision across label holes: gaps whose endpoint
    # indices differ by more than one are masked out of every dense target,
    # but the endpoint indices still pin the phase delta across the gap.
    span_delta = winding_targets.span_delta_loss(
        output["phase"],
        batch["crossing_t"],
        batch["crossing_indices"],
        batch["num_crossings"],
        spacing=float(config.get("spacing", 1.0)),
        min_index_delta=int(config.get("span_delta_min_index_delta", 2)),
        huber_delta=float(config.get("span_delta_huber_delta", 0.25)),
    )
    # Weak phase supervision on position-only columns (patch labels): the
    # phase difference across each labeled gap must be >= 1 winding (hinge)
    # and integer-valued (snap, gated on near-integer predictions).
    position_hinge, position_snap = winding_targets.position_only_phase_loss(
        output["phase"],
        batch["crossing_t"],
        batch["num_crossings"],
        batch["phase_valid"],
        spacing=float(config.get("spacing", 1.0)),
        snap_gate=float(config.get("position_snap_gate", 0.25)),
    )
    total = (
        float(config.get("lambda_phase", 1.0)) * phase
        + float(config.get("lambda_density", 1.0)) * density
        + float(config.get("lambda_span_delta", 0.0)) * span_delta
        + float(config.get("lambda_position_hinge", 0.0)) * position_hinge
        + float(config.get("lambda_position_snap", 0.0)) * position_snap
    )
    metrics = {
        "phase_loss": phase,
        "density_loss": density,
        "span_delta_loss": span_delta,
        "position_hinge_loss": position_hinge,
        "position_snap_loss": position_snap,
    }

    # The crossing-head losses exist only when the model has the head
    # (use_crossing_head; kept for older checkpoints).
    if "crossing_logits" in output:
        crossing = winding_targets.crossing_loss(
            output["crossing_logits"],
            batch["crossing_target"],
            batch["crossing_valid"],
            alpha=float(config.get("crossing_centernet_alpha", 2.0)),
            beta=float(config.get("crossing_centernet_beta", 4.0)),
        )
        # Head agreement: over each supervised span the heatmap's crossing
        # count and the integrated phase increments must tell one story, so
        # the consumer's two observation channels don't disagree.
        consistency = winding_targets.head_consistency_loss(
            output["crossing_logits"],
            output["phase_increments"],
            batch["phase_valid"],
            batch["crossing_target"],
            crossing_sigma_wv=float(config.get("crossing_sigma_wv", 1.0)),
            spacing=float(config.get("spacing", 1.0)),
            huber_delta=float(config.get("consistency_huber_delta", 0.5)),
        )
        # Phase-integer self-distillation: the trusted phase head teaches the
        # crossing head where the focal loss is silent (~crossing_valid).
        distillation = winding_targets.crossing_distillation_loss(
            output["crossing_logits"],
            output["phase"],
            batch["phase_target"],
            batch["phase_valid"],
            batch["crossing_valid"],
            crossing_sigma_wv=float(config.get("crossing_sigma_wv", 1.0)),
            spacing=float(config.get("spacing", 1.0)),
        )
        total = (
            total
            + float(config.get("lambda_crossing", 1.0)) * crossing
            + float(config.get("lambda_consistency", 0.1)) * consistency
            + float(config.get("lambda_crossing_distill", 0.0)) * distillation
        )
        metrics["crossing_loss"] = crossing
        metrics["consistency_loss"] = consistency
        metrics["distill_loss"] = distillation

    # Label-free multiview consistency between the sample's two overlapping
    # slabs. The metric key is emitted whenever pairs are configured (even
    # on batches that drew none) so logging keys stay step-invariant.
    lambda_multiview = float(config.get("lambda_multiview", 0.0))
    if lambda_multiview > 0.0 and "pair_image" in batch:
        if pair_output is None:
            multiview = output["phase"].new_zeros(())
        else:
            # forward_with_pair compacts the second forward to real pairs.
            # Apply the same selector to the primary predictions and both
            # frames so the multiview loss sees aligned, dense mini-batches.
            has_pair = batch["has_pair"].bool()
            pair_count = int(has_pair.sum())
            if pair_output["phase"].shape[0] != pair_count:
                raise ValueError(
                    "pair output batch does not match the selected pairs"
                )
            multiview = winding_targets.multiview_consistency_loss(
                output["phase"][has_pair],
                pair_output["phase"],
                {
                    "origin": batch["slab_origin_zyx"][has_pair],
                    "axis_a": batch["slab_axis_a_zyx"][has_pair],
                    "axis_b": batch["slab_axis_b_zyx"][has_pair],
                    "direction": batch["ray_direction_zyx"][has_pair],
                },
                {
                    "origin": batch["pair_origin_zyx"][has_pair],
                    "axis_a": batch["pair_axis_a_zyx"][has_pair],
                    "axis_b": batch["pair_axis_b_zyx"][has_pair],
                    "direction": batch["pair_direction_zyx"][has_pair],
                },
                batch["slab_valid"][has_pair],
                batch["pair_valid"][has_pair],
                torch.ones_like(batch["has_pair"][has_pair], dtype=torch.bool),
                column_stride=int(config.get("column_stride", 4)),
                spacing=float(config.get("spacing", 1.0)),
                huber_delta=float(config.get("multiview_huber_delta", 0.5)),
            )
        total = total + lambda_multiview * multiview
        metrics["multiview_loss"] = multiview
    return total, metrics


def forward_with_pair(model, batch, config):
    """Model outputs for the batch and, when configured, its pair slabs.

    The pair forward is skipped when the batch drew no pairs; the loss key
    still appears (as zero) via compute_losses so logging stays uniform.
    """
    output = model(batch["slab_image"], batch["slab_valid"])
    pair_output = None
    if (
        float(config.get("lambda_multiview", 0.0)) > 0.0
        and "pair_image" in batch
        and bool(batch["has_pair"].any())
    ):
        has_pair = batch["has_pair"].bool()
        pair_output = model(
            batch["pair_image"][has_pair], batch["pair_valid"][has_pair]
        )
    return output, pair_output


def peak_decoding_config(config):
    spacing = float(config.get("spacing", 1.0))
    min_distance_wv = float(
        config.get(
            "crossing_peak_min_distance_wv",
            2.0 * float(config.get("crossing_sigma_wv", 1.0)),
        )
    )
    return {
        "threshold": float(config.get("crossing_peak_threshold", 0.3)),
        "min_distance": max(1, round(min_distance_wv / spacing)),
    }


def accumulate_winding_metrics(sums, output, batch, config):
    """Consumer-facing quality measures over all supervised columns.

    Crossing detection precision/recall within a physical tolerance,
    slab-centered phase error (one free offset per slab, matching the
    loss), and per-column winding-count error over each column's valid
    span (the quantity fit_spiral's density loss integrates).
    """
    spacing = float(config.get("spacing", 1.0))
    tolerance = float(config.get("crossing_match_tolerance_wv", 2.0)) / spacing
    decoding = peak_decoding_config(config)

    logits = output.get("crossing_logits")
    prob = (
        torch.sigmoid(logits.detach().float()).cpu().numpy()
        if logits is not None
        else None
    )
    phase_pred = output["phase"].detach().float().cpu().numpy()
    phase_target = batch["phase_target"].float().cpu().numpy()
    phase_valid = batch["phase_valid"].cpu().numpy().astype(bool)
    crossing_valid = batch["crossing_valid"].cpu().numpy().astype(bool)
    crossing_t = batch["crossing_t"].float().cpu().numpy()
    num_crossings = batch["num_crossings"].cpu().numpy()
    increments = output["phase_increments"].detach().float().cpu().numpy()
    log_variance = output.get("density_log_variance")
    sigma = (
        (0.5 * log_variance.detach().float()).exp().cpu().numpy()
        if log_variance is not None
        else None
    )
    # Density metrics reuse the exact loss targets and mask: each increment is
    # a per-segment winding integral compared against the rendered label
    # integral, on supervised segments only.
    density_target = batch["density_target"].float().cpu().numpy()
    density_mask = (
        winding_targets.density_supervision_mask(
            batch["density_gap_wv"].cpu(),
            batch["phase_valid"].cpu(),
            min_gap_wv=float(config.get("density_min_gap_wv", 4.0)),
        )
        .numpy()
        .astype(bool)
    )

    batch_size, length = phase_pred.shape[0], phase_pred.shape[-1]
    flat = lambda values: values.reshape(-1, values.shape[-1])  # noqa: E731
    prob_flat = flat(prob) if prob is not None else None
    phase_pred_flat = flat(phase_pred)
    phase_target_flat = flat(phase_target)
    phase_valid_flat = flat(phase_valid)
    crossing_valid_flat = flat(crossing_valid)
    crossing_t_flat = flat(crossing_t)
    increments_flat = flat(increments)
    sigma_flat = flat(sigma) if sigma is not None else None
    density_target_flat = flat(density_target)
    density_mask_flat = flat(density_mask)
    counts_flat = num_crossings.reshape(-1)
    columns_per_slab = phase_pred_flat.shape[0] // batch_size

    # Without a crossing head, crossings decode as the integer passages of
    # the phase, registered per slab against the targets (one free offset,
    # matching the phase loss).
    passage_offsets = np.zeros(batch_size)
    if prob_flat is None:
        for slab in range(batch_size):
            valid = phase_valid[slab].reshape(-1)
            if valid.any():
                passage_offsets[slab] = float(
                    phase_pred[slab].reshape(-1)[valid].mean()
                    - phase_target[slab].reshape(-1)[valid].mean()
                )

    for column in range(len(phase_pred_flat)):
        if counts_flat[column] < 2:
            continue
        if prob_flat is not None:
            peaks = winding_targets.extract_peaks(prob_flat[column], **decoding)
            peak_positions = peaks.astype(np.float64)
        else:
            peak_positions, _ = winding_targets.phase_passages(
                phase_pred_flat[column]
                - passage_offsets[column // columns_per_slab]
            )
            peaks = np.clip(
                np.rint(peak_positions).astype(np.int64), 0, length - 1
            )
        # Peaks in unlabeled spans may be real unlabeled wraps; they are not
        # decidable and must not count as false positives.
        keep = crossing_valid_flat[column][peaks]
        peak_positions = peak_positions[keep]
        tp, fp, fn = winding_targets.match_crossings(
            peak_positions,
            crossing_t_flat[column, : counts_flat[column]] / spacing,
            tolerance=tolerance,
        )
        sums["crossing_tp"] += tp
        sums["crossing_fp"] += fp
        sums["crossing_fn"] += fn

        valid = phase_valid_flat[column]
        if valid.sum() >= 2:
            pred = phase_pred_flat[column][valid]
            target = phase_target_flat[column][valid]
            sums["count_error"] += float(
                abs((pred[-1] - pred[0]) - (target[-1] - target[0]))
            )
            sums["count_columns"] += 1

        mask = density_mask_flat[column]
        if mask.any():
            sums["density_abs_error"] = sums.get("density_abs_error", 0.0) + float(
                np.abs(
                    increments_flat[column][mask] - density_target_flat[column][mask]
                ).mean()
            )
            if sigma_flat is not None:
                sums["density_mean_sigma"] = sums.get(
                    "density_mean_sigma", 0.0
                ) + float(sigma_flat[column][mask].mean())
                sums["density_sigma_columns"] = (
                    sums.get("density_sigma_columns", 0) + 1
                )
            sums["density_columns"] = sums.get("density_columns", 0) + 1

    # Phase error with one free offset per slab, matching the loss and the
    # cross-column coherence the consumer relies on.
    for slab in range(batch_size):
        valid = phase_valid[slab].reshape(-1)
        if valid.sum() < 2:
            continue
        pred = phase_pred[slab].reshape(-1)[valid]
        target = phase_target[slab].reshape(-1)[valid]
        centered = (pred - pred.mean()) - (target - target.mean())
        sums["phase_abs_error"] += float(np.abs(centered).mean())
        sums["phase_slabs"] += 1


def finalize_winding_metrics(sums):
    metrics = {
        "val_crossing_precision": sums["crossing_tp"]
        / max(1, sums["crossing_tp"] + sums["crossing_fp"]),
        "val_crossing_recall": sums["crossing_tp"]
        / max(1, sums["crossing_tp"] + sums["crossing_fn"]),
    }
    if sums["phase_slabs"]:
        metrics["val_phase_mae"] = sums["phase_abs_error"] / sums["phase_slabs"]
    if sums["count_columns"]:
        metrics["val_winding_count_error"] = (
            sums["count_error"] / sums["count_columns"]
        )
    if sums.get("density_columns"):
        metrics["val_density_mae"] = (
            sums["density_abs_error"] / sums["density_columns"]
        )
    if sums.get("density_sigma_columns"):
        metrics["val_density_mean_sigma"] = (
            sums["density_mean_sigma"] / sums["density_sigma_columns"]
        )
    return metrics


def make_dataloader(dataset, config, *, generator, num_workers):
    kwargs = {
        "batch_size": int(config.get("batch_size", 8)),
        # Samples are drawn procedurally (__getitem__ ignores its index), so
        # shuffling the index stream would be a no-op.
        "shuffle": False,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": True,
        "collate_fn": winding_targets.collate_winding_batch,
        "pin_memory": bool(config.get("pin_memory", True)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = max(1, int(config.get("prefetch_factor", 2)))
        kwargs["multiprocessing_context"] = "fork"
        # Draws are procedural, so batch order carries no meaning; letting
        # finished batches overtake a straggling worker keeps one slow draw
        # from stalling the rank (and with DDP, every other rank).
        kwargs["in_order"] = bool(config.get("dataloader_in_order", False))
    return torch.utils.data.DataLoader(dataset, **kwargs)


def dispatched_num_workers(accelerator, requested, dispatch_batches):
    """Use one worker pool when rank zero dispatches global batches.

    ``DataLoaderDispatcher`` constructs an iterator on every rank for PyTorch
    compatibility, even though only rank zero consumes its iterator.  Giving
    nonzero ranks workers would therefore start idle duplicate pools.
    """
    requested = max(0, int(requested))
    if dispatch_batches and not accelerator.is_main_process:
        return 0
    return requested


def checkpoint_model_state_dict(accelerator, model):
    """Return a wrapper-free model state dict for portable checkpoints."""
    return strip_state(accelerator.unwrap_model(model).state_dict())


def initialize_datasets(accelerator, config, *, build_validation=True):
    """Build caches on rank zero before the remaining ranks load them.

    Dataset construction may create the shared segment cache and patch mmap
    pack.  Serializing the first construction prevents every distributed rank
    from doing the same expensive writes and racing to publish the pack.
    """
    with accelerator.main_process_first():
        train_dataset = WindingModelDataset(config)
        if not build_validation:
            val_dataset = None
        elif "val_datasets" in config:
            val_config = dict(config)
            val_config["datasets"] = config["val_datasets"]
            val_dataset = WindingModelDataset(val_config)
        else:
            # Sampling is procedural, so a second instance over the same
            # segments would only duplicate raycaster construction;
            # validation batches are simply fresh draws.
            val_dataset = train_dataset
    return train_dataset, val_dataset


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path):
    """Train a winding phase + crossing model."""

    with open(config_path) as config_file:
        config = json.load(config_file)
    base_config_path = config.pop("base_config", None)
    if base_config_path is not None:
        base_path = Path(config_path).parent / base_config_path
        with base_path.open() as base_config_file:
            resolved = json.load(base_config_file)
        model_overrides = config.pop("model", {})
        resolved.update(config)
        resolved["model"] = {**resolved.get("model", {}), **model_overrides}
        config = resolved

    # The model sizes its transverse attention bias to cover the whole trunk
    # plane, so it needs the slab's transverse size; inject it before the
    # config is saved so checkpoints carry the resolved value.
    model_config = dict(config.get("model") or {})
    model_config["transverse_size"] = int(config.get("transverse_size", 96))
    config["model"] = model_config

    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/config.json", "w") as output_config:
        json.dump(config, output_config, indent=2)
        output_config.write("\n")
    metrics_path = f"{out_dir}/metrics.jsonl"

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dispatch_batches = bool(config.get("dispatch_batches", False))
    validation_mode = str(config.get("validation_mode", "separate"))
    if validation_mode not in {"separate", "train_batch"}:
        raise ValueError(
            "validation_mode must be either 'separate' or 'train_batch'"
        )
    if validation_mode == "train_batch" and "val_datasets" in config:
        raise ValueError(
            "validation_mode='train_batch' cannot be used with val_datasets"
        )
    accelerator = accelerate.Accelerator(
        mixed_precision=str(config.get("mixed_precision", "bf16")),
        gradient_accumulation_steps=int(config.get("grad_acc_steps", 1)),
        dataloader_config=accelerate.DataLoaderConfiguration(
            # Rank zero fetches num_processes per-device batches per step —
            # typically from different workers, so one slow sample delays
            # only its own small batch instead of a serially collated global
            # batch — concatenates them, and dispatches one slice per rank.
            # The dataset pads crossing lists to a fixed width, which keeps
            # independently collated batches concatenation-compatible.
            dispatch_batches=dispatch_batches,
        ),
    )

    if "wandb_project" in config and accelerator.is_main_process:
        wandb.init(
            project=config["wandb_project"],
            entity=config.get("wandb_entity"),
            config=config,
        )

    train_dataset, val_dataset = initialize_datasets(
        accelerator,
        config,
        build_validation=validation_mode == "separate",
    )

    def make_generator(offset):
        generator = torch.Generator()
        generator.manual_seed(seed + accelerator.process_index * 1000 + offset)
        return generator

    train_dataloader = make_dataloader(
        train_dataset,
        config,
        generator=make_generator(0),
        num_workers=dispatched_num_workers(
            accelerator, config.get("num_workers", 8), dispatch_batches
        ),
    )
    val_dataloader = None
    if validation_mode == "separate":
        val_dataloader = make_dataloader(
            val_dataset,
            config,
            generator=make_generator(1),
            num_workers=dispatched_num_workers(
                accelerator, config.get("val_num_workers", 2), dispatch_batches
            ),
        )

    model = WindingModel(config.get("model"))
    if model.column_stride != train_dataset.column_stride:
        raise ValueError(
            f"model column stride {model.column_stride} does not match the "
            f"dataset column stride {train_dataset.column_stride}"
        )
    ema_decay = float(config.get("ema_decay", 0.0))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    ema_start_step = int(config.get("ema_start_step", 0))
    if ema_start_step < 0:
        raise ValueError("ema_start_step must be non-negative")
    ema_model = copy.deepcopy(model) if ema_decay else None
    if ema_model is not None:
        ema_model.requires_grad_(False)
    if config.get("compile_model", False):
        model = torch.compile(model)
        accelerator.print("Model compiled with torch.compile")

    num_iterations = int(config["num_iterations"])
    learning_rate = float(config.get("learning_rate", 3e-4))
    optimizer = create_optimizer(
        {
            "name": config.get("optimizer", "adamw"),
            "learning_rate": learning_rate,
            "weight_decay": float(config.get("weight_decay", 0.01)),
        },
        model,
    )
    # The scheduler advances once per optimizer update (every grad_acc_steps
    # iterations), while num_iterations and scheduler_kwargs count training
    # iterations; convert every step-denominated value to optimizer steps so
    # warmup and decay track iterations regardless of gradient accumulation.
    grad_acc_steps = max(1, int(config.get("grad_acc_steps", 1)))
    scheduler_kwargs = dict(config.get("scheduler_kwargs", {}))
    for key in ("warmup_steps", "first_cycle_steps", "step_size"):
        if key in scheduler_kwargs:
            scheduler_kwargs[key] = max(
                1, -(-int(scheduler_kwargs[key]) // grad_acc_steps)
            )
    lr_scheduler = get_scheduler(
        scheduler_type=str(config.get("scheduler", "diffusers_cosine_warmup")),
        optimizer=optimizer,
        initial_lr=learning_rate,
        max_steps=max(1, -(-num_iterations // grad_acc_steps)),
        **scheduler_kwargs,
    )

    start_iteration = 0
    if "load_ckpt" in config:
        accelerator.print(f"Loading checkpoint {config['load_ckpt']}")
        ckpt = torch.load(config["load_ckpt"], map_location="cpu", weights_only=False)
        # Weights-only warm starts may grow the architecture (e.g. appended
        # attention blocks stay fresh-initialized); a full resume must match
        # the checkpoint exactly.
        strict = not config.get("load_weights_only", False)
        target = accelerator.unwrap_model(model)
        # Checkpoints hold strip_state names; torch.compile prefixes every
        # key with _orig_mod., so load into the wrapped module itself.
        target = getattr(target, "_orig_mod", target)
        missing, unexpected = target.load_state_dict(ckpt["model"], strict=strict)
        if missing:
            accelerator.print(f"  fresh-initialized keys: {sorted(missing)}")
        if unexpected:
            accelerator.print(f"  ignored checkpoint keys: {sorted(unexpected)}")
        if len(missing) + len(unexpected) >= len(ckpt["model"]):
            raise RuntimeError(
                "checkpoint and model state dicts have no overlapping keys"
            )
        if ema_model is not None:
            ema_model.load_state_dict(
                ckpt.get("model_ema", ckpt["model"]), strict=strict
            )
        if not config.get("load_weights_only", False):
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            start_iteration = int(ckpt.get("step", 0))

    # Keep the scheduler out of accelerator.prepare: prepared schedulers are
    # advanced once per process with sharded dataloaders, while num_iterations
    # counts optimizer-update iterations.
    if val_dataloader is None:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
    else:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )
    if ema_model is not None:
        ema_model.to(accelerator.device)

    spacing = float(config.get("spacing", 1.0))
    decoding = peak_decoding_config(config)
    # grad_clip <= 0 disables clipping; the gradient norm is still computed
    # for logging and the non-finite step skip.
    grad_clip = float(config.get("grad_clip", 0.0))
    if grad_clip <= 0.0:
        grad_clip = math.inf
    log_frequency = int(config.get("log_frequency", 250))
    ckpt_frequency = int(config.get("ckpt_frequency", 5000))
    val_batches_per_log = max(1, int(config.get("val_batches_per_log", 4)))

    def save_checkpoint(step, name):
        checkpoint = {
            "model": checkpoint_model_state_dict(accelerator, model),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "config": config,
            "step": step,
        }
        if ema_model is not None:
            checkpoint["model_ema"] = strip_state(ema_model.state_dict())
        torch.save(checkpoint, f"{out_dir}/{name}.pth")

    train_iterator = iter(train_dataloader)
    val_iterator = iter(val_dataloader) if val_dataloader is not None else None
    progress_bar = tqdm(
        total=num_iterations,
        initial=start_iteration,
        disable=(
            not accelerator.is_local_main_process
            or bool(config.get("disable_progress", False))
        ),
    )

    for iteration in range(start_iteration, num_iterations):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        wandb_log = {}
        grad_norm = None
        did_optimizer_step = False
        with accelerator.accumulate(model):
            output, pair_output = forward_with_pair(model, batch, config)
            total_loss, loss_metrics = compute_losses(
                output, batch, config, pair_output
            )
            wandb_log.update(
                {key: value.detach().item() for key, value in loss_metrics.items()}
            )

            if torch.isnan(total_loss).any():
                raise ValueError("loss is NaN")

            do_optimizer_step = True
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                grad_norm = float(
                    grad_norm.detach().item()
                    if torch.is_tensor(grad_norm)
                    else grad_norm
                )
                if not np.isfinite(grad_norm):
                    do_optimizer_step = False
                    accelerator.print(
                        f"Warning: non-finite grad norm at iteration {iteration};"
                        " skipping optimizer step"
                    )
                    wandb_log["skipped_step_nonfinite_grad"] = 1.0
            if do_optimizer_step:
                optimizer.step()
                if accelerator.sync_gradients and not getattr(
                    optimizer, "step_was_skipped", False
                ):
                    lr_scheduler.step()
                    did_optimizer_step = True
            optimizer.zero_grad()

        if did_optimizer_step and ema_model is not None:
            with torch.no_grad():
                online = accelerator.unwrap_model(model)
                for ema_parameter, parameter in zip(
                    ema_model.parameters(), online.parameters(), strict=True
                ):
                    if iteration < ema_start_step:
                        ema_parameter.copy_(parameter.detach())
                    else:
                        ema_parameter.lerp_(parameter.detach(), 1.0 - ema_decay)

        wandb_log["loss"] = total_loss.detach().item()
        wandb_log["current_lr"] = optimizer.param_groups[0]["lr"]
        if grad_norm is not None:
            wandb_log["grad_norm"] = grad_norm

        postfix = {
            "loss": f"{wandb_log['loss']:.4f}",
            "phase": f"{wandb_log['phase_loss']:.4f}",
        }
        # Headless models emit no crossing loss; show the count-bearing
        # terms that exist instead.
        for key, label in (
            ("crossing_loss", "crossing"),
            ("span_delta_loss", "span"),
            ("multiview_loss", "mview"),
        ):
            if key in wandb_log:
                postfix[label] = f"{wandb_log[key]:.4f}"
        progress_bar.set_postfix(postfix)
        progress_bar.update(1)

        should_log = (
            (iteration > 0 or bool(config.get("log_at_step_zero", False)))
            and iteration % log_frequency == 0
        )
        # With separate validation, dispatched dataloaders broadcast every
        # batch collectively, so every rank must advance validation together.
        # ``train_batch`` instead reuses this iteration's predictions: it
        # creates no validation worker pool and incurs no extra model forward.
        # These diagnostics retain the val_* names for dashboard continuity,
        # but are in-sample metrics rather than held-out validation.
        if should_log:
            with torch.no_grad():
                val_sums = {
                    "val_loss": 0.0,
                    "crossing_tp": 0,
                    "crossing_fp": 0,
                    "crossing_fn": 0,
                    "phase_abs_error": 0.0,
                    "phase_slabs": 0,
                    "count_error": 0.0,
                    "count_columns": 0,
                    "density_abs_error": 0.0,
                    "density_mean_sigma": 0.0,
                    "density_columns": 0,
                    "density_sigma_columns": 0,
                }
                for key in loss_metrics:
                    val_sums[f"val_{key}"] = 0.0
                if validation_mode == "train_batch":
                    val_sums["val_loss"] = total_loss.detach().item()
                    for key, value in loss_metrics.items():
                        val_sums[f"val_{key}"] = value.detach().item()
                    accumulate_winding_metrics(val_sums, output, batch, config)
                    first_val = (batch, output)
                    local_validation_batch_count = 1
                else:
                    model.eval()
                    first_val = None
                    for val_index in range(val_batches_per_log):
                        try:
                            val_batch = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(val_dataloader)
                            val_batch = next(val_iterator)

                        with accelerator.autocast():
                            val_output, val_pair_output = forward_with_pair(
                                model, val_batch, config
                            )
                        val_loss, val_metrics = compute_losses(
                            val_output, val_batch, config, val_pair_output
                        )
                        val_sums["val_loss"] += val_loss.item()
                        for key, value in val_metrics.items():
                            val_sums[f"val_{key}"] += value.item()
                        accumulate_winding_metrics(
                            val_sums, val_output, val_batch, config
                        )
                        if val_index == 0:
                            first_val = (val_batch, val_output)
                    local_validation_batch_count = val_batches_per_log

                sum_keys = tuple(val_sums)
                reduced = accelerator.reduce(
                    torch.tensor(
                        [val_sums[key] for key in sum_keys],
                        dtype=torch.float64,
                        device=accelerator.device,
                    ),
                    reduction="sum",
                )
                val_sums = dict(zip(sum_keys, reduced.cpu().tolist()))
                validation_batch_count = (
                    local_validation_batch_count * accelerator.num_processes
                )
                wandb_log["val_loss"] = (
                    val_sums.pop("val_loss") / validation_batch_count
                )
                for key in loss_metrics:
                    val_key = f"val_{key}"
                    wandb_log[val_key] = (
                        val_sums.pop(val_key) / validation_batch_count
                    )
                wandb_log.update(finalize_winding_metrics(val_sums))

                if accelerator.is_main_process:
                    train_img_path = f"{out_dir}/{iteration:06}_train.png"
                    make_winding_visualization(
                        batch,
                        output,
                        train_img_path,
                        spacing=spacing,
                        peak_threshold=decoding["threshold"],
                        peak_min_distance=decoding["min_distance"],
                        density_min_gap_wv=float(
                            config.get("density_min_gap_wv", 4.0)
                        ),
                        crossing_sigma_wv=float(
                            config.get("crossing_sigma_wv", 1.0)
                        ),
                    )
                    val_img_path = train_img_path
                    if validation_mode == "separate":
                        val_img_path = f"{out_dir}/{iteration:06}_val.png"
                        make_winding_visualization(
                            first_val[0],
                            first_val[1],
                            val_img_path,
                            spacing=spacing,
                            peak_threshold=decoding["threshold"],
                            peak_min_distance=decoding["min_distance"],
                            density_min_gap_wv=float(
                                config.get("density_min_gap_wv", 4.0)
                            ),
                            crossing_sigma_wv=float(
                                config.get("crossing_sigma_wv", 1.0)
                            ),
                        )
                    if wandb.run is not None:
                        wandb_log["train_image"] = wandb.Image(train_img_path)
                        wandb_log["val_image"] = wandb.Image(val_img_path)
                if validation_mode == "separate":
                    model.train()

        if should_log and accelerator.is_main_process:
            metrics_record = {
                "step": iteration,
                **{
                    key: value
                    for key, value in wandb_log.items()
                    if isinstance(value, (bool, int, float, str)) or value is None
                },
            }
            with open(metrics_path, "a") as metrics_file:
                json.dump(metrics_record, metrics_file)
                metrics_file.write("\n")
            accelerator.print(json.dumps(metrics_record))

        if (
            (iteration > 0 or bool(config.get("ckpt_at_step_zero", False)))
            and iteration % ckpt_frequency == 0
            and accelerator.is_main_process
        ):
            save_checkpoint(iteration, f"ckpt_{iteration:06}")

        if wandb.run is not None and accelerator.is_main_process:
            wandb.log(wandb_log)

    progress_bar.close()
    if accelerator.is_main_process:
        save_checkpoint(num_iterations, "ckpt_final")


if __name__ == "__main__":
    train()
