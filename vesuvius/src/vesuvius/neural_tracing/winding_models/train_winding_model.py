#!/usr/bin/env python
"""Trainer for the winding phase model.

From intersecting transverse planes along a ray, the model predicts a
monotone relative winding phase and a crossing heatmap per ray sample -- the
dense observations consumed by fit_spiral's neural winding losses
(volume-cartographer/scripts/spiral/neural_winding_losses.py). Phase is
supervised shift-invariantly and canonicalized to increase along the ray;
the consumer applies each ray's known winding direction and registers the
free offset.
"""

import functools
import json
import math
import os
import random

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


def compute_losses(output, batch, config):
    phase = winding_targets.phase_loss(
        output["phase"],
        batch["phase_target"],
        batch["phase_valid"],
        huber_delta=float(config.get("phase_huber_delta", 0.25)),
    )
    crossing = winding_targets.crossing_loss(
        output["crossing_logits"],
        batch["crossing_target"],
        batch["crossing_valid"],
    )
    total = (
        float(config.get("lambda_phase", 1.0)) * phase
        + float(config.get("lambda_crossing", 1.0)) * crossing
    )
    return total, {"phase_loss": phase, "crossing_loss": crossing}


def peak_decoding_config(config):
    spacing = float(config.get("plane_spacing", 1.0))
    return {
        "threshold": float(config.get("crossing_peak_threshold", 0.3)),
        "min_distance": max(
            1,
            round(2.0 * float(config.get("crossing_sigma_wv", 1.0)) / spacing),
        ),
    }


def accumulate_winding_metrics(sums, output, batch, config):
    """Consumer-facing quality measures.

    Crossing detection precision/recall within a physical tolerance,
    mean-centered phase error, and winding-count error over each ray's valid
    span (the quantity fit_spiral's density loss integrates).
    """
    spacing = float(config.get("plane_spacing", 1.0))
    tolerance = float(config.get("crossing_match_tolerance_wv", 2.0)) / spacing
    decoding = peak_decoding_config(config)

    prob = torch.sigmoid(output["crossing_logits"].detach().float()).cpu().numpy()
    phase_pred = output["phase"].detach().float().cpu().numpy()
    phase_target = batch["phase_target"].float().cpu().numpy()
    phase_valid = batch["phase_valid"].cpu().numpy().astype(bool)
    crossing_valid = batch["crossing_valid"].cpu().numpy().astype(bool)
    crossing_t = batch["crossing_t"].float().cpu().numpy()
    num_crossings = batch["num_crossings"].cpu().numpy()

    for ray in range(prob.shape[0]):
        peaks = winding_targets.extract_peaks(prob[ray], **decoding)
        # Peaks in unlabeled spans may be real unlabeled wraps; they are not
        # decidable and must not count as false positives.
        peaks = peaks[crossing_valid[ray][peaks]]
        tp, fp, fn = winding_targets.match_crossings(
            peaks.astype(np.float64),
            crossing_t[ray, : num_crossings[ray]] / spacing,
            tolerance=tolerance,
        )
        sums["crossing_tp"] += tp
        sums["crossing_fp"] += fp
        sums["crossing_fn"] += fn

        valid = phase_valid[ray]
        if valid.sum() >= 2:
            pred = phase_pred[ray][valid]
            target = phase_target[ray][valid]
            centered = (pred - pred.mean()) - (target - target.mean())
            sums["phase_abs_error"] += float(np.abs(centered).mean())
            sums["count_error"] += float(
                abs((pred[-1] - pred[0]) - (target[-1] - target[0]))
            )
            sums["phase_rays"] += 1


def finalize_winding_metrics(sums):
    metrics = {
        "val_crossing_precision": sums["crossing_tp"]
        / max(1, sums["crossing_tp"] + sums["crossing_fp"]),
        "val_crossing_recall": sums["crossing_tp"]
        / max(1, sums["crossing_tp"] + sums["crossing_fn"]),
    }
    if sums["phase_rays"]:
        metrics["val_phase_mae"] = sums["phase_abs_error"] / sums["phase_rays"]
        metrics["val_winding_count_error"] = sums["count_error"] / sums["phase_rays"]
    return metrics


def make_dataloader(dataset, config, *, generator, num_workers):
    kwargs = {
        "batch_size": int(config.get("batch_size", 16)),
        # Samples are drawn procedurally (__getitem__ ignores its index), so
        # shuffling the index stream would be a no-op.
        "shuffle": False,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": True,
        "collate_fn": functools.partial(
            winding_targets.collate_winding_batch,
            crossing_sigma_wv=float(config.get("crossing_sigma_wv", 1.0)),
        ),
        "pin_memory": bool(config.get("pin_memory", True)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = max(1, int(config.get("prefetch_factor", 2)))
        kwargs["multiprocessing_context"] = "spawn"
    return torch.utils.data.DataLoader(dataset, **kwargs)


def checkpoint_model_state_dict(accelerator, model):
    """Return a wrapper-free model state dict for portable checkpoints."""
    return strip_state(accelerator.unwrap_model(model).state_dict())


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path):
    """Train a winding phase + crossing model."""

    with open(config_path) as config_file:
        config = json.load(config_file)

    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    accelerator = accelerate.Accelerator(
        mixed_precision=str(config.get("mixed_precision", "bf16")),
        gradient_accumulation_steps=int(config.get("grad_acc_steps", 1)),
    )

    if "wandb_project" in config and accelerator.is_main_process:
        wandb.init(
            project=config["wandb_project"],
            entity=config.get("wandb_entity"),
            config=config,
        )

    train_dataset = WindingModelDataset(config)
    if "val_datasets" in config:
        val_config = dict(config)
        val_config["datasets"] = config["val_datasets"]
        val_dataset = WindingModelDataset(val_config)
    else:
        # Sampling is procedural, so a second instance over the same segments
        # would only duplicate raycaster construction; validation batches are
        # simply fresh draws.
        val_dataset = train_dataset

    def make_generator(offset):
        generator = torch.Generator()
        generator.manual_seed(seed + accelerator.process_index * 1000 + offset)
        return generator

    train_dataloader = make_dataloader(
        train_dataset,
        config,
        generator=make_generator(0),
        num_workers=max(0, int(config.get("num_workers", 8))),
    )
    val_dataloader = make_dataloader(
        val_dataset,
        config,
        generator=make_generator(1),
        num_workers=max(0, int(config.get("val_num_workers", 2))),
    )

    model = WindingModel(config.get("model"))
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
    lr_scheduler = get_scheduler(
        scheduler_type=str(config.get("scheduler", "diffusers_cosine_warmup")),
        optimizer=optimizer,
        initial_lr=learning_rate,
        max_steps=num_iterations,
        **config.get("scheduler_kwargs", {}),
    )

    start_iteration = 0
    if "load_ckpt" in config:
        accelerator.print(f"Loading checkpoint {config['load_ckpt']}")
        ckpt = torch.load(config["load_ckpt"], map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"])
        if not config.get("load_weights_only", False):
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            start_iteration = int(ckpt.get("step", 0))

    # Keep the scheduler out of accelerator.prepare: prepared schedulers are
    # advanced once per process with sharded dataloaders, while num_iterations
    # counts optimizer-update iterations.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    spacing = float(config.get("plane_spacing", 1.0))
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
        torch.save(
            {
                "model": checkpoint_model_state_dict(accelerator, model),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "config": config,
                "step": step,
            },
            f"{out_dir}/{name}.pth",
        )

    train_iterator = iter(train_dataloader)
    val_iterator = iter(val_dataloader)
    progress_bar = tqdm(
        total=num_iterations,
        initial=start_iteration,
        disable=not accelerator.is_local_main_process,
    )

    for iteration in range(start_iteration, num_iterations):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        wandb_log = {}
        grad_norm = None
        with accelerator.accumulate(model):
            output = model(batch["plane_images"], batch["plane_valid"])
            total_loss, loss_metrics = compute_losses(output, batch, config)
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
            optimizer.zero_grad()

        wandb_log["loss"] = total_loss.detach().item()
        wandb_log["current_lr"] = optimizer.param_groups[0]["lr"]
        if grad_norm is not None:
            wandb_log["grad_norm"] = grad_norm

        progress_bar.set_postfix(
            {
                "loss": f"{wandb_log['loss']:.4f}",
                "phase": f"{wandb_log['phase_loss']:.4f}",
                "crossing": f"{wandb_log['crossing_loss']:.4f}",
            }
        )
        progress_bar.update(1)

        should_log = (
            (iteration > 0 or bool(config.get("log_at_step_zero", False)))
            and iteration % log_frequency == 0
        )
        if should_log and accelerator.is_main_process:
            with torch.no_grad():
                model.eval()
                val_sums = {
                    "val_loss": 0.0,
                    "crossing_tp": 0,
                    "crossing_fp": 0,
                    "crossing_fn": 0,
                    "phase_abs_error": 0.0,
                    "count_error": 0.0,
                    "phase_rays": 0,
                }
                first_val = None
                for val_index in range(val_batches_per_log):
                    try:
                        val_batch = next(val_iterator)
                    except StopIteration:
                        val_iterator = iter(val_dataloader)
                        val_batch = next(val_iterator)

                    with accelerator.autocast():
                        val_output = model(
                            val_batch["plane_images"], val_batch["plane_valid"]
                        )
                    val_loss, val_metrics = compute_losses(
                        val_output, val_batch, config
                    )
                    val_sums["val_loss"] += val_loss.item()
                    for key, value in val_metrics.items():
                        val_sums.setdefault(f"val_{key}", 0.0)
                        val_sums[f"val_{key}"] += value.item()
                    accumulate_winding_metrics(val_sums, val_output, val_batch, config)
                    if val_index == 0:
                        first_val = (val_batch, val_output)

                for key in ("val_loss", "val_phase_loss", "val_crossing_loss"):
                    wandb_log[key] = val_sums.pop(key) / val_batches_per_log
                wandb_log.update(finalize_winding_metrics(val_sums))

                train_img_path = f"{out_dir}/{iteration:06}_train.png"
                val_img_path = f"{out_dir}/{iteration:06}_val.png"
                make_winding_visualization(
                    batch,
                    output,
                    train_img_path,
                    spacing=spacing,
                    peak_threshold=decoding["threshold"],
                    peak_min_distance=decoding["min_distance"],
                )
                make_winding_visualization(
                    first_val[0],
                    first_val[1],
                    val_img_path,
                    spacing=spacing,
                    peak_threshold=decoding["threshold"],
                    peak_min_distance=decoding["min_distance"],
                )
                if wandb.run is not None:
                    wandb_log["train_image"] = wandb.Image(train_img_path)
                    wandb_log["val_image"] = wandb.Image(val_img_path)
                model.train()

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
