#!/usr/bin/env python
"""Evaluate checkpoints and sweep crossing peak-decoding parameters."""

from __future__ import annotations

import contextlib
import functools
import json
import random

import click
import numpy as np
import torch

from vesuvius.neural_tracing.winding_models import winding_targets
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel
from vesuvius.neural_tracing.winding_models.winding_model_dataset import (
    WindingModelDataset,
)


def _csv_numbers(value: str, cast) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _empty_sums() -> dict[str, float | int]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "phase_abs_error": 0.0,
        "phase_slabs": 0,
        "count_error": 0.0,
        "count_columns": 0,
    }


def _update_metrics(sums, output, batch, *, threshold, min_distance, config):
    spacing = float(config.get("spacing", 1.0))
    tolerance = float(config.get("crossing_match_tolerance_wv", 2.0)) / spacing
    logits = output.get("crossing_logits")
    prob = torch.sigmoid(logits.float()).numpy() if logits is not None else None
    phase_pred = output["phase"].float().numpy()
    phase_target = batch["phase_target"].float().numpy()
    phase_valid = batch["phase_valid"].numpy().astype(bool)
    crossing_valid = batch["crossing_valid"].numpy().astype(bool)
    crossing_t = batch["crossing_t"].float().numpy()
    num_crossings = batch["num_crossings"].numpy()

    batch_size, length = phase_pred.shape[0], phase_pred.shape[-1]
    prob_flat = prob.reshape(-1, length) if prob is not None else None
    phase_pred_flat = phase_pred.reshape(-1, length)
    phase_target_flat = phase_target.reshape(-1, length)
    phase_valid_flat = phase_valid.reshape(-1, length)
    crossing_valid_flat = crossing_valid.reshape(-1, length)
    crossing_t_flat = crossing_t.reshape(-1, crossing_t.shape[-1])
    counts_flat = num_crossings.reshape(-1)
    columns_per_slab = phase_pred_flat.shape[0] // batch_size

    # Headless models decode crossings as integer passages of the phase,
    # registered per slab against the targets (one free offset).
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
            peaks = winding_targets.extract_peaks(
                prob_flat[column], threshold=threshold, min_distance=min_distance
            )
            peak_positions = peaks.astype(np.float64)
        else:
            peak_positions, _ = winding_targets.phase_passages(
                phase_pred_flat[column]
                - passage_offsets[column // columns_per_slab]
            )
            peaks = np.clip(
                np.rint(peak_positions).astype(np.int64), 0, length - 1
            )
        keep = crossing_valid_flat[column][peaks]
        tp, fp, fn = winding_targets.match_crossings(
            peak_positions[keep],
            crossing_t_flat[column, : counts_flat[column]] / spacing,
            tolerance=tolerance,
        )
        sums["tp"] += tp
        sums["fp"] += fp
        sums["fn"] += fn

        valid = phase_valid_flat[column]
        if valid.sum() >= 2:
            pred = phase_pred_flat[column][valid]
            target = phase_target_flat[column][valid]
            sums["count_error"] += float(
                abs((pred[-1] - pred[0]) - (target[-1] - target[0]))
            )
            sums["count_columns"] += 1

    for slab in range(prob.shape[0]):
        valid = phase_valid[slab].reshape(-1)
        if valid.sum() < 2:
            continue
        pred = phase_pred[slab].reshape(-1)[valid]
        target = phase_target[slab].reshape(-1)[valid]
        centered = (pred - pred.mean()) - (target - target.mean())
        sums["phase_abs_error"] += float(np.abs(centered).mean())
        sums["phase_slabs"] += 1


def _finalize(sums) -> dict[str, float | int]:
    precision = sums["tp"] / max(1, sums["tp"] + sums["fp"])
    recall = sums["tp"] / max(1, sums["tp"] + sums["fn"])
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "phase_mae": sums["phase_abs_error"] / max(1, sums["phase_slabs"]),
        "winding_count_error": sums["count_error"] / max(1, sums["count_columns"]),
        "tp": sums["tp"],
        "fp": sums["fp"],
        "fn": sums["fn"],
        "phase_slabs": sums["phase_slabs"],
        "count_columns": sums["count_columns"],
    }


@click.command()
@click.argument("checkpoints", nargs=-1, type=click.Path(exists=True))
@click.option("--num-batches", default=32, show_default=True, type=click.IntRange(1))
@click.option("--thresholds", default="0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
@click.option("--min-distances", default="1,2,3,4")
@click.option("--seed", default=4242, show_default=True, type=int)
@click.option("--use-ema/--no-use-ema", default=False, show_default=True)
def evaluate(checkpoints, num_batches, thresholds, min_distances, seed, use_ema):
    """Evaluate one or more CHECKPOINTS on identical procedural batches."""
    if not checkpoints:
        raise click.UsageError("provide at least one checkpoint")
    thresholds = _csv_numbers(thresholds, float)
    min_distances = _csv_numbers(min_distances, int)
    if not thresholds or not min_distances:
        raise click.UsageError("threshold and min-distance sweeps cannot be empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = WindingModel(config.get("model"))
        model.load_state_dict(
            checkpoint.get("model_ema", checkpoint["model"])
            if use_ema
            else checkpoint["model"]
        )
        model.to(device).eval()
        loaded.append((path, config, model))

    if all(model.crossing_head is None for _, _, model in loaded):
        # Phase-passage decode has no peak parameters; sweeping would repeat
        # identical numbers.
        thresholds, min_distances = thresholds[:1], min_distances[:1]
        click.echo(
            "all models are headless (phase-passage decode); collapsing the "
            "peak-parameter sweep",
            err=True,
        )

    data_config = loaded[0][1]
    dataset = WindingModelDataset(data_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=winding_targets.collate_winding_batch,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sweep = {
        path: {
            (threshold, distance): _empty_sums()
            for threshold in thresholds
            for distance in min_distances
        }
        for path, _, _ in loaded
    }
    iterator = iter(loader)
    autocast = (
        functools.partial(torch.autocast, "cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext
    )
    with torch.inference_mode():
        for _ in range(num_batches):
            batch = next(iterator)
            images = batch["slab_image"].to(device, non_blocking=True)
            valid = batch["slab_valid"].to(device, non_blocking=True)
            for path, config, model in loaded:
                with autocast():
                    output = model(images, valid)
                output = {key: value.cpu() for key, value in output.items()}
                for threshold in thresholds:
                    for distance in min_distances:
                        _update_metrics(
                            sweep[path][(threshold, distance)],
                            output,
                            batch,
                            threshold=threshold,
                            min_distance=distance,
                            config=config,
                        )

    report = {
        "num_batches": num_batches,
        "seed": seed,
        "use_ema": use_ema,
        "models": {},
    }
    for path, _, _ in loaded:
        results = []
        for (threshold, distance), sums in sweep[path].items():
            results.append(
                {
                    "threshold": threshold,
                    "min_distance": distance,
                    **_finalize(sums),
                }
            )
        results.sort(key=lambda item: item["f1"], reverse=True)
        report["models"][path] = {
            "best": results[0],
            "fixed_protocol": next(
                (
                    item
                    for item in results
                    if item["threshold"] == 0.3 and item["min_distance"] == 2
                ),
                None,
            ),
            "sweep": results,
        }
    click.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    evaluate()
