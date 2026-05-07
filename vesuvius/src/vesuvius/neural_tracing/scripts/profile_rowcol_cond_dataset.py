#!/usr/bin/env python
"""Profile row/col conditioned dataset sample generation.

This script loads the same JSON config used by train_rowcol_cond.py and runs
EdtSegDataset.__getitem__ under cProfile so slow dataset functions show up by
name instead of being hidden behind DataLoader worker processes.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _ensure_src_on_path() -> None:
    """Allow running this file directly from any working directory."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        src_dir = parent.parent
        if parent.name == "vesuvius" and src_dir.name == "src":
            sys.path.insert(0, str(src_dir))
            return


_ensure_src_on_path()

from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset
from vesuvius.neural_tracing.datasets.rowcol_cond_config import (
    setdefault_rowcol_cond_dataset_config,
    validate_rowcol_cond_dataset_config,
)
from vesuvius.neural_tracing.datasets.targets import RowColTargets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile EdtSegDataset sample generation using a real row/col "
            "conditioning training config."
        )
    )
    parser.add_argument("config_path", type=Path, help="Path to row/col conditioning JSON config.")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to profile.")
    parser.add_argument(
        "--include-prepare-batch",
        action="store_true",
        help="Also collate samples and run RowColTargets.from_batch, matching trainer target preparation.",
    )
    parser.add_argument(
        "--prepare-batch-size",
        type=int,
        default=None,
        help="Batch size for --include-prepare-batch. Defaults to config batch_size.",
    )
    parser.add_argument(
        "--prepare-device",
        default=None,
        help="Device for --include-prepare-batch. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="First dataset index for sequential mode.")
    parser.add_argument(
        "--index-mode",
        choices=("sequential", "random"),
        default="sequential",
        help="How to choose dataset indices.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for random index mode and augmentations.")
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Construct the dataset with apply_augmentation=False.",
    )
    parser.add_argument(
        "--no-perturbation",
        action="store_true",
        help="Construct the dataset with apply_perturbation=False.",
    )
    parser.add_argument(
        "--profile-init",
        action="store_true",
        help="Include dataset construction in the cProfile output.",
    )
    parser.add_argument(
        "--disable-force-recompute-patches",
        action="store_true",
        help="Set force_recompute_patches=False before constructing the dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional raw .prof output path for snakeviz/tuna/pstats.",
    )
    parser.add_argument(
        "--sort",
        default="cumtime",
        choices=("cumtime", "tottime", "calls", "ncalls", "time", "name", "filename", "line"),
        help="pstats sort key.",
    )
    parser.add_argument("--limit", type=int, default=80, help="Number of pstats rows to print.")
    parser.add_argument(
        "--focus",
        default=None,
        help=(
            "Optional printed-output filter regex, e.g. 'neural_tracing' or "
            "'dataset_rowcol_cond'. Profiling still records all functions."
        ),
    )
    return parser.parse_args()


def _load_config(config_path: Path, *, disable_force_recompute_patches: bool) -> dict:
    with config_path.open("r") as f:
        config = json.load(f)
    setdefault_rowcol_cond_dataset_config(config)
    if disable_force_recompute_patches:
        config["force_recompute_patches"] = False
    validate_rowcol_cond_dataset_config(config)
    return config


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_indices(dataset_len: int, args: argparse.Namespace, seed: int) -> list[int]:
    if dataset_len <= 0:
        raise ValueError("Dataset is empty.")
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")

    if args.index_mode == "random":
        rng = random.Random(seed)
        return [rng.randrange(dataset_len) for _ in range(args.samples)]

    start = int(args.start_index)
    return [(start + i) % dataset_len for i in range(args.samples)]


def _collate_with_padding(batch: list[dict]) -> dict:
    """Mirror trainers.train_rowcol_cond.collate_with_padding without importing trainer deps."""
    return {
        "vol": torch.stack([b["vol"] for b in batch]),
        "cond": torch.stack([b["cond"] for b in batch]),
        "cond_direction": [b["cond_direction"] for b in batch],
        "velocity_dir": torch.stack([b["velocity_dir"] for b in batch]),
        "velocity_loss_weight": torch.stack([b["velocity_loss_weight"] for b in batch]),
        "trace_loss_weight": torch.stack([b["trace_loss_weight"] for b in batch]),
        "cond_gt": torch.stack([b["cond_gt"] for b in batch]),
        "masked_seg": torch.stack([b["masked_seg"] for b in batch]),
        "neighbor_seg": torch.stack([b["neighbor_seg"] for b in batch]),
    }


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=False) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _profile_samples(
    dataset: EdtSegDataset,
    indices: list[int],
    *,
    config: dict,
    include_prepare_batch: bool,
    prepare_batch_size: int,
    prepare_device: torch.device,
) -> tuple[float, float, float, float, int]:
    sample_times = []
    prepare_times = []
    pending_batch = []
    for idx in indices:
        start = time.perf_counter()
        sample = dataset[idx]
        # Touch returned tensors so lazy failures surface inside the profile.
        for value in sample.values():
            if torch.is_tensor(value):
                _ = value.shape
        sample_times.append(time.perf_counter() - start)
        if not include_prepare_batch:
            continue

        pending_batch.append(sample)
        if len(pending_batch) < prepare_batch_size:
            continue

        start = time.perf_counter()
        batch = _move_batch_to_device(_collate_with_padding(pending_batch), prepare_device)
        prepared = RowColTargets.from_batch(batch, config)
        for value in prepared.__dict__.values():
            if torch.is_tensor(value):
                _ = value.shape
        if prepare_device.type == "cuda":
            torch.cuda.synchronize(prepare_device)
        prepare_times.append(time.perf_counter() - start)
        pending_batch = []

    if include_prepare_batch and pending_batch:
        start = time.perf_counter()
        batch = _move_batch_to_device(_collate_with_padding(pending_batch), prepare_device)
        prepared = RowColTargets.from_batch(batch, config)
        for value in prepared.__dict__.values():
            if torch.is_tensor(value):
                _ = value.shape
        if prepare_device.type == "cuda":
            torch.cuda.synchronize(prepare_device)
        prepare_times.append(time.perf_counter() - start)

    sample_total = float(sum(sample_times))
    sample_mean = sample_total / len(sample_times)
    prepare_total = float(sum(prepare_times))
    prepare_mean = prepare_total / len(prepare_times) if prepare_times else 0.0
    return sample_total, sample_mean, prepare_total, prepare_mean, len(prepare_times)


def _print_stats(
    profiler: cProfile.Profile,
    *,
    sort: str,
    limit: int,
    focus: str | None,
) -> None:
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(sort)
    if focus:
        # Keep full paths so package filters such as "neural_tracing" can match.
        stats.print_stats(focus, limit)
    else:
        stats.strip_dirs()
        stats.print_stats(limit)
    print(stream.getvalue())


def main() -> None:
    args = _parse_args()
    config = _load_config(
        args.config_path,
        disable_force_recompute_patches=args.disable_force_recompute_patches,
    )
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    _seed_everything(seed)

    apply_augmentation = not args.no_augmentation
    apply_perturbation = not args.no_perturbation

    profiler = cProfile.Profile()
    if args.profile_init:
        profiler.enable()
    init_start = time.perf_counter()
    dataset = EdtSegDataset(
        config,
        apply_augmentation=apply_augmentation,
        apply_perturbation=apply_perturbation,
    )
    init_seconds = time.perf_counter() - init_start
    if args.profile_init:
        profiler.disable()

    indices = _build_indices(len(dataset), args, seed)
    prepare_batch_size = int(args.prepare_batch_size or config.get("batch_size", 1))
    if prepare_batch_size <= 0:
        raise ValueError("--prepare-batch-size must be positive.")
    prepare_device = torch.device(
        args.prepare_device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.include_prepare_batch and prepare_device.type != "cuda":
        raise RuntimeError(
            "--include-prepare-batch requires a CUDA device because RowColTargets uses cupyx EDT."
        )

    profiler.enable()
    (
        total_seconds,
        mean_seconds,
        prepare_seconds,
        mean_prepare_seconds,
        prepared_batches,
    ) = _profile_samples(
        dataset,
        indices,
        config=config,
        include_prepare_batch=bool(args.include_prepare_batch),
        prepare_batch_size=prepare_batch_size,
        prepare_device=prepare_device,
    )
    profiler.disable()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(args.output))

    print("=== Row/Col Dataset Profile ===")
    print(f"config: {args.config_path}")
    print(f"dataset samples: {len(dataset)}")
    print(f"profiled samples: {len(indices)}")
    print(f"index mode: {args.index_mode}")
    print(f"augmentation: {apply_augmentation}")
    print(f"perturbation: {apply_perturbation}")
    print(f"prepare batch: {args.include_prepare_batch}")
    if args.include_prepare_batch:
        print(f"prepare device: {prepare_device}")
        print(f"prepare batch size: {prepare_batch_size}")
        print(f"prepared batches: {prepared_batches}")
    print(f"dataset init seconds: {init_seconds:.3f}")
    print(f"sample loop seconds: {total_seconds:.3f}")
    print(f"mean seconds/sample: {mean_seconds:.6f}")
    if args.include_prepare_batch:
        print(f"prepare loop seconds: {prepare_seconds:.3f}")
        print(f"mean seconds/prepared batch: {mean_prepare_seconds:.6f}")
        print(f"combined sample+prepare seconds: {total_seconds + prepare_seconds:.3f}")
    if args.output is not None:
        print(f"raw profile: {args.output}")
    print()
    _print_stats(profiler, sort=args.sort, limit=args.limit, focus=args.focus)


if __name__ == "__main__":
    main()
