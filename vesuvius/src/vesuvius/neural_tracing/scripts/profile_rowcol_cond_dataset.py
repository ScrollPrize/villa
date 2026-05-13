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
from vesuvius.neural_tracing.datasets.targets import CopyNeighborTargets, RowColTargets


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
        help="Also collate samples and run the active target preparation path.",
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
        "--profile-create-split-masks",
        action="store_true",
        help="Collect and print aggregate stage timings inside EdtSegDataset.create_split_masks.",
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
        "--image-output",
        type=Path,
        default=None,
        help=(
            "Optional PNG path for a target-inspection image. For copy_neighbors, "
            "this shows all model input channels and target tensors for one sample."
        ),
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=None,
        help="Dataset index to render for --image-output. Defaults to the first profiled index.",
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


def _print_create_split_masks_profile(dataset: EdtSegDataset) -> None:
    summary = dataset.create_split_masks_profile_summary()
    attempts = int(summary["attempts"])
    successes = int(summary["successes"])
    if attempts <= 0:
        return

    print("=== create_split_masks Stage Timings ===")
    print(f"attempts: {attempts}")
    print(f"successes: {successes}")
    print(f"success rate: {summary['success_rate']:.3f}")
    print(f"mean total seconds/success: {summary['mean_total']:.6f}")
    print("mean stage seconds/success:")
    for name, seconds in summary["mean_by_stage"].items():
        print(f"  {name}: {seconds:.6f}")
    print()


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
    if "behind_seg" in batch[0] and "front_seg" in batch[0]:
        result = {
            "vol": torch.stack([b["vol"] for b in batch]),
            "cond": torch.stack([b["cond"] for b in batch]),
            "cond_gt": torch.stack([b["cond_gt"] for b in batch]),
            "behind_seg": torch.stack([b["behind_seg"] for b in batch]),
            "front_seg": torch.stack([b["front_seg"] for b in batch]),
        }
        for key in ("dir_priors", "triplet_swap_enabled"):
            if key in batch[0]:
                result[key] = torch.stack([b[key] for b in batch])
        return result

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
    training_mode = str(config.get("training_mode", "rowcol_hidden"))
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
        if training_mode == "copy_neighbors":
            prepared = CopyNeighborTargets.from_batch(batch, config)
        else:
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
        if training_mode == "copy_neighbors":
            prepared = CopyNeighborTargets.from_batch(batch, config)
        else:
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


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.numpy()


def _as_3d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array after squeezing leading dims, got shape {arr.shape}")
    return arr


def _finite_percentile(arr: np.ndarray, percentile: float, fallback: float = 1.0) -> float:
    vals = np.asarray(arr)[np.isfinite(arr)]
    if vals.size == 0:
        return fallback
    value = float(np.percentile(np.abs(vals), percentile))
    return value if np.isfinite(value) and value > 1e-8 else fallback


def _save_copy_neighbor_target_image(
    *,
    dataset: EdtSegDataset,
    index: int,
    config: dict,
    device: torch.device,
    image_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    sample = dataset[index]
    batch = _move_batch_to_device(_collate_with_padding([sample]), device)
    prepared = CopyNeighborTargets.from_batch(batch, config)

    inputs = _tensor_to_numpy(prepared.inputs[0])
    target_seg = _as_3d(_tensor_to_numpy(prepared.target_seg[0]))
    domain = _as_3d(_tensor_to_numpy(prepared.domain[0]))
    velocity = _tensor_to_numpy(prepared.velocity_dir[0])
    velocity_weight = _as_3d(_tensor_to_numpy(prepared.velocity_loss_weight[0]))
    progress_phi = _as_3d(_tensor_to_numpy(prepared.progress_phi[0]))
    progress_weight = _as_3d(_tensor_to_numpy(prepared.progress_phi_weight[0]))
    attract = _tensor_to_numpy(prepared.surface_attract[0])
    attract_weight = _as_3d(_tensor_to_numpy(prepared.surface_attract_weight[0]))
    stop = _as_3d(_tensor_to_numpy(prepared.stop[0]))
    stop_weight = _as_3d(_tensor_to_numpy(prepared.stop_weight[0]))
    target_edt = _as_3d(_tensor_to_numpy(prepared.target_edt[0]))
    seeds = _tensor_to_numpy(prepared.endpoint_seed_points[0])
    seed_mask = _tensor_to_numpy(prepared.endpoint_seed_mask[0]) > 0.0
    endpoint_step_count = (
        int(_tensor_to_numpy(prepared.endpoint_step_count)[0])
        if prepared.endpoint_step_count is not None else None
    )

    velocity_mag = np.linalg.norm(velocity, axis=0)
    attract_mag = np.linalg.norm(attract, axis=0)
    focus = (domain > 0.0) | (target_seg > 0.0) | (inputs[1] > 0.0)
    d, h, w = inputs.shape[1:]
    if np.any(focus):
        z0, y0, x0 = np.median(np.argwhere(focus), axis=0).astype(int).tolist()
    else:
        z0, y0, x0 = d // 2, h // 2, w // 2
    slices = (("z", z0), ("y", y0), ("x", x0))

    attract_vmax = _finite_percentile(attract_mag, 99, fallback=1.0)
    attract_component_vmax = _finite_percentile(attract, 99, fallback=1.0)
    edt_vmax = _finite_percentile(target_edt, 95, fallback=8.0)
    volume_vmax = _finite_percentile(inputs[0], 99, fallback=1.0)
    volume_vmin = float(np.percentile(inputs[0][np.isfinite(inputs[0])], 1)) if np.isfinite(inputs[0]).any() else 0.0

    panels: list[tuple[str, np.ndarray, str, float | None, float | None]] = [
        ("input: volume", inputs[0], "gray", volume_vmin, volume_vmax),
        ("input: source cond", inputs[1], "gray", 0.0, 1.0),
        ("input: side z", inputs[2], "coolwarm", -1.0, 1.0),
        ("input: side y", inputs[3], "coolwarm", -1.0, 1.0),
        ("input: side x", inputs[4], "coolwarm", -1.0, 1.0),
        ("target_seg", target_seg, "gray", 0.0, 1.0),
        ("domain", domain, "gray", 0.0, 1.0),
        ("velocity z", velocity[0], "coolwarm", -1.0, 1.0),
        ("velocity y", velocity[1], "coolwarm", -1.0, 1.0),
        ("velocity x", velocity[2], "coolwarm", -1.0, 1.0),
        ("velocity |v|", velocity_mag, "magma", 0.0, 1.0),
        ("velocity weight", velocity_weight, "gray", 0.0, 1.0),
        ("progress_phi", progress_phi, "viridis", 0.0, 1.0),
        ("progress weight", progress_weight, "gray", 0.0, 1.0),
        ("surface_attr z", attract[0], "coolwarm", -attract_component_vmax, attract_component_vmax),
        ("surface_attr y", attract[1], "coolwarm", -attract_component_vmax, attract_component_vmax),
        ("surface_attr x", attract[2], "coolwarm", -attract_component_vmax, attract_component_vmax),
        ("surface_attr |v|", attract_mag, "magma", 0.0, attract_vmax),
        ("surface_attr weight", attract_weight, "gray", 0.0, 1.0),
        ("stop", stop, "viridis", 0.0, 1.0),
        ("stop weight", stop_weight, "gray", 0.0, 1.0),
        ("target EDT + seeds", target_edt, "magma", 0.0, edt_vmax),
    ]

    def slice_2d(arr: np.ndarray, axis: str, idx: int) -> np.ndarray:
        if axis == "z":
            return arr[idx]
        if axis == "y":
            return arr[:, idx, :]
        if axis == "x":
            return arr[:, :, idx]
        raise ValueError(axis)

    n_cols = len(panels)
    fig, axes = plt.subplots(
        len(slices),
        n_cols,
        figsize=(2.4 * n_cols, 7.6),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (axis, idx) in enumerate(slices):
        for col, (title, arr, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(slice_2d(arr, axis, idx), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}\n{axis}={idx}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if title == "target EDT + seeds":
                active = seeds[seed_mask]
                if active.size > 0:
                    if axis == "z":
                        near = np.abs(active[:, 0] - idx) <= 1.5
                        xy = active[near][:, [2, 1]]
                    elif axis == "y":
                        near = np.abs(active[:, 1] - idx) <= 1.5
                        xy = active[near][:, [2, 0]]
                    else:
                        near = np.abs(active[:, 2] - idx) <= 1.5
                        xy = active[near][:, [1, 0]]
                    if xy.size > 0:
                        ax.scatter(xy[:, 0], xy[:, 1], s=7, c="cyan", edgecolors="black", linewidths=0.2)

    title = (
        f"copy_neighbors sample index={index} | shape={d}x{h}x{w} | "
        f"seeds={int(seed_mask.sum())}"
    )
    if endpoint_step_count is not None:
        title = f"{title} | endpoint_steps={endpoint_step_count}"
    fig.suptitle(title, fontsize=12)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    config = _load_config(
        args.config_path,
        disable_force_recompute_patches=args.disable_force_recompute_patches,
    )
    if args.profile_create_split_masks:
        config["profile_create_split_masks"] = True
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
    training_mode = str(config.get("training_mode", "rowcol_hidden"))
    if args.include_prepare_batch and training_mode != "copy_neighbors" and prepare_device.type != "cuda":
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

    if args.image_output is not None:
        if training_mode != "copy_neighbors":
            raise RuntimeError("--image-output is currently implemented for training_mode='copy_neighbors'.")
        image_index = int(args.image_index if args.image_index is not None else indices[0])
        _save_copy_neighbor_target_image(
            dataset=dataset,
            index=image_index,
            config=config,
            device=prepare_device,
            image_path=args.image_output,
        )

    print("=== Trace ODE Dataset Profile ===")
    print(f"config: {args.config_path}")
    print(f"training mode: {training_mode}")
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
    if args.image_output is not None:
        print(f"target image: {args.image_output}")
    print()
    if args.profile_create_split_masks:
        _print_create_split_masks_profile(dataset)
    _print_stats(profiler, sort=args.sort, limit=args.limit, focus=args.focus)


if __name__ == "__main__":
    main()
