import argparse
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from vesuvius.neural_tracing.dataset import HeatmapDatasetV2, load_datasets
from vesuvius.neural_tracing.benchmarking.test_datasets import (
    HeatmapDatasetV2BilinearLookup,
    HeatmapDatasetV2BBoxGuard,
    HeatmapDatasetV2CoarseDistance,
    HeatmapDatasetV2PrecomputedNormals,
    HeatmapDatasetV2TorchComponent,
)


DATASET_CLASSES = {
    "HeatmapDatasetV2": HeatmapDatasetV2,
    "HeatmapDatasetV2TorchComponent": HeatmapDatasetV2TorchComponent,
    # "HeatmapDatasetV2PrecomputedNormals": HeatmapDatasetV2PrecomputedNormals,
    # "HeatmapDatasetV2CoarseDistance": HeatmapDatasetV2CoarseDistance,
    # "HeatmapDatasetV2BilinearLookup": HeatmapDatasetV2BilinearLookup,
    # "HeatmapDatasetV2BBoxGuard": HeatmapDatasetV2BBoxGuard
}


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compare_dataset_outputs(base_cls, target_cls, config, patches, num_samples: int, seed: int):
    """Check that two dataset implementations emit matching outputs for the same random stream."""
    _set_seed(seed)
    base_iter = iter(base_cls(config, patches))
    _set_seed(seed)
    target_iter = iter(target_cls(config, patches))

    mismatches = []

    for idx in range(num_samples):
        base_sample = next(base_iter)
        target_sample = next(target_iter)

        base_keys = set(base_sample.keys())
        target_keys = set(target_sample.keys())
        if base_keys != target_keys:
            mismatches.append((idx, f"key mismatch: base={sorted(base_keys)}, target={sorted(target_keys)}"))
            continue

        for key in sorted(base_keys):
            b = base_sample[key]
            t = target_sample[key]

            if b.shape != t.shape:
                mismatches.append((idx, f"{key} shape differs: {b.shape} vs {t.shape}"))
                continue

            if not torch.allclose(b, t, rtol=1e-4, atol=1e-5):
                max_diff = (b - t).abs().max().item()
                mismatches.append((idx, f"{key} values differ (max abs diff {max_diff:.3e})"))

    return mismatches


def _time_dataset(dataset_cls, config, patches, num_samples: int, warmup: int, seed: int, label: str):
    _set_seed(seed)
    dataset = dataset_cls(config, patches)
    iterator = iter(dataset)

    total_iters = max(warmup, 0) + num_samples
    pbar = tqdm(total=total_iters, desc=label)

    for _ in range(max(warmup, 0)):
        next(iterator)
        pbar.update(1)

    start = time.perf_counter()
    for _ in range(num_samples):
        next(iterator)
        pbar.update(1)
    pbar.close()
    duration = time.perf_counter() - start

    per_sample = duration / num_samples if num_samples else float("nan")
    samples_per_sec = num_samples / duration if duration > 0 else float("inf")
    return duration, per_sample, samples_per_sec


def main():
    parser = argparse.ArgumentParser(description="Time HeatmapDataset variants.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/vesuvius/neural_tracing/config.json"),
        help="Path to config JSON used to build patches/dataset.",
    )
    parser.add_argument(
        "--dataset-classes",
        nargs="+",
        default=list(DATASET_CLASSES.keys()),
        help=f"Dataset classes to benchmark. Options: {', '.join(DATASET_CLASSES.keys())}",
    )
    parser.add_argument(
        "--compare-classes",
        nargs=2,
        default=None,
        help="If set, compare outputs between these two dataset classes.",
    )
    parser.add_argument(
        "--compare-samples",
        type=int,
        default=0,
        help="If >0, compare outputs for this many samples between two dataset classes.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to measure per dataset class.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup samples to discard before timing.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Which split to draw patches from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to make comparisons reproducible.",
    )
    args = parser.parse_args()

    with args.config.open("r") as f:
        config = json.load(f)

    train_patches, val_patches = load_datasets(config)
    patches = train_patches if args.split == "train" else val_patches

    # Optional correctness check before timing
    if args.compare_samples > 0:
        if args.compare_classes is not None:
            compare_names = args.compare_classes
        elif len(args.dataset_classes) >= 2:
            compare_names = args.dataset_classes[:2]
        else:
            compare_names = None

        if compare_names is None or len(compare_names) != 2:
            print("[compare] Need two dataset classes to compare; skipping comparison.")
        else:
            base_name, target_name = compare_names
            base_cls = DATASET_CLASSES.get(base_name)
            target_cls = DATASET_CLASSES.get(target_name)
            if base_cls is None or target_cls is None:
                print(f"[compare] Unknown dataset class in {compare_names}; available: {', '.join(DATASET_CLASSES.keys())}")
            else:
                print(f"[compare] Checking {args.compare_samples} samples: {base_name} vs {target_name}")
                mismatches = _compare_dataset_outputs(
                    base_cls, target_cls, config, patches, args.compare_samples, args.seed
                )
                if not mismatches:
                    print("[compare] Outputs match for all compared samples.")
                else:
                    for idx, msg in mismatches:
                        print(f"[compare] sample {idx}: {msg}")

    for name in args.dataset_classes:
        dataset_cls = DATASET_CLASSES.get(name)
        if dataset_cls is None:
            print(f"[skip] Unknown dataset class '{name}'. Available: {', '.join(DATASET_CLASSES.keys())}")
            continue

        duration, per_sample, samples_per_sec = _time_dataset(
            dataset_cls, config, patches, args.num_samples, args.warmup, args.seed, name
        )
        print(
            f"{name}: {duration:.3f}s total for {args.num_samples} samples "
            f"({per_sample:.4f}s/sample, {samples_per_sec:.2f} samples/s)"
        )


if __name__ == "__main__":
    main()
