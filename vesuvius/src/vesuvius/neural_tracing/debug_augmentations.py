import argparse
import json
import pathlib
import sys
from typing import Iterable, Tuple

import torch
from tqdm import tqdm

# Make sure the repository root (which contains the 'src' directory) is on sys.path when
# running this file directly.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vesuvius.neural_tracing.dataset import (
    PatchInCubeDataset,
    HeatmapDatasetV2,
    load_datasets,
)


def iter_tensors(prefix: str, obj) -> Iterable[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            yield from iter_tensors(child_prefix, value)
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child_prefix = f"{prefix}[{idx}]"
            yield from iter_tensors(child_prefix, value)


def build_dataset(config: dict):
    if config.get("representation") == "heatmap":
        train_patches, _ = load_datasets(config)
        return HeatmapDatasetV2(config, train_patches)
    return PatchInCubeDataset(config)


def main():
    parser = argparse.ArgumentParser(
        description="Run neural_tracing augmentations repeatedly to surface NaN/Inf issues."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to neural_tracing config JSON (same one used for training).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of augmented batches to draw.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size when drawing from the dataset.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; keep at 0 so augmentation warnings print in the main process.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep scanning after hitting a NaN/Inf batch.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    dataset = build_dataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    loader_iter = iter(dataloader)

    bad_batches = 0
    with torch.no_grad():
        for step in tqdm(range(args.iterations), desc="Augmentation iterations", unit="batch"):
            try:
                batch = next(loader_iter)
            except Exception:
                print(f"Exception while fetching batch at step {step}")
                raise

            bad_keys = []
            for key, tensor in iter_tensors("", batch):
                if tensor is None or (not torch.is_floating_point(tensor) and not torch.is_complex(tensor)):
                    continue
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                if nan_count or inf_count:
                    bad_keys.append(
                        (key, tensor.shape, nan_count, inf_count)
                    )

            if bad_keys:
                bad_batches += 1
                print(f"\nDetected NaN/Inf in batch {step}:")
                for key, shape, nan_count, inf_count in bad_keys:
                    print(
                        f"  - {key} {tuple(shape)} nan={nan_count} inf={inf_count}"
                    )
                if not args.continue_on_error:
                    break

    print(
        f"Finished {args.iterations} iterations; {bad_batches} batches contained NaN/Inf."
    )


if __name__ == "__main__":
    main()
