import argparse
import json
from pathlib import Path

import numpy as np

from .tifxyz_dataset import TifxyzInkDataset


ARRAY_KEYS = (
    "vol",
    "labeled_vox_at_surface",
    "surface_vox",
    "projected_loss_mask",
)


def _as_numpy_array(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def sample_to_numpy(sample):
    return {
        key: _as_numpy_array(sample[key])
        for key in ARRAY_KEYS
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_sample_metadata(sample, dataset_index, arrays):
    patch = sample.get("patch", {})
    sample_idx = int(sample.get("idx", dataset_index))
    return {
        "dataset_index": int(dataset_index),
        "sample_idx": sample_idx,
        "arrays": {
            key: {
                "shape": [int(dim) for dim in array.shape],
                "dtype": str(array.dtype),
            }
            for key, array in arrays.items()
        },
        "patch": _json_safe(patch),
    }


def export_sample(dataset, index, output_dir, overwrite=False):
    output_dir = Path(output_dir)
    sample_dir = output_dir / f"sample_{int(index):06d}"
    if sample_dir.exists() and not overwrite:
        raise FileExistsError(f"{sample_dir} already exists; pass --overwrite to replace files")

    sample_dir.mkdir(parents=True, exist_ok=True)
    sample = dataset[int(index)]
    arrays = sample_to_numpy(sample)
    for key, array in arrays.items():
        np.save(sample_dir / f"{key}.npy", array)

    metadata = build_sample_metadata(
        sample,
        dataset_index=int(index),
        arrays=arrays,
    )
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sample_dir


def parse_indices(value):
    if value is None:
        return None
    indices = []
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("indices must be comma-separated non-negative integers")
        try:
            index = int(part)
        except ValueError as exc:
            raise ValueError("indices must be comma-separated non-negative integers") from exc
        if index < 0:
            raise ValueError("indices must be comma-separated non-negative integers")
        indices.append(index)
    return indices


def _resolve_indices(dataset, count=None, indices=None):
    if count is not None and indices is not None:
        raise ValueError("count and indices are mutually exclusive")
    if count is None and indices is None:
        raise ValueError("count or indices is required")
    dataset_len = len(dataset)
    if indices is None:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > dataset_len:
            raise ValueError(f"count={count} exceeds dataset length {dataset_len}")
        return list(range(count))

    resolved = [int(index) for index in indices]
    out_of_range = [
        index
        for index in resolved
        if index < 0 or index >= dataset_len
    ]
    if out_of_range:
        raise ValueError(f"indices out of range for dataset length {dataset_len}: {out_of_range}")
    return resolved


def export_samples(dataset, output_dir, count=None, indices=None, overwrite=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_indices = _resolve_indices(dataset, count=count, indices=indices)
    samples = []
    for index in resolved_indices:
        sample_dir = export_sample(
            dataset,
            index,
            output_dir,
            overwrite=overwrite,
        )
        samples.append(
            {
                "dataset_index": int(index),
                "path": sample_dir.name,
            }
        )

    manifest = {
        "sample_count": len(samples),
        "samples": samples,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(config):
    return TifxyzInkDataset(
        config,
        apply_augmentation=False,
        apply_perturbation=False,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Export TifxyzInkDataset samples as .npy arrays plus JSON metadata.",
    )
    parser.add_argument("--config", required=True, help="Path to a tifxyz dataset config JSON.")
    parser.add_argument("--output", required=True, help="Directory to write exported samples.")
    parser.add_argument("--count", type=int, help="Export the first N dataset samples.")
    parser.add_argument("--indices", help="Comma-separated dataset indices to export, for example 0,4,9.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files inside existing sample directories.",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        indices = parse_indices(args.indices)
        if args.count is not None and indices is not None:
            parser.error("--count and --indices are mutually exclusive")
        if args.count is None and indices is None:
            parser.error("--count or --indices is required")
        config = load_config(args.config)
        dataset = build_dataset(config)
        export_samples(
            dataset,
            args.output,
            count=args.count,
            indices=indices,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
