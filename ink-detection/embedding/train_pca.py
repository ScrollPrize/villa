#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import DatasetConfig, IndexedInkCropDataset, build_train_augmentations
from model import PCAEmbedder, create_frozen_dino_backbone, normalize_for_backbone


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def safe_json_config(config: dict[str, Any]) -> dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return value

    return _convert(config)


def build_dataset(image_dir: Path, split: str, samples: int, config: dict[str, Any]) -> IndexedInkCropDataset:
    dataset_cfg = DatasetConfig(
        image_dir=image_dir.resolve(),
        split=split,
        seed=int(config["seed"]),
        crop_size=int(config["crop_size"]),
        downsample_factor=int(config["downsample_factor"]),
        samples_per_epoch=samples,
        min_foreground_fraction=float(config["min_foreground_fraction"]),
        max_crop_attempts=int(config["max_crop_attempts"]),
        test_fraction=float(config["test_fraction"]),
        foreground_threshold=float(config["foreground_threshold"]),
        cache_images=True,
    )
    return IndexedInkCropDataset(dataset_cfg, build_train_augmentations(dataset_cfg.crop_size))


def fit_pca_embedder(
    dataloader: DataLoader,
    backbone: torch.nn.Module,
    device: torch.device,
    input_dim: int,
    embedding_dim: int,
    image_key: str = "image",
    max_samples: int | None = None,
) -> tuple[PCAEmbedder, dict[str, Any]]:
    if embedding_dim < 1:
        raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
    if input_dim < embedding_dim:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be <= input_dim ({input_dim}) for PCA adaptation")

    features_batches: list[torch.Tensor] = []
    sample_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting PCA features", dynamic_ncols=True):
            images = batch[image_key].to(device, non_blocking=True)
            features = backbone(normalize_for_backbone(images)).float().cpu()
            if max_samples is not None and sample_count + features.shape[0] > max_samples:
                features = features[: max_samples - sample_count]
            if features.numel() == 0:
                break
            features_batches.append(features)
            sample_count += features.shape[0]
            if max_samples is not None and sample_count >= max_samples:
                break

    if sample_count < 2:
        raise ValueError(f"Need at least 2 sampled crops to fit PCA head, got {sample_count}")

    feature_matrix = torch.cat(features_batches, dim=0)
    mean = feature_matrix.mean(dim=0)
    centered = feature_matrix - mean
    _, _, basis = torch.pca_lowrank(centered, q=embedding_dim, center=False)
    components = basis[:, :embedding_dim].T.contiguous()

    head = PCAEmbedder(input_dim, embedding_dim)
    with torch.no_grad():
        head.net.weight.copy_(components)
        head.net.bias.copy_(-(mean @ components.T))

    explained_variance = centered.pow(2).sum(dim=0).sum().clamp_min(1e-12)
    projected = centered @ components.T
    retained_variance = projected.pow(2).sum()
    stats = {
        "pca_fit_samples": int(sample_count),
        "pca_retained_variance_ratio": float((retained_variance / explained_variance).item()),
    }
    return head, stats


def save_checkpoint(output_dir: Path, head: PCAEmbedder, config: dict[str, Any], name: str = "best.pt") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / name
    torch.save({"model": head.state_dict(), "config": safe_json_config(config)}, checkpoint_path)
    return checkpoint_path


@click.command()
@click.argument("image_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("embedding_pca_runs/latest"))
@click.option("--backbone-name", default="vit_small_patch14_dinov2.lvd142m", show_default=True)
@click.option("--backbone-checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--crop-size", type=click.IntRange(min=32), default=224, show_default=True)
@click.option("--downsample-factor", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--embedding-dim", type=click.IntRange(min=1), default=96, show_default=True)
@click.option("--fit-split", type=click.Choice(["train", "test"]), default="train", show_default=True)
@click.option("--fit-samples", type=click.IntRange(min=2), default=16384, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--num-workers", type=click.IntRange(min=0), default=4, show_default=True)
@click.option("--test-fraction", type=click.FloatRange(min=0.05, max=0.5), default=0.15, show_default=True)
@click.option("--min-foreground-fraction", type=click.FloatRange(min=0.0, max=1.0), default=0.04, show_default=True)
@click.option("--foreground-threshold", type=click.FloatRange(min=0.0, max=1.0), default=0.2, show_default=True)
@click.option("--max-crop-attempts", type=click.IntRange(min=1), default=24, show_default=True)
@click.option("--seed", type=int, default=1337, show_default=True)
@click.option("--device", type=str, default=None)
def main(
    image_dir: Path,
    output_dir: Path,
    backbone_name: str,
    backbone_checkpoint: Path | None,
    crop_size: int,
    downsample_factor: int,
    embedding_dim: int,
    fit_split: str,
    fit_samples: int,
    batch_size: int,
    num_workers: int,
    test_fraction: float,
    min_foreground_fraction: float,
    foreground_threshold: float,
    max_crop_attempts: int,
    seed: int,
    device: str | None,
) -> None:
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone, backbone_dim = create_frozen_dino_backbone(backbone_name, backbone_checkpoint, crop_size, resolved_device)
    config = {
        "image_dir": image_dir.resolve(),
        "output_dir": output_dir.resolve(),
        "adaptation_method": "pca",
        "head_type": "pca",
        "backbone_name": backbone_name,
        "backbone_checkpoint": backbone_checkpoint,
        "crop_size": crop_size,
        "downsample_factor": downsample_factor,
        "embedding_dim": embedding_dim,
        "hidden_dim": embedding_dim,
        "dropout": 0.0,
        "fit_split": fit_split,
        "fit_samples": fit_samples,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "test_fraction": test_fraction,
        "min_foreground_fraction": min_foreground_fraction,
        "foreground_threshold": foreground_threshold,
        "max_crop_attempts": max_crop_attempts,
        "seed": seed,
        "device": str(resolved_device),
        "backbone_dim": backbone_dim,
    }

    dataset = build_dataset(image_dir, fit_split, fit_samples, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    head, pca_stats = fit_pca_embedder(
        dataloader=dataloader,
        backbone=backbone,
        device=resolved_device,
        input_dim=backbone_dim,
        embedding_dim=embedding_dim,
        image_key="image",
        max_samples=fit_samples,
    )

    config.update(pca_stats)
    config["fit_dataset"] = asdict(dataset.config)
    config["fit_split_files"] = [str(path) for path in dataset.paths]

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(safe_json_config(config), handle, indent=2)
    save_checkpoint(output_dir, head, config)


if __name__ == "__main__":
    main()
