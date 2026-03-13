#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import click
import cv2

from dataset import DatasetConfig, IndexedInkCropDataset, build_eval_augmentations, build_train_augmentations


def build_dataset(
    image_dir: Path,
    split: str,
    samples: int,
    seed: int,
    crop_size: int,
    downsample_factor: int,
    test_fraction: float,
    min_foreground_fraction: float,
    foreground_threshold: float,
    max_crop_attempts: int,
) -> IndexedInkCropDataset:
    config = DatasetConfig(
        image_dir=image_dir.resolve(),
        split=split,
        seed=seed,
        crop_size=crop_size,
        downsample_factor=downsample_factor,
        samples_per_epoch=samples,
        min_foreground_fraction=min_foreground_fraction,
        max_crop_attempts=max_crop_attempts,
        test_fraction=test_fraction,
        foreground_threshold=foreground_threshold,
        cache_images=True,
    )
    augmentation = (
        build_train_augmentations(config.crop_size)
        if split == "train"
        else build_eval_augmentations(config.crop_size)
    )
    return IndexedInkCropDataset(config, augmentation)


def dump_split(
    image_dir: Path,
    output_dir: Path,
    split: str,
    samples: int,
    seed: int,
    crop_size: int,
    downsample_factor: int,
    test_fraction: float,
    min_foreground_fraction: float,
    foreground_threshold: float,
    max_crop_attempts: int,
) -> None:
    dataset = build_dataset(
        image_dir=image_dir,
        split=split,
        samples=samples,
        seed=seed,
        crop_size=crop_size,
        downsample_factor=downsample_factor,
        test_fraction=test_fraction,
        min_foreground_fraction=min_foreground_fraction,
        foreground_threshold=foreground_threshold,
        max_crop_attempts=max_crop_attempts,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    count = min(samples, len(dataset))
    for index in range(count):
        sample = dataset[index]
        image = sample["image"][0].mul(255.0).round().to(dtype=sample["image"].dtype).numpy()
        crop_u8 = image.clip(0, 255).astype("uint8")
        filename = (
            f"sample_{int(sample['sample_index']):05d}"
            f"__img_{int(sample['image_index']):03d}"
            f"__y_{int(sample['top']):05d}"
            f"__x_{int(sample['left']):05d}.png"
        )
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), crop_u8)
        manifest.append(
            {
                "sample_index": int(sample["sample_index"]),
                "image_index": int(sample["image_index"]),
                "top": int(sample["top"]),
                "left": int(sample["left"]),
                "foreground_fraction": float(sample["foreground_fraction"]),
                "path": sample["path"],
                "output_path": str(output_path.resolve()),
            }
        )

    payload = {
        "image_dir": str(image_dir.resolve()),
        "split": split,
        "samples_requested": samples,
        "samples_written": count,
        "seed": seed,
        "crop_size": crop_size,
        "downsample_factor": downsample_factor,
        "test_fraction": test_fraction,
        "min_foreground_fraction": min_foreground_fraction,
        "foreground_threshold": foreground_threshold,
        "max_crop_attempts": max_crop_attempts,
        "records": manifest,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@click.command()
@click.argument("image_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("embedding_dataset_dump/latest"))
@click.option("--samples", type=click.IntRange(min=1), default=200, show_default=True)
@click.option("--seed", type=int, default=1337, show_default=True)
@click.option("--crop-size", type=click.IntRange(min=32), default=224, show_default=True)
@click.option("--downsample-factor", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--test-fraction", type=click.FloatRange(min=0.05, max=0.5), default=0.15, show_default=True)
@click.option("--min-foreground-fraction", type=click.FloatRange(min=0.0, max=1.0), default=0.04, show_default=True)
@click.option("--foreground-threshold", type=click.FloatRange(min=0.0, max=1.0), default=0.2, show_default=True)
@click.option("--max-crop-attempts", type=click.IntRange(min=1), default=24, show_default=True)
def main(
    image_dir: Path,
    output_dir: Path,
    samples: int,
    seed: int,
    crop_size: int,
    downsample_factor: int,
    test_fraction: float,
    min_foreground_fraction: float,
    foreground_threshold: float,
    max_crop_attempts: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        dump_split(
            image_dir=image_dir,
            output_dir=output_dir / split,
            split=split,
            samples=samples,
            seed=seed,
            crop_size=crop_size,
            downsample_factor=downsample_factor,
            test_fraction=test_fraction,
            min_foreground_fraction=min_foreground_fraction,
            foreground_threshold=foreground_threshold,
            max_crop_attempts=max_crop_attempts,
        )


if __name__ == "__main__":
    main()
