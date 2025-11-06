#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.ndimage import zoom
from tifffile import TiffFile, imread, imwrite
from tqdm import tqdm


Shape3D = Tuple[int, int, int]


@dataclass(frozen=True)
class ResizeTask:
    source: Path
    destination: Path
    original_shape: Shape3D
    target_shape: Shape3D
    scale_factor: float
    order: int  # 1 for trilinear, 0 for nearest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uniformly scale all 3D TIFF volumes so the smallest image reaches the target size, "
            "then center crop everything down to that target."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing images/ and labels/ subdirectories.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=3,
        metavar=("DEPTH", "HEIGHT", "WIDTH"),
        required=True,
        help="Desired output size (D H W) after scaling and center cropping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory that will receive images/ and labels/ outputs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Process pool size for resizing work. Defaults to CPU count minus one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate planned work without writing any files.",
    )
    return parser.parse_args()


def load_volume_shape(path: Path) -> Shape3D:
    with TiffFile(path) as tif:
        series = tif.series[0]
        shape = series.shape
    if len(shape) != 3:
        raise ValueError(f"{path} is not 3D (shape={shape})")
    return int(shape[0]), int(shape[1]), int(shape[2])


def gather_tifs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    candidates = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ]
    return sorted(candidates)


def choose_smallest_shape(volume_shapes: dict[Path, Shape3D]) -> tuple[Path, Shape3D]:
    def score(item: tuple[Path, Shape3D]) -> tuple[int, Shape3D]:
        _, shape = item
        voxels = math.prod(shape)
        return voxels, shape

    smallest_item = min(volume_shapes.items(), key=score)
    return smallest_item


def ensure_shape(volume: np.ndarray, desired_shape: Shape3D) -> np.ndarray:
    if volume.shape == desired_shape:
        return volume

    slices = tuple(slice(0, min(cur, tgt)) for cur, tgt in zip(volume.shape, desired_shape))
    trimmed = volume[slices]
    if trimmed.shape != desired_shape:
        pad_width = [(0, tgt - cur) for cur, tgt in zip(trimmed.shape, desired_shape)]
        trimmed = np.pad(trimmed, pad_width=pad_width, mode="edge")
    return trimmed


def center_crop(volume: np.ndarray, target_shape: Shape3D) -> np.ndarray:
    if any(cur < tgt for cur, tgt in zip(volume.shape, target_shape)):
        raise ValueError(
            f"Cannot center crop from shape {volume.shape} down to smaller target {target_shape}."
        )
    slices = []
    for cur, tgt in zip(volume.shape, target_shape):
        start = (cur - tgt) // 2
        end = start + tgt
        slices.append(slice(start, end))
    return volume[tuple(slices)]


def resize_volume(volume: np.ndarray, task: ResizeTask) -> np.ndarray:
    scale_factor = task.scale_factor
    scaled_shape = tuple(math.ceil(dim * scale_factor) for dim in volume.shape)

    if any(dim == 0 for dim in scaled_shape):
        raise ValueError(f"Invalid scaled shape computed from {volume.shape} with factor {scale_factor}.")

    if scaled_shape == volume.shape:
        resized = volume
    else:
        zoom_factors = tuple(scaled / original for original, scaled in zip(volume.shape, scaled_shape))

        if task.order == 1 and not np.issubdtype(volume.dtype, np.floating):
            working = volume.astype(np.float32, copy=False)
        else:
            working = volume

        resized = zoom(
            working,
            zoom=zoom_factors,
            order=task.order,
            prefilter=False,
            mode="nearest",
            grid_mode=False,
        )

        resized = ensure_shape(resized, scaled_shape)

        if task.order == 1:
            if np.issubdtype(volume.dtype, np.integer):
                info = np.iinfo(volume.dtype)
                resized = np.clip(resized, info.min, info.max)
            resized = resized.astype(volume.dtype, copy=False)
        else:
            resized = resized.astype(volume.dtype, copy=False)

    return resized

def process_task(task: ResizeTask, dry_run: bool) -> str:
    src = task.source
    dst = task.destination

    scaled_shape = tuple(math.ceil(dim * task.scale_factor) for dim in task.original_shape)

    if any(scaled < tgt for scaled, tgt in zip(scaled_shape, task.target_shape)):
        raise ValueError(
            f"Scale factor {task.scale_factor:.4f} insufficient to reach target {task.target_shape} "
            f"from original shape {task.original_shape} ({src})."
        )

    if dry_run:
        if scaled_shape == task.original_shape and task.original_shape == task.target_shape:
            return f"[DRY-RUN] {src.name}: already matches target; copy to destination."
        if scaled_shape == task.original_shape:
            return (
                f"[DRY-RUN] {src.name}: scale factor x{task.scale_factor:.4f} leaves shape unchanged; "
                f"center crop {task.original_shape} -> {task.target_shape}."
            )
        return (
            f"[DRY-RUN] {src.name}: scale x{task.scale_factor:.4f} "
            f"{task.original_shape} -> {scaled_shape}, then center crop to {task.target_shape}."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)

    if scaled_shape == task.original_shape and task.original_shape == task.target_shape:
        shutil.copy2(src, dst)
        return f"{src.name}: copied"

    volume = imread(src)
    resized = resize_volume(volume, task)
    if any(cur < tgt for cur, tgt in zip(resized.shape, task.target_shape)):
        raise ValueError(
            f"Resized volume shape {resized.shape} is smaller than target {task.target_shape} for {src}"
        )
    cropped = center_crop(resized, task.target_shape) if resized.shape != task.target_shape else resized
    imwrite(dst, cropped)
    return f"{src.name}: scaled to {resized.shape} and cropped to {task.target_shape}"


def run_task(task: ResizeTask) -> str:
    return process_task(task, dry_run=False)


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    target_shape = tuple(int(x) for x in args.target_size)
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    out_images_dir = output_dir / "images"
    out_labels_dir = output_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    image_files = gather_tifs(images_dir)
    if not image_files:
        raise FileNotFoundError(f"No TIFF images found in {images_dir}")

    image_shapes = {}
    for path in tqdm(image_files, desc="Scanning images", unit="file"):
        image_shapes[path] = load_volume_shape(path)

    smallest_image_path, smallest_shape = choose_smallest_shape(image_shapes)

    scale_factor = max(
        tgt / cur if cur else float("inf") for cur, tgt in zip(smallest_shape, target_shape)
    )
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError(
            f"Unable to compute valid scale factor from smallest image shape {smallest_shape} "
            f"and target {target_shape}."
        )

    print(
        f"Smallest image: {smallest_image_path.name} shape={smallest_shape}, "
        f"scale factor={scale_factor:.4f}"
    )

    tasks: list[ResizeTask] = []
    for path in image_files:
        tasks.append(
            ResizeTask(
                source=path,
                destination=out_images_dir / path.name,
                original_shape=image_shapes[path],
                target_shape=target_shape,
                scale_factor=scale_factor,
                order=1,
            )
        )

    label_files = gather_tifs(labels_dir)
    label_shapes = {}
    for path in tqdm(label_files, desc="Scanning labels", unit="file"):
        label_shapes[path] = load_volume_shape(path)

    for path in label_files:
        tasks.append(
            ResizeTask(
                source=path,
                destination=out_labels_dir / path.name,
                original_shape=label_shapes[path],
                target_shape=target_shape,
                scale_factor=scale_factor,
                order=0,
            )
        )

    if args.dry_run:
        for task in tasks:
            message = process_task(task, dry_run=True)
            print(message)
        return

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    num_workers = args.num_workers
    if num_workers < 1:
        num_workers = 1

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(run_task, tasks),
                total=len(tasks),
                desc="Resizing volumes",
                unit="file",
            )
        )

    for message in results:
        print(message)


if __name__ == "__main__":
    main()
