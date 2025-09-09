#!/usr/bin/env python3
"""
Script to randomly sample data from a dataset.

Supports two modes:
- Paired mode (default): randomly sample image/label pairs from
  ``source/images`` and ``source/labels`` and COPY them to destination.
- Images-only fallback: if no labels are present or no pairs can be found,
  randomly sample images from ``source/images`` and MOVE them to
  ``destination/images``.

This keeps existing behavior for paired datasets and makes the script usable
for unlabeled datasets by moving only images when labels are absent.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


def get_matching_files(images_dir: Path, labels_dir: Path, label_suffix: str = '') -> List[Tuple[Path, Path]]:
    """
    Get matching image and label file pairs.
    
    Args:
        images_dir: Path to images directory
        labels_dir: Path to labels directory
        label_suffix: Suffix that labels have (e.g., '_label', '_mask')
        
    Returns:
        List of (image_path, label_path) tuples
    """
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy', '.npz'}
    image_files: List[Path] = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))
    
    # Match with corresponding labels
    matched_pairs = []
    for img_path in image_files:
        # Look for matching label file (same name + suffix, possibly different extension)
        stem = img_path.stem
        label_pattern = f'{stem}{label_suffix}.*'
        label_candidates = list(labels_dir.glob(label_pattern))
        
        if label_candidates:
            # Take the first matching label file
            matched_pairs.append((img_path, label_candidates[0]))
        else:
            # No label match for this image
            print(f"Warning: No matching label found for {img_path.name} (searched for {label_pattern})")
    
    return matched_pairs


def _discover_images(images_dir: Path) -> List[Path]:
    """Return a list of image files in ``images_dir``.

    Recognizes common raster formats and numpy arrays.
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy', '.npz'}
    files: List[Path] = []
    for ext in image_extensions:
        files.extend(images_dir.glob(f'*{ext}'))
        files.extend(images_dir.glob(f'*{ext.upper()}'))
    return files


def random_sample_and_copy(
    source_dir: Path,
    dest_dir: Path,
    num_samples: int,
    seed: int = None,
    label_suffix: str = ''
) -> None:
    """
    Randomly sample image-label pairs and copy to destination.
    
    Args:
        source_dir: Source directory containing 'images' and 'labels' subdirectories
        dest_dir: Destination directory
        num_samples: Number of samples to select
        seed: Random seed for reproducibility
        label_suffix: Suffix that labels have (e.g., '_label', '_mask')
    """
    # Verify source structure
    images_dir = source_dir / 'images'
    labels_dir = source_dir / 'labels'

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    labels_exist = labels_dir.exists()

    # Get matching pairs if labels exist
    matched_pairs: List[Tuple[Path, Path]] = []
    if labels_exist:
        matched_pairs = get_matching_files(images_dir, labels_dir, label_suffix)
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # If we have matched pairs, proceed with original paired copy behavior
    if matched_pairs:
        if num_samples > len(matched_pairs):
            print(f"Warning: Requested {num_samples} samples but only {len(matched_pairs)} pairs available.")
            num_samples = len(matched_pairs)

        selected_pairs = random.sample(matched_pairs, num_samples)

        # Create destination directories
        dest_images_dir = dest_dir / 'images'
        dest_labels_dir = dest_dir / 'labels'
        dest_images_dir.mkdir(parents=True, exist_ok=True)
        dest_labels_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        print(f"Copying {num_samples} randomly selected pairs...")
        for i, (img_path, label_path) in enumerate(selected_pairs, 1):
            # Copy image
            dest_img_path = dest_images_dir / img_path.name
            shutil.copy2(img_path, dest_img_path)

            # Copy label
            dest_label_path = dest_labels_dir / label_path.name
            shutil.copy2(label_path, dest_label_path)

            if i % 10 == 0:
                print(f"Copied {i}/{num_samples} pairs")

        print(f"Successfully copied {num_samples} image-label pairs to {dest_dir}")
        return

    # Fallback: images-only mode when labels are absent or no pairs found
    if not labels_exist:
        print(f"Labels directory not found at {labels_dir}. Proceeding in images-only mode.")
    else:
        print("No matching image-label pairs found. Proceeding in images-only mode (moving only images).")

    image_files = _discover_images(images_dir)
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    if num_samples > len(image_files):
        print(f"Warning: Requested {num_samples} samples but only {len(image_files)} images available.")
        num_samples = len(image_files)

    selected_images = random.sample(image_files, num_samples)

    dest_images_dir = dest_dir / 'images'
    dest_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Moving {num_samples} randomly selected images...")
    for i, img_path in enumerate(selected_images, 1):
        dest_img_path = dest_images_dir / img_path.name
        shutil.move(str(img_path), str(dest_img_path))
        if i % 10 == 0:
            print(f"Moved {i}/{num_samples} images")

    print(f"Successfully moved {num_samples} images to {dest_images_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample from a dataset. Copies image/label pairs when labels are present; "
            "otherwise moves only images (images-only mode)."
        )
    )
    parser.add_argument(
        'source',
        type=Path,
        help='Source directory containing images/ and labels/ subdirectories'
    )
    parser.add_argument(
        'destination',
        type=Path,
        help='Destination directory for sampled data'
    )
    parser.add_argument(
        'num_samples',
        type=int,
        help='Number of items to sample (pairs when labels exist, images otherwise)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible sampling'
    )
    parser.add_argument(
        '--label-suffix',
        type=str,
        default='',
        help='Suffix that label files have (e.g., "_label", "_mask")'
    )
    
    args = parser.parse_args()
    
    try:
        random_sample_and_copy(
            args.source,
            args.destination,
            args.num_samples,
            args.seed,
            args.label_suffix
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
