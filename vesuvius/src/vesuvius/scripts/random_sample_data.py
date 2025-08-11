#!/usr/bin/env python3
"""
Script to randomly sample images and corresponding labels from a source directory
and copy them to a destination directory.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple


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
    image_files = []
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
            print(f"Warning: No matching label found for {img_path.name} (searched for {label_pattern})")
    
    return matched_pairs


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
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Get matching pairs
    matched_pairs = get_matching_files(images_dir, labels_dir, label_suffix)
    
    if not matched_pairs:
        raise ValueError("No matching image-label pairs found")
    
    if num_samples > len(matched_pairs):
        print(f"Warning: Requested {num_samples} samples but only {len(matched_pairs)} pairs available.")
        num_samples = len(matched_pairs)
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Random sampling
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


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample images and labels from a dataset"
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
        help='Number of image-label pairs to sample'
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