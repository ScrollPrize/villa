#!/usr/bin/env python3
"""
Mask images by setting pixels to 0 wherever the corresponding label equals 2 (ignore label).
"""

import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import tifffile
from tqdm import tqdm


# Directories
IMAGES_DIR = Path("/mnt/raid_nvme/datasets/raw/Dataset110_kaggle/eval/images")
LABELS_DIR = Path("/mnt/raid_nvme/datasets/raw/Dataset110_kaggle/eval/labels")
OUTPUT_DIR = Path("/mnt/raid_nvme/datasets/raw/Dataset110_kaggle/eval/images_masked")

IGNORE_LABEL = 2


def get_label_path(image_path: Path) -> Path:
    """Map image filename to label filename.

    Image: sample_XXXXX_0000.tif -> Label: sample_XXXXX.tif
    """
    # Remove the _0000 suffix
    stem = image_path.stem  # sample_XXXXX_0000
    label_stem = stem.rsplit("_", 1)[0]  # sample_XXXXX
    return LABELS_DIR / f"{label_stem}.tif"


def process_file(image_path: Path) -> str:
    """Process a single image file, masking ignore labels."""
    label_path = get_label_path(image_path)
    output_path = OUTPUT_DIR / image_path.name

    # Load image and label
    image = tifffile.imread(image_path)
    label = tifffile.imread(label_path)

    # Create mask where label == ignore label and set those pixels to 0
    mask = label == IGNORE_LABEL
    image[mask] = 0

    # Save with LZW compression
    tifffile.imwrite(output_path, image, compression='lzw')

    return image_path.name


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = sorted(IMAGES_DIR.glob("sample_*_0000.tif"))
    print(f"Found {len(image_files)} images to process")

    # Verify all labels exist
    missing_labels = []
    for img_path in image_files:
        label_path = get_label_path(img_path)
        if not label_path.exists():
            missing_labels.append(label_path)

    if missing_labels:
        print(f"Error: {len(missing_labels)} label files missing:")
        for p in missing_labels[:5]:
            print(f"  {p}")
        return

    # Process in parallel
    num_workers = min(cpu_count(), 16)  # Cap at 16 workers
    print(f"Processing with {num_workers} workers...")

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_file, image_files),
            total=len(image_files),
            desc="Masking images"
        ))

    print(f"Done! Processed {len(results)} files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
