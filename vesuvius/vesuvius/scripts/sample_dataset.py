#!/usr/bin/env python3
import os
import random
import shutil
from pathlib import Path

# Set paths
base_dir = Path("/mnt/raid_nvme/datasets/nnUNet_raw/Dataset059")
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# Create output directories
output_base = base_dir.parent / "Dataset_059_10_percent_sample"
output_images = output_base / "images"
output_labels = output_base / "labels"

# Create directories if they don't exist
output_images.mkdir(parents=True, exist_ok=True)
output_labels.mkdir(parents=True, exist_ok=True)

# Get all image files
image_files = sorted([f for f in images_dir.iterdir() if f.suffix == '.tif'])
print(f"Total images found: {len(image_files)}")

# Calculate 10% sample size
sample_size = int(len(image_files) * 0.1)
print(f"Sampling {sample_size} images (10%)")

# Randomly sample images
sampled_images = random.sample(image_files, sample_size)

# Copy sampled images and their corresponding labels
copied_count = 0
for image_path in sampled_images:
    # Get the base name without extension
    base_name = image_path.stem
    
    # Construct label filename
    label_name = f"{base_name}_surface.tif"
    label_path = labels_dir / label_name
    
    # Check if label exists
    if label_path.exists():
        # Copy image
        shutil.copy2(image_path, output_images / image_path.name)
        # Copy label
        shutil.copy2(label_path, output_labels / label_name)
        copied_count += 1
    else:
        print(f"Warning: Label not found for {image_path.name}")

print(f"\nSuccessfully copied {copied_count} image-label pairs to:")
print(f"  Images: {output_images}")
print(f"  Labels: {output_labels}")