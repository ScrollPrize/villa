#!/usr/bin/env python3
"""
Script to move image-label pairs where labels contain only the ignore label.
For nnUNet datasets where images have _0000 suffix and labels don't.
"""
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import json


def contains_only_ignore_label(label_path, ignore_value=2):
    """Check if a label file contains only the ignore label value."""
    # Load the label image
    label = np.array(Image.open(label_path))
    
    # Get unique values in the label
    unique_values = np.unique(label)
    
    # Check if it only contains the ignore value
    return len(unique_values) == 1 and unique_values[0] == ignore_value


def update_dataset_json(dataset_path, new_count):
    """Update the numTraining field in dataset.json."""
    json_path = dataset_path / "dataset.json"
    
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        old_count = data.get('numTraining', 0)
        data['numTraining'] = new_count
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"\nUpdated dataset.json: numTraining {old_count} -> {new_count}")
    else:
        print(f"\nWarning: dataset.json not found at {json_path}")


def move_ignore_only_pairs(dataset_path, ignore_value=2, dry_run=False):
    """
    Move image-label pairs where labels contain only the ignore label.
    
    Args:
        dataset_path: Path to dataset folder containing imagesTr and labelsTr
        ignore_value: The value of the ignore label (default: 2)
        dry_run: If True, only print what would be moved without actually moving
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "imagesTr"
    labels_dir = dataset_path / "labelsTr"
    
    # Create output directories
    images_ignore_dir = dataset_path / "imagesTr_ignore_only"
    labels_ignore_dir = dataset_path / "labelsTr_ignore_only"
    
    if not dry_run:
        images_ignore_dir.mkdir(exist_ok=True)
        labels_ignore_dir.mkdir(exist_ok=True)
    
    # Get all label files
    label_files = sorted(list(labels_dir.glob("*.tif")))
    
    moved_count = 0
    total_count = len(label_files)
    
    print(f"Checking {total_count} label files...")
    
    for label_path in tqdm(label_files):
        # Check if label contains only ignore value
        if contains_only_ignore_label(label_path, ignore_value):
            # Get corresponding image name (add _0000 suffix)
            label_name = label_path.stem
            image_name = f"{label_name}_0000.tif"
            image_path = images_dir / image_name
            
            if not image_path.exists():
                print(f"Warning: Image not found for label {label_path.name}: {image_path}")
                continue
            
            # Move files
            if dry_run:
                print(f"Would move: {label_path.name} and {image_name}")
            else:
                # Move label
                shutil.move(str(label_path), str(labels_ignore_dir / label_path.name))
                # Move image
                shutil.move(str(image_path), str(images_ignore_dir / image_name))
                
            moved_count += 1
    
    print(f"\nSummary:")
    print(f"Total labels checked: {total_count}")
    print(f"Pairs with ignore-only labels: {moved_count}")
    print(f"Percentage: {moved_count/total_count*100:.1f}%")
    
    if not dry_run:
        print(f"\nMoved files to:")
        print(f"  Images: {images_ignore_dir}")
        print(f"  Labels: {labels_ignore_dir}")
        
        # Update dataset.json with new count
        update_dataset_json(dataset_path, total_count - moved_count)


def main():
    parser = argparse.ArgumentParser(description="Move image-label pairs where labels contain only ignore value")
    parser.add_argument("dataset_path", help="Path to dataset folder containing imagesTr and labelsTr")
    parser.add_argument("--ignore-value", type=int, default=2, help="Value of ignore label (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved without actually moving")
    
    args = parser.parse_args()
    
    # Validate paths
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    if not (dataset_path / "imagesTr").exists():
        print(f"Error: imagesTr folder not found in {dataset_path}")
        return
        
    if not (dataset_path / "labelsTr").exists():
        print(f"Error: labelsTr folder not found in {dataset_path}")
        return
    
    # Run the moving process
    move_ignore_only_pairs(dataset_path, args.ignore_value, args.dry_run)


if __name__ == "__main__":
    main()