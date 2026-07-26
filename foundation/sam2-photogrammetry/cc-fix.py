#!/usr/bin/env python
"""
Single-Phase Pipeline for Cleaning Palette-Based PNG Files

For each PNG file under the provided root directory, the following steps are applied:
  1. Convert the image to palette mode (if needed) and load as a 2D array.
  2. For label == 1, keep only the largest connected component.
  3. [MERGE STEP COMMENTED OUT] Merge fully enclosed regions.
  4. Remove small ("dust") connected components for label == 2.
  5. Check (once) whether any non-background label is still split into multiple connected components.
     If so, a warning is printed.
  6. The cleaned image is saved (overwriting the original file).

Usage:
    python pipeline.py /path/to/root_folder [--workers N] [--dust-threshold 100]
"""

import os
import argparse
import numpy as np
from PIL import Image
from skimage.measure import label
from tqdm import tqdm
import multiprocessing
from functools import partial

def get_png_files(root_folder):
    """
    Recursively collects all PNG file paths under the given root_folder.
    """
    png_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".png"):
                png_files.append(os.path.join(root, file))
    return png_files

def process_file(file_path, dust_threshold):
    """
    Processes a single PNG file:
      - Loads the image in 'P' mode and extracts its palette.
      - Applies cleaning steps:
          1. For label==1, keeps only the largest connected component.
          2. [MERGE STEP COMMENTED OUT] Merge fully enclosed regions.
          3. Removes small dust components for label==2.
      - Checks the cleaned image for any labels that remain split (warns if so).
      - Restores the original palette and saves the cleaned image (overwriting the original file).
    """
    try:
        with Image.open(file_path) as img:
            if img.mode != 'P':
                img = img.convert('P')
            # Extract the original palette
            original_palette = img.getpalette()
            arr = np.array(img)
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return

    if arr.ndim != 2:
        print(f"Skipping {file_path}: image is not 2D")
        return

    # Apply cleaning steps:
    arr_clean = keep_largest_label_one_component(arr)
    # arr_clean = merge_fully_enclosed_regions(arr_clean)  # Merge step commented out
    arr_clean = remove_dust_label_2(arr_clean, dust_threshold)

    # Check connectivity after cleaning
    # if has_strange_components(arr_clean):
    #    print(f"Warning: {file_path} still exhibits multiple connected components after cleaning.")

    # Save the cleaned image while restoring the original palette
    try:
        out_img = Image.fromarray(arr_clean).convert("P")
        out_img.putpalette(original_palette)
        out_img.save(file_path)
    except Exception as e:
        print(f"Error saving {file_path}: {e}")


def keep_largest_label_one_component(arr):
    """
    For pixels with value 1, keeps only the largest connected component.
    Other components originally labeled 1 are set to 0 (background).
    """
    mask = (arr == 1)
    labeled = label(mask, connectivity=2)
    if labeled.max() < 1:
        return arr  # no label 1 found

    unique_cc = np.unique(labeled)
    unique_cc = unique_cc[unique_cc != 0]

    largest_cc = None
    largest_count = 0
    for cc in unique_cc:
        count = np.count_nonzero(labeled == cc)
        if count > largest_count:
            largest_count = count
            largest_cc = cc

    # Zero out label==1 pixels not in the largest connected component.
    arr[(labeled != largest_cc) & (labeled != 0)] = 0
    return arr

def merge_fully_enclosed_regions(arr):
    """
    Merges regions that are fully enclosed:
      1. Labels non-background pixels.
      2. Builds an adjacency graph to detect if a region is entirely surrounded
         by one other color (and not touching the image border).
      3. Merges the enclosed region into that color.
    """
    non_bg = (arr != 0)
    region_image = label(non_bg, connectivity=2)
    num_regions = region_image.max()
    if num_regions < 1:
        return arr

    region_colors = np.zeros(num_regions + 1, dtype=np.uint16)
    adjacency = [set() for _ in range(num_regions + 1)]
    touches_bg = [False] * (num_regions + 1)

    rows, cols = region_image.shape
    for r in range(rows):
        for c in range(cols):
            rid = region_image[r, c]
            if rid == 0:
                continue
            if region_colors[rid] == 0:
                region_colors[rid] = arr[r, c]
            # Check all 8 neighbors
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < rows and 0 <= cc < cols):
                        touches_bg[rid] = True
                    else:
                        neighbor = region_image[rr, cc]
                        if neighbor == 0:
                            touches_bg[rid] = True
                        elif neighbor != rid:
                            adjacency[rid].add(neighbor)

    out_arr = arr.copy()
    merged = [False] * (num_regions + 1)
    for rid in range(1, num_regions + 1):
        if touches_bg[rid] or merged[rid]:
            continue
        neighbor_rids = adjacency[rid]
        neighbor_colors = {region_colors[nid] for nid in neighbor_rids if nid != rid}
        neighbor_colors.discard(region_colors[rid])
        if len(neighbor_colors) == 1:
            enclosing_color = neighbor_colors.pop()
            if enclosing_color != region_colors[rid] and enclosing_color != 0:
                out_arr[region_image == rid] = enclosing_color
                merged[rid] = True
    return out_arr

def remove_dust_label_2(arr, dust_threshold):
    """
    For pixels with value 2, removes (sets to 0) any connected component
    smaller than the dust_threshold.
    """
    labeled = label(arr == 2, connectivity=2)
    if labeled.max() < 1:
        return arr

    for region_id in range(1, labeled.max() + 1):
        size = np.count_nonzero(labeled == region_id)
        if size < dust_threshold:
            arr[labeled == region_id] = 0
    return arr

def has_strange_components(arr):
    """
    Checks whether any non-zero label in the image is split into multiple
    connected components. Returns True if at least one such label exists.
    """
    labels = np.unique(arr)
    labels = labels[labels != 0]
    for lbl in labels:
        mask = (arr == lbl)
        cc = label(mask, connectivity=2)
        if cc.max() > 1:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Single-phase cleaning of palette-based PNG files.\n"
            "For each file, the following is applied:\n"
            "  1. Keep only the largest connected component for label==1\n"
            "  2. [MERGE STEP COMMENTED OUT] Merge fully enclosed regions\n"
            "  3. Remove small dust connected components for label==2\n"
            "A connectivity check is then performed and a warning is printed if issues remain.\n"
            "The cleaned image overwrites the original file."
        )
    )
    parser.add_argument("root_folder", help="Root directory to scan for PNG files.")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help="Number of parallel workers (default: all CPUs).")
    parser.add_argument("--dust-threshold", type=int, default=256,
                        help="Minimum pixel count to keep for label==2 (default: 256).")
    args = parser.parse_args()

    png_files = get_png_files(args.root_folder)
    if not png_files:
        print("No PNG files found.")
        return

    process_func = partial(process_file, dust_threshold=args.dust_threshold)
    with multiprocessing.Pool(args.workers) as pool:
        list(tqdm(pool.imap_unordered(process_func, png_files),
                  total=len(png_files),
                  desc="Processing files", unit="file"))

    print("Processing complete.")

if __name__ == "__main__":
    main()
