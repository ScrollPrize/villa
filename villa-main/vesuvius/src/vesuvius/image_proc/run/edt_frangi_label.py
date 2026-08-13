#!/usr/bin/env python
import os
import glob
import argparse
import multiprocessing

import numpy as np
import tifffile
from tqdm import tqdm

from vesuvius.image_proc.distance import dilate_by_inverse_edt
from vesuvius.image_proc.features.ridges_vessels import (
    detect_ridges_2d,
    detect_ridges_3d,
)


def process_file(file_path, output_folder, dilation_distance=3, ridge_threshold=0.5):
    try:
        volume = tifffile.imread(file_path)
        
        # Check if input is 2D or 3D
        if volume.ndim == 2:
            # 2D processing
            binary_volume = (volume > 0).astype(np.uint8)
            dilated_volume = dilate_by_inverse_edt(binary_volume, dilation_distance)
            dilated_float = dilated_volume.astype(np.float32)
            ridges = detect_ridges_2d(dilated_float)
            binary_ridges = (ridges > ridge_threshold).astype(np.uint8)
        elif volume.ndim == 3:
            # 3D processing
            binary_volume = (volume > 0).astype(np.uint8)
            dilated_volume = dilate_by_inverse_edt(binary_volume, dilation_distance)
            dilated_float = dilated_volume.astype(np.float32)
            ridges = detect_ridges_3d(dilated_float)
            binary_ridges = (ridges > ridge_threshold).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image dimensions: {volume.ndim}. Only 2D and 3D images are supported.")
        
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)
        tifffile.imwrite(output_path, binary_ridges, compression='packbits')
        return file_path, True
    except Exception as e:
        return file_path, False, str(e)


# Define a top-level worker function to avoid lambda pickling issues
def worker(args):
    return process_file(*args)


# ---------------------------
# Main Routine Using Multiprocessing and tqdm
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Process a folder of 2D or 3D TIFF files with inverse EDT dilation and custom ridge detection."
    )
    parser.add_argument("input_folder", help="Folder containing input TIFF files (2D or 3D).")
    parser.add_argument("output_folder", help="Folder where processed TIFF files will be saved.")
    parser.add_argument("--dilation_distance", type=float, default=3, help="Dilation distance (in pixels/voxels).")
    parser.add_argument("--ridge_threshold", type=float, default=0.5,
                        help="Threshold for binarizing the ridge detection.")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(),
                        help="Number of worker processes.")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    file_list = glob.glob(os.path.join(args.input_folder, "*.tif"))
    if not file_list:
        print(f"No .tif files found in {args.input_folder}")
        return

    tasks = [
        (f, args.output_folder, args.dilation_distance, args.ridge_threshold)
        for f in file_list
    ]

    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker, tasks),
                total=len(tasks),
                desc="Processing Files"
            )
        )

    successes = [res for res in results if res[1] is True]
    failures = [res for res in results if res[1] is not True]

    print(f"Processed {len(successes)} files successfully.")
    if failures:
        print(f"{len(failures)} files failed:")
        for fail in failures:
            print(f"File: {fail[0]}, Error: {fail[2]}")


if __name__ == "__main__":
    main()
