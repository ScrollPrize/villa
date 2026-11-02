#!/usr/bin/env python
import os
import glob
import argparse
import multiprocessing

import numpy as np
from vc_reader import images
from tqdm import tqdm

import vc_reader.transforms as A

_DISTANCE = A.DistanceTransform(mode="outside", implicit_background=True)
_RIDGES = A.Pipeline([A.GaussianBlur(sigma=2, boundary="reflect", truncate=4, full_width=False),
                      A.Normalize("minmax"), A.Ridges()], backend="cpu")


def process_file(file_path, output_folder, dilation_distance=3, ridge_threshold=0.5):
    try:
        volume = images.read(file_path)
        
        if volume.ndim not in (2, 3):
            raise ValueError(f"Unsupported image dimensions: {volume.ndim}. Only 2D and 3D images are supported.")
        binary = (volume > 0).astype(np.uint8)
        distance = _DISTANCE(binary[None])[0]
        dilated = (1.0 / (distance.astype(np.float64) + 1e-6) > 1.0 / dilation_distance).astype(np.float32)
        ridges = _RIDGES({"image": dilated[None]}, seed=0)["image"][0]
        binary_ridges = (ridges > ridge_threshold).astype(np.uint8)

        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)
        images.write_tiff(output_path, binary_ridges, compression='packbits')
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
