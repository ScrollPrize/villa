#!/usr/bin/env python
import os
import glob
import argparse
import multiprocessing

import numpy as np
import tifffile
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, gaussian_filter, convolve, map_coordinates
from scipy import ndimage
from numpy import linalg as LA


# ---------------------------
# Utility and Processing Functions
# ---------------------------
def divide_nonzero(array1, array2, eps=1e-10):
    denominator = np.copy(array2)
    denominator[denominator == 0] = eps
    return np.divide(array1, denominator)


def normalize(volume):
    minim = np.min(volume)
    maxim = np.max(volume)
    if maxim - minim == 0:
        return volume
    volume = volume - minim
    volume = volume / (maxim - minim)
    return volume


def hessian(volume, gauss_sigma=2, sigma=6):
    volume_smoothed = gaussian_filter(volume, sigma=gauss_sigma)
    volume_smoothed = normalize(volume_smoothed)

    joint_hessian = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], 3, 3), dtype=float)

    Dz = np.gradient(volume_smoothed, axis=0, edge_order=2)
    joint_hessian[:, :, :, 2, 2] = np.gradient(Dz, axis=0, edge_order=2)
    del Dz

    Dy = np.gradient(volume_smoothed, axis=1, edge_order=2)
    joint_hessian[:, :, :, 1, 1] = np.gradient(Dy, axis=1, edge_order=2)
    joint_hessian[:, :, :, 1, 2] = np.gradient(Dy, axis=0, edge_order=2)
    del Dy

    Dx = np.gradient(volume_smoothed, axis=2, edge_order=2)
    joint_hessian[:, :, :, 0, 0] = np.gradient(Dx, axis=2, edge_order=2)
    joint_hessian[:, :, :, 0, 1] = np.gradient(Dx, axis=1, edge_order=2)
    joint_hessian[:, :, :, 0, 2] = np.gradient(Dx, axis=0, edge_order=2)
    del Dx

    joint_hessian = joint_hessian * (sigma ** 2)
    zero_mask = np.trace(joint_hessian, axis1=3, axis2=4) == 0
    return joint_hessian, zero_mask


def detect_ridges(volume, gamma=1.5, beta1=0.5, beta2=0.5, gauss_sigma=2, sigma=6):
    joint_hessian, zero_mask = hessian(volume, gauss_sigma, sigma)
    eigvals = LA.eigvalsh(joint_hessian, 'U')
    idxs = np.argsort(np.abs(eigvals), axis=-1)
    eigvals = np.take_along_axis(eigvals, idxs, axis=-1)
    eigvals[zero_mask, :] = 0

    L1 = np.abs(eigvals[:, :, :, 0])
    L2 = np.abs(eigvals[:, :, :, 1])
    L3 = eigvals[:, :, :, 2]
    L3abs = np.abs(L3)

    S = np.sqrt(np.square(eigvals).sum(axis=-1))
    background_term = 1 - np.exp(-0.5 * np.square(S / gamma))

    Ra = divide_nonzero(L2, L3abs)
    planar_term = np.exp(-0.5 * np.square(Ra / beta1))

    Rb = divide_nonzero(L1, np.sqrt(L2 * L3abs))
    blob_term = np.exp(-0.5 * np.square(Rb / beta2))

    ridges = background_term * planar_term * blob_term
    ridges[L3 > 0] = 0
    return ridges


def dilate_by_inverse_edt(binary_volume, dilation_distance):
    eps = 1e-6
    edt = distance_transform_edt(1 - binary_volume)
    inv_edt = 1.0 / (edt + eps)
    threshold = 1.0 / dilation_distance
    dilated = (inv_edt > threshold).astype(np.uint8)
    return dilated


def process_file(file_path, output_folder, dilation_distance=3, ridge_threshold=0.5):
    try:
        volume = tifffile.imread(file_path)
        binary_volume = (volume > 0).astype(np.uint8)
        dilated_volume = dilate_by_inverse_edt(binary_volume, dilation_distance)
        dilated_float = dilated_volume.astype(np.float32)
        ridges = detect_ridges(dilated_float)
        binary_ridges = (ridges > ridge_threshold).astype(np.uint8)
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)
        tifffile.imwrite(output_path, binary_ridges)
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
        description="Process a folder of multipage TIFFs with inverse EDT dilation and custom ridge detection."
    )
    parser.add_argument("input_folder", help="Folder containing input multipage TIFF files.")
    parser.add_argument("output_folder", help="Folder where processed TIFF files will be saved.")
    parser.add_argument("--dilation_distance", type=float, default=3, help="Dilation distance (in voxels).")
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
