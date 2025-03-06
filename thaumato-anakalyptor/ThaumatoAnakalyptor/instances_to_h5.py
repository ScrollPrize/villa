#!/usr/bin/env python3
"""
instances_to_h5.py

This tool converts produced instance archives (stored as 7z or tar files) into a single HDF5 file.
Each archive is expected to contain one or more PLY files (e.g. “surface_0.ply”, “surface_1.ply”, …)
and, optionally, matching metadata JSON files named like “metadata_surface_0.json” for each surface.

Usage:
    python instances_to_h5.py --input_dir /path/to/archives --output_h5 /path/to/output.h5 [--threads N]
"""

import os
import argparse
import tempfile
import shutil
import json
import time
import tarfile
from math import ceil
from tqdm import tqdm

import h5py
import numpy as np
import open3d as o3d
import py7zr
from concurrent.futures import ProcessPoolExecutor, as_completed

def setup_h5_partials_folder(h5_path):
    """Creates a fresh h5_partials folder, deleting it first if it exists."""
    partials_dir = os.path.join(os.path.dirname(h5_path), "h5_partials")
    if os.path.exists(partials_dir):
        response = input(f"The folder '{partials_dir}' already exists. Delete and recreate? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without modifying h5_partials.")
            exit(1)
        shutil.rmtree(partials_dir)
    os.makedirs(partials_dir)
    return partials_dir

def load_ply_file(filepath):
    """
    Loads a PLY file using Open3D and returns its points, normals, and colors.
    """
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    colors = np.asarray(pcd.colors)
    return points, normals, colors


def extract_archive(archive_path, extract_dir):
    """
    Extracts a 7z or tar archive into the given directory.
    """
    if archive_path.lower().endswith('.7z'):
        with py7zr.SevenZipFile(archive_path, 'r') as archive:
            archive.extractall(path=extract_dir)
    elif archive_path.lower().endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar:
            tar.extractall(path=extract_dir)
    else:
        raise ValueError(f"Unsupported archive format for {archive_path}")


def process_instance_archive(archive_path, h5_file, group_prefix="", compression="lzf", compression_level=4):
    """
    Extracts an instance archive, reads its PLY and metadata files, and writes them into the HDF5 file.
    
    Each archive becomes an HDF5 group (named after the archive filename). Inside that group, each
    surface is stored in a subgroup (named after the PLY file, without extension) with dataset:
      - "points"
      - "colors" (only the second entry per point is stored)
    
    Any metadata is stored as attributes.
    
    Returns:
        extraction_duration (float): Time taken for archive extraction (seconds).
        group_creation_duration (float): Time taken for processing PLY files and writing HDF5 groups (seconds).
    """
    temp_dir = tempfile.mkdtemp()
    extraction_duration = 0.0
    group_creation_duration = 0.0
    try:
        t0 = time.perf_counter()
        extract_archive(archive_path, temp_dir)
        extraction_duration = time.perf_counter() - t0

        # Search recursively for PLY files.
        ply_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.ply'):
                    ply_files.append(os.path.join(root, file))
                    
        if not ply_files:
            print(f"[WARNING] No .ply files found in archive {archive_path}.")
            return extraction_duration, group_creation_duration

        t1 = time.perf_counter()
        archive_basename = os.path.basename(archive_path)
        group_name = os.path.splitext(archive_basename)[0]
        if group_prefix:
            group_name = os.path.join(group_prefix, group_name)
        if group_name in h5_file:
            del h5_file[group_name]
        grp = h5_file.create_group(group_name)
        
        for ply_file in ply_files:
            base = os.path.splitext(os.path.basename(ply_file))[0]
            metadata_file = os.path.join(os.path.dirname(ply_file), f"metadata_{base}.json")
            
            try:
                points, normals, colors = load_ply_file(ply_file)
            except Exception as e:
                print(f"[ERROR] Reading {ply_file}: {e}")
                continue

            # Use only the second entry of the colors data
            if colors.ndim == 2 and colors.shape[1] >= 2:
                colors = colors[:, 1]
            else:
                print(f"[WARNING] Colors data in {ply_file} does not have at least 2 columns.")
                continue

            metadata = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    print(f"[WARNING] Reading metadata file {metadata_file}: {e}")

            surface_grp = grp.create_group(base)
            surface_grp.create_dataset("points", data=points, compression=compression,
                                         compression_opts=compression_level if compression == 'gzip' else None)
            # Normals are no longer saved.
            surface_grp.create_dataset("colors", data=colors, compression=compression,
                                         compression_opts=compression_level if compression == 'gzip' else None)
            
            for key, value in metadata.items():
                try:
                    surface_grp.attrs[key] = value
                except Exception as e:
                    print(f"[WARNING] Setting attribute {key} for {base}: {e}")
        group_creation_duration = time.perf_counter() - t1

        return extraction_duration, group_creation_duration

    finally:
        shutil.rmtree(temp_dir)


def process_archives_chunk(archives, partial_h5_filename, group_prefix, compression, compression_level=4):
    """
    Processes a list of archives and writes the results into a partial HDF5 file.
    
    Returns a tuple (total_extraction_time, total_group_time).
    """
    total_extraction_time = 0.0
    total_group_time = 0.0
    with h5py.File(partial_h5_filename, "w") as h5_file:
        for archive in tqdm(archives, desc=f"Thread processing {os.path.basename(partial_h5_filename)}", leave=False):
            extr_time, grp_time = process_instance_archive(archive, h5_file, group_prefix, compression, compression_level=compression_level)
            total_extraction_time += extr_time
            total_group_time += grp_time
    return total_extraction_time, total_group_time


def merge_h5_files(partial_files, final_filename):
    """
    Merges multiple partial HDF5 files into one final HDF5 file.
    """
    print(f"Merging {len(partial_files)} partial HDF5 files into {final_filename}")
    with h5py.File(final_filename, "w") as h5_final:
        for pf in tqdm(partial_files, desc="Merging partial HDF5 files"):
            with h5py.File(pf, "r") as h5_part:
                for group in tqdm(h5_part, desc=f"Merging {os.path.basename(pf)}"):
                    # Copy each group from the partial file into the final file.
                    h5_part.copy(group, h5_final, name=group)


def main():
    parser = argparse.ArgumentParser(
        description="Convert instance archives (7z/tar) to a single HDF5 file with compressed datasets."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing instance archives (7z or tar files).")
    parser.add_argument("--output_h5", type=str, required=True,
                        help="Path for the output HDF5 file.")
    parser.add_argument("--compression", type=str, default="lzf",
                        help="Compression algorithm to use for HDF5 datasets (e.g., 'lzf', 'gzip').")
    parser.add_argument("--group_prefix", type=str, default="",
                        help="Optional prefix for group names in the HDF5 file.")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads (processes) to use. If >1, partial HDF5 files are created and merged.")
    parser.add_argument("--compression_level", type=int, default=4,
                        help="Compression level (0-9, only for gzip, where 0 is no compression and 9 is max compression).")
    args = parser.parse_args()

    print(f"Converting instance archives in {args.input_dir} to HDF5 file: {args.output_h5}")

    # Find all archive files.
    archives = [os.path.join(args.input_dir, f) for f in tqdm(os.listdir(args.input_dir))
                if f.lower().endswith(('.7z', '.tar'))]
    print(f"Found {len(archives)} archives.")
    archives.sort()
    if not archives:
        print("[ERROR] No archives found in the specified input directory.")
        return

    print(f"Found {len(archives)} archives. Converting to HDF5...")

    if os.path.exists(args.output_h5):
        print(f"[WARNING] Output HDF5 file {args.output_h5} already exists.")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without overwriting the existing file.")
            return

    if args.threads <= 1:
        # Process sequentially.
        total_extraction_time = 0.0
        total_group_time = 0.0
        with h5py.File(args.output_h5, "w") as h5_file:
            for archive in tqdm(archives, desc="Processing archives"):
                extr_time, grp_time = process_instance_archive(archive, h5_file, args.group_prefix, args.compression, args.compression_level)
                total_extraction_time += extr_time
                total_group_time += grp_time
                tqdm.write(f"Processed {os.path.basename(archive)}: extr={extr_time:.2f}s, grp={grp_time:.2f}s")
        print(f"Conversion complete. HDF5 file saved as: {args.output_h5}")
        print(f"Total extraction time: {total_extraction_time:.2f} sec")
        print(f"Total group creation time: {total_group_time:.2f} sec")
    else:
        # Process in parallel using partial HDF5 files.
        n_threads = args.threads
        # Split archives evenly among threads.
        chunk_size = ceil(len(archives) / n_threads)
        archive_chunks = [archives[i:i+chunk_size] for i in range(0, len(archives), chunk_size)]
        partial_files = []
        partials_dir = setup_h5_partials_folder(args.output_h5)
        futures = []
        total_extraction_time = 0.0
        total_group_time = 0.0

        with ProcessPoolExecutor(max_workers=n_threads) as executor:
            for i, chunk in enumerate(archive_chunks):
                partial_filename = os.path.join(partials_dir, os.path.basename(f"{args.output_h5}.part{i}"))
                print(f"Thread {i} will write to: {partial_filename}")
                partial_files.append(partial_filename)
                futures.append(executor.submit(process_archives_chunk, chunk, partial_filename,
                                               args.group_prefix, args.compression, args.compression_level))
            for future in as_completed(futures):
                extr_time, grp_time = future.result()
                total_extraction_time += extr_time
                total_group_time += grp_time

        # Merge partial HDF5 files.
        merge_h5_files(partial_files, args.output_h5)
        # Remove partial files.
        for pf in partial_files:
            os.remove(pf)
        print(f"Conversion complete. Final HDF5 file saved as: {args.output_h5}")
        print(f"Total extraction time: {total_extraction_time:.2f} sec")
        print(f"Total group creation time: {total_group_time:.2f} sec")


if __name__ == "__main__":
    main()
