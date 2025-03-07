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

import threading
import queue

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
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(path=extract_dir)
    else:
        raise ValueError(f"Unsupported archive format for {archive_path}")

def process_instance_archive(archive_path, h5_file, group_prefix="", compression="lzf", compression_level=4):
    """
    Extracts an instance archive, reads its PLY and metadata files, and writes them into the HDF5 file.
    
    Each archive becomes an HDF5 group (named after the archive filename). Inside that group, each
    surface is stored in a subgroup (named after the PLY file, without extension) with datasets:
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
    
    Returns a tuple (total_extraction_time, total_group_time, archives_processed).
    """
    total_extraction_time = 0.0
    total_group_time = 0.0
    with h5py.File(partial_h5_filename, "w") as h5_file:
        for archive in archives:
            extr_time, grp_time = process_instance_archive(archive, h5_file, group_prefix, compression, compression_level=compression_level)
            total_extraction_time += extr_time
            total_group_time += grp_time
    return total_extraction_time, total_group_time, len(archives)

def merge_worker(merge_queue, final_filename, merge_pbar):
    """
    Continuously monitors the merge_queue for completed partial files.
    For each partial file, counts its groups, updates the merge progress bar total,
    merges the groups into the final HDF5 file (opened in append mode), and deletes the partial file.
    A None value in the queue signals that no more partial files are coming.
    """
    while True:
        pf = merge_queue.get()
        if pf is None:
            merge_queue.task_done()
            break
        try:
            # Count the number of groups in the partial file.
            with h5py.File(pf, "r") as h5_part:
                groups = list(h5_part.keys())
            num_groups = len(groups)
            # Increase the total of the merge progress bar.
            merge_pbar.total += num_groups
            merge_pbar.refresh()
            # Open the final file in append mode and copy groups.
            with h5py.File(final_filename, "a") as h5_final:
                with h5py.File(pf, "r") as h5_part:
                    for group in groups:
                        h5_part.copy(group, h5_final, name=group)
                        merge_pbar.update(1)
            # Delete the partial file after merging.
            os.remove(pf)
        except Exception as e:
            print(f"[ERROR] Merging partial file {pf}: {e}")
        finally:
            merge_queue.task_done()

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
                        help="Number of threads (processes) to use. If >1, partial HDF5 files are created and merged on the fly.")
    parser.add_argument("--compression_level", type=int, default=4,
                        help="Compression level (0-9, only for gzip, where 0 is no compression and 9 is max compression).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print more detailed information during processing.")
    args = parser.parse_args()

    print(f"Converting instance archives in {args.input_dir} to HDF5 file: {args.output_h5}")

    # Find all archive files.
    archives = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                if f.lower().endswith(('.7z', '.tar'))]
    print(f"Found {len(archives)} archives.")
    archives.sort()
    if not archives:
        print("[ERROR] No archives found in the specified input directory.")
        return

    if os.path.exists(args.output_h5):
        while True:
            print(f"[WARNING] Output HDF5 file {args.output_h5} already exists.")
            response = input("Do you want to overwrite(y) or remove(r) it? (y/r/n): ")
            if response.lower() == 'r':
                print(f"Removing existing file {args.output_h5}.")
                os.remove(args.output_h5)
                break
            elif response.lower() == 'y':
                print(f"Overwriting existing file {args.output_h5}.")
                break
            elif response.lower() == 'n':
                print("Exiting without overwriting the existing file.")
                return
            else:
                print("Invalid response. Please enter 'y', 'r', or 'n'.")
                

    if args.threads <= 1:
        # Process sequentially.
        total_extraction_time = 0.0
        total_group_time = 0.0
        with h5py.File(args.output_h5, "w") as h5_file:
            for archive in tqdm(archives, desc="Processing archives"):
                extr_time, grp_time = process_instance_archive(archive, h5_file, args.group_prefix, args.compression, args.compression_level)
                total_extraction_time += extr_time
                total_group_time += grp_time
                if args.verbose:
                    tqdm.write(f"Processed {os.path.basename(archive)}: extr={extr_time:.2f}s, grp={grp_time:.2f}s")
        print(f"Conversion complete. HDF5 file saved as: {args.output_h5}")
        print(f"Total extraction time: {total_extraction_time:.2f} sec")
        print(f"Total group creation time: {total_group_time:.2f} sec")
    else:
        # Process in parallel with many small chunks and on-the-fly merging.
        n_threads = args.threads
        chunk_size = ceil(len(archives) / (n_threads * 2))  # use more, smaller chunks
        chunk_size = min(chunk_size, 100)  # limit chunk size to avoid too large files
        chunk_size = max(chunk_size, 5)  # ensure at least 5 archives per chunk
        archive_chunks = [archives[i:i+chunk_size] for i in range(0, len(archives), chunk_size)]
        print(f"Using {n_threads} threads to process {len(archives)} archives in {len(archive_chunks)} chunks.")
        partials_dir = setup_h5_partials_folder(args.output_h5)

        # Prepare a queue and start the merge worker thread.
        merge_queue = queue.Queue()
        merge_pbar = tqdm(desc="Merging partial HDF5 files", total=0)
        merge_thread = threading.Thread(target=merge_worker, args=(merge_queue, args.output_h5, merge_pbar))
        merge_thread.start()

        total_extraction_time = 0.0
        total_group_time = 0.0
        start_time = time.perf_counter()

        future_to_pf = {}
        with ProcessPoolExecutor(max_workers=n_threads) as executor:
            for i, chunk in enumerate(archive_chunks):
                partial_filename = os.path.join(partials_dir, f"{os.path.basename(args.output_h5)}.part{i}")
                future = executor.submit(process_archives_chunk, chunk, partial_filename,
                                           args.group_prefix, args.compression, args.compression_level)
                future_to_pf[future] = partial_filename

            pbar = tqdm(total=len(archives), desc="Processing archives")
            for future in as_completed(future_to_pf):
                extr_time, grp_time, count = future.result()
                total_extraction_time += extr_time
                total_group_time += grp_time
                pbar.update(count)
                # As soon as a chunk finishes, push its partial file for merging.
                merge_queue.put(future_to_pf[future])
            pbar.close()

        # Wait until all partial files have been processed by the merge worker.
        merge_queue.join()
        # Signal the merge worker to exit.
        merge_queue.put(None)
        merge_thread.join()
        merge_pbar.close()

        total_time = time.perf_counter() - start_time
        print(f"Conversion complete. Final HDF5 file saved as: {args.output_h5}")
        print(f"Total extraction time: {total_extraction_time:.2f} sec")
        print(f"Total group creation time: {total_group_time:.2f} sec")
        print(f"Total processing time: {total_time:.2f} sec")

if __name__ == "__main__":
    main()
