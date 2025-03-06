#!/usr/bin/env python3
"""
instances_to_h5.py

This tool converts produced instance archives (stored as 7z or tar files) into a single HDF5 file.
Each archive is expected to contain one or more PLY files (e.g. “surface_0.ply”, “surface_1.ply”, …)
and, optionally, matching metadata JSON files named like “metadata_surface_0.json” for each surface.

Usage:
    python instances_to_h5.py --input_dir /path/to/archives --output_h5 /path/to/output.h5
"""

import os
import argparse
import tempfile
import shutil
import json
import glob

import h5py
import numpy as np
import open3d as o3d
import py7zr
import tarfile
from tqdm import tqdm


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


def process_instance_archive(archive_path, h5_file, group_prefix="", compression="lzf"):
    """
    Extracts an instance archive, reads its PLY and metadata files, and writes them into the HDF5 file.
    
    Each archive becomes an HDF5 group (named after the archive filename). Inside that group, each
    surface is stored in a subgroup (named after the PLY file, without extension) with datasets:
      - "points"
      - "normals"
      - "colors"
    and with any metadata (e.g. score, distance, coefficients, etc.) stored as attributes.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        extract_archive(archive_path, temp_dir)
        # Search recursively for PLY files.
        ply_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.ply'):
                    ply_files.append(os.path.join(root, file))
                    
        if not ply_files:
            print(f"[WARNING] No .ply files found in archive {archive_path}.")
            return

        # Create a group in the HDF5 file for this archive.
        archive_basename = os.path.basename(archive_path)
        group_name = os.path.splitext(archive_basename)[0]
        if group_prefix:
            group_name = os.path.join(group_prefix, group_name)
        # Remove the group if it already exists.
        if group_name in h5_file:
            del h5_file[group_name]
        grp = h5_file.create_group(group_name)
        
        # Process each PLY file.
        for ply_file in ply_files:
            base = os.path.splitext(os.path.basename(ply_file))[0]
            # Expect a metadata file named like "metadata_<base>.json"
            metadata_file = os.path.join(os.path.dirname(ply_file), f"metadata_{base}.json")
            
            try:
                points, normals, colors = load_ply_file(ply_file)
            except Exception as e:
                print(f"[ERROR] Reading {ply_file}: {e}")
                continue

            metadata = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    print(f"[WARNING] Reading metadata file {metadata_file}: {e}")

            # Create a subgroup for this surface.
            surface_grp = grp.create_group(base)
            surface_grp.create_dataset("points", data=points, compression=compression)
            surface_grp.create_dataset("normals", data=normals, compression=compression)
            surface_grp.create_dataset("colors", data=colors, compression=compression)
            
            # Save metadata as attributes.
            for key, value in metadata.items():
                try:
                    surface_grp.attrs[key] = value
                except Exception as e:
                    print(f"[WARNING] Setting attribute {key} for {base}: {e}")

    finally:
        # Clean up the temporary extraction directory.
        shutil.rmtree(temp_dir)


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
    args = parser.parse_args()

    # Find all archive files (both .7z and .tar) in the input directory.
    archives = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                if f.lower().endswith(('.7z', '.tar'))]
    archives.sort()
    if not archives:
        print("[ERROR] No archives found in the specified input directory.")
        return

    print(f"Found {len(archives)} archives. Converting to HDF5...")
    
    # Open the HDF5 file for writing.
    with h5py.File(args.output_h5, "w") as h5_file:
        for archive in tqdm(archives, desc="Processing archives"):
            try:
                process_instance_archive(archive, h5_file, group_prefix=args.group_prefix, compression=args.compression)
            except Exception as e:
                print(f"[ERROR] Processing {archive}: {e}")

    print(f"Conversion complete. HDF5 file saved as: {args.output_h5}")


if __name__ == "__main__":
    main()
