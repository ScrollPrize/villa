#!/usr/bin/env python3
"""
Script to rebase OME-Zarr pyramid layers by removing initial layers and shifting indices.

Usage:
  python rebase_ome_zarr_pyramid.py <input_dir> <output_dir> <displacement>
"""
import argparse
import os
import shutil
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description='Rebase OME-Zarr pyramid layers by removing initial levels and shifting indices.')
    parser.add_argument(
        'input', help='Path to input OME-Zarr directory')
    parser.add_argument(
        'output', help='Path to output OME-Zarr directory')
    parser.add_argument(
        'displacement', type=int,
        help='Pyramid layer displacement: layer that will become new 0')
    return parser.parse_args()

def main():
    args = parse_args()
    in_dir = args.input
    out_dir = args.output
    displacement = args.displacement

    if not os.path.isdir(in_dir):
        print(f"Error: input directory '{in_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)
    if os.path.exists(out_dir):
        print(f"Error: output directory '{out_dir}' already exists. Please remove it or choose another path.", file=sys.stderr)
        sys.exit(1)

    # Copy entire input tree to output
    try:
        shutil.copytree(in_dir, out_dir)
    except Exception as e:
        print(f"Error copying input to output: {e}", file=sys.stderr)
        sys.exit(1)

    # Identify and remove layers below displacement
    entries = os.listdir(out_dir)
    layer_dirs = [e for e in entries if e.isdigit() and os.path.isdir(os.path.join(out_dir, e))]
    for layer in layer_dirs:
        idx = int(layer)
        if idx < displacement:
            shutil.rmtree(os.path.join(out_dir, layer))

    # Rename remaining layers by shifting indices
    entries = os.listdir(out_dir)
    layer_dirs = sorted(
        [e for e in entries if e.isdigit() and os.path.isdir(os.path.join(out_dir, e))],
        key=lambda x: int(x)
    )
    for layer in layer_dirs:
        old_idx = int(layer)
        new_idx = old_idx - displacement
        if new_idx < 0:
            continue
        old_path = os.path.join(out_dir, layer)
        new_path = os.path.join(out_dir, str(new_idx))
        os.rename(old_path, new_path)

    # Update multiscales metadata in .zattrs
    zattrs_path = os.path.join(out_dir, '.zattrs')
    if not os.path.isfile(zattrs_path):
        print(f"Error: .zattrs file not found in '{out_dir}'. Is this a valid OME-Zarr?", file=sys.stderr)
        sys.exit(1)
    with open(zattrs_path, 'r') as f:
        attrs = json.load(f)
    ms_list = attrs.get('multiscales')
    if not ms_list or not isinstance(ms_list, list):
        print("Error: no 'multiscales' metadata found in .zattrs.", file=sys.stderr)
        sys.exit(1)
    ms = ms_list[0]
    datasets = ms.get('datasets')
    if not datasets or not isinstance(datasets, list):
        print("Error: no 'datasets' in multiscales metadata.", file=sys.stderr)
        sys.exit(1)
    if displacement < 0 or displacement >= len(datasets):
        print(f"Error: displacement {displacement} out of range (0..{len(datasets)-1}).", file=sys.stderr)
        sys.exit(1)
    new_datasets = []
    for new_idx, ds in enumerate(datasets[displacement:]):
        entry = dict(ds)
        entry['path'] = str(new_idx)
        new_datasets.append(entry)
    ms['datasets'] = new_datasets
    attrs['multiscales'][0] = ms
    try:
        with open(zattrs_path, 'w') as f:
            json.dump(attrs, f, indent=2)
    except Exception as e:
        print(f"Error writing updated .zattrs: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Rebased OME-Zarr pyramid with displacement {displacement}. Output at '{out_dir}'.")

if __name__ == '__main__':
    main()