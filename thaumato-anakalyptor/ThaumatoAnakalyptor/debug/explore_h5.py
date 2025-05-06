#!/usr/bin/env python3
import argparse
import h5py

def print_h5_structure(name, obj):
    """Callback for h5py.File.visititems to print groups & datasets."""
    if isinstance(obj, h5py.Group):
        # name is '' for the root group
        grp_name = name if name else '/'
        print(f"Group:   {grp_name}")
        for k, v in obj.attrs.items():
            print(f"    ↳ attr: {k} = {v!r}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        print(f"    ↳ shape: {obj.shape}, dtype: {obj.dtype}")
        for k, v in obj.attrs.items():
            print(f"    ↳ attr:  {k} = {v!r}")

def main():
    p = argparse.ArgumentParser(
        description="Inspect an HDF5 file: list groups, datasets, shapes, dtypes, and attributes."
    )
    p.add_argument("h5_file", help="Path to input .h5/.hdf5 file")
    args = p.parse_args()

    with h5py.File(args.h5_file, "r") as h5f:
        # Print any root-level attributes
        if h5f.attrs:
            print("Root attributes:")
            for k, v in h5f.attrs.items():
                print(f"  {k} = {v!r}")
            print()

        # Walk and print everything under the root
        h5f.visititems(print_h5_structure)

if __name__ == "__main__":
    main()
