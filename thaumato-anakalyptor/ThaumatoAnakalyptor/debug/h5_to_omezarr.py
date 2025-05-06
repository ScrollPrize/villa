#!/usr/bin/env python3
import argparse
import h5py
import zarr

# Recursively copy HDF5 groups/datasets into Zarr
def copy_h5_to_zarr(h5_group, z_group):
    for name, obj in h5_group.items():
        if isinstance(obj, h5py.Dataset):
            # Copy the dataset
            z_ds = z_group.create_dataset(
                name,
                data=obj[...],
                chunks=obj.chunks or True,
                compressor="blosc"
            )
            # Copy any HDF5 attributes
            for k, v in obj.attrs.items():
                z_ds.attrs[k] = v
        elif isinstance(obj, h5py.Group):
            # Make a new Zarr group and recurse
            sub_z = z_group.create_group(name)
            for k, v in obj.attrs.items():
                sub_z.attrs[k] = v
            copy_h5_to_zarr(obj, sub_z)

def main():
    p = argparse.ArgumentParser(
        description="Convert an HDF5 file into an OME-Zarr directory store"
    )
    p.add_argument("input_h5", help="Path to input .h5 file")
    p.add_argument("output_zarr", help="Path to output Zarr directory")
    args = p.parse_args()

    # Open source HDF5
    h5f = h5py.File(args.input_h5, "r")

    # Create/overwrite Zarr store
    store = zarr.DirectoryStore(args.output_zarr)
    zroot = zarr.group(store=store, overwrite=True)

    # Add minimal OME-NGFF multiscales metadata
    # Adjust axes, units, etc. as needed for your data!
    zroot.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "converted",
        "datasets": [{"path": "0"}],
        "axes": [
            {"name": "z", "type": "space", "unit": "µm"},
            {"name": "y", "type": "space", "unit": "µm"},
            {"name": "x", "type": "space", "unit": "µm"},
        ],
    }]
    zroot.attrs["@type"] = ["multiscales", "default"]

    # Copy everything into the first (and only) level
    copy_h5_to_zarr(h5f, zroot.require_group("0"))

    h5f.close()
    print(f"✔️  Converted {args.input_h5} → {args.output_zarr}")

if __name__ == "__main__":
    main()
