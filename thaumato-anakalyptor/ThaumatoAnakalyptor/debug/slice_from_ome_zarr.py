#!/usr/bin/env python3

import sys
import numpy as np
import imageio.v2 as imageio
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

def main():
    if len(sys.argv) < 3:
        print("Usage: python save_z_slice.py <zarr_path> <z-index>")
        sys.exit(1)
    
    zarr_path = sys.argv[1]
    z_index = int(sys.argv[2])
    
    # Parse and open the OME-Zarr store
    store = parse_url(zarr_path)
    reader = Reader(store)
    nodes = list(reader())

    # For simplicity, assume the first node contains the image you want
    # and that we use scale 0 (highest resolution)
    # Typically, `node.data[0]` => scale 0
    node = nodes[0]
    data = node.data[0]  # This is a dask array

    # OME-NGFF standard often arranges axes as (C, Z, Y, X) for 3D data
    # If your data has a different shape (e.g. (Z, Y, X)), adjust accordingly.
    # Here we pick channel 0 and the requested z-index:
    slice_2d = data[z_index, :, :].compute()

    # Normalize to [0, 255] for 8-bit JPG
    min_val, max_val = slice_2d.min(), slice_2d.max()
    if min_val == max_val:
        # Avoid divide-by-zero if the slice is uniform
        img_8bit = np.zeros_like(slice_2d, dtype=np.uint8)
    else:
        img_8bit = ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Save as JPG
    output_filename = f"z_slice_{z_index}.jpg"
    imageio.imwrite(output_filename, img_8bit)
    print(f"Saved {output_filename}")

if __name__ == "__main__":
    main()
