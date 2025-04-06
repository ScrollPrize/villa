import os
import zarr
import numpy as np
import tifffile
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial


def parse_filename(filename):
    # Parse scroll_zstart_ystart_xstart_size from filename
    parts = os.path.basename(filename).split('_')
    return {
        'scroll': parts[0],
        'z': int(parts[1]),
        'y': int(parts[2]),
        'x': int(parts[3]),
        'size': int(parts[4])
    }


def process_tif_pair(args):
    image_path, label_path, image_zarr, label_zarr = args

    # Parse coordinates from filename
    coords = parse_filename(image_path)

    # Read data
    image = tifffile.imread(image_path)
    label = tifffile.imread(label_path)

    # Write to zarr at correct position
    z_slice = slice(coords['z'], coords['z'] + coords['size'])
    y_slice = slice(coords['y'], coords['y'] + coords['size'])
    x_slice = slice(coords['x'], coords['x'] + coords['size'])

    image_zarr[z_slice, y_slice, x_slice] = image
    label_zarr[z_slice, y_slice, x_slice] = label

    return f"Processed chunk at z={coords['z']}, y={coords['y']}, x={coords['x']}"


def main():
    # Paths
    base_dir = "/home/sean/Downloads/Dataset001_sk-fibers-20250124"  # Update this
    images_dir = os.path.join(base_dir, "imagesTr")
    labels_dir = os.path.join(base_dir, "labelsTr")

    # Get all image files (excluding _std files)
    image_files = sorted([f for f in glob(os.path.join(images_dir, "*.tif"))
                          if "_std" not in f])
    label_files = sorted([f for f in glob(os.path.join(labels_dir, "*.tif"))
                          if "_std" not in f])

    # Get volume dimensions from filenames
    coords_list = [parse_filename(f) for f in image_files]
    max_z = max(c['z'] + c['size'] for c in coords_list)
    max_y = max(c['y'] + c['size'] for c in coords_list)
    max_x = max(c['x'] + c['size'] for c in coords_list)
    chunk_size = coords_list[0]['size']

    # Create zarr arrays
    store_images = zarr.DirectoryStore('output_images.zarr')
    store_labels = zarr.DirectoryStore('output_labels.zarr')

    image_zarr = zarr.create(shape=(max_z, max_y, max_x),
                             chunks=(chunk_size, chunk_size, chunk_size),
                             dtype=np.uint8,
                             store=store_images,
                             fill_value=0,
                             write_empty_chunks=False)

    label_zarr = zarr.create(shape=(max_z, max_y, max_x),
                             chunks=(chunk_size, chunk_size, chunk_size),
                             dtype=np.uint8,
                             store=store_labels,
                             fill_value=0,
                             write_empty_chunks=False
                             )

    # Prepare arguments for parallel processing
    process_args = list(zip(image_files, label_files,
                            [image_zarr] * len(image_files),
                            [label_zarr] * len(image_files)))

    # Process in parallel with progress bar
    with Pool() as pool:
        list(tqdm(pool.imap(process_tif_pair, process_args),
                  total=len(process_args),
                  desc="Processing chunks"))


if __name__ == "__main__":
    main()