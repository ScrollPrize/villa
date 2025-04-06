import os
import argparse
import zarr
import numpy as np
import cv2
import imageio
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Slice a normals_final dataset (3,z,y,x) from Zarr into image files.")
    parser.add_argument("--zarr_path", type=str, required=True,
                        help="Path to the Zarr store.")
    parser.add_argument("--dataset_name", type=str, default="normals_final",
                        help="Name of the dataset in the Zarr store.")
    parser.add_argument("--output_dir", type=str, default="./normals_slices",
                        help="Directory for the slices.")
    parser.add_argument("--use_16bit", action="store_true",
                        help="If set, writes 16-bit images (PNG/TIFF) instead of 8-bit JPEG.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Open Zarr store
    zarr_store = zarr.open(args.zarr_path, mode="r")
    if args.dataset_name not in zarr_store:
        raise ValueError(f"Dataset '{args.dataset_name}' not found in Zarr store '{args.zarr_path}'.")

    normals_data = zarr_store[args.dataset_name]  # expected shape: (3, z, y, x)
    if normals_data.shape[0] != 3:
        raise ValueError("Expected normals dataset shape (3,z,y,x).")

    _, z_dim, _, _ = normals_data.shape
    print(f"Found dataset '{args.dataset_name}' with shape {normals_data.shape}.")

    for z in tqdm(range(z_dim), desc="Writing slices"):
        # Extract slice (3, y, x)
        slice_data = normals_data[:, z, :, :]

        # Transpose to (y, x, 3)
        slice_data = np.transpose(slice_data, (1, 2, 0))

        if args.use_16bit:
            # Save 16-bit PNG (or TIFF) for full range
            out_name = os.path.join(args.output_dir, f"normals_z{z:04d}.png")

            # Make sure we have uint16
            if slice_data.dtype != np.uint16:
                # If it's float32 or something else, convert accordingly
                # e.g., if the data is in [-1..1], map to [0..65535], etc.
                slice_data_16bit = np.clip(slice_data, 0, 65535).astype(np.uint16)
            else:
                slice_data_16bit = slice_data

            imageio.imwrite(out_name, slice_data_16bit)
        else:
            # Save 8-bit JPEG for easy viewing
            out_name = os.path.join(args.output_dir, f"normals_z{z:04d}.jpg")

            # If data is [0..65535], scale to [0..255]
            if slice_data.dtype == np.uint16:
                slice_data_8bit = (slice_data / 256).astype(np.uint8)
            else:
                # If you already have float in [0..1], multiply by 255
                # Or if you trust the values are already in [0..255], just cast
                slice_data_8bit = slice_data.astype(np.uint8)

            cv2.imwrite(out_name, slice_data_8bit)

    print(f"Done! Images saved in {args.output_dir}.")

if __name__ == "__main__":
    main()
