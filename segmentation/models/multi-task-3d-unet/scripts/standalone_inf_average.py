import os
import zarr
import numpy as np
from numcodecs import Blosc
from tqdm import tqdm

def finalize_predictions(store_path: str,
                        targets_config: dict,
                        chunk_size_z: int = 128,
                        chunk_size_y: int = 128,
                        chunk_size_x: int = 128,
                        do_average: bool = True):
    """
    1) Optionally averages 'xxx_sum' by 'xxx_count' if do_average=True,
       otherwise leaves the raw sum.
    2) Scales/casts to final dtype.

    Args:
        store_path (str): Path to the Zarr store containing *_sum and *_count datasets.
        targets_config (dict): Dictionary of {target_name: { ... }} for each output channel.
        chunk_size_z, chunk_size_y, chunk_size_x (int): Tiling sizes used for reading/writing in chunks.
        do_average (bool): If True, divide sum by count. If False, leave it as sum.
    """

    # Open the Zarr store in read/write mode
    zarr_store = zarr.open(store_path, mode='r+')
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

    for tgt_name, tgt_info in targets_config.items():
        sum_name = f"{tgt_name}_sum"      # e.g. "sheet_sum"
        count_name = f"{tgt_name}_count"  # e.g. "sheet_count"

        if sum_name not in zarr_store or count_name not in zarr_store:
            print(f"Skipping '{tgt_name}' because {sum_name} or {count_name} is missing.")
            continue

        sum_ds = zarr_store[sum_name]    # float32 sums
        cnt_ds = zarr_store[count_name]  # float32 counts

        # 1) AVERAGE PASS (optional)
        if do_average:
            print(f"Averaging overlaps for '{tgt_name}'...")
        else:
            print(f"Skipping division (keeping sums) for '{tgt_name}'...")

        z = sum_ds.shape[-3]
        y = sum_ds.shape[-2]
        x = sum_ds.shape[-1]

        # If do_average=True, we do sum / count in chunks
        if do_average:
            for z0 in tqdm(range(0, z, chunk_size_z), desc=f"{tgt_name} avg Z"):
                z1 = min(z0 + chunk_size_z, z)
                for y0 in range(0, y, chunk_size_y):
                    y1 = min(y0 + chunk_size_y, y)
                    for x0 in range(0, x, chunk_size_x):
                        x1 = min(x0 + chunk_size_x, x)

                        sum_block = sum_ds[..., z0:z1, y0:y1, x0:x1]
                        cnt_block = cnt_ds[z0:z1, y0:y1, x0:x1]

                        # Avoid divide-by-zero
                        mask = (cnt_block > 0)
                        sum_block[..., mask] /= cnt_block[mask]

                        sum_ds[..., z0:z1, y0:y1, x0:x1] = sum_block

        # 2) CREATE FINAL DATASET & CAST
        final_name = f"{tgt_name}_final"
        if final_name in zarr_store:
            # If there's already a final dataset, remove or overwrite it
            del zarr_store[final_name]

        # Decide the default dtype: normals => uint16, else => uint8
        final_dtype = 'uint16' if tgt_name.lower() == "normals" else 'uint8'

        final_ds = zarr_store.create_dataset(
            name=final_name,
            shape=sum_ds.shape,
            chunks=sum_ds.chunks,
            dtype=final_dtype,
            compressor=compressor,
            fill_value=0
        )

        # 3) SCALE & CAST
        print(f"Scaling and casting '{tgt_name}' to int...")
        chunk_size_z = sum_ds.chunks[-3]
        chunk_size_y = sum_ds.chunks[-2]
        chunk_size_x = sum_ds.chunks[-1]

        for z0 in tqdm(range(0, z, chunk_size_z), desc=f"{tgt_name} cast Z"):
            z1 = min(z0 + chunk_size_z, z)
            for y0 in range(0, y, chunk_size_y):
                y1 = min(y0 + chunk_size_y, y)
                for x0 in range(0, x, chunk_size_x):
                    x1 = min(x0 + chunk_size_x, x)

                    float_block = sum_ds[..., z0:z1, y0:y1, x0:x1]

                    # Example logic:
                    #  - "normals" => [-1..1] => [0..65000]
                    #  - everything else => [0..1] => [0..255]
                    if tgt_name.lower() == "normals":
                        int_block = (float_block + 1.0) / 2.0  # [-1..1] -> [0..1]
                        int_block *= 65000.0
                        np.clip(int_block, 0, 65000, out=int_block)
                        int_block = int_block.astype(np.uint16)
                    else:
                        int_block = float_block * 255.0
                        np.clip(int_block, 0, 255, out=int_block)
                        int_block = int_block.astype(np.uint8)

                    final_ds[..., z0:z1, y0:y1, x0:x1] = int_block

        # Optionally delete the sum / count arrays
        # del zarr_store[sum_name]
        # del zarr_store[count_name]

    print("Done!")


if __name__ == "__main__":
    store_path = "/mnt/raid_nvme/inference_out/predictions.zarr"

    # Minimal dictionary describing your tasks
    # (so we know which ones to handle)
    targets_config = {
        "sheet":   {"channels": 1},
        "normals": {"channels": 3},
        # add more as needed...
    }

    # Example usage:
    # finalize_predictions(store_path, targets_config, do_average=True)
    # finalize_predictions(store_path, targets_config, do_average=False)

    finalize_predictions(store_path, targets_config, do_average=False)
