import zarr
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def _read_chunk(
    input_zarr_path,
    z_global_start,
    y_global_start,
    x_global_start,
    chunk_coords
):
    """
    Worker function to read a chunk of data from input_zarr_path.
    Returns (out_z_slice, out_y_slice, out_x_slice, data_chunk).
    """
    (z, z_end, y, y_end, x, x_end) = chunk_coords

    # Open array directly (read-only)
    arr = zarr.open(input_zarr_path, mode='r')

    # Read the chunk
    data_chunk = arr[z:z_end, y:y_end, x:x_end]

    # Calculate output (local) slicing, i.e. offsets in the sub-volume
    out_z_start = z - z_global_start
    out_z_end   = out_z_start + (z_end - z)
    out_y_start = y - y_global_start
    out_y_end   = out_y_start + (y_end - y)
    out_x_start = x - x_global_start
    out_x_end   = out_x_start + (x_end - x)

    return (out_z_start, out_z_end,
            out_y_start, out_y_end,
            out_x_start, out_x_end,
            data_chunk)

def cut_zarr_bounding_box_chunked(
    input_zarr_path,
    output_zarr_path,
    z_start=0, z_stop=100,
    y_start=0, y_stop=100,
    x_start=0, x_stop=100,
    compressor=None,
    chunks=None,
    n_workers=1
):
    """
    Cuts a bounding box from a root-level Zarr array and writes it chunk-by-chunk
    to a new Zarr store, tracking progress with TQDM and optionally using
    multiprocessing to speed up chunk reading.

    Parameters
    ----------
    input_zarr_path : str
        Path to the input Zarr store.
    output_zarr_path : str
        Path to the output Zarr store (will be overwritten if it exists).
    z_start, z_stop : int
        Start and stop indices for the z dimension.
    y_start, y_stop : int
        Start and stop indices for the y dimension.
    x_start, x_stop : int
        Start and stop indices for the x dimension.
    compressor : numcodecs.abc.Codec or None
        Compressor to use when creating the new Zarr array. If None,
        defaults to the input array's compressor.
    chunks : tuple of int or None
        Chunk size for the new array. If None, defaults to the
        input array's chunk sizes.
    n_workers : int
        Number of parallel worker processes for reading chunks.
    """
    # -- 1. Open the input array (read-only)
    input_array = zarr.open(input_zarr_path, mode='r')

    # -- 2. Determine the shape of the sub-volume
    sub_z = z_stop - z_start
    sub_y = y_stop - y_start
    sub_x = x_stop - x_start
    sub_shape = (sub_z, sub_y, sub_x)
    print(f"Output shape will be {sub_shape}")

    # -- 3. Determine chunking and compressor
    if compressor is None:
        compressor = input_array.compressor
    if chunks is None:
        chunks = input_array.chunks  # preserve original chunk sizes
    chunk_z, chunk_y, chunk_x = chunks  # unpack for 3D
    print(f"Using chunks={chunks} and compressor={compressor}")

    # -- 4. Create (or overwrite) the output array
    output_array = zarr.open(
        output_zarr_path,
        mode='w',
        shape=sub_shape,
        chunks=chunks,
        dtype=input_array.dtype,
        compressor=compressor
    )

    # -- 5. Build a list of chunk coordinates
    print("Building chunk coordinates...")
    chunk_coords_list = []
    for z in range(z_start, z_stop, chunk_z):
        z_end = min(z_stop, z + chunk_z)
        for y in range(y_start, y_stop, chunk_y):
            y_end = min(y_stop, y + chunk_y)
            for x in range(x_start, x_stop, chunk_x):
                x_end = min(x_stop, x + chunk_x)
                chunk_coords_list.append((z, z_end, y, y_end, x, x_end))

    # -- 6. Set up multiprocessing for reading
    read_func = partial(
        _read_chunk,
        input_zarr_path,
        z_start,
        y_start,
        x_start
    )

    if n_workers > 1:
        pool = mp.Pool(n_workers)
        results_iter = pool.imap(read_func, chunk_coords_list)
    else:
        results_iter = (read_func(cc) for cc in chunk_coords_list)

    # -- 7. Read and write chunks
    print("Processing chunks...")
    for (out_z_start, out_z_end,
         out_y_start, out_y_end,
         out_x_start, out_x_end,
         data_chunk) in tqdm(results_iter, total=len(chunk_coords_list)):

        output_array[out_z_start:out_z_end,
                    out_y_start:out_y_end,
                    out_x_start:out_x_end] = data_chunk

    # Close the pool if we created one
    if n_workers > 1:
        pool.close()
        pool.join()

    print(f"Successfully wrote bounding box "
          f"[z:{z_start}:{z_stop}, y:{y_start}:{y_stop}, x:{x_start}:{x_stop}] "
          f"from '{input_zarr_path}' to '{output_zarr_path}'")


if __name__ == "__main__":
    # Example usage
    input_zarr = "/mnt/raid_nvme/s1.zarr"
    output_zarr = "/mnt/raid_nvme/s1_5000_8000.zarr"

    cut_zarr_bounding_box_chunked(
        input_zarr_path=input_zarr,
        output_zarr_path=output_zarr,
        z_start=5000, z_stop=8000,
        y_start=0,  y_stop=7888,
        x_start=0,  x_stop=8096,
        n_workers=16
    )