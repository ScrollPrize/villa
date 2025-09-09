import os
import zarr
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tifffile


def normalize_to_8bit(data):
    """Convert floating point data to 8-bit with proper scaling."""
    # Handle potential NaN or inf values
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)

    # Get the actual min and max
    data_min = data.min()
    data_max = data.max()

    # Avoid division by zero
    if data_max == data_min:
        return np.zeros_like(data, dtype=np.uint8)

    # Scale to 0-255
    scaled = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    return scaled

def process_block(args):
    bz, by, bx, zarr_path, resolution, output_directory, block_size, overlap, block_shape = args
    os.makedirs(output_directory, exist_ok=True)
    
    block_filename = f"cell_yxz_{by+1:03}_{bx+1:03}_{bz+1:03}.tif"
    block_path = os.path.join(output_directory, block_filename)

    if os.path.exists(block_path):
        print(f"'{block_filename}' already exists in the output directory. Skipping.")
        return

    # Open zarr array at specified resolution
    store = zarr.open(zarr_path, mode='r')
    data = store[str(resolution)]
    
    # Calculate padded block size
    padded_size = block_size + 2 * overlap
    
    # Calculate start positions with overlap
    z_start = max(0, bz * block_size - overlap)
    y_start = max(0, by * block_size - overlap)
    x_start = max(0, bx * block_size - overlap)

    # Calculate end positions with overlap
    z_end = min(block_shape[0], (bz + 1) * block_size + overlap)
    y_end = min(block_shape[1], (by + 1) * block_size + overlap)
    x_end = min(block_shape[2], (bx + 1) * block_size + overlap)

    # Create block with padding - use uint8 as target dtype
    block = np.zeros((padded_size, padded_size, padded_size), dtype=np.uint8)

    # Calculate where in the padded block to place the data
    block_z_start = overlap - (bz * block_size - z_start)
    block_y_start = overlap - (by * block_size - y_start)
    block_x_start = overlap - (bx * block_size - x_start)

    # Extract data from zarr
    actual_data = data[z_start:z_end, y_start:y_end, x_start:x_end]

    # Convert to 8-bit
    if actual_data.dtype != np.uint8:
        actual_data = normalize_to_8bit(actual_data)
    
    # Place data in padded block
    block[
        block_z_start:block_z_start + actual_data.shape[0],
        block_y_start:block_y_start + actual_data.shape[1],
        block_x_start:block_x_start + actual_data.shape[2]
    ] = actual_data

    # Save as tiff
    tifffile.imwrite(block_path, block, compression='zlib')
    #tifffile.imwrite(block_path, block)

def generate_grid_blocks(zarr_path, output_directory, resolution, block_size, overlap, num_threads):
    # Open zarr to get dimensions and check data type
    store = zarr.open(zarr_path, mode='r')
    data = store[str(resolution)]
    block_shape = data.shape
    
    print(f"Input data type: {data.dtype}")
    print(f"Input shape: {block_shape}")
    
    # Calculate number of blocks needed
    blocks_in_z = int(np.ceil(block_shape[0] / block_size))
    blocks_in_y = int(np.ceil(block_shape[1] / block_size))
    blocks_in_x = int(np.ceil(block_shape[2] / block_size))
    
    print(f"Will generate {blocks_in_z * blocks_in_y * blocks_in_x} blocks")
    print(f"Using {num_threads} processes")

    tasks = [
        (bz, by, bx, zarr_path, resolution, output_directory, block_size, overlap, block_shape)
        for bz in range(blocks_in_z)
        for by in range(blocks_in_y)
        for bx in range(blocks_in_x)
    ]

    with Pool(processes=num_threads) as pool:
        for _ in tqdm(pool.imap_unordered(process_block, tasks), total=len(tasks)):
            pass

    print('Grid blocks have been generated.')

def main():
    zarr_path = "/mnt/raid_nvme/datasets/scroll1_predictions/gp_legendary-medial-cubes-softmax_ome.zarr" # Path to your OME-Zarr dataset
    output_directory = "/mnt/raid_nvme/datasets/scroll1_predictions/gp_legendary-medial-cubes-softmax_grids"  # Path where you want to save the grid blocks
    resolution = 0  # Specify which resolution level to use (0,1,2,3,4,5)
    block_size = 500  # Define your block size here
    overlap = 50  # Overlap size in pixels
    num_threads = 16
    generate_grid_blocks(zarr_path, output_directory, resolution, block_size, overlap, num_threads)

if __name__ == '__main__':
    main()