import os
import zarr
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tifffile

def get_zarr_data(store, resolution):
    """Helper function to get zarr data handling both multi-resolution and single arrays."""
    try:
        # First try to access as multi-resolution
        if isinstance(store, zarr.Array):
            return store
        try:
            return store[str(resolution)]
        except (KeyError, ValueError, TypeError, IndexError):
            # If resolution access fails, try getting array directly
            for key in store.keys():
                if isinstance(store[key], zarr.Array):
                    return store[key]
            # If we get here, return the store itself as a last resort
            return store
    except Exception as e:
        raise ValueError(f"Could not access zarr data: {str(e)}")

def normalize_to_8bit(data):
    """Convert floating point data to 8-bit with proper scaling."""
    try:
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
    except Exception as e:
        print(f"Error in normalize_to_8bit: {str(e)}")
        raise


def get_required_blocks(csv_path, block_size, block_shape):
    """Read CSV and determine which blocks need to be processed."""
    try:
        df = pd.read_csv(csv_path)

        # Calculate number of blocks in each dimension from volume shape
        blocks_in_z = int(np.ceil(block_shape[0] / block_size))
        blocks_in_y = int(np.ceil(block_shape[1] / block_size))
        blocks_in_x = int(np.ceil(block_shape[2] / block_size))

        print(f"Total possible blocks: {blocks_in_z * blocks_in_y * blocks_in_x}")

        # The CSV indices are already in terms of blocks, so we use them directly
        # Subtract 1 because the CSV uses 1-based indexing but our processing uses 0-based
        df['block_y'] = df['jy'] - 1
        df['block_x'] = df['jx'] - 1
        df['block_z'] = df['jz'] - 1

        # Get unique block combinations
        unique_blocks = df[['block_z', 'block_y', 'block_x']].drop_duplicates().values

        print(f"Found {len(unique_blocks)} blocks containing points of interest")

        return unique_blocks

    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        raise


def process_block(args):
    bz, by, bx, zarr_path, resolution, output_directory, block_size, overlap, block_shape, debug = args
    try:
        os.makedirs(output_directory, exist_ok=True)

        block_filename = f"cell_yxz_{by + 1:03}_{bx + 1:03}_{bz + 1:03}.tif"
        block_path = os.path.join(output_directory, block_filename)

        if os.path.exists(block_path):
            if debug:
                print(f"'{block_filename}' already exists in the output directory. Skipping.")
            return

        store = zarr.open(zarr_path, mode='r')
        data = get_zarr_data(store, resolution)

        if debug:
            print(f"Processing block {block_filename}")
            print(f"Block shape: {block_shape}")

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

        if debug:
            print(f"Extraction region: z[{z_start}:{z_end}], y[{y_start}:{y_end}], x[{x_start}:{x_end}]")

        # Create block with padding - use uint8 as target dtype
        block = np.zeros((padded_size, padded_size, padded_size), dtype=np.uint8)

        # Calculate where in the padded block to place the data
        block_z_start = overlap - (bz * block_size - z_start)
        block_y_start = overlap - (by * block_size - y_start)
        block_x_start = overlap - (bx * block_size - x_start)

        # Extract data from zarr
        actual_data = data[z_start:z_end, y_start:y_end, x_start:x_end]

        if debug:
            print(f"Extracted data shape: {actual_data.shape}")
            print(f"Data type: {actual_data.dtype}")

        # Convert to 8-bit
        if actual_data.dtype != np.uint8:
            actual_data = normalize_to_8bit(actual_data)

        try:
            # Place data in padded block
            block[
            block_z_start:block_z_start + actual_data.shape[0],
            block_y_start:block_y_start + actual_data.shape[1],
            block_x_start:block_x_start + actual_data.shape[2]
            ] = actual_data
        except ValueError as e:
            print(f"\nDetailed error info for {block_filename}:")
            print(f"Block shape: {block.shape}")
            print(f"Data shape: {actual_data.shape}")
            print(f"Insertion position: ({block_z_start}:{block_z_start + actual_data.shape[0]}, "
                  f"{block_y_start}:{block_y_start + actual_data.shape[1]}, "
                  f"{block_x_start}:{block_x_start + actual_data.shape[2]})")
            raise e

        # Save as tiff
        tifffile.imwrite(block_path, block, compression='zlib')

        if debug:
            print(f"Successfully saved {block_filename}")

    except Exception as e:
        print(f"Error processing block {by + 1}_{bx + 1}_{bz + 1}: {str(e)}")
        raise


def get_blocks_from_directory(directory):
    """Get list of blocks from existing files in a directory."""
    try:
        import re
        pattern = re.compile(r"cell_yxz_(\d{3})_(\d{3})_(\d{3})\.tif")
        blocks = []

        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                # Convert back to 0-based indexing
                y, x, z = [int(i) - 1 for i in match.groups()]
                blocks.append([z, y, x])  # order is z,y,x to match current block order

        blocks = np.array(blocks)
        print(f"Found {len(blocks)} blocks in directory")
        return blocks

    except Exception as e:
        print(f"Error reading directory: {str(e)}")
        raise


def generate_grid_blocks(zarr_path, output_directory, resolution, block_size, overlap, num_threads, debug=False,
                         csv_path=None, template_dir=None):
    """
    Generate grid blocks either from CSV coordinates or by matching existing files.

    Parameters:
        csv_path: Path to CSV file with coordinates
        template_dir: Directory containing existing files to match
        (one of csv_path or template_dir should be provided)
    """
    if csv_path and template_dir:
        raise ValueError("Please provide either csv_path or template_dir, not both")
    if not csv_path and not template_dir:
        raise ValueError("Must provide either csv_path or template_dir")

    store = zarr.open(zarr_path, mode='r')
    data = get_zarr_data(store, resolution)
    block_shape = data.shape

    print(f"Input data type: {data.dtype}")
    print(f"Input shape: {block_shape}")

    # Get required blocks either from CSV or directory
    if csv_path:
        required_blocks = get_required_blocks(csv_path, block_size, block_shape)
        print(f"Number of blocks from CSV: {len(required_blocks)}")
    else:
        required_blocks = get_blocks_from_directory(template_dir)
        print(f"Number of blocks from template directory: {len(required_blocks)}")

    tasks = [
        (bz, by, bx, zarr_path, resolution, output_directory, block_size, overlap, block_shape, debug)
        for bz, by, bx in required_blocks
    ]

    with Pool(processes=num_threads) as pool:
        for _ in tqdm(pool.imap_unordered(process_block, tasks), total=len(tasks)):
            pass

    print('Grid blocks have been generated.')


def main():
    # Configuration
    zarr_path = "/mnt/raid_nvme/datasets/scroll1_predictions/gp_legendary-medial-cubes-softmax.zarr"
    output_directory = "/mnt/raid_nvme/datasets/scroll1_predictions/gp_legendary-medial-cubes-softmax_grids"
    resolution = 0
    block_size = 500
    overlap = 50
    num_threads = 8
    debug = False

    # Choose ONE of these:
    # csv_path = "/mnt/raid_nvme/scroll1.volpkg/scroll_1_54_mask.csv"  # Option 1: Use CSV
    template_dir = "/mnt/raid_nvme/datasets/scroll1_predictions/050_resencextraDAdistloss_sm_grids"  # Option 2: Use existing directory as template

    # Print configuration
    print("Configuration:")
    print(f"- Zarr path: {zarr_path}")
    #if csv_path:
     #   print(f"- CSV path: {csv_path}")
    if template_dir:
        print(f"- Template directory: {template_dir}")
    print(f"- Output directory: {output_directory}")
    print(f"- Resolution: {resolution}")
    print(f"- Block size: {block_size}")
    print(f"- Overlap: {overlap}")
    print(f"- Number of threads: {num_threads}")
    print(f"- Debug mode: {debug}")
    print("-" * 50)

    try:
        generate_grid_blocks(
            zarr_path=zarr_path,
            output_directory=output_directory,
            resolution=resolution,
            block_size=block_size,
            overlap=overlap,
            num_threads=num_threads,
            debug=debug,
            #csv_path=csv_path,
            template_dir=template_dir
        )
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()