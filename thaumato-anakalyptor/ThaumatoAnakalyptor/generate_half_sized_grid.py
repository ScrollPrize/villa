### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import os
import shutil
from tqdm import tqdm
import tifffile
import numpy as np
from skimage.transform import resize
from multiprocessing import Pool, cpu_count


def downsample_folder_tifs_singlethreaded(input_directory, output_directory, downsample_factor=2):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through all files in the input directory
    files = list(os.listdir(input_directory))

    #Filter all even {i}.tifs ie downscale in z direction
    files = [f for f in files if f.endswith('.tif') and int(f.split('.')[0]) % 2 == 0]
    print(f"Found {len(files)} even tif files to downsample.")

    # check that all even tif from min to max are present
    min_file = min([int(f.split('.')[0]) for f in files])
    max_file = max([int(f.split('.')[0]) for f in files])
    for i in range(min_file, max_file + 1):
        if i % 2 == 0:
            if (f"{i:04}.tif" not in files) and (f"{i:05}.tif" not in files):
                print(f"Missing {i:05}.tif")
                raise Exception("Missing tif files")

    for filename in tqdm(files):
        if filename.endswith('.tif'):
            filepath = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            # Check if image already exists in the output directory
            if os.path.exists(output_path):
                continue
            with tifffile.TiffFile(filepath) as tif:
                image = tif.asarray()
                # Check if the image is not empty
                if image.size == 0:
                    print(f"Warning: '{filename}' is empty.")
                    continue
                # Downsample the image using the specified factor
                downsampled_image = resize(image, 
                                        (image.shape[0] // downsample_factor, image.shape[1] // downsample_factor),
                                        anti_aliasing=False,
                                        preserve_range=True).astype(image.dtype)

                tifffile.imwrite(output_path, downsampled_image)

    print('Downsampling complete.')

def downsample_image(args):
    input_directory, output_directory, filename, downsample_factor = args
    filepath = os.path.join(input_directory, filename)
    output_path = os.path.join(output_directory, filename)

    # Check if image already exists in the output directory
    if os.path.exists(output_path):
        return f"'{filename}' already exists in the output directory. Skipping."
    # print(f"Downsampling {filename}.")
    
    with tifffile.TiffFile(filepath) as tif:
        image = tif.asarray()
        # Check if the image is not empty
        if image.size == 0:
            return f"Warning: '{filename}' is empty. Skipping."
        # Downsample the image using the specified factor
        downsampled_image = resize(image, 
                                   (image.shape[0] // downsample_factor, image.shape[1] // downsample_factor),
                                   anti_aliasing=False,
                                   preserve_range=True).astype(image.dtype)

        # Save the downsampled image
        tifffile.imwrite(output_path, downsampled_image)
    return f"Downsampled and saved '{filename}'."

def downsample_folder_tifs(input_directory, output_directory, downsample_factor=2,  num_threads=1):
    if downsample_factor == 1:
        print("Downsample factor is 1, skipping.")
        return
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = [f for f in os.listdir(input_directory) if f.endswith('.tif') and int(f.split('.')[0]) % downsample_factor == 0]
    print(f"Found {len(files)} even tif files to downsample.")

    # Check that all downsample_factor-th tif from min to max are present
    min_file = min([int(f.split('.')[0]) for f in files])
    max_file = max([int(f.split('.')[0]) for f in files])
    for i in range(min_file, max_file + 1, downsample_factor):  # Adjusted the range to step by downsample_factor for efficiency
        if (f"{i:04}.tif" not in files) and (f"{i:05}.tif" not in files):
            raise Exception(f"Missing {i:04}.tif")

    # Prepare the arguments for each process
    tasks = [(input_directory, output_directory, f, downsample_factor) for f in files]

    # Initialize a Pool of processes
    with Pool(processes=num_threads) as pool:
        # Process the files in parallel
        for _ in tqdm(pool.imap_unordered(downsample_image, tasks), total=len(tasks)):
            pass

    print('Downsampling complete.')

def process_block(args):
    bz, by, bx, directory_path, block_size, nz, ny, nx, tif_files, standard_size = args
    block_directory = directory_path + '_grids'
    block = np.zeros((block_size, block_size, block_size), dtype=np.uint16)
    block_filename = f"cell_yxz_{by+1:03}_{bx+1:03}_{bz+1:03}.tif"
    block_path = os.path.join(block_directory, block_filename)

    if os.path.exists(block_path) and os.path.getsize(block_path) == standard_size:
        print(f"'{block_filename}' already exists in the output directory. Skipping.")
        return  # Skip if file exists and size matches
    elif os.path.exists(block_path):
        print(f"Warning: '{block_filename}' exists but has the wrong size. Overwriting. {os.path.getsize(block_path)} != {block_size**3 * np.uint16().itemsize}")

    for z in range(block_size):
        z_index = bz * block_size + z
        if z_index >= nz:
            break
        image_path = os.path.join(directory_path, tif_files[z_index])
        image_slice = tifffile.imread(image_path)
        y_slice, x_slice = (slice(b * block_size, min((b + 1) * block_size, d)) for b, d in ((by, ny), (bx, nx)))
        block[z, :y_slice.stop - y_slice.start, :x_slice.stop - x_slice.start] = image_slice[y_slice, x_slice]

    tifffile.imwrite(block_path, block)

def generate_grid_blocks(directory_path, block_size, num_threads):
    block_directory = directory_path + '_grids'
    os.makedirs(block_directory, exist_ok=True)
    tif_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.tif')])

    sample_image = tifffile.imread(os.path.join(directory_path, tif_files[0]))
    nz, ny, nx = len(tif_files), *sample_image.shape
    blocks_in_x, blocks_in_y, blocks_in_z = (int(np.ceil(d / block_size)) for d in (nx, ny, nz))

    # Get standard_size. is -1 if not at least two cells exist and first two not having the same size
    standard_size = -1
    # get grid cells
    grid_cells = [f for f in os.listdir(directory_path + '_grids') if f.endswith('.tif') and f.startswith('cell_yxz_')]
    if len(grid_cells) >= 2:
        # get first two grid cells
        grid_cell_1 = tifffile.imread(os.path.join(directory_path + '_grids', grid_cells[0]))
        grid_cell_1_size = os.path.getsize(os.path.join(directory_path + '_grids', grid_cells[0]))
        grid_cell_2 = tifffile.imread(os.path.join(directory_path + '_grids', grid_cells[1]))
        grid_cell_2_size = os.path.getsize(os.path.join(directory_path + '_grids', grid_cells[1]))
        if grid_cell_1.shape == grid_cell_2.shape and grid_cell_1_size == grid_cell_2_size:
            standard_size = grid_cell_1_size
            print(f"Found standard size: {standard_size}")

    tasks = [(bz, by, bx, directory_path, block_size, nz, ny, nx, tif_files, standard_size) 
             for bz in range(blocks_in_z) for by in range(blocks_in_y) for bx in range(blocks_in_x)]

    # multiprocessing
    num_pools = max(1, num_threads // 3)
    with Pool(processes=num_pools) as pool:
        for _ in tqdm(pool.imap_unordered(process_block, tasks), total=len(tasks)):
            pass

    print('Grid blocks have been generated.')

def fix_zyx(original_directory, new_directory):
    # Create new directory if it does not exist
    os.makedirs(new_directory, exist_ok=True)

    # Iterate over all files in the original directory
    for filename in tqdm(os.listdir(original_directory)):
        if filename.endswith('.tif') and filename.startswith('cell_yxz_'):
            # Full paths for old and new files
            old_filepath = os.path.join(original_directory, filename)
            new_filepath = os.path.join(new_directory, filename)
            
            # Read the original .tif file
            image = tifffile.imread(old_filepath)
            
            # Transpose each slice along the z-axis
            image = np.swapaxes(image, 1, 2)
            image = np.swapaxes(image, 0, 1)

            # Save the transposed image to the new directory
            tifffile.imwrite(new_filepath, image)
    print('XY transposition has been completed.')

def fix_naming(original_directory, new_directory):
    # Create new directory if it does not exist
    os.makedirs(new_directory, exist_ok=True)

    # Iterate over all files in the original directory
    for filename in os.listdir(original_directory):
        if filename.endswith('.tif') and filename.startswith('cell_yxz_'):
            # Extract the block indices from the filename
            parts = filename.rstrip('.tif').split('_')
            if len(parts) == 5:
                bz, by, bx = parts[2], parts[3], parts[4]
                print(by, bx, bz)
                # New filename with corrected order
                new_filename = f"cell_yxz_{int(by):03}_{int(bx):03}_{int(bz):03}.tif"
                
                # Full paths for old and new files
                old_filepath = os.path.join(original_directory, filename)
                new_filepath = os.path.join(new_directory, new_filename)

                # Move and rename the file to the new directory
                shutil.move(old_filepath, new_filepath)

def fix_naming_xy(original_directory, new_directory):
    # Create new directory if it does not exist
    os.makedirs(new_directory, exist_ok=True)

    # Iterate over all files in the original directory
    for filename in tqdm(os.listdir(original_directory)):
        if filename.endswith('.tif') and filename.startswith('cell_yxz_'):
            x, y, z = filename.split(".")[0].split('_')[2:5]
            filename_transposed = f'cell_yxz_{y}_{x}_{z}.tif'
            # Full paths for old and new files
            old_filepath = os.path.join(original_directory, filename)
            new_filepath = os.path.join(new_directory, filename_transposed)
            
            # Read the original .tif file
            image = tifffile.imread(old_filepath)
            
            # Save the transposed image to the new directory
            tifffile.imwrite(new_filepath, image)
    print('XY transposition has been completed.')

def fix_xy_transpose(original_directory, new_directory):
    # Create new directory if it does not exist
    os.makedirs(new_directory, exist_ok=True)

    # Iterate over all files in the original directory
    for filename in tqdm(os.listdir(original_directory)):
        if filename.endswith('.tif') and filename.startswith('cell_yxz_'):
            x, y, z = filename.split(".")[0].split('_')[2:5]
            filename_transposed = f'cell_yxz_{y}_{x}_{z}.tif'
            # Full paths for old and new files
            old_filepath = os.path.join(original_directory, filename)
            new_filepath = os.path.join(new_directory, filename_transposed)
            
            # Read the original .tif file
            image = tifffile.imread(old_filepath)
            
            # Transpose each slice along the z-axis
            image = np.swapaxes(image, 0, 1)

            # Save the transposed image to the new directory
            tifffile.imwrite(new_filepath, image)
    print('XY transposition has been completed.')

def compute(input_directory, output_directory, downsample_factor, num_threads):
    downsample_folder_tifs(input_directory, output_directory, downsample_factor, num_threads)
    generate_grid_blocks(output_directory, 500, num_threads)

def main():
    # Path to the directory containing the .tif files
    input_directory = '/media/julian/SSD4TB/PHerc0332.volpkg/volumes/20231027191953'
    output_directory = '/media/julian/SSD4TB/PHerc0332.volpkg/volumes/2dtifs_8um'
    downsample_factor = 2

    import argparse
    parser = argparse.ArgumentParser(description="Downsample a folder of tif files and generate grid blocks. This file might contain bugs, please test if it works propperly (the blocks should have the same naming and orientation as the spelufo gridblocks).")
    parser.add_argument("--input_directory", type=str, help="Path to the input directory containing the tif files", default=input_directory)
    parser.add_argument("--output_directory", type=str, help="Path to the output directory", default=output_directory)
    parser.add_argument("--downsample_factor", type=int, help="Downsample factor (int)", default=downsample_factor)
    parser.add_argument("--num_threads", type=int, help="Number of threads to use for processing", default=cpu_count())

    # Take arguments back over
    args = parser.parse_args()

    # Print the arguments
    print(f"Arguments for generating downsampled grid cells: \n{args}")

    input_directory = args.input_directory
    output_directory = args.output_directory
    downsample_factor = args.downsample_factor
    num_threads = args.num_threads

    # Compute the downsampled tifs and grid blocks
    compute(input_directory, output_directory, downsample_factor, num_threads)

if __name__ == '__main__':
    main()
