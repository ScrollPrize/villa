import os
import re
import numpy as np
import tifffile as tiff
import zarr
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import cpu_count

def collect_grid_coordinates(base_dir):
    pattern = re.compile(r"cell_yxz_(\d{3})_(\d{3})_(\d{3})\.tif")
    grid_coords = [(int(m.group(1)), int(m.group(2)), int(m.group(3))) 
                   for m in (pattern.match(f) for f in os.listdir(base_dir)) 
                   if m is not None]
    return sorted(grid_coords)

def read_tiff_to_array(filename):
    try:
        return tiff.memmap(filename, mode='r')
    except (ValueError, Exception):
        return tiff.imread(filename)

def determine_zarr_shape(grid_coords, full_volume_shape=None, grid_size=500):
    if not grid_coords:
        raise ValueError("No grid coordinates found")
        
    if full_volume_shape is not None:
        return full_volume_shape
    else:
        max_jy = max(jy for jy, _, _ in grid_coords)
        max_jx = max(jx for _, jx, _ in grid_coords)
        max_jz = max(jz for _, _, jz in grid_coords)
        return ((max_jz) * grid_size, (max_jy) * grid_size, (max_jx) * grid_size)

def process_single_tiff(args):
    try:
        zarr_array, base_dir, coords, grid_size, trim_size, value_map, debug = args
        jy, jx, jz = coords
        filename = os.path.join(base_dir, f"cell_yxz_{jy:03d}_{jx:03d}_{jz:03d}.tif")
        
        if not os.path.exists(filename):
            if debug:
                print(f"Warning: File not found: {filename}")
            return
            
        # Read the TIFF data
        tiff_data = read_tiff_to_array(filename)
        
        if debug:
            print(f"Original block shape for {filename}: {tiff_data.shape}")
            
        # Verify the input block size
        expected_size = grid_size + 2 * trim_size
        if tiff_data.shape != (expected_size, expected_size, expected_size):
            print(f"Warning: Unexpected block size in {filename}: {tiff_data.shape}")
            print(f"Expected ({expected_size}, {expected_size}, {expected_size})")
            
        # Convert 1-based indices to 0-based for array positioning
        # Calculate the base position in the final array
        z_start = (jz - 1) * grid_size
        y_start = (jy - 1) * grid_size
        x_start = (jx - 1) * grid_size
        
        # Trim the overlapping regions
        if trim_size > 0:
            tiff_data = tiff_data[
                trim_size:-trim_size,
                trim_size:-trim_size,
                trim_size:-trim_size
            ]
            
        if debug:
            print(f"Trimmed block shape: {tiff_data.shape}")
            print(f"Writing to position: ({z_start}:{z_start+grid_size}, {y_start}:{y_start+grid_size}, {x_start}:{x_start+grid_size})")
        
        # Apply value mapping if provided
        if value_map is not None:
            mapped_data = np.zeros_like(tiff_data)
            for original_value, new_value in value_map.items():
                mapped_data[tiff_data == original_value] = new_value
            tiff_data = mapped_data
        
        try:
            zarr_array[
                z_start:z_start + grid_size,
                y_start:y_start + grid_size,
                x_start:x_start + grid_size
            ] = tiff_data
            
        except ValueError as e:
            print(f"\nDetailed error info for {filename}:")
            print(f"Block shape after trimming: {tiff_data.shape}")
            print(f"Attempted write region: {grid_size}x{grid_size}x{grid_size}")
            print(f"Start indices: ({z_start}, {y_start}, {x_start})")
            print(f"End indices: ({z_start+grid_size}, {y_start+grid_size}, {x_start+grid_size})")
            raise e
            
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        raise

def main(base_dir, zarr_store_path, trim_size=0, full_volume_shape=None, max_workers=4, 
         chunk_size=128, value_map=None, debug=False):
    original_grid_size = 500  # Original size of input blocks
    effective_grid_size = original_grid_size - 2 * trim_size  # Size after trimming
    
    print("Collecting grid coordinates...")
    grid_coords = collect_grid_coordinates(base_dir)
    if not grid_coords:
        raise ValueError("No valid TIFF files found in the input directory")
    
    if debug:
        sample_files = sorted(os.listdir(base_dir))[:5]
        print("\nChecking sample file shapes:")
        for f in sample_files:
            if f.endswith('.tif'):
                shape = tiff.imread(os.path.join(base_dir, f)).shape
                print(f"{f}: {shape}")
    
    print("\nDetermining array shape...")
    if full_volume_shape:
        zarr_shape = full_volume_shape
    else:
        # Adjust shape calculation to account for trimming
        zarr_shape = determine_zarr_shape(grid_coords, full_volume_shape, effective_grid_size)
    print(f"Creating Zarr array with shape: {zarr_shape}")
    
    # Determine appropriate dtype
    if value_map:
        max_value = max(value_map.values())
        if max_value <= 255:
            zarr_dtype = np.uint8
        elif max_value <= 65535:
            zarr_dtype = np.uint16
        else:
            zarr_dtype = np.uint32
    else:
        zarr_dtype = np.uint8
    
    print(f"Using dtype: {zarr_dtype}")
    
    compressor = zarr.Blosc(cname='zstd', clevel=3)
    
    print(f"Creating Zarr array with chunk size: {chunk_size}...")
    zarr_store = zarr.DirectoryStore(zarr_store_path)
    zarr_array = zarr.create(
        store=zarr_store,
        shape=zarr_shape,
        dtype=zarr_dtype,
        chunks=(chunk_size, chunk_size, chunk_size),
        compressor=compressor,
        fill_value=0,
        overwrite=True
    )
    
    process_args = [(zarr_array, base_dir, coords, original_grid_size, trim_size, value_map, debug) 
                   for coords in grid_coords]
    
    print(f"Processing {len(process_args)} TIFF files in parallel...")
    print(f"Trimming {trim_size} pixels from each edge of blocks")
    if value_map:
        print(f"Using value mapping: {value_map}")
        
    failed_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for args in process_args:
            future = executor.submit(process_single_tiff, args)
            futures.append(future)
        
        for future in tqdm(futures, total=len(futures), desc="Converting TIFF files"):
            try:
                future.result()
            except Exception as e:
                failed_files.append(str(e))
    
    if failed_files:
        print("\nWarning: Some files failed to process:")
        for error in failed_files:
            print(error)
    else:
        print("\nAll files processed successfully!")
    
    print(f"Final Zarr array shape: {zarr_array.shape}")
    return zarr_array

if __name__ == "__main__":
    base_dir = "/mnt/raid_nvme/scroll3.volpkg/050_extra_da_s3_thresh"
    zarr_store_path = "/mnt/raid_nvme/scroll3.volpkg/050_extra_da_s3_thresh.zarr"
    scroll = '3'
    map = True
    
    # specify the full volume shape (z, y, x)
    s1_volume_shape = (14376, 7888, 8096) # scroll1
    s2_volume_shape = (14428, 10112, 11984) # scroll2
    s3_volume_shape = (9778, 3550, 3400) # scroll3
    s4_volume_shape = (11174, 3440, 3340) # scroll4
    
    if scroll == '1':
        full_volume_shape = s1_volume_shape
    if scroll == '2':
        full_volume_shape = s2_volume_shape
    if scroll == '3':
        full_volume_shape = s3_volume_shape
    if scroll == '4':
        full_volume_shape = s4_volume_shape

    
    if map == True:
        # Define your value mapping here
        value_map = {
            0: 0,      # keep 0 as 0
            1: 255    # map 1 to 255
        }
    
    if map == False: 
        value_map = None
    
    # Define trim size (number of pixels to trim from each edge)
    trim_size = 50  # Example: if your overlap was 50 pixels, trim 50 from each edge
    
    max_workers = 8
    chunk_size = 128
    debug = False
    
    zarr_array = main(base_dir, zarr_store_path, 
                     trim_size=trim_size,
                     full_volume_shape=full_volume_shape, 
                     max_workers=max_workers, 
                     chunk_size=chunk_size, 
                     value_map=value_map,
                     debug=debug)
