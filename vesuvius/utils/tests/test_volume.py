import os
import sys
import numpy as np
import torch
import time

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

# Import the Volume class - will import directly to avoid circular imports
try:
    from data.volume import Volume
    print("Successfully imported Volume class")
except Exception as e:
    print(f"Error importing Volume class: {e}")
    sys.exit(1)

def test_zscore_normalization():
    """
    Test zscore normalization in Volume to check for NaN values.
    """
    print("\n=== Testing Z-Score Normalization ===\n")
    
    # Load the production zarr file
    production_zarr_path = "/mnt/raid_nvme/s5_test.zarr"
    
    if not os.path.exists(production_zarr_path):
        print(f"Production zarr not found at {production_zarr_path}, skipping test")
        return

    # Test with zscore normalization
    print("Testing with normalization_scheme='zscore'...")
    vol_zscore = Volume(type='zarr', path=production_zarr_path, normalize=True, 
                        normalization_scheme='zscore', verbose=True)
    
    # Get shape
    shape = vol_zscore.shape()
    print(f"Volume shape: {shape}")
    
    # Take a more significant portion of the volume - 192³ cubic patch
    z_size = min(192, shape[0])
    y_size = min(192, shape[1])
    x_size = min(192, shape[2])
    
    # Try to get a slice and check for NaN values
    start_time = time.time()
    zscore_slice = vol_zscore[0:z_size, 0:y_size, 0:x_size]
    slice_time = time.time() - start_time
    
    print(f"Z-score normalized slice shape: {zscore_slice.shape}")
    print(f"Z-score normalized slice time: {slice_time:.2f} seconds")
    
    # Check for NaN values
    nan_count = np.isnan(zscore_slice).sum()
    print(f"NaN count in z-score normalized slice: {nan_count}")
    print(f"Total elements in slice: {zscore_slice.size}")
    print(f"Percentage of NaN values: {nan_count / zscore_slice.size * 100:.4f}%")
    
    # Check data distribution
    if nan_count == 0:
        print("\nZ-score normalized data statistics:")
        print(f"Min value: {np.min(zscore_slice):.4f}")
        print(f"Max value: {np.max(zscore_slice):.4f}")
        print(f"Mean value: {np.mean(zscore_slice):.4f}")
        print(f"Standard deviation: {np.std(zscore_slice):.4f}")
        
        # Count values outside normal range
        out_of_range = np.sum(np.abs(zscore_slice) > 5.0)
        print(f"Values with magnitude > 5.0: {out_of_range} ({out_of_range / zscore_slice.size * 100:.4f}%)")
        
        # Count extreme values
        extreme_values = np.sum(np.abs(zscore_slice) > 10.0) 
        print(f"Values with magnitude > 10.0: {extreme_values} ({extreme_values / zscore_slice.size * 100:.4f}%)")
    
    if nan_count > 0:
        print("\nWARNING: Found NaN values in z-score normalized data!")
        print("This is likely causing the inference issues with NaN outputs.")
        
        # Create a simple heatmap of where NaNs are occurring
        print("\nNaN heatmap (slice 0):")
        nan_heatmap = np.isnan(zscore_slice[0]).astype(int)
        print(f"Slice 0 NaN count: {nan_heatmap.sum()}")
        
        # Show where the standard deviation is 0 (causes NaNs in zscore)
        print("\nChecking for regions with zero standard deviation:")
        std_values = np.std(vol_zscore[0:z_size, 0:y_size, 0:x_size], axis=0)
        zero_std_count = (std_values == 0).sum()
        print(f"Regions with zero standard deviation: {zero_std_count}")
        print(f"Percentage of regions with zero std: {zero_std_count / std_values.size * 100:.4f}%")
    
    # Test with no normalization
    print("\nTesting with normalization_scheme='none'...")
    vol_none = Volume(type='zarr', path=production_zarr_path, normalize=True, 
                      normalization_scheme='none', verbose=True)
    
    none_slice = vol_none[0:z_size, 0:y_size, 0:x_size]
    nan_count_none = np.isnan(none_slice).sum()
    print(f"NaN count with normalization_scheme='none': {nan_count_none}")
    
    if nan_count_none == 0:
        print("\nStandard normalized data statistics (normalize=True, normalization_scheme='none'):")
        print(f"Min value: {np.min(none_slice):.4f}")
        print(f"Max value: {np.max(none_slice):.4f}")
        print(f"Mean value: {np.mean(none_slice):.4f}")
        print(f"Standard deviation: {np.std(none_slice):.4f}")
    
    # Test with minmax normalization
    print("\nTesting with normalization_scheme='minmax'...")
    vol_minmax = Volume(type='zarr', path=production_zarr_path, normalize=True, 
                        normalization_scheme='minmax', verbose=True)
    
    minmax_slice = vol_minmax[0:z_size, 0:y_size, 0:x_size]
    nan_count_minmax = np.isnan(minmax_slice).sum()
    print(f"NaN count with normalization_scheme='minmax': {nan_count_minmax}")
    
    if nan_count_minmax == 0:
        print("\nMinMax normalized data statistics:")
        print(f"Min value: {np.min(minmax_slice):.4f}")
        print(f"Max value: {np.max(minmax_slice):.4f}")
        print(f"Mean value: {np.mean(minmax_slice):.4f}")
        print(f"Standard deviation: {np.std(minmax_slice):.4f}")
    
    print("\n=== Z-Score Normalization Test Complete ===")

def test_volume():
    """
    Test the Volume class with various data sources:
    1. Scroll 1 from remote HTTP
    2. Local zarr file
    3. Remote zarr via HTTP
    """
    print("\n=== Testing Volume Class with Multiple Data Sources ===\n")

    # 1. Load the volume with verbose logging
    print("Loading Scroll 1...")
    start_time = time.time()
    vol = Volume(type='scroll', scroll_id=1, verbose=True)
    load_time = time.time() - start_time
    print(f"Volume loaded in {load_time:.2f} seconds")
    
    # 2. Get and print the shape
    shape = vol.shape()
    print(f"Volume shape: {shape}")
    
    # 3. Get a small cube with NumPy-style slicing
    print("\nTesting basic slicing...")
    z_center, y_center, x_center = shape[0]//2, shape[1]//2, shape[2]//2
    z_start, y_start, x_start = z_center-128, y_center-128, x_center-128
    z_end, y_end, x_end = z_center+128, y_center+128, x_center+128
    
    print(f"Slicing a 256³ subvolume from coordinates:")
    print(f"Z: {z_start}:{z_end}, Y: {y_start}:{y_end}, X: {x_start}:{x_end}")
    
    start_time = time.time()
    subcube = vol[z_start:z_end, y_start:y_end, x_start:x_end]
    slice_time = time.time() - start_time
    
    print(f"Sliced subvolume in {slice_time:.2f} seconds")
    print(f"Subvolume shape: {subcube.shape}")
    print(f"Subvolume dtype: {subcube.dtype}")
    print(f"Subvolume min value: {np.min(subcube)}")
    print(f"Subvolume max value: {np.max(subcube)}")
    
    # 4. Test normalization
    print("\nTesting normalization...")
    vol_norm = Volume(type='scroll', scroll_id=1, normalize=True, verbose=True)
    
    start_time = time.time()
    subcube_norm = vol_norm[z_start:z_end, y_start:y_end, x_start:x_end]
    norm_time = time.time() - start_time
    
    print(f"Sliced normalized subvolume in {norm_time:.2f} seconds")
    print(f"Normalized subvolume shape: {subcube_norm.shape}")
    print(f"Normalized subvolume dtype: {subcube_norm.dtype}")
    print(f"Normalized subvolume min value: {np.min(subcube_norm)}")
    print(f"Normalized subvolume max value: {np.max(subcube_norm)}")
    
    # 5. Test conversion to uint8
    print("\nTesting conversion to uint8...")
    vol_uint8 = Volume(type='scroll', scroll_id=1, normalize=True, 
                       return_as_type='np.uint8', verbose=True)
    
    start_time = time.time()
    subcube_uint8 = vol_uint8[z_start:z_end, y_start:y_end, x_start:x_end]
    uint8_time = time.time() - start_time
    
    print(f"Sliced uint8 subvolume in {uint8_time:.2f} seconds")
    print(f"uint8 subvolume shape: {subcube_uint8.shape}")
    print(f"uint8 subvolume dtype: {subcube_uint8.dtype}")
    print(f"uint8 subvolume min value: {np.min(subcube_uint8)}")
    print(f"uint8 subvolume max value: {np.max(subcube_uint8)}")
    
    # 6. Test conversion to tensor
    print("\nTesting conversion to PyTorch tensor...")
    vol_tensor = Volume(type='scroll', scroll_id=1, normalize=True, 
                       return_as_tensor=True, verbose=True)
    
    start_time = time.time()
    subcube_tensor = vol_tensor[z_start:z_end, y_start:y_end, x_start:x_end]
    tensor_time = time.time() - start_time
    
    print(f"Sliced tensor subvolume in {tensor_time:.2f} seconds")
    print(f"Tensor subvolume shape: {subcube_tensor.shape}")
    print(f"Tensor subvolume dtype: {subcube_tensor.dtype}")
    print(f"Tensor subvolume min value: {torch.min(subcube_tensor).item()}")
    print(f"Tensor subvolume max value: {torch.max(subcube_tensor).item()}")
    
    # 7. Test with caching on and off
    print("\nTesting caching...")
    
    # First with caching off
    vol_no_cache = Volume(type='scroll', scroll_id=1, cache=False, verbose=True)
    
    start_time = time.time()
    subcube_no_cache = vol_no_cache[z_start:z_end, y_start:y_end, x_start:x_end]
    no_cache_time = time.time() - start_time
    
    print(f"Sliced without cache in {no_cache_time:.2f} seconds")
    
    # Then with caching on
    vol_with_cache = Volume(type='scroll', scroll_id=1, cache=True, 
                           cache_pool=1e9, verbose=True)  # 1GB cache
    
    start_time = time.time()
    subcube_with_cache = vol_with_cache[z_start:z_end, y_start:y_end, x_start:x_end]
    with_cache_time = time.time() - start_time
    
    print(f"Sliced with cache in {with_cache_time:.2f} seconds")
    
    # Do a second read to see caching benefits
    start_time = time.time()
    subcube_with_cache_2 = vol_with_cache[z_start:z_end, y_start:y_end, x_start:x_end]
    with_cache_time_2 = time.time() - start_time
    
    print(f"Second slice with cache in {with_cache_time_2:.2f} seconds")
    
    # 8. Test smaller and larger cube sizes
    print("\nTesting different cube sizes...")
    
    # Small cube (32³)
    start_time = time.time()
    small_cube = vol[z_center-16:z_center+16, y_center-16:y_center+16, x_center-16:x_center+16]
    small_time = time.time() - start_time
    print(f"32³ cube in {small_time:.2f} seconds, shape: {small_cube.shape}")
    
    # Medium cube (128³)
    start_time = time.time()
    medium_cube = vol[z_center-64:z_center+64, y_center-64:y_center+64, x_center-64:x_center+64]
    medium_time = time.time() - start_time
    print(f"128³ cube in {medium_time:.2f} seconds, shape: {medium_cube.shape}")
    
    # Large cube (512³) - if possible
    try:
        start_time = time.time()
        large_cube = vol[z_center-256:z_center+256, y_center-256:y_center+256, x_center-256:x_center+256]
        large_time = time.time() - start_time
        print(f"512³ cube in {large_time:.2f} seconds, shape: {large_cube.shape}")
    except Exception as e:
        print(f"Could not load 512³ cube: {e}")
    
    # 9. Test local zarr directly
    print("\n=== Testing Local Zarr Access ===")
    local_zarr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s5_test.zarr")
    
    if os.path.exists(local_zarr_path):
        print(f"Loading local zarr from: {local_zarr_path}")
        try:
            start_time = time.time()
            vol_local = Volume(type='zarr', path=local_zarr_path, verbose=True)
            local_load_time = time.time() - start_time
            print(f"Local zarr loaded in {local_load_time:.2f} seconds")
            
            # Get shape and test slicing
            local_shape = vol_local.shape()
            print(f"Local zarr shape: {local_shape}")
            
            # Take appropriate slice
            z_size = min(64, local_shape[0])
            y_size = min(64, local_shape[1])
            x_size = min(64, local_shape[2])
            
            start_time = time.time()
            local_slice = vol_local[0:z_size, 0:y_size, 0:x_size]
            local_slice_time = time.time() - start_time
            
            print(f"Local zarr slice shape: {local_slice.shape}")
            print(f"Local zarr slice time: {local_slice_time:.2f} seconds")
            print(f"Local zarr min/max: {np.min(local_slice)}, {np.max(local_slice)}")
            
            # Test normalization
            vol_local_norm = Volume(type='zarr', path=local_zarr_path, normalize=True)
            local_norm = vol_local_norm[0:z_size, 0:y_size, 0:x_size]
            print(f"Local zarr normalized shape: {local_norm.shape}")
            print(f"Local zarr normalized dtype: {local_norm.dtype}")
            print(f"Local zarr normalized min/max: {np.min(local_norm)}, {np.max(local_norm)}")
            
        except Exception as e:
            print(f"Error with local zarr: {e}")
    else:
        print(f"Local zarr not found at {local_zarr_path}")
    
    # 10. Test remote HTTP zarr directly
    print("\n=== Testing HTTP Zarr Access ===")
    http_zarr_url = "https://dl.ash2txt.org/community-uploads/bruniss/test_vols/s5_test.zarr/"
    
    print(f"Loading HTTP zarr from: {http_zarr_url}")
    try:
        start_time = time.time()
        vol_http = Volume(type='zarr', path=http_zarr_url, verbose=True)
        http_load_time = time.time() - start_time
        print(f"HTTP zarr loaded in {http_load_time:.2f} seconds")
        
        # Get shape and test slicing
        http_shape = vol_http.shape()
        print(f"HTTP zarr shape: {http_shape}")
        
        # Take appropriate slice
        z_size = min(64, http_shape[0])
        y_size = min(64, http_shape[1])
        x_size = min(64, http_shape[2])
        
        start_time = time.time()
        http_slice = vol_http[0:z_size, 0:y_size, 0:x_size]
        http_slice_time = time.time() - start_time
        
        print(f"HTTP zarr slice shape: {http_slice.shape}")
        print(f"HTTP zarr slice time: {http_slice_time:.2f} seconds")
        print(f"HTTP zarr min/max: {np.min(http_slice)}, {np.max(http_slice)}")
        
        # Test normalization
        vol_http_norm = Volume(type='zarr', path=http_zarr_url, normalize=True)
        http_norm = vol_http_norm[0:z_size, 0:y_size, 0:x_size]
        print(f"HTTP zarr normalized shape: {http_norm.shape}")
        print(f"HTTP zarr normalized dtype: {http_norm.dtype}")
        print(f"HTTP zarr normalized min/max: {np.min(http_norm)}, {np.max(http_norm)}")
        
    except Exception as e:
        print(f"Error with HTTP zarr: {e}")
    
    print("\n=== Volume Class Tests Completed Successfully ===")
    
    return {
        "shape": shape,
        "slice_time": slice_time,
        "norm_time": norm_time,
        "uint8_time": uint8_time,
        "tensor_time": tensor_time,
        "no_cache_time": no_cache_time,
        "with_cache_time": with_cache_time,
        "with_cache_time_2": with_cache_time_2,
        "small_time": small_time,
        "medium_time": medium_time
    }

if __name__ == "__main__":
    try:
        # Run the z-score normalization test first
        test_zscore_normalization()
        
        # Uncomment to run the full volume test
        # results = test_volume()
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)