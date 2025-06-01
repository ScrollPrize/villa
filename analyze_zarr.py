import zarr
import numpy as np
import fsspec

# Open the output zarr using fsspec for consistent pattern
print("Opening zarr file...")
zarr_path = '/mnt/raid_nvme/s5_test_surf.zarr'
mapper = fsspec.get_mapper(zarr_path)
z = zarr.open(mapper, mode='r')

# Print basic information
print(f"Shape: {z.shape}")
print(f"Data type: {z.dtype}")

# Extract a small region to analyze (adjust indices if needed)
print("Extracting data sample...")
try:
    data = z[0, 100:200, 100:200, 100:200]
except:
    # If the zarr has a different shape, try a more flexible approach
    print("Trying alternative indexing...")
    if len(z.shape) == 4:  # (C, Z, Y, X)
        mid_z = z.shape[1] // 2
        mid_y = z.shape[2] // 2
        mid_x = z.shape[3] // 2
        size = 50
        data = z[0, 
                 max(0, mid_z-size):min(z.shape[1], mid_z+size),
                 max(0, mid_y-size):min(z.shape[2], mid_y+size),
                 max(0, mid_x-size):min(z.shape[3], mid_x+size)]
    else:
        # For other shapes, take the first elements
        print(f"Using flexible indexing for shape {z.shape}")
        data = z[tuple(slice(0, min(1, s)) if i == 0 else slice(0, min(100, s)) 
                for i, s in enumerate(z.shape))]

# Check value range
print(f"Min: {data.min()}, Max: {data.max()}")

# Check if values are only 0 and 1
unique_vals = np.unique(data)
if len(unique_vals) <= 10:
    print(f"Unique values: {unique_vals}")
else:
    print(f"Number of unique values: {len(unique_vals)}")
    print(f"Sample of unique values: {np.sort(unique_vals)[:5]} ... {np.sort(unique_vals)[-5:]}")

# Count number of 0s and 1s (exact)
exact_zeros = np.count_nonzero(data == 0)
exact_ones = np.count_nonzero(data == 1)
total_elements = data.size
print(f"Exact 0s: {exact_zeros} ({exact_zeros/total_elements:.2%} of data)")
print(f"Exact 1s: {exact_ones} ({exact_ones/total_elements:.2%} of data)")
print(f"Other values: {total_elements - exact_zeros - exact_ones} ({(total_elements - exact_zeros - exact_ones)/total_elements:.2%} of data)")

# Count values in ranges
near_zero = np.count_nonzero((data > 0) & (data < 0.1))
near_one = np.count_nonzero((data > 0.9) & (data < 1))
mid_range = np.count_nonzero((data >= 0.1) & (data <= 0.9))
print(f"Values 0-0.1 (near zero): {near_zero} ({near_zero/total_elements:.2%} of data)")
print(f"Values 0.1-0.9 (mid-range): {mid_range} ({mid_range/total_elements:.2%} of data)")
print(f"Values 0.9-1 (near one): {near_one} ({near_one/total_elements:.2%} of data)")

# Generate histogram for distribution
print("\nValue distribution histogram:")
hist, bins = np.histogram(data.flatten(), bins=20)
for i in range(len(hist)):
    print(f"{bins[i]:.4f} - {bins[i+1]:.4f}: {hist[i]} ({hist[i]/total_elements:.2%})")

# Check for NaN or Inf values
nan_count = np.count_nonzero(np.isnan(data))
inf_count = np.count_nonzero(np.isinf(data))
print(f"\nNaN values: {nan_count}")
print(f"Inf values: {inf_count}")
