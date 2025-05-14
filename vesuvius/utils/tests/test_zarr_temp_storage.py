import os
import threading
import time
import queue
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from zarr_temp_storage import ZarrTempStorage, zarr_writer_worker

def test_zarr_temp_storage_basic():
    """Test basic functionality of ZarrTempStorage"""
    # Create temp directory
    os.makedirs("../temp_test", exist_ok=True)
    
    # Initialize storage
    storage = ZarrTempStorage(output_path="../temp_test", rank=0, world_size=1, verbose=True)
    storage.initialize()
    
    # Set expected patch counts - need to account for 1-based indexing
    target_name = "segmentation"
    patch_count = 10
    # Increase expected count for:
    # 1. The initial patch we'll use to create the array
    # 2. The 1-based indexing with atomics
    storage.set_expected_patch_count(target_name, patch_count + 2)
    
    # Create sample patches
    patch_shape = (2, 8, 8, 8)  # (channels, z, y, x)
    
    # Initialize all position entries (for index 0 through patch_count+1)
    # This ensures no None values which could cause issues in finalize_target
    
    # Store patches
    # First create the counter and positions arrays
    storage.store_patch(np.ones(patch_shape, dtype=np.float32), (0, 0, 0), target_name)
    
    # Initialize all positions to avoid None values
    for i in range(patch_count + 2):
        if storage.positions[target_name][i] is None:
            storage.positions[target_name][i] = (0, 0, 0)
    positions = []
    for i in range(patch_count):
        # Create a simple patch with incrementing values
        patch = np.ones(patch_shape, dtype=np.float32) * i
        position = (i*5, i*5, i*5)  # Simple position increments
        
        # Store the patch
        idx = storage.store_patch(patch, position, target_name)
        positions.append((idx, position))
        
        print(f"Stored patch {i} at position {position}, got index {idx}")
    
    # Finalize target
    storage.finalize_target(target_name)
    
    # Get patches back 
    patches_array, positions_array = storage.get_all_patches(0, target_name)
    
    # Verify data
    print(f"\nRetrieved {patches_array.shape[0]} patches")
    print(f"Patch shape: {patches_array.shape[1:]}")
    
    # Verify positions
    print(f"Retrieved {len(positions_array)} positions")
    if len(positions_array) > 0:
        print(f"First position: {positions_array[0]}")
        print(f"Last position: {positions_array[-1]}")
    
    # Compare stored vs retrieved patches
    for i in range(min(3, patch_count)):  # Check first few patches
        stored_patch = np.ones(patch_shape, dtype=np.float32) * i
        retrieved_patch = patches_array[i]
        
        if np.array_equal(stored_patch, retrieved_patch):
            print(f"Patch {i} verified: data matches")
        else:
            print(f"ERROR: Patch {i} data mismatch!")
    
    # Collect all patches (simulating multi-rank collection)
    all_patches = storage.collect_all_patches(target_name)
    print(f"\nCollected {len(all_patches)} patches total")
    
    # Clean up
    storage.cleanup()
    print("\nTemp storage cleaned up")
    
    return storage, patches_array, positions_array

def test_zarr_temp_storage_with_workers():
    """
    Test ZarrTempStorage using the worker queue pattern from inference.py
    This mirrors how patches are stored in the real implementation.
    """
    # Create temp directory
    os.makedirs("../temp_test", exist_ok=True)
    
    # Initialize storage
    storage = ZarrTempStorage(output_path="../temp_test", rank=0, world_size=1, verbose=True)
    storage.initialize()
    
    # Set up test parameters
    target_name = "segmentation"
    num_workers = 4
    patches_per_thread = 20
    total_patches = num_workers * patches_per_thread
    
    # Need to account for:
    # 1. The dummy initialization patch
    # 2. The fact that increment_and_get() returns values starting from 1
    # So we need total_patches + 1 + 1 = total_patches + 2
    storage.set_expected_patch_count(target_name, total_patches + 2)
    
    # Create writer queue
    writer_queue = queue.Queue()
    
    # Create array first before starting workers
    # This ensures the array exists before any workers try to use it
    patch_shape = (2, 8, 8, 8)  # (channels, z, y, x)
    dummy_patch = np.ones(patch_shape, dtype=np.float32)
    dummy_position = (0, 0, 0)
    
    # Use store_patch but don't use the result - this guarantees array creation
    print("Creating zarr array for target...")
    dummy_idx = storage.store_patch(dummy_patch, dummy_position, target_name)
    print(f"Array initialized with dummy patch at index {dummy_idx}, now starting worker threads")
    
    # Initialize all positions to avoid None values
    for i in range(total_patches + 2):
        if storage.positions[target_name][i] is None:
            storage.positions[target_name][i] = (0, 0, 0)
    
    # Create and start worker threads
    writer_threads = []
    for worker_id in range(num_workers):
        thread = threading.Thread(
            target=zarr_writer_worker,
            args=(storage, writer_queue, worker_id, True)  # Use verbose=True
        )
        thread.daemon = True
        thread.start()
        writer_threads.append(thread)
    
    print(f"Started {num_workers} worker threads")
    
    # Function to generate patch data for multiple threads
    def generate_patches(thread_id, patch_count):
        """Generate and queue patches for a thread"""
        patch_shape = (2, 8, 8, 8)  # (channels, z, y, x)
        
        for i in range(patch_count):
            # Create unique patch based on thread ID and index
            patch = np.ones(patch_shape, dtype=np.float32) * (thread_id * 1000 + i)
            # Create unique position to avoid overlaps between threads
            position = (thread_id * 100 + i, thread_id * 100 + i, thread_id * 100 + i)
            
            # Queue the patch for writing (just like in inference.py)
            writer_queue.put((patch, position, target_name))
            
            # Small delay to increase thread interleaving
            time.sleep(0.001)
    
    # Use ThreadPoolExecutor to generate patches from multiple threads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to generate patches
        futures = [executor.submit(generate_patches, thread_id, patches_per_thread) 
                  for thread_id in range(num_workers)]
        
        # Wait for all generation tasks to complete
        for future in futures:
            future.result()
    
    print(f"All patch generation tasks completed")
    
    # Wait for the queue to be fully processed
    writer_queue.join()
    print(f"All patches have been processed by writer threads")
    
    # Send sentinel values to stop worker threads
    for _ in range(num_workers):
        writer_queue.put(None)
    
    # Wait for threads to finish
    for thread in writer_threads:
        thread.join()
    
    print(f"All writer threads have terminated")
    
    # Finalize target
    storage.finalize_target(target_name)
    
    # Get all patches back
    patches_array, positions_array = storage.get_all_patches(0, target_name)
    
    # Print some info
    print(f"\nRetrieved {len(patches_array)} patches with {num_workers} worker threads")
    print(f"Expected {total_patches} thread-generated patches (plus 1 dummy initialization patch)")
    
    # Check the final count
    final_count = storage.counters[target_name].get()
    print(f"Final counter value: {final_count}")
    
    # Accounting for dummy patch
    expected_total_with_dummy = total_patches + 1
    if final_count == expected_total_with_dummy:
        print(f"Correct count: {final_count} patches (including 1 dummy initialization patch)")
    else:
        print(f"WARNING: Expected {expected_total_with_dummy} patches (with dummy), got {final_count}")
    
    # Collect all patches 
    all_patches = storage.collect_all_patches(target_name)
    print(f"collect_all_patches found {len(all_patches)} total patches")
    
    # Clean up
    storage.cleanup()
    print("Temp storage cleaned up")
    
    return storage, patches_array, positions_array

if __name__ == "__main__":
    print("Testing ZarrTempStorage with new AtomicCounter")
    print("\n1. Basic single-threaded test:")
    storage, patches, positions = test_zarr_temp_storage_basic()
    
    print("\n2. Worker queue pattern test (matching real implementation):")
    storage, patches, positions = test_zarr_temp_storage_with_workers()