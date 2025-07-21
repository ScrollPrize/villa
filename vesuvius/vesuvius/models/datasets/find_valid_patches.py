import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import zarr

def _chunker(seq, chunk_size):
    """Yield successive 'chunk_size'-sized chunks from 'seq'."""
    for pos in range(0, len(seq), chunk_size):
        yield seq[pos:pos + chunk_size]
        
def compute_bounding_box_3d(mask):
    """
    Given a 3D boolean array (True where labeled, False otherwise),
    returns (minz, maxz, miny, maxy, minx, maxx).
    If there are no nonzero elements, returns None.
    """
    nonzero_coords = np.argwhere(mask)
    if nonzero_coords.size == 0:
        return None

    minz, miny, minx = nonzero_coords.min(axis=0)
    maxz, maxy, maxx = nonzero_coords.max(axis=0)
    return (minz, maxz, miny, maxy, minx, maxx)

def bounding_box_volume(bbox):
    """
    Given a bounding box (minz, maxz, miny, maxy, minx, maxx),
    returns the volume (number of voxels) inside the box.
    """
    minz, maxz, miny, maxy, minx, maxx = bbox
    return ((maxz - minz + 1) *
            (maxy - miny + 1) *
            (maxx - minx + 1))

def check_patch_chunk(chunk, sheet_label, patch_size, bbox_threshold=0.5, label_threshold=0.05):
    """
    Worker function to check each patch in 'chunk' with both:
      - bounding box coverage >= bbox_threshold
      - overall labeled voxel ratio >= label_threshold
    """
    pD, pH, pW = patch_size
    valid_positions = []

    for (z, y, x) in chunk:
        patch = sheet_label[z:z + pD, y:y + pH, x:x + pW]
        # Compute bounding box of nonzero pixels in this patch
        bbox = compute_bounding_box_3d(patch > 0)
        if bbox is None:
            # No nonzero voxels at all -> skip
            continue

        # 1) Check bounding box coverage
        bb_vol = bounding_box_volume(bbox)
        patch_vol = patch.size  # pD * pH * pW
        if bb_vol / patch_vol < bbox_threshold:
            continue

        # 2) Check overall labeled fraction
        labeled_ratio = np.count_nonzero(patch) / patch_vol
        if labeled_ratio < label_threshold:
            continue

        # If we passed both checks, add to valid positions
        valid_positions.append((z, y, x))

    return valid_positions

def find_valid_patches(label_arrays,
                        label_names,
                        patch_size,
                        bbox_threshold=0.97,  # bounding-box coverage fraction
                        label_threshold=0.10,  # minimum % of voxels labeled,
                        min_z = 0,
                        min_y = 0,
                        min_x = 0,
                        max_z = None,
                        max_y = None,
                        max_x = None,
                        num_workers=4,
                        downsample_level=1):  
    """
    Finds patches that contain:
      - a bounding box of labeled voxels >= bbox_threshold fraction of the patch volume
      - an overall labeled voxel fraction >= label_threshold
    
    Args:
        label_arrays: List of zarr arrays (label volumes) - should be OME-ZARR root groups
        label_names: List of names for each volume (filename without suffix)
        patch_size: (pZ, pY, pX) tuple for FULL RESOLUTION patches
        bbox_threshold: minimum bounding box coverage fraction
        label_threshold: minimum labeled voxel fraction
        min_z, min_y, min_x: minimum coordinates for patch extraction (full resolution)
        max_z, max_y, max_x: maximum coordinates for patch extraction (full resolution)
        num_workers: number of processes for parallel processing
        downsample_level: Resolution level to use for patch finding (0=full res, 1=2x downsample, etc.)
    
    Returns:
        List of dictionaries with 'volume_idx', 'volume_name', and 'start_pos' (coordinates at full resolution)
    """
    if len(label_arrays) != len(label_names):
        raise ValueError("Number of label arrays must match number of label names")
    
    pZ, pY, pX = patch_size
    all_valid_patches = []
    
    # Calculate downsampled patch size
    downsample_factor = 2 ** downsample_level
    downsampled_patch_size = (pZ // downsample_factor, pY // downsample_factor, pX // downsample_factor)
    
    print(
        f"Finding valid patches of size: {patch_size} (full resolution) "
        f"using downsample level {downsample_level} with patch size {downsampled_patch_size} "
        f"with bounding box coverage >= {bbox_threshold} and labeled fraction >= {label_threshold}."
    )
    
    # Outer progress bar for volumes
    for vol_idx, (label_array, label_name) in enumerate(tqdm(
        zip(label_arrays, label_names), 
        total=len(label_arrays),
        desc="Processing volumes",
        position=0
    )):
        print(f"\nProcessing volume '{label_name}' ({vol_idx + 1}/{len(label_arrays)})")
        
        # Access the appropriate resolution level for patch finding
        try:
            if downsample_level == 0:
                # Use full resolution
                if hasattr(label_array, '0'):
                    downsampled_array = label_array['0']
                else:
                    downsampled_array = label_array
            else:
                # Use downsampled level
                if hasattr(label_array, str(downsample_level)):
                    downsampled_array = label_array[str(downsample_level)]
                else:
                    print(f"Warning: Downsample level {downsample_level} not found in {label_name}, using level 0")
                    downsampled_array = label_array['0'] if hasattr(label_array, '0') else label_array
        except Exception as e:
            print(f"Error accessing resolution level {downsample_level} for {label_name}: {e}")
            # Fallback to the array itself
            downsampled_array = label_array
        
        # Set volume-specific bounds (scaled to downsampled resolution)
        vol_min_z = min_z // downsample_factor
        vol_min_y = min_y // downsample_factor
        vol_min_x = min_x // downsample_factor
        vol_max_z = downsampled_array.shape[0] if max_z is None else max_z // downsample_factor
        vol_max_y = downsampled_array.shape[1] if max_y is None else max_y // downsample_factor
        vol_max_x = downsampled_array.shape[2] if max_x is None else max_x // downsample_factor
        
        # Generate possible start positions for this volume (at downsampled resolution)
        dpZ, dpY, dpX = downsampled_patch_size
        z_step = dpZ // 2
        y_step = dpY // 2
        x_step = dpX // 2
        all_positions = []
        for z in range(vol_min_z, vol_max_z - dpZ + 2, z_step):
            for y in range(vol_min_y, vol_max_y - dpY + 2, y_step):
                for x in range(vol_min_x, vol_max_x - dpX + 2, x_step):
                    all_positions.append((z, y, x))
        
        if len(all_positions) == 0:
            print(f"No valid positions found for volume '{label_name}' - skipping")
            continue
        
        chunk_size = max(1, len(all_positions) // (num_workers * 2))
        position_chunks = list(_chunker(all_positions, chunk_size))
        
        # Process patches for this volume
        valid_positions_vol = []
        with Pool(processes=num_workers) as pool:
            results = [
                pool.apply_async(
                    check_patch_chunk,
                    (
                        chunk,
                        downsampled_array,
                        downsampled_patch_size,
                        bbox_threshold,  # pass bounding box threshold
                        label_threshold  # pass label fraction threshold
                    )
                )
                for chunk in position_chunks
            ]
            for r in tqdm(results, 
                         desc=f"Checking patches in {label_name}", 
                         total=len(results),
                         position=1,
                         leave=False):
                valid_positions_vol.extend(r.get())
        
        # Add results with proper volume tracking - scale coordinates back to full resolution
        for (z, y, x) in valid_positions_vol:
            # Scale coordinates back to full resolution
            full_res_z = z * downsample_factor
            full_res_y = y * downsample_factor
            full_res_x = x * downsample_factor
            
            all_valid_patches.append({
                'volume_idx': vol_idx,
                'volume_name': label_name,
                'start_pos': [full_res_z, full_res_y, full_res_x]
            })
        
        print(f"Found {len(valid_positions_vol)} valid patches in '{label_name}'")
    
    # Final summary
    print(f"\nTotal valid patches found across all {len(label_arrays)} volumes: {len(all_valid_patches)}")
    
    return all_valid_patches
