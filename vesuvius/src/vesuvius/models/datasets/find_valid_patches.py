from typing import Sequence, Union

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
    Given a 2D or 3D boolean array (True where labeled, False otherwise),
    returns bounding box coordinates.
    For 3D: (minz, maxz, miny, maxy, minx, maxx)
    For 2D: (miny, maxy, minx, maxx)
    If there are no nonzero elements, returns None.
    """
    nonzero_coords = np.argwhere(mask)
    if nonzero_coords.size == 0:
        return None

    if len(mask.shape) == 3:
        minz, miny, minx = nonzero_coords.min(axis=0)
        maxz, maxy, maxx = nonzero_coords.max(axis=0)
        return (minz, maxz, miny, maxy, minx, maxx)
    else:  # 2D
        miny, minx = nonzero_coords.min(axis=0)
        maxy, maxx = nonzero_coords.max(axis=0)
        return (miny, maxy, minx, maxx)

def bounding_box_volume(bbox):
    """
    Given a bounding box, returns the volume/area (number of voxels/pixels) inside the box.
    For 3D: bbox = (minz, maxz, miny, maxy, minx, maxx)
    For 2D: bbox = (miny, maxy, minx, maxx)
    """
    if len(bbox) == 6:
        # 3D
        minz, maxz, miny, maxy, minx, maxx = bbox
        return ((maxz - minz + 1) *
                (maxy - miny + 1) *
                (maxx - minx + 1))
    else:
        # 2D
        miny, maxy, minx, maxx = bbox
        return ((maxy - miny + 1) *
                (maxx - minx + 1))

def _resolve_channel_index(selector: Union[int, Sequence[int]], extra_shape: Sequence[int]) -> int:
    """Convert a channel selector description into a flattened index."""

    total = int(np.prod(extra_shape)) if extra_shape else 1

    if isinstance(selector, int):
        idx = int(selector)
    elif isinstance(selector, (tuple, list)):
        if len(selector) != len(extra_shape):
            raise ValueError(
                "Channel selector dimensionality does not match channel axes"
            )
        idx = 0
        for sel, size in zip(selector, extra_shape):
            sel_int = int(sel)
            if sel_int < 0:
                sel_int += int(size)
            if not (0 <= sel_int < int(size)):
                raise ValueError(
                    f"Channel selector index {sel} out of bounds for axis size {size}"
                )
            idx = idx * int(size) + sel_int
    else:
        raise TypeError("Channel selector must be an int or a sequence of ints")

    if idx < 0:
        idx += total
    if not (0 <= idx < total):
        raise ValueError(
            f"Resolved channel index {idx} out of bounds for flattened size {total}"
        )
    return idx


def collapse_patch_to_spatial(
    patch: np.ndarray,
    *,
    spatial_ndim: int,
    channel_selector: Union[int, Sequence[int], None],
) -> np.ndarray:
    """Reduce a patch with extra channel axes down to spatial-only values."""

    arr = np.asarray(patch)
    if arr.ndim < spatial_ndim:
        raise ValueError(
            f"Patch ndim {arr.ndim} is incompatible with spatial dimensions {spatial_ndim}"
        )

    if arr.ndim == spatial_ndim:
        return arr

    spatial_shape = arr.shape[:spatial_ndim]
    extra_shape = arr.shape[spatial_ndim:]
    flat = arr.reshape(spatial_shape + (int(np.prod(extra_shape)),))

    if channel_selector is None:
        return np.linalg.norm(flat, axis=-1)

    idx = _resolve_channel_index(channel_selector, extra_shape)
    return flat[..., idx]


def check_patch_chunk(
    chunk,
    sheet_label,
    patch_size,
    bbox_threshold=0.5,
    label_threshold=0.05,
    channel_selector: Union[int, Sequence[int], None] = None,
):
    """Identify valid label patches within a chunk of candidate positions."""

    spatial_ndim = len(patch_size)
    is_2d = spatial_ndim == 2
    valid_positions = []

    if is_2d:
        pH, pW = patch_size[-2:]
        for (y, x) in chunk:
            patch = sheet_label[y:y + pH, x:x + pW]
            patch = collapse_patch_to_spatial(
                patch,
                spatial_ndim=2,
                channel_selector=channel_selector,
            )
            mask = np.abs(patch) > 0
            bbox = compute_bounding_box_3d(mask)
            if bbox is None:
                continue

            bb_vol = bounding_box_volume(bbox)
            patch_vol = patch.size
            if bb_vol / patch_vol < bbox_threshold:
                continue

            labeled_ratio = np.count_nonzero(mask) / patch_vol
            if labeled_ratio < label_threshold:
                continue

            valid_positions.append((y, x))
    else:
        pD, pH, pW = patch_size
        for (z, y, x) in chunk:
            patch = sheet_label[z:z + pD, y:y + pH, x:x + pW]
            patch = collapse_patch_to_spatial(
                patch,
                spatial_ndim=3,
                channel_selector=channel_selector,
            )
            mask = np.abs(patch) > 0
            bbox = compute_bounding_box_3d(mask)
            if bbox is None:
                continue

            bb_vol = bounding_box_volume(bbox)
            patch_vol = patch.size
            if bb_vol / patch_vol < bbox_threshold:
                continue

            labeled_ratio = np.count_nonzero(mask) / patch_vol
            if labeled_ratio < label_threshold:
                continue

            valid_positions.append((z, y, x))

    return valid_positions

def find_valid_patches(
    label_arrays,
    label_names,
    patch_size,
    bbox_threshold=0.97,  # bounding-box coverage fraction
    label_threshold=0.10,  # minimum % of voxels labeled,
    min_z=0,
    min_y=0,
    min_x=0,
    max_z=None,
    max_y=None,
    max_x=None,
    num_workers=4,
    downsample_level=1,
    channel_selectors: Sequence[Union[int, Sequence[int], None]] | None = None,
):
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
    
    all_valid_patches = []
    
    # Calculate downsampled patch size
    downsample_factor = 2 ** downsample_level
    downsampled_patch_size = tuple(p // downsample_factor for p in patch_size)
    
    if downsample_level == 0:
        print(
            f"Finding valid patches of size: {patch_size} at full resolution "
            f"with bounding box coverage >= {bbox_threshold} and labeled fraction >= {label_threshold}."
        )
    else:
        print(
            f"Finding valid patches with bounding box coverage >= {bbox_threshold} and labeled fraction >= {label_threshold}.\n"
            f"Target patch size: {patch_size} (full resolution)\n"
            f"Will attempt to use downsample level {downsample_level} for faster processing (would use patch size {downsampled_patch_size})"
        )
    
    # Outer progress bar for volumes
    if channel_selectors is not None and len(channel_selectors) != len(label_arrays):
        raise ValueError("channel_selectors must match number of label arrays")

    for vol_idx, (label_array, label_name) in enumerate(tqdm(
        zip(label_arrays, label_names), 
        total=len(label_arrays),
        desc="Processing volumes",
        position=0
    )):
        print(f"\nProcessing volume '{label_name}' ({vol_idx + 1}/{len(label_arrays)})")

        selector = None
        if channel_selectors is not None:
            selector = channel_selectors[vol_idx]

        # Access the appropriate resolution level for patch finding
        actual_downsample_factor = downsample_factor
        actual_downsampled_patch_size = downsampled_patch_size

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
                    # For non-multi-resolution zarrs, fall back to full resolution
                    downsampled_array = label_array['0'] if hasattr(label_array, '0') else label_array
                    # Update factors since we're using full resolution
                    actual_downsample_factor = 1
                    actual_downsampled_patch_size = patch_size
                    print(f"Using full resolution for {label_name} (patch size {actual_downsampled_patch_size})")
        except Exception as e:
            print(f"Error accessing resolution level {downsample_level} for {label_name}: {e}")
            # Fallback to the array itself at full resolution
            downsampled_array = label_array
            actual_downsample_factor = 1
            actual_downsampled_patch_size = patch_size
            print(f"Using full resolution for {label_name} (patch size {actual_downsampled_patch_size})")
        
        # Check if data is 2D or 3D based on patch dimensionality
        spatial_ndim = len(actual_downsampled_patch_size)
        is_2d = spatial_ndim == 2

        # Adjust patch size for 2D data if needed
        if is_2d and len(actual_downsampled_patch_size) == 3:
            # For 2D data with 3D patch size, use last 2 dimensions
            actual_downsampled_patch_size = actual_downsampled_patch_size[-2:]
            print(f"Adjusted patch size for 2D data: {actual_downsampled_patch_size}")
        
        if is_2d:
            # For 2D data, we only have y and x dimensions
            vol_min_y = min_y // actual_downsample_factor if min_y is not None else 0
            vol_min_x = min_x // actual_downsample_factor if min_x is not None else 0
            spatial_shape = downsampled_array.shape[:2]
            vol_max_y = spatial_shape[0] if max_y is None else max_y // actual_downsample_factor
            vol_max_x = spatial_shape[1] if max_x is None else max_x // actual_downsample_factor
            
            # Generate possible start positions for 2D data
            dpY, dpX = actual_downsampled_patch_size[-2:]  # Take last two dimensions
            y_step = dpY  # Use full patch size for stepping
            x_step = dpX  # Use full patch size for stepping
            all_positions = []
            for y in range(vol_min_y, vol_max_y - dpY + 2, y_step):
                for x in range(vol_min_x, vol_max_x - dpX + 2, x_step):
                    all_positions.append((y, x))
        else:
            # For 3D data (existing logic)
            vol_min_z = min_z // actual_downsample_factor if min_z is not None else 0
            vol_min_y = min_y // actual_downsample_factor if min_y is not None else 0
            vol_min_x = min_x // actual_downsample_factor if min_x is not None else 0
            spatial_shape = downsampled_array.shape[:3]
            vol_max_z = spatial_shape[0] if max_z is None else max_z // actual_downsample_factor
            vol_max_y = spatial_shape[1] if max_y is None else max_y // actual_downsample_factor
            vol_max_x = spatial_shape[2] if max_x is None else max_x // actual_downsample_factor
            
            # Generate possible start positions for this volume (at downsampled resolution)
            dpZ, dpY, dpX = actual_downsampled_patch_size
            z_step = dpZ  # Use full patch size for stepping
            y_step = dpY  # Use full patch size for stepping
            x_step = dpX  # Use full patch size for stepping
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
                        actual_downsampled_patch_size,
                        bbox_threshold,  # pass bounding box threshold
                        label_threshold,  # pass label fraction threshold
                        selector,
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
        for pos in valid_positions_vol:
            if is_2d:
                # 2D position (y, x)
                y, x = pos
                full_res_y = y * actual_downsample_factor
                full_res_x = x * actual_downsample_factor
                
                all_valid_patches.append({
                    'volume_idx': vol_idx,
                    'volume_name': label_name,
                    'start_pos': [full_res_y, full_res_x]
                })
            else:
                # 3D position (z, y, x)
                z, y, x = pos
                full_res_z = z * actual_downsample_factor
                full_res_y = y * actual_downsample_factor
                full_res_x = x * actual_downsample_factor
                
                all_valid_patches.append({
                    'volume_idx': vol_idx,
                    'volume_name': label_name,
                    'start_pos': [full_res_z, full_res_y, full_res_x]
                })
        
        print(f"Found {len(valid_positions_vol)} valid patches in '{label_name}'")
    
    # Final summary
    print(f"\nTotal valid patches found across all {len(label_arrays)} volumes: {len(all_valid_patches)}")
    
    return all_valid_patches
