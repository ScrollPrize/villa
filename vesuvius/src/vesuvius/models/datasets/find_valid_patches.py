import logging
import time
from typing import Sequence, Union

import numpy as np
from tqdm import tqdm
import zarr
from numpy.lib.stride_tricks import as_strided

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

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


def reduce_block_to_scalar(
    block: np.ndarray,
    *,
    spatial_ndim: int,
    channel_selector: Union[int, Sequence[int], None],
) -> np.ndarray:
    """Collapse extra channel axes for a larger block slice."""

    arr = np.asarray(block)
    if channel_selector is not None:
        if isinstance(channel_selector, (tuple, list)):
            indices = tuple(int(v) for v in channel_selector)
        else:
            indices = (int(channel_selector),)
        return arr[(...,) + indices]

    if arr.ndim == spatial_ndim:
        return arr

    spatial_shape = arr.shape[:spatial_ndim]
    extra_shape = arr.shape[spatial_ndim:]
    flat = arr.reshape(spatial_shape + (int(np.prod(extra_shape)),))
    return np.linalg.norm(flat, axis=-1)


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

    collapse_selector: Union[int, Sequence[int], None] = channel_selector
    direct_selector = None

    extra_ndim = 0
    try:
        extra_ndim = max(0, sheet_label.ndim - spatial_ndim)
    except AttributeError:
        extra_ndim = 0

    if extra_ndim and isinstance(channel_selector, (tuple, list)) and len(channel_selector) == extra_ndim:
        direct_selector = tuple(int(v) for v in channel_selector)
        collapse_selector = None

    if is_2d:
        pH, pW = patch_size[-2:]
        for (y, x) in chunk:
            base_slice = (slice(y, y + pH), slice(x, x + pW))
            if direct_selector is not None:
                patch = sheet_label[base_slice + direct_selector]
            else:
                patch = sheet_label[base_slice]
            patch = collapse_patch_to_spatial(
                patch,
                spatial_ndim=2,
                channel_selector=collapse_selector,
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
            base_slice = (slice(z, z + pD), slice(y, y + pH), slice(x, x + pW))
            if direct_selector is not None:
                patch = sheet_label[base_slice + direct_selector]
            else:
                patch = sheet_label[base_slice]
            patch = collapse_patch_to_spatial(
                patch,
                spatial_ndim=3,
                channel_selector=collapse_selector,
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

        def _resolve_resolution(array_obj, level_key):
            """Best-effort accessor for multi-resolution zarr groups."""

            key = str(level_key)

            # Attribute access (zarr Groups expose arrays as attributes)
            try:
                candidate = getattr(array_obj, key)
            except AttributeError:
                candidate = None
            except Exception:
                candidate = None
            else:
                if candidate is not None:
                    return candidate

            # Mapping-style / __getitem__ access
            try:
                return array_obj[key]
            except Exception:
                return None

        logger.info(
            "Resolving downsample level %s for volume '%s' (fallback level 0 if unavailable)",
            downsample_level,
            label_name,
        )

        try:
            if downsample_level == 0:
                # Use full resolution
                candidate = _resolve_resolution(label_array, '0')
                downsampled_array = candidate if candidate is not None else label_array
            else:
                candidate = _resolve_resolution(label_array, downsample_level)
                if candidate is not None:
                    downsampled_array = candidate
                    logger.info(
                        "Using downsample level %s for '%s' with patch size %s",
                        downsample_level,
                        label_name,
                        actual_downsampled_patch_size,
                    )
                else:
                    # For non-multi-resolution zarrs, fall back to full resolution
                    candidate_full = _resolve_resolution(label_array, '0')
                    downsampled_array = candidate_full if candidate_full is not None else label_array
                    # Update factors since we're using full resolution
                    actual_downsample_factor = 1
                    actual_downsampled_patch_size = patch_size
                    print(f"Using full resolution for {label_name} (patch size {actual_downsampled_patch_size})")
                    logger.info(
                        "Falling back to full resolution for '%s' (patch size %s)",
                        label_name,
                        actual_downsampled_patch_size,
                    )
        except Exception as e:
            print(f"Error accessing resolution level {downsample_level} for {label_name}: {e}")
            # Fallback to the array itself at full resolution
            downsampled_array = label_array
            actual_downsample_factor = 1
            actual_downsampled_patch_size = patch_size
            print(f"Using full resolution for {label_name} (patch size {actual_downsampled_patch_size})")
            logger.info(
                "Exception while resolving level %s for '%s': %s. Falling back to full resolution.",
                downsample_level,
                label_name,
                e,
            )
        
        # Check if data is 2D or 3D based on patch dimensionality
        spatial_ndim = len(actual_downsampled_patch_size)
        is_2d = spatial_ndim == 2

        # Adjust patch size for 2D data if needed
        if is_2d and len(actual_downsampled_patch_size) == 3:
            # For 2D data with 3D patch size, use last 2 dimensions
            actual_downsampled_patch_size = actual_downsampled_patch_size[-2:]
            print(f"Adjusted patch size for 2D data: {actual_downsampled_patch_size}")
        
        position_gen_start = time.perf_counter()

        spatial_shape = downsampled_array.shape[:spatial_ndim]

        if is_2d:
            vol_min_y = min_y // actual_downsample_factor if min_y is not None else 0
            vol_min_x = min_x // actual_downsample_factor if min_x is not None else 0
            vol_max_y = spatial_shape[0] if max_y is None else max_y // actual_downsample_factor
            vol_max_x = spatial_shape[1] if max_x is None else max_x // actual_downsample_factor

            dpY, dpX = actual_downsampled_patch_size[-2:]
            y_starts = list(range(vol_min_y, max(vol_min_y, vol_max_y - dpY + 1), dpY))
            x_starts = list(range(vol_min_x, max(vol_min_x, vol_max_x - dpX + 1), dpX))
        else:
            vol_min_z = min_z // actual_downsample_factor if min_z is not None else 0
            vol_min_y = min_y // actual_downsample_factor if min_y is not None else 0
            vol_min_x = min_x // actual_downsample_factor if min_x is not None else 0
            vol_max_z = spatial_shape[0] if max_z is None else max_z // actual_downsample_factor
            vol_max_y = spatial_shape[1] if max_y is None else max_y // actual_downsample_factor
            vol_max_x = spatial_shape[2] if max_x is None else max_x // actual_downsample_factor

            dpZ, dpY, dpX = actual_downsampled_patch_size
            z_starts = list(range(vol_min_z, max(vol_min_z, vol_max_z - dpZ + 1), dpZ))
            y_starts = list(range(vol_min_y, max(vol_min_y, vol_max_y - dpY + 1), dpY))
            x_starts = list(range(vol_min_x, max(vol_min_x, vol_max_x - dpX + 1), dpX))

        generate_elapsed = time.perf_counter() - position_gen_start
        candidate_count = (
            len(y_starts) * len(x_starts) if is_2d else len(z_starts) * len(y_starts) * len(x_starts)
        )

        logger.info(
            "Volume '%s': downsampled array shape %s, target patch %s, candidate positions %d (generated in %.2fs)",
            label_name,
            getattr(downsampled_array, 'shape', None),
            actual_downsampled_patch_size,
            candidate_count,
            generate_elapsed,
        )

        if candidate_count == 0:
            print(f"No valid positions found for volume '{label_name}' - skipping")
            continue

        chunk_shape = getattr(downsampled_array, 'chunks', None)
        if not chunk_shape:
            chunk_shape = actual_downsampled_patch_size + (0,) * (max(0, 3 - spatial_ndim))

        patch_volume = int(np.prod(actual_downsampled_patch_size))

        valid_positions_vol = []
        block_start = time.perf_counter()

        if is_2d:
            chunk_y_patches = max(1, chunk_shape[0] // dpY)
            chunk_x_patches = max(1, chunk_shape[1] // dpX)
            chunk_y_patches = min(chunk_y_patches, len(y_starts))
            chunk_x_patches = min(chunk_x_patches, len(x_starts))

            for yi in range(0, len(y_starts), chunk_y_patches):
                y_group = y_starts[yi: yi + chunk_y_patches]
                y_start = y_group[0]
                y_stop = y_group[-1] + dpY

                for xi in range(0, len(x_starts), chunk_x_patches):
                    x_group = x_starts[xi: xi + chunk_x_patches]
                    x_start = x_group[0]
                    x_stop = x_group[-1] + dpX

                    block = downsampled_array[y_start:y_stop, x_start:x_stop]
                    block = reduce_block_to_scalar(
                        block,
                        spatial_ndim=2,
                        channel_selector=selector,
                    )
                    block_mask = np.asarray(block != 0)
                    if not np.any(block_mask):
                        continue

                    block_mask = np.ascontiguousarray(block_mask)
                    y_len = len(y_group)
                    x_len = len(x_group)
                    strides = block_mask.strides
                    patches_view = as_strided(
                        block_mask,
                        shape=(y_len, x_len, dpY, dpX),
                        strides=(strides[0] * dpY, strides[1] * dpX, strides[0], strides[1]),
                        writeable=False,
                    )

                    patches_flat = patches_view.reshape(y_len, x_len, -1)
                    labeled_counts = patches_flat.sum(axis=-1)
                    label_fraction = labeled_counts / patch_volume

                    y_any = patches_view.any(axis=-1)
                    x_any = patches_view.any(axis=-2)

                    y_has = y_any.any(axis=-1)
                    x_has = x_any.any(axis=-1)

                    y_first = np.argmax(y_any, axis=-1)
                    y_last = y_any.shape[-1] - 1 - np.argmax(y_any[..., ::-1], axis=-1)
                    y_width = np.where(y_has, (y_last - y_first + 1), 0)

                    x_first = np.argmax(x_any, axis=-1)
                    x_last = x_any.shape[-1] - 1 - np.argmax(x_any[..., ::-1], axis=-1)
                    x_width = np.where(x_has, (x_last - x_first + 1), 0)

                    bbox_fraction = (y_width * x_width) / patch_volume

                    valid_mask = (label_fraction >= label_threshold) & (bbox_fraction >= bbox_threshold)

                    if not np.any(valid_mask):
                        continue

                    valid_idx = np.argwhere(valid_mask)
                    for (yy, xx) in valid_idx:
                        pos_y = y_group[yy]
                        pos_x = x_group[xx]
                        valid_positions_vol.append((pos_y, pos_x))
        else:
            chunk_z_patches = max(1, chunk_shape[0] // dpZ)
            chunk_y_patches = max(1, chunk_shape[1] // dpY)
            chunk_x_patches = max(1, chunk_shape[2] // dpX)
            chunk_z_patches = min(chunk_z_patches, len(z_starts))
            chunk_y_patches = min(chunk_y_patches, len(y_starts))
            chunk_x_patches = min(chunk_x_patches, len(x_starts))

            for zi in range(0, len(z_starts), chunk_z_patches):
                z_group = z_starts[zi: zi + chunk_z_patches]
                z_start = z_group[0]
                z_stop = z_group[-1] + dpZ

                for yi in range(0, len(y_starts), chunk_y_patches):
                    y_group = y_starts[yi: yi + chunk_y_patches]
                    y_start = y_group[0]
                    y_stop = y_group[-1] + dpY

                    for xi in range(0, len(x_starts), chunk_x_patches):
                        x_group = x_starts[xi: xi + chunk_x_patches]
                        x_start = x_group[0]
                        x_stop = x_group[-1] + dpX

                        block = downsampled_array[z_start:z_stop, y_start:y_stop, x_start:x_stop]
                        block = reduce_block_to_scalar(
                            block,
                            spatial_ndim=3,
                            channel_selector=selector,
                        )
                        block_mask = np.asarray(block != 0)
                        if not np.any(block_mask):
                            continue

                        block_mask = np.ascontiguousarray(block_mask)
                        z_len = len(z_group)
                        y_len = len(y_group)
                        x_len = len(x_group)
                        strides = block_mask.strides
                        patches_view = as_strided(
                            block_mask,
                            shape=(z_len, y_len, x_len, dpZ, dpY, dpX),
                            strides=(
                                strides[0] * dpZ,
                                strides[1] * dpY,
                                strides[2] * dpX,
                                strides[0],
                                strides[1],
                                strides[2],
                            ),
                            writeable=False,
                        )

                        patches_flat = patches_view.reshape(z_len, y_len, x_len, -1)
                        labeled_counts = patches_flat.sum(axis=-1)
                        label_fraction = labeled_counts / patch_volume

                        z_any = patches_view.any(axis=(4, 5))
                        y_any = patches_view.any(axis=(3, 5))
                        x_any = patches_view.any(axis=(3, 4))

                        z_has = z_any.any(axis=-1)
                        y_has = y_any.any(axis=-1)
                        x_has = x_any.any(axis=-1)

                        z_first = np.argmax(z_any, axis=-1)
                        z_last = z_any.shape[-1] - 1 - np.argmax(z_any[..., ::-1], axis=-1)
                        z_width = np.where(z_has, (z_last - z_first + 1), 0)

                        y_first = np.argmax(y_any, axis=-1)
                        y_last = y_any.shape[-1] - 1 - np.argmax(y_any[..., ::-1], axis=-1)
                        y_width = np.where(y_has, (y_last - y_first + 1), 0)

                        x_first = np.argmax(x_any, axis=-1)
                        x_last = x_any.shape[-1] - 1 - np.argmax(x_any[..., ::-1], axis=-1)
                        x_width = np.where(x_has, (x_last - x_first + 1), 0)

                        bbox_fraction = (z_width * y_width * x_width) / patch_volume

                        valid_mask = (label_fraction >= label_threshold) & (
                            bbox_fraction >= bbox_threshold
                        )

                        if not np.any(valid_mask):
                            continue

                        valid_idx = np.argwhere(valid_mask)
                        for (zz, yy, xx) in valid_idx:
                            pos_z = z_group[zz]
                            pos_y = y_group[yy]
                            pos_x = x_group[xx]
                            valid_positions_vol.append((pos_z, pos_y, pos_x))

        elapsed = time.perf_counter() - block_start
        logger.info(
            "Volume '%s': %d valid positions identified (%.2f%% of candidates) in %.2fs",
            label_name,
            len(valid_positions_vol),
            (len(valid_positions_vol) / candidate_count * 100.0) if candidate_count else 0.0,
            elapsed,
        )
        
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
