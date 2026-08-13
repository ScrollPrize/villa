import numpy as np

def arr_to_f32_zyx(arr, axis_order):
    if not (3 <= arr.ndim <= 4):
        raise ValueError(f"Array must be 3D or 4D, got {arr.ndim}D")

    original_dtype = arr.dtype
    arr = arr.astype(np.float32)



    current_axes = axis_order.lower().split(',')
    target_axes = ['z', 'y', 'x']

    if 'c' in current_axes:
        if arr.ndim != 4:
            raise ValueError("Array must be 4D when channel dimension is specified")
        target_axes = ['c'] + target_axes
    elif arr.ndim == 4:
        raise ValueError("Must specify 'c' in axis_order for 4D arrays")

    transpose_order = [current_axes.index(ax) for ax in target_axes]
    arr = np.transpose(arr, transpose_order)

    # Drop channel dimension if it equals 1
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)

    return arr

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import fsspec
import zarr

def chunker(seq, chunk_size):
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


def find_valid_patches(
        target_array,
        patch_size,
        bbox_threshold=0.97,  # bounding-box coverage fraction
        label_threshold=0.10,  # minimum % of voxels labeled
        num_workers=4,
        overlap_fraction=0.25,
        axis_order='zyx'
):
    """
    Finds patches that have:
      - A bounding box of labeled voxels >= bbox_threshold fraction of the patch volume
      - Overall labeled voxel fraction >= label_threshold

    Parameters:
        target_array: Input array to find patches in
        patch_size: Size of patches to extract. Should match the dimension order
        bbox_threshold: Minimum fraction of bounding box that must be covered
        label_threshold: Minimum fraction of voxels that must be labeled
        num_workers: Number of parallel workers
        overlap_fraction: Fraction of overlap between patches
        axis_order: Order of dimensions in the input array ('zyx' or 'xyz')

    Returns:
        List of dictionaries with 'volume_idx' and 'start_pos', where start_pos is always in ZYX order
    """
    # Validate and normalize axis order
    axis_order = axis_order.lower()
    if axis_order not in ['zyx', 'xyz']:
        raise ValueError("axis_order must be either 'zyx' or 'xyz'")

    # Create dimension mapping
    dim_map = {
        'zyx': {'z': 0, 'y': 1, 'x': 2},
        'xyz': {'z': 2, 'y': 1, 'x': 0}
    }[axis_order]

    # Create inverse mapping for converting back to ZYX
    inverse_map = {v: i for i, (k, v) in enumerate(dim_map.items())}

    # Extract patch dimensions based on axis order
    if len(patch_size) == 3:
        patch_dims = {
            'z': patch_size[dim_map['z']],
            'y': patch_size[dim_map['y']],
            'x': patch_size[dim_map['x']]
        }
        pZ, pY, pX = patch_dims['z'], patch_dims['y'], patch_dims['x']
    else:
        # Assume patch_size includes batch/channel dimensions
        patch_dims = {
            'z': patch_size[-3 + dim_map['z']],
            'y': patch_size[-3 + dim_map['y']],
            'x': patch_size[-3 + dim_map['x']]
        }
        pZ, pY, pX = patch_dims['z'], patch_dims['y'], patch_dims['x']

    # Extract array dimensions based on axis order
    array_dims = {
        'z': target_array.shape[-3 + dim_map['z']],
        'y': target_array.shape[-3 + dim_map['y']],
        'x': target_array.shape[-3 + dim_map['x']]
    }
    Zdim, Ydim, Xdim = array_dims['z'], array_dims['y'], array_dims['x']

    # Convert overlap fraction -> step (stride)
    z_stride = max(1, int(pZ * (1 - overlap_fraction)))
    y_stride = max(1, int(pY * (1 - overlap_fraction)))
    x_stride = max(1, int(pX * (1 - overlap_fraction)))

    # Generate all possible positions
    all_positions = []
    for z in range(0, Zdim - pZ + 1, z_stride):
        for y in range(0, Ydim - pY + 1, y_stride):
            for x in range(0, Xdim - pX + 1, x_stride):
                # Create position in input array's order for patch checking
                pos_input_order = [0, 0, 0]
                pos_input_order[dim_map['z']] = z
                pos_input_order[dim_map['y']] = y
                pos_input_order[dim_map['x']] = x
                all_positions.append(tuple(pos_input_order))

    chunk_size = max(1, len(all_positions) // (num_workers * 2))
    position_chunks = list(chunker(all_positions, chunk_size))

    print(
        f"Finding valid patches of size: {patch_size} "
        f"with bounding box coverage >= {bbox_threshold} and labeled fraction >= {label_threshold}."
    )

    valid_positions_ref = []
    with Pool(processes=num_workers) as pool:
        results = [
            pool.apply_async(
                check_patch_chunk,
                (
                    chunk,
                    target_array,
                    patch_size,
                    bbox_threshold,
                    label_threshold
                )
            )
            for chunk in position_chunks
        ]
        for r in tqdm(results, desc="Checking patches", total=len(results)):
            valid_positions_ref.extend(r.get())

    # Convert positions back to ZYX order for output
    valid_patches = []
    for pos in valid_positions_ref:
        # Convert from input order to ZYX order
        pos_zyx = [0, 0, 0]
        for i, coord in enumerate(pos):
            zyx_index = inverse_map[i]
            pos_zyx[zyx_index] = coord

        valid_patches.append({'volume_idx': 0, 'start_pos': pos_zyx})

    print(
        f"Found {len(valid_positions_ref)} valid patches in reference volume. "
        f"Total {len(valid_patches)} across all volumes."
    )

    return valid_patches

