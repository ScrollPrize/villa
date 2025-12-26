import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

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

def zero_ignore_labels(array: np.ndarray, ignore_label: Union[int, float]) -> np.ndarray:
    """Return a copy of ``array`` with the ignore label value zeroed out."""

    arr = np.asarray(array)
    if arr.size == 0:
        return arr

    if isinstance(ignore_label, float) and np.isnan(ignore_label):
        mask = np.isnan(arr)
    else:
        try:
            mask = (arr == ignore_label)
        except TypeError:
            return arr

    if not np.any(mask):
        return arr

    result = arr.copy()
    result[mask] = 0
    return result


def check_patch_chunk(
    chunk,
    sheet_label,
    patch_size,
    bbox_threshold=0.5,
    label_threshold=0.05,
    channel_selector: Union[int, Sequence[int], None] = None,
    ignore_label: Optional[Union[int, float]] = None,
    valid_patch_value: Optional[Union[int, float]] = None,
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
            if ignore_label is not None:
                patch = zero_ignore_labels(patch, ignore_label)
            patch = collapse_patch_to_spatial(
                patch,
                spatial_ndim=2,
                channel_selector=collapse_selector,
            )
            if valid_patch_value is not None:
                mask = (patch == valid_patch_value)
            else:
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
            if ignore_label is not None:
                patch = zero_ignore_labels(patch, ignore_label)
            patch = collapse_patch_to_spatial(
                patch,
                spatial_ndim=3,
                channel_selector=collapse_selector,
            )
            if valid_patch_value is not None:
                mask = (patch == valid_patch_value)
            else:
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


def _collect_unlabeled_fg_from_image_only(
    image_array,
    vol_idx: int,
    label_name: str,
    patch_size: Tuple[int, ...],
    valid_patch_find_resolution: int,
    downsample_factor: int,
    downsampled_patch_size: Tuple[int, ...],
    min_z: int,
    min_y: int,
    min_x: int,
    max_z: Optional[int],
    max_y: Optional[int],
    max_x: Optional[int],
    unlabeled_fg_threshold: float,
    unlabeled_fg_bbox_threshold: float,
) -> List[Dict[str, object]]:
    """
    Collect unlabeled foreground patches from a volume with no labels.

    This function is used for self-supervised training where volumes have
    image data but no labels. It filters patches by image content only.

    Returns:
        List of patch dictionaries with 'volume_idx', 'volume_name', 'start_pos'
    """
    unlabeled_fg_patches = []

    def _resolve_resolution(array_obj, level_key):
        """Access resolution level from a zarr group or return array directly."""
        key = str(level_key)
        if hasattr(array_obj, 'shape') and hasattr(array_obj, 'dtype'):
            if not hasattr(array_obj, 'keys'):
                return array_obj
        try:
            candidate = array_obj[key]
            if hasattr(candidate, 'shape'):
                return candidate
        except Exception:
            pass
        return None

    # Resolve image array at appropriate resolution level
    actual_downsample_factor = downsample_factor
    actual_downsampled_patch_size = downsampled_patch_size

    try:
        candidate = _resolve_resolution(image_array, valid_patch_find_resolution)
        if candidate is not None:
            downsampled_image = candidate
            logger.info(
                "Using image level %s for unlabeled volume '%s' with patch size %s",
                valid_patch_find_resolution,
                label_name,
                actual_downsampled_patch_size,
            )
        else:
            candidate_full = _resolve_resolution(image_array, '0')
            if candidate_full is not None:
                downsampled_image = candidate_full
            else:
                downsampled_image = image_array
            actual_downsample_factor = 1
            actual_downsampled_patch_size = patch_size
            logger.info(
                "Image level %s unavailable for '%s', using full resolution",
                valid_patch_find_resolution,
                label_name,
            )
    except Exception as e:
        logger.warning(
            "Error resolving image level %s for '%s': %s. Using array directly.",
            valid_patch_find_resolution, label_name, e,
        )
        downsampled_image = image_array
        actual_downsample_factor = 1
        actual_downsampled_patch_size = patch_size

    if downsampled_image is None:
        logger.warning("Volume '%s': no usable image data; skipping", label_name)
        return []

    # Determine spatial dimensionality
    spatial_ndim = len(actual_downsampled_patch_size)
    is_2d = spatial_ndim == 2

    # Get spatial shape from image
    image_shape = downsampled_image.shape
    spatial_shape = image_shape[:spatial_ndim] if len(image_shape) >= spatial_ndim else image_shape

    # Generate candidate positions
    if is_2d:
        dpY, dpX = actual_downsampled_patch_size[-2:]
        vol_min_y = min_y // actual_downsample_factor if min_y is not None else 0
        vol_min_x = min_x // actual_downsample_factor if min_x is not None else 0
        vol_max_y = spatial_shape[0] if max_y is None else max_y // actual_downsample_factor
        vol_max_x = spatial_shape[1] if max_x is None else max_x // actual_downsample_factor

        y_starts = list(range(vol_min_y, max(vol_min_y, vol_max_y - dpY + 1), dpY))
        x_starts = list(range(vol_min_x, max(vol_min_x, vol_max_x - dpX + 1), dpX))
        candidate_count = len(y_starts) * len(x_starts)
    else:
        dpZ, dpY, dpX = actual_downsampled_patch_size
        vol_min_z = min_z // actual_downsample_factor if min_z is not None else 0
        vol_min_y = min_y // actual_downsample_factor if min_y is not None else 0
        vol_min_x = min_x // actual_downsample_factor if min_x is not None else 0
        vol_max_z = spatial_shape[0] if max_z is None else max_z // actual_downsample_factor
        vol_max_y = spatial_shape[1] if max_y is None else max_y // actual_downsample_factor
        vol_max_x = spatial_shape[2] if max_x is None else max_x // actual_downsample_factor

        z_starts = list(range(vol_min_z, max(vol_min_z, vol_max_z - dpZ + 1), dpZ))
        y_starts = list(range(vol_min_y, max(vol_min_y, vol_max_y - dpY + 1), dpY))
        x_starts = list(range(vol_min_x, max(vol_min_x, vol_max_x - dpX + 1), dpX))
        candidate_count = len(z_starts) * len(y_starts) * len(x_starts)

    logger.info(
        "Unlabeled volume '%s': image shape %s, target patch %s, candidate positions %d",
        label_name,
        getattr(downsampled_image, 'shape', None),
        actual_downsampled_patch_size,
        candidate_count,
    )

    if candidate_count == 0:
        logger.info("No valid positions found for unlabeled volume '%s'", label_name)
        return []

    patch_volume = int(np.prod(actual_downsampled_patch_size))
    chunk_shape = getattr(downsampled_image, 'chunks', None)
    if not chunk_shape:
        chunk_shape = actual_downsampled_patch_size + (0,) * (max(0, 3 - spatial_ndim))

    block_start = time.perf_counter()

    if is_2d:
        chunk_y_patches = max(1, chunk_shape[0] // dpY)
        chunk_x_patches = max(1, chunk_shape[1] // dpX)
        chunk_y_patches = min(chunk_y_patches, len(y_starts))
        chunk_x_patches = min(chunk_x_patches, len(x_starts))

        y_blocks = list(range(0, len(y_starts), chunk_y_patches))
        for yi in tqdm(y_blocks, desc=f"  {label_name} (unlabeled)", position=1, leave=False):
            y_group = y_starts[yi: yi + chunk_y_patches]
            y_start = y_group[0]
            y_stop = y_group[-1] + dpY

            for xi in range(0, len(x_starts), chunk_x_patches):
                x_group = x_starts[xi: xi + chunk_x_patches]
                x_start = x_group[0]
                x_stop = x_group[-1] + dpX

                image_block = np.asarray(downsampled_image[y_start:y_stop, x_start:x_stop])
                if image_block.ndim > 2:
                    image_block = image_block.reshape(image_block.shape[-2:])

                image_nonzero = np.asarray(image_block != 0)
                if not np.any(image_nonzero):
                    continue

                image_nonzero = np.ascontiguousarray(image_nonzero)
                y_len = len(y_group)
                x_len = len(x_group)

                img_patches_view = as_strided(
                    image_nonzero,
                    shape=(y_len, x_len, dpY, dpX),
                    strides=(
                        image_nonzero.strides[0] * dpY,
                        image_nonzero.strides[1] * dpX,
                        image_nonzero.strides[0],
                        image_nonzero.strides[1],
                    ),
                    writeable=False,
                )
                img_patches_flat = img_patches_view.reshape(y_len, x_len, -1)
                img_nonzero_counts = img_patches_flat.sum(axis=-1)
                img_fraction = img_nonzero_counts / patch_volume

                # Compute image bbox coverage (2D)
                img_y_any = img_patches_view.any(axis=-1)
                img_x_any = img_patches_view.any(axis=-2)
                img_y_has = img_y_any.any(axis=-1)
                img_x_has = img_x_any.any(axis=-1)
                img_y_first = np.argmax(img_y_any, axis=-1)
                img_y_last = img_y_any.shape[-1] - 1 - np.argmax(img_y_any[..., ::-1], axis=-1)
                img_y_width = np.where(img_y_has, (img_y_last - img_y_first + 1), 0)
                img_x_first = np.argmax(img_x_any, axis=-1)
                img_x_last = img_x_any.shape[-1] - 1 - np.argmax(img_x_any[..., ::-1], axis=-1)
                img_x_width = np.where(img_x_has, (img_x_last - img_x_first + 1), 0)
                img_bbox_fraction = (img_y_width * img_x_width) / patch_volume

                # Find patches meeting thresholds
                valid_mask = (
                    (img_fraction >= unlabeled_fg_threshold) &
                    (img_bbox_fraction >= unlabeled_fg_bbox_threshold)
                )

                if np.any(valid_mask):
                    valid_idx = np.argwhere(valid_mask)
                    for (yy, xx) in valid_idx:
                        pos_y = y_group[yy]
                        pos_x = x_group[xx]
                        full_res_y = pos_y * actual_downsample_factor
                        full_res_x = pos_x * actual_downsample_factor
                        unlabeled_fg_patches.append({
                            'volume_idx': vol_idx,
                            'volume_name': label_name,
                            'start_pos': [full_res_y, full_res_x],
                        })
    else:
        # 3D case
        chunk_z_patches = max(1, chunk_shape[0] // dpZ)
        chunk_y_patches = max(1, chunk_shape[1] // dpY)
        chunk_x_patches = max(1, chunk_shape[2] // dpX)
        chunk_z_patches = min(chunk_z_patches, len(z_starts))
        chunk_y_patches = min(chunk_y_patches, len(y_starts))
        chunk_x_patches = min(chunk_x_patches, len(x_starts))

        z_blocks = list(range(0, len(z_starts), chunk_z_patches))
        for zi in tqdm(z_blocks, desc=f"  {label_name} (unlabeled)", position=1, leave=False):
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

                    image_block = np.asarray(
                        downsampled_image[z_start:z_stop, y_start:y_stop, x_start:x_stop]
                    )
                    if image_block.ndim > 3:
                        image_block = image_block.reshape(image_block.shape[-3:])

                    image_nonzero = np.asarray(image_block != 0)
                    if not np.any(image_nonzero):
                        continue

                    image_nonzero = np.ascontiguousarray(image_nonzero)
                    z_len = len(z_group)
                    y_len = len(y_group)
                    x_len = len(x_group)

                    img_patches_view = as_strided(
                        image_nonzero,
                        shape=(z_len, y_len, x_len, dpZ, dpY, dpX),
                        strides=(
                            image_nonzero.strides[0] * dpZ,
                            image_nonzero.strides[1] * dpY,
                            image_nonzero.strides[2] * dpX,
                            image_nonzero.strides[0],
                            image_nonzero.strides[1],
                            image_nonzero.strides[2],
                        ),
                        writeable=False,
                    )
                    img_patches_flat = img_patches_view.reshape(z_len, y_len, x_len, -1)
                    img_nonzero_counts = img_patches_flat.sum(axis=-1)
                    img_fraction = img_nonzero_counts / patch_volume

                    # Compute image bbox coverage (3D)
                    img_z_any = img_patches_view.any(axis=(4, 5))
                    img_y_any = img_patches_view.any(axis=(3, 5))
                    img_x_any = img_patches_view.any(axis=(3, 4))
                    img_z_has = img_z_any.any(axis=-1)
                    img_y_has = img_y_any.any(axis=-1)
                    img_x_has = img_x_any.any(axis=-1)
                    img_z_first = np.argmax(img_z_any, axis=-1)
                    img_z_last = img_z_any.shape[-1] - 1 - np.argmax(img_z_any[..., ::-1], axis=-1)
                    img_z_width = np.where(img_z_has, (img_z_last - img_z_first + 1), 0)
                    img_y_first = np.argmax(img_y_any, axis=-1)
                    img_y_last = img_y_any.shape[-1] - 1 - np.argmax(img_y_any[..., ::-1], axis=-1)
                    img_y_width = np.where(img_y_has, (img_y_last - img_y_first + 1), 0)
                    img_x_first = np.argmax(img_x_any, axis=-1)
                    img_x_last = img_x_any.shape[-1] - 1 - np.argmax(img_x_any[..., ::-1], axis=-1)
                    img_x_width = np.where(img_x_has, (img_x_last - img_x_first + 1), 0)
                    img_bbox_fraction = (img_z_width * img_y_width * img_x_width) / patch_volume

                    # Find patches meeting thresholds
                    valid_mask = (
                        (img_fraction >= unlabeled_fg_threshold) &
                        (img_bbox_fraction >= unlabeled_fg_bbox_threshold)
                    )

                    if np.any(valid_mask):
                        valid_idx = np.argwhere(valid_mask)
                        for (zz, yy, xx) in valid_idx:
                            pos_z = z_group[zz]
                            pos_y = y_group[yy]
                            pos_x = x_group[xx]
                            full_res_z = pos_z * actual_downsample_factor
                            full_res_y = pos_y * actual_downsample_factor
                            full_res_x = pos_x * actual_downsample_factor
                            unlabeled_fg_patches.append({
                                'volume_idx': vol_idx,
                                'volume_name': label_name,
                                'start_pos': [full_res_z, full_res_y, full_res_x],
                            })

    elapsed = time.perf_counter() - block_start
    logger.info(
        "Unlabeled volume '%s': %d unlabeled FG patches identified (%.2f%% of candidates) in %.2fs",
        label_name,
        len(unlabeled_fg_patches),
        (len(unlabeled_fg_patches) / candidate_count * 100.0) if candidate_count else 0.0,
        elapsed,
    )

    return unlabeled_fg_patches


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
    valid_patch_find_resolution=1,
    channel_selectors: Sequence[Union[int, Sequence[int], None]] | None = None,
    ignore_labels: Sequence[Optional[Union[int, float]]] | None = None,
    valid_patch_values: Sequence[Optional[Union[int, float]]] | None = None,
    collect_bg_only: bool = False,
    # Unlabeled foreground detection for semi-supervised learning
    image_arrays: Sequence[object] | None = None,
    collect_unlabeled_fg: bool = False,
    unlabeled_fg_threshold: float = 0.05,  # Min fraction of non-zero image voxels
    unlabeled_fg_bbox_threshold: float = 0.15,  # Min bbox coverage for image data
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
        valid_patch_find_resolution: Resolution level to use for patch finding (0=full res, 1=2x downsample, etc.)
        ignore_labels: Optional per-volume ignore values that should be treated as background.
        valid_patch_values: Optional per-volume values to use for valid patch detection.
            When set, only voxels matching this value count as labeled (instead of all non-zero).
        collect_bg_only: When True and ignore_label is set, collect background-only patches
            (patches where all labeled voxels are 0 after masking ignore labels).
        image_arrays: Optional list of zarr arrays (image volumes) for unlabeled foreground detection.
            Must match length of label_arrays when provided.
        collect_unlabeled_fg: When True and image_arrays provided, collect patches that are
            effectively unlabeled (no valid labels after ignore masking) but have image data.
            These are used for semi-supervised learning (unlabeled samples with data).
        unlabeled_fg_threshold: Minimum fraction of non-zero image voxels for a patch to qualify
            as unlabeled foreground.
        unlabeled_fg_bbox_threshold: Minimum bounding box coverage of non-zero image voxels.

    Returns:
        Dictionary with 'fg_patches', 'bg_patches', and 'unlabeled_fg_patches' lists,
        each containing dictionaries with 'volume_idx', 'volume_name', and 'start_pos'
        (coordinates at full resolution)
    """
    if len(label_arrays) != len(label_names):
        raise ValueError("Number of label arrays must match number of label names")
    if ignore_labels is not None and len(ignore_labels) != len(label_arrays):
        raise ValueError("ignore_labels must match number of label arrays when provided")
    if valid_patch_values is not None and len(valid_patch_values) != len(label_arrays):
        raise ValueError("valid_patch_values must match number of label arrays when provided")
    if collect_unlabeled_fg and image_arrays is None:
        raise ValueError("image_arrays must be provided when collect_unlabeled_fg=True")
    if image_arrays is not None and len(image_arrays) != len(label_arrays):
        raise ValueError("image_arrays must match number of label arrays when provided")

    all_valid_patches = []
    all_bg_patches = []
    all_unlabeled_fg_patches = []
    
    # Calculate downsampled patch size
    downsample_factor = 2 ** valid_patch_find_resolution
    downsampled_patch_size = tuple(p // downsample_factor for p in patch_size)
    
    if valid_patch_find_resolution == 0:
        print(
            f"Finding valid patches of size: {patch_size} at full resolution "
            f"with bounding box coverage >= {bbox_threshold} and labeled fraction >= {label_threshold}."
        )
    else:
        print(
            f"Finding valid patches with bounding box coverage >= {bbox_threshold} and labeled fraction >= {label_threshold}.\n"
            f"Target patch size: {patch_size} (full resolution)\n"
            f"Will attempt to use downsample level {valid_patch_find_resolution} for faster processing (would use patch size {downsampled_patch_size})"
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
        ignore_label = None
        if ignore_labels is not None:
            ignore_label = ignore_labels[vol_idx]
        valid_patch_value = None
        if valid_patch_values is not None:
            valid_patch_value = valid_patch_values[vol_idx]

        # Get image array for unlabeled foreground detection
        image_array = None
        if image_arrays is not None:
            image_array = image_arrays[vol_idx]

        if label_array is None:
            # When there are no labels, we can still collect unlabeled FG patches
            # by checking image content only
            if collect_unlabeled_fg and image_array is not None:
                logger.info(
                    "Volume '%s' has no labels at index %d; collecting unlabeled FG patches from image data",
                    label_name,
                    vol_idx,
                )
                # Debug: check what type of object image_array is
                logger.info(
                    "DEBUG: image_array type=%s, hasattr(keys)=%s, hasattr(shape)=%s",
                    type(image_array).__name__,
                    hasattr(image_array, 'keys'),
                    hasattr(image_array, 'shape'),
                )
                if hasattr(image_array, 'keys'):
                    logger.info("DEBUG: image_array keys=%s", list(image_array.keys())[:6])
                if hasattr(image_array, 'shape'):
                    logger.info("DEBUG: image_array shape=%s", image_array.shape)
                # Process this volume using only image data for unlabeled FG detection
                # All patches are considered unlabeled since there are no labels
                unlabeled_only_patches = _collect_unlabeled_fg_from_image_only(
                    image_array=image_array,
                    vol_idx=vol_idx,
                    label_name=label_name,
                    patch_size=patch_size,
                    valid_patch_find_resolution=valid_patch_find_resolution,
                    downsample_factor=downsample_factor,
                    downsampled_patch_size=downsampled_patch_size,
                    min_z=min_z, min_y=min_y, min_x=min_x,
                    max_z=max_z, max_y=max_y, max_x=max_x,
                    unlabeled_fg_threshold=unlabeled_fg_threshold,
                    unlabeled_fg_bbox_threshold=unlabeled_fg_bbox_threshold,
                )
                all_unlabeled_fg_patches.extend(unlabeled_only_patches)
                print(f"Found {len(unlabeled_only_patches)} unlabeled foreground patches in '{label_name}'")
                continue
            else:
                logger.warning(
                    "Volume '%s' has no label array available at index %d; skipping validation",
                    label_name,
                    vol_idx,
                )
                continue

        # Access the appropriate resolution level for patch finding
        actual_downsample_factor = downsample_factor
        actual_downsampled_patch_size = downsampled_patch_size

        def _resolve_resolution(array_obj, level_key):
            """Access resolution level from a zarr group or return array directly."""
            key = str(level_key)

            # If it's already an array (not a group), return it
            if hasattr(array_obj, 'shape') and hasattr(array_obj, 'dtype'):
                if not hasattr(array_obj, 'keys'):
                    return array_obj

            # Try accessing as group[level]
            try:
                candidate = array_obj[key]
                if hasattr(candidate, 'shape'):
                    return candidate
            except Exception:
                pass

            return None

        logger.info(
            "Resolving downsample level %s for volume '%s'",
            valid_patch_find_resolution,
            label_name,
        )

        try:
            # Try to get the requested resolution level
            candidate = _resolve_resolution(label_array, valid_patch_find_resolution)
            if candidate is not None:
                downsampled_array = candidate
                logger.info(
                    "Using level %s for '%s' with patch size %s",
                    valid_patch_find_resolution,
                    label_name,
                    actual_downsampled_patch_size,
                )
            else:
                # Fall back to level 0 or the array itself
                candidate_full = _resolve_resolution(label_array, '0')
                if candidate_full is not None:
                    downsampled_array = candidate_full
                else:
                    downsampled_array = label_array
                actual_downsample_factor = 1
                actual_downsampled_patch_size = patch_size
                logger.info(
                    "Level %s unavailable for '%s', using full resolution (patch size %s)",
                    valid_patch_find_resolution,
                    label_name,
                    actual_downsampled_patch_size,
                )
            if downsampled_array is None:
                logger.warning(
                    "Volume '%s': unable to resolve label data for validation; skipping",
                    label_name,
                )
                continue
        except Exception as e:
            logger.warning(
                "Error resolving level %s for '%s': %s. Using array directly.",
                valid_patch_find_resolution, label_name, e,
            )
            downsampled_array = label_array
            actual_downsample_factor = 1
            actual_downsampled_patch_size = patch_size

        if downsampled_array is None:
            logger.warning("Volume '%s': no usable label data; skipping", label_name)
            continue

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
        bg_positions_vol = []
        unlabeled_fg_positions_vol = []
        # BG collection is only valid when ignore_label is set
        should_collect_bg = collect_bg_only and ignore_label is not None
        # Unlabeled FG collection requires image array
        should_collect_unlabeled_fg = collect_unlabeled_fg and image_array is not None

        # Resolve image array for unlabeled FG detection
        downsampled_image_array = None
        if should_collect_unlabeled_fg:
            try:
                candidate = _resolve_resolution(image_array, valid_patch_find_resolution)
                if candidate is not None:
                    downsampled_image_array = candidate
                else:
                    candidate_full = _resolve_resolution(image_array, '0')
                    downsampled_image_array = candidate_full if candidate_full is not None else image_array
            except Exception as e:
                logger.warning(
                    "Error resolving image level %s for '%s': %s. Using array directly.",
                    valid_patch_find_resolution, label_name, e,
                )
                downsampled_image_array = image_array

        block_start = time.perf_counter()

        if is_2d:
            chunk_y_patches = max(1, chunk_shape[0] // dpY)
            chunk_x_patches = max(1, chunk_shape[1] // dpX)
            chunk_y_patches = min(chunk_y_patches, len(y_starts))
            chunk_x_patches = min(chunk_x_patches, len(x_starts))

            y_blocks = list(range(0, len(y_starts), chunk_y_patches))
            for yi in tqdm(y_blocks, desc=f"  {label_name} blocks", position=1, leave=False):
                y_group = y_starts[yi: yi + chunk_y_patches]
                y_start = y_group[0]
                y_stop = y_group[-1] + dpY

                for xi in range(0, len(x_starts), chunk_x_patches):
                    x_group = x_starts[xi: xi + chunk_x_patches]
                    x_start = x_group[0]
                    x_stop = x_group[-1] + dpX

                    block = np.asarray(downsampled_array[y_start:y_stop, x_start:x_stop])
                    # Check BG criteria on original block before zeroing ignore labels
                    if should_collect_bg:
                        block_for_bg = reduce_block_to_scalar(
                            block, spatial_ndim=2, channel_selector=selector
                        )
                        has_fg = np.any(block_for_bg == valid_patch_value)
                        has_ignore = np.any(block_for_bg == ignore_label)
                        has_bg = np.any(block_for_bg == 0)
                        if has_ignore and has_bg and not has_fg:
                            for yy in range(len(y_group)):
                                for xx in range(len(x_group)):
                                    bg_positions_vol.append((y_group[yy], x_group[xx]))
                    if ignore_label is not None:
                        block = zero_ignore_labels(block, ignore_label)
                    block = reduce_block_to_scalar(
                        block,
                        spatial_ndim=2,
                        channel_selector=selector,
                    )
                    if valid_patch_value is not None:
                        block_mask = np.asarray(block == valid_patch_value)
                    else:
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

                    # Collect valid (labeled) patches
                    if np.any(valid_mask):
                        valid_idx = np.argwhere(valid_mask)
                        for (yy, xx) in valid_idx:
                            pos_y = y_group[yy]
                            pos_x = x_group[xx]
                            valid_positions_vol.append((pos_y, pos_x))

                    # Collect unlabeled foreground patches (labels fail, but image has data)
                    if should_collect_unlabeled_fg and downsampled_image_array is not None:
                        # Patches that are effectively unlabeled (fail label threshold)
                        unlabeled_mask = ~valid_mask
                        if np.any(unlabeled_mask):
                            # Load corresponding image block
                            image_block = np.asarray(
                                downsampled_image_array[y_start:y_stop, x_start:x_stop]
                            )
                            # Reduce to scalar (handle multi-channel)
                            if image_block.ndim > 2:
                                image_block = image_block.reshape(image_block.shape[-2:])
                            image_nonzero = np.asarray(image_block != 0)
                            if np.any(image_nonzero):
                                image_nonzero = np.ascontiguousarray(image_nonzero)
                                img_patches_view = as_strided(
                                    image_nonzero,
                                    shape=(y_len, x_len, dpY, dpX),
                                    strides=(
                                        image_nonzero.strides[0] * dpY,
                                        image_nonzero.strides[1] * dpX,
                                        image_nonzero.strides[0],
                                        image_nonzero.strides[1],
                                    ),
                                    writeable=False,
                                )
                                img_patches_flat = img_patches_view.reshape(y_len, x_len, -1)
                                img_nonzero_counts = img_patches_flat.sum(axis=-1)
                                img_fraction = img_nonzero_counts / patch_volume

                                # Compute image bbox coverage
                                img_y_any = img_patches_view.any(axis=-1)
                                img_x_any = img_patches_view.any(axis=-2)
                                img_y_has = img_y_any.any(axis=-1)
                                img_x_has = img_x_any.any(axis=-1)
                                img_y_first = np.argmax(img_y_any, axis=-1)
                                img_y_last = img_y_any.shape[-1] - 1 - np.argmax(img_y_any[..., ::-1], axis=-1)
                                img_y_width = np.where(img_y_has, (img_y_last - img_y_first + 1), 0)
                                img_x_first = np.argmax(img_x_any, axis=-1)
                                img_x_last = img_x_any.shape[-1] - 1 - np.argmax(img_x_any[..., ::-1], axis=-1)
                                img_x_width = np.where(img_x_has, (img_x_last - img_x_first + 1), 0)
                                img_bbox_fraction = (img_y_width * img_x_width) / patch_volume

                                # Unlabeled FG: unlabeled AND image has data
                                unlabeled_fg_mask = (
                                    unlabeled_mask &
                                    (img_fraction >= unlabeled_fg_threshold) &
                                    (img_bbox_fraction >= unlabeled_fg_bbox_threshold)
                                )
                                if np.any(unlabeled_fg_mask):
                                    unlabeled_fg_idx = np.argwhere(unlabeled_fg_mask)
                                    for (yy, xx) in unlabeled_fg_idx:
                                        pos_y = y_group[yy]
                                        pos_x = x_group[xx]
                                        unlabeled_fg_positions_vol.append((pos_y, pos_x))
        else:
            chunk_z_patches = max(1, chunk_shape[0] // dpZ)
            chunk_y_patches = max(1, chunk_shape[1] // dpY)
            chunk_x_patches = max(1, chunk_shape[2] // dpX)
            chunk_z_patches = min(chunk_z_patches, len(z_starts))
            chunk_y_patches = min(chunk_y_patches, len(y_starts))
            chunk_x_patches = min(chunk_x_patches, len(x_starts))

            z_blocks = list(range(0, len(z_starts), chunk_z_patches))
            for zi in tqdm(z_blocks, desc=f"  {label_name} blocks", position=1, leave=False):
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

                        block = np.asarray(
                            downsampled_array[z_start:z_stop, y_start:y_stop, x_start:x_stop]
                        )
                        # Check BG criteria on original block before zeroing ignore labels
                        if should_collect_bg:
                            block_for_bg = reduce_block_to_scalar(
                                block, spatial_ndim=3, channel_selector=selector
                            )
                            has_fg = np.any(block_for_bg == valid_patch_value)
                            has_ignore = np.any(block_for_bg == ignore_label)
                            has_bg = np.any(block_for_bg == 0)
                            if has_ignore and has_bg and not has_fg:
                                for zz in range(len(z_group)):
                                    for yy in range(len(y_group)):
                                        for xx in range(len(x_group)):
                                            bg_positions_vol.append((z_group[zz], y_group[yy], x_group[xx]))
                        if ignore_label is not None:
                            block = zero_ignore_labels(block, ignore_label)
                        block = reduce_block_to_scalar(
                            block,
                            spatial_ndim=3,
                            channel_selector=selector,
                        )
                        if valid_patch_value is not None:
                            block_mask = np.asarray(block == valid_patch_value)
                        else:
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

                        # Collect valid (labeled) patches
                        if np.any(valid_mask):
                            valid_idx = np.argwhere(valid_mask)
                            for (zz, yy, xx) in valid_idx:
                                pos_z = z_group[zz]
                                pos_y = y_group[yy]
                                pos_x = x_group[xx]
                                valid_positions_vol.append((pos_z, pos_y, pos_x))

                        # Collect unlabeled foreground patches (labels fail, but image has data)
                        if should_collect_unlabeled_fg and downsampled_image_array is not None:
                            # Patches that are effectively unlabeled (fail label threshold)
                            unlabeled_mask = ~valid_mask
                            if np.any(unlabeled_mask):
                                # Load corresponding image block
                                image_block = np.asarray(
                                    downsampled_image_array[z_start:z_stop, y_start:y_stop, x_start:x_stop]
                                )
                                # Reduce to scalar (handle multi-channel)
                                if image_block.ndim > 3:
                                    image_block = image_block.reshape(image_block.shape[-3:])
                                image_nonzero = np.asarray(image_block != 0)
                                if np.any(image_nonzero):
                                    image_nonzero = np.ascontiguousarray(image_nonzero)
                                    img_patches_view = as_strided(
                                        image_nonzero,
                                        shape=(z_len, y_len, x_len, dpZ, dpY, dpX),
                                        strides=(
                                            image_nonzero.strides[0] * dpZ,
                                            image_nonzero.strides[1] * dpY,
                                            image_nonzero.strides[2] * dpX,
                                            image_nonzero.strides[0],
                                            image_nonzero.strides[1],
                                            image_nonzero.strides[2],
                                        ),
                                        writeable=False,
                                    )
                                    img_patches_flat = img_patches_view.reshape(z_len, y_len, x_len, -1)
                                    img_nonzero_counts = img_patches_flat.sum(axis=-1)
                                    img_fraction = img_nonzero_counts / patch_volume

                                    # Compute image bbox coverage (3D)
                                    img_z_any = img_patches_view.any(axis=(4, 5))
                                    img_y_any = img_patches_view.any(axis=(3, 5))
                                    img_x_any = img_patches_view.any(axis=(3, 4))
                                    img_z_has = img_z_any.any(axis=-1)
                                    img_y_has = img_y_any.any(axis=-1)
                                    img_x_has = img_x_any.any(axis=-1)
                                    img_z_first = np.argmax(img_z_any, axis=-1)
                                    img_z_last = img_z_any.shape[-1] - 1 - np.argmax(img_z_any[..., ::-1], axis=-1)
                                    img_z_width = np.where(img_z_has, (img_z_last - img_z_first + 1), 0)
                                    img_y_first = np.argmax(img_y_any, axis=-1)
                                    img_y_last = img_y_any.shape[-1] - 1 - np.argmax(img_y_any[..., ::-1], axis=-1)
                                    img_y_width = np.where(img_y_has, (img_y_last - img_y_first + 1), 0)
                                    img_x_first = np.argmax(img_x_any, axis=-1)
                                    img_x_last = img_x_any.shape[-1] - 1 - np.argmax(img_x_any[..., ::-1], axis=-1)
                                    img_x_width = np.where(img_x_has, (img_x_last - img_x_first + 1), 0)
                                    img_bbox_fraction = (img_z_width * img_y_width * img_x_width) / patch_volume

                                    # Unlabeled FG: unlabeled AND image has data
                                    unlabeled_fg_mask = (
                                        unlabeled_mask &
                                        (img_fraction >= unlabeled_fg_threshold) &
                                        (img_bbox_fraction >= unlabeled_fg_bbox_threshold)
                                    )
                                    if np.any(unlabeled_fg_mask):
                                        unlabeled_fg_idx = np.argwhere(unlabeled_fg_mask)
                                        for (zz, yy, xx) in unlabeled_fg_idx:
                                            pos_z = z_group[zz]
                                            pos_y = y_group[yy]
                                            pos_x = x_group[xx]
                                            unlabeled_fg_positions_vol.append((pos_z, pos_y, pos_x))

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

        # Add BG-only patches if collected
        for pos in bg_positions_vol:
            if is_2d:
                y, x = pos
                full_res_y = y * actual_downsample_factor
                full_res_x = x * actual_downsample_factor

                all_bg_patches.append({
                    'volume_idx': vol_idx,
                    'volume_name': label_name,
                    'start_pos': [full_res_y, full_res_x]
                })
            else:
                z, y, x = pos
                full_res_z = z * actual_downsample_factor
                full_res_y = y * actual_downsample_factor
                full_res_x = x * actual_downsample_factor

                all_bg_patches.append({
                    'volume_idx': vol_idx,
                    'volume_name': label_name,
                    'start_pos': [full_res_z, full_res_y, full_res_x]
                })

        # Add unlabeled FG patches if collected
        for pos in unlabeled_fg_positions_vol:
            if is_2d:
                y, x = pos
                full_res_y = y * actual_downsample_factor
                full_res_x = x * actual_downsample_factor

                all_unlabeled_fg_patches.append({
                    'volume_idx': vol_idx,
                    'volume_name': label_name,
                    'start_pos': [full_res_y, full_res_x]
                })
            else:
                z, y, x = pos
                full_res_z = z * actual_downsample_factor
                full_res_y = y * actual_downsample_factor
                full_res_x = x * actual_downsample_factor

                all_unlabeled_fg_patches.append({
                    'volume_idx': vol_idx,
                    'volume_name': label_name,
                    'start_pos': [full_res_z, full_res_y, full_res_x]
                })

        print(f"Found {len(valid_positions_vol)} valid patches in '{label_name}'")
        if bg_positions_vol:
            print(f"Found {len(bg_positions_vol)} BG-only patches in '{label_name}'")
        if unlabeled_fg_positions_vol:
            print(f"Found {len(unlabeled_fg_positions_vol)} unlabeled foreground patches in '{label_name}'")

    # Final summary
    print(f"\nTotal valid patches found across all {len(label_arrays)} volumes: {len(all_valid_patches)}")
    if all_bg_patches:
        print(f"Total BG-only patches found: {len(all_bg_patches)}")
    if all_unlabeled_fg_patches:
        print(f"Total unlabeled foreground patches found: {len(all_unlabeled_fg_patches)}")

    return {
        'fg_patches': all_valid_patches,
        'bg_patches': all_bg_patches,
        'unlabeled_fg_patches': all_unlabeled_fg_patches,
    }
