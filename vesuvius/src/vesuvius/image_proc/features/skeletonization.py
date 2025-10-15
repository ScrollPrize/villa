"""
Utilities for skeletonizing binary masks slice-by-slice.

The helper here mirrors the ROI-based approach used in a few evaluation
utilities, but is exposed in a lightweight module so it can be re-used by
scripts without reaching into model code.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

__all__ = ["skeletonize_stack_2d"]


def _roi_bounds(mask2d: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (y0, y1, x0, x1) bounds with 1px padding for a non-empty mask."""
    ys, xs = np.nonzero(mask2d)
    if ys.size == 0:
        return None

    y0 = max(int(ys.min()) - 1, 0)
    y1 = int(ys.max()) + 2
    x0 = max(int(xs.min()) - 1, 0)
    x1 = int(xs.max()) + 2
    return y0, y1, x0, x1


_NEIGHBOR_KERNEL_2D = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def _branch_points_from_skeleton(skeleton2d: np.ndarray) -> int:
    """Count branch/junction pixels (>=3 neighbors) in a binary skeleton."""
    if skeleton2d.size == 0 or not skeleton2d.any():
        return 0
    skel_u8 = skeleton2d.astype(np.uint8, copy=False)
    neighbor_counts = convolve(skel_u8, _NEIGHBOR_KERNEL_2D, mode="constant", cval=0)
    branch_mask = (skel_u8 == 1) & (neighbor_counts >= 3)
    return int(branch_mask.sum())


def _skeletonize_slice(
    mask2d: np.ndarray,
    use_roi: bool,
    return_branch_points: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """
    Skeletonize a single 2D slice, optionally cropping to a tight ROI.

    Parameters
    ----------
    mask2d:
        2D boolean/binary array.
    use_roi:
        When True, skeletonize only a tight bounding box around the foreground.
    return_branch_points:
        If True, also return the number of branch points detected in the skeleton.
    """
    if not mask2d.any():
        empty = np.zeros(mask2d.shape, dtype=np.uint8)
        if return_branch_points:
            return empty, 0
        return empty

    if not use_roi:
        skel_full = skeletonize(mask2d).astype(np.uint8)
        if return_branch_points:
            return skel_full, _branch_points_from_skeleton(skel_full)
        return skel_full

    H, W = mask2d.shape
    bounds = _roi_bounds(mask2d)
    if bounds is None:
        empty = np.zeros((H, W), dtype=np.uint8)
        if return_branch_points:
            return empty, 0
        return empty

    y0, y1, x0, x1 = bounds
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(H, y1)
    x1 = min(W, x1)

    roi = mask2d[y0:y1, x0:x1]
    if roi.size == 0:
        empty = np.zeros((H, W), dtype=np.uint8)
        if return_branch_points:
            return empty, 0
        return empty

    skel_roi = skeletonize(roi).astype(np.uint8)

    out = np.zeros((H, W), dtype=np.uint8)
    out[y0:y1, x0:x1] = skel_roi
    if return_branch_points:
        return out, _branch_points_from_skeleton(out)
    return out


def skeletonize_stack_2d(
    mask: np.ndarray,
    use_roi: bool = True,
    return_branch_counts: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply 2D skeletonization slice-wise over a binary 2D or 3D mask.

    Parameters
    ----------
    mask:
        Binary/bool array with shape (Z, Y, X) or (Y, X). Non-zero values are
        treated as foreground.
    use_roi:
        If True (default), only the tight bounding box around the foreground is
        skeletonized to avoid unnecessary work on empty regions.
    return_branch_counts:
        If True, also return an array with the branch-point count per slice.

    Returns
    -------
    np.ndarray
        Skeletonized mask with the same shape as the input. Output dtype is uint8.
    tuple (np.ndarray, np.ndarray)
        Returned when ``return_branch_counts`` is True. Second element contains the
        branch-point counts per slice (shape (Z,)).
    """
    if mask.ndim not in (2, 3):
        raise ValueError(f"mask must be 2D or 3D, got shape {mask.shape}")

    mask_bool = mask.astype(bool, copy=False)
    squeeze = False
    if mask_bool.ndim == 2:
        mask_bool = mask_bool[np.newaxis, ...]
        squeeze = True

    out = np.zeros(mask_bool.shape, dtype=np.uint8)
    branch_counts = np.zeros(mask_bool.shape[0], dtype=np.int32) if return_branch_counts else None
    for z in range(mask_bool.shape[0]):
        slice_mask = mask_bool[z]
        if slice_mask.any():
            if return_branch_counts:
                skel_slice, bp = _skeletonize_slice(
                    slice_mask,
                    use_roi=use_roi,
                    return_branch_points=True,
                )
                out[z] = skel_slice
                branch_counts[z] = bp
            else:
                out[z] = _skeletonize_slice(slice_mask, use_roi=use_roi)

    if squeeze:
        if return_branch_counts:
            return out[0], branch_counts  # type: ignore[return-value]
        return out[0]

    if return_branch_counts:
        return out, branch_counts  # type: ignore[return-value]
    return out
