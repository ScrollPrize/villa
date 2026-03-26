import numpy as np


def compute_native_crop_bbox_from_patch_points(
    patch_zyxs,
    valid_mask,
    target_shape_zyx,
) -> tuple[int, int, int, int, int, int]:
    patch_zyxs = np.asarray(patch_zyxs)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    valid_pts = patch_zyxs[valid_mask]
    if valid_pts.size == 0:
        raise ValueError("No valid tifxyz points found for patch")

    mins = valid_pts.min(axis=0).astype(np.int64)
    maxs = valid_pts.max(axis=0).astype(np.int64)
    target_shape_zyx = np.asarray(target_shape_zyx, dtype=np.int64)

    actual_zyx_shape = (maxs - mins + 1).astype(np.int64)
    shape_diff = target_shape_zyx - actual_zyx_shape

    trim_before = np.maximum(-shape_diff, 0) // 2
    trim_after = np.maximum(-shape_diff, 0) - trim_before
    mins = mins + trim_before
    maxs = maxs - trim_after

    adjusted_shape = (maxs - mins + 1).astype(np.int64)
    remaining_diff = target_shape_zyx - adjusted_shape

    pad_before = np.maximum(remaining_diff, 0) // 2
    pad_after = np.maximum(remaining_diff, 0) - pad_before
    mins = mins - pad_before
    maxs = maxs + pad_after

    return (
        int(mins[0]),
        int(mins[1]),
        int(mins[2]),
        int(maxs[0] + 1),
        int(maxs[1] + 1),
        int(maxs[2] + 1),
    )
