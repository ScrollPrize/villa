import numpy as np

def divide_nonzero(array1, array2, eps=1e-10):
    denominator = np.copy(array2)
    denominator[denominator == 0] = eps
    return np.divide(array1, denominator)


def arr_to_indices(arr, subsample_rate=20, pts_over=0, downsample_factor=1):
    """
    - Optionally downsamples the grid by an integer factor before thresholding.
    - Optionally subsamples the resulting point list by taking every Nth point.
    - Supports 2D or 3D grids and returns points in the ORIGINAL coordinate system
      along with the original shape.
    """

    arr = np.asarray(arr)
    if arr.ndim not in (2, 3):
        raise ValueError("arr_to_indices only supports 2D or 3D arrays")

    if downsample_factor is None or downsample_factor < 1:
        downsample_factor = 1
    else:
        downsample_factor = int(downsample_factor)

    if downsample_factor > 1:
        # Simple decimation for speed (sufficient for label masks)
        slc = tuple(slice(None, None, downsample_factor) for _ in range(arr.ndim))
        img_ds = arr[slc]
        pts = np.where(img_ds > pts_over)
        coords = [axis * downsample_factor for axis in pts]
    else:
        pts = np.where(arr > pts_over)
        coords = list(pts)

    # Subsample the point list to thin it further
    if subsample_rate is None or subsample_rate < 1:
        subsample_rate = 1
    else:
        subsample_rate = int(subsample_rate)
    coords = [axis[::subsample_rate] for axis in coords]

    # Map to x, y (, z) order regardless of input dimensionality
    ordered_coords = coords[::-1]
    pts_nd = np.column_stack(ordered_coords) if ordered_coords else np.empty((0, arr.ndim), dtype=int)

    return pts_nd, arr.shape