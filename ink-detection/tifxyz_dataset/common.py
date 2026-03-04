import numpy as np
from vesuvius.neural_tracing.datasets.common import normalize_zscore

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None


def _empty_patch_generation_stats():
    return {
        "segments_considered": 0,
        "segments_tried": 0,
        "segments_missing_ink": 0,
        "segments_autofixed_padding": 0,
        "segments_without_positive_points": 0,
        "candidate_bboxes": 0,
        "rejected_positive_fraction": 0,
        "rejected_span": 0,
        "kept_patches": 0,
    }


def _normalize_patch_size_zyx(patch_size):
    patch_size_zyx = np.asarray(patch_size, dtype=np.int32).reshape(-1)
    if patch_size_zyx.size == 1:
        patch_size_zyx = np.repeat(patch_size_zyx, 3)
    if patch_size_zyx.size != 3 or np.any(patch_size_zyx <= 0):
        raise ValueError(
            f"patch_size must be a positive int or [z, y, x], got {patch_size!r}"
        )
    return patch_size_zyx

# we have two "known" padded sizes -- multiples of 64 or 256, which are leftover padding from old inference scripts
# that were used to generate labels
def _known_padded_size(base_size, multiple):
    base_size = int(base_size)
    multiple = int(multiple)
    if base_size % multiple == 0:
        return base_size + multiple
    return ((base_size + multiple - 1) // multiple) * multiple


def _dimension_matches_known_padding(actual, expected, multiple):
    actual = int(actual)
    expected = int(expected)
    if actual == expected:
        return True
    if abs(actual - expected) == 1:
        return True
    small = min(actual, expected)
    big = max(actual, expected)
    return big == _known_padded_size(small, multiple)

# if our dimension matches what we know are common padding multiples, we can remove it
# though this is kind of risky because unless we actually look at the label every time we dont really know
# if the padding is correctly removed or added...
def _fix_known_bottom_right_padding(label, expected_shape, multiples):
    expected_h, expected_w = int(expected_shape[0]), int(expected_shape[1])
    actual_h, actual_w = int(label.shape[0]), int(label.shape[1])

    for multiple in multiples:
        if not _dimension_matches_known_padding(actual_h, expected_h, multiple):
            continue
        if not _dimension_matches_known_padding(actual_w, expected_w, multiple):
            continue

        if (actual_h - expected_h) > 1 and np.any(label[expected_h:actual_h, :] != 0):
            continue
        if (actual_w - expected_w) > 1 and np.any(label[:, expected_w:actual_w] != 0):
            continue

        fixed = label[: min(actual_h, expected_h), : min(actual_w, expected_w)]
        pad_h = max(0, expected_h - fixed.shape[0])
        pad_w = max(0, expected_w - fixed.shape[1])
        if pad_h or pad_w:
            fixed = np.pad(fixed, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        return fixed, int(multiple)

    return None, None


def _points_within_bbox(points_zyx, bbox):
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    return (
        (points_zyx[:, 0] >= float(z_min))
        & (points_zyx[:, 0] < float(z_max) + 1.0)
        & (points_zyx[:, 1] >= float(y_min))
        & (points_zyx[:, 1] < float(y_max) + 1.0)
        & (points_zyx[:, 2] >= float(x_min))
        & (points_zyx[:, 2] < float(x_max) + 1.0)
    )


def _points_within_minmax(points_zyx, min_corner, max_corner):
    points = np.asarray(points_zyx, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    min_corner = np.asarray(min_corner, dtype=np.float32).reshape(3)
    max_corner = np.asarray(max_corner, dtype=np.float32).reshape(3)
    return (
        (points[:, 0] >= min_corner[0]) & (points[:, 0] < max_corner[0]) &
        (points[:, 1] >= min_corner[1]) & (points[:, 1] < max_corner[1]) &
        (points[:, 2] >= min_corner[2]) & (points[:, 2] < max_corner[2])
    )


def _points_to_voxels(points_local, crop_size):
    crop_size_arr = np.asarray(crop_size, dtype=np.int64).reshape(3)
    vox = np.zeros(tuple(int(v) for v in crop_size_arr.tolist()), dtype=np.float32)
    if points_local is None:
        return vox
    points = np.asarray(points_local, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return vox

    finite = np.isfinite(points).all(axis=1)
    if not bool(np.any(finite)):
        return vox
    coords = np.rint(points[finite]).astype(np.int64, copy=False)
    in_bounds = (
        (coords[:, 0] >= 0) & (coords[:, 0] < crop_size_arr[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < crop_size_arr[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < crop_size_arr[2])
    )
    if bool(np.any(in_bounds)):
        coords = coords[in_bounds]
        vox[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    return vox


if njit is not None:
    @njit(cache=True)
    def _splat_points_trilinear_numba(points, size_z, size_y, size_x):
        vox = np.zeros((size_z, size_y, size_x), dtype=np.float32)
        n_points = points.shape[0]
        for i in range(n_points):
            pz = points[i, 0]
            py = points[i, 1]
            px = points[i, 2]
            if not (np.isfinite(pz) and np.isfinite(py) and np.isfinite(px)):
                continue

            z0 = int(np.floor(pz))
            y0 = int(np.floor(py))
            x0 = int(np.floor(px))
            dz = pz - z0
            dy = py - y0
            dx = px - x0

            for oz in range(2):
                zi = z0 + oz
                if zi < 0 or zi >= size_z:
                    continue
                wz = (1.0 - dz) if oz == 0 else dz
                if wz <= 0.0:
                    continue
                for oy in range(2):
                    yi = y0 + oy
                    if yi < 0 or yi >= size_y:
                        continue
                    wy = (1.0 - dy) if oy == 0 else dy
                    if wy <= 0.0:
                        continue
                    for ox in range(2):
                        xi = x0 + ox
                        if xi < 0 or xi >= size_x:
                            continue
                        wx = (1.0 - dx) if ox == 0 else dx
                        if wx <= 0.0:
                            continue
                        vox[zi, yi, xi] += wz * wy * wx
        return vox
else:  # pragma: no cover - only used when numba missing
    _splat_points_trilinear_numba = None


def _points_to_voxels_trilinear(points_local, crop_size, threshold=1e-4, use_numba=True):
    crop_size_arr = np.asarray(crop_size, dtype=np.int64).reshape(3)
    crop_size_tuple = tuple(int(v) for v in crop_size_arr.tolist())
    points = np.asarray(points_local, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return np.zeros(crop_size_tuple, dtype=np.float32)

    finite = np.isfinite(points).all(axis=1)
    if not bool(np.any(finite)):
        return np.zeros(crop_size_tuple, dtype=np.float32)
    points = points[finite]

    if use_numba and _splat_points_trilinear_numba is not None:
        vox_accum = _splat_points_trilinear_numba(
            points,
            int(crop_size_arr[0]),
            int(crop_size_arr[1]),
            int(crop_size_arr[2]),
        )
        return (vox_accum > float(threshold)).astype(np.float32, copy=False)

    vox_accum = np.zeros(crop_size_tuple, dtype=np.float32)
    base = np.floor(points).astype(np.int64, copy=False)
    frac = points - base.astype(np.float32, copy=False)

    for oz in (0, 1):
        z_idx = base[:, 0] + oz
        wz = (1.0 - frac[:, 0]) if oz == 0 else frac[:, 0]
        for oy in (0, 1):
            y_idx = base[:, 1] + oy
            wy = (1.0 - frac[:, 1]) if oy == 0 else frac[:, 1]
            for ox in (0, 1):
                x_idx = base[:, 2] + ox
                wx = (1.0 - frac[:, 2]) if ox == 0 else frac[:, 2]
                w = wz * wy * wx
                valid = (
                    (w > 0.0)
                    & (z_idx >= 0)
                    & (z_idx < crop_size_arr[0])
                    & (y_idx >= 0)
                    & (y_idx < crop_size_arr[1])
                    & (x_idx >= 0)
                    & (x_idx < crop_size_arr[2])
                )
                if bool(np.any(valid)):
                    np.add.at(
                        vox_accum,
                        (z_idx[valid], y_idx[valid], x_idx[valid]),
                        w[valid].astype(np.float32, copy=False),
                    )
    return (vox_accum > float(threshold)).astype(np.float32, copy=False)


def _estimate_surface_normals_zyx(x_grid, y_grid, z_grid, valid_mask, eps=1e-6):
    x = np.asarray(x_grid, dtype=np.float32)
    y = np.asarray(y_grid, dtype=np.float32)
    z = np.asarray(z_grid, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)

    p = np.stack([z, y, x], axis=-1).astype(np.float32, copy=False)
    p_prev_r = np.roll(p, 1, axis=0)
    p_next_r = np.roll(p, -1, axis=0)
    p_prev_c = np.roll(p, 1, axis=1)
    p_next_c = np.roll(p, -1, axis=1)

    v_prev_r = np.roll(valid, 1, axis=0)
    v_next_r = np.roll(valid, -1, axis=0)
    v_prev_c = np.roll(valid, 1, axis=1)
    v_next_c = np.roll(valid, -1, axis=1)

    tangent_r = np.zeros_like(p, dtype=np.float32)
    tangent_c = np.zeros_like(p, dtype=np.float32)

    center_r = v_prev_r & v_next_r & valid
    forward_r = (~v_prev_r) & v_next_r & valid
    backward_r = v_prev_r & (~v_next_r) & valid
    tangent_r[center_r] = 0.5 * (p_next_r[center_r] - p_prev_r[center_r])
    tangent_r[forward_r] = p_next_r[forward_r] - p[forward_r]
    tangent_r[backward_r] = p[backward_r] - p_prev_r[backward_r]

    center_c = v_prev_c & v_next_c & valid
    forward_c = (~v_prev_c) & v_next_c & valid
    backward_c = v_prev_c & (~v_next_c) & valid
    tangent_c[center_c] = 0.5 * (p_next_c[center_c] - p_prev_c[center_c])
    tangent_c[forward_c] = p_next_c[forward_c] - p[forward_c]
    tangent_c[backward_c] = p[backward_c] - p_prev_c[backward_c]

    normals = np.cross(tangent_r, tangent_c).astype(np.float32, copy=False)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    good = valid & np.isfinite(norm[..., 0]) & (norm[..., 0] > float(eps))
    out = np.zeros_like(normals, dtype=np.float32)
    out[good] = normals[good] / norm[good]
    return out


def _build_normal_offset_mask_from_labeled_points(
    points_world_zyx,
    normals_zyx,
    min_corner,
    crop_size,
    label_distance,
    sample_step=0.5,
    trilinear_threshold=1e-4,
    use_numba=True,
):
    points = np.asarray(points_world_zyx, dtype=np.float32)
    normals = np.asarray(normals_zyx, dtype=np.float32)
    crop_size_arr = np.asarray(crop_size, dtype=np.int64).reshape(3)
    crop_size_tuple = tuple(int(v) for v in crop_size_arr.tolist())
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return np.zeros(crop_size_tuple, dtype=np.float32)
    if normals.ndim != 2 or normals.shape != points.shape:
        return np.zeros(crop_size_tuple, dtype=np.float32)

    label_distance = float(label_distance)
    sample_step = float(sample_step)
    if sample_step <= 0.0:
        sample_step = 0.5

    n_norm = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1) & (n_norm > 1e-6)
    if not bool(np.any(valid)):
        return np.zeros(crop_size_tuple, dtype=np.float32)

    points = points[valid]
    normals = normals[valid] / n_norm[valid, None]
    min_corner = np.asarray(min_corner, dtype=np.float32).reshape(1, 3)

    if label_distance <= 0.0:
        local_points = points - min_corner
        return _points_to_voxels(local_points, crop_size_tuple)

    n_samples = max(2, int(np.ceil((2.0 * label_distance) / sample_step)) + 1)
    offsets = np.linspace(-label_distance, label_distance, num=n_samples, dtype=np.float32)
    sampled = points[:, None, :] + offsets[None, :, None] * normals[:, None, :]
    local_points = sampled.reshape(-1, 3) - min_corner
    return _points_to_voxels_trilinear(
        local_points,
        crop_size_tuple,
        threshold=trilinear_threshold,
        use_numba=bool(use_numba),
    )

# simple "dominant" span finder
def _required_span_axes(points_zyx):
    y_span = float(np.max(points_zyx[:, 1]) - np.min(points_zyx[:, 1]))
    x_span = float(np.max(points_zyx[:, 2]) - np.min(points_zyx[:, 2]))
    return ("z", "y" if y_span >= x_span else "x")


# ensure the segment covers the entire z height of the crop, and in its dominant axis spans at least some percentage across it.
# this helps ensure we don't have patches which contain only a tiny corner of the segment 
def _passes_min_span(points_zyx, patch_size_zyx, min_span_ratio):
    if points_zyx.shape[0] == 0:
        return False, (0.0, 0.0, 0.0)

    spans = (
        float(np.max(points_zyx[:, 0]) - np.min(points_zyx[:, 0])),
        float(np.max(points_zyx[:, 1]) - np.min(points_zyx[:, 1])),
        float(np.max(points_zyx[:, 2]) - np.min(points_zyx[:, 2])),
    )
    axis_to_idx = {"z": 0, "y": 1, "x": 2}
    size_minus_one = (
        max(0.0, float(patch_size_zyx[0]) - 1.0),
        max(0.0, float(patch_size_zyx[1]) - 1.0),
        max(0.0, float(patch_size_zyx[2]) - 1.0),
    )
    for axis in _required_span_axes(points_zyx):
        axis_idx = axis_to_idx[axis]
        if spans[axis_idx] < float(min_span_ratio) * size_minus_one[axis_idx]:
            return False, spans
    return True, spans


def _read_volume_crop_from_patch_dict(patch, crop_size, min_corner, max_corner):
    """Read a [z, y, x] crop from a patch dict and z-score normalize it."""
    volume = patch["volume"]
    if not hasattr(volume, "shape"):
        volume = volume[str(int(patch["scale"]))]

    crop_size = tuple(int(v) for v in crop_size)
    min_corner = np.asarray(min_corner, dtype=np.int64).reshape(3)
    max_corner = np.asarray(max_corner, dtype=np.int64).reshape(3)

    vol_crop = np.zeros(crop_size, dtype=volume.dtype)
    vol_shape = np.asarray(volume.shape, dtype=np.int64)
    src_starts = np.maximum(min_corner, 0)
    src_ends = np.minimum(max_corner, vol_shape)
    dst_starts = src_starts - min_corner
    dst_ends = dst_starts + (src_ends - src_starts)

    if np.all(src_ends > src_starts):
        vol_crop[
            dst_starts[0]:dst_ends[0],
            dst_starts[1]:dst_ends[1],
            dst_starts[2]:dst_ends[2],
        ] = volume[
            src_starts[0]:src_ends[0],
            src_starts[1]:src_ends[1],
            src_starts[2]:src_ends[2],
        ]
    return normalize_zscore(vol_crop)
