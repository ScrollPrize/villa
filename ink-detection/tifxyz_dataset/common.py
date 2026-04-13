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


def _load_segment_ink_mask(dataset, segment):
    import cv2

    segment_uuid = str(segment.uuid)
    cached = dataset._segment_ink_mask_cache.get(segment_uuid)
    if cached is not None:
        return cached

    grid = dataset._get_segment_stored_grid(segment)
    stored_h, stored_w = tuple(int(v) for v in grid["shape"])
    segment.use_stored_resolution()
    scale_y, scale_x = getattr(segment, "_scale", (1.0, 1.0))
    scale_y = float(scale_y) if np.isfinite(scale_y) and float(scale_y) > 0.0 else 1.0
    scale_x = float(scale_x) if np.isfinite(scale_x) and float(scale_x) > 0.0 else 1.0
    expected_shape = (
        int(stored_h / scale_y),
        int(stored_w / scale_x),
    )
    ink_label_path = None
    ink_meta = next(
        (label for label in segment.list_labels() if label.get("name") == "inklabels"),
        None,
    )
    if ink_meta is not None and ink_meta.get("path") is not None:
        ink_label_path = str(ink_meta["path"])
    if ink_label_path is None:
        path_map = getattr(dataset, "_segment_ink_label_path_by_uuid", {})
        ink_label_path = path_map.get(segment_uuid)
    if ink_label_path is None:
        out = np.zeros(expected_shape, dtype=bool)
        dataset._segment_ink_mask_cache[segment_uuid] = out
        return out

    ink_label = cv2.imread(str(ink_label_path), cv2.IMREAD_UNCHANGED)
    if ink_label is None:
        out = np.zeros(expected_shape, dtype=bool)
        dataset._segment_ink_mask_cache[segment_uuid] = out
        return out
    if ink_label.ndim == 3:
        if ink_label.shape[2] == 4:
            ink_label = cv2.cvtColor(ink_label, cv2.COLOR_BGRA2GRAY)
        else:
            ink_label = cv2.cvtColor(ink_label, cv2.COLOR_BGR2GRAY)

    if tuple(int(v) for v in ink_label.shape) != expected_shape:
        fixed_label, _ = _fix_known_bottom_right_padding(
            ink_label,
            expected_shape,
            dataset.auto_fix_padding_multiples,
        )
        if fixed_label is not None:
            ink_label = fixed_label
        else:
            ink_label = (
                cv2.resize(
                    (ink_label > 0).astype(np.uint8),
                    (int(expected_shape[1]), int(expected_shape[0])),
                    interpolation=cv2.INTER_AREA,
                )
                > 0
            ).astype(np.uint8)

    out = np.asarray(ink_label > 0, dtype=bool)
    dataset._segment_ink_mask_cache[segment_uuid] = out
    return out


def _get_segment_positive_points_zyx(dataset, segment):
    return _get_segment_positive_samples(dataset, segment)["points_zyx"]


def _normalize_distance_pair(value, name):
    if np.isscalar(value):
        distance = float(value)
        if not np.isfinite(distance):
            raise ValueError(f"{name} must be finite, got {value!r}")
        distance = max(0.0, distance)
        return distance, distance

    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if int(arr.size) != 2:
        raise ValueError(
            f"{name} must be a scalar or a two-value sequence [positive, negative], got {value!r}"
        )

    pos_distance = float(arr[0])
    neg_distance = float(arr[1])
    if not np.isfinite(pos_distance) or not np.isfinite(neg_distance):
        raise ValueError(f"{name} values must be finite, got {value!r}")
    return max(0.0, pos_distance), max(0.0, neg_distance)


def _build_surface_supervision_from_ink_mask(ink_mask, bg_dilate_distance):
    import cv2

    ink = np.asarray(ink_mask, dtype=bool)
    out = np.full(ink.shape, 100, dtype=np.uint8)
    out[ink] = 1

    bg_radius = max(0, int(bg_dilate_distance))
    if bg_radius <= 0 or not bool(np.any(ink)):
        return out

    background_src = (~ink).astype(np.uint8, copy=False)
    distances = cv2.distanceTransform(background_src, cv2.DIST_L2, 5)
    near_bg = (~ink) & np.isfinite(distances) & (distances <= float(bg_radius))
    out[near_bg] = 0
    return out


def _get_segment_positive_samples(dataset, segment):
    import cv2

    segment_uuid = str(segment.uuid)
    cached = dataset._segment_positive_samples_cache.get(segment_uuid)
    if cached is not None:
        return cached

    grid = dataset._get_segment_stored_grid(segment)
    ink_mask = _load_segment_ink_mask(dataset, segment)
    if tuple(int(v) for v in ink_mask.shape) != tuple(int(v) for v in grid["shape"]):
        ink_mask = (
            cv2.resize(
                ink_mask.astype(np.uint8),
                (int(grid["shape"][1]), int(grid["shape"][0])),
                interpolation=cv2.INTER_AREA,
            )
            > 0
        )

    positive_mask = np.asarray(grid["valid"] & ink_mask, dtype=bool)
    if not bool(np.any(positive_mask)):
        out = {
            "rows": np.empty((0,), dtype=np.int32),
            "cols": np.empty((0,), dtype=np.int32),
            "points_zyx": np.empty((0, 3), dtype=np.float32),
        }
        dataset._segment_positive_samples_cache[segment_uuid] = out
        dataset._segment_positive_points_cache[segment_uuid] = out["points_zyx"]
        return out

    row_idx, col_idx = np.where(positive_mask)
    row_idx = np.asarray(row_idx, dtype=np.int32)
    col_idx = np.asarray(col_idx, dtype=np.int32)
    points_zyx = np.stack(
        [
            grid["z"][row_idx, col_idx],
            grid["y"][row_idx, col_idx],
            grid["x"][row_idx, col_idx],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    out = {
        "rows": row_idx,
        "cols": col_idx,
        "points_zyx": points_zyx,
    }
    dataset._segment_positive_samples_cache[segment_uuid] = out
    dataset._segment_positive_points_cache[segment_uuid] = points_zyx
    return out


def _get_segment_background_samples(dataset, segment):
    import cv2

    segment_uuid = str(segment.uuid)
    cached = dataset._segment_background_samples_cache.get(segment_uuid)
    if cached is not None:
        return cached

    grid = dataset._get_segment_stored_grid(segment)
    surface_supervision = dataset._load_segment_surface_supervision(segment)
    if tuple(int(v) for v in surface_supervision.shape) != tuple(int(v) for v in grid["shape"]):
        surface_supervision = cv2.resize(
            surface_supervision.astype(np.uint8),
            (int(grid["shape"][1]), int(grid["shape"][0])),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8, copy=False)

    background_mask = np.asarray(grid["valid"] & (surface_supervision == 0), dtype=bool)
    if not bool(np.any(background_mask)):
        out = {
            "rows": np.empty((0,), dtype=np.int32),
            "cols": np.empty((0,), dtype=np.int32),
            "points_zyx": np.empty((0, 3), dtype=np.float32),
        }
        dataset._segment_background_samples_cache[segment_uuid] = out
        return out

    row_idx, col_idx = np.where(background_mask)
    row_idx = np.asarray(row_idx, dtype=np.int32)
    col_idx = np.asarray(col_idx, dtype=np.int32)
    points_zyx = np.stack(
        [
            grid["z"][row_idx, col_idx],
            grid["y"][row_idx, col_idx],
            grid["x"][row_idx, col_idx],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    out = {
        "rows": row_idx,
        "cols": col_idx,
        "points_zyx": points_zyx,
    }
    dataset._segment_background_samples_cache[segment_uuid] = out
    return out


def _sample_patch_supervision_grid(dataset, segment, min_corner, max_corner, extra_bbox_pad=0.0):
    from vesuvius.tifxyz import interpolate_at_points

    grid = dataset._get_segment_stored_grid(segment)
    x_stored = grid["x"]
    y_stored = grid["y"]
    z_stored = grid["z"]
    valid_mask = grid["valid"]
    n_rows_stored, n_cols_stored = x_stored.shape
    crop_size_tuple = tuple(int(v) for v in dataset.patch_size_zyx)
    min_corner_f = np.asarray(min_corner, dtype=np.float32).reshape(3)
    max_corner_f = np.asarray(max_corner, dtype=np.float32).reshape(3)

    bbox_pad = max(float(dataset.surface_bbox_pad), float(extra_bbox_pad))
    if bbox_pad < 0.0:
        bbox_pad = 0.0

    interp_method = str(dataset.surface_interp_method).strip().lower()
    if interp_method not in {"catmull_rom", "linear", "bspline"}:
        interp_method = "catmull_rom"

    segment.use_stored_resolution()
    scale_y, scale_x = getattr(segment, "_scale", (1.0, 1.0))
    scale_y = float(scale_y) if np.isfinite(scale_y) and float(scale_y) > 0.0 else 1.0
    scale_x = float(scale_x) if np.isfinite(scale_x) and float(scale_x) > 0.0 else 1.0

    expanded_min = min_corner_f - bbox_pad
    expanded_max = max_corner_f + bbox_pad
    in_bbox = (
        valid_mask
        & (z_stored >= expanded_min[0])
        & (z_stored < expanded_max[0])
        & (y_stored >= expanded_min[1])
        & (y_stored < expanded_max[1])
        & (x_stored >= expanded_min[2])
        & (x_stored < expanded_max[2])
    )

    if not bool(np.any(in_bbox)):
        empty_mask = np.zeros((0, 0), dtype=bool)
        empty_grid = np.zeros((0, 0, 3), dtype=np.float32)
        empty_class = np.zeros((0, 0), dtype=np.uint8)
        return {
            "local_grid": empty_grid,
            "world_grid": empty_grid,
            "valid_interp": empty_mask,
            "in_patch": empty_mask,
            "class_codes": empty_class,
            "normals_zyx": empty_grid,
            "normals_valid": empty_mask,
            "crop_size": crop_size_tuple,
        }

    rows, cols = np.where(in_bbox)
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    kernel_pad = 2 if interp_method in {"catmull_rom", "bspline"} else 1
    row_min = max(0, row_min - kernel_pad)
    row_max = min(n_rows_stored - 1, row_max + kernel_pad)
    col_min = max(0, col_min - kernel_pad)
    col_max = min(n_cols_stored - 1, col_max + kernel_pad)

    n_rows_local = row_max - row_min + 1
    n_cols_local = col_max - col_min + 1
    query_h = 1 if n_rows_local <= 1 else max(n_rows_local, int(round(n_rows_local / scale_y)))
    query_w = 1 if n_cols_local <= 1 else max(n_cols_local, int(round(n_cols_local / scale_x)))
    query_rows = np.linspace(row_min, row_max, query_h, dtype=np.float32)
    query_cols = np.linspace(col_min, col_max, query_w, dtype=np.float32)
    query_y, query_x = np.meshgrid(query_rows, query_cols, indexing="ij")

    x_int, y_int, z_int, int_valid = interpolate_at_points(
        x_stored,
        y_stored,
        z_stored,
        valid_mask,
        query_y,
        query_x,
        scale=(1.0, 1.0),
        method=interp_method,
        invalid_value=-1.0,
    )
    world_grid = np.stack([z_int, y_int, x_int], axis=-1).astype(np.float32, copy=False)
    valid_interp = np.asarray(int_valid, dtype=bool)
    valid_interp &= np.isfinite(world_grid).all(axis=-1)
    in_patch = (
        valid_interp
        & (world_grid[..., 0] >= min_corner_f[0])
        & (world_grid[..., 0] < max_corner_f[0])
        & (world_grid[..., 1] >= min_corner_f[1])
        & (world_grid[..., 1] < max_corner_f[1])
        & (world_grid[..., 2] >= min_corner_f[2])
        & (world_grid[..., 2] < max_corner_f[2])
    )

    normals_grid = _get_segment_normals_zyx(dataset, segment)
    nz_int, ny_int, nx_int, normals_valid = interpolate_at_points(
        normals_grid[..., 0],
        normals_grid[..., 1],
        normals_grid[..., 2],
        valid_mask,
        query_y,
        query_x,
        scale=(1.0, 1.0),
        method=interp_method,
        invalid_value=np.nan,
    )
    normals_zyx = np.stack([nz_int, ny_int, nx_int], axis=-1).astype(np.float32, copy=False)
    normals_valid = np.asarray(normals_valid, dtype=bool)
    normals_valid &= np.isfinite(normals_zyx).all(axis=-1)

    class_codes = np.full(query_y.shape, 100, dtype=np.uint8)
    ink_mask_full = _load_segment_ink_mask(dataset, segment)
    if ink_mask_full.size > 0:
        full_h, full_w = ink_mask_full.shape
        query_rows_full = query_y / scale_y
        query_cols_full = query_x / scale_x
        label_rows = np.rint(query_rows_full).astype(np.int64, copy=False)
        label_cols = np.rint(query_cols_full).astype(np.int64, copy=False)
        in_label_bounds = (
            (label_rows >= 0)
            & (label_rows < int(full_h))
            & (label_cols >= 0)
            & (label_cols < int(full_w))
        )
        if bool(np.any(in_label_bounds)):
            halo = max(0, int(dataset.bg_dilate_distance))
            rows_in = label_rows[in_label_bounds]
            cols_in = label_cols[in_label_bounds]
            row_min = max(0, int(rows_in.min()) - halo)
            row_max = min(int(full_h) - 1, int(rows_in.max()) + halo)
            col_min = max(0, int(cols_in.min()) - halo)
            col_max = min(int(full_w) - 1, int(cols_in.max()) + halo)

            ink_local = ink_mask_full[row_min:row_max + 1, col_min:col_max + 1]
            supervision_local = _build_surface_supervision_from_ink_mask(
                ink_local,
                bg_dilate_distance=dataset.bg_dilate_distance,
            )
            class_codes[in_label_bounds] = supervision_local[
                rows_in - row_min,
                cols_in - col_min,
            ]

    local_grid = world_grid - min_corner_f.reshape(1, 1, 3)
    return {
        "local_grid": local_grid,
        "world_grid": world_grid,
        "valid_interp": valid_interp,
        "in_patch": in_patch,
        "class_codes": class_codes,
        "normals_zyx": normals_zyx,
        "normals_valid": normals_valid,
        "crop_size": crop_size_tuple,
    }


def _voxelize_surface_from_sampled_grid(dataset, segment, min_corner, max_corner, crop_size, sampled_grid=None):
    from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid_masked

    crop_size_tuple = tuple(int(v) for v in crop_size)
    sampled = sampled_grid
    if sampled is None:
        sampled = _sample_patch_supervision_grid(
            dataset,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            extra_bbox_pad=0.0,
        )
    local_grid = sampled["local_grid"]
    in_patch = sampled["in_patch"]
    if local_grid.size == 0 or not bool(np.any(in_patch)):
        return np.zeros(crop_size_tuple, dtype=np.float32)
    return voxelize_surface_grid_masked(local_grid, crop_size_tuple, in_patch).astype(
        np.float32,
        copy=False,
    )


def _voxelize_positive_labels_from_sampled_grid(dataset, segment, min_corner, max_corner, crop_size, sampled_grid=None):
    crop_size_tuple = tuple(int(v) for v in crop_size)
    sampled = sampled_grid
    if sampled is None:
        sampled = _sample_patch_supervision_grid(
            dataset,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            extra_bbox_pad=0.0,
        )
    positive_mask = sampled["in_patch"] & (sampled["class_codes"] == 1)
    if not bool(np.any(positive_mask)):
        return np.zeros(crop_size_tuple, dtype=np.float32)
    return _points_to_voxels(sampled["local_grid"][positive_mask], crop_size_tuple)


def _voxelize_background_surface_labels_from_sampled_grid(
    dataset,
    segment,
    min_corner,
    max_corner,
    crop_size,
    sampled_grid=None,
):
    crop_size_tuple = tuple(int(v) for v in crop_size)
    sampled = sampled_grid
    if sampled is None:
        sampled = _sample_patch_supervision_grid(
            dataset,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            extra_bbox_pad=0.0,
        )
    background_mask = sampled["in_patch"] & (sampled["class_codes"] == 0)
    if not bool(np.any(background_mask)):
        return np.zeros(crop_size_tuple, dtype=np.float32)
    return _points_to_voxels(sampled["local_grid"][background_mask], crop_size_tuple)


def _voxelize_class_tolerance_band_from_sample(
    dataset,
    sampled_grid,
    min_corner,
    max_corner,
    crop_size,
    class_value,
    label_distance,
):
    crop_size_tuple = tuple(int(v) for v in crop_size)
    pos_distance, neg_distance = _normalize_distance_pair(label_distance, name="label_distance")
    max_distance = max(pos_distance, neg_distance)
    class_mask = sampled_grid["class_codes"] == int(class_value)

    if max_distance <= 0.0:
        in_patch = sampled_grid["in_patch"] & class_mask
        if not bool(np.any(in_patch)):
            return np.zeros(crop_size_tuple, dtype=np.float32)
        return _points_to_voxels(sampled_grid["local_grid"][in_patch], crop_size_tuple)

    expanded_mask = sampled_grid["valid_interp"] & class_mask & sampled_grid["normals_valid"]
    if not bool(np.any(expanded_mask)):
        return np.zeros(crop_size_tuple, dtype=np.float32)

    points_world = sampled_grid["world_grid"][expanded_mask].astype(np.float32, copy=False)
    normals_zyx = sampled_grid["normals_zyx"][expanded_mask].astype(np.float32, copy=False)
    expand = max_distance + 1.0
    expanded_min = np.asarray(min_corner, dtype=np.float32) - expand
    expanded_max = np.asarray(max_corner, dtype=np.float32) + expand
    in_expanded = _points_within_minmax(points_world, expanded_min, expanded_max)
    if not bool(np.any(in_expanded)):
        return np.zeros(crop_size_tuple, dtype=np.float32)

    return _build_normal_offset_mask_from_labeled_points(
        points_world[in_expanded],
        normals_zyx[in_expanded],
        min_corner=min_corner,
        crop_size=crop_size_tuple,
        label_distance=(pos_distance, neg_distance),
        sample_step=float(dataset.normal_sample_step),
        trilinear_threshold=float(dataset.normal_trilinear_threshold),
        use_numba=bool(dataset.use_numba_for_normal_mask),
    )


def _voxelize_label_tolerance_band(dataset, segment, min_corner, max_corner, crop_size, sampled_grid=None):
    sampled = sampled_grid
    if sampled is None:
        sampled = _sample_patch_supervision_grid(
            dataset,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            extra_bbox_pad=float(dataset.label_distance_max) + 1.0,
        )
    return _voxelize_class_tolerance_band_from_sample(
        dataset,
        sampled,
        min_corner=min_corner,
        max_corner=max_corner,
        crop_size=crop_size,
        class_value=1,
        label_distance=(dataset.label_distance_pos, dataset.label_distance_neg),
    )


def _voxelize_background_tolerance_band(dataset, segment, min_corner, max_corner, crop_size, sampled_grid=None):
    sampled = sampled_grid
    if sampled is None:
        sampled = _sample_patch_supervision_grid(
            dataset,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            extra_bbox_pad=float(dataset.bg_distance_max) + 1.0,
        )
    return _voxelize_class_tolerance_band_from_sample(
        dataset,
        sampled,
        min_corner=min_corner,
        max_corner=max_corner,
        crop_size=crop_size,
        class_value=0,
        label_distance=(dataset.bg_distance_pos, dataset.bg_distance_neg),
    )


def _build_projected_loss_mask_volume(dataset, segment, min_corner, max_corner, crop_size, sampled_grid=None):
    crop_size_tuple = tuple(int(v) for v in crop_size)
    out = np.full(crop_size_tuple, 2.0, dtype=np.float32)
    sampled = sampled_grid
    if sampled is None:
        sampled = _sample_patch_supervision_grid(
            dataset,
            segment,
            min_corner=min_corner,
            max_corner=max_corner,
            extra_bbox_pad=max(float(dataset.bg_distance_max), float(dataset.label_distance_max)) + 1.0,
        )
    background_tolerance_vox = _voxelize_background_tolerance_band(
        dataset,
        segment,
        min_corner=min_corner,
        max_corner=max_corner,
        crop_size=crop_size_tuple,
        sampled_grid=sampled,
    )
    out[background_tolerance_vox > 0.0] = 0.0

    label_tolerance_vox = _voxelize_label_tolerance_band(
        dataset,
        segment,
        min_corner=min_corner,
        max_corner=max_corner,
        crop_size=crop_size_tuple,
        sampled_grid=sampled,
    )
    out[label_tolerance_vox > 0.0] = 1.0
    return out


def _build_surface_label_volume(positive_label_vox, background_label_vox, crop_size):
    crop_size_tuple = tuple(int(v) for v in crop_size)
    out = np.full(crop_size_tuple, 2.0, dtype=np.float32)
    out[background_label_vox > 0.0] = 0.0
    out[positive_label_vox > 0.0] = 1.0
    return out


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
    v_prev_r[0, :] = False
    v_next_r[-1, :] = False
    v_prev_c[:, 0] = False
    v_next_c[:, -1] = False

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


def _get_segment_normals_zyx(dataset, segment):
    segment_uuid = str(segment.uuid)
    cached = dataset._segment_normal_cache.get(segment_uuid)
    if cached is not None:
        return cached

    grid = dataset._get_segment_stored_grid(segment)
    normals = _estimate_surface_normals_zyx(
        grid["x"],
        grid["y"],
        grid["z"],
        grid["valid"],
    )
    dataset._segment_normal_cache[segment_uuid] = normals
    return normals


def _voxelize_positive_labels(dataset, segment, min_corner, max_corner, crop_size):
    positive_points_world = _get_segment_positive_points_zyx(dataset, segment)
    if positive_points_world.shape[0] == 0:
        return np.zeros(tuple(int(v) for v in crop_size), dtype=np.float32)

    in_bbox = _points_within_minmax(positive_points_world, min_corner, max_corner)
    if not bool(np.any(in_bbox)):
        return np.zeros(tuple(int(v) for v in crop_size), dtype=np.float32)

    local_points = positive_points_world[in_bbox] - np.asarray(min_corner, dtype=np.float32)[None, :]
    return _points_to_voxels(local_points, crop_size)


def _voxelize_surface(dataset, segment, min_corner, max_corner, crop_size):
    from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid_masked
    from vesuvius.tifxyz import interpolate_at_points

    crop_size_tuple = tuple(int(v) for v in crop_size)
    min_corner_f = np.asarray(min_corner, dtype=np.float32).reshape(3)
    max_corner_f = np.asarray(max_corner, dtype=np.float32).reshape(3)
    grid = dataset._get_segment_stored_grid(segment)
    x_stored = grid["x"]
    y_stored = grid["y"]
    z_stored = grid["z"]
    valid_mask = grid["valid"]
    n_rows_stored, n_cols_stored = x_stored.shape

    bbox_pad = max(0.0, float(getattr(dataset, "surface_bbox_pad", 2.0)))
    interp_method = "catmull_rom"
    segment.use_stored_resolution()
    scale_y, scale_x = getattr(segment, "_scale", (1.0, 1.0))
    scale_y = float(scale_y) if np.isfinite(scale_y) and float(scale_y) > 0.0 else 1.0
    scale_x = float(scale_x) if np.isfinite(scale_x) and float(scale_x) > 0.0 else 1.0

    expanded_min = min_corner_f - bbox_pad
    expanded_max = max_corner_f + bbox_pad

    in_bbox = (
        valid_mask
        & (z_stored >= expanded_min[0])
        & (z_stored < expanded_max[0])
        & (y_stored >= expanded_min[1])
        & (y_stored < expanded_max[1])
        & (x_stored >= expanded_min[2])
        & (x_stored < expanded_max[2])
    )
    if not bool(np.any(in_bbox)):
        return np.zeros(crop_size_tuple, dtype=np.float32)

    rows, cols = np.where(in_bbox)
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    kernel_pad = 2
    row_min = max(0, row_min - kernel_pad)
    row_max = min(n_rows_stored - 1, row_max + kernel_pad)
    col_min = max(0, col_min - kernel_pad)
    col_max = min(n_cols_stored - 1, col_max + kernel_pad)

    n_rows_local = row_max - row_min + 1
    n_cols_local = col_max - col_min + 1
    query_h = 1 if n_rows_local <= 1 else max(n_rows_local, int(round(n_rows_local / scale_y)))
    query_w = 1 if n_cols_local <= 1 else max(n_cols_local, int(round(n_cols_local / scale_x)))
    query_rows = np.linspace(row_min, row_max, query_h, dtype=np.float32)
    query_cols = np.linspace(col_min, col_max, query_w, dtype=np.float32)
    query_y, query_x = np.meshgrid(query_rows, query_cols, indexing="ij")

    x_int, y_int, z_int, int_valid = interpolate_at_points(
        x_stored,
        y_stored,
        z_stored,
        valid_mask,
        query_y,
        query_x,
        scale=(1.0, 1.0),
        method=interp_method,
        invalid_value=-1.0,
    )
    zyx_world = np.stack([z_int, y_int, x_int], axis=-1).astype(np.float32, copy=False)
    valid_interp = np.asarray(int_valid, dtype=bool)
    valid_interp &= np.isfinite(zyx_world).all(axis=-1)
    valid_interp &= (
        (zyx_world[..., 0] >= min_corner_f[0])
        & (zyx_world[..., 0] < max_corner_f[0])
        & (zyx_world[..., 1] >= min_corner_f[1])
        & (zyx_world[..., 1] < max_corner_f[1])
        & (zyx_world[..., 2] >= min_corner_f[2])
        & (zyx_world[..., 2] < max_corner_f[2])
    )
    if not bool(np.any(valid_interp)):
        return np.zeros(crop_size_tuple, dtype=np.float32)

    local_grid = zyx_world - min_corner_f.reshape(1, 1, 3)
    return voxelize_surface_grid_masked(local_grid, crop_size_tuple, valid_interp).astype(
        np.float32,
        copy=False,
    )


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

    if np.isscalar(label_distance):
        pos_distance = float(label_distance)
        neg_distance = float(label_distance)
    else:
        distance_arr = np.asarray(label_distance, dtype=np.float32).reshape(-1)
        if int(distance_arr.size) != 2:
            raise ValueError(
                f"label_distance must be a scalar or [positive, negative], got {label_distance!r}"
            )
        pos_distance = float(distance_arr[0])
        neg_distance = float(distance_arr[1])
    if not np.isfinite(pos_distance) or not np.isfinite(neg_distance):
        raise ValueError(f"label_distance values must be finite, got {label_distance!r}")
    pos_distance = max(0.0, pos_distance)
    neg_distance = max(0.0, neg_distance)

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

    max_distance = max(pos_distance, neg_distance)
    if max_distance <= 0.0:
        local_points = points - min_corner
        return _points_to_voxels(local_points, crop_size_tuple)

    n_samples = max(2, int(np.ceil((pos_distance + neg_distance) / sample_step)) + 1)
    offsets = np.linspace(-neg_distance, pos_distance, num=n_samples, dtype=np.float32)
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
