"""Native ZYX crop, tifxyz selection, surface projection, and mask geometry."""

from __future__ import annotations

import cc3d
import numpy as np
from numba import njit
from scipy import ndimage
from scipy.ndimage import distance_transform_edt


NATIVE_COARSE_PAD_LEVEL0_VOXELS = 20.0
SURFACE_MASK_MAX_DISTANCE_LEVEL0_VOXELS = 10.0


def native_volume_downsample_factor(resolution: int) -> int:
    """Return the native-coordinate factor for an OME-Zarr pyramid level."""

    try:
        level = int(resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"native volume resolution must be an integer, got {resolution!r}"
        ) from exc
    if level < 0:
        raise ValueError(f"native volume resolution must be >= 0, got {level!r}")
    return 1 << level


def native_tifxyz_pyramid_params(resolution: int) -> tuple[int, float, int]:
    """Return `(flat-grid stride, native-coordinate scale, coarse pad)`."""

    factor = native_volume_downsample_factor(resolution)
    pad = max(1, int(np.ceil(NATIVE_COARSE_PAD_LEVEL0_VOXELS / float(factor))))
    return factor, 1.0 / float(factor), pad


def read_tifxyz_on_flat_grid(
    patch_tifxyz,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    flat_grid_stride: int = 1,
    native_coordinate_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read physical XYZ rasters as a sampled native ZYX position grid."""
    stride = int(flat_grid_stride)
    if stride <= 0:
        raise ValueError(f"flat_grid_stride must be positive, got {stride!r}")
    scale = float(native_coordinate_scale)
    if scale <= 0.0:
        raise ValueError(f"native_coordinate_scale must be positive, got {scale!r}")
    full_x, full_y, full_z, valid = patch_tifxyz[
        slice(int(y0) * stride, int(y1) * stride, stride),
        slice(int(x0) * stride, int(x1) * stride, stride),
    ]
    positions_zyx = np.stack([full_z, full_y, full_x], axis=-1).astype(
        np.float32, copy=False
    )
    if scale != 1.0:
        positions_zyx = positions_zyx * scale
    return positions_zyx, np.asarray(valid, dtype=bool)


def compute_native_crop_bbox(
    positions_zyx: np.ndarray,
    valid_mask: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """Center a fixed-shape native crop on the valid patch-point extent."""
    valid_points = np.asarray(positions_zyx)[np.asarray(valid_mask, dtype=bool)]
    if valid_points.size == 0:
        raise ValueError("No valid tifxyz points found for patch")
    mins = valid_points.min(axis=0).astype(np.int64)
    maxs = valid_points.max(axis=0).astype(np.int64)
    target = np.asarray(target_shape_zyx, dtype=np.int64)
    shape_diff = target - (maxs - mins + 1)
    trim_before = np.maximum(-shape_diff, 0) // 2
    trim_after = np.maximum(-shape_diff, 0) - trim_before
    mins += trim_before
    maxs -= trim_after
    remaining = target - (maxs - mins + 1)
    pad_before = np.maximum(remaining, 0) // 2
    pad_after = np.maximum(remaining, 0) - pad_before
    mins -= pad_before
    maxs += pad_after
    return (
        int(mins[0]),
        int(mins[1]),
        int(mins[2]),
        int(maxs[0] + 1),
        int(maxs[1] + 1),
        int(maxs[2] + 1),
    )


def maybe_select_flat_pixels(
    positions_zyx: np.ndarray,
    valid_mask: np.ndarray,
    crop_bbox_zyx: tuple[int, int, int, int, int, int],
):
    """Return the tight flat-grid window whose points land in a native crop."""
    positions_zyx = np.asarray(positions_zyx)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    crop_start = np.asarray(crop_bbox_zyx[:3], dtype=np.int64)
    crop_stop = np.asarray(crop_bbox_zyx[3:], dtype=np.int64)
    within = valid_mask & np.isfinite(positions_zyx).all(axis=-1)
    within &= (positions_zyx >= crop_start).all(axis=-1)
    within &= (positions_zyx < crop_stop).all(axis=-1)
    if not np.any(within):
        return None
    rows = np.flatnonzero(np.any(within, axis=1))
    columns = np.flatnonzero(np.any(within, axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    return (
        (y0, y1, x0, x1),
        positions_zyx[y0:y1, x0:x1],
        within[y0:y1, x0:x1],
    )


def _stored_resolution_window(
    patch_tifxyz,
    crop_bbox_zyx,
    *,
    coarse_native_pad: int,
    coarse_positions_zyx: np.ndarray,
    coarse_valid: np.ndarray,
    native_coordinate_scale: float,
    flat_grid_stride: int,
):
    scale = float(native_coordinate_scale)
    stride = int(flat_grid_stride)
    if scale <= 0.0 or stride <= 0:
        raise ValueError("native coordinate scale and flat grid stride must be positive")
    coarse_positions_zyx = np.asarray(coarse_positions_zyx, dtype=np.float32)
    if scale != 1.0:
        coarse_positions_zyx = coarse_positions_zyx * scale
    coarse_bbox = tuple(
        int(value) + (-coarse_native_pad if index < 3 else coarse_native_pad)
        for index, value in enumerate(crop_bbox_zyx)
    )
    selection = maybe_select_flat_pixels(
        coarse_positions_zyx, coarse_valid, coarse_bbox
    )
    if selection is None:
        return None
    (coarse_y0, coarse_y1, coarse_x0, coarse_x1), _, _ = selection
    stored_h, stored_w = (int(value) for value in coarse_positions_zyx.shape[:2])
    full_h, full_w = (int(value) for value in patch_tifxyz.full_resolution_shape)
    if stored_h <= 0 or stored_w <= 0:
        raise ValueError(
            "stored-resolution tifxyz grid must have positive shape, "
            f"got {(stored_h, stored_w)!r}"
        )
    # Expand by one stored cell before mapping to full resolution so exact
    # full-resolution refinement cannot miss an intersection at a coarse edge.
    coarse_y0, coarse_y1 = max(0, coarse_y0 - 1), min(stored_h, coarse_y1 + 1)
    coarse_x0, coarse_x1 = max(0, coarse_x0 - 1), min(stored_w, coarse_x1 + 1)
    full_y0 = max(0, int(np.floor(coarse_y0 * full_h / float(stored_h))))
    full_y1 = min(full_h, int(np.ceil(coarse_y1 * full_h / float(stored_h))))
    full_x0 = max(0, int(np.floor(coarse_x0 * full_w / float(stored_w))))
    full_x1 = min(full_w, int(np.ceil(coarse_x1 * full_w / float(stored_w))))
    flat_y0, flat_x0 = full_y0 // stride, full_x0 // stride
    sampled_y0, sampled_x0 = flat_y0 * stride, flat_x0 * stride
    sampled_y1 = min(full_h, int(np.ceil(full_y1 / float(stride))) * stride)
    sampled_x1 = min(full_w, int(np.ceil(full_x1 / float(stride))) * stride)
    full_x, full_y, full_z, full_valid = patch_tifxyz[
        slice(sampled_y0, sampled_y1, stride),
        slice(sampled_x0, sampled_x1, stride),
    ]
    positions_zyx = np.stack([full_z, full_y, full_x], axis=-1).astype(
        np.float32, copy=False
    )
    if scale != 1.0:
        positions_zyx *= scale
    return positions_zyx, np.asarray(full_valid, dtype=bool), flat_y0, flat_x0


def select_flat_pixels_via_stored_resolution(
    patch_tifxyz,
    crop_bbox_zyx,
    *,
    coarse_native_pad: int,
    coarse_positions_zyx: np.ndarray,
    coarse_valid: np.ndarray,
    native_coordinate_scale: float = 1.0,
    flat_grid_stride: int = 1,
    required: bool = True,
):
    """Refine a coarse tifxyz intersection to an exact full-grid support window."""
    window = _stored_resolution_window(
        patch_tifxyz,
        crop_bbox_zyx,
        coarse_native_pad=coarse_native_pad,
        coarse_positions_zyx=coarse_positions_zyx,
        coarse_valid=coarse_valid,
        native_coordinate_scale=native_coordinate_scale,
        flat_grid_stride=flat_grid_stride,
    )
    if window is None:
        if required:
            raise ValueError(
                f"crop_bbox {crop_bbox_zyx!r} does not intersect any valid flat tifxyz pixels"
            )
        return None
    positions_zyx, valid, base_y0, base_x0 = window
    selection = maybe_select_flat_pixels(positions_zyx, valid, crop_bbox_zyx)
    if selection is None:
        if required:
            raise ValueError(
                f"crop_bbox {crop_bbox_zyx!r} does not intersect any valid flat tifxyz pixels"
            )
        return None
    (local_y0, local_y1, local_x0, local_x1), support, support_valid = selection
    support_bbox = (
        base_y0 + local_y0,
        base_y0 + local_y1,
        base_x0 + local_x0,
        base_x0 + local_x1,
    )
    return support_bbox, support, support_valid


def project_flat_patch(
    flat_patch: np.ndarray,
    positions_zyx: np.ndarray,
    valid_mask: np.ndarray,
    crop_bbox_zyx: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    """Scatter nonzero flat values into a native crop using maximum reduction."""
    z0, y0, x0, z1, y1, x1 = crop_bbox_zyx
    output = np.zeros(
        (z1 - z0, y1 - y0, x1 - x0), dtype=np.asarray(flat_patch).dtype
    )
    valid = np.asarray(valid_mask, dtype=bool) & (np.asarray(flat_patch) != 0)
    valid &= np.isfinite(positions_zyx).all(axis=-1)
    if not np.any(valid):
        return output
    mapped = np.asarray(positions_zyx)[valid].astype(np.int64, copy=False)
    local = mapped - np.asarray((z0, y0, x0), dtype=np.int64)
    within = (local >= 0).all(axis=1) & (local < np.asarray(output.shape)).all(axis=1)
    if not np.any(within):
        return output
    local = local[within]
    values = np.asarray(flat_patch)[valid][within]
    flat_indices = np.ravel_multi_index(local.T, output.shape)
    np.maximum.at(output.reshape(-1), flat_indices, values)
    return output


def project_surface_distance(
    positions_zyx: np.ndarray,
    valid_mask: np.ndarray,
    crop_bbox_zyx: tuple[int, int, int, int, int, int],
    *,
    max_distance_voxels: float = 10.0,
) -> np.ndarray:
    occupancy = project_flat_patch(
        np.ones(np.asarray(valid_mask).shape, dtype=np.float32),
        positions_zyx,
        valid_mask,
        crop_bbox_zyx,
    ) > 0
    if not np.any(occupancy):
        return occupancy.astype(np.float32)
    if max_distance_voxels <= 0.0:
        return occupancy.astype(np.float32, copy=False)
    distance = distance_transform_edt(~occupancy)
    return np.clip(1.0 - distance / max_distance_voxels, 0.0, 1.0).astype(
        np.float32, copy=False
    )


@njit(cache=True)
def _mark(output, z, y, x, z0, y0, x0):
    local_z = int(z) - z0
    local_y = int(y) - y0
    local_x = int(x) - x0
    if (
        local_z >= 0
        and local_z < output.shape[0]
        and local_y >= 0
        and local_y < output.shape[1]
        and local_x >= 0
        and local_x < output.shape[2]
    ):
        output[local_z, local_y, local_x] = 1


@njit(cache=True)
def _draw_line(output, start, stop, z0, y0, x0):
    delta_z = stop[0] - start[0]
    delta_y = stop[1] - start[1]
    delta_x = stop[2] - start[2]
    steps = int(np.ceil(max(abs(delta_z), abs(delta_y), abs(delta_x))))
    if steps <= 0:
        _mark(output, start[0], start[1], start[2], z0, y0, x0)
        return

    inverse_steps = 1.0 / float(steps)
    for step in range(steps + 1):
        value = float(step) * inverse_steps
        _mark(
            output,
            start[0] + delta_z * value,
            start[1] + delta_y * value,
            start[2] + delta_x * value,
            z0,
            y0,
            x0,
        )


@njit(cache=True)
def _chebyshev_distance(start, stop):
    distance = abs(stop[0] - start[0])
    delta = abs(stop[1] - start[1])
    if delta > distance:
        distance = delta
    delta = abs(stop[2] - start[2])
    if delta > distance:
        distance = delta
    return distance


@njit(cache=True)
def _dense_steps(distance):
    steps = int(np.ceil(float(distance) * 2.0))
    if steps < 1:
        return 1
    return steps


@njit(cache=True)
def _draw_bilinear(output, p00, p01, p10, p11, z0, y0, x0):
    row_distance = _chebyshev_distance(p00, p10)
    distance = _chebyshev_distance(p01, p11)
    if distance > row_distance:
        row_distance = distance
    column_distance = _chebyshev_distance(p00, p01)
    distance = _chebyshev_distance(p10, p11)
    if distance > column_distance:
        column_distance = distance
    row_steps = _dense_steps(row_distance)
    column_steps = _dense_steps(column_distance)
    inverse_row_steps = 1.0 / float(row_steps)
    inverse_column_steps = 1.0 / float(column_steps)
    for row_step in range(row_steps + 1):
        row_t = float(row_step) * inverse_row_steps
        inverse_row = 1.0 - row_t
        left_z = p00[0] * inverse_row + p10[0] * row_t
        left_y = p00[1] * inverse_row + p10[1] * row_t
        left_x = p00[2] * inverse_row + p10[2] * row_t
        right_z = p01[0] * inverse_row + p11[0] * row_t
        right_y = p01[1] * inverse_row + p11[1] * row_t
        right_x = p01[2] * inverse_row + p11[2] * row_t
        for column_step in range(column_steps + 1):
            column_t = float(column_step) * inverse_column_steps
            inverse_column = 1.0 - column_t
            _mark(
                output,
                left_z * inverse_column + right_z * column_t,
                left_y * inverse_column + right_y * column_t,
                left_x * inverse_column + right_x * column_t,
                z0,
                y0,
                x0,
            )


@njit(cache=True)
def _draw_trilinear(
    output,
    lower_p00,
    lower_p01,
    lower_p10,
    lower_p11,
    upper_p00,
    upper_p01,
    upper_p10,
    upper_p11,
    z0,
    y0,
    x0,
):
    offset_distance = _chebyshev_distance(lower_p00, upper_p00)
    distance = _chebyshev_distance(lower_p01, upper_p01)
    if distance > offset_distance:
        offset_distance = distance
    distance = _chebyshev_distance(lower_p10, upper_p10)
    if distance > offset_distance:
        offset_distance = distance
    distance = _chebyshev_distance(lower_p11, upper_p11)
    if distance > offset_distance:
        offset_distance = distance

    row_distance = _chebyshev_distance(lower_p00, lower_p10)
    distance = _chebyshev_distance(lower_p01, lower_p11)
    if distance > row_distance:
        row_distance = distance
    distance = _chebyshev_distance(upper_p00, upper_p10)
    if distance > row_distance:
        row_distance = distance
    distance = _chebyshev_distance(upper_p01, upper_p11)
    if distance > row_distance:
        row_distance = distance

    column_distance = _chebyshev_distance(lower_p00, lower_p01)
    distance = _chebyshev_distance(lower_p10, lower_p11)
    if distance > column_distance:
        column_distance = distance
    distance = _chebyshev_distance(upper_p00, upper_p01)
    if distance > column_distance:
        column_distance = distance
    distance = _chebyshev_distance(upper_p10, upper_p11)
    if distance > column_distance:
        column_distance = distance

    offset_steps = _dense_steps(offset_distance)
    row_steps = _dense_steps(row_distance)
    column_steps = _dense_steps(column_distance)
    inverse_offset_steps = 1.0 / float(offset_steps)
    inverse_row_steps = 1.0 / float(row_steps)
    inverse_column_steps = 1.0 / float(column_steps)
    for offset_step in range(offset_steps + 1):
        offset_t = float(offset_step) * inverse_offset_steps
        inverse_offset = 1.0 - offset_t
        p00_z = lower_p00[0] * inverse_offset + upper_p00[0] * offset_t
        p00_y = lower_p00[1] * inverse_offset + upper_p00[1] * offset_t
        p00_x = lower_p00[2] * inverse_offset + upper_p00[2] * offset_t
        p01_z = lower_p01[0] * inverse_offset + upper_p01[0] * offset_t
        p01_y = lower_p01[1] * inverse_offset + upper_p01[1] * offset_t
        p01_x = lower_p01[2] * inverse_offset + upper_p01[2] * offset_t
        p10_z = lower_p10[0] * inverse_offset + upper_p10[0] * offset_t
        p10_y = lower_p10[1] * inverse_offset + upper_p10[1] * offset_t
        p10_x = lower_p10[2] * inverse_offset + upper_p10[2] * offset_t
        p11_z = lower_p11[0] * inverse_offset + upper_p11[0] * offset_t
        p11_y = lower_p11[1] * inverse_offset + upper_p11[1] * offset_t
        p11_x = lower_p11[2] * inverse_offset + upper_p11[2] * offset_t
        for row_step in range(row_steps + 1):
            row_t = float(row_step) * inverse_row_steps
            inverse_row = 1.0 - row_t
            left_z = p00_z * inverse_row + p10_z * row_t
            left_y = p00_y * inverse_row + p10_y * row_t
            left_x = p00_x * inverse_row + p10_x * row_t
            right_z = p01_z * inverse_row + p11_z * row_t
            right_y = p01_y * inverse_row + p11_y * row_t
            right_x = p01_x * inverse_row + p11_x * row_t
            for column_step in range(column_steps + 1):
                column_t = float(column_step) * inverse_column_steps
                inverse_column = 1.0 - column_t
                _mark(
                    output,
                    left_z * inverse_column + right_z * column_t,
                    left_y * inverse_column + right_y * column_t,
                    left_x * inverse_column + right_x * column_t,
                    z0,
                    y0,
                    x0,
                )


@njit(cache=True)
def _offset_position(positions, normals, row, column, offset, output_position):
    point_z = positions[row, column, 0]
    point_y = positions[row, column, 1]
    point_x = positions[row, column, 2]
    normal_z = normals[row, column, 0]
    normal_y = normals[row, column, 1]
    normal_x = normals[row, column, 2]
    if (
        not np.isfinite(point_z)
        or not np.isfinite(point_y)
        or not np.isfinite(point_x)
        or not np.isfinite(normal_z)
        or not np.isfinite(normal_y)
        or not np.isfinite(normal_x)
    ):
        return False

    magnitude = np.sqrt(
        normal_z * normal_z
        + normal_y * normal_y
        + normal_x * normal_x
    )
    if magnitude <= 1e-6:
        return False

    inverse_magnitude = 1.0 / magnitude
    output_position[0] = point_z + offset * normal_z * inverse_magnitude
    output_position[1] = point_y + offset * normal_y * inverse_magnitude
    output_position[2] = point_x + offset * normal_x * inverse_magnitude
    return True


@njit(cache=True)
def _project_mask_along_normals(
    mask, positions, normals, valid, crop_start, output, half_thickness
):
    z0 = int(crop_start[0])
    y0 = int(crop_start[1])
    x0 = int(crop_start[2])
    radius = int(np.ceil(half_thickness))
    current = np.empty((3,), dtype=np.float32)
    previous = np.empty((3,), dtype=np.float32)
    right = np.empty((3,), dtype=np.float32)
    down = np.empty((3,), dtype=np.float32)
    diagonal = np.empty((3,), dtype=np.float32)
    previous_right = np.empty((3,), dtype=np.float32)
    previous_down = np.empty((3,), dtype=np.float32)
    previous_diagonal = np.empty((3,), dtype=np.float32)
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            if mask[row, column] == 0 or not valid[row, column]:
                continue
            has_previous = False
            has_previous_cell = False
            for step in range(-radius, radius + 1):
                if abs(step) > half_thickness + 1e-6:
                    continue
                if not _offset_position(positions, normals, row, column, float(step), current):
                    break
                _mark(output, current[0], current[1], current[2], z0, y0, x0)
                if has_previous:
                    _draw_line(output, previous, current, z0, y0, x0)
                right_ok = (
                    column + 1 < mask.shape[1]
                    and mask[row, column + 1] != 0
                    and valid[row, column + 1]
                    and _offset_position(positions, normals, row, column + 1, float(step), right)
                )
                if right_ok:
                    _draw_line(output, current, right, z0, y0, x0)
                down_ok = (
                    row + 1 < mask.shape[0]
                    and mask[row + 1, column] != 0
                    and valid[row + 1, column]
                    and _offset_position(positions, normals, row + 1, column, float(step), down)
                )
                if down_ok:
                    _draw_line(output, current, down, z0, y0, x0)
                diagonal_ok = (
                    row + 1 < mask.shape[0]
                    and column + 1 < mask.shape[1]
                    and mask[row + 1, column + 1] != 0
                    and valid[row + 1, column + 1]
                    and _offset_position(
                        positions,
                        normals,
                        row + 1,
                        column + 1,
                        float(step),
                        diagonal,
                    )
                )
                cell_ok = right_ok and down_ok and diagonal_ok
                if cell_ok:
                    _draw_bilinear(
                        output,
                        current,
                        right,
                        down,
                        diagonal,
                        z0,
                        y0,
                        x0,
                    )
                    if has_previous_cell:
                        _draw_trilinear(
                            output,
                            previous,
                            previous_right,
                            previous_down,
                            previous_diagonal,
                            current,
                            right,
                            down,
                            diagonal,
                            z0,
                            y0,
                            x0,
                        )
                previous[:] = current
                if cell_ok:
                    previous_right[:] = right
                    previous_down[:] = down
                    previous_diagonal[:] = diagonal
                has_previous = True
                has_previous_cell = cell_ok


def project_binary_mask_along_normals(
    flat_mask: np.ndarray,
    positions_zyx: np.ndarray,
    normals_zyx: np.ndarray | None,
    valid_mask: np.ndarray,
    crop_bbox_zyx: tuple[int, int, int, int, int, int],
    *,
    half_thickness_voxels: float,
) -> np.ndarray:
    """Project a binary flat mask through a native normal-offset thickness."""
    if half_thickness_voxels < 0.0:
        raise ValueError("half_thickness_voxels must be >= 0")
    flat_mask = (np.asarray(flat_mask) > 0).astype(np.uint8, copy=False)
    if half_thickness_voxels <= 0.0:
        return project_flat_patch(
            flat_mask, positions_zyx, valid_mask, crop_bbox_zyx
        ) > 0
    if normals_zyx is None:
        raise ValueError("normals_zyx is required when projecting with thickness")
    positions_zyx = np.asarray(positions_zyx, dtype=np.float32)
    normals_zyx = np.asarray(normals_zyx, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    if positions_zyx.shape[:2] != flat_mask.shape or positions_zyx.shape[-1] != 3:
        raise ValueError(
            "positions_zyx must have shape (*flat_mask.shape, 3), "
            f"got flat_mask={flat_mask.shape!r}, positions={positions_zyx.shape!r}"
        )
    if normals_zyx.shape[:2] != flat_mask.shape or normals_zyx.shape[-1] != 3:
        raise ValueError(
            "normals_zyx must have shape (*flat_mask.shape, 3), "
            f"got flat_mask={flat_mask.shape!r}, normals={normals_zyx.shape!r}"
        )
    if valid_mask.shape != flat_mask.shape:
        raise ValueError(
            "valid_mask must match flat_mask shape, "
            f"got flat_mask={flat_mask.shape!r}, valid_mask={valid_mask.shape!r}"
        )
    z0, y0, x0, z1, y1, x1 = crop_bbox_zyx
    output = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.uint8)
    _project_mask_along_normals(
        np.ascontiguousarray(flat_mask),
        np.ascontiguousarray(positions_zyx),
        np.ascontiguousarray(normals_zyx),
        np.ascontiguousarray(valid_mask),
        np.asarray((z0, y0, x0), dtype=np.int64),
        output,
        float(half_thickness_voxels),
    )
    return output > 0


def project_labels_and_supervision(
    *,
    positions_zyx: np.ndarray,
    valid_mask: np.ndarray,
    inklabels_flat: np.ndarray,
    supervision_flat: np.ndarray,
    crop_bbox_zyx: tuple[int, int, int, int, int, int],
    normals_zyx: np.ndarray,
    label_half_thickness: float,
    background_half_thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project mutually exclusive binary foreground/background support."""
    labels = np.asarray(inklabels_flat) > 0
    supervision = np.asarray(supervision_flat) > 0
    background = supervision & ~labels
    labels_native = project_binary_mask_along_normals(
        labels,
        positions_zyx,
        normals_zyx,
        valid_mask,
        crop_bbox_zyx,
        half_thickness_voxels=label_half_thickness,
    )
    background_native = project_binary_mask_along_normals(
        background,
        positions_zyx,
        normals_zyx,
        valid_mask,
        crop_bbox_zyx,
        half_thickness_voxels=background_half_thickness,
    )
    background_native &= ~labels_native
    return (
        labels_native.astype(np.float32, copy=False),
        (labels_native | background_native).astype(np.float32, copy=False),
    )


def filter_support_components(
    *,
    support_bbox_yx: tuple[int, int, int, int],
    positions_zyx: np.ndarray,
    valid_mask: np.ndarray,
    inklabels_flat: np.ndarray,
    supervision_flat: np.ndarray,
    crop_bbox_zyx: tuple[int, int, int, int, int, int],
    patch_bbox_zyx: tuple[int, int, int, int, int, int],
    max_supervision_grid_distance: float | None,
):
    """Keep native connected components seeded by active flat supervision."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid_mask):
        return support_bbox_yx, positions_zyx, valid_mask, inklabels_flat, supervision_flat
    occupancy = project_flat_patch(
        np.ones(valid_mask.shape, dtype=np.uint8),
        positions_zyx,
        valid_mask,
        crop_bbox_zyx,
    ) > 0
    if not np.any(occupancy):
        return support_bbox_yx, positions_zyx, valid_mask, inklabels_flat, supervision_flat
    components = cc3d.connected_components(
        occupancy.astype(np.uint8, copy=False), connectivity=26
    )
    supervision_native = project_flat_patch(
        (np.asarray(supervision_flat) > 0).astype(np.uint8, copy=False),
        positions_zyx,
        valid_mask,
        crop_bbox_zyx,
    )
    kept = np.unique(components[supervision_native > 0])
    kept = kept[kept > 0]
    if kept.size == 0:
        return support_bbox_yx, positions_zyx, valid_mask, inklabels_flat, supervision_flat
    z0, y0, x0 = crop_bbox_zyx[:3]
    finite = valid_mask & np.isfinite(positions_zyx).all(axis=-1)
    if not np.any(finite):
        return support_bbox_yx, positions_zyx, valid_mask, inklabels_flat, supervision_flat
    mapped = positions_zyx[finite].astype(np.int64, copy=False)
    local = mapped - np.asarray((z0, y0, x0), dtype=np.int64)
    shape = np.asarray(components.shape, dtype=np.int64)
    within = (local >= 0).all(axis=1) & (local < shape).all(axis=1)
    if not np.any(within):
        return support_bbox_yx, positions_zyx, valid_mask, inklabels_flat, supervision_flat
    rows, columns = np.nonzero(finite)
    rows = rows[within]
    columns = columns[within]
    selected_local = local[within]
    ids = components[selected_local[:, 0], selected_local[:, 1], selected_local[:, 2]]
    keep_flat = np.isin(ids, kept)
    if not np.any(keep_flat):
        return (
            support_bbox_yx,
            positions_zyx,
            valid_mask,
            inklabels_flat,
            supervision_flat,
        )
    filtered = np.zeros_like(valid_mask)
    filtered[rows[keep_flat], columns[keep_flat]] = True
    support_y0, support_y1, support_x0, support_x1 = support_bbox_yx
    patch_y0, patch_y1 = patch_bbox_zyx[1], patch_bbox_zyx[4]
    patch_x0, patch_x1 = patch_bbox_zyx[2], patch_bbox_zyx[5]
    row0 = max(0, patch_y0 - support_y0)
    row1 = min(support_y1 - support_y0, patch_y1 - support_y0)
    column0 = max(0, patch_x0 - support_x0)
    column1 = min(support_x1 - support_x0, patch_x1 - support_x0)
    seed = np.zeros_like(filtered)
    if row1 > row0 and column1 > column0:
        seed[row0:row1, column0:column1] = (
            np.asarray(supervision_flat)[row0:row1, column0:column1] > 0
        )
    seed &= filtered
    if not np.any(seed):
        seed = (np.asarray(supervision_flat) > 0) & filtered
    if np.any(seed):
        flat_components, _ = ndimage.label(
            filtered, structure=np.ones((3, 3), dtype=np.uint8)
        )
        flat_component_ids = np.unique(flat_components[seed])
        flat_component_ids = flat_component_ids[flat_component_ids > 0]
        if flat_component_ids.size:
            filtered = np.isin(flat_components, flat_component_ids)
        if max_supervision_grid_distance is not None:
            max_distance = float(max_supervision_grid_distance)
            if not np.isfinite(max_distance) or max_distance < 0:
                raise ValueError(
                    "max_supervision_grid_distance must be finite and >= 0, "
                    f"got {max_distance!r}"
                )
            filtered &= distance_transform_edt(~seed) <= max_distance
    if not np.any(filtered):
        return support_bbox_yx, positions_zyx, filtered, inklabels_flat, supervision_flat
    row_ids = np.flatnonzero(np.any(filtered, axis=1))
    column_ids = np.flatnonzero(np.any(filtered, axis=0))
    row0, row1 = int(row_ids[0]), int(row_ids[-1]) + 1
    col0, col1 = int(column_ids[0]), int(column_ids[-1]) + 1
    return (
        (support_y0 + row0, support_y0 + row1, support_x0 + col0, support_x0 + col1),
        positions_zyx[row0:row1, col0:col1],
        filtered[row0:row1, col0:col1],
        np.asarray(inklabels_flat)[row0:row1, col0:col1],
        np.asarray(supervision_flat)[row0:row1, col0:col1],
    )
