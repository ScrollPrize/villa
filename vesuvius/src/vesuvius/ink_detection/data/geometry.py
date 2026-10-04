"""Native ZYX crop, tifxyz selection, surface projection, and mask geometry."""

from __future__ import annotations

import cc3d
import numpy as np
from numba import njit
from scipy import ndimage
from scipy.ndimage import distance_transform_edt


NATIVE_COARSE_PAD_LEVEL0_VOXELS = 20.0
SURFACE_MASK_MAX_DISTANCE_LEVEL0_VOXELS = 10.0
# Tifxyz interpolates full-resolution positions from the 4x4 stored points
# around them with Catmull-Rom weights. In 1D the negative weights at offset t
# are -t(1-t)^2/2 and -t^2(1-t)/2, which sum to t(1-t)/2 <= 1/8, so in 2D the
# negative weights sum to at most 1/8 + 1/8 + 2/64. An interpolated coordinate
# therefore lies within this fraction of the 16 points' range of their extremes.
CATMULL_ROM_NEGATIVE_WEIGHT = 0.28125


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


class StoredResolutionIndex:
    """Per-tile bounds of a stored-resolution tifxyz grid, for crop window queries.

    `maybe_select_flat_pixels` over the whole stored grid costs O(grid) for
    every crop. This index is built once per segment and keeps, for each
    `tile` x `tile` block of the grid, the min and max of its valid scaled
    positions. A query then tests only the blocks whose bounds can reach the
    crop, with the same comparisons, so it returns exactly the window the full
    scan returns.
    """

    def __init__(
        self,
        coarse_positions_zyx: np.ndarray,
        coarse_valid: np.ndarray,
        *,
        native_coordinate_scale: float = 1.0,
        tile: int = 32,
    ) -> None:
        self.scale = float(native_coordinate_scale)
        if self.scale <= 0.0:
            raise ValueError("native coordinate scale must be positive")
        self.tile = int(tile)
        self.positions = np.asarray(coarse_positions_zyx, dtype=np.float32)
        self.valid = np.asarray(coarse_valid, dtype=bool)
        height, width = self.positions.shape[:2]
        rows = -(-height // self.tile)
        columns = -(-width // self.tile)
        self.shape = (height, width)
        self.lower = np.empty((rows, columns, 3), dtype=np.float32)
        self.upper = np.empty((rows, columns, 3), dtype=np.float32)
        pad = ((0, 0), (0, columns * self.tile - width), (0, 0))
        for row in range(rows):
            y0, y1 = row * self.tile, min(height, (row + 1) * self.tile)
            scaled = self._scaled(y0, y1, 0, width)
            usable = self.valid[y0:y1] & np.isfinite(scaled).all(axis=-1)
            band = (y1 - y0, columns, self.tile, 3)
            # Tiles without usable points keep +inf / -inf and are never reached.
            for bound, fill, reduce in (
                (self.lower, np.inf, np.min),
                (self.upper, -np.inf, np.max),
            ):
                values = np.where(usable[..., None], scaled, np.float32(fill))
                values = np.pad(values, pad, constant_values=fill)
                bound[row] = reduce(values.reshape(band), axis=(0, 2))

    def _scaled(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        block = self.positions[y0:y1, x0:x1]
        # Same float32 product the full-grid path computes for these points.
        return block * self.scale if self.scale != 1.0 else block

    def window(self, crop_bbox_zyx) -> tuple[int, int, int, int] | None:
        """Return `(y0, y1, x0, x1)` as `maybe_select_flat_pixels` would, or None."""

        crop_start = np.asarray(crop_bbox_zyx[:3], dtype=np.int64)
        crop_stop = np.asarray(crop_bbox_zyx[3:], dtype=np.int64)
        reachable = (self.upper >= crop_start).all(axis=-1)
        reachable &= (self.lower < crop_stop).all(axis=-1)
        rows, columns = [], []
        for row, column in zip(*np.nonzero(reachable)):
            y0, x0 = int(row) * self.tile, int(column) * self.tile
            y1 = min(self.shape[0], y0 + self.tile)
            x1 = min(self.shape[1], x0 + self.tile)
            positions = self._scaled(y0, y1, x0, x1)
            within = self.valid[y0:y1, x0:x1] & np.isfinite(positions).all(axis=-1)
            within &= (positions >= crop_start).all(axis=-1)
            within &= (positions < crop_stop).all(axis=-1)
            if np.any(within):
                hit_rows = y0 + np.flatnonzero(np.any(within, axis=1))
                hit_columns = x0 + np.flatnonzero(np.any(within, axis=0))
                rows += (int(hit_rows[0]), int(hit_rows[-1]))
                columns += (int(hit_columns[0]), int(hit_columns[-1]))
        if not rows:
            return None
        return min(rows), max(rows) + 1, min(columns), max(columns) + 1


def _stored_resolution_bounds(
    patch_tifxyz,
    crop_bbox_zyx,
    *,
    coarse_native_pad: int,
    coarse_positions_zyx: np.ndarray,
    coarse_valid: np.ndarray,
    native_coordinate_scale: float,
    flat_grid_stride: int,
    coarse_index: StoredResolutionIndex | None = None,
):
    """Full-resolution `(y0, y1, x0, x1)` of the flat window to refine, or None."""
    scale = float(native_coordinate_scale)
    stride = int(flat_grid_stride)
    if scale <= 0.0 or stride <= 0:
        raise ValueError("native coordinate scale and flat grid stride must be positive")
    coarse_bbox = tuple(
        int(value) + (-coarse_native_pad if index < 3 else coarse_native_pad)
        for index, value in enumerate(crop_bbox_zyx)
    )
    if coarse_index is not None:
        if coarse_index.scale != scale:
            raise ValueError(
                f"coarse index was built for scale {coarse_index.scale!r}, "
                f"not {scale!r}"
            )
        coarse_window = coarse_index.window(coarse_bbox)
        if coarse_window is None:
            return None
        coarse_y0, coarse_y1, coarse_x0, coarse_x1 = coarse_window
        stored_h, stored_w = coarse_index.shape
    else:
        coarse_positions_zyx = np.asarray(coarse_positions_zyx, dtype=np.float32)
        if scale != 1.0:
            coarse_positions_zyx = coarse_positions_zyx * scale
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
    return sampled_y0, sampled_y1, sampled_x0, sampled_x1


def _read_flat_window(
    patch_tifxyz,
    bounds: tuple[int, int, int, int],
    *,
    native_coordinate_scale: float,
    flat_grid_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Native positions and validity of the flat-grid points inside `bounds`."""
    y0, y1, x0, x1 = bounds
    stride = int(flat_grid_stride)
    full_x, full_y, full_z, full_valid = patch_tifxyz[
        slice(y0, y1, stride),
        slice(x0, x1, stride),
    ]
    positions_zyx = np.stack([full_z, full_y, full_x], axis=-1).astype(
        np.float32, copy=False
    )
    scale = float(native_coordinate_scale)
    if scale != 1.0:
        positions_zyx *= scale
    return positions_zyx, np.asarray(full_valid, dtype=bool)


def _stored_resolution_window(
    patch_tifxyz,
    crop_bbox_zyx,
    *,
    coarse_native_pad: int,
    coarse_positions_zyx: np.ndarray,
    coarse_valid: np.ndarray,
    native_coordinate_scale: float,
    flat_grid_stride: int,
    coarse_index: StoredResolutionIndex | None = None,
):
    bounds = _stored_resolution_bounds(
        patch_tifxyz,
        crop_bbox_zyx,
        coarse_native_pad=coarse_native_pad,
        coarse_positions_zyx=coarse_positions_zyx,
        coarse_valid=coarse_valid,
        native_coordinate_scale=native_coordinate_scale,
        flat_grid_stride=flat_grid_stride,
        coarse_index=coarse_index,
    )
    if bounds is None:
        return None
    positions_zyx, valid = _read_flat_window(
        patch_tifxyz,
        bounds,
        native_coordinate_scale=native_coordinate_scale,
        flat_grid_stride=flat_grid_stride,
    )
    stride = int(flat_grid_stride)
    return positions_zyx, valid, bounds[0] // stride, bounds[2] // stride


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
    coarse_index: StoredResolutionIndex | None = None,
):
    """Refine a coarse tifxyz intersection to an exact full-grid support window.

    Pass a `StoredResolutionIndex` built from the same coarse positions, valid
    mask and scale to avoid scanning the whole stored grid for every crop; the
    result is the same.
    """
    window = _stored_resolution_window(
        patch_tifxyz,
        crop_bbox_zyx,
        coarse_native_pad=coarse_native_pad,
        coarse_positions_zyx=coarse_positions_zyx,
        coarse_valid=coarse_valid,
        native_coordinate_scale=native_coordinate_scale,
        flat_grid_stride=flat_grid_stride,
        coarse_index=coarse_index,
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


def catmull_rom_vertex_mask(patch_tifxyz) -> np.ndarray | None:
    """Stored-grid validity behind `patch_tifxyz`'s full-resolution positions.

    Returns None unless those positions come from Tifxyz's Catmull-Rom
    interpolation, the case `select_flat_pixel_bands_via_stored_resolution`
    can bound.
    """
    if (
        getattr(patch_tifxyz, "interp_method", None) != "catmull_rom"
        or getattr(patch_tifxyz, "resolution", None) != "full"
    ):
        return None
    return np.asarray(patch_tifxyz.valid_vertex_mask, dtype=bool)


def _stored_cells(patch_tifxyz, start: int, stop: int, step: int, axis: int):
    """Stored-grid cell of each sampled full-resolution row or column.

    Uses the float32 arithmetic of `Tifxyz.__getitem__`; a point in cell `c`
    is interpolated from stored rows or columns `c - 1` to `c + 2`.
    """
    stored = np.arange(start, stop, step).astype(np.float32)
    stored = stored * patch_tifxyz.get_scale_tuple()[axis]
    return np.asarray(stored, dtype=np.float32).astype(np.int64)


def _reduce_4x4(values: np.ndarray, reduce) -> np.ndarray:
    """Apply `reduce` over every 4x4 block of the first two axes."""
    values = reduce(reduce(values[:-3], values[1:-2]), reduce(values[2:-1], values[3:]))
    return reduce(
        reduce(values[:, :-3], values[:, 1:-2]),
        reduce(values[:, 2:-1], values[:, 3:]),
    )


def _merge_touching(boxes: list[list[int]]) -> list[list[int]]:
    """Merge `[y0, y1, x0, x1]` boxes until no two of them overlap or touch."""
    while True:
        merged: list[list[int]] = []
        for y0, y1, x0, x1 in boxes:
            for other in merged:
                rows_meet = y0 <= other[1] and other[0] <= y1
                if rows_meet and x0 <= other[3] and other[2] <= x1:
                    other[:] = (
                        min(y0, other[0]),
                        max(y1, other[1]),
                        min(x0, other[2]),
                        max(x1, other[3]),
                    )
                    break
            else:
                merged.append([y0, y1, x0, x1])
        if len(merged) == len(boxes):
            return merged
        boxes = merged


def _band_bounds(
    patch_tifxyz,
    bounds: tuple[int, int, int, int],
    crop_bbox_zyx,
    *,
    coarse_positions_zyx: np.ndarray,
    vertex_mask: np.ndarray,
    native_coordinate_scale: float,
    flat_grid_stride: int,
) -> list[tuple[int, int, int, int]]:
    """Split a full-resolution window into parts that hold every sampled
    point whose interpolated position can land in the crop. Parts never touch,
    so no two of their points are neighbours on the flat grid."""
    y0, y1, x0, x1 = bounds
    stride = int(flat_grid_stride)
    rows = _stored_cells(patch_tifxyz, y0, y1, stride, 0)
    columns = _stored_cells(patch_tifxyz, x0, x1, stride, 1)
    height, width = coarse_positions_zyx.shape[:2]
    if not rows.size or not columns.size or rows[0] >= height or columns[0] >= width:
        return []
    row0, row1 = int(rows[0]), int(min(rows[-1], height - 1)) + 1
    column0, column1 = int(columns[0]), int(min(columns[-1], width - 1)) + 1
    # A cell's points come from the 4x4 stored points around it, clamped to the
    # grid, and are valid only when all 16 are.
    near_rows = np.clip(np.arange(row0 - 1, row1 + 2), 0, height - 1)[:, None]
    near_columns = np.clip(np.arange(column0 - 1, column1 + 2), 0, width - 1)[None, :]
    controls = coarse_positions_zyx[near_rows, near_columns]
    lower = _reduce_4x4(controls, np.minimum).astype(np.float64)
    upper = _reduce_4x4(controls, np.maximum).astype(np.float64)
    usable = _reduce_4x4(vertex_mask[near_rows, near_columns], np.logical_and)
    scale = float(native_coordinate_scale)
    with np.errstate(invalid="ignore"):
        # One voxel on top of the bound covers float32 rounding of the result;
        # non-finite control points leave the cell unbounded.
        margin = (upper - lower) * CATMULL_ROM_NEGATIVE_WEIGHT + 1.0
        lower = np.where(np.isnan(margin), -np.inf, (lower - margin) * scale)
        upper = np.where(np.isnan(margin), np.inf, (upper + margin) * scale)
    candidate = usable & (upper >= np.asarray(crop_bbox_zyx[:3])).all(axis=-1)
    candidate &= (lower < np.asarray(crop_bbox_zyx[3:])).all(axis=-1)
    if not np.any(candidate):
        return []
    labels, _ = ndimage.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    parts = []
    for cells in ndimage.find_objects(labels):
        band_rows = np.searchsorted(rows, (row0 + cells[0].start, row0 + cells[0].stop))
        band_columns = np.searchsorted(
            columns, (column0 + cells[1].start, column0 + cells[1].stop)
        )
        if band_rows[1] > band_rows[0] and band_columns[1] > band_columns[0]:
            parts.append([*map(int, band_rows), *map(int, band_columns)])
    return [
        (
            y0 + top * stride,
            min(y1, y0 + bottom * stride),
            x0 + left * stride,
            min(x1, x0 + right * stride),
        )
        for top, bottom, left, right in _merge_touching(parts)
    ]


def select_flat_pixel_bands_via_stored_resolution(
    patch_tifxyz,
    crop_bbox_zyx,
    *,
    coarse_native_pad: int,
    coarse_positions_zyx: np.ndarray,
    coarse_valid: np.ndarray,
    native_coordinate_scale: float = 1.0,
    flat_grid_stride: int = 1,
    required: bool = True,
    coarse_index: StoredResolutionIndex | None = None,
    vertex_mask: np.ndarray | None = None,
) -> list[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]]:
    """`select_flat_pixels_via_stored_resolution`, as separate bands.

    A crop that meets several windings of a segment refines a window spanning
    all of them, mostly points far from the crop. Given `vertex_mask` from
    `catmull_rom_vertex_mask`, only the stored cells of that window whose
    interpolated points can land in the crop are refined, in disjoint bands.
    Returns `(support_bbox, positions, valid)` per band, each cropped to its
    in-crop points; together they hold exactly the in-crop points of the whole
    window, with the same positions. Without a vertex mask the whole window is
    the only band.
    """
    stride = int(flat_grid_stride)
    bounds = _stored_resolution_bounds(
        patch_tifxyz,
        crop_bbox_zyx,
        coarse_native_pad=coarse_native_pad,
        coarse_positions_zyx=coarse_positions_zyx,
        coarse_valid=coarse_valid,
        native_coordinate_scale=native_coordinate_scale,
        flat_grid_stride=stride,
        coarse_index=coarse_index,
    )
    if bounds is None:
        parts = []
    elif vertex_mask is None:
        parts = [bounds]
    else:
        parts = _band_bounds(
            patch_tifxyz,
            bounds,
            crop_bbox_zyx,
            coarse_positions_zyx=np.asarray(coarse_positions_zyx, dtype=np.float32),
            vertex_mask=vertex_mask,
            native_coordinate_scale=native_coordinate_scale,
            flat_grid_stride=stride,
        )
    bands = []
    for part in parts:
        positions_zyx, valid = _read_flat_window(
            patch_tifxyz,
            part,
            native_coordinate_scale=native_coordinate_scale,
            flat_grid_stride=stride,
        )
        selection = maybe_select_flat_pixels(positions_zyx, valid, crop_bbox_zyx)
        if selection is None:
            continue
        (local_y0, local_y1, local_x0, local_x1), support, support_valid = selection
        base_y0, base_x0 = part[0] // stride, part[2] // stride
        support_bbox = (
            base_y0 + local_y0,
            base_y0 + local_y1,
            base_x0 + local_x0,
            base_x0 + local_x1,
        )
        bands.append((support_bbox, support, support_valid))
    if not bands and required:
        raise ValueError(
            f"crop_bbox {crop_bbox_zyx!r} does not intersect any valid flat tifxyz pixels"
        )
    return bands


def paste_bands(boxes, arrays, bbox, *, fill) -> np.ndarray:
    """Place per-band flat arrays into one array covering `bbox`."""
    y0, y1, x0, x1 = bbox
    first = np.asarray(arrays[0])
    output = np.full((y1 - y0, x1 - x0) + first.shape[2:], fill, dtype=first.dtype)
    for (band_y0, band_y1, band_x0, band_x1), array in zip(boxes, arrays):
        top, bottom = max(band_y0, y0), min(band_y1, y1)
        left, right = max(band_x0, x0), min(band_x1, x1)
        if bottom > top and right > left:
            output[top - y0 : bottom - y0, left - x0 : right - x0] = np.asarray(array)[
                top - band_y0 : bottom - band_y0, left - band_x0 : right - band_x0
            ]
    return output


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


_UNREACHED = np.iinfo(np.int32).max
_OFFSETS_WITHIN: dict[float, np.ndarray] = {}


def _offsets_within(max_distance: float) -> np.ndarray:
    """Integer `(dz, dy, dx, squared norm)` rows with norm below `max_distance`."""

    offsets = _OFFSETS_WITHIN.get(max_distance)
    if offsets is None:
        radius = int(np.ceil(max_distance))
        steps = np.arange(-radius, radius + 1, dtype=np.int32)
        dz, dy, dx = np.meshgrid(steps, steps, steps, indexing="ij")
        squared = dz * dz + dy * dy + dx * dx
        keep = squared < max_distance * max_distance
        offsets = np.stack([dz[keep], dy[keep], dx[keep], squared[keep]], axis=1)
        _OFFSETS_WITHIN[max_distance] = offsets
    return offsets


@njit(cache=True)
def _local_squared_distance(occupancy, offsets, unreached):
    depth, height, width = occupancy.shape
    squared = np.full((depth, height, width), unreached, dtype=np.int32)
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if not occupancy[z, y, x]:
                    continue
                for index in range(offsets.shape[0]):
                    zz = z + offsets[index, 0]
                    yy = y + offsets[index, 1]
                    xx = x + offsets[index, 2]
                    if (
                        zz >= 0
                        and zz < depth
                        and yy >= 0
                        and yy < height
                        and xx >= 0
                        and xx < width
                        and offsets[index, 3] < squared[zz, yy, xx]
                    ):
                        squared[zz, yy, xx] = offsets[index, 3]
    return squared


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
    offsets = _offsets_within(float(max_distance_voxels))
    # Only distances below max_distance_voxels survive the clip, and each of those
    # is the norm of an integer offset in `offsets`, so searching those offsets
    # around the occupied voxels gives the same values as the full transform.
    # It costs one update per occupied voxel and offset, so keep the full
    # transform for large radii on dense crops.
    if np.count_nonzero(occupancy) * len(offsets) <= 16 * occupancy.size:
        squared = _local_squared_distance(occupancy, offsets, _UNREACHED)
        distance = np.full(occupancy.shape, np.inf)
        reached = squared != _UNREACHED
        distance[reached] = np.sqrt(squared[reached].astype(np.float64))
    else:
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


def _union_bbox(boxes) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        max(box[1] for box in boxes),
        min(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _nearby_groups(boxes, reach: int) -> list[list[int]]:
    """Group boxes so that boxes whose pixels may lie within `reach` of each
    other (Chebyshev bound) share a group."""
    parent = list(range(len(boxes)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            a, b = boxes[first], boxes[second]
            row_gap = max(0, b[0] - a[1] + 1, a[0] - b[1] + 1)
            column_gap = max(0, b[2] - a[3] + 1, a[2] - b[3] + 1)
            if max(row_gap, column_gap) <= reach:
                parent[root(first)] = root(second)
    groups: dict[int, list[int]] = {}
    for index in range(len(boxes)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def filter_support_bands(
    bands,
    supervision_flat,
    *,
    crop_bbox_zyx: tuple[int, int, int, int, int, int],
    patch_bbox_zyx: tuple[int, int, int, int, int, int],
    max_supervision_grid_distance: float | None,
) -> tuple[tuple[int, int, int, int], list[np.ndarray]]:
    """Keep native connected components seeded by active flat supervision.

    `bands` holds disjoint `(support_bbox, positions, valid)` flat bands and
    `supervision_flat` their supervision. The result is the same as for one
    window spanning all bands: native components, the seed choice and the
    kept bbox are global, while flat components and seed distances are
    computed per group of bands close enough to touch or to reach each
    other's seeds. Returns the support bbox and each band's mask of points to
    keep.
    """
    boxes = [tuple(int(value) for value in band[0]) for band in bands]
    valid = [np.asarray(band[2], dtype=bool) for band in bands]
    support_bbox = _union_bbox(boxes)
    if not any(np.any(mask) for mask in valid):
        return support_bbox, valid
    points = np.concatenate(
        [np.asarray(band[1])[mask] for band, mask in zip(bands, valid)]
    )[:, None]
    point_valid = np.ones(points.shape[:2], dtype=bool)
    occupancy = project_flat_patch(
        point_valid.astype(np.uint8),
        points,
        point_valid,
        crop_bbox_zyx,
    ) > 0
    if not np.any(occupancy):
        return support_bbox, valid
    components = cc3d.connected_components(
        occupancy.astype(np.uint8, copy=False), connectivity=26
    )
    supervised = np.concatenate(
        [(np.asarray(flat) > 0)[mask] for flat, mask in zip(supervision_flat, valid)]
    )
    supervision_native = project_flat_patch(
        supervised[:, None].astype(np.uint8),
        points,
        point_valid,
        crop_bbox_zyx,
    )
    kept = np.unique(components[supervision_native > 0])
    kept = kept[kept > 0]
    if kept.size == 0:
        return support_bbox, valid
    finite = np.isfinite(points[:, 0]).all(axis=-1)
    if not np.any(finite):
        return support_bbox, valid
    local = points[finite, 0].astype(np.int64, copy=False)
    local = local - np.asarray(crop_bbox_zyx[:3], dtype=np.int64)
    shape = np.asarray(components.shape, dtype=np.int64)
    within = (local >= 0).all(axis=1) & (local < shape).all(axis=1)
    if not np.any(within):
        return support_bbox, valid
    selected_local = local[within]
    keep_point = np.zeros(points.shape[0], dtype=bool)
    keep_point[np.flatnonzero(finite)[within]] = np.isin(
        components[selected_local[:, 0], selected_local[:, 1], selected_local[:, 2]],
        kept,
    )
    if not np.any(keep_point):
        return support_bbox, valid
    filtered, start = [], 0
    for mask in valid:
        count = int(np.count_nonzero(mask))
        band_filtered = np.zeros_like(mask)
        band_filtered[mask] = keep_point[start : start + count]
        filtered.append(band_filtered)
        start += count
    patch_y0, patch_y1 = patch_bbox_zyx[1], patch_bbox_zyx[4]
    patch_x0, patch_x1 = patch_bbox_zyx[2], patch_bbox_zyx[5]
    seeds = []
    for (y0, y1, x0, x1), flat, band_filtered in zip(boxes, supervision_flat, filtered):
        seed = np.zeros_like(band_filtered)
        row0, row1 = max(0, patch_y0 - y0), min(y1 - y0, patch_y1 - y0)
        column0, column1 = max(0, patch_x0 - x0), min(x1 - x0, patch_x1 - x0)
        if row1 > row0 and column1 > column0:
            seed[row0:row1, column0:column1] = (
                np.asarray(flat)[row0:row1, column0:column1] > 0
            )
        seeds.append(seed & band_filtered)
    if not any(np.any(seed) for seed in seeds):
        seeds = [
            (np.asarray(flat) > 0) & band_filtered
            for flat, band_filtered in zip(supervision_flat, filtered)
        ]
    if any(np.any(seed) for seed in seeds):
        reach = 1
        if max_supervision_grid_distance is not None:
            max_distance = float(max_supervision_grid_distance)
            if not np.isfinite(max_distance) or max_distance < 0:
                raise ValueError(
                    "max_supervision_grid_distance must be finite and >= 0, "
                    f"got {max_distance!r}"
                )
            reach = max(1, int(np.ceil(max_distance)))
        for group in _nearby_groups(boxes, reach):
            group_boxes = [boxes[index] for index in group]
            group_bbox = _union_bbox(group_boxes)
            group_filtered, group_seed = (
                paste_bands(
                    group_boxes,
                    [masks[index] for index in group],
                    group_bbox,
                    fill=False,
                )
                for masks in (filtered, seeds)
            )
            flat_components, _ = ndimage.label(
                group_filtered, structure=np.ones((3, 3), dtype=np.uint8)
            )
            flat_component_ids = np.unique(flat_components[group_seed])
            group_filtered = np.isin(
                flat_components, flat_component_ids[flat_component_ids > 0]
            )
            if max_supervision_grid_distance is not None and np.any(group_seed):
                group_filtered &= distance_transform_edt(~group_seed) <= max_distance
            for index, (y0, y1, x0, x1) in zip(group, group_boxes):
                filtered[index] = filtered[index] & group_filtered[
                    y0 - group_bbox[0] : y1 - group_bbox[0],
                    x0 - group_bbox[2] : x1 - group_bbox[2],
                ]
    kept_boxes = []
    for (y0, _, x0, _), band_filtered in zip(boxes, filtered):
        if np.any(band_filtered):
            row_ids = np.flatnonzero(np.any(band_filtered, axis=1))
            column_ids = np.flatnonzero(np.any(band_filtered, axis=0))
            kept_boxes.append(
                (
                    y0 + int(row_ids[0]),
                    y0 + int(row_ids[-1]) + 1,
                    x0 + int(column_ids[0]),
                    x0 + int(column_ids[-1]) + 1,
                )
            )
    if not kept_boxes:
        return support_bbox, filtered
    return _union_bbox(kept_boxes), filtered


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
    kept_bbox, (kept,) = filter_support_bands(
        [(support_bbox_yx, positions_zyx, valid_mask)],
        [supervision_flat],
        crop_bbox_zyx=crop_bbox_zyx,
        patch_bbox_zyx=patch_bbox_zyx,
        max_supervision_grid_distance=max_supervision_grid_distance,
    )
    row0 = kept_bbox[0] - support_bbox_yx[0]
    column0 = kept_bbox[2] - support_bbox_yx[2]
    window = (
        slice(row0, row0 + kept_bbox[1] - kept_bbox[0]),
        slice(column0, column0 + kept_bbox[3] - kept_bbox[2]),
    )
    return (
        kept_bbox,
        positions_zyx[window],
        kept[window],
        np.asarray(inklabels_flat)[window],
        np.asarray(supervision_flat)[window],
    )
