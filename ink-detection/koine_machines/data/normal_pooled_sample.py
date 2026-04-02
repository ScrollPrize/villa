import cc3d
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import torch


def _select_flat_pixels_for_native_crop(patch_zyxs, valid_mask, crop_bbox):
    patch_zyxs = np.asarray(patch_zyxs)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    crop_start = np.asarray(crop_bbox[:3], dtype=np.int64)
    crop_stop = np.asarray(crop_bbox[3:], dtype=np.int64)

    finite_mask = np.isfinite(patch_zyxs).all(axis=-1)
    within = valid_mask & finite_mask
    within &= patch_zyxs[..., 0] >= crop_start[0]
    within &= patch_zyxs[..., 0] < crop_stop[0]
    within &= patch_zyxs[..., 1] >= crop_start[1]
    within &= patch_zyxs[..., 1] < crop_stop[1]
    within &= patch_zyxs[..., 2] >= crop_start[2]
    within &= patch_zyxs[..., 2] < crop_stop[2]
    if not np.any(within):
        raise ValueError(f"crop_bbox {crop_bbox!r} does not intersect any valid flat tifxyz pixels")

    row_hits = np.any(within, axis=1)
    col_hits = np.any(within, axis=0)
    row_indices = np.flatnonzero(row_hits)
    col_indices = np.flatnonzero(col_hits)
    support_y0 = int(row_indices[0])
    support_y1 = int(row_indices[-1]) + 1
    support_x0 = int(col_indices[0])
    support_x1 = int(col_indices[-1]) + 1
    return (
        (support_y0, support_y1, support_x0, support_x1),
        patch_zyxs[support_y0:support_y1, support_x0:support_x1],
        within[support_y0:support_y1, support_x0:support_x1],
    )


def _select_flat_pixels_for_native_crop_via_stored_resolution(
    patch_tifxyz,
    crop_bbox,
    *,
    coarse_native_pad=20,
    coarse_patch_zyxs=None,
    coarse_valid=None,
    return_halo=False,
):
    coarse_native_pad = int(coarse_native_pad)
    coarse_crop_bbox = (
        int(crop_bbox[0]) - coarse_native_pad,
        int(crop_bbox[1]) - coarse_native_pad,
        int(crop_bbox[2]) - coarse_native_pad,
        int(crop_bbox[3]) + coarse_native_pad,
        int(crop_bbox[4]) + coarse_native_pad,
        int(crop_bbox[5]) + coarse_native_pad,
    )

    if coarse_patch_zyxs is None:
        coarse_patch_zyxs = np.asarray(
            patch_tifxyz.get_zyxs(stored_resolution=True),
            dtype=np.float32,
        )
    else:
        coarse_patch_zyxs = np.asarray(coarse_patch_zyxs, dtype=np.float32)

    if coarse_valid is None:
        coarse_valid = np.isfinite(coarse_patch_zyxs).all(axis=-1)
        coarse_valid &= (coarse_patch_zyxs >= 0).all(axis=-1)
    else:
        coarse_valid = np.asarray(coarse_valid, dtype=bool)

    (coarse_y0, coarse_y1, coarse_x0, coarse_x1), _, _ = _select_flat_pixels_for_native_crop(
        coarse_patch_zyxs,
        coarse_valid,
        coarse_crop_bbox,
    )

    stored_h, stored_w = (int(v) for v in coarse_patch_zyxs.shape[:2])
    full_h, full_w = (int(v) for v in patch_tifxyz.full_resolution_shape)
    if stored_h <= 0 or stored_w <= 0:
        raise ValueError(f"stored-resolution tifxyz grid must have positive shape, got {(stored_h, stored_w)!r}")

    factor_y = full_h / float(stored_h)
    factor_x = full_w / float(stored_w)

    # Expand by one stored cell before mapping back to full resolution so the
    # exact full-res refinement can't miss intersections near a coarse edge.
    coarse_y0 = max(0, coarse_y0 - 1)
    coarse_y1 = min(stored_h, coarse_y1 + 1)
    coarse_x0 = max(0, coarse_x0 - 1)
    coarse_x1 = min(stored_w, coarse_x1 + 1)

    full_y0 = max(0, int(np.floor(coarse_y0 * factor_y)))
    full_y1 = min(full_h, int(np.ceil(coarse_y1 * factor_y)))
    full_x0 = max(0, int(np.floor(coarse_x0 * factor_x)))
    full_x1 = min(full_w, int(np.ceil(coarse_x1 * factor_x)))

    full_x, full_y, full_z, full_valid = patch_tifxyz[full_y0:full_y1, full_x0:full_x1]
    full_patch_zyxs = np.stack([full_z, full_y, full_x], axis=-1)
    (local_y0, local_y1, local_x0, local_x1), support_patch_zyxs, support_valid = _select_flat_pixels_for_native_crop(
        full_patch_zyxs,
        full_valid,
        crop_bbox,
    )
    support_bbox = (
        full_y0 + local_y0,
        full_y0 + local_y1,
        full_x0 + local_x0,
        full_x0 + local_x1,
    )
    if not return_halo:
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
        )

    halo_local_y0 = max(0, local_y0 - 1)
    halo_local_y1 = min(full_patch_zyxs.shape[0], local_y1 + 1)
    halo_local_x0 = max(0, local_x0 - 1)
    halo_local_x1 = min(full_patch_zyxs.shape[1], local_x1 + 1)
    support_patch_zyxs_halo = full_patch_zyxs[halo_local_y0:halo_local_y1, halo_local_x0:halo_local_x1]
    support_valid_halo = full_valid[halo_local_y0:halo_local_y1, halo_local_x0:halo_local_x1]
    trim_slices = (
        slice(local_y0 - halo_local_y0, local_y1 - halo_local_y0),
        slice(local_x0 - halo_local_x0, local_x1 - halo_local_x0),
    )
    return (
        support_bbox,
        support_patch_zyxs,
        support_valid,
        support_patch_zyxs_halo,
        support_valid_halo,
        trim_slices,
    )


def _project_flat_patch_to_native_crop(flat_patch, patch_zyxs, valid_mask, crop_bbox):
    z0, y0, x0, z1, y1, x1 = (int(v) for v in crop_bbox)
    output = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.asarray(flat_patch).dtype)

    positive_mask = np.asarray(flat_patch) != 0
    valid_mask = np.asarray(valid_mask, dtype=bool) & positive_mask
    if not np.any(valid_mask):
        return output

    patch_zyxs = np.asarray(patch_zyxs)
    finite_mask = np.isfinite(patch_zyxs).all(axis=-1)
    valid_mask &= finite_mask
    if not np.any(valid_mask):
        return output

    mapped_zyxs = patch_zyxs[valid_mask].astype(np.int64, copy=False)
    local_zyxs = mapped_zyxs - np.asarray((z0, y0, x0), dtype=np.int32)
    within_crop = (
        (local_zyxs[:, 0] >= 0)
        & (local_zyxs[:, 0] < output.shape[0])
        & (local_zyxs[:, 1] >= 0)
        & (local_zyxs[:, 1] < output.shape[1])
        & (local_zyxs[:, 2] >= 0)
        & (local_zyxs[:, 2] < output.shape[2])
    )
    if not np.any(within_crop):
        return output

    local_zyxs = local_zyxs[within_crop]
    values = np.asarray(flat_patch)[valid_mask][within_crop]
    flat_indices = np.ravel_multi_index(local_zyxs.T, output.shape)
    np.maximum.at(output.reshape(-1), flat_indices, values)
    return output


def _project_valid_surface_mask_to_native_crop(patch_zyxs, valid_mask, crop_bbox):
    flat_mask = np.ones(np.asarray(valid_mask).shape, dtype=np.float32)
    surface_occupancy = _project_flat_patch_to_native_crop(flat_mask, patch_zyxs, valid_mask, crop_bbox)
    surface_occupancy = surface_occupancy > 0
    if not np.any(surface_occupancy):
        return surface_occupancy.astype(np.float32)

    max_distance_voxels = 10.0
    distance = distance_transform_edt(~surface_occupancy)
    surface_distance_field = np.clip(1.0 - (distance / max_distance_voxels), 0.0, 1.0)
    return surface_distance_field.astype(np.float32, copy=False)


def _dilate_binary_mask_via_distance_transform(mask, *, max_distance_voxels):
    mask = np.asarray(mask, dtype=bool)
    max_distance_voxels = float(max_distance_voxels)
    if max_distance_voxels <= 0.0 or not np.any(mask):
        return mask

    distance = distance_transform_edt(~mask)
    return distance <= max_distance_voxels


def _project_flat_labels_and_supervision_to_native_crop(
    *,
    support_patch_zyxs,
    support_valid,
    support_inklabels_flat_patch,
    support_supervision_flat_patch,
    crop_bbox,
    label_dilation_distance=0.0,
    supervision_dilation_distance=0.0,
):
    inklabels_crop = _project_flat_patch_to_native_crop(
        (np.asarray(support_inklabels_flat_patch) > 0).astype(np.uint8, copy=False),
        support_patch_zyxs,
        support_valid,
        crop_bbox,
    ) > 0
    supervision_crop = _project_flat_patch_to_native_crop(
        (np.asarray(support_supervision_flat_patch) > 0).astype(np.uint8, copy=False),
        support_patch_zyxs,
        support_valid,
        crop_bbox,
    ) > 0

    inklabels_crop = _dilate_binary_mask_via_distance_transform(
        inklabels_crop,
        max_distance_voxels=label_dilation_distance,
    )
    supervision_background = supervision_crop & ~inklabels_crop
    supervision_background = _dilate_binary_mask_via_distance_transform(
        supervision_background,
        max_distance_voxels=supervision_dilation_distance,
    )
    supervision_background &= ~inklabels_crop
    supervision_crop = inklabels_crop | supervision_background
    return (
        inklabels_crop.astype(np.float32, copy=False),
        supervision_crop.astype(np.float32, copy=False),
    )


def _filter_support_by_flat_seeded_component(
    *,
    support_bbox,
    support_valid,
    support_supervision_flat_patch,
    patch_bbox,
    max_grid_distance=None,
):
    support_valid = np.asarray(support_valid, dtype=bool)
    if not np.any(support_valid):
        return support_valid

    support_y0, support_y1, support_x0, support_x1 = (int(v) for v in support_bbox)
    patch_y0, patch_y1, patch_x0, patch_x1 = (
        int(patch_bbox[1]),
        int(patch_bbox[4]),
        int(patch_bbox[2]),
        int(patch_bbox[5]),
    )

    row0 = max(0, patch_y0 - support_y0)
    row1 = min(support_y1 - support_y0, patch_y1 - support_y0)
    col0 = max(0, patch_x0 - support_x0)
    col1 = min(support_x1 - support_x0, patch_x1 - support_x0)

    seed_mask = np.zeros_like(support_valid, dtype=bool)
    if row1 > row0 and col1 > col0:
        seed_mask[row0:row1, col0:col1] = (
            np.asarray(support_supervision_flat_patch)[row0:row1, col0:col1] > 0
        )
    seed_mask &= support_valid

    if not np.any(seed_mask):
        seed_mask = (np.asarray(support_supervision_flat_patch) > 0) & support_valid
    if not np.any(seed_mask):
        return support_valid

    labeled_components, _ = ndimage.label(
        support_valid,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    kept_component_ids = np.unique(labeled_components[seed_mask])
    kept_component_ids = kept_component_ids[kept_component_ids > 0]
    if kept_component_ids.size == 0:
        return support_valid

    filtered_valid = np.isin(labeled_components, kept_component_ids)
    if max_grid_distance is not None:
        filtered_valid &= _filter_support_by_2d_distance_from_supervision(
            filtered_valid,
            seed_mask.astype(np.uint8, copy=False),
            max_grid_distance=max_grid_distance,
        )
    return filtered_valid


def _tighten_support_window(
    support_bbox,
    support_patch_zyxs,
    support_valid,
    support_inklabels_flat_patch,
    support_supervision_flat_patch,
):
    support_valid = np.asarray(support_valid, dtype=bool)
    if not np.any(support_valid):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    row_hits = np.any(support_valid, axis=1)
    col_hits = np.any(support_valid, axis=0)
    row_indices = np.flatnonzero(row_hits)
    col_indices = np.flatnonzero(col_hits)
    row_start = int(row_indices[0])
    row_stop = int(row_indices[-1]) + 1
    col_start = int(col_indices[0])
    col_stop = int(col_indices[-1]) + 1
    support_y0, _, support_x0, _ = (int(v) for v in support_bbox)
    tightened_bbox = (
        support_y0 + row_start,
        support_y0 + row_stop,
        support_x0 + col_start,
        support_x0 + col_stop,
    )
    return (
        tightened_bbox,
        np.asarray(support_patch_zyxs)[row_start:row_stop, col_start:col_stop],
        support_valid[row_start:row_stop, col_start:col_stop],
        np.asarray(support_inklabels_flat_patch)[row_start:row_stop, col_start:col_stop],
        np.asarray(support_supervision_flat_patch)[row_start:row_stop, col_start:col_stop],
    )


def _filter_support_by_2d_distance_from_supervision(
    support_valid,
    support_supervision_flat_patch,
    *,
    max_grid_distance,
):
    support_valid = np.asarray(support_valid, dtype=bool)
    if not np.any(support_valid):
        return support_valid

    max_grid_distance = float(max_grid_distance)
    if not np.isfinite(max_grid_distance) or max_grid_distance < 0:
        raise ValueError(
            f"max_grid_distance must be a finite value >= 0, got {max_grid_distance!r}"
        )

    supervision_seed_mask = (np.asarray(support_supervision_flat_patch) > 0) & support_valid
    if not np.any(supervision_seed_mask):
        return support_valid

    grid_distance = distance_transform_edt(~supervision_seed_mask)
    return support_valid & (grid_distance <= max_grid_distance)


def _slice_support_halo_for_subwindow(
    support_patch_zyxs_halo,
    support_valid_halo,
    trim_slices,
    base_support_bbox,
    subwindow_bbox,
):
    base_support_y0, _, base_support_x0, _ = (int(v) for v in base_support_bbox)
    subwindow_y0, subwindow_y1, subwindow_x0, subwindow_x1 = (int(v) for v in subwindow_bbox)
    row_start = subwindow_y0 - base_support_y0
    row_stop = subwindow_y1 - base_support_y0
    col_start = subwindow_x0 - base_support_x0
    col_stop = subwindow_x1 - base_support_x0

    halo_row_start = max(0, int(trim_slices[0].start) + row_start - 1)
    halo_row_stop = min(support_patch_zyxs_halo.shape[0], int(trim_slices[0].start) + row_stop + 1)
    halo_col_start = max(0, int(trim_slices[1].start) + col_start - 1)
    halo_col_stop = min(support_patch_zyxs_halo.shape[1], int(trim_slices[1].start) + col_stop + 1)

    return (
        np.asarray(support_patch_zyxs_halo)[halo_row_start:halo_row_stop, halo_col_start:halo_col_stop],
        np.asarray(support_valid_halo)[halo_row_start:halo_row_stop, halo_col_start:halo_col_stop],
        (
            slice(int(trim_slices[0].start) + row_start - halo_row_start, int(trim_slices[0].start) + row_stop - halo_row_start),
            slice(int(trim_slices[1].start) + col_start - halo_col_start, int(trim_slices[1].start) + col_stop - halo_col_start),
        ),
    )


def _filter_support_components_by_active_supervision(
    *,
    support_bbox,
    support_patch_zyxs,
    support_valid,
    support_inklabels_flat_patch,
    support_supervision_flat_patch,
    crop_bbox,
    patch_bbox,
    max_supervision_grid_distance=None,
):
    support_valid = np.asarray(support_valid, dtype=bool)
    if not np.any(support_valid):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    occupancy = _project_flat_patch_to_native_crop(
        np.ones(np.asarray(support_valid).shape, dtype=np.uint8),
        support_patch_zyxs,
        support_valid,
        crop_bbox,
    )
    occupancy = occupancy > 0
    if not np.any(occupancy):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    components = cc3d.connected_components(
        occupancy.astype(np.uint8, copy=False),
        connectivity=26,
    )
    supervision_native = _project_flat_patch_to_native_crop(
        (np.asarray(support_supervision_flat_patch) > 0).astype(np.uint8, copy=False),
        support_patch_zyxs,
        support_valid,
        crop_bbox,
    )
    kept_component_ids = np.unique(components[supervision_native > 0])
    kept_component_ids = kept_component_ids[kept_component_ids > 0]
    if kept_component_ids.size == 0:
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    patch_zyxs = np.asarray(support_patch_zyxs)
    finite_mask = np.isfinite(patch_zyxs).all(axis=-1)
    flat_valid = support_valid & finite_mask
    if not np.any(flat_valid):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    z0, y0, x0, z1, y1, x1 = (int(v) for v in crop_bbox)
    mapped_zyxs = patch_zyxs[flat_valid].astype(np.int64, copy=False)
    local_zyxs = mapped_zyxs - np.asarray((z0, y0, x0), dtype=np.int64)
    within_crop = (
        (local_zyxs[:, 0] >= 0)
        & (local_zyxs[:, 0] < (z1 - z0))
        & (local_zyxs[:, 1] >= 0)
        & (local_zyxs[:, 1] < (y1 - y0))
        & (local_zyxs[:, 2] >= 0)
        & (local_zyxs[:, 2] < (x1 - x0))
    )
    if not np.any(within_crop):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    row_indices, col_indices = np.nonzero(flat_valid)
    row_indices = row_indices[within_crop]
    col_indices = col_indices[within_crop]
    local_zyxs = local_zyxs[within_crop]
    component_ids = components[
        local_zyxs[:, 0],
        local_zyxs[:, 1],
        local_zyxs[:, 2],
    ]
    keep_flat = np.isin(component_ids, kept_component_ids)
    if not np.any(keep_flat):
        return (
            support_bbox,
            support_patch_zyxs,
            support_valid,
            support_inklabels_flat_patch,
            support_supervision_flat_patch,
        )

    filtered_valid = np.zeros_like(support_valid, dtype=bool)
    filtered_valid[row_indices[keep_flat], col_indices[keep_flat]] = True
    filtered_valid = _filter_support_by_flat_seeded_component(
        support_bbox=support_bbox,
        support_valid=filtered_valid,
        support_supervision_flat_patch=support_supervision_flat_patch,
        patch_bbox=patch_bbox,
        max_grid_distance=max_supervision_grid_distance,
    )
    return _tighten_support_window(
        support_bbox,
        support_patch_zyxs,
        filtered_valid,
        support_inklabels_flat_patch,
        support_supervision_flat_patch,
    )


def _compute_normals_local_zyx_from_position_halo(
    support_patch_zyxs_halo,
    support_valid_halo,
    trim_slices,
):
    support_patch_zyxs_halo = np.asarray(support_patch_zyxs_halo, dtype=np.float32)
    support_valid_halo = np.asarray(support_valid_halo, dtype=bool)

    z = support_patch_zyxs_halo[..., 0]
    y = support_patch_zyxs_halo[..., 1]
    x = support_patch_zyxs_halo[..., 2]

    halo_h, halo_w = z.shape
    nx = np.full((halo_h, halo_w), np.nan, dtype=np.float32)
    ny = np.full((halo_h, halo_w), np.nan, dtype=np.float32)
    nz = np.full((halo_h, halo_w), np.nan, dtype=np.float32)

    if halo_h >= 3 and halo_w >= 3:
        interior_valid = (
            support_valid_halo[1:-1, 1:-1]
            & support_valid_halo[1:-1, :-2]
            & support_valid_halo[1:-1, 2:]
            & support_valid_halo[:-2, 1:-1]
            & support_valid_halo[2:, 1:-1]
        )

        tx_x = x[1:-1, 2:] - x[1:-1, :-2]
        tx_y = y[1:-1, 2:] - y[1:-1, :-2]
        tx_z = z[1:-1, 2:] - z[1:-1, :-2]

        ty_x = x[2:, 1:-1] - x[:-2, 1:-1]
        ty_y = y[2:, 1:-1] - y[:-2, 1:-1]
        ty_z = z[2:, 1:-1] - z[:-2, 1:-1]

        n_x = ty_y * tx_z - ty_z * tx_y
        n_y = ty_z * tx_x - ty_x * tx_z
        n_z = ty_x * tx_y - ty_y * tx_x

        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        norm = np.where(norm > 1e-10, norm, np.nan)

        n_x = np.where(interior_valid, n_x / norm, np.nan)
        n_y = np.where(interior_valid, n_y / norm, np.nan)
        n_z = np.where(interior_valid, n_z / norm, np.nan)

        nx[1:-1, 1:-1] = n_x.astype(np.float32, copy=False)
        ny[1:-1, 1:-1] = n_y.astype(np.float32, copy=False)
        nz[1:-1, 1:-1] = n_z.astype(np.float32, copy=False)

    row_slice, col_slice = trim_slices
    return np.stack(
        [
            nz[row_slice, col_slice],
            ny[row_slice, col_slice],
            nx[row_slice, col_slice],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _build_normal_pooled_flat_metadata(
    *,
    support_patch_zyxs,
    support_valid,
    support_patch_zyxs_halo,
    support_valid_halo,
    trim_slices,
    support_inklabels_flat_patch,
    support_supervision_flat_patch,
    crop_bbox,
):
    normals_local_zyx = _compute_normals_local_zyx_from_position_halo(
        support_patch_zyxs_halo,
        support_valid_halo,
        trim_slices,
    )

    flat_points_local_zyx = (
        np.asarray(support_patch_zyxs, dtype=np.float32)
        - np.asarray(crop_bbox[:3], dtype=np.float32)
    )

    flat_target = (np.asarray(support_inklabels_flat_patch) > 0).astype(np.float32, copy=False)
    flat_supervision = (np.asarray(support_supervision_flat_patch) > 0).astype(np.float32, copy=False)

    flat_valid = np.asarray(support_valid, dtype=bool)
    flat_valid &= np.isfinite(flat_points_local_zyx).all(axis=-1)
    flat_valid &= np.isfinite(normals_local_zyx).all(axis=-1)
    crop_shape_zyx = (
        np.asarray(crop_bbox[3:], dtype=np.float32)
        - np.asarray(crop_bbox[:3], dtype=np.float32)
    )
    flat_valid &= (flat_points_local_zyx >= 0.0).all(axis=-1)
    flat_valid &= (flat_points_local_zyx <= (crop_shape_zyx - 1.0)).all(axis=-1)

    normal_magnitudes = np.linalg.norm(normals_local_zyx, axis=-1)
    flat_valid &= normal_magnitudes > 1e-6

    safe_normals = np.zeros_like(normals_local_zyx, dtype=np.float32)
    safe_points = np.zeros_like(flat_points_local_zyx, dtype=np.float32)
    if np.any(flat_valid):
        safe_normals[flat_valid] = (
            normals_local_zyx[flat_valid]
            / normal_magnitudes[flat_valid, None]
        ).astype(np.float32, copy=False)
        safe_points[flat_valid] = flat_points_local_zyx[flat_valid].astype(np.float32, copy=False)

    return {
        'flat_target': torch.from_numpy(flat_target).float().unsqueeze(0),
        'flat_supervision': torch.from_numpy(flat_supervision).float().unsqueeze(0),
        'flat_valid': torch.from_numpy(flat_valid.astype(np.float32, copy=False)).float().unsqueeze(0),
        'flat_points_local_zyx': torch.from_numpy(safe_points).float(),
        'flat_normals_local_zyx': torch.from_numpy(safe_normals).float(),
    }


def _pack_normal_pooled_augmentation_data(data):
    flat_valid_mask = data['flat_valid'][0] > 0
    keypoints = data['flat_points_local_zyx'][flat_valid_mask]
    flat_normals = data['flat_normals_local_zyx'][flat_valid_mask]

    augmentation_data = {
        'image': data['image'],
        'surface_mask': data['surface_mask'],
        'regression_keys': ['surface_mask'],
        'keypoints': keypoints,
        'flat_normals': flat_normals,
        'vector_keys': ['flat_normals'],
        'crop_shape': tuple(int(v) for v in data['image'].shape[1:]),
    }
    return augmentation_data, flat_valid_mask


def _restore_normal_pooled_augmentation_data(augmented, original_data, flat_valid_mask):
    restored = {
        'image': augmented['image'],
        'surface_mask': augmented['surface_mask'],
        'flat_target': original_data['flat_target'],
        'flat_supervision': original_data['flat_supervision'],
        'flat_valid': original_data['flat_valid'],
        'flat_points_local_zyx': torch.zeros_like(original_data['flat_points_local_zyx']),
        'flat_normals_local_zyx': torch.zeros_like(original_data['flat_normals_local_zyx']),
    }
    restored['flat_points_local_zyx'][flat_valid_mask] = augmented['keypoints']
    restored['flat_normals_local_zyx'][flat_valid_mask] = augmented['flat_normals']
    return restored
