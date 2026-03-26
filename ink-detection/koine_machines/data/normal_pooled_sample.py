import numpy as np
import torch


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
