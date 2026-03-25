import numpy as np
import torch


def _build_normal_pooled_flat_metadata(
    *,
    patch_tifxyz,
    support_bbox,
    support_patch_zyxs,
    support_valid,
    support_inklabels_flat_patch,
    support_supervision_flat_patch,
    crop_bbox,
):
    support_y0, support_y1, support_x0, support_x1 = (int(v) for v in support_bbox)
    nx, ny, nz = patch_tifxyz.get_normals(support_y0, support_y1, support_x0, support_x1)
    normals_local_zyx = np.stack([nz, ny, nx], axis=-1).astype(np.float32, copy=False)

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
