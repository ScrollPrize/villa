from __future__ import annotations

import numpy as np

from koine_machines.common.common import open_zarr


def _surface_patch_bbox(surface: int, y0: int, x0: int, patch_size) -> tuple[int, int, int, int, int, int]:
    depth, height, width = (int(v) for v in patch_size)
    z0 = int(surface - depth // 2)
    return (
        z0,
        int(y0),
        int(x0),
        z0 + depth,
        int(y0) + height,
        int(x0) + width,
    )


def _labeled_patch_coverage(label_patch) -> float:
    patch = np.asarray(label_patch)
    if patch.size == 0:
        return 0.0

    labeled_ys, labeled_xs = np.nonzero(patch)
    if labeled_ys.size == 0:
        return 0.0

    labeled_area = (
        (int(labeled_ys.max()) - int(labeled_ys.min()) + 1)
        * (int(labeled_xs.max()) - int(labeled_xs.min()) + 1)
    )
    return float(labeled_area) / float(patch.size)


def _nonzero_patch_coverage(image_patch) -> float:
    patch = np.asarray(image_patch)
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch)) / float(patch.size)


def _resolve_scan_scale(segment, volume_auth):
    patch_finding_scale = segment.config.get("patch_finding_scale", None)
    if patch_finding_scale is not None and int(patch_finding_scale) != int(segment.scale):
        scan_scale = int(patch_finding_scale)
        print(f"Patch finding using scan scale {scan_scale} (volume scale {segment.scale})")
        scan_volume = open_zarr(segment.image_volume, resolution=scan_scale, auth=volume_auth)
        full_res_volume = open_zarr(segment.image_volume, resolution=segment.scale, auth=volume_auth)
        scale_factor_y = full_res_volume.shape[1] / scan_volume.shape[1]
        scale_factor_x = full_res_volume.shape[2] / scan_volume.shape[2]
        surface = int(full_res_volume.shape[0] // 2)
    else:
        scan_scale = int(segment.scale)
        scan_volume = open_zarr(segment.image_volume, resolution=segment.scale, auth=volume_auth)
        scale_factor_y = 1.0
        scale_factor_x = 1.0
        surface = int(scan_volume.shape[0] // 2)
    scan_surface = int(scan_volume.shape[0] // 2)
    return scan_scale, scan_volume, scan_surface, surface, scale_factor_y, scale_factor_x


def find_segment_patches(segment, patch_cls):
    volume_auth = segment.config.get("volume_auth_json")
    patch_size = tuple(int(v) for v in segment.patch_size)

    scan_scale, scan_volume, scan_surface, surface, scale_factor_y, scale_factor_x = _resolve_scan_scale(segment, volume_auth)

    # Open all masks at scan scale so per-patch checks are fast
    supervision_mask = open_zarr(segment.supervision_mask, resolution=scan_scale, auth=volume_auth)
    inklabels = open_zarr(segment.inklabels, resolution=scan_scale, auth=volume_auth)
    validation_mask = None
    if segment.validation_mask is not None:
        validation_mask = open_zarr(segment.validation_mask, resolution=scan_scale, auth=volume_auth)

    # Masks may not downsample Z the same way the image volume does,
    # so compute the surface index from the mask's own Z dimension.
    mask_scan_surface = int(supervision_mask.shape[0] // 2)

    scan_patch_h = max(1, int(round(patch_size[1] / scale_factor_y)))
    scan_patch_w = max(1, int(round(patch_size[2] / scale_factor_x)))

    surface_slice = np.asarray(supervision_mask[mask_scan_surface])
    ys, xs = np.nonzero(surface_slice)
    if len(ys) == 0:
        raise ValueError(f"{segment.supervision_mask} contains no nonzero voxels at scan scale")

    stride_scan = max(1, int(round(
        int(patch_size[1] * float(segment.config["patch_overlap"])) / scale_factor_y
    )))
    if stride_scan <= 0:
        raise ValueError(f"patch_overlap produced non-positive stride")

    patch_corners_scan = np.unique(
        np.stack([ys // stride_scan * stride_scan, xs // stride_scan * stride_scan], axis=1),
        axis=0,
    )

    training_patches = []
    validation_patches = []
    for y0_scan, x0_scan in patch_corners_scan.tolist():
        y0_s, x0_s = int(y0_scan), int(x0_scan)
        supervision_patch = supervision_mask[
            mask_scan_surface,
            y0_s:y0_s + scan_patch_h,
            x0_s:x0_s + scan_patch_w,
        ]

        has_validation_supervision = False
        has_training_supervision = bool(supervision_patch.size > 0 and np.any(supervision_patch))
        if validation_mask is not None:
            validation_patch = validation_mask[
                mask_scan_surface,
                y0_s:y0_s + scan_patch_h,
                x0_s:x0_s + scan_patch_w,
            ]
            has_validation_supervision = bool(validation_patch.size > 0 and np.any(validation_patch))
            if has_training_supervision and has_validation_supervision:
                has_training_supervision = bool(
                    np.any(np.asarray(supervision_patch) & ~np.asarray(validation_patch))
                )

        # Scale coordinates back to segment.scale for the patch bbox
        y0 = int(round(y0_scan * scale_factor_y))
        x0 = int(round(x0_scan * scale_factor_x))
        patch_bbox_zyx = _surface_patch_bbox(surface, y0, x0, patch_size)

        if has_validation_supervision:
            validation_patches.append(
                patch_cls(
                    segment=segment,
                    bbox=patch_bbox_zyx,
                    is_validation=True,
                    supervision_mask_override=segment.validation_mask,
                )
            )

        label_patch = inklabels[
            mask_scan_surface,
            y0_s:y0_s + scan_patch_h,
            x0_s:x0_s + scan_patch_w,
        ]
        if has_training_supervision and _labeled_patch_coverage(label_patch) >= float(segment.config["patch_min_labeled_coverage"]):
            training_patches.append(
                patch_cls(
                    segment=segment,
                    bbox=patch_bbox_zyx,
                )
            )

    if len(training_patches) == 0 and len(validation_patches) == 0:
        raise ValueError(f"{segment.inklabels} produced no valid patches")

    return training_patches, validation_patches


def find_segment_unlabeled_patches(segment, patch_cls):
    volume_auth = segment.config.get("volume_auth_json")
    patch_size = tuple(int(v) for v in segment.patch_size)
    min_data_coverage = float(segment.config.get("unlabeled_patch_min_data_coverage", 0.15))

    scan_scale, scan_volume, scan_surface, surface, scale_factor_y, scale_factor_x = _resolve_scan_scale(segment, volume_auth)

    # Open masks at scan scale too
    supervision_mask = None
    if segment.supervision_mask is not None:
        supervision_mask = open_zarr(segment.supervision_mask, resolution=scan_scale, auth=volume_auth)
    validation_mask = None
    if segment.validation_mask is not None:
        validation_mask = open_zarr(segment.validation_mask, resolution=scan_scale, auth=volume_auth)

    # Masks may not downsample Z the same way the image volume does
    sup_mask_surface = int(supervision_mask.shape[0] // 2) if supervision_mask is not None else scan_surface
    val_mask_surface = int(validation_mask.shape[0] // 2) if validation_mask is not None else scan_surface

    surface_slice = np.asarray(scan_volume[scan_surface])
    ys, xs = np.nonzero(surface_slice)
    if len(ys) == 0:
        raise ValueError(f"{segment.image_volume} contains no nonzero projected data")

    stride_scan = max(1, int(round(
        int(patch_size[1] * float(segment.config["patch_overlap"])) / scale_factor_y
    )))
    if stride_scan <= 0:
        raise ValueError(f"patch_overlap produced non-positive stride")

    patch_corners_scan = np.unique(
        np.stack([ys // stride_scan * stride_scan, xs // stride_scan * stride_scan], axis=1),
        axis=0,
    )

    scan_patch_h = max(1, int(round(patch_size[1] / scale_factor_y)))
    scan_patch_w = max(1, int(round(patch_size[2] / scale_factor_x)))

    training_patches = []
    for y0_scan, x0_scan in patch_corners_scan.tolist():
        y0_s, x0_s = int(y0_scan), int(x0_scan)
        image_patch = scan_volume[
            scan_surface,
            y0_s:y0_s + scan_patch_h,
            x0_s:x0_s + scan_patch_w,
        ]
        if _nonzero_patch_coverage(image_patch) < 0.25:
            continue

        if supervision_mask is not None:
            supervision_patch = supervision_mask[
                sup_mask_surface,
                y0_s:y0_s + scan_patch_h,
                x0_s:x0_s + scan_patch_w,
            ]
            if np.any(supervision_patch):
                continue
        if validation_mask is not None:
            validation_patch = validation_mask[
                val_mask_surface,
                y0_s:y0_s + scan_patch_h,
                x0_s:x0_s + scan_patch_w,
            ]
            if np.any(validation_patch):
                continue

        # Scale coordinates back to segment.scale for the patch bbox
        y0 = int(round(y0_scan * scale_factor_y))
        x0 = int(round(x0_scan * scale_factor_x))
        patch_bbox_zyx = _surface_patch_bbox(surface, y0, x0, patch_size)

        training_patches.append(
            patch_cls(
                segment=segment,
                bbox=patch_bbox_zyx,
                is_unlabeled=True,
            )
        )

    if len(training_patches) == 0:
        print(f"Warning: {segment.image_volume} produced no valid unlabeled patches, skipping")

    return training_patches, []


__all__ = ["find_segment_patches", "find_segment_unlabeled_patches"]
