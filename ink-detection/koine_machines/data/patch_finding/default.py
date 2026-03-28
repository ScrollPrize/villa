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


def find_segment_patches(segment, patch_cls):
    volume_auth = segment.config.get("volume_auth_json")
    patch_size = tuple(int(v) for v in segment.patch_size)

    supervision_mask = open_zarr(segment.supervision_mask, resolution=segment.scale, auth=volume_auth)
    inklabels = open_zarr(segment.inklabels, resolution=segment.scale, auth=volume_auth)
    validation_mask = None
    if segment.validation_mask is not None:
        validation_mask = open_zarr(segment.validation_mask, resolution=segment.scale, auth=volume_auth)

    surface = int(supervision_mask.shape[0] // 2)
    surface_slice = supervision_mask[surface]
    ys, xs = np.nonzero(surface_slice)
    if len(ys) == 0:
        raise ValueError(f"{segment.supervision_mask} contains no nonzero voxels")

    stride = int(patch_size[1] * float(segment.config["patch_overlap"]))
    if stride <= 0:
        raise ValueError(f"patch_overlap produced non-positive stride={stride}")

    patch_corners_top_left = np.unique(
        np.stack([ys // stride * stride, xs // stride * stride], axis=1),
        axis=0,
    )

    training_patches = []
    validation_patches = []
    for y0, x0 in patch_corners_top_left.tolist():
        patch_bbox_zyx = _surface_patch_bbox(surface, int(y0), int(x0), patch_size)
        supervision_patch = supervision_mask[
            surface,
            int(y0):int(y0) + patch_size[1],
            int(x0):int(x0) + patch_size[2],
        ]

        has_validation_supervision = False
        has_training_supervision = bool(supervision_patch.size > 0 and np.any(supervision_patch))
        if validation_mask is not None:
            validation_patch = validation_mask[
                surface,
                int(y0):int(y0) + patch_size[1],
                int(x0):int(x0) + patch_size[2],
            ]
            has_validation_supervision = bool(validation_patch.size > 0 and np.any(validation_patch))
            if has_training_supervision and has_validation_supervision:
                has_training_supervision = bool(
                    np.any(np.asarray(supervision_patch) & ~np.asarray(validation_patch))
                )
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
            surface,
            int(y0):int(y0) + patch_size[1],
            int(x0):int(x0) + patch_size[2],
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


__all__ = ["find_segment_patches"]
