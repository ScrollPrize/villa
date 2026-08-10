"""Default labeled and unlabeled flat-patch discovery."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from vesuvius.ink_detection.types import Patch, Segment


def surface_patch_bbox(
    surface: int, y0: int, x0: int, patch_size: tuple[int, int, int]
) -> tuple[int, int, int, int, int, int]:
    depth, height, width = patch_size
    z0 = int(surface - depth // 2)
    return z0, int(y0), int(x0), z0 + depth, int(y0) + height, int(x0) + width


def labeled_patch_coverage(label_patch: np.ndarray) -> float:
    """Return nonzero bounding-rectangle area over patch area."""
    patch = np.asarray(label_patch)
    if patch.size == 0:
        return 0.0
    labeled_y, labeled_x = np.nonzero(patch)
    if labeled_y.size == 0:
        return 0.0
    area = (
        int(labeled_y.max()) - int(labeled_y.min()) + 1
    ) * (int(labeled_x.max()) - int(labeled_x.min()) + 1)
    return float(area) / float(patch.size)


def combined_patch_discovery_support(
    supervision_slice: np.ndarray, validation_slice: np.ndarray | None = None
) -> np.ndarray:
    support = np.asarray(supervision_slice) > 0
    if validation_slice is not None:
        validation = np.asarray(validation_slice) > 0
        if validation.shape != support.shape:
            raise ValueError(
                f"validation support {validation.shape} != supervision {support.shape}"
            )
        support |= validation
    return support


def _scan_geometry(segment: Segment, open_volume: Callable):
    finding = segment.data_config.patch_finding
    if finding.scan_scale is not None and finding.scan_scale != segment.scale:
        scan_volume = open_volume(segment.image_volume, finding.scan_scale)
        full_volume = open_volume(segment.image_volume, segment.scale)
        scale_y = full_volume.shape[1] / scan_volume.shape[1]
        scale_x = full_volume.shape[2] / scan_volume.shape[2]
        surface = int(full_volume.shape[0] // 2)
        scan_scale = finding.scan_scale
    else:
        scan_volume = open_volume(segment.image_volume, segment.scale)
        scale_y = scale_x = 1.0
        surface = int(scan_volume.shape[0] // 2)
        scan_scale = segment.scale
    return scan_scale, scan_volume, int(scan_volume.shape[0] // 2), surface, scale_y, scale_x


def find_segment_patches(
    segment: Segment, open_volume: Callable[[str, int], object]
) -> tuple[list[Patch], list[Patch]]:
    """Find labeled train/validation patches in deterministic origin order."""
    if segment.supervision_mask is None or segment.inklabels is None:
        raise ValueError("labeled patch discovery requires supervision and inklabels")
    patch_size = segment.patch_size
    scan_scale, _, _, surface, scale_y, scale_x = _scan_geometry(
        segment, open_volume
    )
    supervision = open_volume(segment.supervision_mask, scan_scale)
    inklabels = open_volume(segment.inklabels, scan_scale)
    validation = (
        None
        if segment.validation_mask is None
        else open_volume(segment.validation_mask, scan_scale)
    )
    mask_surface = int(supervision.shape[0] // 2)
    validation_surface = (
        None if validation is None else int(validation.shape[0] // 2)
    )
    support = combined_patch_discovery_support(
        supervision[mask_surface],
        None if validation is None else validation[validation_surface],
    )
    ys, xs = np.nonzero(support)
    if len(ys) == 0:
        raise ValueError(f"{segment.supervision_mask} contains no nonzero voxels at scan scale")
    stride = max(
        1,
        int(
            round(
                int(patch_size[1] * segment.data_config.patch_finding.overlap)
                / scale_y
            )
        ),
    )
    corners = np.unique(
        np.stack([ys // stride * stride, xs // stride * stride], axis=1), axis=0
    )
    scan_h = max(1, int(round(patch_size[1] / scale_y)))
    scan_w = max(1, int(round(patch_size[2] / scale_x)))
    training: list[Patch] = []
    held_out: list[Patch] = []
    for y_scan, x_scan in corners.tolist():
        y_scan, x_scan = int(y_scan), int(x_scan)
        supervision_patch = supervision[
            mask_surface, y_scan : y_scan + scan_h, x_scan : x_scan + scan_w
        ]
        has_training = bool(supervision_patch.size and np.any(supervision_patch))
        has_validation = False
        if validation is not None:
            validation_patch = validation[
                validation_surface,
                y_scan : y_scan + scan_h,
                x_scan : x_scan + scan_w,
            ]
            has_validation = bool(validation_patch.size and np.any(validation_patch))
            if has_training and has_validation:
                has_training = bool(
                    np.any(np.asarray(supervision_patch) & ~np.asarray(validation_patch))
                )
        y0 = int(round(y_scan * scale_y))
        x0 = int(round(x_scan * scale_x))
        bbox = surface_patch_bbox(surface, y0, x0, patch_size)
        if has_validation:
            held_out.append(
                Patch(
                    segment=segment,
                    bbox=bbox,
                    is_validation=True,
                    supervision_mask_override=segment.validation_mask,
                )
            )
        label_patch = inklabels[
            mask_surface, y_scan : y_scan + scan_h, x_scan : x_scan + scan_w
        ]
        if has_training and labeled_patch_coverage(label_patch) >= (
            segment.data_config.patch_finding.min_labeled_coverage
        ):
            training.append(Patch(segment=segment, bbox=bbox))
    if not training and not held_out:
        raise ValueError(f"{segment.inklabels} produced no valid patches")
    return training, held_out


def find_segment_unlabeled_patches(
    segment: Segment, open_volume: Callable[[str, int], object]
) -> tuple[list[Patch], list[Patch]]:
    """Find image-supported flat patches outside all known supervision."""
    patch_size = segment.patch_size
    scan_scale, scan_volume, scan_surface, surface, scale_y, scale_x = _scan_geometry(
        segment, open_volume
    )
    supervision = (
        None
        if segment.supervision_mask is None
        else open_volume(segment.supervision_mask, scan_scale)
    )
    validation = (
        None
        if segment.validation_mask is None
        else open_volume(segment.validation_mask, scan_scale)
    )
    surface_slice = np.asarray(scan_volume[scan_surface])
    ys, xs = np.nonzero(surface_slice)
    if len(ys) == 0:
        raise ValueError(f"{segment.image_volume} contains no nonzero projected data")
    stride = max(
        1,
        int(
            round(
                int(patch_size[1] * segment.data_config.patch_finding.overlap)
                / scale_y
            )
        ),
    )
    corners = np.unique(
        np.stack([ys // stride * stride, xs // stride * stride], axis=1), axis=0
    )
    scan_h = max(1, int(round(patch_size[1] / scale_y)))
    scan_w = max(1, int(round(patch_size[2] / scale_x)))
    sup_surface = (
        scan_surface if supervision is None else int(supervision.shape[0] // 2)
    )
    val_surface = scan_surface if validation is None else int(validation.shape[0] // 2)
    training: list[Patch] = []
    for y_scan, x_scan in corners.tolist():
        y_scan, x_scan = int(y_scan), int(x_scan)
        image_patch = scan_volume[
            scan_surface, y_scan : y_scan + scan_h, x_scan : x_scan + scan_w
        ]
        coverage = 0.0 if image_patch.size == 0 else (
            float(np.count_nonzero(image_patch)) / float(image_patch.size)
        )
        if coverage < 0.25:
            continue
        if supervision is not None and np.any(
            supervision[
                sup_surface, y_scan : y_scan + scan_h, x_scan : x_scan + scan_w
            ]
        ):
            continue
        if validation is not None and np.any(
            validation[
                val_surface, y_scan : y_scan + scan_h, x_scan : x_scan + scan_w
            ]
        ):
            continue
        y0 = int(round(y_scan * scale_y))
        x0 = int(round(x_scan * scale_x))
        training.append(
            Patch(
                segment=segment,
                bbox=surface_patch_bbox(surface, y0, x0, patch_size),
                is_unlabeled=True,
            )
        )
    return training, []
