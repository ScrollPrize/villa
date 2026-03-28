from __future__ import annotations

import numpy as np

from koine_machines.common.common import open_zarr

def _label_tile_is_empty(label_tile) -> bool:
    tile = np.asarray(label_tile)
    if tile.size == 0:
        return True
    if np.issubdtype(tile.dtype, np.floating):
        return bool(np.all(tile < 0.01))
    if np.issubdtype(tile.dtype, np.integer):
        return bool(np.all(tile < 3))
    return bool(np.all(tile.astype(np.float32, copy=False) < 0.01))


def _candidate_patch_starts(limit: int, *, size: int, tile_size: int, stride: int) -> np.ndarray:
    max_tile_start = int(limit) - int(tile_size)
    max_patch_start = int(limit) - int(size)
    if max_tile_start < 0 or max_patch_start < 0:
        return np.zeros((0,), dtype=np.int32)

    starts = set()
    for tile_start in range(0, max_tile_start + 1, int(stride)):
        for offset in range(0, int(tile_size), int(size)):
            patch_start = int(tile_start + offset)
            if patch_start > max_patch_start:
                continue
            starts.add(patch_start)
    if not starts:
        return np.zeros((0,), dtype=np.int32)
    return np.asarray(sorted(starts), dtype=np.int32)


def _extract_patch_coordinates_full_grid(
    label_mask,
    valid_mask,
    *,
    size: int,
    tile_size: int,
    stride: int,
    filter_empty_tile: bool,
) -> tuple[tuple[int, int, int, int], ...]:
    max_x = int(valid_mask.shape[1] - tile_size)
    max_y = int(valid_mask.shape[0] - tile_size)
    if max_x < 0 or max_y < 0:
        return tuple()

    x1_list = range(0, max_x + 1, stride)
    y1_list = range(0, max_y + 1, stride)
    seen = set()
    xyxys: list[tuple[int, int, int, int]] = []

    for y_tile in y1_list:
        for x_tile in x1_list:
            if filter_empty_tile and _label_tile_is_empty(
                label_mask[y_tile:y_tile + tile_size, x_tile:x_tile + tile_size]
            ):
                continue

            tile_valid = valid_mask[y_tile:y_tile + tile_size, x_tile:x_tile + tile_size]
            tile_has_invalid = not bool(tile_valid.all())
            if tile_has_invalid and filter_empty_tile:
                continue

            for yi in range(0, tile_size, size):
                for xi in range(0, tile_size, size):
                    y1 = int(y_tile + yi)
                    x1 = int(x_tile + xi)
                    y2 = int(y1 + size)
                    x2 = int(x1 + size)
                    key = (x1, y1, x2, y2)
                    if key in seen:
                        continue
                    if tile_has_invalid and (not filter_empty_tile) and not bool(valid_mask[y1:y2, x1:x2].all()):
                        continue
                    seen.add(key)
                    xyxys.append(key)
    return tuple(xyxys)


def build_patch_index(
    mask,
    fragment_mask,
    *,
    size: int,
    tile_size: int,
    stride: int,
    filter_empty_tile: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = int(size)
    tile_size = int(tile_size)
    stride = int(stride)
    if size <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("size, tile_size, and stride must be positive")
    if tile_size % size != 0:
        raise ValueError(f"tile_size={tile_size} must be divisible by size={size}")

    label_mask = None if mask is None else np.asarray(mask)
    fragment_mask = np.asarray(fragment_mask)
    if label_mask is not None and label_mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={tuple(label_mask.shape)}")
    if fragment_mask.ndim != 2:
        raise ValueError(f"fragment_mask must be 2D, got shape={tuple(fragment_mask.shape)}")
    if label_mask is not None and tuple(label_mask.shape) != tuple(fragment_mask.shape):
        raise ValueError(
            "mask and fragment_mask must have the same shape, "
            f"got {tuple(label_mask.shape)} and {tuple(fragment_mask.shape)}"
        )
    if label_mask is None and bool(filter_empty_tile):
        raise ValueError("mask is required when filter_empty_tile=True")

    valid_mask = fragment_mask.astype(bool, copy=False)
    if filter_empty_tile:
        xyxys = np.asarray(
            _extract_patch_coordinates_full_grid(
                label_mask,
                valid_mask,
                size=size,
                tile_size=tile_size,
                stride=stride,
                filter_empty_tile=True,
            ),
            dtype=np.int64,
        )
        return (
            np.zeros((0, 4), dtype=np.int32),
            xyxys.reshape(-1, 4),
            np.full((int(xyxys.shape[0]),), -1, dtype=np.int32),
        )

    x_starts = _candidate_patch_starts(valid_mask.shape[1], size=size, tile_size=tile_size, stride=stride)
    y_starts = _candidate_patch_starts(valid_mask.shape[0], size=size, tile_size=tile_size, stride=stride)
    if int(x_starts.size) == 0 or int(y_starts.size) == 0 or not bool(valid_mask.any()):
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0, 4), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )

    raw_bboxes = np.asarray(component_bboxes(valid_mask, connectivity=2), dtype=np.int32)
    if int(raw_bboxes.shape[0]) == 0:
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0, 4), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )

    kept_bboxes: list[tuple[int, int, int, int]] = []
    xyxys: list[tuple[int, int, int, int]] = []
    bbox_indices: list[int] = []
    seen = set()
    for y0, y1, x0, x1 in raw_bboxes.tolist():
        max_x1 = int(x1) - int(size)
        max_y1 = int(y1) - int(size)
        if max_x1 < int(x0) or max_y1 < int(y0):
            continue
        x_lo = int(np.searchsorted(x_starts, int(x0), side="left"))
        x_hi = int(np.searchsorted(x_starts, int(max_x1), side="right"))
        y_lo = int(np.searchsorted(y_starts, int(y0), side="left"))
        y_hi = int(np.searchsorted(y_starts, int(max_y1), side="right"))
        local_rows: list[tuple[int, int, int, int]] = []
        for y_patch in y_starts[y_lo:y_hi].tolist():
            for x_patch in x_starts[x_lo:x_hi].tolist():
                x2 = int(x_patch + size)
                y2 = int(y_patch + size)
                key = (int(x_patch), int(y_patch), x2, y2)
                if key in seen:
                    continue
                if not bool(valid_mask[int(y_patch):y2, int(x_patch):x2].all()):
                    continue
                seen.add(key)
                local_rows.append(key)
        if not local_rows:
            continue
        bbox_index = int(len(kept_bboxes))
        kept_bboxes.append((int(y0), int(y1), int(x0), int(x1)))
        xyxys.extend(local_rows)
        bbox_indices.extend([bbox_index] * len(local_rows))

    if not xyxys:
        return (
            np.zeros((0, 4), dtype=np.int32),
            np.zeros((0, 4), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.asarray(kept_bboxes, dtype=np.int32),
        np.asarray(xyxys, dtype=np.int64),
        np.asarray(bbox_indices, dtype=np.int32),
    )


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
    patch_size = tuple(int(v) for v in segment.patch_size)
    if int(patch_size[1]) != int(patch_size[2]):
        raise ValueError(
            "subtiling patch finding requires square y/x patch_size, "
            f"got {tuple(int(v) for v in patch_size)}"
        )

    volume_auth = segment.config.get("volume_auth_json")
    supervision_mask = open_zarr(segment.supervision_mask, resolution=segment.scale, auth=volume_auth)
    inklabels = open_zarr(segment.inklabels, resolution=segment.scale, auth=volume_auth)
    validation_mask = None
    if segment.validation_mask is not None:
        validation_mask = open_zarr(segment.validation_mask, resolution=segment.scale, auth=volume_auth)

    surface = int(supervision_mask.shape[0] // 2)
    patch_size_yx = int(patch_size[1])
    default_stride = int(patch_size_yx * float(segment.config["patch_overlap"]))
    tile_size = int(segment.config.get("patch_finding_tile_size", patch_size_yx))
    stride = int(segment.config.get("patch_finding_stride", default_stride))
    filter_empty_tile = bool(segment.config.get("patch_finding_filter_empty_tile", False))

    _, xyxys, _ = build_patch_index(
        mask=inklabels[surface],
        fragment_mask=supervision_mask[surface],
        size=patch_size_yx,
        tile_size=tile_size,
        stride=stride,
        filter_empty_tile=filter_empty_tile,
    )

    training_patches = []
    validation_patches = []
    for x1, y1, x2, y2 in xyxys.tolist():
        del x2, y2
        patch_bbox_zyx = _surface_patch_bbox(surface, int(y1), int(x1), patch_size)
        supervision_patch = supervision_mask[
            surface,
            int(y1):int(y1) + patch_size[1],
            int(x1):int(x1) + patch_size[2],
        ]

        has_validation_supervision = False
        has_training_supervision = bool(supervision_patch.size > 0 and np.any(supervision_patch))
        if validation_mask is not None:
            validation_patch = validation_mask[
                surface,
                int(y1):int(y1) + patch_size[1],
                int(x1):int(x1) + patch_size[2],
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
            int(y1):int(y1) + patch_size[1],
            int(x1):int(x1) + patch_size[2],
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


__all__ = ["build_patch_index", "find_segment_patches"]
