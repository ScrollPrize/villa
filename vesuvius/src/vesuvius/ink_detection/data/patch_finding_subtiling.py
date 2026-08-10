"""Filter-empty-tile subtiling patch discovery."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from vesuvius.ink_detection.types import Patch, Segment


def _label_tile_is_empty(label_tile: np.ndarray) -> bool:
    tile = np.asarray(label_tile)
    if tile.size == 0:
        return True
    if np.issubdtype(tile.dtype, np.floating):
        return bool(np.all(tile < 0.01))
    if np.issubdtype(tile.dtype, np.integer):
        return bool(np.all(tile < 3))
    return bool(np.all(tile.astype(np.float32, copy=False) < 0.01))


def build_patch_index(
    mask: np.ndarray,
    fragment_mask: np.ndarray,
    *,
    size: int,
    tile_size: int,
    stride: int,
    filter_empty_tile: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact working full-grid index; unfiltered mode is undefined."""
    size, tile_size, stride = int(size), int(tile_size), int(stride)
    if size <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("size, tile_size, and stride must be positive")
    if tile_size % size:
        raise ValueError(f"tile_size={tile_size} must be divisible by size={size}")
    if not filter_empty_tile:
        raise ValueError(
            "patch_finding_type='subtiling' does not support "
            "patch_finding_filter_empty_tile=false; set "
            "patch_finding_filter_empty_tile=true"
        )
    label_mask = np.asarray(mask)
    valid_mask = np.asarray(fragment_mask)
    if label_mask.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError(
            f"mask and fragment_mask must be 2D, got {label_mask.shape!r} and {valid_mask.shape!r}"
        )
    if label_mask.shape != valid_mask.shape:
        raise ValueError(
            "mask and fragment_mask must have the same shape, "
            f"got {label_mask.shape!r} and {valid_mask.shape!r}"
        )
    valid_mask = valid_mask.astype(bool, copy=False)
    seen: set[tuple[int, int, int, int]] = set()
    coordinates: list[tuple[int, int, int, int]] = []
    max_y = valid_mask.shape[0] - tile_size
    max_x = valid_mask.shape[1] - tile_size
    if max_y >= 0 and max_x >= 0:
        for y_tile in range(0, max_y + 1, stride):
            for x_tile in range(0, max_x + 1, stride):
                if _label_tile_is_empty(
                    label_mask[
                        y_tile : y_tile + tile_size,
                        x_tile : x_tile + tile_size,
                    ]
                ):
                    continue
                if not bool(
                    valid_mask[
                        y_tile : y_tile + tile_size,
                        x_tile : x_tile + tile_size,
                    ].all()
                ):
                    continue
                for yi in range(0, tile_size, size):
                    for xi in range(0, tile_size, size):
                        key = (
                            x_tile + xi,
                            y_tile + yi,
                            x_tile + xi + size,
                            y_tile + yi + size,
                        )
                        if key not in seen:
                            seen.add(key)
                            coordinates.append(key)
    xyxys = np.asarray(coordinates, dtype=np.int64).reshape(-1, 4)
    return (
        np.zeros((0, 4), dtype=np.int32),
        xyxys,
        np.full((xyxys.shape[0],), -1, dtype=np.int32),
    )


def find_segment_patches(
    segment: Segment, open_volume: Callable[[str, int], object]
) -> tuple[list[Patch], list[Patch]]:
    """Find labeled patches using filter-empty-tile subtiling."""
    if segment.patch_size[1] != segment.patch_size[2]:
        raise ValueError(
            "subtiling patch finding requires square y/x patch_size, "
            f"got {segment.patch_size!r}"
        )
    if segment.supervision_mask is None or segment.inklabels is None:
        raise ValueError("subtiling requires supervision and inklabels")
    supervision = open_volume(segment.supervision_mask, segment.scale)
    inklabels = open_volume(segment.inklabels, segment.scale)
    validation = (
        None
        if segment.validation_mask is None
        else open_volume(segment.validation_mask, segment.scale)
    )
    surface = int(supervision.shape[0] // 2)
    patch_size = segment.patch_size
    finding = segment.data_config.patch_finding
    size = patch_size[1]
    tile_size = size if finding.tile_size is None else finding.tile_size
    default_stride = int(size * finding.overlap)
    stride = default_stride if finding.stride is None else finding.stride
    _, xyxys, _ = build_patch_index(
        inklabels[surface],
        supervision[surface],
        size=size,
        tile_size=tile_size,
        stride=stride,
        filter_empty_tile=finding.filter_empty_tile,
    )
    training: list[Patch] = []
    held_out: list[Patch] = []
    for x1, y1, _, _ in xyxys.tolist():
        z0 = surface - patch_size[0] // 2
        bbox = (
            z0,
            int(y1),
            int(x1),
            z0 + patch_size[0],
            int(y1) + patch_size[1],
            int(x1) + patch_size[2],
        )
        supervision_patch = supervision[
            surface,
            int(y1) : int(y1) + patch_size[1],
            int(x1) : int(x1) + patch_size[2],
        ]
        has_training = bool(supervision_patch.size and np.any(supervision_patch))
        has_validation = False
        if validation is not None:
            validation_patch = validation[
                surface,
                int(y1) : int(y1) + patch_size[1],
                int(x1) : int(x1) + patch_size[2],
            ]
            has_validation = bool(validation_patch.size and np.any(validation_patch))
            if has_training and has_validation:
                has_training = bool(
                    np.any(np.asarray(supervision_patch) & ~np.asarray(validation_patch))
                )
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
            surface,
            int(y1) : int(y1) + patch_size[1],
            int(x1) : int(x1) + patch_size[2],
        ]
        labeled_y, labeled_x = np.nonzero(label_patch)
        coverage = 0.0
        if labeled_y.size:
            coverage = float(
                (int(labeled_y.max()) - int(labeled_y.min()) + 1)
                * (int(labeled_x.max()) - int(labeled_x.min()) + 1)
            ) / float(label_patch.size)
        if has_training and coverage >= finding.min_labeled_coverage:
            training.append(Patch(segment=segment, bbox=bbox))
    if not training and not held_out:
        raise ValueError(f"{segment.inklabels} produced no valid patches")
    return training, held_out
