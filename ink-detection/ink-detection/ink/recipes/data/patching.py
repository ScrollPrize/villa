from __future__ import annotations

import numpy as np

from ink.recipes.components import component_bboxes


def _label_tile_is_empty(label_tile) -> bool:
    tile = np.asarray(label_tile)
    if tile.size == 0:
        return True
    if np.issubdtype(tile.dtype, np.floating):
        return bool(np.all(tile < 0.01))
    if np.issubdtype(tile.dtype, np.integer):
        return bool(np.all(tile < 3))
    return bool(np.all(tile.astype(np.float32, copy=False) < 0.01))


def candidate_patch_starts(limit: int, *, size: int, tile_size: int, stride: int) -> np.ndarray:
    max_tile_start = int(limit) - int(tile_size)
    max_patch_start = int(limit) - int(size)
    if max_tile_start < 0 or max_patch_start < 0:
        return np.zeros((0,), dtype=np.int32)

    tile_starts = np.arange(0, max_tile_start + 1, int(stride), dtype=np.int32)
    offsets = np.arange(0, int(tile_size), int(size), dtype=np.int32)
    if int(tile_starts.size) == 0 or int(offsets.size) == 0:
        return np.zeros((0,), dtype=np.int32)
    patch_starts = (tile_starts[:, None] + offsets[None, :]).reshape(-1)
    patch_starts = patch_starts[patch_starts <= max_patch_start]
    if int(patch_starts.size) == 0:
        return np.zeros((0,), dtype=np.int32)
    return np.unique(patch_starts).astype(np.int32, copy=False)


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

    x_starts = candidate_patch_starts(valid_mask.shape[1], size=size, tile_size=tile_size, stride=stride)
    y_starts = candidate_patch_starts(valid_mask.shape[0], size=size, tile_size=tile_size, stride=stride)
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

__all__ = [
    "build_patch_index",
    "candidate_patch_starts",
]
