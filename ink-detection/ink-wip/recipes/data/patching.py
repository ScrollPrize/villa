from __future__ import annotations

import numpy as np


def _label_tile_is_empty(label_tile) -> bool:
    tile = np.asarray(label_tile)
    if tile.size == 0:
        return True
    if np.issubdtype(tile.dtype, np.floating):
        return bool(np.all(tile < 0.01))
    if np.issubdtype(tile.dtype, np.integer):
        return bool(np.all(tile < 3))
    return bool(np.all(tile.astype(np.float32, copy=False) < 0.01))


def extract_patch_coordinates(
    mask,
    fragment_mask,
    *,
    size: int,
    tile_size: int,
    stride: int,
    filter_empty_tile: bool = False,
) -> tuple[tuple[int, int, int, int], ...]:
    size = int(size)
    tile_size = int(tile_size)
    stride = int(stride)
    if size <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("size, tile_size, and stride must be positive")
    if tile_size % size != 0:
        raise ValueError(f"tile_size={tile_size} must be divisible by size={size}")

    label_mask = np.asarray(mask)
    fragment_mask = np.asarray(fragment_mask)
    if label_mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={tuple(label_mask.shape)}")
    if fragment_mask.ndim != 2:
        raise ValueError(f"fragment_mask must be 2D, got shape={tuple(fragment_mask.shape)}")
    if tuple(label_mask.shape) != tuple(fragment_mask.shape):
        raise ValueError(
            "mask and fragment_mask must have the same shape, "
            f"got {tuple(label_mask.shape)} and {tuple(fragment_mask.shape)}"
        )

    max_x = int(fragment_mask.shape[1] - tile_size)
    max_y = int(fragment_mask.shape[0] - tile_size)
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

            tile = fragment_mask[y_tile:y_tile + tile_size, x_tile:x_tile + tile_size]
            tile_has_invalid = bool(np.any(tile == 0))
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
                    if tile_has_invalid and (not filter_empty_tile) and np.any(fragment_mask[y1:y2, x1:x2] == 0):
                        continue
                    seen.add(key)
                    xyxys.append(key)

    return tuple(xyxys)


__all__ = ["extract_patch_coordinates"]
