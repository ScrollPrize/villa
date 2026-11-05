"""Shared OME-Zarr layout for flat label images embedded in ZYX volumes."""

from __future__ import annotations

from typing import Sequence


ARRAY_DIMENSIONS = ["z", "y", "x"]
DEFAULT_LEVELS = 6
VOLUME_DEPTH = 65
LABEL_SLICE = 32
CHUNK_SHAPE = (VOLUME_DEPTH, 128, 128)
LABEL_CODEC = {"cname": "zstd", "level": 3, "shuffle": 2}


def pyramid_shapes(
    image_shape: Sequence[int], levels: int
) -> list[tuple[int, int, int]]:
    """Return ceil-halved YX shapes embedded at the fixed Z depth."""

    if levels < 1:
        raise ValueError("levels must be at least 1")
    height, width = int(image_shape[0]), int(image_shape[1])
    shapes = []
    for _ in range(levels):
        shapes.append((VOLUME_DEPTH, height, width))
        height = (height + 1) // 2
        width = (width + 1) // 2
    return shapes
