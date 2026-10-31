"""Shared OME-Zarr layout for flat label images embedded in ZYX volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from vc_reader.reader import ZarrWriteGroup, ZarrWriter, create_group, ome_multiscales


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


def create_label_group(output_path: Path, *, levels: int, reduction: str = "nearest") -> ZarrWriteGroup:
    """Replace ``output_path`` with a group declaring a fixed-Z, halved-YX pyramid."""

    multiscales = ome_multiscales(name=output_path.stem, levels=levels - 1, factors=(1, 2, 2), reduction=reduction)
    return create_group(output_path, attrs={"multiscales": [multiscales]}, overwrite=True, auth=None)


def create_label_array(
    group: ZarrWriteGroup,
    name: str,
    *,
    shape: tuple[int, int, int],
    dtype: Any,
) -> ZarrWriter:
    """Create one fixed-chunk label level with slash-separated keys; all-zero chunks stay off disk."""

    array = group.create_array(
        name,
        shape=shape,
        chunks=CHUNK_SHAPE,
        dtype=dtype,
        fill_value=0,
        separator="/",
        overwrite=True,
        skip_empty=True,
        **LABEL_CODEC,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ARRAY_DIMENSIONS
    return array
