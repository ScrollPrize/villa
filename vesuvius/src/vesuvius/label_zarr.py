"""Shared OME-Zarr layout for flat label images embedded in ZYX volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from numcodecs import Blosc
import zarr


AXES = [
    {"name": "z", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]
ARRAY_DIMENSIONS = ["z", "y", "x"]
DEFAULT_LEVELS = 6
VOLUME_DEPTH = 65
LABEL_SLICE = 32
CHUNK_SHAPE = (VOLUME_DEPTH, 128, 128)
LABEL_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
_ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


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


def multiscales_metadata(name: str, levels: int) -> dict[str, object]:
    """Return OME-NGFF 0.4 metadata for the fixed-Z, halved-YX pyramid."""

    datasets = []
    for level in range(levels):
        scale_factor = float(2**level)
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1.0, scale_factor, scale_factor],
                    }
                ],
            }
        )
    return {
        "multiscales": [
            {
                "name": name,
                "version": "0.4",
                "axes": AXES,
                "datasets": datasets,
            }
        ]
    }


def create_label_group(output_path: Path, *, levels: int):
    """Create an explicit-Zarr-v2 group with the shared multiscale metadata."""

    kwargs: dict[str, object] = {"mode": "w"}
    if _ZARR_V3:
        kwargs["zarr_format"] = 2
    group = zarr.open_group(str(output_path), **kwargs)
    group.attrs.update(multiscales_metadata(output_path.stem, levels))
    return group


def create_label_array(
    group,
    name: str,
    *,
    shape: tuple[int, int, int],
    dtype: Any,
):
    """Create one fixed-chunk label level with v2 slash-separated keys."""

    array = create_v2_array(
        group,
        name,
        shape=shape,
        chunks=CHUNK_SHAPE,
        dtype=dtype,
        compressor=LABEL_COMPRESSOR,
        fill_value=0,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ARRAY_DIMENSIONS
    return array


def create_v2_array(
    group,
    name: str,
    *,
    shape,
    chunks,
    dtype,
    compressor,
    fill_value,
):
    """Create one explicit-Zarr-v2 array with slash-separated chunk keys."""

    kwargs = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "compressor": compressor,
        "fill_value": fill_value,
        "overwrite": True,
    }
    if _ZARR_V3:
        array = group.create_array(
            name,
            **kwargs,
            chunk_key_encoding={"name": "v2", "separator": "/"},
            config={"write_empty_chunks": False},
        )
    else:
        array = group.create_dataset(
            name,
            **kwargs,
            dimension_separator="/",
            write_empty_chunks=False,
        )
    return array
