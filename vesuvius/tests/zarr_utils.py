from __future__ import annotations

from pathlib import Path
from typing import Any

import zarr


def create_v2_array(
    path: Path,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: Any,
):
    """Create a slash-keyed Zarr v2 array with either supported Zarr API."""
    if hasattr(zarr.Group, "create_array"):
        return zarr.open_array(
            str(path),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=None,
            zarr_format=2,
            dimension_separator="/",
        )
    return zarr.open(
        str(path),
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=None,
        dimension_separator="/",
    )


def create_v2_group_array(
    root_path: Path,
    name: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: Any,
):
    """Create a Zarr v2 group and one slash-keyed child array."""
    if hasattr(zarr.Group, "create_array"):
        root = zarr.open_group(str(root_path), mode="w", zarr_format=2)
        return root.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=None,
            chunk_key_encoding={"name": "v2", "separator": "/"},
        )
    root = zarr.open_group(str(root_path), mode="w")
    return root.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=None,
        dimension_separator="/",
    )
