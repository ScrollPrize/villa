from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
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


_VOLUME_ROOT = tempfile.TemporaryDirectory(prefix="vesuvius-test-volumes-")
_VOLUME_PATHS: dict[str, str] = {}


def zarr_volume(data: np.ndarray, *, chunks: tuple[int, ...] = (16, 16, 16)) -> str:
    """Path of a local Zarr v2 array holding ``data``, for readers that open paths.

    Each distinct array is written once per test session.
    """
    array = np.ascontiguousarray(data)
    digest = hashlib.sha1(array.tobytes())
    digest.update(repr((array.shape, array.dtype.str, chunks)).encode())
    key = digest.hexdigest()
    path = _VOLUME_PATHS.get(key)
    if path is None:
        path = str(Path(_VOLUME_ROOT.name) / f"{key}.zarr")
        create_v2_array(Path(path), shape=array.shape, chunks=chunks, dtype=array.dtype)[:] = array
        _VOLUME_PATHS[key] = path
    return path
