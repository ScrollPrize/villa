from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from vc_reader.reader import ZarrWriter, create_group


def create_v2_array(
    path: Path,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: Any,
) -> ZarrWriter:
    """Create an uncompressed, slash-keyed Zarr v2 array, replacing anything at ``path``."""
    return ZarrWriter.create(path, shape=shape, chunks=chunks, dtype=dtype, compressor=None, separator="/", overwrite=True)


def create_v2_group_array(
    root_path: Path,
    name: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: Any,
) -> ZarrWriter:
    """Create a Zarr v2 group, replacing anything at ``root_path``, and one child array as in :func:`create_v2_array`."""
    group = create_group(root_path, overwrite=True)
    return group.create_array(name, shape=shape, chunks=chunks, dtype=dtype, compressor=None, separator="/")


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
