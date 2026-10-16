"""Readers for the data the catalog points at.

Requires the ``reader`` extra (numpy) and the native extension
``vc_reader._native``, which is built with the package.
"""

from .disk_cache import CacheLayer, vc3d_cache_root, vc3d_volume_id
from .zarr_meta import ArrayMeta
from .zarr_reader import MemoryCache, ZarrReader, open_zarr

__all__ = [
    "ArrayMeta",
    "CacheLayer",
    "MemoryCache",
    "ZarrReader",
    "open_zarr",
    "vc3d_cache_root",
    "vc3d_volume_id",
]
