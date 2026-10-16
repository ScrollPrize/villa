"""Readers for the data the catalog points at.

Requires the ``reader`` extra (numpy) and the native extension
``vc_reader._native``, which is built with the package.

Readers can be used in forked processes such as DataLoader workers: they rebuild
their native threads there on first use. See :mod:`vc_reader.reader.zarr_reader`.
"""

from .disk_cache import CacheLayer, vc3d_cache_root, vc3d_volume_id
from .zarr_meta import ArrayMeta
from .zarr_reader import MemoryCache, ZarrReader, open_zarr, output_pool_stats, set_output_pool_bytes

__all__ = [
    "ArrayMeta",
    "CacheLayer",
    "MemoryCache",
    "ZarrReader",
    "open_zarr",
    "output_pool_stats",
    "set_output_pool_bytes",
    "vc3d_cache_root",
    "vc3d_volume_id",
]
