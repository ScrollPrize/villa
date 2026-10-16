"""Readers for the data the catalog points at.

Requires the ``reader`` extra (numpy) and the native extension
``vc_reader._native``, which is built with the package.

Readers can be used in forked processes such as DataLoader workers: they rebuild
their native threads there on first use. See :mod:`vc_reader.reader.zarr_reader`.
"""

from ..errors import CacheMissError
from .auth import S3Credentials, load_basic_auth
from .disk_cache import CacheLayer, vc3d_cache_root, vc3d_volume_id
from .zarr_group import ZarrGroup, open_group
from .zarr_meta import ArrayMeta
from .zarr_reader import MemoryCache, ZarrReader, open_zarr, output_pool_stats, set_output_pool_bytes
from .prefetch import PrefetchClient, PrefetchRequest, PrefetchService, PrefetchWindow

__all__ = [
    "ArrayMeta",
    "CacheLayer",
    "CacheMissError",
    "MemoryCache",
    "PrefetchClient",
    "PrefetchRequest",
    "PrefetchService",
    "PrefetchWindow",
    "S3Credentials",
    "ZarrGroup",
    "ZarrReader",
    "load_basic_auth",
    "open_group",
    "open_zarr",
    "output_pool_stats",
    "set_output_pool_bytes",
    "vc3d_cache_root",
    "vc3d_volume_id",
]
