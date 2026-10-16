"""Readers for the data the catalog points at.

Requires the ``reader`` extra (numpy, urllib3; numcodecs for compressed arrays).
"""

from .cache import MemoryChunkCache
from .codecs import CodecUnavailable
from .disk_cache import DiskCachedSource, vc3d_cache_root, vc3d_volume_id
from .sources import ChunkSource, HttpSource, LocalSource
from .zarr_meta import ArrayMeta
from .zarr_reader import ZarrReader, open_zarr

__all__ = [
    "ArrayMeta",
    "ChunkSource",
    "CodecUnavailable",
    "DiskCachedSource",
    "HttpSource",
    "LocalSource",
    "MemoryChunkCache",
    "ZarrReader",
    "open_zarr",
    "vc3d_cache_root",
    "vc3d_volume_id",
]
