"""Array and image readers, loaded only when their public names are used.

Image decoding needs the ``labels`` extra. Zarr access additionally needs the
native reader extension. Neither is imported merely to describe a source.
"""

from importlib import import_module

from ..errors import CacheMissError

_MODULES = {
    "ArrayMeta": "zarr_meta",
    "CacheLayer": "disk_cache",
    "DecodedCache": "decoded_cache",
    "MemoryCache": "memory_cache",
    "PrefetchClient": "prefetch",
    "PrefetchRequest": "prefetch",
    "PrefetchService": "prefetch",
    "PrefetchWindow": "prefetch",
    "S3Credentials": "auth",
    "SourceReader": "source_reader",
    "ZarrGroup": "zarr_group",
    "ZarrReader": "zarr_reader",
    "load_basic_auth": "auth",
    "open_group": "zarr_group",
    "open_zarr": "zarr_reader",
    "output_pool_stats": "zarr_reader",
    "set_output_pool_bytes": "zarr_reader",
    "vc3d_cache_root": "disk_cache",
    "vc3d_volume_id": "disk_cache",
}

__all__ = ["CacheMissError", *_MODULES]


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
