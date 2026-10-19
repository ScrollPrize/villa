"""Python bindings for Volume Cartographer.

Load native modules on demand so object storage does not import volume bindings.
"""
from importlib import import_module

__all__ = ["Volume", "set_chunk_cache_budget", "set_chunk_cache_io_threads", "objects"]


def __getattr__(name):
    if name == "objects":
        value = import_module(f".{name}", __name__)
    elif name in ("Volume", "set_chunk_cache_budget", "set_chunk_cache_io_threads"):
        value = getattr(import_module(".volume", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
