"""
Patch validation and caching for OME-Zarr datasets.

This module provides utilities for pre-computing and caching valid patch
positions for training datasets.
"""
from .cache import (
    PatchCacheData,
    PatchCacheParams,
    PatchEntry,
    SCHEMA_VERSION,
    build_cache_params,
    load_patch_cache,
    save_patch_cache,
    try_load_patch_cache,
)
from .generate import generate_patch_caches, PatchCacheResult

__all__ = [
    "PatchCacheData",
    "PatchCacheParams",
    "PatchCacheResult",
    "PatchEntry",
    "SCHEMA_VERSION",
    "build_cache_params",
    "generate_patch_caches",
    "load_patch_cache",
    "save_patch_cache",
    "try_load_patch_cache",
]
