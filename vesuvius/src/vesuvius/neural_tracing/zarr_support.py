from __future__ import annotations

from typing import Any


def zarr_chunk_key(array: Any, chunk_coords: tuple[int, ...]) -> str | None:
    """Return a store-relative chunk key for Zarr 2 or Zarr 3 arrays."""
    legacy_encoder = getattr(array, "_chunk_key", None)
    if callable(legacy_encoder):
        return str(legacy_encoder(chunk_coords))

    metadata = getattr(array, "metadata", None)
    metadata_encoder = getattr(metadata, "encode_chunk_key", None)
    if not callable(metadata_encoder):
        return None

    encoded = str(metadata_encoder(chunk_coords)).lstrip("/")
    array_path = str(getattr(array, "path", "")).strip("/")
    return f"{array_path}/{encoded}" if array_path else encoded


def read_zarr_store_bytes(store: Any, key: str) -> bytes:
    """Read one raw store value through the Zarr 2 or Zarr 3 store API."""
    get_sync = getattr(store, "get_sync", None)
    if callable(get_sync):
        value = get_sync(key)
    else:
        value = store[key]
    if value is None:
        raise KeyError(key)
    to_bytes = getattr(value, "to_bytes", None)
    return to_bytes() if callable(to_bytes) else bytes(value)
