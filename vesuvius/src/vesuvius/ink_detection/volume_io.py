"""Local/remote Zarr opening, resolution selection, padding, and disk caching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import zarr

from vesuvius.data.utils import open_zarr as open_vesuvius_zarr


LOGGER = logging.getLogger(__name__)

_PUBLIC_S3_VOLUME_SUBSTRING = "vesuvius-challenge-open-data"
ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def _cache_snapshot(cache_dir: Path) -> list[tuple[int, int, Path]]:
    snapshot = []
    for directory, _, filenames in os.walk(cache_dir):
        for filename in filenames:
            if filename.endswith(".partial"):
                continue
            path = Path(directory) / filename
            try:
                stat = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not path.is_file():
                continue
            snapshot.append((stat.st_mtime_ns, stat.st_size, path))
    snapshot.sort()
    return snapshot


def _evict_to_watermark(
    snapshot: list[tuple[int, int, Path]], max_bytes: int
) -> int:
    total = sum(size for _, size, _ in snapshot)
    target_bytes = 0.9 * max_bytes
    for _, size, path in snapshot:
        if total <= target_bytes:
            break
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Windows refuses to unlink a file that another process still holds
            # open, so a sibling DataLoader worker reading this entry turns the
            # sweep into a PermissionError. The entry survives and therefore
            # still occupies its bytes: skip it without crediting the budget and
            # evict a different one instead.
            continue
        total -= size
    return total


def _best_effort_cache_store(cache_path: Path) -> Any:
    """Build the cache-side store, tolerating mutations the OS refuses.

    The disk cache is an optimization, never a source of truth, so a failed
    cache write or eviction must not abort the read that triggered it. That
    distinction only becomes visible on Windows: zarr commits each cache entry
    with ``os.replace`` and drops entries with ``unlink``, and Windows denies
    both while another process holds the target open. Several DataLoader
    workers sharing one ``cache_dir`` collide on exactly the same keys, so the
    failure is routine rather than exotic -- four concurrent readers of one
    volume fail roughly half the time. POSIX ``rename(2)``/``unlink(2)`` permit
    both operations, which is why the cache looks reliable on Linux and macOS.

    Losing the write costs one refetch; raising costs the whole run.
    """
    from zarr.storage import LocalStore

    class _BestEffortCacheStore(LocalStore):
        async def get(
            self,
            key: str,
            prototype: Any = None,
            byte_range: Any = None,
        ) -> Any:
            try:
                return await super().get(key, prototype, byte_range)
            except OSError as error:
                # LocalStore.get already reports an unreadable entry as a miss
                # for FileNotFoundError and friends; on Windows a sibling
                # process committing the same key denies the read instead, which
                # means exactly the same thing here.
                LOGGER.debug("treating unreadable cache entry %s as a miss: %s", key, error)
                return None

        async def _set(self, key: str, value: Any, exclusive: bool = False) -> None:
            try:
                await super()._set(key, value, exclusive=exclusive)
            except FileExistsError:
                raise
            except OSError as error:
                LOGGER.debug("skipped caching %s: %s", key, error)

        async def delete(self, key: str) -> None:
            try:
                await super().delete(key)
            except OSError as error:
                LOGGER.debug("skipped evicting %s: %s", key, error)

        async def delete_dir(self, prefix: str) -> None:
            try:
                await super().delete_dir(prefix)
            except OSError as error:
                LOGGER.debug("skipped evicting %s: %s", prefix, error)

    return _BestEffortCacheStore(cache_path)


def load_volume_auth(auth_json_path: str | Path | None) -> tuple[str, str] | None:
    """Read the exact username/password JSON boundary used for HTTPS volumes."""
    if auth_json_path is None:
        return None
    with Path(auth_json_path).open("r", encoding="utf-8") as stream:
        authored = json.load(stream)
    if not isinstance(authored, dict) or "username" not in authored or "password" not in authored:
        raise ValueError("volume auth JSON requires username and password")
    return str(authored["username"]), str(authored["password"])


def disk_cache_subdir(source_path: str, cache_dir: Path) -> Path:
    digest = hashlib.sha1(str(source_path).encode()).hexdigest()[:12]
    return Path(cache_dir) / digest


def _available_top_level_keys(root: Any) -> tuple[str, ...]:
    if not hasattr(root, "keys"):
        return ()
    return tuple(sorted(str(key) for key in root.keys()))


def _missing_node_error(message: str) -> Exception:
    error_type = getattr(zarr.errors, "NodeNotFoundError", None)
    if error_type is None:
        error_type = getattr(zarr.errors, "PathNotFoundError", KeyError)
    return error_type(message)


def open_volume_root(
    path: str | Path,
    auth_json_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    cache_max_gb: float | None = None,
):
    """Open a Zarr root with process-owned remote transport and optional cache."""
    path_text = str(path)
    storage_options: dict[str, Any] = {}
    is_public_s3 = (
        path_text.startswith("s3://")
        and _PUBLIC_S3_VOLUME_SUBSTRING in path_text
    )
    if is_public_s3:
        storage_options["anon"] = True
    auth = load_volume_auth(auth_json_path)
    if not is_public_s3 and path_text.startswith("https://") and auth is not None:
        storage_options["client_kwargs"] = {
            "auth": aiohttp.BasicAuth(auth[0], auth[1])
        }
    is_remote = path_text.startswith(("s3://", "http://", "https://"))

    if cache_dir is not None:
        if not ZARR_V3:
            raise NotImplementedError(
                "volume disk cache requires zarr 3; "
                f"installed zarr is {zarr.__version__}"
            )
        from zarr.experimental.cache_store import CacheStore
        from zarr.storage import LocalStore

        maximum_bytes = (
            None if cache_max_gb is None else int(float(cache_max_gb) * 1e9)
        )
        if maximum_bytes is not None and maximum_bytes < 0:
            raise ValueError("cache_max_gb must be nonnegative or None")
        cache_path = disk_cache_subdir(path_text, Path(cache_dir))
        cache_path.mkdir(parents=True, exist_ok=True)
        if maximum_bytes is not None:
            snapshot = _cache_snapshot(cache_path)
            if sum(size for _, size, _ in snapshot) > maximum_bytes:
                _evict_to_watermark(snapshot, maximum_bytes)
        if is_remote:
            remote_options = dict(storage_options)
            remote_options["skip_instance_cache"] = True
            source_store = zarr.storage.FsspecStore.from_url(
                path_text,
                storage_options=remote_options,
                read_only=True,
            )
        else:
            source_store = LocalStore(path_text, read_only=True)
        store = CacheStore(
            store=source_store,
            cache_store=_best_effort_cache_store(cache_path),
            max_size=maximum_bytes,
        )
        return zarr.open(store=store, mode="r")

    if is_remote and ZARR_V3:
        storage_options["skip_instance_cache"] = True
        store = zarr.storage.FsspecStore.from_url(
            path_text,
            storage_options=storage_options,
            read_only=True,
        )
        return zarr.open(store=store, mode="r")

    return open_vesuvius_zarr(
        path_text, mode="r", storage_options=storage_options
    )


def select_volume_level(
    root: Any,
    resolution: int | str,
    *,
    source: str,
    root_array_is_requested_level: bool = False,
) -> Any:
    """Select one resolution from an already opened array or group root."""

    if hasattr(root, "shape"):
        if not root_array_is_requested_level and str(resolution) not in {"0", ""}:
            raise _missing_node_error(
                f"{source.rstrip('/')}/{resolution} (resolution {str(resolution)!r} "
                f"in zarr array {source!r})"
            )
        return root
    try:
        return root[str(resolution)]
    except KeyError as exc:
        message = (
            f"{source.rstrip('/')}/{resolution} (resolution {str(resolution)!r} "
            f"in zarr store {source!r})"
        )
        try:
            available = _available_top_level_keys(root)
        except Exception:
            available = ()
        if available:
            message += "; available top-level keys: " + ", ".join(available[:20])
        raise _missing_node_error(message) from exc


def open_volume(
    path: str | Path,
    resolution: int | str,
    auth_json_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    cache_max_gb: float | None = None,
    root_array_is_requested_level: bool = False,
):
    """Open one Zarr pyramid level through the shared root boundary."""

    root = open_volume_root(
        path,
        auth_json_path,
        cache_dir=cache_dir,
        cache_max_gb=cache_max_gb,
    )
    return select_volume_level(
        root,
        resolution,
        source=str(path),
        root_array_is_requested_level=root_array_is_requested_level,
    )


def read_bbox_with_padding(
    volume: Any,
    bbox_zyx: tuple[int, int, int, int, int, int],
    *,
    fill_value: int | float = 0,
) -> tuple[np.ndarray, tuple[slice, slice, slice] | None]:
    """Read a positive ZYX bbox, padding only outside the array bounds."""
    z0, y0, x0, z1, y1, x1 = (int(value) for value in bbox_zyx)
    expected_shape = z1 - z0, y1 - y0, x1 - x0
    if any(size <= 0 for size in expected_shape):
        raise ValueError(f"bbox must define a positive crop, got {bbox_zyx!r}")
    shape = tuple(int(value) for value in volume.shape[:3])
    starts = max(0, z0), max(0, y0), max(0, x0)
    stops = min(shape[0], z1), min(shape[1], y1), min(shape[2], x1)
    output = np.full(expected_shape, fill_value, dtype=np.dtype(volume.dtype))
    if any(stop <= start for start, stop in zip(starts, stops)):
        return output, None
    crop = np.asarray(
        volume[
            starts[0] : stops[0],
            starts[1] : stops[1],
            starts[2] : stops[2],
        ]
    )
    destination_starts = starts[0] - z0, starts[1] - y0, starts[2] - x0
    destination = tuple(
        slice(start, start + size)
        for start, size in zip(destination_starts, crop.shape)
    )
    output[destination] = crop
    return output, destination
