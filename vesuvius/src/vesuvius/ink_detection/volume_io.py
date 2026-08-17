"""Local/remote Zarr opening, resolution selection, padding, and disk caching."""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import zarr

from vesuvius.data.utils import open_zarr as open_vesuvius_zarr


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
) -> list[tuple[int, int, Path]]:
    """Delete oldest entries until the snapshot fits under the watermark.

    Returns the entries that survived, still sorted oldest-mtime-first, so the
    caller can hand exactly the retained set to ``_seed_cache_store_lru``.
    """
    total = sum(size for _, size, _ in snapshot)
    target_bytes = 0.9 * max_bytes
    evicted = 0
    for _, size, path in snapshot:
        if total <= target_bytes:
            break
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        total -= size
        evicted += 1
    return snapshot[evicted:]


def _seed_cache_store_lru(
    store: Any, cache_path: Path, snapshot: list[tuple[int, int, Path]]
) -> bool:
    """Register already-cached files in a fresh CacheStore's LRU accounting.

    ``CacheStore`` builds its LRU bookkeeping in memory and starts it empty in
    every new process: it never scans ``cache_store`` on construction, so files
    left by an earlier run are invisible to it. It therefore writes a full
    ``max_size`` of new data on top of whatever the open-time sweep retained,
    and a hit on a pre-existing file neither counts toward the budget nor
    refreshes that file's LRU position. Seeding the accounting from the files
    that are actually on disk makes ``max_size`` a true bound on the cache
    directory rather than a per-process allowance, and lets reuse of an
    inherited entry keep it alive.

    The alternative of shrinking ``max_size`` to the remaining headroom is
    worse: the sweep leaves 90% of the budget in place, so headroom collapses
    to 10% (or to 0 when the directory is already at budget), and a
    ``max_size`` below one chunk makes ``CacheStore`` write every value to disk
    and then decline to track it -- an unbounded, never-evicted cache. Evicting
    far enough to free a whole budget would instead throw away the cross-run
    reuse the disk cache exists to provide.

    This reaches into ``CacheStore._state`` because zarr exposes no public seam
    for pre-populating the LRU (``cache_info()`` is read-only). The proper
    home for this is ``CacheStore.__init__``; until it lands upstream, the
    access is guarded and reported rather than assumed.

    Returns True when the accounting was seeded.
    """
    state = getattr(store, "_state", None)
    required = ("cache_order", "key_sizes", "key_insert_times", "current_size")
    if state is None or not all(hasattr(state, name) for name in required):
        return False
    monotonic_now = time.monotonic()
    wall_now = time.time()
    total = 0
    for mtime_ns, size, path in snapshot:
        try:
            key = path.relative_to(cache_path).as_posix()
        except ValueError:
            continue
        # snapshot is oldest-first, so insertion order is already LRU order.
        state.cache_order[key] = None
        state.key_sizes[key] = size
        # Age entries by mtime so a finite max_age_seconds would still expire
        # them; with the "infinity" default this value is never read.
        state.key_insert_times[key] = monotonic_now - max(
            0.0, wall_now - mtime_ns / 1e9
        )
        total += size
    state.current_size = total
    return True


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
        retained: list[tuple[int, int, Path]] = []
        if maximum_bytes is not None:
            snapshot = _cache_snapshot(cache_path)
            if sum(size for _, size, _ in snapshot) > maximum_bytes:
                retained = _evict_to_watermark(snapshot, maximum_bytes)
            else:
                retained = snapshot
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
            cache_store=LocalStore(cache_path),
            max_size=maximum_bytes,
        )
        # Seed before the first read: zarr.open() itself fetches metadata
        # through this store, and those writes must be charged against the
        # bytes already on disk rather than against an empty budget.
        if maximum_bytes is not None and not _seed_cache_store_lru(
            store, cache_path, retained
        ):
            warnings.warn(
                "installed zarr CacheStore exposes no LRU state to seed "
                f"(zarr {zarr.__version__}); the volume cache budget of "
                f"{maximum_bytes} bytes is enforced only by the open-time "
                "sweep, so this process may grow the cache directory past it",
                RuntimeWarning,
                stacklevel=2,
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
