"""Persistent on-disk chunk cache for remote zarr stores.

Read-through store wrappers that mirror a remote zarr store into a local
directory, one file per chunk key, namespaced by the remote URL. Cached bytes
survive process exit, so repeat epochs, DataLoader workers and separate runs
all share the same chunks.

Promoted here from ``neural_tracing/datasets/common.py`` so the rest of
``vesuvius.data`` can use it; that module re-exports these names, so its
existing imports keep working.
"""

import asyncio
import os
import threading
import time
import warnings
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import zarr
import zarr.storage


_ZARR_V3 = int(zarr.__version__.split('.', 1)[0]) >= 3

if _ZARR_V3:
    from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
else:
    # These names are only used by the Zarr 3 store implementation below.
    OffsetByteRequest = RangeByteRequest = SuffixByteRequest = ()


@dataclass
class ZarrCacheTraceStats:
    cache_hits: int = 0
    downloads: int = 0
    negative_hits: int = 0
    missing: int = 0
    cache_hit_bytes: int = 0
    download_bytes: int = 0
    cache_hit_ms: float = 0.0
    download_ms: float = 0.0
    missing_ms: float = 0.0


_CACHE_TRACE_LOCAL = threading.local()


def begin_zarr_cache_trace() -> None:
    _CACHE_TRACE_LOCAL.stats = ZarrCacheTraceStats()


def end_zarr_cache_trace() -> ZarrCacheTraceStats:
    stats = getattr(_CACHE_TRACE_LOCAL, "stats", None)
    _CACHE_TRACE_LOCAL.stats = None
    return stats if isinstance(stats, ZarrCacheTraceStats) else ZarrCacheTraceStats()


def _record_zarr_cache_event(kind: str, *, byte_count: int = 0, elapsed_ms: float = 0.0) -> None:
    stats = getattr(_CACHE_TRACE_LOCAL, "stats", None)
    if not isinstance(stats, ZarrCacheTraceStats):
        return
    if kind == "cache_hit":
        stats.cache_hits += 1
        stats.cache_hit_bytes += int(byte_count)
        stats.cache_hit_ms += float(elapsed_ms)
    elif kind == "download":
        stats.downloads += 1
        stats.download_bytes += int(byte_count)
        stats.download_ms += float(elapsed_ms)
    elif kind == "negative_hit":
        stats.negative_hits += 1
        stats.missing_ms += float(elapsed_ms)
    elif kind == "missing":
        stats.missing += 1
        stats.missing_ms += float(elapsed_ms)


class OfflineCacheMiss(Exception):
    """Raised when a zarr chunk fetch is attempted in offline mode but the
    chunk is not present in the local DiskCacheStore cache (neither as data
    nor as a negative marker).

    Used by testing/dev flows that want to train on whatever happens to be
    cached already, without issuing any network requests."""


# Exceptions that are NEVER retried — they won't resolve by waiting.
# OfflineCacheMiss must be here because it's our own marker and would
# otherwise be caught by the broad OSError check.
_NEVER_RETRY_EXCEPTIONS: tuple = (
    OfflineCacheMiss,
    PermissionError,        # OSError subclass; permanent denial
    IsADirectoryError,      # OSError subclass; structural problem
    NotADirectoryError,     # OSError subclass; structural problem
)

# Exceptions that ARE retried with backoff. OSError covers ConnectionError,
# TimeoutError, asyncio.TimeoutError (Python 3.11+), aiohttp.ClientConnectionError
# (which inherits OSError), most fsspec wrapped errors, etc. Anything in
# _NEVER_RETRY_EXCEPTIONS is excluded earlier in the except chain.
#
# Note: zarr's FsspecStore.get already converts FileNotFoundError/KeyError
# (the genuine "missing chunk" cases) into a None return via its
# allowed_exceptions filter, so they never reach this retry layer.
_RETRYABLE_EXCEPTIONS: tuple = (OSError,)

# Add botocore exceptions when available so S3 endpoint/credential failures
# get retried too. botocore.BotoCoreError is the base for connection/timeout
# errors; ClientError covers HTTP 4xx/5xx returned by the service. We retry
# both — permanent ClientErrors (NoSuchKey, AccessDenied) will burn the budget
# and then propagate.
try:
    import botocore.exceptions as _botocore_exceptions
    _RETRYABLE_EXCEPTIONS = _RETRYABLE_EXCEPTIONS + (
        _botocore_exceptions.BotoCoreError,
        _botocore_exceptions.ClientError,
    )
    del _botocore_exceptions
except ImportError:
    pass


# Suffix appended to the cached path to mark a "known-missing" chunk.
# Zarr chunk keys don't contain this pattern, so there's no collision
# with a real cached chunk filename.
_NEGATIVE_MARKER_SUFFIX = ".__notfound__"

# Store metadata is exempt from negative caching. Chunk absence is durable for
# an immutable published volume; metadata absence is not. Volume.load_ome_metadata
# probes candidate paths that 404 by design, and a volume still being uploaded
# gains its metadata later, so a marker written once would answer "missing" for
# the life of the cache directory. The cost of skipping it is a couple of
# no-byte 404 probes per open.
_METADATA_BASENAMES = frozenset({".zarray", ".zgroup", ".zattrs", ".zmetadata", "zarr.json"})


def _is_metadata_key(key: str) -> bool:
    return key.rsplit("/", 1)[-1] in _METADATA_BASENAMES


def _cache_subdir(cache_dir: str, url: str) -> str:
    """Return the subtree of `cache_dir` that holds `url`.

    Cache entries are namespaced by the normalized remote URL to prevent
    cross-dataset chunk-key collisions. Zarr chunk keys are relative paths
    inside one store (e.g. "c/0/1/2"), so without a per-URL prefix every
    dataset would write to the same paths under cache_dir.
    """
    normalized = url.rstrip('/')
    scheme, sep, rest = normalized.partition('://')
    # os.path.join discards everything before an absolute component, so an
    # absolute remainder (file:///x.zarr) would otherwise land outside the root.
    subdir = os.path.join(scheme, rest.lstrip('/')) if sep else normalized.lstrip('/')
    root = os.path.abspath(cache_dir)
    path = os.path.normpath(os.path.join(root, subdir))
    if os.path.commonpath([root, path]) != root:
        raise ValueError(f"cache path for {url!r} escapes cache_dir")
    return path


def _atomic_write_bytes(target: str, data: bytes) -> None:
    """Write `data` to `target` atomically.

    Uses a per-process/thread temp file in the same directory + os.replace,
    which is atomic on POSIX. Concurrent readers always see either the
    old content or the new — never a partially written file.
    """
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = f"{target}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# Bytes written per cache root since that root was last swept. Walking the
# whole tree on every chunk write would cost more than the download it saves,
# so a sweep only runs once the writes since the last one add up.
_EVICT_STATE_LOCK = threading.Lock()
_EVICT_STATE: dict[str, int] = {}
_EVICT_CHECK_EVERY = 64 * 2**20

# A temp file this old belongs to a writer that died before its os.replace;
# nothing will ever rename it, so the sweep reclaims it.
_ORPHAN_TEMP_AGE_SECONDS = 3600.0

# Written at the cache root the first time a store takes an empty or new
# directory, so eviction can tell a directory it owns from one the user keeps
# other files in.
_CACHE_STAMP_NAME = ".vesuvius-chunk-cache"
_UNSTAMPED_ROOTS_WARNED: set[str] = set()


def _ensure_cache_stamp(cache_root: str) -> None:
    """Claim `cache_root` for this cache, if it is free to claim.

    Best effort: a root that already holds unrelated files stays unstamped, and
    _maybe_evict then declines to sweep it.
    """
    stamp = os.path.join(cache_root, _CACHE_STAMP_NAME)
    try:
        if os.path.exists(stamp):
            return
        if os.path.isdir(cache_root) and os.listdir(cache_root):
            return
        os.makedirs(cache_root, exist_ok=True)
        with open(stamp, 'ab'):
            pass
    except OSError:
        pass


def _warn_unstamped_root(cache_root: str) -> None:
    """Warn once per root that its size cap cannot be enforced."""
    with _EVICT_STATE_LOCK:
        if cache_root in _UNSTAMPED_ROOTS_WARNED:
            return
        _UNSTAMPED_ROOTS_WARNED.add(cache_root)
    warnings.warn(
        f"chunk cache root {cache_root!r} holds files this cache did not write "
        f"(no {_CACHE_STAMP_NAME} stamp), so max_bytes is not enforced there; "
        f"point cache_dir at a directory of its own to cap it",
        stacklevel=3,
    )


def _check_max_bytes(max_bytes: int | None) -> None:
    """Reject a non-positive cap.

    ``None`` means unbounded, so a 0 or negative cap would otherwise be
    indistinguishable from it in every ``if self._max_bytes`` check below and
    silently give the caller an unbounded cache. Callers that want no caching
    should not build the store.
    """
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be a positive byte count or None")


def _touch(path: str) -> None:
    """Record a cache hit as recent use, for eviction ordering.

    Best effort: another process may have evicted the file between the read and
    this call, which costs nothing but that entry's place in the order.
    """
    try:
        os.utime(path)
    except OSError:
        pass


def _maybe_evict(cache_root: str, max_bytes: int, written: int) -> None:
    """Trim the cache tree below `max_bytes`, oldest mtime first.

    Recency is the file mtime, refreshed on every cache hit, so this is an LRU.
    The byte counter is per-process: with several processes sharing one cache
    root each sweeps on its own writes, which keeps the cap approximate but
    still bounded. Files another process removes or is midway through writing
    are skipped rather than treated as errors.

    Only a stamped root is swept, so a cache_dir that also holds the user's own
    files never loses them.
    """
    with _EVICT_STATE_LOCK:
        pending = _EVICT_STATE.get(cache_root, 0) + written
        if pending < _EVICT_CHECK_EVERY and pending < max_bytes:
            _EVICT_STATE[cache_root] = pending
            return
        _EVICT_STATE[cache_root] = 0

    if not os.path.exists(os.path.join(cache_root, _CACHE_STAMP_NAME)):
        _warn_unstamped_root(cache_root)
        return

    entries = []
    total = 0
    now = time.time()
    for dirpath, _dirnames, filenames in os.walk(cache_root):
        for name in filenames:
            # Partial writes belong to whoever is writing them; they become
            # cache entries only once os.replace renames them. One left behind
            # by a killed worker is nobody's, so age it out.
            if ".tmp." in name:
                path = os.path.join(dirpath, name)
                try:
                    if now - os.stat(path).st_mtime > _ORPHAN_TEMP_AGE_SECONDS:
                        os.remove(path)
                except OSError:
                    pass
                continue
            if name == _CACHE_STAMP_NAME and dirpath == cache_root:
                continue
            path = os.path.join(dirpath, name)
            try:
                info = os.stat(path)
            except OSError:
                continue
            entries.append((info.st_mtime, info.st_size, path))
            total += info.st_size

    if total <= max_bytes:
        return

    entries.sort()
    for _mtime, size, path in entries:
        try:
            os.remove(path)
        except OSError:
            continue
        total -= size
        if total <= max_bytes:
            break


class DiskCacheStoreV3(getattr(zarr.storage, 'WrapperStore', object)):
    """Read-only Zarr v3 store wrapper that lazily caches remote bytes to disk."""

    def __init__(
        self,
        remote: Any,
        cache_dir: str,
        url: str,
        offline: bool = False,
        retry_budget_seconds: float = 0.0,
        max_bytes: int | None = None,
    ) -> None:
        super().__init__(remote)
        _check_max_bytes(max_bytes)
        self._remote = remote
        # The cap applies to the whole user-supplied cache root, not just this
        # store's URL-namespaced subdirectory.
        self._cache_root = cache_dir
        self._max_bytes = max_bytes
        self._offline = offline
        self._retry_budget_seconds = float(retry_budget_seconds)
        self._url = str(url)
        self._cache_dir = _cache_subdir(cache_dir, url)
        _ensure_cache_stamp(cache_dir)

    # Also exposed on the instance because callers read it off the store
    # (neural_tracing/fiber_trace/train.py).
    _NEGATIVE_MARKER_SUFFIX = _NEGATIVE_MARKER_SUFFIX

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    def with_read_only(self, read_only: bool = False):
        if not read_only:
            raise NotImplementedError("DiskCacheStore is always read-only")
        return type(self)(
            self._remote.with_read_only(True),
            self._cache_root,
            self._url,
            offline=self._offline,
            retry_budget_seconds=self._retry_budget_seconds,
            max_bytes=self._max_bytes,
        )

    async def _remote_get_with_retry(self, key, prototype, byte_range=None):
        """Read one key from the wrapped remote store with backoff retries."""
        if self._retry_budget_seconds <= 0.0:
            return await self._remote.get(key, prototype, byte_range)

        deadline = time.monotonic() + self._retry_budget_seconds
        delay = 1.0
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._remote.get(key, prototype, byte_range)
            except _NEVER_RETRY_EXCEPTIONS:
                raise
            except _RETRYABLE_EXCEPTIONS as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(
                        f"[DiskCacheStore] giving up on {key!r} after "
                        f"{attempt} attempts, "
                        f"{self._retry_budget_seconds:.0f}s budget exhausted: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    raise
                wait = min(delay, remaining, 60.0)
                print(
                    f"[DiskCacheStore] transient error fetching {key!r} "
                    f"(attempt {attempt}): {type(exc).__name__}: {exc}; "
                    f"retrying in {wait:.1f}s "
                    f"(remaining budget {remaining:.0f}s)",
                    flush=True,
                )
                await asyncio.sleep(wait)
                delay = min(delay * 2.0, 60.0)

    @staticmethod
    def _buffer_from_cached_bytes(data, prototype, byte_range=None):
        if isinstance(byte_range, RangeByteRequest):
            data = data[byte_range.start:byte_range.end]
        elif isinstance(byte_range, OffsetByteRequest):
            data = data[byte_range.offset:]
        elif isinstance(byte_range, SuffixByteRequest):
            # Indexed from the front: data[-0:] would be the whole object.
            data = data[max(len(data) - byte_range.suffix, 0):]
        elif byte_range is not None:
            raise ValueError(f"unexpected byte range request: {byte_range!r}")
        return prototype.buffer.from_bytes(data)

    async def get(self, key, prototype, byte_range=None):
        start = time.perf_counter()
        cached = os.path.join(self._cache_dir, key)
        marker = cached + self._NEGATIVE_MARKER_SUFFIX

        # Positive cache hit.
        if os.path.isfile(cached):
            try:
                with open(cached, 'rb') as f:
                    data = f.read()
                if self._max_bytes:
                    _touch(cached)
                _record_zarr_cache_event(
                    "cache_hit",
                    byte_count=len(data),
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                )
                return self._buffer_from_cached_bytes(data, prototype, byte_range)
            except FileNotFoundError:
                # Raced with a concurrent replace; fall through to re-fetch.
                pass
        # Negative cache hit → known-missing, skip the remote round-trip.
        # Metadata markers are ignored online so a re-probe can find metadata
        # that exists now, but offline they are the only answer available: a
        # cache tree written before that exemption must not start raising.
        if os.path.isfile(marker) and (self._offline or not _is_metadata_key(key)):
            _record_zarr_cache_event(
                "negative_hit",
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            )
            return None

        if self._offline:
            raise OfflineCacheMiss(
                f"offline mode: chunk {key!r} not in local cache "
                f"({self._cache_dir})"
            )

        # A partial response cannot populate the full-object disk cache. Range
        # reads are uncommon for Zarr chunks, so delegate cache misses to the
        # remote store and preserve its exact ByteRequest semantics.
        result = await self._remote_get_with_retry(key, prototype, byte_range)
        if result is None:
            if not _is_metadata_key(key):
                try:
                    _atomic_write_bytes(marker, b"")
                except OSError:
                    pass
            _record_zarr_cache_event(
                "missing",
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            )
            return None

        if byte_range is None:
            data = result.to_bytes()
            byte_count = len(data)
            # The bytes are already in hand; a full cache dir or a read-only one
            # costs the next reader a refetch, not this reader its result.
            try:
                _atomic_write_bytes(cached, data)
            except OSError:
                pass
            else:
                if self._max_bytes:
                    _maybe_evict(self._cache_root, self._max_bytes, byte_count)
        else:
            byte_count = len(result.to_bytes())
        _record_zarr_cache_event(
            "download",
            byte_count=byte_count,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )
        return result

    async def exists(self, key):
        cached = os.path.join(self._cache_dir, key)
        if os.path.isfile(cached):
            # Answered from disk instead of the remote, so it counts as use.
            if self._max_bytes:
                _touch(cached)
            return True
        if not _is_metadata_key(key) and os.path.isfile(cached + self._NEGATIVE_MARKER_SUFFIX):
            return False
        if self._offline:
            return False
        return await self._remote.exists(key)

    async def get_partial_values(self, prototype, key_ranges):
        return await asyncio.gather(*(
            self.get(key, prototype, byte_range)
            for key, byte_range in key_ranges
        ))

    async def set(self, key, value):
        raise PermissionError("read-only cache store")

    async def delete(self, key):
        raise PermissionError("read-only cache store")

    def close(self) -> None:
        close_fn = getattr(self._remote, "close", None)
        if callable(close_fn):
            close_fn()

    def __eq__(self, other):
        return (
            isinstance(other, DiskCacheStoreV3)
            and self._remote == other._remote
            and self._cache_dir == other._cache_dir
        )


# Zarr 2 wraps a bare MutableMapping in KVStore, which would hide the listdir
# and getsize below and send zarr back to its key-iterating fallbacks. Its Store
# ABC is passed through untouched, so subclass that where it exists.
_V2_STORE_BASE = MutableMapping if _ZARR_V3 else getattr(zarr.storage, 'Store', MutableMapping)


class DiskCacheStoreV2(_V2_STORE_BASE):
    """Read-only Zarr v2 mapping that lazily caches remote bytes to disk."""

    _NEGATIVE_MARKER_SUFFIX = _NEGATIVE_MARKER_SUFFIX

    def __init__(
        self,
        remote: MutableMapping,
        cache_dir: str,
        url: str,
        offline: bool = False,
        retry_budget_seconds: float = 0.0,
        max_bytes: int | None = None,
    ) -> None:
        _check_max_bytes(max_bytes)
        self._remote = remote
        # The cap applies to the whole user-supplied cache root, not just this
        # store's URL-namespaced subdirectory.
        self._cache_root = cache_dir
        self._max_bytes = max_bytes
        self._offline = offline
        self._retry_budget_seconds = float(retry_budget_seconds)
        self._url = str(url)
        self._cache_dir = _cache_subdir(cache_dir, url)
        _ensure_cache_stamp(cache_dir)

    def with_read_only(self, read_only: bool = False):
        if not read_only:
            raise NotImplementedError("DiskCacheStore is always read-only")
        return type(self)(
            self._remote,
            self._cache_root,
            self._url,
            offline=self._offline,
            retry_budget_seconds=self._retry_budget_seconds,
            max_bytes=self._max_bytes,
        )

    def _remote_get_with_retry(self, key):
        if self._retry_budget_seconds <= 0.0:
            return self._remote[key]

        deadline = time.monotonic() + self._retry_budget_seconds
        delay = 1.0
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._remote[key]
            except _NEVER_RETRY_EXCEPTIONS:
                raise
            except _RETRYABLE_EXCEPTIONS as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(
                        f"[DiskCacheStore] giving up on {key!r} after "
                        f"{attempt} attempts, "
                        f"{self._retry_budget_seconds:.0f}s budget exhausted: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    raise
                wait = min(delay, remaining, 60.0)
                print(
                    f"[DiskCacheStore] transient error fetching {key!r} "
                    f"(attempt {attempt}): {type(exc).__name__}: {exc}; "
                    f"retrying in {wait:.1f}s "
                    f"(remaining budget {remaining:.0f}s)",
                    flush=True,
                )
                time.sleep(wait)
                delay = min(delay * 2.0, 60.0)

    def __getitem__(self, key):
        start = time.perf_counter()
        cached = os.path.join(self._cache_dir, key)
        marker = cached + self._NEGATIVE_MARKER_SUFFIX

        # Positive cache hit.
        if os.path.isfile(cached):
            try:
                with open(cached, 'rb') as f:
                    data = f.read()
                if self._max_bytes:
                    _touch(cached)
                _record_zarr_cache_event(
                    "cache_hit",
                    byte_count=len(data),
                    elapsed_ms=(time.perf_counter() - start) * 1000.0,
                )
                return data
            except FileNotFoundError:
                # Raced with a concurrent replace; fall through to re-fetch.
                pass
        # Negative cache hit → known-missing, skip the remote round-trip.
        # Metadata markers are ignored online so a re-probe can find metadata
        # that exists now, but offline they are the only answer available: a
        # cache tree written before that exemption must not start raising
        # OfflineCacheMiss where it used to raise KeyError.
        if os.path.isfile(marker) and (self._offline or not _is_metadata_key(key)):
            _record_zarr_cache_event(
                "negative_hit",
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            )
            raise KeyError(key)
        if self._offline:
            raise OfflineCacheMiss(
                f"offline mode: chunk {key!r} not in local cache "
                f"({self._cache_dir})"
            )

        try:
            result = self._remote_get_with_retry(key)
        except KeyError:
            if not _is_metadata_key(key):
                try:
                    _atomic_write_bytes(marker, b"")
                except OSError:
                    pass
            _record_zarr_cache_event(
                "missing",
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            )
            raise

        result = bytes(result)
        # The bytes are already in hand; a full cache dir or a read-only one
        # costs the next reader a refetch, not this reader its result.
        try:
            _atomic_write_bytes(cached, result)
        except OSError:
            pass
        else:
            if self._max_bytes:
                _maybe_evict(self._cache_root, self._max_bytes, len(result))
        _record_zarr_cache_event(
            "download",
            byte_count=len(result),
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )
        return result

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        cached = os.path.join(self._cache_dir, key)
        if os.path.isfile(cached):
            # Answered from disk instead of the remote, so it counts as use.
            if self._max_bytes:
                _touch(cached)
            return True
        if not _is_metadata_key(key) and os.path.isfile(cached + self._NEGATIVE_MARKER_SUFFIX):
            return False
        if self._offline:
            return False
        return key in self._remote

    def __iter__(self):
        return iter(self._remote)

    def __len__(self):
        return len(self._remote)

    def listdir(self, path: str = ""):
        # Delegated, never cached: zarr's fallback for a store without listdir
        # iterates every key, which on an FSStore is a full recursive listing of
        # the remote. Opening a group listing its children is the common path.
        return zarr.storage.listdir(self._remote, path)

    def getsize(self, path=None):
        return zarr.storage.getsize(self._remote, path)

    def __setitem__(self, key, value):
        raise PermissionError("read-only cache store")

    def __delitem__(self, key):
        raise PermissionError("read-only cache store")

    def close(self) -> None:
        close_fn = getattr(self._remote, "close", None)
        if callable(close_fn):
            close_fn()

    def __eq__(self, other):
        return (
            isinstance(other, DiskCacheStoreV2)
            and self._remote == other._remote
            and self._cache_dir == other._cache_dir
        )


DiskCacheStore = DiskCacheStoreV3 if _ZARR_V3 else DiskCacheStoreV2
