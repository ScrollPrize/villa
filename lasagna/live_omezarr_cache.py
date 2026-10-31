from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

from vc_reader.reader import ZarrReader

# flock() is POSIX. Windows has no fcntl, so take the same non-blocking
# shared/exclusive advisory lock through LockFileEx, which is the Win32 call
# with matching semantics (msvcrt.locking can only do exclusive locks, and the
# cache relies on several readers holding a shared lock at once).
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    import msvcrt

    _LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
    _LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
    _ERROR_LOCK_VIOLATION = 33
    _WHOLE_FILE_LOW = 0xFFFFFFFF
    _WHOLE_FILE_HIGH = 0xFFFFFFFF

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.LockFileEx.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_Overlapped),
    ]
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = [
        wintypes.HANDLE, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_Overlapped),
    ]
    _kernel32.UnlockFileEx.restype = wintypes.BOOL

    def _lock_file(handle, *, exclusive: bool) -> None:
        flags = _LOCKFILE_FAIL_IMMEDIATELY
        if exclusive:
            flags |= _LOCKFILE_EXCLUSIVE_LOCK
        overlapped = _Overlapped()
        win_handle = msvcrt.get_osfhandle(handle.fileno())
        if not _kernel32.LockFileEx(
            win_handle, flags, 0, _WHOLE_FILE_LOW, _WHOLE_FILE_HIGH,
            ctypes.byref(overlapped),
        ):
            code = ctypes.get_last_error()
            if code == _ERROR_LOCK_VIOLATION:
                raise BlockingIOError(code, "file is locked by another process")
            raise ctypes.WinError(code)

    def _unlock_file(handle) -> None:
        overlapped = _Overlapped()
        win_handle = msvcrt.get_osfhandle(handle.fileno())
        _kernel32.UnlockFileEx(
            win_handle, 0, _WHOLE_FILE_LOW, _WHOLE_FILE_HIGH,
            ctypes.byref(overlapped),
        )
else:
    import fcntl

    def _lock_file(handle, *, exclusive: bool) -> None:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)

    def _unlock_file(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


DEFAULT_LIVE_CACHE_GIB = 10 * 1024
DEFAULT_LIVE_FETCH_AHEAD_TILES = 10_000


def _downloader():
    try:
        from lasagna.scripts import download_omezarr
    except ImportError:  # pragma: no cover - legacy top-level package mode.
        from scripts import download_omezarr
    return download_omezarr


@dataclass(frozen=True)
class SelectedLevelSource:
    group_root: Path
    level_path: Path
    level: int
    source_uri: str
    bucket: str
    prefix: str
    anon: bool
    region: str | None
    shape: tuple[int, int, int]
    chunks: tuple[int, int, int]
    dimension_separator: str
    zarray: dict[str, Any]


class SelectedLevelLock:
    """Advisory shared/exclusive lock for one local OME-Zarr level."""

    def __init__(self, group_root: Path, level: int, *, exclusive: bool) -> None:
        self.group_root = Path(group_root)
        self.level = int(level)
        self.exclusive = bool(exclusive)
        self.path = self.group_root / ".dl_cache" / f"{self.level}.level.lock"
        self._handle = None

    def __enter__(self) -> "SelectedLevelLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        try:
            _lock_file(self._handle, exclusive=self.exclusive)
        except BlockingIOError as error:
            self._handle.close()
            self._handle = None
            kind = "mutator" if self.exclusive else "reader"
            raise RuntimeError(
                f"OME-Zarr level {self.level} is locked by another cache reader/mutator: "
                f"{self.path} (requested {kind} lock)"
            ) from error
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self._handle is None:
            return
        try:
            _unlock_file(self._handle)
        finally:
            self._handle.close()
            self._handle = None


def selected_level_lock_for_input(
    input_path: str | os.PathLike[str], *, exclusive: bool,
) -> SelectedLevelLock | None:
    path = Path(str(input_path).rstrip("/"))
    if not path.name.isdigit() or path.is_symlink():
        return None
    group_root = path.parent
    if not (group_root / ".zattrs").is_file():
        return None
    return SelectedLevelLock(group_root, int(path.name), exclusive=exclusive)


def path_has_download_source(input_path: str | os.PathLike[str]) -> bool:
    """Return whether a nearby group .zattrs advertises a remote source."""
    check = Path(str(input_path).rstrip("/")).expanduser().resolve()
    for _ in range(6):
        zattrs_path = check / ".zattrs"
        if zattrs_path.is_file():
            try:
                attrs = _read_json(zattrs_path)
            except (OSError, ValueError, json.JSONDecodeError):
                return False
            download = attrs.get("_download")
            return isinstance(download, dict) and isinstance(download.get("source"), str)
        if check.parent == check:
            break
        check = check.parent
    return False


_ZARRAY_COMPATIBILITY_FIELDS = (
    "shape",
    "chunks",
    "dtype",
    "order",
    "compressor",
    "filters",
    "fill_value",
    "dimension_separator",
)


def _metadata_value(zarray: dict[str, Any], field: str) -> Any:
    if field == "dimension_separator":
        return zarray.get(field, ".")
    return zarray.get(field)


def _validate_zarray_compatible(
    local: dict[str, Any], remote: dict[str, Any], *, local_path: Path,
) -> None:
    mismatches = [
        field
        for field in _ZARRAY_COMPATIBILITY_FIELDS
        if _metadata_value(local, field) != _metadata_value(remote, field)
    ]
    if mismatches:
        detail = ", ".join(
            f"{field}: local={_metadata_value(local, field)!r} "
            f"remote={_metadata_value(remote, field)!r}"
            for field in mismatches
        )
        raise ValueError(
            f"local selected-level metadata does not match its remote source at "
            f"{local_path}: {detail}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
    except FileExistsError:
        pass


def prepare_selected_level_source(
    input_path: str | os.PathLike[str],
) -> SelectedLevelSource:
    """Prepare/validate metadata for one local numeric OME-Zarr-v2 level."""
    dl = _downloader()
    raw_path = Path(str(input_path).rstrip("/")).expanduser()
    if not raw_path.name.isdigit():
        raise ValueError(
            f"live fetch requires an input path ending in a numeric OME-Zarr level: {input_path}"
        )
    if raw_path.is_symlink():
        raise ValueError(f"live fetch refuses a symlink selected-level path: {raw_path}")
    group_root = raw_path.parent.resolve()
    level_path = group_root / raw_path.name
    level = int(raw_path.name)
    zattrs_path = group_root / ".zattrs"
    if not zattrs_path.is_file():
        raise ValueError(
            f"live fetch requires local group metadata with _download source: {zattrs_path}"
        )
    root_attrs = _read_json(zattrs_path)
    download_meta = root_attrs.get("_download")
    if not isinstance(download_meta, dict) or not isinstance(download_meta.get("source"), str):
        raise ValueError(f"missing valid _download source metadata in {zattrs_path}")
    source_uri = str(download_meta["source"]).rstrip("/")
    bucket, prefix = dl._parse_s3_uri(source_uri)
    anon = bool(download_meta.get("anon", False))
    region_value = download_meta.get("region")
    region = str(region_value) if region_value else None

    remote_zarray = dl._s3_read_json(
        bucket, f"{prefix}/{level}/.zarray", anon, region=region,
    )
    if not isinstance(remote_zarray, dict):
        raise ValueError(f"remote selected-level .zarray is not an object: {source_uri}/{level}")
    local_zarray_path = level_path / ".zarray"
    if local_zarray_path.is_file():
        local_zarray = _read_json(local_zarray_path)
        _validate_zarray_compatible(local_zarray, remote_zarray, local_path=local_zarray_path)
    else:
        _write_json_exclusive(local_zarray_path, remote_zarray)
        local_zarray = _read_json(local_zarray_path)
        _validate_zarray_compatible(local_zarray, remote_zarray, local_path=local_zarray_path)

    for metadata_name in (".zgroup",):
        local_metadata = group_root / metadata_name
        if local_metadata.exists():
            continue
        try:
            remote_metadata = dl._s3_read_json(
                bucket, f"{prefix}/{metadata_name}", anon, region=region,
            )
        except Exception:
            continue
        if isinstance(remote_metadata, dict):
            _write_json_exclusive(local_metadata, remote_metadata)
    local_level_attrs = level_path / ".zattrs"
    if not local_level_attrs.exists():
        try:
            remote_level_attrs = dl._s3_read_json(
                bucket, f"{prefix}/{level}/.zattrs", anon, region=region,
            )
        except Exception:
            remote_level_attrs = None
        if isinstance(remote_level_attrs, dict):
            _write_json_exclusive(local_level_attrs, remote_level_attrs)

    shape = tuple(int(value) for value in local_zarray["shape"])
    chunks = tuple(int(value) for value in local_zarray["chunks"])
    if len(shape) != 3 or len(chunks) != 3 or any(value <= 0 for value in (*shape, *chunks)):
        raise ValueError(
            f"live fetch requires positive 3-D shape/chunks, got shape={shape} chunks={chunks}"
        )
    separator = str(local_zarray.get("dimension_separator", "."))
    if separator not in {".", "/"}:
        raise ValueError(f"unsupported Zarr-v2 dimension_separator {separator!r}")
    return SelectedLevelSource(
        group_root=group_root,
        level_path=level_path,
        level=level,
        source_uri=source_uri,
        bucket=bucket,
        prefix=prefix,
        anon=anon,
        region=region,
        shape=shape,
        chunks=chunks,
        dimension_separator=separator,
        zarray=local_zarray,
    )


class LiveOmeZarrCache:
    """One selected level read from its remote source through a bounded disk cache, with read-ahead.

    The reader's cache drops its least recently used chunks past ``max_bytes``. Chunks a plain
    download left in the level are not used.
    """

    def __init__(
        self,
        input_path: str | os.PathLike[str],
        *,
        max_bytes: int = DEFAULT_LIVE_CACHE_GIB * (1 << 30),
        lookahead_tiles: int = DEFAULT_LIVE_FETCH_AHEAD_TILES,
        workers: int = 64,
    ) -> None:
        if int(max_bytes) <= 0:
            raise ValueError("live cache max_bytes must be > 0")
        if int(lookahead_tiles) <= 0:
            raise ValueError("live fetch lookahead_tiles must be > 0")
        if int(workers) <= 0:
            raise ValueError("live fetch workers must be > 0")
        self.source = prepare_selected_level_source(input_path)
        self.max_bytes = int(max_bytes)
        self.lookahead_tiles = int(lookahead_tiles)
        self.workers = int(workers)
        self._reader: ZarrReader | None = None
        self._final_reader_stats: dict[str, Any] | None = None
        self._inventory_futures: dict[int, Future[set[tuple[int, int]]]] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._inventory_pool = ThreadPoolExecutor(
            max_workers=min(16, self.workers), thread_name_prefix="live-zarr-list",
        )
        # 64 concurrent tile reads keep up with the old 256-worker pipeline on Open Data (planning, C).
        self._read_pool = ThreadPoolExecutor(
            max_workers=min(64, self.workers), thread_name_prefix="live-zarr-read",
        )
        self._stats = {
            "requested_regions": 0, "materialized_regions": 0,
            "listed_planes": 0, "listed_chunks": 0, "missing_chunks": 0,
        }
        self._started_at = time.monotonic()
        print(
            f"[live-fetch] level={self.source.level} shape={self.source.shape} "
            f"chunks={self.source.chunks} target={self.max_bytes / (1 << 40):.2f}TiB "
            f"lookahead_tiles={self.lookahead_tiles} workers={self.workers}",
            flush=True,
        )

    def __enter__(self) -> "LiveOmeZarrCache":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def open_reader(self, *, memory_cache: int | None = None, threads: int | None = None) -> ZarrReader:
        """The reader inference reads tiles from; read-ahead goes through the same one."""
        source = self.source
        self._reader = ZarrReader.open(
            _downloader()._s3_location(f"{source.source_uri}/{source.level}", source.anon, source.region),
            cache_dir=str(source.group_root / ".live_cache"),
            cache_max_bytes=self.max_bytes,
            memory_cache=memory_cache,
            threads=threads,
            connections=self.workers,
            auth=None if source.anon else "aws",
        )
        return self._reader

    def _list_plane(self, iz: int) -> set[tuple[int, int]]:
        dl = _downloader()
        level_prefix = f"{self.source.prefix}/{self.source.level}"
        list_prefix = dl._inventory_prefix(
            level_prefix, self.source.dimension_separator, int(iz),
        )
        result: set[tuple[int, int]] = set()
        y_chunks = math.ceil(self.source.shape[1] / self.source.chunks[1])
        x_chunks = math.ceil(self.source.shape[2] / self.source.chunks[2])
        for object_key in dl._s3_iter_objects(
            self.source.bucket,
            list_prefix,
            self.source.anon,
            region=self.source.region,
        ):
            chunk_key = dl._chunk_key_from_s3_object(
                object_key, level_prefix, self.source.dimension_separator,
            )
            if chunk_key is None:
                continue
            parts = chunk_key.split(self.source.dimension_separator)
            if len(parts) != 3 or not all(part.isdigit() for part in parts):
                continue
            cz, cy, cx = (int(part) for part in parts)
            if cz == int(iz) and 0 <= cy < y_chunks and 0 <= cx < x_chunks:
                result.add((cy, cx))
        with self._lock:
            self._stats["listed_planes"] += 1
            self._stats["listed_chunks"] += len(result)
            self._stats["missing_chunks"] += max(0, y_chunks * x_chunks - len(result))
        return result

    def _plane_inventory(self, iz: int) -> Future[set[tuple[int, int]]]:
        with self._lock:
            future = self._inventory_futures.get(int(iz))
            if future is None:
                future = self._inventory_pool.submit(self._list_plane, int(iz))
                self._inventory_futures[int(iz)] = future
            return future

    def _chunk_ranges(
        self, bounds_zyx: tuple[int, int, int, int, int, int],
    ) -> tuple[range, range, range]:
        z0, z1, y0, y1, x0, x1 = bounds_zyx
        cz, cy, cx = self.source.chunks
        return (
            range(max(0, z0 // cz), min(math.ceil(self.source.shape[0] / cz), math.ceil(z1 / cz))),
            range(max(0, y0 // cy), min(math.ceil(self.source.shape[1] / cy), math.ceil(y1 / cy))),
            range(max(0, x0 // cx), min(math.ceil(self.source.shape[2] / cx), math.ceil(x1 / cx))),
        )

    def _read_ahead(self, bounds_zyx: tuple[int, int, int, int, int, int]) -> bool:
        """Reads the region into the caches; False when nothing is stored there."""
        present = self.region_has_remote_chunks(bounds_zyx)
        if present:
            assert self._reader is not None
            z0, z1, y0, y1, x0, x1 = bounds_zyx
            self._reader[max(0, z0):z1, max(0, y0):y1, max(0, x0):x1]
        with self._lock:
            self._stats["materialized_regions"] += 1
        return present

    def request_region(
        self, bounds_zyx: tuple[int, int, int, int, int, int],
    ) -> Future[bool]:
        with self._lock:
            if self._closed:
                raise RuntimeError("live cache is closed")
            self._stats["requested_regions"] += 1
        return self._read_pool.submit(self._read_ahead, bounds_zyx)

    def region_has_remote_chunks(
        self, bounds_zyx: tuple[int, int, int, int, int, int],
    ) -> bool:
        """Authoritatively test sparse source support without local-state bias."""
        z_values, y_values, x_values = self._chunk_ranges(bounds_zyx)
        inventory_tasks = {iz: self._plane_inventory(iz) for iz in z_values}
        for iz, task in inventory_tasks.items():
            present = task.result()
            if any((iy, ix) in present for iy in y_values for ix in x_values):
                return True
        return False

    def snapshot(self) -> dict[str, Any]:
        reader_stats = self._final_reader_stats
        if reader_stats is None:
            reader_stats = self._reader.stats() if self._reader is not None else {}
        with self._lock:
            elapsed = max(1.0e-6, time.monotonic() - self._started_at)
            downloaded_bytes = int(reader_stats.get("remote_bytes", 0))
            return {
                "enabled": 1,
                "level": self.source.level,
                "target_bytes": self.max_bytes,
                "lookahead_tiles": self.lookahead_tiles,
                "active_inventory_planes": sum(not task.done() for task in self._inventory_futures.values()),
                "inflight_regions": self._stats["requested_regions"] - self._stats["materialized_regions"],
                "elapsed_seconds": elapsed,
                "downloaded_chunks": int(reader_stats.get("remote_fetches", 0)),
                "downloaded_bytes": downloaded_bytes,
                "disk_hits": int(reader_stats.get("disk_hits", 0)),
                "download_rate_bytes_s": downloaded_bytes / elapsed,
                **self._stats,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._read_pool.shutdown(wait=True, cancel_futures=True)
        self._inventory_pool.shutdown(wait=True, cancel_futures=True)
        if self._reader is not None:
            self._reader.flush()
            self._final_reader_stats = self._reader.stats()
            self._reader.close()
        snapshot = self.snapshot()
        print(
            f"[live-fetch] done downloaded={snapshot['downloaded_bytes'] / (1 << 40):.2f}TiB "
            f"in {snapshot['downloaded_chunks']} chunks, {snapshot['disk_hits']} from disk",
            flush=True,
        )
