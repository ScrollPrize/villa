from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator


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
        mode = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(self._handle.fileno(), mode | fcntl.LOCK_NB)
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
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
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


def _chunk_key(separator: str, iz: int, iy: int, ix: int) -> str:
    return separator.join(str(value) for value in (iz, iy, ix))


def _chunk_path(source: SelectedLevelSource, iz: int, iy: int, ix: int) -> Path:
    key = _chunk_key(source.dimension_separator, iz, iy, ix)
    return source.level_path / Path(key.replace("/", os.sep))


def _valid_regular_file(path: Path) -> bool:
    try:
        stat = path.lstat()
    except OSError:
        return False
    return not path.is_symlink() and path.is_file() and stat.st_size > 0


def _iter_plane_files(source: SelectedLevelSource, iz: int) -> Iterator[Path]:
    if source.dimension_separator == ".":
        prefix = f"{int(iz)}."
        try:
            entries = os.scandir(source.level_path)
        except OSError:
            return
        with entries:
            for entry in entries:
                if not entry.name.startswith(prefix) or entry.is_symlink():
                    continue
                parts = entry.name.split(".")
                if len(parts) == 3 and all(part.isdigit() for part in parts):
                    path = Path(entry.path)
                    if _valid_regular_file(path):
                        yield path
        return

    z_path = source.level_path / str(int(iz))
    if z_path.is_symlink() or not z_path.is_dir():
        return
    try:
        y_entries = os.scandir(z_path)
    except OSError:
        return
    with y_entries:
        for y_entry in y_entries:
            if y_entry.is_symlink() or not y_entry.name.isdigit() or not y_entry.is_dir(follow_symlinks=False):
                continue
            try:
                x_entries = os.scandir(y_entry.path)
            except OSError:
                continue
            with x_entries:
                for x_entry in x_entries:
                    if x_entry.is_symlink() or not x_entry.name.isdigit():
                        continue
                    path = Path(x_entry.path)
                    if _valid_regular_file(path):
                        yield path


def inventory_selected_level(source: SelectedLevelSource) -> tuple[dict[int, int], dict[int, int]]:
    plane_bytes: dict[int, int] = defaultdict(int)
    plane_counts: dict[int, int] = defaultdict(int)
    nz = math.ceil(source.shape[0] / source.chunks[0])
    if source.dimension_separator == "/":
        try:
            entries = os.scandir(source.level_path)
        except OSError:
            return {}, {}
        with entries:
            z_values = sorted(
                int(entry.name)
                for entry in entries
                if entry.name.isdigit() and not entry.is_symlink()
                and entry.is_dir(follow_symlinks=False) and int(entry.name) < nz
            )
    else:
        z_values_set: set[int] = set()
        try:
            entries = os.scandir(source.level_path)
        except OSError:
            return {}, {}
        with entries:
            for entry in entries:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    continue
                parts = entry.name.split(".")
                if len(parts) == 3 and all(part.isdigit() for part in parts):
                    iz = int(parts[0])
                    if iz < nz:
                        z_values_set.add(iz)
        z_values = sorted(z_values_set)
    for iz in z_values:
        for path in _iter_plane_files(source, iz):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            plane_bytes[iz] += int(size)
            plane_counts[iz] += 1
    return dict(plane_bytes), dict(plane_counts)


class LiveOmeZarrCache:
    """Bounded, Z-forward selected-level materialization cache."""

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
        raw_path = Path(str(input_path).rstrip("/")).expanduser()
        if not raw_path.name.isdigit():
            raise ValueError(
                f"live fetch requires an input path ending in a numeric OME-Zarr level: {input_path}"
            )
        self._level_lock = SelectedLevelLock(
            raw_path.parent.resolve(), int(raw_path.name), exclusive=True,
        )
        self._level_lock.__enter__()
        try:
            self.source = prepare_selected_level_source(input_path)
        except BaseException:
            self._level_lock.__exit__(None, None, None)
            raise
        self.max_bytes = int(max_bytes)
        self.lookahead_tiles = int(lookahead_tiles)
        self.workers = int(workers)
        try:
            self._plane_bytes, self._plane_counts = inventory_selected_level(self.source)
        except BaseException:
            self._level_lock.__exit__(None, None, None)
            raise
        self._resident_bytes = sum(self._plane_bytes.values())
        self._resident_chunks = sum(self._plane_counts.values())
        self._peak_resident_bytes = self._resident_bytes
        self._safe_plane_exclusive = 0
        self._inventory_futures: dict[int, Future[set[tuple[int, int]]]] = {}
        self._chunk_futures: dict[tuple[int, int, int], Future[int]] = {}
        self._active_planes: dict[int, int] = defaultdict(int)
        self._lock = threading.RLock()
        self._closed = False
        self._inventory_pool = None
        self._download_pool = None
        self._planner_pool = None
        try:
            self._inventory_pool = ThreadPoolExecutor(
                max_workers=min(16, self.workers), thread_name_prefix="live-zarr-list",
            )
            self._download_pool = ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="live-zarr-get",
            )
            self._planner_pool = ThreadPoolExecutor(
                max_workers=min(32, self.workers), thread_name_prefix="live-zarr-plan",
            )
        except BaseException:
            for pool in (self._planner_pool, self._download_pool, self._inventory_pool):
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
            self._level_lock.__exit__(None, None, None)
            raise
        self._stats = defaultdict(int)
        self._started_at = time.monotonic()
        print(
            f"[live-fetch] level={self.source.level} shape={self.source.shape} "
            f"chunks={self.source.chunks} resident={self._resident_bytes / (1 << 40):.2f}TiB "
            f"target={self.max_bytes / (1 << 40):.2f}TiB "
            f"lookahead_tiles={self.lookahead_tiles} workers={self.workers}",
            flush=True,
        )

    def __enter__(self) -> "LiveOmeZarrCache":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

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
                assert self._inventory_pool is not None
                future = self._inventory_pool.submit(self._list_plane, int(iz))
                self._inventory_futures[int(iz)] = future
            return future

    def _download_chunk(self, iz: int, iy: int, ix: int) -> int:
        dl = _downloader()
        local_path = _chunk_path(self.source, iz, iy, ix)
        remote_key = (
            f"{self.source.prefix}/{self.source.level}/"
            f"{_chunk_key(self.source.dimension_separator, iz, iy, ix)}"
        )
        for attempt in range(3):
            try:
                size = dl._download_chunk_atomic(
                    self.source.bucket,
                    remote_key,
                    str(local_path),
                    self.source.anon,
                    region=self.source.region,
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        with self._lock:
            self._plane_bytes[iz] = self._plane_bytes.get(iz, 0) + int(size)
            self._plane_counts[iz] = self._plane_counts.get(iz, 0) + 1
            self._resident_bytes += int(size)
            self._resident_chunks += 1
            self._peak_resident_bytes = max(self._peak_resident_bytes, self._resident_bytes)
            self._stats["downloaded_chunks"] += 1
            self._stats["downloaded_bytes"] += int(size)
        return int(size)

    def _ensure_chunk(self, iz: int, iy: int, ix: int) -> Future[int] | None:
        path = _chunk_path(self.source, iz, iy, ix)
        if _valid_regular_file(path):
            with self._lock:
                self._stats["reused_chunks"] += 1
            return None
        key = (int(iz), int(iy), int(ix))
        with self._lock:
            existing = self._chunk_futures.get(key)
            if existing is not None:
                return existing
            assert self._download_pool is not None
            future = self._download_pool.submit(self._download_chunk, *key)
            self._chunk_futures[key] = future

        def forget(done: Future[int], *, chunk_key=key) -> None:
            with self._lock:
                if self._chunk_futures.get(chunk_key) is done:
                    self._chunk_futures.pop(chunk_key, None)

        future.add_done_callback(forget)
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

    def _materialize_region(
        self, bounds_zyx: tuple[int, int, int, int, int, int],
    ) -> bool:
        z_values, y_values, x_values = self._chunk_ranges(bounds_zyx)
        planes = tuple(z_values)
        with self._lock:
            for iz in planes:
                self._active_planes[iz] += 1
        try:
            inventory_tasks = {iz: self._plane_inventory(iz) for iz in planes}
            inventories = {iz: task.result() for iz, task in inventory_tasks.items()}
            futures: list[Future[int]] = []
            has_present = False
            for iz in planes:
                present = inventories[iz]
                for iy in y_values:
                    for ix in x_values:
                        if (iy, ix) not in present:
                            continue
                        has_present = True
                        future = self._ensure_chunk(iz, iy, ix)
                        if future is not None:
                            futures.append(future)
            for future in futures:
                future.result()
            with self._lock:
                self._stats["materialized_regions"] += 1
            return has_present
        finally:
            with self._lock:
                for iz in planes:
                    self._active_planes[iz] -= 1
                    if self._active_planes[iz] <= 0:
                        self._active_planes.pop(iz, None)

    def request_region(
        self, bounds_zyx: tuple[int, int, int, int, int, int],
    ) -> Future[bool]:
        with self._lock:
            if self._closed:
                raise RuntimeError("live cache is closed")
            self._stats["requested_regions"] += 1
        assert self._planner_pool is not None
        return self._planner_pool.submit(self._materialize_region, bounds_zyx)

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

    def _evict_plane(self, iz: int) -> tuple[int, int]:
        removed_bytes = 0
        removed_chunks = 0
        for path in tuple(_iter_plane_files(self.source, iz)):
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError:
                continue
            removed_bytes += int(size)
            removed_chunks += 1
        if self.source.dimension_separator == "/":
            z_path = self.source.level_path / str(int(iz))
            if not z_path.is_symlink() and z_path.is_dir():
                try:
                    y_paths = tuple(z_path.iterdir())
                except OSError:
                    y_paths = ()
                for y_path in y_paths:
                    if y_path.is_symlink() or not y_path.is_dir():
                        continue
                    try:
                        y_path.rmdir()
                    except OSError:
                        pass
                try:
                    z_path.rmdir()
                except OSError:
                    pass
        with self._lock:
            remaining_bytes = max(0, self._plane_bytes.get(iz, 0) - removed_bytes)
            remaining_chunks = max(0, self._plane_counts.get(iz, 0) - removed_chunks)
            if remaining_bytes:
                self._plane_bytes[iz] = remaining_bytes
            else:
                self._plane_bytes.pop(iz, None)
            if remaining_chunks:
                self._plane_counts[iz] = remaining_chunks
            else:
                self._plane_counts.pop(iz, None)
            self._resident_bytes = max(0, self._resident_bytes - removed_bytes)
            self._resident_chunks = max(0, self._resident_chunks - removed_chunks)
            self._stats["evicted_planes"] += 1
            self._stats["evicted_chunks"] += removed_chunks
            self._stats["evicted_bytes"] += removed_bytes
            self._inventory_futures.pop(iz, None)
        return removed_bytes, removed_chunks

    def advance_safe_boundary(self, input_voxel_z: int) -> None:
        safe_exclusive = max(0, int(input_voxel_z) // self.source.chunks[0])
        with self._lock:
            self._safe_plane_exclusive = max(self._safe_plane_exclusive, safe_exclusive)
        while True:
            with self._lock:
                if self._resident_bytes <= self.max_bytes:
                    break
                eligible = [
                    iz
                    for iz, byte_count in self._plane_bytes.items()
                    if byte_count > 0 and iz < self._safe_plane_exclusive
                    and self._active_planes.get(iz, 0) == 0
                    and not any(key[0] == iz for key in self._chunk_futures)
                ]
                if not eligible:
                    self._stats["over_target_events"] += 1
                    break
                iz = min(eligible)
            removed_bytes, removed_chunks = self._evict_plane(iz)
            print(
                f"[live-fetch] evict zchunk={iz} chunks={removed_chunks} "
                f"bytes={removed_bytes / (1 << 30):.2f}GiB "
                f"resident={self.snapshot()['resident_bytes'] / (1 << 40):.2f}TiB",
                flush=True,
            )
            if removed_chunks == 0:
                break
        with self._lock:
            for iz in tuple(self._inventory_futures):
                if (
                    iz < self._safe_plane_exclusive
                    and self._active_planes.get(iz, 0) == 0
                    and not any(key[0] == iz for key in self._chunk_futures)
                    and self._inventory_futures[iz].done()
                ):
                    self._inventory_futures.pop(iz, None)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = max(1.0e-6, time.monotonic() - self._started_at)
            return {
                "enabled": 1,
                "level": self.source.level,
                "target_bytes": self.max_bytes,
                "lookahead_tiles": self.lookahead_tiles,
                "resident_bytes": self._resident_bytes,
                "resident_chunks": self._resident_chunks,
                "peak_resident_bytes": self._peak_resident_bytes,
                "safe_plane_exclusive": self._safe_plane_exclusive,
                "active_inventory_planes": len(self._inventory_futures),
                "inflight_downloads": len(self._chunk_futures),
                "elapsed_seconds": elapsed,
                "download_rate_bytes_s": self._stats["downloaded_bytes"] / elapsed,
                "over_target": int(self._resident_bytes > self.max_bytes),
                **{str(key): int(value) for key, value in self._stats.items()},
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        assert self._planner_pool is not None
        assert self._download_pool is not None
        assert self._inventory_pool is not None
        try:
            self._planner_pool.shutdown(wait=True, cancel_futures=True)
            self._download_pool.shutdown(wait=True, cancel_futures=True)
            self._inventory_pool.shutdown(wait=True, cancel_futures=True)
            snapshot = self.snapshot()
            print(
                f"[live-fetch] done resident={snapshot['resident_bytes'] / (1 << 40):.2f}TiB "
                f"peak={snapshot['peak_resident_bytes'] / (1 << 40):.2f}TiB "
                f"downloaded={snapshot.get('downloaded_bytes', 0) / (1 << 40):.2f}TiB "
                f"evicted={snapshot.get('evicted_bytes', 0) / (1 << 40):.2f}TiB",
                flush=True,
            )
        finally:
            self._level_lock.__exit__(None, None, None)
