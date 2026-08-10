"""Local/remote Zarr opening, resolution selection, padding, and disk caching."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp
import numpy as np
import zarr

from vesuvius.data.utils import open_zarr as open_vesuvius_zarr


_PUBLIC_S3_BUCKET = "vesuvius-challenge-open-data"


def normalize_volume_url(path: str | Path) -> tuple[str, bool]:
    """Return the canonical URL and whether it names the public S3 bucket."""
    path_text = str(path)
    parsed = urlsplit(path_text)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme == "s3" and hostname == _PUBLIC_S3_BUCKET:
        return f"s3://{_PUBLIC_S3_BUCKET}{parsed.path}", True
    if parsed.scheme not in {"http", "https"}:
        return path_text, False

    virtual_prefix = f"{_PUBLIC_S3_BUCKET}.s3"
    virtual_host = hostname == f"{virtual_prefix}.amazonaws.com" or (
        hostname.startswith((f"{virtual_prefix}.", f"{virtual_prefix}-"))
        and hostname.endswith(".amazonaws.com")
    )
    if virtual_host:
        object_path = parsed.path.lstrip("/")
        suffix = f"/{object_path}" if object_path else ""
        return f"s3://{_PUBLIC_S3_BUCKET}{suffix}", True

    path_style_host = hostname == "s3.amazonaws.com" or (
        hostname.startswith(("s3.", "s3-"))
        and hostname.endswith(".amazonaws.com")
    )
    path_parts = parsed.path.lstrip("/").split("/", 1)
    if path_style_host and path_parts[0] == _PUBLIC_S3_BUCKET:
        suffix = f"/{path_parts[1]}" if len(path_parts) == 2 else ""
        return f"s3://{_PUBLIC_S3_BUCKET}{suffix}", True
    return path_text, False


@contextmanager
def _locked_cache(cache_dir: Path):
    import fcntl

    lock_path = cache_dir / ".cache.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _cache_snapshot(cache_dir: Path) -> list[tuple[int, int, Path]]:
    snapshot = []
    for entry in os.scandir(cache_dir):
        if entry.name == ".cache.lock" or entry.name.endswith(".tmp"):
            continue
        try:
            if not entry.is_file(follow_symlinks=False):
                continue
            stat = entry.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        snapshot.append((stat.st_mtime_ns, stat.st_size, Path(entry.path)))
    snapshot.sort(key=lambda item: (item[0], item[2].name))
    return snapshot


def _remove_temporary_files_locked(cache_dir: Path) -> None:
    for entry in os.scandir(cache_dir):
        if not entry.name.endswith(".tmp"):
            continue
        try:
            Path(entry.path).unlink()
        except FileNotFoundError:
            continue


def _evict_to_budget_locked(cache_dir: Path, max_bytes: int | None) -> None:
    _remove_temporary_files_locked(cache_dir)
    if max_bytes is None:
        return
    snapshot = _cache_snapshot(cache_dir)
    total = sum(size for _, size, _ in snapshot)
    if total <= max_bytes:
        return
    target_bytes = 0.9 * max_bytes
    while True:
        snapshot = _cache_snapshot(cache_dir)
        total = sum(size for _, size, _ in snapshot)
        if total <= target_bytes or not snapshot:
            return
        try:
            snapshot[0][2].unlink()
        except FileNotFoundError:
            continue


def _install_cache_value(
    cache_dir: Path, path: Path, value: bytes, max_bytes: int | None
) -> None:
    temporary_path: Path | None = None
    try:
        with _locked_cache(cache_dir):
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=cache_dir,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                stream.write(value)
                temporary_path = Path(stream.name)
            os.replace(temporary_path, path)
            temporary_path = None
            _evict_to_budget_locked(cache_dir, max_bytes)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if hasattr(zarr.storage, "WrapperStore"):

    from zarr.abc.store import (
        OffsetByteRequest,
        RangeByteRequest,
        Store,
        SuffixByteRequest,
    )

    def _make_fsspec_store(
        url: str, storage_options: dict[str, Any], read_only: bool
    ):
        return zarr.storage.FsspecStore.from_url(
            url,
            storage_options=storage_options,
            read_only=read_only,
        )


    class PidAwareFsspecStore(Store):
        """Build a fresh fsspec-backed read-only store in each process."""

        def __init__(
            self,
            url: str,
            storage_options: dict[str, Any],
            *,
            read_only: bool = True,
            store_factory=None,
            pid_provider=None,
        ) -> None:
            super().__init__(read_only=read_only)
            self.url = str(url)
            self.storage_options = dict(storage_options)
            self._store_factory = (
                _make_fsspec_store if store_factory is None else store_factory
            )
            self._pid_provider = (
                os.getpid if pid_provider is None else pid_provider
            )
            self._process_store = None
            self._process_id: int | None = None

        def __getstate__(self):
            state = self.__dict__.copy()
            state["_process_store"] = None
            state["_process_id"] = None
            return state

        def __eq__(self, other: object) -> bool:
            return (
                isinstance(other, PidAwareFsspecStore)
                and self.url == other.url
                and self.storage_options == other.storage_options
                and self.read_only == other.read_only
            )

        def with_read_only(self, read_only: bool = False):
            return type(self)(
                self.url,
                self.storage_options,
                read_only=read_only,
                store_factory=self._store_factory,
                pid_provider=self._pid_provider,
            )

        @property
        def supports_writes(self) -> bool:
            return False

        @property
        def supports_deletes(self) -> bool:
            return False

        @property
        def supports_listing(self) -> bool:
            return True

        def _current_store(self):
            process_id = int(self._pid_provider())
            if self._process_store is None or self._process_id != process_id:
                options = dict(self.storage_options)
                options["skip_instance_cache"] = True
                self._process_store = self._store_factory(
                    self.url, options, self.read_only
                )
                self._process_id = process_id
            return self._process_store

        async def get(self, key, prototype, byte_range=None):
            return await self._current_store().get(key, prototype, byte_range)

        async def get_partial_values(self, prototype, key_ranges):
            return await self._current_store().get_partial_values(
                prototype, key_ranges
            )

        async def exists(self, key: str) -> bool:
            return await self._current_store().exists(key)

        async def set(self, key, value) -> None:
            raise NotImplementedError("PID-aware volume stores do not support writes")

        async def delete(self, key: str) -> None:
            raise NotImplementedError("PID-aware volume stores do not support deletes")

        async def list(self):
            async for key in self._current_store().list():
                yield key

        async def list_prefix(self, prefix: str):
            async for key in self._current_store().list_prefix(prefix):
                yield key

        async def list_dir(self, prefix: str):
            async for key in self._current_store().list_dir(prefix):
                yield key

        def close(self) -> None:
            if (
                self._process_store is not None
                and self._process_id == int(self._pid_provider())
            ):
                self._process_store.close()
            self._process_store = None
            self._process_id = None
            super().close()


    def _apply_byte_range(value: bytes, byte_range) -> bytes:
        if byte_range is None:
            return value
        if isinstance(byte_range, RangeByteRequest):
            return value[byte_range.start : byte_range.end]
        if isinstance(byte_range, OffsetByteRequest):
            return value[byte_range.offset :]
        if isinstance(byte_range, SuffixByteRequest):
            if byte_range.suffix == 0:
                return b""
            return value[-byte_range.suffix :]
        raise ValueError(f"Unexpected byte_range, got {byte_range}.")

    class AsyncDiskCachedStore(zarr.storage.WrapperStore):
        """Zarr-3 read-through cache for complete compressed store values."""

        def __init__(self, store, cache_dir: Path, max_bytes: int | None) -> None:
            super().__init__(store)
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.max_bytes = max_bytes
            if max_bytes is not None and max_bytes < 0:
                raise ValueError("max_bytes must be nonnegative or None")
            with _locked_cache(self.cache_dir):
                _evict_to_budget_locked(self.cache_dir, self.max_bytes)

        def _path(self, key: str) -> Path:
            return self.cache_dir / key.replace("/", "__")

        def _with_store(self, store):
            return type(self)(store, self.cache_dir, self.max_bytes)

        async def get(self, key, prototype, byte_range=None):
            path = self._path(key)
            try:
                value = path.read_bytes()
            except FileNotFoundError:
                source_value = await self._store.get(key, prototype, None)
                if source_value is None:
                    return None
                value = source_value.to_bytes()
                _install_cache_value(
                    self.cache_dir, path, value, self.max_bytes
                )
            else:
                try:
                    os.utime(path)
                except FileNotFoundError:
                    pass
            return prototype.buffer.from_bytes(_apply_byte_range(value, byte_range))

        async def get_partial_values(self, prototype, key_ranges):
            return [
                await self.get(key, prototype, byte_range)
                for key, byte_range in key_ranges
            ]

        async def _get_many(self, requests):
            for key, prototype, byte_range in requests:
                yield key, await self.get(key, prototype, byte_range)

else:
    AsyncDiskCachedStore = None
    PidAwareFsspecStore = None


def load_volume_auth(auth_json_path: str | Path | None) -> tuple[str, str] | None:
    """Read the exact username/password JSON boundary used for HTTPS volumes."""
    if auth_json_path is None:
        return None
    with Path(auth_json_path).open("r", encoding="utf-8") as stream:
        authored = json.load(stream)
    if not isinstance(authored, dict) or "username" not in authored or "password" not in authored:
        raise ValueError("volume auth JSON requires username and password")
    return str(authored["username"]), str(authored["password"])


class DiskCachedMapping(MutableMapping[str, bytes]):
    """Read-through compressed-byte cache with atomic writes and mtime LRU."""

    def __init__(
        self, source: MutableMapping[str, bytes], cache_dir: Path, max_bytes: int | None
    ) -> None:
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative or None")
        with _locked_cache(self.cache_dir):
            _evict_to_budget_locked(self.cache_dir, self.max_bytes)

    def _path(self, key: str) -> Path:
        return self.cache_dir / key.replace("/", "__")

    def __getitem__(self, key: str) -> bytes:
        path = self._path(key)
        try:
            value = path.read_bytes()
        except FileNotFoundError:
            value = self.source[key]
            _install_cache_value(self.cache_dir, path, value, self.max_bytes)
            return value
        try:
            os.utime(path)
        except FileNotFoundError:
            pass
        return value

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (self._path(key).exists() or key in self.source)

    def __setitem__(self, key: str, value: bytes) -> None:
        raise NotImplementedError("read-only store")

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("read-only store")

    def __iter__(self):
        return iter(self.source)

    def __len__(self) -> int:
        return len(self.source)


def disk_cache_subdir(source_path: str, cache_dir: Path) -> Path:
    digest = hashlib.sha1(str(source_path).encode()).hexdigest()[:12]
    return Path(cache_dir) / digest


def _available_top_level_keys(root: Any) -> tuple[str, ...]:
    if not hasattr(root, "group_keys"):
        return ()
    return tuple(
        sorted(
            {
                *(str(key) for key in root.group_keys()),
                *(str(key) for key in root.array_keys()),
            }
        )
    )


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
    """Open a Zarr root while preserving public-S3 and Basic-Auth rules."""
    path_text, is_public_s3 = normalize_volume_url(path)
    storage_options: dict[str, Any] = {}
    if is_public_s3:
        storage_options["anon"] = True
    auth = load_volume_auth(auth_json_path)
    if not is_public_s3 and path_text.startswith("https://") and auth is not None:
        storage_options["client_kwargs"] = {
            "auth": aiohttp.BasicAuth(auth[0], auth[1])
        }
    is_remote = path_text.startswith(("s3://", "http://", "https://"))
    if cache_dir is not None and AsyncDiskCachedStore is not None:
        if is_remote:
            source_store = PidAwareFsspecStore(
                path_text.rstrip("/"),
                storage_options,
            )
        else:
            source_store = zarr.storage.LocalStore(path_text, read_only=True)
        maximum_bytes = (
            None if cache_max_gb is None else int(float(cache_max_gb) * 1e9)
        )
        store = AsyncDiskCachedStore(
            source_store,
            disk_cache_subdir(path_text, Path(cache_dir)),
            maximum_bytes,
        )
        root = zarr.open(store=store, mode="r")
    elif PidAwareFsspecStore is not None and is_remote:
        root = zarr.open(
            store=PidAwareFsspecStore(path_text.rstrip("/"), storage_options),
            mode="r",
        )
    elif cache_dir is not None:
        import fsspec

        if path_text.startswith("s3://"):
            source = fsspec.get_mapper(path_text, **storage_options)
        elif path_text.startswith(("http://", "https://")):
            source = fsspec.get_mapper(path_text, **storage_options)
        else:
            source = zarr.storage.DirectoryStore(path_text)
        maximum_bytes = (
            None if cache_max_gb is None else int(float(cache_max_gb) * 1e9)
        )
        root = zarr.open(
            store=DiskCachedMapping(
                source,
                disk_cache_subdir(path_text, Path(cache_dir)),
                maximum_bytes,
            ),
            mode="r",
        )
    else:
        root = open_vesuvius_zarr(
            path_text, mode="r", storage_options=storage_options
        )
    return root


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
