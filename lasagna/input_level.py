"""One OME-Zarr input level for tiled inference, read through vc_reader.

The level is a local directory or a remote ``s3://`` / ``https://`` URL; both go
through the same :class:`~vc_reader.reader.ZarrReader`, so a remote level needs
no separate download step. Remote chunks are cached decoded in ``cache_dir``
(bounded by ``cache_max_bytes``), and tiles can be read ahead into that cache
before the GPU needs them. A per-z-plane chunk inventory, listed through
:class:`~vc_reader.reader.writer.ObjectStore`, tells sparse volumes' empty
regions apart without reading them.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import math
import os
import threading
import time
from typing import Any

from vc_reader.reader import ZarrReader
from vc_reader.reader.writer import ObjectStore

DEFAULT_READ_AHEAD_TILES = 10_000
DEFAULT_INPUT_CONNECTIONS = 64

BoundsZYX = tuple[int, int, int, int, int, int]


def is_remote_location(location: str | os.PathLike[str]) -> bool:
    """Whether ``location`` is a URL rather than a local path."""
    return "://" in os.fspath(location)


def split_level_location(location: str | os.PathLike[str]) -> tuple[str, int | None]:
    """The OME-Zarr group location and numeric level of a level location.

    ``volume.zarr/2`` gives ``("volume.zarr", 2)``; a location whose last
    component is not a number gives ``(location, None)``. Works for URLs, which
    ``pathlib`` would mangle.
    """
    loc = os.fspath(location).rstrip("/")
    parent, _, name = loc.rpartition("/")
    if parent and name.isdigit():
        return parent, int(name)
    return loc, None


class InputLevel:
    """A 3-D Zarr level opened for tiled inference.

    ``region_has_chunks`` answers from a chunk listing of the level (one listing
    per z-chunk plane, cached), so empty regions of a sparse volume are skipped
    without reading. ``request_region`` reads a region into the reader's caches
    on a thread pool; the tiled loop uses it for read-ahead on remote levels.
    """

    def __init__(
        self,
        location: str | os.PathLike[str],
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        cache_max_bytes: int | None = None,
        memory_cache: int | None = None,
        threads: int | None = None,
        connections: int = DEFAULT_INPUT_CONNECTIONS,
        auth: Any = "auto",
        read_ahead_tiles: int = DEFAULT_READ_AHEAD_TILES,
    ) -> None:
        if int(connections) <= 0:
            raise ValueError("input connections must be > 0")
        if int(read_ahead_tiles) <= 0:
            raise ValueError("read_ahead_tiles must be > 0")
        if cache_max_bytes is not None and int(cache_max_bytes) <= 0:
            raise ValueError("cache_max_bytes must be > 0")
        self.location = os.fspath(location).rstrip("/")
        self.remote = is_remote_location(self.location)
        self.group_location, self.level = split_level_location(self.location)
        self.cache_dir = None if cache_dir is None else os.fspath(cache_dir)
        self.cache_max_bytes = None if cache_max_bytes is None else int(cache_max_bytes)
        self.connections = int(connections)
        self.read_ahead_tiles = int(read_ahead_tiles)
        self.reader = ZarrReader.open(
            self.location,
            cache_dir=self.cache_dir,
            cache_max_bytes=self.cache_max_bytes,
            memory_cache=memory_cache,
            threads=threads,
            connections=self.connections,
            auth=auth,
        )
        self.shape = tuple(int(v) for v in self.reader.shape)
        self.chunks = tuple(int(v) for v in self.reader.chunks)
        if len(self.shape) != 3 or len(self.chunks) != 3 or any(v <= 0 for v in (*self.shape, *self.chunks)):
            raise ValueError(
                f"tiled inference input must be a 3-D array with positive chunks, "
                f"got shape={self.shape} chunks={self.chunks}: {self.location}"
            )
        raw = self.reader.meta.raw
        if int(raw.get("zarr_format", 2)) == 3:
            self._chunk_prefix, self._separator = "c/", "/"
        else:
            self._chunk_prefix, self._separator = "", str(raw.get("dimension_separator", "."))
        # Listing needs the same credentials as reading on S3; other stores list anonymously
        # (a local directory) or not at all (a plain HTTP server, handled in region_has_chunks).
        self._store = ObjectStore(
            self.location,
            auth=auth if self.location.startswith("s3://") else None,
            connections=self.connections,
        )
        self._listable: bool | None = None
        self._planes: dict[int, Future[set[tuple[int, int]]]] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._final_stats: dict[str, Any] | None = None
        self._stats = {
            "requested_regions": 0, "materialized_regions": 0,
            "listed_planes": 0, "listed_chunks": 0, "missing_chunks": 0,
        }
        self._started_at = time.monotonic()
        self._list_pool = ThreadPoolExecutor(
            max_workers=min(16, self.connections), thread_name_prefix="input-level-list",
        )
        self._read_pool = ThreadPoolExecutor(
            max_workers=self.connections, thread_name_prefix="input-level-read",
        )
        print(
            f"[input] level={self.level} location={self.location} shape={self.shape} "
            f"chunks={self.chunks} remote={self.remote} cache_dir={self.cache_dir} "
            f"cache_max_bytes={self.cache_max_bytes} read_ahead_tiles={self.read_ahead_tiles} "
            f"connections={self.connections}",
            flush=True,
        )

    def __enter__(self) -> "InputLevel":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    # ---- chunk inventory -------------------------------------------------

    def _plane_prefix(self, iz: int) -> str:
        return f"{self._chunk_prefix}{int(iz)}{self._separator}"

    def _list_plane(self, iz: int) -> set[tuple[int, int]]:
        prefix = self._plane_prefix(iz)
        y_chunks = math.ceil(self.shape[1] / self.chunks[1])
        x_chunks = math.ceil(self.shape[2] / self.chunks[2])
        present: set[tuple[int, int]] = set()
        for key in self._store.keys(prefix):
            parts = key[len(self._chunk_prefix):].split(self._separator)
            if len(parts) != 3 or not all(part.isdigit() for part in parts):
                continue
            cz, cy, cx = (int(part) for part in parts)
            if cz == int(iz) and 0 <= cy < y_chunks and 0 <= cx < x_chunks:
                present.add((cy, cx))
        with self._lock:
            self._stats["listed_planes"] += 1
            self._stats["listed_chunks"] += len(present)
            self._stats["missing_chunks"] += max(0, y_chunks * x_chunks - len(present))
        return present

    def _plane(self, iz: int) -> Future[set[tuple[int, int]]]:
        with self._lock:
            future = self._planes.get(int(iz))
            if future is None:
                future = self._planes[int(iz)] = self._list_pool.submit(self._list_plane, int(iz))
            return future

    def _chunk_ranges(self, bounds: BoundsZYX) -> tuple[range, range, range]:
        z0, z1, y0, y1, x0, x1 = bounds
        cz, cy, cx = self.chunks
        return (
            range(max(0, z0 // cz), min(math.ceil(self.shape[0] / cz), math.ceil(z1 / cz))),
            range(max(0, y0 // cy), min(math.ceil(self.shape[1] / cy), math.ceil(y1 / cy))),
            range(max(0, x0 // cx), min(math.ceil(self.shape[2] / cx), math.ceil(x1 / cx))),
        )

    def region_has_chunks(self, bounds: BoundsZYX) -> bool:
        """Whether any chunk of the level is stored inside ``bounds`` (z0,z1,y0,y1,x0,x1).

        A store that cannot be listed (a plain HTTP server) reports every region
        as stored; the reader then returns ``fill_value`` for chunks it finds missing.
        """
        z_values, y_values, x_values = self._chunk_ranges(bounds)
        if not (z_values and y_values and x_values):
            return False
        if self._listable is False:
            return True
        planes = {iz: self._plane(iz) for iz in z_values}
        for task in planes.values():
            try:
                present = task.result()
            except Exception as error:
                with self._lock:
                    if self._listable is None:
                        print(
                            f"[input] chunk listing unavailable ({type(error).__name__}: {error}); "
                            "reading every region",
                            flush=True,
                        )
                        self._listable = False
                return True
            if any((iy, ix) in present for iy in y_values for ix in x_values):
                self._listable = True
                return True
        self._listable = True
        return False

    # ---- read-ahead --------------------------------------------------------

    def _materialize(self, bounds: BoundsZYX) -> bool:
        present = self.region_has_chunks(bounds)
        if present:
            z0, z1, y0, y1, x0, x1 = bounds
            self.reader[max(0, z0):z1, max(0, y0):y1, max(0, x0):x1]
        with self._lock:
            self._stats["materialized_regions"] += 1
        return present

    def request_region(self, bounds: BoundsZYX) -> Future[bool]:
        """Read ``bounds`` into the reader's caches; the result says whether anything is stored there."""
        with self._lock:
            if self._closed:
                raise RuntimeError("input level is closed")
            self._stats["requested_regions"] += 1
        return self._read_pool.submit(self._materialize, bounds)

    # ---- lifecycle ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        reader_stats = self._final_stats
        if reader_stats is None:
            reader_stats = self.reader.stats()
        with self._lock:
            elapsed = max(1.0e-6, time.monotonic() - self._started_at)
            fetched_bytes = int(reader_stats.get("remote_bytes", 0))
            return {
                "location": self.location,
                "level": self.level,
                "remote": self.remote,
                "cache_dir": self.cache_dir,
                "cache_max_bytes": self.cache_max_bytes,
                "read_ahead_tiles": self.read_ahead_tiles,
                "connections": self.connections,
                "elapsed_seconds": elapsed,
                "inflight_regions": self._stats["requested_regions"] - self._stats["materialized_regions"],
                "fetched_chunks": int(reader_stats.get("remote_fetches", 0)),
                "fetched_bytes": fetched_bytes,
                "fetch_rate_bytes_s": fetched_bytes / elapsed,
                "disk_hits": int(reader_stats.get("disk_hits", 0)),
                "memory_hits": int(reader_stats.get("memory_hits", 0)),
                **self._stats,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._read_pool.shutdown(wait=True, cancel_futures=True)
        self._list_pool.shutdown(wait=True, cancel_futures=True)
        self.reader.flush()
        self._final_stats = self.reader.stats()
        self.reader.close()
        snapshot = self.snapshot()
        if self.remote:
            print(
                f"[input] done fetched={snapshot['fetched_bytes'] / (1 << 30):.2f}GiB "
                f"in {snapshot['fetched_chunks']} chunks, {snapshot['disk_hits']} from disk cache",
                flush=True,
            )


__all__ = [
    "DEFAULT_INPUT_CONNECTIONS",
    "DEFAULT_READ_AHEAD_TILES",
    "InputLevel",
    "is_remote_location",
    "split_level_location",
]
