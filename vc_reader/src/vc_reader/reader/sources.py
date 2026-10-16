"""Where chunks come from: a local zarr directory, an HTTP(S) endpoint, or a
remote endpoint behind a persistent on-disk cache of decoded chunks.

Every source returns a full chunk as an ndarray, or None for a chunk that is
not stored (it reads as ``fill_value``).
"""

from __future__ import annotations

import mmap
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .zarr_meta import ArrayMeta, ChunkDecoder

_MADV_WILLNEED = getattr(mmap, "MADV_WILLNEED", None)


def read_file_into(path: str, nbytes: int) -> np.ndarray | None:
    """Read a whole file of exactly ``nbytes`` into a new uint8 array; None if absent."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        return None
    try:
        buf = np.empty(nbytes, dtype=np.uint8)
        view = memoryview(buf)
        got = 0
        while got < nbytes:
            n = os.readv(fd, [view[got:]])
            if n == 0:
                raise ValueError(f"{path}: {got} bytes, expected {nbytes}")
            got += n
        return buf
    finally:
        os.close(fd)


def willneed(arr: np.memmap, lo: tuple[int, ...], hi: tuple[int, ...]) -> None:
    """Hint the kernel to read the pages of a mapped 3D chunk that ``[lo, hi)`` touches.

    Ported from vesuvius crop_sampling.volume._willneed_rows: a fault on a file
    mapping otherwise reads a whole read-around window one blocking fault at a
    time. Advisory only; values are unchanged.
    """
    mm = getattr(arr, "_mmap", None)
    if mm is None or _MADV_WILLNEED is None or arr.ndim != 3 or getattr(arr, "offset", 0) != 0:
        return
    page = mmap.PAGESIZE
    plane, row = arr.strides[0], arr.strides[1]
    z0, z1, first = int(lo[0]), int(hi[0]), int(lo[1]) * row
    need = (int(hi[1]) - int(lo[1])) * row
    spans = (
        [(z0 * plane + first, (z1 - 1) * plane + first + need)]
        if plane - need <= 3 * need
        else [(z * plane + first, z * plane + first + need) for z in range(z0, z1)]
    )
    try:
        for s, e in spans:
            s = s // page * page
            mm.madvise(_MADV_WILLNEED, s, -(-e // page) * page - s)
    except OSError:
        pass


def _pread_into(fd: int, view: memoryview, offset: int) -> int:
    """Fill ``view`` from file offset ``offset``; returns bytes read (short only at EOF)."""
    got = 0
    n = len(view)
    while got < n:
        if _HAVE_PREADV:
            k = os.preadv(fd, [view[got:]], offset + got)
        else:
            os.lseek(fd, offset + got, os.SEEK_SET)
            k = os.readv(fd, [view[got:]])
        if k == 0:
            break
        got += k
    return got


_HAVE_PREADV = hasattr(os, "preadv")
_FADV_WILLNEED = getattr(os, "POSIX_FADV_WILLNEED", None) if hasattr(os, "posix_fadvise") else None
_O_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)

#: How LocalSource reads uncompressed chunks into a region.
READ_MODES = ("mmap", "pread", "slab")


class ChunkSource:
    """Base class. ``get`` and ``read_into`` must be thread-safe."""

    meta: ArrayMeta
    #: Stable identity used to key shared memory caches.
    identity: str
    #: True when ``get`` can return a memory map (cheap, lazily paged).
    maps_files: bool = False

    def get(self, idx: tuple[int, ...], *, mapped: bool = False) -> np.ndarray | None:
        raise NotImplementedError

    def read_into(self, idx: tuple[int, ...], src: tuple[slice, ...], dst: np.ndarray) -> bool:
        """Copy ``chunk[src]`` into ``dst``; False (``dst`` untouched) if the chunk is not stored."""
        arr = self.get(idx)
        if arr is None:
            return False
        dst[...] = arr[src]
        return True

    def close(self) -> None:
        pass


class LocalSource(ChunkSource):
    """Chunks of a zarr v2 array in a local directory.

    ``read_mode`` selects how uncompressed chunks are copied into a region:

    * ``"slab"`` (default): ``pread`` only the contiguous byte range of the
      chunk the region touches (first to last needed z/y row) into a reused
      per-thread buffer.
    * ``"pread"``: ``pread`` the whole chunk into a reused per-thread buffer.
    * ``"mmap"``: memory-map the chunk and hint the touched pages with
      ``MADV_WILLNEED`` (the vesuvius ``ChunkedArray`` approach).
    """

    def __init__(self, root: str | os.PathLike[str], meta: ArrayMeta | None = None, *, read_mode: str = "slab"):
        if read_mode not in READ_MODES:
            raise ValueError(f"read_mode must be one of {READ_MODES}")
        self.root = os.path.abspath(os.fspath(root))
        self.meta = meta or ArrayMeta.from_bytes(Path(self.root, ".zarray").read_bytes())
        self.identity = "file://" + self.root
        self.read_mode = read_mode
        self.maps_files = self.meta.is_raw and read_mode == "mmap"
        self._decoder = ChunkDecoder(self.meta)
        self._tls = threading.local()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_tls"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tls = threading.local()

    def path(self, idx: tuple[int, ...]) -> str:
        return os.path.join(self.root, self.meta.chunk_key(idx))

    def _scratch(self) -> np.ndarray:
        buf = getattr(self._tls, "buf", None)
        if buf is None:
            buf = self._tls.buf = np.empty(self.meta.chunk_nbytes, dtype=np.uint8)
        return buf

    @property
    def reads_ranges(self) -> bool:
        """True when chunks are read with pread (``slab``/``pread`` modes on raw 3D chunks)."""
        return self.meta.is_raw and self.read_mode != "mmap" and self.meta.ndim == 3

    def _layout(self, src):
        """File spans to read for ``chunk[src]``, and how they lie in the scratch buffer.

        Returns ``(spans, offset, strides)``: ``spans`` are ``(file offset, length)``
        packed back to back into the buffer, and the ``[z, y, x-all]`` view of the
        region starts at ``offset`` with ``strides``.
        """
        m = self.meta
        item = m.dtype.itemsize
        _, cy, cx = m.chunks
        row, plane = cx * item, cy * cx * item
        z0, z1, y0, y1 = src[0].start, src[0].stop, src[1].start, src[1].stop
        if self.read_mode == "pread":
            return [(0, m.chunk_nbytes)], z0 * plane + y0 * row, (plane, row, item)
        # One contiguous read from the first needed row to the last. Splitting it
        # into per-plane reads saves at most a plane of copying per z, but costs a
        # syscall per z (measured: ~14 syscalls per piece, 2x slower warm), and
        # the readahead hint covers the whole range either way.
        start = z0 * plane + y0 * row
        return [(start, (z1 - 1) * plane + y1 * row - start)], 0, (plane, row, item)

    def _read_fd_into(self, fd, idx, src, dst) -> None:
        buf = self._scratch()
        view = memoryview(buf)
        spans, offset, strides = self._layout(src)
        pos = 0
        for off, n in spans:
            if _pread_into(fd, view[pos:pos + n], off) != n:
                raise ValueError(f"{self.path(idx)}: short chunk file")
            pos += n
        vals = np.ndarray((src[0].stop - src[0].start, src[1].stop - src[1].start, self.meta.chunks[2]),
                          dtype=self.meta.dtype, buffer=buf, offset=offset, strides=strides)
        dst[...] = vals[:, :, src[2]]

    def read_into(self, idx, src, dst):
        if not self.reads_ranges:
            return super().read_into(idx, src, dst)
        try:
            fd = os.open(self.path(idx), _O_FLAGS)
        except FileNotFoundError:
            return False
        try:
            self._read_fd_into(fd, idx, src, dst)
            return True
        finally:
            os.close(fd)

    def open_hinted(self, pieces) -> list[int | None]:
        """Open the chunk file of every ``(idx, src, dst)`` piece and queue readahead for its bytes.

        The ranges go to the kernel as asynchronous readahead
        (``POSIX_FADV_WILLNEED``), so cold pieces are fetched from the device
        concurrently rather than one blocking read at a time. Advisory only:
        the values read are unchanged. Returns one fd per piece (None if the
        chunk is not stored); the caller closes them.
        """
        fds: list[int | None] = []
        try:
            for idx, src, _ in pieces:
                try:
                    fd = os.open(self.path(idx), _O_FLAGS)
                except FileNotFoundError:
                    fds.append(None)
                    continue
                fds.append(fd)
                if _FADV_WILLNEED is not None:
                    spans = self._layout(src)[0]
                    first, last = spans[0][0], spans[-1][0] + spans[-1][1]
                    try:
                        os.posix_fadvise(fd, first, last - first, _FADV_WILLNEED)
                    except OSError:
                        pass
        except BaseException:
            close_all(fds)
            raise
        return fds

    def read_fd_into(self, fd: int | None, piece) -> bool:
        """Copy one piece from a file descriptor returned by :meth:`open_hinted`."""
        if fd is None:
            return False
        self._read_fd_into(fd, *piece)
        return True

    def read_many_into(self, pieces) -> list[bool]:
        """``read_into`` for several ``(idx, src, dst)`` pieces from one thread, hinted first."""
        if not self.reads_ranges or len(pieces) == 1:
            return [self.read_into(*p) for p in pieces]
        fds = self.open_hinted(pieces)
        try:
            return [self.read_fd_into(fd, p) for fd, p in zip(fds, pieces)]
        finally:
            close_all(fds)

    def get(self, idx, *, mapped=False):
        path = self.path(idx)
        m = self.meta
        if m.is_raw:
            if mapped:
                try:
                    return np.memmap(path, dtype=m.dtype, mode="r", shape=m.chunks)
                except FileNotFoundError:
                    return None
            buf = read_file_into(path, m.chunk_nbytes)
            return None if buf is None else buf.view(m.dtype).reshape(m.chunks)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            return None
        return self._decoder.decode(data)


class HttpSource(ChunkSource):
    """Chunks over anonymous HTTP(S) GETs, using a shared urllib3 connection pool.

    At most ``connections`` requests are in flight at once, however many threads
    ask: on a congested link more parallel transfers are slower in total
    (measured to us-east-1: 16 at once 1.2 s, 32 at once 2.9 s for 64 MiB).
    ``timeout`` bounds the gap between received bytes, so a stalled
    connection is abandoned and retried quickly. A 404 is a missing chunk.
    Connection, read and 5xx errors are retried with exponential backoff; any
    other status raises.
    """

    def __init__(
        self,
        url: str,
        meta: ArrayMeta | None = None,
        *,
        connections: int = 16,
        retries: int = 5,
        timeout: float = 10.0,
        pool: Any = None,
    ):
        import urllib3

        self.url = url.rstrip("/")
        self.identity = self.url
        self.retries = retries
        self._slots = threading.BoundedSemaphore(connections)
        self._pool = pool or urllib3.PoolManager(
            num_pools=4,
            maxsize=connections,
            block=False,
            timeout=urllib3.Timeout(connect=10.0, read=timeout),
            retries=False,
            headers={"User-Agent": "vc_reader"},
        )
        if meta is None:
            zarray = self.fetch(".zarray")
            if zarray is None:
                raise FileNotFoundError(f"no zarr v2 array at {self.url} (.zarray not found)")
            meta = ArrayMeta.from_bytes(zarray)
        self.meta = meta
        self._decoder = ChunkDecoder(self.meta)

    def _request(self, key: str, sink):
        """GET ``url/key`` with retries; ``sink(response)`` consumes the body. None on 404."""
        import urllib3

        delay = 0.25
        for attempt in range(self.retries + 1):
            r = None
            clean = False
            try:
                with self._slots:
                    r = self._pool.request("GET", f"{self.url}/{key}", preload_content=False, decode_content=False)
                    if r.status == 404:
                        r.drain_conn()
                        clean = True
                        return None
                    if r.status == 200:
                        result = sink(r)
                        clean = True
                        return result
                body = r.read(512)
                if r.status < 500 and r.status != 429:
                    raise OSError(f"GET {self.url}/{key}: HTTP {r.status}: {body[:200]!r}")
                err: Exception = OSError(f"HTTP {r.status}")
            except (urllib3.exceptions.HTTPError, ConnectionError, TimeoutError) as exc:
                err = exc
            finally:
                if r is not None:
                    if not clean:
                        r.close()  # never hand a half-read connection back to the pool
                    r.release_conn()
            if attempt == self.retries:
                raise OSError(f"GET {self.url}/{key} failed after {attempt + 1} attempts: {err}") from err
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
        raise AssertionError("unreachable")

    def fetch(self, key: str) -> bytes | None:
        def sink(r):
            data = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                import gzip

                data = gzip.decompress(data)
            return data

        return self._request(key, sink)

    def get(self, idx, *, mapped=False):
        m = self.meta
        key = m.chunk_key(idx)
        if m.is_raw:
            nbytes = m.chunk_nbytes

            def sink(r):
                buf = np.empty(nbytes, dtype=np.uint8)
                view = memoryview(buf)
                got = 0
                while got < nbytes:
                    n = r.readinto(view[got:])
                    if not n:
                        break
                    got += n
                if got != nbytes or r.read(1):
                    import urllib3

                    raise urllib3.exceptions.ProtocolError(f"{self.url}/{key}: body is not {nbytes} bytes")
                return buf.view(m.dtype).reshape(m.chunks)

            return self._request(key, sink)
        data = self._request(key, lambda r: r.read())
        return None if data is None else self._decoder.decode(data)

    def get_with_payload(self, idx) -> tuple[np.ndarray | None, Any]:
        """The decoded chunk and the bytes the server stores for it (for verbatim mirrors)."""
        if self.meta.is_raw:
            arr = self.get(idx)
            return arr, None if arr is None else memoryview(arr.reshape(-1).view(np.uint8))
        data = self._request(self.meta.chunk_key(idx), lambda r: r.read())
        return (None, None) if data is None else (self._decoder.decode(data), data)

    def close(self) -> None:
        self._pool.clear()


def close_all(fds) -> None:
    for fd in fds:
        if fd is not None:
            os.close(fd)


def atomic_write(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def http_url(url: str) -> str:
    """https URL for an ``s3://`` URL (public buckets only), otherwise unchanged."""
    if not url.startswith("s3://"):
        return url
    from ..artifacts import to_https

    https = to_https(url)
    if https is None:
        bucket, _, key = url[len("s3://"):].partition("/")
        https = f"https://{bucket}.s3.amazonaws.com/{key}"
        warnings.warn(f"no known public endpoint for {url}; trying {https} anonymously", stacklevel=3)
    return https
