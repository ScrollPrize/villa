"""Persistent on-disk chunk caches, in the formats other tools already write.

A remote array can be cached in any of these, and an existing cache in any of
them is read and appended to in its own format:

* ``decoded`` (vc_reader, and vesuvius' ``RemoteChunkedArray``):
  ``<root>/s3/<bucket>/<path>/<level>/`` is an uncompressed zarr v2 array of
  decoded chunks. vc_reader marks missing chunks with ``<key>.missing``;
  vesuvius writes them as fill-valued chunks, which read the same.
* ``vc3d-mirror`` (VC3D ``ZarrMirror``): ``<root>/<name>-<fnv1a64(url)>/`` mirrors
  the source group. Chunks are the source's stored bytes (any source codec,
  e.g. blosc or volcomp) and ``<key>.empty`` marks a missing chunk.
* ``vc3d-delta3d`` (VC3D ``Delta3d``): the same directory with a
  ``.vc_delta3d_cache`` marker. Chunks are decoded voxels re-encoded with the
  lossless VC-Delta3D codec (its ``.zarray`` says so), and missing chunks are
  marked under ``.vc_cache_empty/<level>/<key>.empty``.

VC3D caches are used under VC3D's shared lease (``flock`` on
``<root>/.<id>.vc_cache.lock``): VC3D only replaces a cache directory when it
can take that lock exclusively. vc_reader does not update VC3D's byte
accounting; VC3D picks up the added files on its next cache scan.
"""

from __future__ import annotations

import configparser
import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

import numpy as np

from .codecs import DELTA3D_ID, delta3d_payload_lossless, ensure_registered, is_lossless_codec
from .sources import ChunkSource, LocalSource, atomic_write
from .zarr_meta import ArrayMeta

#: Returned by layer lookups when the cache knows nothing about a chunk.
NOT_CACHED = object()
MISSING_SUFFIX = ".missing"
VC3D_EMPTY_SUFFIX = ".empty"
VC3D_DELTA3D_MARKER = ".vc_delta3d_cache"
VC3D_DELTA3D_MARKER_CONTENTS = b"D3D1\n"
VC3D = "vc3d"  # cache_dir entry meaning "VC3D's configured remote cache root"


# -- VC3D conventions ------------------------------------------------------------------


def fnv1a64(text: str) -> int:
    h = 14695981039346656037
    for b in text.encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def vc3d_volume_id(url: str) -> str:
    """VC3D's cache directory name for a remote volume: ``<last path part>-<fnv1a64 hex>``.

    Mirrors ``deriveRemoteVolumeId`` in volume-cartographer core/src/Volume.cpp.
    """
    url = url.rstrip("/")
    name = url.rsplit("/", 1)[-1] or "remote"
    return f"{name}-{fnv1a64(url):016x}"


def vc3d_cache_root() -> Path | None:
    """VC3D's remote cache root, resolved the way VC3D resolves it; None if it does not exist.

    ``[viewer] remote_cache_dir`` in ``$VC3D_CONFIG_DIR/VC3D.ini`` (default
    ``~/.VC3D/VC3D.ini``), else ``/volpkgs/remote_cache``,
    ``/ephemeral/remote_cache``, ``~/.VC3D/remote_cache``.
    """
    config_dir = Path(os.environ.get("VC3D_CONFIG_DIR") or Path.home() / ".VC3D")
    candidates: list[Path] = []
    ini = config_dir / "VC3D.ini"
    if ini.is_file():
        parser = configparser.ConfigParser(interpolation=None, strict=False)
        try:
            parser.read(ini, encoding="utf-8")
            configured = parser.get("viewer", "remote_cache_dir", fallback="").strip().strip('"')
        except (configparser.Error, UnicodeDecodeError):
            configured = ""
        if configured:
            candidates.append(Path(configured).expanduser())
    for base in (Path("/volpkgs"), Path("/ephemeral")):
        if base.is_dir():
            candidates.append(base / "remote_cache")
            break
    candidates.append(Path.home() / ".VC3D" / "remote_cache")
    return next((c for c in candidates if c.is_dir()), None)


class Vc3dLease:
    """VC3D's shared cache lease: ``flock(LOCK_SH)`` on ``<root>/.<id>.vc_cache.lock``."""

    def __init__(self, volume_dir: Path):
        self.path = volume_dir.parent / f".{volume_dir.name}.vc_cache.lock"
        self._fd: int | None = None

    def acquire(self) -> bool:
        try:
            import fcntl
        except ImportError:  # no flock (Windows): VC3D caches are not used
            return False
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)  # closing the descriptor drops the flock
            self._fd = None


# -- layers ----------------------------------------------------------------------------


class CacheLayer:
    """One cache directory holding chunks of one array, in one format."""

    kind = "?"

    def __init__(self, root: str, stored: ArrayMeta, read_mode: str):
        self.root = root
        self.local = LocalSource(root, stored, read_mode=read_mode)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.root!r})"

    def chunk_path(self, idx: tuple[int, ...]) -> str:
        return self.local.path(idx)

    def missing_path(self, idx: tuple[int, ...]) -> str:
        raise NotImplementedError

    @property
    def maps_files(self) -> bool:
        return self.local.maps_files

    def read_into(self, idx, src, dst) -> bool | None:
        """Copy a cached chunk's region: True if copied, False if known missing, None if not cached."""
        try:
            if self.local.read_into(idx, src, dst):
                return True
        except FileNotFoundError:  # evicted between open and read by another process
            return None
        return False if os.path.exists(self.missing_path(idx)) else None

    def get(self, idx, mapped: bool = False):
        """The decoded chunk, None if known missing, or ``NOT_CACHED``."""
        try:
            arr = self.local.get(idx, mapped=mapped)
        except FileNotFoundError:
            return NOT_CACHED
        if arr is not None:
            return arr
        return None if os.path.exists(self.missing_path(idx)) else NOT_CACHED

    def store(self, idx, arr: np.ndarray, payload: Any) -> None:
        raise NotImplementedError

    def store_missing(self, idx) -> None:
        atomic_write(self.missing_path(idx), b"")

    def close(self) -> None:
        pass


class DecodedLayer(CacheLayer):
    kind = "decoded"

    def missing_path(self, idx):
        return self.chunk_path(idx) + MISSING_SUFFIX

    def store(self, idx, arr, payload):
        atomic_write(self.chunk_path(idx), memoryview(np.ascontiguousarray(arr)).cast("B"))


class Vc3dMirrorLayer(CacheLayer):
    kind = "vc3d-mirror"

    def __init__(self, root, stored, read_mode, lease: Vc3dLease):
        super().__init__(root, stored, read_mode)
        self.lease = lease

    def missing_path(self, idx):
        return self.chunk_path(idx) + VC3D_EMPTY_SUFFIX

    def store(self, idx, arr, payload):
        atomic_write(self.chunk_path(idx), payload)  # the source's bytes, verbatim

    def close(self):
        self.lease.release()


class Vc3dDelta3dLayer(CacheLayer):
    kind = "vc3d-delta3d"

    def __init__(self, root, stored, read_mode, lease: Vc3dLease, level_key: str, allow_lossy: bool):
        super().__init__(root, stored, read_mode)
        self.lease = lease
        self.volume_dir = os.path.dirname(root)
        self.level_key = level_key
        self.allow_lossy = allow_lossy
        import vc_delta3d

        self._encoder = vc_delta3d.Delta3D(quant=1)

    def missing_path(self, idx):
        rel = os.path.join(self.level_key, self.local.meta.chunk_key(idx))
        return os.path.join(self.volume_dir, ".vc_cache_empty", rel + VC3D_EMPTY_SUFFIX)

    def _payload(self, idx):
        try:
            with open(self.chunk_path(idx), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def get(self, idx, mapped=False):
        payload = self._payload(idx)
        if payload is None:
            return None if os.path.exists(self.missing_path(idx)) else NOT_CACHED
        if not self.allow_lossy and not delta3d_payload_lossless(payload):
            return NOT_CACHED  # quantized: not the source values; refetch exact ones
        return self.local._decoder.decode(payload)

    def read_into(self, idx, src, dst):
        arr = self.get(idx)
        if arr is NOT_CACHED:
            return None
        if arr is None:
            return False
        dst[...] = arr[src]
        return True

    def store(self, idx, arr, payload):
        atomic_write(self.chunk_path(idx), self._encoder.encode(np.ascontiguousarray(arr)))

    def close(self):
        self.lease.release()


# -- discovery ---------------------------------------------------------------------------


def decoded_cache_path(cache_dir: str | os.PathLike[str], url: str) -> str:
    """``cache_dir/<scheme>/<host or bucket>/<path>``; known public buckets are keyed by their s3:// form.

    That is the layout of vesuvius' ``RemoteChunkedArray`` caches, so they are
    found and reused, and the https and s3 spellings of a URL share one cache.
    """
    from ..artifacts import to_s3

    u = urlsplit((to_s3(url) or url).rstrip("/"))
    return os.path.join(os.fspath(cache_dir), u.scheme, u.netloc, u.path.lstrip("/"))


def _same_array(stored: dict, source: ArrayMeta, *, ignore_codec: bool) -> bool:
    d = source.dtype
    try:
        same_dtype = np.dtype(stored.get("dtype")).newbyteorder("=") == d.newbyteorder("=")
    except TypeError:
        return False
    same = (
        same_dtype
        and list(stored.get("shape") or ()) == list(source.shape)
        and list(stored.get("chunks") or ()) == list(source.chunks)
        and (stored.get("dimension_separator") or ".") == source.separator
        and np.asarray(stored.get("fill_value") or 0, d) == np.asarray(source.fill_value, d)
    )
    if not same or ignore_codec:
        return same
    return (stored.get("compressor") == (dict(source.compressor) if source.compressor else None)
            and list(stored.get("filters") or ()) == [dict(f) for f in source.filters]
            and (stored.get("order") or "C") == source.order)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def _decoded_layer(root: Path, source: ArrayMeta, url: str, read_mode: str, create: bool) -> DecodedLayer | None:
    stored = _read_json(root / ".zarray")
    decoded = ArrayMeta.from_zarray(source.decoded_zarray())
    if stored is None:
        if not create:
            return None
        os.makedirs(root, exist_ok=True)
        record = {"vc_reader": {"source_url": url, "source_zarray": dict(source.raw)}}
        atomic_write(str(root / ".zattrs"), json.dumps(record, indent=1).encode())
        atomic_write(str(root / ".zarray"), json.dumps(source.decoded_zarray(), indent=1).encode())
        return DecodedLayer(str(root), decoded, read_mode)
    record = (_read_json(root / ".zattrs") or {}).get("vc_reader")
    if record is not None:
        ok = record.get("source_zarray") == dict(source.raw)
    else:  # written by another tool (vesuvius): usable if it holds the same decoded array
        ok = (stored.get("compressor") is None and not stored.get("filters")
              and (stored.get("order") or "C") == "C" and _same_array(stored, source, ignore_codec=True))
    if not ok:
        raise ValueError(f"disk cache {root} holds a different array than {url} "
                         "(the remote .zarray changed?); delete that cache directory to refill it")
    return DecodedLayer(str(root), decoded, read_mode)


def _vc3d_layer(volume_dir: Path, level_key: str, source: ArrayMeta, read_mode: str,
                allow_lossy: bool, writable: bool) -> CacheLayer | None:
    lease = Vc3dLease(volume_dir)
    if not lease.acquire():
        warnings.warn(f"VC3D cache {volume_dir} is being replaced by another process (its lease is held "
                      "exclusively); not using it", stacklevel=4)
        return None
    try:
        delta3d = (volume_dir / VC3D_DELTA3D_MARKER).read_bytes() == VC3D_DELTA3D_MARKER_CONTENTS
    except FileNotFoundError:
        delta3d = False
    root = volume_dir / level_key
    stored = _read_json(root / ".zarray")
    if stored is None and writable:
        # VC3D publishes each level's metadata when it first touches the level; do the same.
        stored = vc3d_delta3d_zarray(source) if delta3d else dict(source.raw)
        os.makedirs(root, exist_ok=True)
        atomic_write(str(root / ".zarray"), (json.dumps(stored, indent=2) + "\n").encode())
    if stored is None:
        lease.release()
        return None
    try:
        if delta3d:
            comp = stored.get("compressor") or {}
            if comp.get("id") != DELTA3D_ID or not _same_array(stored, source, ignore_codec=True):
                raise ValueError(f"VC3D Delta3D cache {root} does not hold {source.shape} {source.dtype} chunks")
            if not allow_lossy and not is_lossless_codec(comp):
                lease.release()
                return None
            ensure_registered(DELTA3D_ID)
            return Vc3dDelta3dLayer(str(root), ArrayMeta.from_zarray(stored), read_mode, lease, level_key, allow_lossy)
        if not _same_array(stored, source, ignore_codec=False):
            raise ValueError(f"VC3D cache {root} mirrors a different array (the remote .zarray changed?)")
        return Vc3dMirrorLayer(str(root), ArrayMeta.from_zarray(stored), read_mode, lease)
    except BaseException:
        lease.release()
        raise


def vc3d_delta3d_zarray(source: ArrayMeta) -> dict:
    """The ``.zarray`` VC3D writes for a level of a Delta3D cache (configureDelta3dV2Array)."""
    meta = dict(source.raw, compressor={"id": DELTA3D_ID, "quant": 1}, filters=None)
    dt = str(meta.get("dtype", ""))
    if dt and dt[0] != "|":
        meta["dtype"] = ("<" if np.little_endian else ">") + dt[1:]
    return meta


def _url_spellings(url: str) -> list[str]:
    from ..artifacts import to_https, to_s3

    out = []
    for u in (url, to_https(url), to_s3(url)):
        if u and u.rstrip("/") not in out:
            out.append(u.rstrip("/"))
    return out


def open_layers(
    cache_dirs: Sequence[str | os.PathLike[str]],
    url: str,
    source: ArrayMeta,
    *,
    group_url: str | None,
    level_key: str | None,
    read_mode: str = "slab",
    writable: bool = True,
    allow_lossy: bool = False,
) -> tuple[list[CacheLayer], CacheLayer | None]:
    """Every existing cache of ``url`` under ``cache_dirs`` (in order), and the layer to append to.

    Appends go to the first layer that already holds the array. If none does
    and ``writable``, a new decoded cache is created under the first directory.
    ``group_url``/``level_key`` (the OME-Zarr group and the level's path in it)
    are needed to find VC3D caches, which are per volume.
    """
    roots: list[Path] = []
    for d in cache_dirs:
        if os.fspath(d) == VC3D:
            r = vc3d_cache_root()
            if r is not None:
                roots.append(r)
        else:
            roots.append(Path(d).expanduser())
    layers: list[CacheLayer] = []
    for root in roots:
        if group_url is not None and level_key is not None:
            for spelling in _url_spellings(group_url):
                vdir = root / vc3d_volume_id(spelling)
                if vdir.is_dir():
                    layer = _vc3d_layer(vdir, level_key, source, read_mode, allow_lossy, writable)
                    if layer is not None:
                        layers.append(layer)
                    break
        layer = _decoded_layer(Path(decoded_cache_path(root, url)), source, url, read_mode, create=False)
        if layer is not None:
            layers.append(layer)
    if not writable:
        return layers, None
    if layers:
        return layers, layers[0]
    if not roots:
        return layers, None
    created = _decoded_layer(Path(decoded_cache_path(roots[0], url)), source, url, read_mode, create=True)
    return [created], created


# -- source --------------------------------------------------------------------------------


class DiskCachedSource(ChunkSource):
    """A remote source behind one or more disk cache layers (see the module docstring).

    Reads try each layer in order; a miss is fetched from the remote and stored
    in ``write_layer`` (in its format). Values served are exactly the source's
    stored values, unless ``allow_lossy_cache`` let a quantized cache answer.
    """

    def __init__(self, remote, layers: list[CacheLayer], write_layer: CacheLayer | None):
        self.remote = remote
        self.meta = remote.meta
        self.identity = remote.identity
        self.layers = layers
        self.write_layer = write_layer
        self.maps_files = bool(layers) and all(layer.maps_files for layer in layers)
        self._write_failed = False

    @classmethod
    def open(cls, remote, cache_dirs: Iterable[str | os.PathLike[str]], url: str, *, group_url=None,
             level_key=None, read_mode="slab", writable=True, allow_lossy=False) -> DiskCachedSource:
        layers, write_layer = open_layers(list(cache_dirs), url, remote.meta, group_url=group_url,
                                          level_key=level_key, read_mode=read_mode, writable=writable,
                                          allow_lossy=allow_lossy)
        return cls(remote, layers, write_layer)

    @property
    def root(self) -> str | None:
        """Directory of the layer misses are written to."""
        return self.write_layer.root if self.write_layer else None

    def __repr__(self) -> str:
        return f"DiskCachedSource({self.identity!r}, layers={self.layers})"

    def read_cached_into(self, idx, src, dst) -> bool | None:
        for layer in self.layers:
            hit = layer.read_into(idx, src, dst)
            if hit is not None:
                return hit
        return None

    def get(self, idx, *, mapped=False):
        for layer in self.layers:
            hit = layer.get(idx, mapped)
            if hit is not NOT_CACHED:
                return hit
        arr, payload = self.remote.get_with_payload(idx)
        self._store(idx, arr, payload)
        return arr

    def _store(self, idx, arr, payload) -> None:
        layer = self.write_layer
        if layer is None or self._write_failed:
            return
        try:
            if arr is None:
                layer.store_missing(idx)
            else:
                layer.store(idx, arr, payload)
        except OSError as exc:
            self._write_failed = True
            warnings.warn(f"disk cache {layer.root}: writes disabled after error: {exc}", stacklevel=3)

    def close(self) -> None:
        for layer in self.layers:
            layer.close()
        self.remote.close()
