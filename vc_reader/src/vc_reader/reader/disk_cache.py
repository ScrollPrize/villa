"""Finding the on-disk chunk caches of a remote array, in the formats other tools already write.

The native reader reads and appends chunks; this module only finds caches,
checks that they hold the requested array, creates level metadata where an
append needs it, and holds VC3D's lease. Supported layouts:

* ``decoded`` (vc_reader, and vesuvius' ``RemoteChunkedArray``):
  ``<root>/s3/<bucket>/<path>/<level>/`` is a zarr v2 array of the decoded
  chunks, uncompressed or (vc_reader's ``compress_cache``) losslessly
  Delta3D-encoded, as its ``.zarray`` says. vc_reader marks missing chunks with
  ``<key>.missing``; vesuvius writes them as fill-valued chunks, which read the
  same.
* ``vc3d-mirror`` (VC3D ``ZarrMirror``): ``<root>/<name>-<fnv1a64(url)>/`` mirrors
  the source group. Chunks are the source's stored bytes and ``<key>.empty``
  marks a missing chunk.
* ``vc3d-delta3d`` (VC3D ``Delta3d``): the same directory with a
  ``.vc_delta3d_cache`` marker. Chunks are decoded voxels re-encoded with the
  lossless VC-Delta3D codec, and missing chunks are marked under
  ``.vc_cache_empty/<level>/<key>.empty``.

VC3D caches are used under VC3D's shared lease (``flock`` on
``<root>/.<id>.vc_cache.lock``): VC3D only replaces a cache directory when it
can take that lock exclusively. vc_reader does not update VC3D's byte
accounting; VC3D picks up the added files on its next cache scan.
"""

from __future__ import annotations

import configparser
import json
import os
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

import numpy as np

from .zarr_meta import ArrayMeta

DELTA3D_ID = "vc-delta3d"
VC3D_DELTA3D_MARKER = ".vc_delta3d_cache"
VC3D_DELTA3D_MARKER_CONTENTS = b"D3D1\n"
VC3D = "vc3d"  # cache_dir entry meaning "VC3D's configured remote cache root"


def atomic_write(path: str | os.PathLike[str], data: bytes) -> None:
    path = os.fspath(path)
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


@dataclass
class CacheLayer:
    """One cache directory holding chunks of one array, in one format."""

    kind: str  # "decoded", "vc3d-mirror" or "vc3d-delta3d"
    root: str  # directory of the level's chunk files
    stored: ArrayMeta  # how the layer stores chunks (its own .zarray)
    empty_root: str = ""  # vc3d-delta3d: where missing-chunk markers live
    allow_lossy: bool = False
    lease: Vc3dLease | None = field(default=None, repr=False)

    def native_spec(self) -> tuple[str, str, str, str, bool]:
        return (self.kind, self.root, self.stored.codec_id, self.empty_root, self.allow_lossy)

    def close(self) -> None:
        if self.lease is not None:
            self.lease.release()


def decoded_cache_path(cache_dir: str | os.PathLike[str], url: str) -> str:
    """``cache_dir/<scheme>/<host or bucket>/<path>``; known public buckets are keyed by their s3:// form.

    That is the layout of vesuvius' ``RemoteChunkedArray`` caches, so they are
    found and reused, and the https and s3 spellings of a URL share one cache.
    """
    from ..artifacts import to_s3

    u = urlsplit((to_s3(url) or url).rstrip("/"))
    return os.path.join(os.fspath(cache_dir), u.scheme, u.netloc, u.path.lstrip("/"))


def is_lossless_codec(config: dict | None) -> bool:
    """Whether a cache's compressor reproduces the decoded chunks exactly."""
    if not config or config.get("id") != DELTA3D_ID:
        return True
    return int(config.get("quant", 1) or 1) <= 1


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


def check_compressible(source: ArrayMeta) -> None:
    """Raise unless chunks of this array can be stored Delta3D-encoded (3-D, uint8 or uint16)."""
    if source.ndim != 3 or source.dtype.kind != "u" or source.dtype.itemsize not in (1, 2):
        raise ValueError(f"compress_cache needs 3-D uint8 or uint16 chunks; this array is "
                         f"{source.ndim}-D {source.dtype}")


def _decoded_layer(root: Path, source: ArrayMeta, url: str, create: bool, compress: bool = False,
                   allow_lossy: bool = False) -> CacheLayer | None:
    stored = _read_json(root / ".zarray")
    if stored is None:
        if not create:
            return None
        if compress:
            check_compressible(source)
        stored = delta3d_zarray(source) if compress else source.decoded_zarray()
        record = {"vc_reader": {"source_url": url, "source_zarray": dict(source.raw)}}
        atomic_write(root / ".zattrs", json.dumps(record, indent=1).encode())
        atomic_write(root / ".zarray", json.dumps(stored, indent=1).encode())
        return CacheLayer("decoded", str(root), ArrayMeta.from_zarray(stored))
    record = (_read_json(root / ".zattrs") or {}).get("vc_reader")
    if record is not None:  # ours: uncompressed or Delta3D, as its .zarray says
        comp = stored.get("compressor")
        ok = (record.get("source_zarray") == dict(source.raw) and not stored.get("filters")
              and (comp is None or comp.get("id") == DELTA3D_ID))
        if ok and not allow_lossy and not is_lossless_codec(comp):
            return None
    else:  # written by another tool (vesuvius): usable if it holds the same decoded array
        ok = (stored.get("compressor") is None and not stored.get("filters")
              and (stored.get("order") or "C") == "C" and _same_array(stored, source, ignore_codec=True))
    if not ok:
        raise ValueError(f"disk cache {root} holds a different array than {url} "
                         "(the remote .zarray changed?); delete that cache directory to refill it")
    return CacheLayer("decoded", str(root), ArrayMeta.from_zarray(stored), allow_lossy=allow_lossy)


def delta3d_zarray(source: ArrayMeta) -> dict:
    """``.zarray`` of the decoded chunks stored with lossless Delta3D.

    The same metadata VC3D writes for a level of its Delta3D caches
    (configureDelta3dV2Array).
    """
    meta = dict(source.raw, compressor={"id": DELTA3D_ID, "quant": 1}, filters=None)
    dt = str(meta.get("dtype", ""))
    if dt and dt[0] != "|":
        meta["dtype"] = ("<" if np.little_endian else ">") + dt[1:]
    return meta


def _vc3d_layer(volume_dir: Path, level_key: str, source: ArrayMeta, allow_lossy: bool,
                writable: bool) -> CacheLayer | None:
    lease = Vc3dLease(volume_dir)
    if not lease.acquire():
        warnings.warn(f"VC3D cache {volume_dir} is being replaced by another process (its lease is held "
                      "exclusively); not using it", stacklevel=4)
        return None
    try:
        try:
            delta3d = (volume_dir / VC3D_DELTA3D_MARKER).read_bytes() == VC3D_DELTA3D_MARKER_CONTENTS
        except FileNotFoundError:
            delta3d = False
        root = volume_dir / level_key
        stored = _read_json(root / ".zarray")
        if stored is None and writable:
            # VC3D publishes each level's metadata when it first touches the level; do the same.
            stored = delta3d_zarray(source) if delta3d else dict(source.raw)
            atomic_write(root / ".zarray", (json.dumps(stored, indent=2) + "\n").encode())
        if stored is None:
            lease.release()
            return None
        if delta3d:
            comp = stored.get("compressor") or {}
            if comp.get("id") != DELTA3D_ID or not _same_array(stored, source, ignore_codec=True):
                raise ValueError(f"VC3D Delta3D cache {root} does not hold {source.shape} {source.dtype} chunks")
            if not allow_lossy and not is_lossless_codec(comp):
                lease.release()
                return None
            empty_root = str(volume_dir / ".vc_cache_empty" / level_key)
            return CacheLayer("vc3d-delta3d", str(root), ArrayMeta.from_zarray(stored), empty_root,
                              allow_lossy, lease)
        if not _same_array(stored, source, ignore_codec=False):
            raise ValueError(f"VC3D cache {root} mirrors a different array (the remote .zarray changed?)")
        return CacheLayer("vc3d-mirror", str(root), ArrayMeta.from_zarray(stored), lease=lease)
    except BaseException:
        lease.release()
        raise


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
    writable: bool = True,
    allow_lossy: bool = False,
    compress: bool = False,
) -> tuple[list[CacheLayer], CacheLayer | None]:
    """Every existing cache of ``url`` under ``cache_dirs`` (in order), and the layer to append to.

    Appends go to the first layer that already holds the array, in its format.
    If none does and ``writable``, a new decoded cache is created under the
    first directory, Delta3D-encoded if ``compress``.
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
    try:
        for root in roots:
            if group_url is not None and level_key is not None:
                for spelling in _url_spellings(group_url):
                    vdir = root / vc3d_volume_id(spelling)
                    if vdir.is_dir():
                        layer = _vc3d_layer(vdir, level_key, source, allow_lossy, writable)
                        if layer is not None:
                            layers.append(layer)
                        break
            layer = _decoded_layer(Path(decoded_cache_path(root, url)), source, url, create=False,
                                   allow_lossy=allow_lossy)
            if layer is not None:
                layers.append(layer)
        if not writable:
            return layers, None
        if layers:
            return layers, layers[0]
        if not roots:
            return layers, None
        created = _decoded_layer(Path(decoded_cache_path(roots[0], url)), source, url, create=True,
                                 compress=compress)
        return [created], created
    except BaseException:
        for layer in layers:
            layer.close()
        raise


__all__: Sequence[Any] = ["CacheLayer", "open_layers", "decoded_cache_path", "vc3d_cache_root", "vc3d_volume_id",
                          "delta3d_zarray", "Vc3dLease"]
