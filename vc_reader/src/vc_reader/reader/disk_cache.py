"""Finding the on-disk chunk caches of a remote array, in the formats other tools already write.

The C++ reader finds caches, checks that they hold the requested array, creates
level metadata where an append needs it, holds VC3D's lease, reads and appends
chunks (``vc_reader/zarr.hpp``; native clients use the same code). Supported
layouts:

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
can take that lock exclusively.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Sequence

from .zarr_meta import ArrayMeta

VC3D = "vc3d"  # cache_dir entry meaning "VC3D's configured remote cache root"


def _native():
    from .. import _reader

    return _reader


def disk_budget(root: str, max_bytes: int | None):
    """One C++ eviction index per directory and process, shared by writers."""
    return _native().disk_budget(str(root), max_bytes)


def atomic_write(path: str | os.PathLike[str], data: bytes) -> None:
    _native().atomic_write(os.fspath(path), bytes(data))


def vc3d_volume_id(url: str) -> str:
    """VC3D's cache directory name for a remote volume: ``<last path part>-<fnv1a64 hex>``."""
    return _native().vc3d_volume_id(url)


def vc3d_cache_root() -> Path | None:
    """VC3D's remote cache root, resolved the way VC3D resolves it; None if it does not exist.

    ``[viewer] remote_cache_dir`` in ``$VC3D_CONFIG_DIR/VC3D.ini`` (default
    ``~/.VC3D/VC3D.ini``), else ``/volpkgs/remote_cache``,
    ``/ephemeral/remote_cache``, ``~/.VC3D/remote_cache``.
    """
    root = _native().vc3d_cache_root()
    return Path(root) if root is not None else None


def decoded_cache_path(cache_dir: str | os.PathLike[str], url: str) -> str:
    """``cache_dir/<scheme>/<host or bucket>/<path>``; known public buckets are keyed by their s3:// form.

    That is the layout of vesuvius' ``RemoteChunkedArray`` caches, so they are
    found and reused, and the https and s3 spellings of a URL share one cache.
    """
    return _native().decoded_cache_path(os.fspath(cache_dir), url)


def cache_roots(cache_dirs: Sequence[str | os.PathLike[str]]) -> list[Path]:
    """The cache directories, with ``"vc3d"`` resolved (and dropped if VC3D has no cache root)."""
    return [Path(d) for d in _native().cache_roots([os.fspath(d) for d in cache_dirs])]


def cached_source_zarray(
    cache_dirs: Sequence[str | os.PathLike[str]], url: str, *, group_url: str | None, level_key: str | None,
) -> dict | None:
    """The remote ``.zarray`` of ``url`` as the caches under ``cache_dirs`` record it, for opening offline."""
    found = _native().cached_source_zarray([os.fspath(d) for d in cache_dirs], url, group_url, level_key)
    return json.loads(found) if found is not None else None


def cached_node_dirs(cache_dirs: Sequence[str | os.PathLike[str]], url: str) -> list[Path]:
    """Every cache directory that mirrors the remote node (group or array) at ``url``, in search order."""
    return [Path(d) for d in _native().cached_node_dirs([os.fspath(d) for d in cache_dirs], url)]


class CacheLayer:
    """One cache directory holding chunks of one array, in one format."""

    def __init__(self, layers: Any, index: int):
        self._layers, self._index = layers, index
        kind, root, stored, empty_root, allow_lossy, budget_root = layers.layer(index)
        self.kind: str = kind  # "decoded", "vc3d-mirror" or "vc3d-delta3d"
        self.root: str = root  # directory of the level's chunk files
        self.stored: ArrayMeta = ArrayMeta.from_zarray(json.loads(stored))  # how the layer stores chunks
        self.empty_root: str = empty_root  # vc3d-delta3d: where missing-chunk markers live
        self.allow_lossy: bool = allow_lossy
        self.budget_root: str = budget_root  # only the selected decoded writer when a limit is configured

    def __repr__(self) -> str:
        return f"CacheLayer({self.kind!r}, {self.root!r})"

    def reader_spec(self) -> tuple[str, str, str, str, bool]:
        return (self.kind, self.root, self.stored.codec_id, self.empty_root, self.allow_lossy)

    def close(self) -> None:
        """Releases VC3D's lease on this layer's cache, if it holds one."""
        if self._layers is not None:
            self._layers.release(self._index)
            self._layers = None


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
    cache_max_bytes: int | None = None,
) -> tuple[list[CacheLayer], CacheLayer | None]:
    """Every existing cache of ``url`` under ``cache_dirs`` (in order), and the layer to append to.

    Appends go to the first layer that already holds the array, in its format.
    If none does and ``writable``, a new decoded cache is created under the
    first directory, Delta3D-encoded if ``compress``.
    ``group_url``/``level_key`` (the OME-Zarr group and the level's path in it)
    are needed to find VC3D caches, which are per volume.
    """
    native = _native().open_layers([os.fspath(d) for d in cache_dirs], url, json.dumps(dict(source.raw)),
                                   group_url, level_key, writable, allow_lossy, compress, cache_max_bytes)
    for busy in native.busy:
        warnings.warn(f"VC3D cache {busy} is being replaced by another process (its lease is held "
                      "exclusively); not using it", stacklevel=3)
    layers = [CacheLayer(native, i) for i in range(len(native))]
    return layers, layers[native.writer] if native.writer >= 0 else None


__all__: Sequence[Any] = ["CacheLayer", "open_layers", "decoded_cache_path", "vc3d_cache_root", "vc3d_volume_id",
                          "cache_roots", "cached_source_zarray", "cached_node_dirs"]
