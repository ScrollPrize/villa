"""The on-disk chunk caches of remote arrays, in the formats other tools already write.

The native reader finds caches, checks that they hold the requested array, creates level
metadata where an append needs it, holds VC3D's lease, reads and appends chunks
(``vc_reader/zarr.hpp``). Supported layouts:

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
can take that lock exclusively. The entry ``"vc3d"`` in ``cache_dir`` stands for
VC3D's configured remote cache root.
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import _reader

CacheLayer = _reader.CacheLayer
disk_budget = _reader.disk_budget
vc3d_volume_id = _reader.vc3d_volume_id


def vc3d_cache_root() -> Path | None:
    """VC3D's remote cache root, resolved the way VC3D resolves it; None if it does not exist.

    ``[viewer] remote_cache_dir`` in ``$VC3D_CONFIG_DIR/VC3D.ini`` (default
    ``~/.VC3D/VC3D.ini``), else ``/volpkgs/remote_cache``,
    ``/ephemeral/remote_cache``, ``~/.VC3D/remote_cache``.
    """
    root = _reader.vc3d_cache_root()
    return Path(root) if root is not None else None


def decoded_cache_path(cache_dir: str | os.PathLike[str], url: str) -> str:
    """``cache_dir/<scheme>/<host or bucket>/<path>``; known public buckets are keyed by their s3:// form.

    That is the layout of vesuvius' ``RemoteChunkedArray`` caches, so they are
    found and reused, and the https and s3 spellings of a URL share one cache.
    """
    return _reader.decoded_cache_path(os.fspath(cache_dir), url)


__all__ = ["CacheLayer", "decoded_cache_path", "disk_budget", "vc3d_cache_root", "vc3d_volume_id"]
