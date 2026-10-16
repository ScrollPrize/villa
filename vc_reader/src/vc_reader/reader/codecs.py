"""Chunk codecs beyond numcodecs' built-ins: VC-Delta3D and volcomp.

* ``vc-delta3d`` (``{"id": "vc-delta3d", "quant": Q}``): Volume Cartographer's
  Delta3D codec (https://github.com/ScrollPrize/vc-delta3d), used for VC3D's
  compressed remote caches. Decoding needs the ``vc-delta3d`` package, 0.1.1 or
  newer so decoding releases the GIL (``pip install 'vc-delta3d>=0.1.1'``; part
  of the ``reader`` extra on Python 3.12+), which registers it with numcodecs.
  Quantization width 1 is lossless; wider bins are lossy.
* ``volcomp`` (``{"id": "volcomp", "q": Q}``): the 3-D DCT codec from
  https://github.com/SuperOptimizer/volume-compressor, used by VC3D for
  volcomp-compressed volumes. Decoding needs that repository's ``volcomp_zarr``
  Python binding, built with its ``libvolcomp``. ``q = 0`` is lossless; ``q > 0``
  is lossy.
"""

from __future__ import annotations

from typing import Any, Mapping

DELTA3D_ID = "vc-delta3d"
VOLCOMP_ID = "volcomp"
_DELTA3D_MAGIC = b"D3D1"


class CodecUnavailable(ImportError):
    """A chunk codec is named by the array's metadata but cannot be loaded."""


def ensure_registered(codec_id: str) -> None:
    """Make ``codec_id`` resolvable by ``numcodecs.get_codec`` (no-op for built-in codecs)."""
    import numcodecs

    if codec_id == DELTA3D_ID:
        try:
            import vc_delta3d  # noqa: F401  (registers the codec on import)
        except ImportError as exc:
            raise CodecUnavailable(
                "this array is compressed with vc-delta3d; install it with `pip install 'vc-delta3d>=0.1.1'` "
                "(Python 3.12+)"
            ) from exc
    elif codec_id == VOLCOMP_ID:
        if numcodecs.registry.codec_registry.get(VOLCOMP_ID) is None:
            numcodecs.register_codec(_volcomp_codec_class())


def _volcomp_codec_class():
    try:
        from volcomp_zarr import _lib
    except (ImportError, OSError) as exc:
        raise CodecUnavailable(
            "this array is compressed with volcomp; install the volcomp_zarr binding from "
            "https://github.com/SuperOptimizer/volume-compressor (python/README.md) with its libvolcomp"
        ) from exc
    import numcodecs.abc
    import numpy as np

    class Volcomp(numcodecs.abc.Codec):
        """Decode-only numcodecs adapter for zarr v2 ``{"id": "volcomp", "q": Q}``."""

        codec_id = VOLCOMP_ID

        def __init__(self, q: float = 8.0, **config: Any):
            self.q = float(q)
            self.config = config

        def encode(self, buf):
            return _lib.encode(np.ascontiguousarray(buf, dtype=np.uint8).tobytes(), self.q)

        def decode(self, buf, out=None):
            raw = bytes(memoryview(buf))
            if out is not None:
                _lib.decode(raw, np.frombuffer(out, dtype=np.uint8))
                return out
            dest = np.empty(_lib.CHUNK_VOXELS, dtype=np.uint8)
            _lib.decode(raw, dest)
            return dest

        def get_config(self):
            return {"id": self.codec_id, "q": self.q, **self.config}

    return Volcomp


def is_lossless_codec(config: Mapping[str, Any] | None) -> bool:
    """Whether a compressor config reproduces its input exactly (unknown codecs: assumed lossless)."""
    if not config:
        return True
    cid = config.get("id")
    if cid == DELTA3D_ID:
        return int(config.get("quant", 1) or 1) <= 1
    if cid == VOLCOMP_ID:
        return float(config.get("q", 0) or 0) == 0.0
    return True


def delta3d_payload_lossless(payload: bytes | memoryview) -> bool:
    """True if a Delta3D payload was written losslessly (its quantization bin width is 0 or 1)."""
    head = bytes(payload[:7])
    return head[:4] == _DELTA3D_MAGIC and head[6] <= 1
