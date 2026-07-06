"""numcodecs wrappers for Volume Cartographer compression codecs."""

from __future__ import annotations

import numpy as np
import numcodecs

from . import vcz1


class Vcz1(numcodecs.abc.Codec):
    """numcodecs wrapper for the C++ VCZ1 chunk codec."""

    codec_id = "vcz1"

    def __init__(self, codec: str = "rans", quant: int = 1):
        if codec not in {"rans", "zstd"}:
            raise ValueError("codec must be 'rans' or 'zstd'")
        if not 1 <= int(quant) <= 255:
            raise ValueError("quant must be in [1, 255]")
        self.codec = codec
        self.quant = int(quant)

    def encode(self, buf):
        a = np.asarray(buf)
        if a.ndim != 3:
            raise ValueError("vcz1 expects 3D chunks")
        if a.dtype not in (np.uint8, np.uint16):
            raise ValueError("vcz1 supports uint8 and uint16 chunks")
        if a.flags.c_contiguous:
            return vcz1.compress_array(a, self.quant, self.codec)
        a = np.ascontiguousarray(a)
        return vcz1.compress_array(a, self.quant, self.codec)

    def decode(self, buf, out=None):
        payload = buf if isinstance(buf, bytes) else bytes(memoryview(buf))
        z, y, x = _vcz1_shape(payload)
        elem_size = payload[5]
        expected_size = z * y * x * elem_size
        if out is not None:
            out_bytes = np.frombuffer(out, dtype=np.uint8)
            if out_bytes.size != expected_size:
                raise ValueError(
                    f"output buffer has {out_bytes.size} bytes, expected {expected_size}"
                )
            vcz1.decompress_into(payload, out_bytes)
            return out
        return vcz1.decompress(payload, expected_size)

    def get_config(self):
        return {"id": self.codec_id, "codec": self.codec, "quant": self.quant}


def register() -> None:
    """Register VCZ1 in the active numcodecs process registry."""

    numcodecs.register_codec(Vcz1)


def _vcz1_shape(payload) -> tuple[int, int, int]:
    if len(payload) < 20 or payload[:4] != b"VCZ1":
        raise ValueError("not a VCZ1 payload")
    return (
        int.from_bytes(payload[8:12], "little"),
        int.from_bytes(payload[12:16], "little"),
        int.from_bytes(payload[16:20], "little"),
    )
