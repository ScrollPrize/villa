"""numcodecs compatibility wrapper for Volume Cartographer's VCZ1 codec."""

from __future__ import annotations

import numcodecs
import numpy as np

from . import vcz1


class Vcz1(numcodecs.abc.Codec):
    """Read and write chunks using the historical VCZ1 identifier."""

    codec_id = "vcz1"

    def __init__(self, codec: str = "rans", quant: int = 1):
        if codec != "rans":
            raise ValueError("VCZ1 compatibility only supports rANS entropy coding")
        if not 1 <= int(quant) <= 255:
            raise ValueError("quant must be in [1, 255]")
        self.quant = int(quant)

    def encode(self, buf):
        array = np.asarray(buf)
        if array.ndim != 3:
            raise ValueError("vcz1 expects 3D chunks")
        if array.dtype not in (np.uint8, np.uint16):
            raise ValueError("vcz1 supports uint8 and uint16 chunks")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return vcz1.compress_array(array, self.quant)

    def decode(self, buf, out=None):
        payload = buf if isinstance(buf, bytes) else bytes(memoryview(buf))
        z, y, x = _shape(payload)
        expected_size = z * y * x * payload[5]
        if out is not None:
            out_bytes = np.frombuffer(out, dtype=np.uint8)
            if out_bytes.size != expected_size:
                raise ValueError(
                    f"output buffer has {out_bytes.size} bytes, "
                    f"expected {expected_size}"
                )
            vcz1.decompress_into(payload, out_bytes)
            return out
        return vcz1.decompress(payload, expected_size)

    def get_config(self):
        return {"id": self.codec_id, "quant": self.quant}


def register() -> None:
    """Register VCZ1 in the active numcodecs process registry."""

    numcodecs.register_codec(Vcz1)


def _shape(payload) -> tuple[int, int, int]:
    if len(payload) < 20 or payload[:4] != b"VCZ1":
        raise ValueError("not a VCZ1 payload")
    return (
        int.from_bytes(payload[8:12], "little"),
        int.from_bytes(payload[12:16], "little"),
        int.from_bytes(payload[16:20], "little"),
    )


register()

__all__ = ["Vcz1", "register"]
