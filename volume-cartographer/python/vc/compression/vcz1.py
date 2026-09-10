"""Compatibility adapter for the former native ``vc.compression.vcz1`` module."""

from vc_delta3d import (
    compress as _compress,
    compress_array as _compress_array,
    decompress_into_with_magic as _decompress_into_with_magic,
    decompress_with_magic as _decompress_with_magic,
)

_DELTA3D_MAGIC = b"D3D1"
_VCZ1_MAGIC = b"VCZ1"


def _with_magic(payload, source: bytes, target: bytes) -> bytes:
    data = payload if isinstance(payload, bytes) else bytes(memoryview(payload))
    if data[:4] != source:
        return data
    return target + data[4:]


def compress(raw, z, y, x, elem_size, quant=1, codec="rans"):
    if codec != "rans":
        raise ValueError("VCZ1 compatibility only supports rANS entropy coding")
    encoded = _compress(raw, z, y, x, elem_size, quant)
    return _with_magic(encoded, _DELTA3D_MAGIC, _VCZ1_MAGIC)


def compress_array(array, quant=1, codec="rans"):
    if codec != "rans":
        raise ValueError("VCZ1 compatibility only supports rANS entropy coding")
    encoded = _compress_array(array, quant)
    return _with_magic(encoded, _DELTA3D_MAGIC, _VCZ1_MAGIC)


def decompress(payload, expected_size):
    data = payload if isinstance(payload, bytes) else bytes(memoryview(payload))
    return _decompress_with_magic(data, expected_size, _VCZ1_MAGIC)


def decompress_into(payload, output):
    data = payload if isinstance(payload, bytes) else bytes(memoryview(payload))
    return _decompress_into_with_magic(data, output, _VCZ1_MAGIC)

__all__ = ["compress", "compress_array", "decompress", "decompress_into"]
