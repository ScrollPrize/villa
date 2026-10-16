"""Regenerate LZW reference payloads with libtiff, independent of vc_reader.

Run ``python tests/make_lzw_fixtures.py`` with libtiff installed. The TIFF
container is temporary; only its raw, predictor-free LZW strip is retained.
"""
from __future__ import annotations

import ctypes as c
from ctypes.util import find_library
from pathlib import Path
from tempfile import TemporaryDirectory


def reference_bytes(kind: str) -> bytes:
    if kind == "repeat":
        return b"ABABABA" * 2048 + bytes(2048)
    state = 12345
    values = bytearray()
    for _ in range(32768):
        state = (1664525 * state + 1013904223) & 0xffffffff
        values.append(state >> 24)
    return bytes(values)


def main():
    lib = c.CDLL(find_library("tiff"))
    lib.TIFFOpen.argtypes, lib.TIFFOpen.restype = [c.c_char_p, c.c_char_p], c.c_void_p
    lib.TIFFSetField.argtypes, lib.TIFFSetField.restype = [c.c_void_p, c.c_uint32], c.c_int
    lib.TIFFClose.argtypes = [c.c_void_p]
    lib.TIFFWriteEncodedStrip.argtypes = [c.c_void_p, c.c_uint32, c.c_void_p, c.c_ssize_t]
    lib.TIFFWriteEncodedStrip.restype = c.c_ssize_t
    lib.TIFFReadRawStrip.argtypes = [c.c_void_p, c.c_uint32, c.c_void_p, c.c_ssize_t]
    lib.TIFFReadRawStrip.restype = c.c_ssize_t
    lib.TIFFRawStripSize.argtypes, lib.TIFFRawStripSize.restype = [c.c_void_p, c.c_uint32], c.c_ssize_t
    with TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "reference.tif").encode()
        for kind in ("repeat", "noise"):
            data = reference_bytes(kind)
            tiff = lib.TIFFOpen(path, b"w")
            assert tiff
            try:
                for tag, value in ((256, len(data)), (257, 1), (258, 8), (259, 5),
                                   (262, 1), (277, 1), (278, 1), (284, 1)):
                    assert lib.TIFFSetField(tiff, tag, c.c_uint32(value))
                source = c.create_string_buffer(data)
                assert lib.TIFFWriteEncodedStrip(tiff, 0, source, len(data)) == len(data)
            finally:
                lib.TIFFClose(tiff)
            tiff = lib.TIFFOpen(path, b"r")
            assert tiff
            try:
                size = lib.TIFFRawStripSize(tiff, 0)
                assert size > 0
                payload = c.create_string_buffer(size)
                assert lib.TIFFReadRawStrip(tiff, 0, payload, size) == size
                (Path(__file__).parent / "data" / f"lzw_{kind}.bin").write_bytes(payload.raw)
            finally:
                lib.TIFFClose(tiff)


if __name__ == "__main__":
    main()
