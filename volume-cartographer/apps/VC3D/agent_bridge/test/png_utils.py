"""Minimal, stdlib-only PNG decoder used to sanity-check screenshot.capture
output (verify it is a real, non-blank image) without depending on Pillow.

Supports the subset of PNG that Qt's QImage::save(..., "PNG") produces in
practice: 8-bit depth, non-interlaced, color types 0 (gray), 2 (RGB), 4
(gray+alpha), 6 (RGBA). Falls back gracefully (raises PngUnsupported) for
anything else so callers can fall back to a size-based heuristic.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngUnsupported(Exception):
    pass


@dataclass
class DecodedPng:
    width: int
    height: int
    channels: int
    pixels: np.ndarray  # shape (height, width, channels), dtype uint8


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png(data: bytes) -> DecodedPng:
    if data[:8] != PNG_SIGNATURE:
        raise PngUnsupported("not a PNG (bad signature)")

    pos = 8
    idat = bytearray()
    width = height = bit_depth = color_type = interlace = None

    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        cdata = data[pos + 8:pos + 8 + length]
        pos += 12 + length

        if ctype == b"IHDR":
            (width, height, bit_depth, color_type, _comp, _filt, interlace) = \
                struct.unpack(">IIBBBBB", cdata)
        elif ctype == b"IDAT":
            idat.extend(cdata)
        elif ctype == b"IEND":
            break

    if width is None:
        raise PngUnsupported("no IHDR chunk found")
    if bit_depth != 8:
        raise PngUnsupported(f"unsupported bit depth {bit_depth}")
    if interlace != 0:
        raise PngUnsupported("interlaced PNG not supported")

    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise PngUnsupported(f"unsupported color type {color_type}")
    channels = channels_by_type[color_type]

    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    expected_len = (stride + 1) * height
    if len(raw) < expected_len:
        raise PngUnsupported("decompressed data shorter than expected")

    out = np.zeros((height, width, channels), dtype=np.uint8)
    prev_row = np.zeros(stride, dtype=np.int16)
    offset = 0
    for y in range(height):
        filter_type = raw[offset]
        offset += 1
        row = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=offset).astype(np.int16)
        offset += stride

        if filter_type == 0:
            pass
        elif filter_type == 1:  # Sub
            for x in range(channels, stride):
                row[x] = (row[x] + row[x - channels]) & 0xFF
        elif filter_type == 2:  # Up
            row = (row + prev_row) & 0xFF
        elif filter_type == 3:  # Average
            for x in range(stride):
                a = row[x - channels] if x >= channels else 0
                b = prev_row[x]
                row[x] = (row[x] + ((a + b) // 2)) & 0xFF
        elif filter_type == 4:  # Paeth
            for x in range(stride):
                a = row[x - channels] if x >= channels else 0
                b = prev_row[x]
                c = prev_row[x - channels] if x >= channels else 0
                row[x] = (row[x] + _paeth(a, b, c)) & 0xFF
        else:
            raise PngUnsupported(f"unsupported filter type {filter_type}")

        out[y] = row.astype(np.uint8).reshape(width, channels)
        prev_row = row

    return DecodedPng(width=width, height=height, channels=channels, pixels=out)


def is_nontrivial_image(data: bytes, min_stddev: float = 1.0) -> tuple[bool, str]:
    """Best-effort check that PNG bytes decode to a real, non-blank image.

    Returns (ok, detail). Falls back to a conservative size-based heuristic
    if the PNG can't be decoded with our minimal decoder.
    """
    try:
        decoded = decode_png(data)
    except PngUnsupported as e:
        # Fallback: a truly blank/constant image compresses to a tiny file;
        # anything above a few KB for a real window grab is very unlikely to
        # be blank. Not as strong a check, but avoids a hard dependency.
        ok = len(data) > 4096
        return ok, f"png decode unsupported ({e}); size-heuristic only: {len(data)} bytes"

    stddev = float(decoded.pixels.astype(np.float64).std())
    nonzero_fraction = float(np.count_nonzero(decoded.pixels)) / decoded.pixels.size
    ok = stddev >= min_stddev and nonzero_fraction > 0.0
    detail = (f"{decoded.width}x{decoded.height}x{decoded.channels}, "
              f"stddev={stddev:.3f}, nonzero_fraction={nonzero_fraction:.4f}")
    return ok, detail
