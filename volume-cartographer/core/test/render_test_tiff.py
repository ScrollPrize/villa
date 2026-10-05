"""Standard-library TIFF writer shared by the vc_render_tifxyz CLI tests.

A tifxyz surface is three float32 TIFFs (x.tif, y.tif, z.tif). The renderer tests build their surfaces with this,
so they need no numpy or tifffile and run on any machine that can build the renderer.
"""

from pathlib import Path
import struct


def write_float_tiff(path, width, height, values):
    """A baseline single-strip float32 greyscale TIFF, which is what a tifxyz coordinate plane is."""
    data = struct.pack("<%df" % len(values), *values)
    entries = [(256, 3, 1, width), (257, 3, 1, height), (258, 3, 1, 32), (259, 3, 1, 1),
               (262, 3, 1, 1), (273, 4, 1, 0), (277, 3, 1, 1), (278, 3, 1, height),
               (279, 4, 1, len(data)), (339, 3, 1, 3)]          # 339 SampleFormat 3 = IEEE float
    ifd_offset = 8
    pixel_offset = ifd_offset + 2 + 12 * len(entries) + 4
    entries = [(t, ty, n, pixel_offset if t == 273 else v) for (t, ty, n, v) in entries]
    out = bytearray(struct.pack("<2sHI", b"II", 42, ifd_offset))
    out += struct.pack("<H", len(entries))
    for tag, typ, count, value in sorted(entries):
        # a value of 2 bytes or 4 bytes lives in the entry itself, left-justified
        raw = struct.pack("<I", value) if typ == 4 else struct.pack("<HH", value, 0)
        out += struct.pack("<HHI", tag, typ, count) + raw
    out += struct.pack("<I", 0)                                 # no second IFD
    out += data
    Path(path).write_bytes(bytes(out))
