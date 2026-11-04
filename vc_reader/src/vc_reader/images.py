"""Plain image files: TIFF and PNG/JPEG reading, TIFF writing (``vcr::sources``).

``read`` is a whole file. ``open`` is a :class:`~vc_reader.reader.SourceReader`: indexing reads a window,
decoding only the strips and tiles it touches. Both take the first series and level, and name a
page stack's axes z, y, x. Bilevel files read by their TIFF photometric, so white is True on a
min-is-white file.
"""
import os

import numpy as np

from . import _reader
from ._reader import TiffWriter as _NativeWriter
from .reader.source_reader import SourceReader
from .reader.zarr_reader import _make_settings, _open_options

_NO_DECODE_LIMIT = 1 << 62


def _options_for(reader_options):
    def options_for(location):
        settings, _ = _make_settings(location, **reader_options)
        return _open_options(settings, None)
    return options_for


def read(location, *, page=None, **reader_options):
    """The whole file as an array, or one ``page`` of a page stack. Reader options as ``ZarrReader.open``."""
    return _reader._read_image(os.fspath(location), _options_for(reader_options), page)


def open(location, *, decoded_cache=None, decoded_cache_bytes=None, max_decode_bytes=None, **reader_options):  # noqa: A001
    """The file as a reader for windows, ``.shape``, ``.dtype``, ``.axes``. A window may be any size unless ``max_decode_bytes`` says otherwise."""
    if decoded_cache is None and max_decode_bytes is None:
        max_decode_bytes = _NO_DECODE_LIMIT
    source = _reader._image_source(os.fspath(location), _options_for(reader_options))
    return SourceReader(source, decoded_cache=decoded_cache, decoded_cache_bytes=decoded_cache_bytes, max_decode_bytes=max_decode_bytes, **reader_options)


def memmap(path, shape=None, dtype=None, *, mode='r'):
    """A local uncompressed single-page TIFF as a ``numpy.memmap``. ``ValueError`` when its values cannot be mapped.

    With ``shape`` and ``dtype``, first creates the file, all zeros, and maps it for writing.
    """
    path = os.fspath(path)
    if shape is not None:
        with TiffWriter(path, shape, dtype) as writer:
            writer.write_blank_page()
        mode = 'r+'
    offset, stored, typestr = _reader._tiff_mapping(path)
    return np.memmap(path, dtype=np.dtype(typestr), mode=mode, offset=offset, shape=stored)


def write_tiff(path, image, *, compression='none', tile=None, pages=None):
    """Writes ``(rows, cols)`` or ``(rows, cols, channels)``, or with ``pages`` a leading page axis. Atomic.

    Unless ``pages`` says, a third axis of up to 4 is channels, of more is pages, and a fourth means both.

    ``compression``: none, lzw, deflate, packbits, zstd. ``tile``: rows and columns, rounded down to
    a multiple of 16 and capped to the image; without it, strips of about 256 KiB. 3 or 4 channels are RGB(A), else min-is-black.
    """
    _reader._write_tiff(path, image, compression or 'none', tile, pages)


def TiffWriter(path, shape, dtype, *, compression='none', tile=None, pages=1, pixel_um=None):
    """A TIFF too large to hold, written by ``write_tiles(iterable)`` row by row, ``write_tile(row, col, tile)`` at ``writer.tile`` steps, or ``write_page``.

    ``shape`` is ``(rows, cols)`` or ``(rows, cols, channels)``. ``pixel_um`` is written as the resolution tags. Use as a context manager.
    """
    return _NativeWriter(path, np.dtype(dtype).str, *shape[:2], *(shape[2:] or (1,)), compression=compression or 'none', tile=tile, pages=pages, pixel_um=pixel_um)


__all__ = ['TiffWriter', 'memmap', 'open', 'read', 'write_tiff']
