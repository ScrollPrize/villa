"""Surfaces stored as tifxyz directories (``vcr::tifxyz``).

``read`` returns a :class:`Surface`: ``xyz`` (rows, cols, 3; XYZ float32, -1 where a vertex is
missing) and ``valid``, read-only views of the stored grid. ``resample`` and ``points_at`` take
stored-grid positions and return NaN where no patch covers one; ``"smooth"`` is a bicubic
Hermite patch (Catmull-Rom on interior cells), ``"linear"`` a bilinear one.
"""
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

from . import _reader
from ._reader import TifxyzSurface as Surface


def _options_for(reader_options):
    from .reader.zarr_reader import _make_settings, _open_options

    def options_for(location):
        settings, _ = _make_settings(location, **reader_options)
        return _open_options(settings, None)
    return options_for


def read(location, *, mask=True, **reader_options):
    """The whole surface. ``mask=False`` ignores ``mask.tif``; reader options as ``ZarrReader.open``."""
    return _reader._tifxyz_read(os.fspath(location), mask, _options_for(reader_options))


def read_mask(location, shape, **reader_options):
    """``mask.tif`` as a bool array over a grid of ``shape``, or None without one."""
    return _reader._tifxyz_read_mask(os.fspath(location), shape[0], shape[1], _options_for(reader_options))


@dataclass(frozen=True)
class Metadata:
    uuid: str
    scale: tuple[float, float]  # stored cells per full-resolution pixel: rows, columns
    bbox: tuple[float, float, float, float, float, float] | None
    area: float | None
    extra: dict[str, Any] = field(default_factory=dict)


def read_metadata(location, **reader_options):
    return Metadata(*_reader._tifxyz_read_metadata(os.fspath(location), _options_for(reader_options)))


@dataclass(frozen=True)
class Entry:
    location: str
    metadata: Metadata


def list_surfaces(folder, *, recursive=True, z_range=None, **reader_options):
    """Tifxyz directories below ``folder``, by location. Those with unreadable metadata warn and are left out."""
    out = []
    for location, uuid, scale, bbox, area, extra, error in _reader._tifxyz_list(
            os.fspath(folder), recursive, z_range, _options_for(reader_options)):
        if error:
            warnings.warn(f'{location}: {error}', stacklevel=2)
        else:
            out.append(Entry(location, Metadata(uuid, scale, bbox, area, extra)))
    return out


def write(path, surface, *, compression='lzw', tile=1024, mask=True, overwrite=False, pixel_um=None):
    """Writes the directory; a surface without a UUID gets one, kept on ``surface``.

    ``pixel_um`` is written as the TIFFs' resolution tags (micrometres per pixel).
    """
    _reader._tifxyz_write(path, surface, compression, tile, mask, overwrite, pixel_um)


def full_extent(stored, scale):
    """A grid's extent at full resolution, as ``vc_render_tifxyz`` sizes its output."""
    return _reader._tifxyz_full_extent(stored, scale)


__all__ = ['Entry', 'Metadata', 'Surface', 'full_extent', 'list_surfaces', 'read', 'read_mask', 'read_metadata', 'write']
