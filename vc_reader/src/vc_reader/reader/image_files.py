"""Metadata inspection and bounded local image/page access.

Decoders are imported inside their format-specific paths. TIFF files without
explicit spatial metadata require declared axes for their anonymous dimensions.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np


@dataclass(frozen=True)
class FileInfo:
    shape: tuple
    axes: tuple
    dtype: np.dtype
    page_shape: tuple
    page_axes: tuple
    series: int = 0
    level: int = 0


def local_path(location):
    if urlsplit(location).scheme in ('http', 'https', 's3'):
        raise NotImplementedError('remote image materialization is not implemented yet')
    return Path(location).expanduser().absolute()


def identity(path):
    stat = path.stat()
    return str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _axes(declared, observed, shape):
    if declared is not None:
        if len(declared) != len(shape):
            raise ValueError(f'declared axes {declared} do not match storage shape {shape}')
        for given, found in zip(declared, observed):
            if found is not None and given != found:
                raise ValueError(f'declared axes {declared} contradict storage axes {observed}')
        axes = declared
    else:
        if None in observed:
            raise ValueError(f'ambiguous storage axes {observed}; declare Source.axes explicitly')
        axes = observed
    if len(set(axes)) != len(axes) or not set(axes) <= {'x', 'y', 'z', 'c', 't'}:
        raise ValueError(f'unsupported or repeated storage axes {axes}')
    if not {'x', 'y'} <= set(axes):
        raise ValueError('image storage must have x and y axes')
    return tuple(axes)


class ImageFile:
    """Immutable file metadata; decoder handles exist only inside page reads."""

    def __init__(self, location, format, selection, axes=None):
        self.path = local_path(location)
        self.token = identity(self.path)
        self.format = format or {'.tif': 'tiff', '.tiff': 'tiff', '.png': 'png',
                                 '.jpg': 'jpeg', '.jpeg': 'jpeg'}.get(self.path.suffix.lower())
        if self.format not in ('tiff', 'png', 'jpeg'):
            raise NotImplementedError(f'local image format {self.format!r} is not supported')
        if selection.array_path is not None or selection.corresponding is not None:
            raise NotImplementedError('image array-path/corresponding-level resolution is not implemented yet')
        self.palette = None
        if self.format == 'tiff':
            self.info = self._tiff_info(selection, axes)
        else:
            if selection.series is not None or selection.level is not None:
                raise ValueError('PNG/JPEG sources do not have selectable series or levels')
            self.info = self._pillow_info(axes)
        self.prefix_shape = self.info.shape[:len(self.info.shape) - len(self.info.page_shape)]

    def check_identity(self):
        if identity(self.path) != self.token:
            raise ValueError(f'{self.path}: source changed since opening; reopen the source')

    def _pillow_info(self, axes):
        from PIL import Image, ImageMode

        with Image.open(self.path) as image:
            if image.format.lower() != self.format:
                raise ValueError(f'declared format {self.format} contradicts file format {image.format}')
            if getattr(image, 'n_frames', 1) != 1:
                raise ValueError('animated/multi-frame images require an explicit supported stack')
            bands = len(image.getbands())
            shape = (image.height, image.width) + ((bands,) if bands > 1 else ())
            observed = ('y', 'x') + (('c',) if bands > 1 else ())
            dtype = np.dtype(ImageMode.getmode(image.mode).typestr)
            if image.mode == 'P':
                self.palette = {'mode': image.palette.mode,
                                'values': tuple(image.getpalette()),
                                'transparency': image.info.get('transparency')}
            resolved = _axes(axes, observed, shape)
            return FileInfo(shape, resolved, dtype, shape, resolved)

    def _tiff_info(self, selection, axes):
        import tifffile

        with tifffile.TiffFile(self.path) as tif:
            index = selection.series
            if index is None:
                if len(tif.series) != 1:
                    raise ValueError(f'TIFF has {len(tif.series)} series; select series explicitly')
                index = 0
            if isinstance(index, str):
                candidates = [i for i, series in enumerate(tif.series) if series.name == index]
                if len(candidates) != 1:
                    raise ValueError(f'TIFF series name {index!r} is absent or ambiguous')
                index = candidates[0]
            if index >= len(tif.series):
                raise ValueError(f'TIFF series {index} does not exist')
            series = tif.series[index]
            level = selection.level
            if level is None:
                if len(series.levels) > 1:
                    raise ValueError('pyramidal TIFF requires an explicit level selection')
                level = 0
            if level >= len(series.levels):
                raise ValueError(f'TIFF level {level} does not exist in series {index}')
            series = series.levels[level]
            first = series.pages[0]
            if first is None:
                raise ValueError('TIFF first page is missing')
            shape, page_shape = tuple(series.shape), tuple(first.shape)
            observed = tuple({'X': 'x', 'Y': 'y', 'Z': 'z', 'T': 't',
                              'C': 'c', 'S': 'c'}.get(a) for a in series.axes)
            resolved = _axes(axes, observed, shape)
            prefix = len(shape) - len(page_shape)
            if prefix < 0 or shape[prefix:] != page_shape or math.prod(shape[:prefix]) != len(series.pages):
                raise ValueError('TIFF series/page layout cannot be read as independent pages; use a chunked source')
            return FileInfo(shape, resolved, np.dtype(series.dtype), page_shape,
                            resolved[prefix:], index, level)

    @contextmanager
    def page(self, index, cache, scratch_bytes):
        """Borrow a selected page. A mapped TIFF never enters the decoded LRU."""
        self.check_identity()
        if self.format != 'tiff':
            from PIL import Image

            nbytes = math.prod(self.info.page_shape) * self.info.dtype.itemsize
            # Pillow owns an internal image plus exported bytes/array. Include
            # packed-mode expansion, encoded input, row filters and codec state.
            working = 8 * nbytes + self.token[3] + (1 << 20)

            def load():
                with Image.open(self.path) as image:
                    result = np.array(image)
                self._check_page(result)
                self.check_identity()
                return result

            with cache.use((self.token, 'pillow'), decode_bytes=working,
                           scratch_bytes=scratch_bytes, load=load) as array:
                yield array
            return

        import tifffile

        with tifffile.TiffFile(self.path) as tif:
            page = tif.series[self.info.series].levels[self.info.level].pages[index]
            if page is None:
                raise ValueError(f'TIFF page {index} is missing')
            if tuple(page.shape) != self.info.page_shape or np.dtype(page.dtype) != self.info.dtype:
                raise ValueError(f'TIFF page {index} shape/dtype differs from the selected series')
            if page.is_memmappable:
                with cache.working(scratch_bytes):
                    array = page.asarray(out='memmap', maxworkers=1)
                    try:
                        yield array
                        self.check_identity()
                    finally:
                        array._mmap.close()
                return
            nbytes = math.prod(self.info.page_shape) * self.info.dtype.itemsize
            # Edge tiles may extend far beyond the image dimensions. Include
            # the full decoded segment rather than just the visible pixels.
            segment_pixels = (page.tilewidth * page.tilelength * (page.tiledepth or 1)
                              if page.is_tiled else page.imagewidth * page.rowsperstrip)
            segment_bytes = segment_pixels * page.samplesperpixel * self.info.dtype.itemsize
            working = 8 * (nbytes + segment_bytes) + sum(page.databytecounts) + (1 << 20)

            def load():
                try:
                    array = page.asarray(maxworkers=1, buffersize=min(working, 1 << 20))
                except (ValueError, KeyError) as error:
                    if 'imagecodecs' in str(error).lower() or 'codec' in str(error).lower():
                        raise ValueError(f'TIFF codec {page.compression.name}: {error}; '
                                         'the optional imagecodecs decoder may be required') from error
                    raise
                self._check_page(array)
                self.check_identity()
                return array

            key = (self.token, 'tiff', self.info.series, self.info.level, index)
            with cache.use(key, decode_bytes=working, scratch_bytes=scratch_bytes, load=load) as array:
                yield array

    def _check_page(self, array):
        if array.shape != self.info.page_shape or array.dtype != self.info.dtype:
            raise ValueError('decoded shape/dtype differs from image header')
