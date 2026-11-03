"""Metadata inspection and bounded local image/page access.

Decoders are imported inside their format-specific paths. TIFF files without
explicit spatial metadata require declared axes for their anonymous dimensions.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import math
import itertools
import os
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

    def __init__(self, location, format, selection, axes=None, files=None):
        from .encoded_files import EncodedFiles
        self.files = files if files is not None else EncodedFiles()
        self.location = location
        self.path = Path(urlsplit(location).path)
        self.bytes_read = 0
        self.format = format or {'.tif': 'tiff', '.tiff': 'tiff', '.png': 'png',
                                 '.jpg': 'jpeg', '.jpeg': 'jpeg'}.get(self.path.suffix.lower())
        if self.format not in ('tiff', 'png', 'jpeg'):
            raise ValueError(f'image format {self.format!r} is not supported')
        if selection.array_path is not None or selection.corresponding is not None:
            raise ValueError('image files select series/level, not Zarr array paths or implicit corresponding levels')
        self.palette = None
        self.storage_from_array = self.storage_units = None
        self.channel_names = self.time_names = ()
        with self.files.open(location) as (handle, token):
            self.token = token
            self.encoded_size = os.fstat(handle.fileno()).st_size
            if self.format == 'tiff':
                self.info = self._tiff_info(selection, axes, handle)
            else:
                if selection.series is not None or selection.level is not None:
                    raise ValueError('PNG/JPEG sources do not have selectable series or levels')
                self.info = self._pillow_info(axes, handle)
        self.prefix_shape = self.info.shape[:len(self.info.shape) - len(self.info.page_shape)]

    def _pillow_info(self, axes, handle):
        from PIL import Image, ImageMode

        with Image.open(handle) as image:
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
                                'values': tuple(image.palette.palette),
                                'transparency': image.info.get('transparency')}
            resolved = _axes(axes, observed, shape)
            return FileInfo(shape, resolved, dtype, shape, resolved)

    def _tiff_info(self, selection, axes, handle):
        import tifffile

        with tifffile.TiffFile(handle, name=self.path.name) as tif:
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
            self.level_count = len(series.levels)
            level = selection.level
            if level is None:
                if len(series.levels) > 1:
                    raise ValueError('pyramidal TIFF requires an explicit level selection')
                level = 0
            if level >= len(series.levels):
                raise ValueError(f'TIFF level {level} does not exist in series {index}')
            series = series.levels[level]
            first = series.pages[0]
            from ..affine import Affine
            unit = first.tags.get('ResolutionUnit') if first is not None else None
            multiplier = {2: 25400., 3: 10000.}.get(int(unit.value)) if unit else None
            if multiplier is not None and 'Z' not in series.axes:
                xres, yres = first.tags.get('XResolution'), first.tags.get('YResolution')
                if xres is not None and yres is not None:
                    x, y = xres.value, yres.value
                    if any(v <= 0 for v in (*x, *y)):
                        raise ValueError('TIFF physical resolution must be positive')
                    self.storage_from_array = Affine.from_rows(((multiplier*x[1]/x[0], 0, 0, 0),
                                                               (0, multiplier*y[1]/y[0], 0, 0), (0, 0, 1, 0)))
                    self.storage_units = 'micrometer'
            if tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                images = [node for node in root if node.tag.rsplit('}', 1)[-1] == 'Image']
                if index < len(images):
                    pixels = next((node for node in images[index] if node.tag.rsplit('}', 1)[-1] == 'Pixels'), None)
                    if pixels is not None:
                        self.channel_names = tuple(node.get('Name') for node in pixels if node.tag.rsplit('}', 1)[-1] == 'Channel')
                        from .source_resolution import _UNITS
                        spatial = 'XYZ' if 'Z' in series.axes else 'XY'
                        if level == 0 and all(pixels.get('PhysicalSize'+a) is not None for a in spatial):
                            spacing = []
                            for a in 'XYZ':
                                value = pixels.get('PhysicalSize'+a)
                                if value is None:
                                    spacing.append(1.)
                                else:
                                    units = pixels.get('PhysicalSize'+a+'Unit', 'µm')
                                    if units not in _UNITS:
                                        raise ValueError(f'unknown OME spatial unit {units}')
                                    spacing.append(float(value)*_UNITS[units])
                            if any(not math.isfinite(v) or v <= 0 for v in spacing):
                                raise ValueError('OME physical spacing must be finite and positive')
                            self.storage_from_array = Affine.from_rows(tuple(tuple(spacing[i] if i == j else 0 for j in range(3))+(0,) for i in range(3)))
                            self.storage_units = 'micrometer'

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
    def region(self, index, slices, cache, scratch_bytes):
        """Borrow only the requested portion; tile/strip decoding is selective."""
        with self.files.open(self.location) as (handle, token):
            if token != self.token:
                raise ValueError(f'{self.location}: source changed since opening; reopen the source')
            if self.format != 'tiff':
                from PIL import Image
                nbytes = math.prod(self.info.page_shape) * self.info.dtype.itemsize
                working = 8 * nbytes + self.encoded_size + (1 << 20)

                def load():
                    with Image.open(handle) as image:
                        result = np.array(image)
                    self.bytes_read += self.encoded_size
                    self._check_page(result)
                    return result

                with cache.use((self.token, 'pillow'), decode_bytes=working,
                               scratch_bytes=scratch_bytes, load=load) as array:
                    yield array[slices]
                return
            import tifffile
            with tifffile.TiffFile(handle, name=self.path.name) as tif:
                page = tif.series[self.info.series].levels[self.info.level].pages[index]
                if page is None:
                    raise ValueError(f'TIFF page {index} is missing')
                if tuple(page.shape) != self.info.page_shape or np.dtype(page.dtype) != self.info.dtype:
                    raise ValueError(f'TIFF page {index} shape/dtype differs from the selected series')
                if page.is_memmappable:
                    with cache.working(scratch_bytes):
                        array = page.asarray(out='memmap', maxworkers=1)
                        try:
                            region = array[slices]
                            self.bytes_read += region.nbytes
                            yield region
                        finally:
                            array._mmap.close()
                    return
                with self._segments(page, index, slices, cache, scratch_bytes) as region:
                    yield region

    @contextmanager
    def _segments(self, page, page_index, slices, cache, scratch_bytes):
        # tifffile's decoder coordinates are (separate sample, depth, Y, X,
        # contiguous sample). This mapping preserves planar/interleaved channels.
        layout = page.keyframe
        shape = layout.shaped
        dimensions = {'Z': 1, 'Y': 2, 'X': 3, 'S': 0 if layout.planarconfig == 2 else 4}
        bounds = [(0, n) for n in shape]
        result_shape = []
        for axis, selection, n in zip(layout.axes, slices, layout.shape):
            if isinstance(selection, int):
                bounds[dimensions[axis]] = (selection, selection+1)
            else:
                lo, hi, step = selection.indices(n)
                if step != 1:
                    raise ValueError('TIFF regions require unit strides')
                bounds[dimensions[axis]] = lo, hi
                result_shape.append(hi-lo)
        height = layout.tilelength if layout.is_tiled else layout.rowsperstrip
        width = layout.tilewidth if layout.is_tiled else layout.imagewidth
        depth = (layout.tiledepth or 1) if layout.is_tiled else 1
        nz, ny, nx = (math.ceil(shape[1]/depth), math.ceil(shape[2]/height), math.ceil(shape[3]/width))
        choices = [range(*bounds[0]), range(bounds[1][0]//depth, (bounds[1][1]-1)//depth+1),
                   range(bounds[2][0]//height, (bounds[2][1]-1)//height+1),
                   range(bounds[3][0]//width, (bounds[3][1]-1)//width+1)]
        normalized_shape = tuple(hi-lo for lo, hi in bounds)
        region_bytes = math.prod(normalized_shape) * self.info.dtype.itemsize
        with cache.working(region_bytes + scratch_bytes):
            output = np.zeros(normalized_shape, dtype=self.info.dtype)
            try:
                decode = page.keyframe.decode
            except (ValueError, KeyError) as error:
                raise ValueError(f'TIFF codec {layout.compression.name}: {error}; install the required optional decoder') from error
            kwargs = {'_fullsize': layout.is_tiled}
            if layout.compression in (6, 7, 34892, 33007):
                kwargs.update(jpegtables=page.jpegtables, jpegheader=page.keyframe.jpegheader)
            segment_bytes = depth * height * width * shape[4] * self.info.dtype.itemsize
            for sample, z, y, x in itertools.product(*choices):
                segment = ((sample*nz+z)*ny+y)*nx+x
                count, offset = page.databytecounts[segment], page.dataoffsets[segment]
                if not count or not offset:
                    raise ValueError(f'TIFF page {page_index} segment {segment} is missing')
                def load():
                    fh = page.parent.filehandle
                    fh.seek(offset)
                    encoded = fh.read(count)
                    self.bytes_read += len(encoded)
                    if len(encoded) != count:
                        raise ValueError('truncated TIFF segment')
                    array, position, extent = decode(encoded, segment, **kwargs)
                    if array is None:
                        raise ValueError('TIFF segment contains no decoded values')
                    return array
                key = (self.token, 'tiff', self.info.series, self.info.level, page_index, segment)
                with cache.use(key, decode_bytes=8*segment_bytes+count+(1 << 16), scratch_bytes=0, load=load) as array:
                    position = (sample, z*depth, y*height, x*width, 0)
                    tile = array.reshape((1, *array.shape))
                    source_slices, target_slices = [], []
                    for (lo, hi), at, n in zip(bounds, position, tile.shape):
                        a, b = max(lo, at), min(hi, at+n)
                        source_slices.append(slice(a-at, b-at))
                        target_slices.append(slice(a-lo, b-lo))
                    output[tuple(target_slices)] = tile[tuple(source_slices)]
            yield output.reshape(tuple(result_shape))

    def _check_page(self, array):
        if array.shape != self.info.page_shape or array.dtype != self.info.dtype:
            raise ValueError('decoded shape/dtype differs from image header')
