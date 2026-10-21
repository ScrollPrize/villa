"""Storage integration: Zarr metadata, file transport, caches, registration."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import base64
import hashlib
import http.server
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import zlib
from unittest.mock import patch

import numpy as np
from PIL import Image
import tifffile

if os.environ.get('VC_READER_BUILD_DIR'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from extension_build import load_extensions
    load_extensions(Path(os.environ['VC_READER_BUILD_DIR']))

from vc_reader import (Affine, Companion, Encoding, Frame, LevelRequest, Plane, QueryContext,
                       Selection, Source, Storage, VolumeRegistration, CacheMissError)
from vc_reader.reader import S3Credentials, MemoryCache, ZarrReader
from vc_reader.reader.disk_cache import disk_budget, decoded_cache_path


def process_read(reader):
    try:
        return reader.read((0, 0), (3, 4))
    finally:
        reader.close()


def read_remote_source(source, options):
    with source.open(**options) as reader:
        return reader.read((0, 0), (5, 7))


def array(path, data, attrs=None, fill=0):
    path.mkdir(parents=True, exist_ok=True)
    (path / '.zarray').write_text(json.dumps(dict(zarr_format=2, shape=data.shape, chunks=data.shape,
        dtype=data.dtype.str, compressor=None, filters=None, fill_value=fill, order='C')))
    (path / '.zattrs').write_text(json.dumps(attrs or {}))
    (path / '.'.join('0' for _ in data.shape)).write_bytes(data.tobytes())


def pyramid(path, scales=(1, 2, 4), units=None):
    path.mkdir()
    (path / '.zgroup').write_text('{"zarr_format": 2}')
    axes = [dict(name=a, type='space', **({'unit': units} if units else {})) for a in 'zyx']
    datasets = []
    for i, scale in enumerate(scales):
        datasets.append(dict(path=str(i), coordinateTransformations=[dict(type='scale', scale=[scale]*3)]))
        array(path / str(i), np.full((3, 5, 7), i+1, dtype=np.uint16))
    (path / '.zattrs').write_text(json.dumps(dict(multiscales=[dict(name='main', axes=axes, datasets=datasets)])))


class Handler(http.server.SimpleHTTPRequestHandler):
    counts = None
    lock = None

    def log_message(self, *args):
        pass

    def do_GET(self):
        hook = getattr(self.server, 'request_hook', None)
        if hook is not None and hook(self):
            return
        ignore_range = self.path.startswith('/ignore-range/')
        bad_range = self.path.startswith('/bad-range/')
        with self.lock:
            self.counts[self.path] = self.counts.get(self.path, 0) + 1
            attempt = self.counts[self.path]
        if ignore_range:
            self.path = self.path[len('/ignore-range'):]
        if bad_range:
            self.path = self.path[len('/bad-range'):]
        if self.path.startswith('/denied/'):
            self.send_error(403)
            return
        if self.path.startswith('/retry/'):
            if attempt == 1:
                self.send_error(503)
                return
            self.path = self.path[len('/retry'):]
        if self.path.startswith('/truncated/'):
            self.path = self.path[len('/truncated'):]
            if attempt == 1:
                self.send_response(200)
                self.send_header('Content-Length', '10000')
                self.end_headers()
                self.wfile.write(b'incomplete')
                self.close_connection = True
                return
        if self.path.startswith('/basic/'):
            if self.headers.get('Authorization') != 'Basic ' + base64.b64encode(b'user:password').decode():
                self.send_error(401)
                return
            self.path = self.path[len('/basic'):]
        if self.path.startswith('/bucket/'):
            if not self.headers.get('Authorization', '').startswith('AWS4-HMAC-SHA256 '):
                self.send_error(403)
                return
            self.path = self.path[len('/bucket'):]
        path = Path(self.translate_path(self.path))
        if path.is_file():
            data = path.read_bytes()
            tag = '"' + hashlib.sha256(data).hexdigest() + '"'
            if self.headers.get('If-Match') not in (None, tag):
                self.send_error(412)
                return
            if self.headers.get('If-None-Match') == tag:
                self.send_response(304)
                self.send_header('ETag', tag)
                self.end_headers()
                return
            if self.headers.get('Range') and not ignore_range:
                first, last = self.headers['Range'].removeprefix('bytes=').split('-')
                start = int(first) if first else max(0, len(data)-int(last))
                end = min(len(data)-1, int(last)) if first and last else len(data)-1
                if start > end:
                    self.send_error(416)
                    return
                self.send_response(206)
                self.send_header('ETag', tag)
                self.send_header('Content-Range', f'bytes {start+int(bad_range)}-{end}/{len(data)}')
                self.send_header('Content-Length', str(end-start+1))
                self.end_headers()
                self.wfile.write(data[start:end+1])
                return
            self.send_response(200)
            self.send_header('ETag', tag)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()


class StorageTests(FixtureCase):
    def test_zarr_worker_reopen_and_budgeted_encoding(self):
        path = self.root/'worker.zarr'
        data = np.arange(35, dtype=np.uint16).reshape(5, 7)
        array(path, data, {'_ARRAY_DIMENSIONS': ['y', 'x']})
        source = Source('worker', Storage(str(path), 'zarr'))
        with source.open() as reader:
            reader.read((0, 0), (3, 4))
            for method in ('spawn', 'fork'):
                if method not in multiprocessing.get_all_start_methods():
                    continue
                with multiprocessing.get_context(method).Pool(2) as pool:
                    for result in pool.map(process_read, [reader, reader]):
                        np.testing.assert_array_equal(result, data[:3, :4])
            np.testing.assert_array_equal(reader.read_many([((0, 0), (3, 4))])[0], data[:3, :4])
        encoded = replace(source, encoding=Encoding('integer', ((0, 0),), 'uint8', fallback=1))
        with encoded.open(max_decode_bytes=100) as reader:
            with self.assertRaisesRegex(MemoryError, 'max_decode_bytes'):
                reader.read((0, 0), (5, 7))

    def test_palette_open_does_not_decode_and_large_tiff_small_crop(self):
        path = self.root/'palette.png'
        image = Image.fromarray(np.zeros((20, 31), 'u1')).convert('P')
        image.save(path)
        with patch('PIL.PngImagePlugin.PngImageFile.load', side_effect=AssertionError('pixel decode')):
            with Source('palette', Storage(str(path), 'png')).open() as reader:
                self.assertIsNotNone(reader.palette)
        path = self.root/'large.tif'
        data = np.ones((2048, 3072), dtype=np.uint16)
        tifffile.imwrite(path, data, tile=(32, 32), compression='deflate')
        with Source('large', Storage(str(path), 'tiff')).open(max_decode_bytes=100_000) as reader:
            np.testing.assert_array_equal(reader.read((17, 39), (2, 3)), 1)
            self.assertLess(reader.stats()['image_payload_bytes'], 1024)
            self.assertLessEqual(reader.decoded_cache.stats()['peak_working_bytes'], 100_000)

    def test_tiff_physical_pyramid_correspondence_and_incomplete_ome(self):
        path = self.root/'pyramid.tif'
        with tifffile.TiffWriter(path) as tif:
            tif.write(np.ones((64, 80), 'u1'), subifds=1, resolution=(10000, 10000),
                      resolutionunit='CENTIMETER', metadata={'axes': 'YX'})
            tif.write(np.full((32, 40), 2, 'u1'), subfiletype=1,
                      resolution=(5000, 5000), resolutionunit='CENTIMETER')
        reg = VolumeRegistration('s', 'v', affine=Affine.identity())
        source = Source('physical', Storage(str(path), 'tiff', selection=Selection(level=0)), registration=reg)
        with source.with_selection(level=1).open() as reader:
            self.assertEqual(reader.registration.affine.linear, ((2., 0., 0.), (0., 2., 0.), (0., 0., 1.)))
            np.testing.assert_array_equal(reader.read((0, 0), (2, 3)), 2)
        with source.with_selection(corresponding=LevelRequest(Frame('s', 'v', 1))).open() as reader:
            self.assertEqual(reader.resolved.selection.level, 1)
        with self.assertRaisesRegex(ValueError, 'exact TIFF levels'):
            source.with_selection(corresponding=LevelRequest(Frame('s', 'v', 2))).open()
        path = self.root/'incomplete.ome.tif'
        tifffile.imwrite(path, np.zeros((2, 5, 7), 'u1'), photometric='minisblack',
                          metadata={'axes': 'ZYX', 'PhysicalSizeX': 2., 'PhysicalSizeY': 2.})
        with Source('ome', Storage(str(path), 'tiff')).open() as reader:
            self.assertIsNone(reader.resolved.storage_from_array)

    def test_zarr_selection_encoding_missing_chunks_and_padding(self):
        path = self.root / 'image.zarr'
        data = np.arange(2*3*4*5, dtype=np.uint64).reshape(2, 3, 4, 5) + 2**40
        array(path, data, {'_ARRAY_DIMENSIONS': ['t', 'c', 'y', 'x'],
                           'channel_names': ['red', 'green', 'blue'], 'time_names': ['a', 'b']})
        source = Source('zarr', Storage(str(path), 'zarr'))
        with self.assertRaisesRegex(ValueError, 'time selection'):
            source.open()
        selected = source.with_selection(channels=('blue', 'red'), time='b')
        with selected.open() as reader:
            self.assertEqual(reader.shape, (2, 4, 5))
            np.testing.assert_array_equal(reader.read((1, 2), (3, 3)), data[1, [2, 0], 1:, 2:])
        labels = self.root / 'labels.zarr'
        array(labels, np.ones((3, 4), dtype=np.uint8), {'_ARRAY_DIMENSIONS': ['y', 'x']}, fill=7)
        (labels / '0.0').unlink()
        source = Source('labels', Storage(str(labels), 'zarr'), fill_value=2**40+3,
                        encoding=Encoding('integer', ((7, 2**40+1),), 'uint64'))
        with source.open() as reader:
            out = reader.read((-1, -1), (3, 4))
            self.assertEqual(out.dtype, np.dtype('uint64'))
            self.assertEqual(out[0, 0], 2**40+3)
            self.assertEqual(out[1, 1], 2**40+1)
        with replace(source, fill_value=None).open() as reader:
            self.assertEqual(reader.fill_value, 2**40+1)

    def test_zarr_time_selection_without_channels(self):
        for axes in ('tyx', 'tzyx', 'zyxt'):
            spatial_shape = (3, 4) if axes == 'tyx' else (2, 3, 4)
            values = np.arange(math.prod(spatial_shape), dtype=np.uint16).reshape(spatial_shape)
            for count in (1, 2):
                with self.subTest(axes=axes, times=count):
                    data = np.stack([values + 100*i for i in range(count)], axis=axes.index('t'))
                    path = self.root / f'{axes}-{count}.zarr'
                    array(path, data, {'_ARRAY_DIMENSIONS': list(axes)})
                    source = Source('time', Storage(str(path), 'zarr'))
                    if count > 1:
                        source = source.with_selection(time=1)
                    expected = values + 100*(count-1)
                    start = (0,) * len(spatial_shape)
                    with source.open() as reader:
                        np.testing.assert_array_equal(reader.read(start, spatial_shape), expected)
                        for result in reader.read_many([(start, spatial_shape)] * 2):
                            np.testing.assert_array_equal(result, expected)
                        np.testing.assert_array_equal(reader[...], expected)
                        np.testing.assert_array_equal(reader[-1, ..., 1:], expected[-1, ..., 1:])

    def test_pyramid_selection_resolution_and_metadata_override(self):
        path = self.root / 'pyramid.zarr'
        pyramid(path, scales=(4, 8, 16), units='micrometer')
        reg = VolumeRegistration('s', 'v', affine=Affine.scale(128), origin='catalog-fallback',
                                 declared_grid=Selection(level=0), volume_spacing_um=(2.,)*3)
        source = Source('pyramid', Storage(str(path), 'zarr', pyramid=True, grid_level=7), registration=reg)
        with source.with_selection(level=1).open() as reader:
            self.assertEqual(reader.registration.affine, Affine.scale(4))
        request = source.with_selection(corresponding=LevelRequest(Frame('s', 'v', 2), -5))
        with request.open() as reader:
            self.assertEqual(reader.resolved.selection.array_path, '1')
            self.assertEqual(reader.registration.affine, Affine.scale(4))
        with self.assertRaisesRegex(ValueError, 'reference level 0'):
            source.with_selection(corresponding=LevelRequest(Frame('s', 'v', 0), -7)).open()
        with self.assertRaisesRegex(ValueError, 'source level 8'):
            source.with_selection(level=8).open()
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            source.open()
        unknown = replace(source, registration=replace(reg, volume_spacing_um=None))
        with self.assertRaisesRegex(ValueError, 'volume_spacing_um'):
            unknown.with_selection(level=0).open()

    def test_explicit_crop_origin_changed_grid_units_and_no_double_scale(self):
        path = self.root / 'pyramid.zarr'
        pyramid(path, scales=(1000, 2000, 4000), units='nanometer')
        affine = Affine.from_rows(((2, 0, 0, 2**26+.25), (0, 2, 0, 80), (0, 0, 2, 40)))
        source = Source('crop', Storage(str(path), 'zarr', selection=Selection(level=0)),
                        registration=VolumeRegistration('s', 'v', affine=affine))
        with source.open() as reader:
            self.assertEqual(reader.registration.affine, affine)
        with source.with_selection(level=1).open() as reader:
            self.assertEqual(reader.registration.affine, affine @ Affine.scale(2))
            self.assertEqual(reader.registration.affine.translation[0], 2**26+.25)
        self.assertNotEqual(np.float32(2**26+.25).item(), 2**26+.25)
        placed = replace(source, registration=VolumeRegistration('s', 'v', affine=Affine.scale(3),
                          mode='storage', storage_frame='physical', storage_units='micrometer'))
        with placed.with_selection(level=1).open() as reader:
            self.assertEqual(reader.registration.affine, Affine.scale(6))
        with self.assertRaisesRegex(ValueError, 'storage_units'):
            replace(placed, registration=replace(placed.registration, storage_units=None)).open()
        attrs = json.loads((path / '.zattrs').read_text())
        attrs['multiscales'][0]['axes'][0]['unit'] = 'unknown'
        (path / '.zattrs').write_text(json.dumps(attrs))
        with self.assertRaisesRegex(ValueError, 'unknown'):
            source.open()

    def test_mapped_correspondence_and_conflicting_axes_transforms(self):
        path = self.root / 'pyramid.zarr'
        pyramid(path)
        source = Source('mapped', Storage(str(path), 'zarr', selection=Selection(level=0)),
                        registration=VolumeRegistration('s', 'a', affine=Affine.identity()),
                        query=QueryContext(Frame('s', 'b', 2), 'transform', Affine.scale(2)))
        with source.with_selection(corresponding=LevelRequest(Frame('s', 'b', 2))).open() as reader:
            self.assertEqual(reader.resolved.selection.array_path, '1')
        (path / '0' / '.zattrs').write_text(json.dumps({'_ARRAY_DIMENSIONS': ['x', 'y', 'z']}))
        with self.assertRaisesRegex(ValueError, 'axes'):
            source.open()
        (path / '0' / '.zattrs').write_text(json.dumps({'coordinateTransformations': [{'type': 'scale', 'scale': [9, 9, 9]}]}))
        with self.assertRaisesRegex(ValueError, 'conflicting transformations'):
            source.open()

    def test_companions_and_raw_unregistered_arrays(self):
        parent, child = self.root / 'parent.zarr', self.root / 'child.zarr'
        attrs = {'_ARRAY_DIMENSIONS': ['y', 'x']}
        array(parent, np.ones((3, 5), 'u1'), attrs)
        array(child, np.full((3, 5), 2, 'u1'), attrs)
        mask = Source('supervision', Storage(str(child), 'zarr'), role='label')
        source = Source('parent', Storage(str(parent), 'zarr'), companions=(Companion('supervision', mask, True),))
        with source.open() as reader:
            self.assertIsNone(reader.registration)
            np.testing.assert_array_equal(reader.companions['supervision'].read((0, 0), (3, 5)), 2)
            output = reader.read((0, 0), (3, 5))
        np.testing.assert_array_equal(output, 1)
        array(child, np.zeros((4, 5), 'u1'), attrs)
        with self.assertRaisesRegex(ValueError, 'same_grid shape'):
            source.open()

    def test_tiff_tiles_strips_and_separate_channel_encoding(self):
        for kwargs in ({'tile': (16, 16)}, {'rowsperstrip': 5}):
            for data, axes, options in ((np.arange(33*51, dtype=np.uint16).reshape(33, 51), 'YX', {}),
                    (np.full((33, 51, 3), (3, 7, 11), dtype=np.uint8), 'YXS', {'photometric': 'rgb'}),
                    (np.broadcast_to(np.array([3, 7, 11], 'u1')[:, None, None], (3, 33, 51)), 'SYX',
                     {'photometric': 'rgb', 'planarconfig': 'separate'})):
                with self.subTest(kwargs=kwargs, axes=axes):
                    path = self.root / 'tiled.tif'
                    tifffile.imwrite(path, data, compression='deflate', metadata={'axes': axes}, **kwargs, **options)
                    source = Source('tiff', Storage(str(path), 'tiff'))
                    if options:
                        source = replace(source, encoding=Encoding('color', (((3, 7, 11), 2**40+1),), 'uint64'))
                    with source.open() as reader:
                        output = reader.read((14, 13), (7, 8))
                        np.testing.assert_array_equal(output, 2**40+1 if options else data[14:21, 13:21])
                        count = reader.stats()['image_payload_bytes']
                        reader.read((14, 13), (7, 8))
                        self.assertEqual(reader.stats()['image_payload_bytes'], count)
        # Three independent grayscale pages carrying explicitly declared color components.
        data = np.broadcast_to(np.array([3, 7, 11], 'u1')[:, None, None], (3, 33, 51))
        path = self.root / 'pages.tif'
        tifffile.imwrite(path, data, metadata={'axes': 'CYX'}, photometric='minisblack', compression='deflate')
        source = Source('pages', Storage(str(path), 'tiff'), encoding=Encoding('color', (((3, 7, 11), 2**40),), 'uint64'))
        with source.open() as reader:
            np.testing.assert_array_equal(reader.read((7, 9), (5, 7)), 2**40)
        stack = replace(source, storage=Storage(format='stack', locations=(str(path), str(path))),
                        axes=('z', 'c', 'y', 'x'))
        with stack.open() as reader:
            np.testing.assert_array_equal(reader.read((0, 7, 9), (2, 5, 7)), 2**40)


class RemoteFixtureCase(FixtureCase):
    def setUp(self):
        super().setUp()
        self.counts = {}
        def handler(*args, **kwargs):
            cls = type('TestHandler', (Handler,), {'counts': self.counts, 'lock': threading.Lock()})
            return cls(*args, directory=str(self.root), **kwargs)
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.cache = self.root / 'cache'
        self.data = np.arange(35, dtype=np.uint8).reshape(5, 7)
        Image.fromarray(self.data).save(self.root / 'image.png')

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        super().tearDown()


class RemoteTests(RemoteFixtureCase):
    def test_offline_s3_zarr_skips_auth_and_region_discovery(self):
        from vc_reader.reader import ZarrGroup
        from vc_reader.reader.auth import Access
        from vc_reader.reader.transport import cache_url
        group = self.root/'offline-group'
        group.mkdir()
        (group/'.zgroup').write_text('{"zarr_format": 2}')
        (group/'.zattrs').write_text(json.dumps(dict(multiscales=[dict(datasets=[dict(path='0')])])))
        values = np.arange(12, dtype=np.uint8).reshape(3, 4)
        array(group/'0', values)
        url = 's3://bucket/offline-group'
        keys = S3Credentials('example', 'test-secret', region='us-east-1', endpoint_url=self.url)
        with ZarrGroup.open(url, cache_dir=self.cache, auth=keys) as reader:
            np.testing.assert_array_equal(reader.level(0)[:], values)
        before = dict(self.counts)
        with patch('vc_reader.reader.auth.bucket_region', side_effect=AssertionError('region discovery')), \
             patch('vc_reader.reader.auth._AwsSource', side_effect=AssertionError('credential discovery')), \
             patch('vc_reader._reader.http_get', side_effect=AssertionError('metadata fetch')):
            for auth in ('auto', 'aws', None, S3Credentials('example', 'test-secret')):
                with self.subTest(auth=type(auth).__name__):
                    with ZarrReader.open(url+'/0', cache_dir=self.cache, offline=True, auth=auth) as reader:
                        np.testing.assert_array_equal(reader[:], values)
                    with ZarrGroup.open(url, cache_dir=self.cache, offline=True, auth=auth) as reader:
                        np.testing.assert_array_equal(reader.level(0)[:], values)
        self.assertEqual(before, self.counts)
        self.assertEqual(cache_url(url, Access('anonymous')), cache_url(url, Access('s3')))

    def test_shared_memory_cache_separates_lossy_and_exact_reads(self):
        data = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
        array(self.root/'volume.zarr', data)
        url = self.url+'/volume.zarr'
        metadata = json.loads((self.root/'volume.zarr/.zarray').read_text())
        lossy = Path(decoded_cache_path(self.cache, url))
        lossy.mkdir(parents=True)
        (lossy/'.zarray').write_text(json.dumps(dict(metadata, compressor={'id': 'vc-delta3d', 'quant': 8})))
        (lossy/'.zattrs').write_text(json.dumps({'vc_reader': {'source_url': url, 'source_zarray': metadata}}))
        # VC-Delta3D quant=8 payload for arange(64).reshape(4,4,4), compressed
        # here with zlib to keep the fixture independent of Python codec extras.
        (lossy/'0.0.0').write_bytes(zlib.decompress(base64.b64decode(
            'eJxzMXYxZGTkmMjCwMAAww48DCiggYG+fAeGUTCQYJ55hSUriMGoaMOChZ4/hxHIxi3PsNbye8Pe6v8A9AcPSA==')))
        memory = MemoryCache(4096)
        with ZarrReader.open(url, cache_dir=self.cache, allow_lossy_cache=True,
                             memory_cache=memory, cache_writes=False) as reader:
            quantized = reader[:]
            self.assertTrue(np.any(quantized != data))
        exact_cache = self.root/'exact'
        with ZarrReader.open(url, cache_dir=exact_cache, memory_cache=memory) as reader:
            np.testing.assert_array_equal(reader[:], data)
        # An exact cache hit stays available to another exact reader offline.
        with ZarrReader.open(url, cache_dir=exact_cache, memory_cache=memory, offline=True) as reader:
            np.testing.assert_array_equal(reader[:], data)
            self.assertEqual(reader.stats()['memory_hits'], 1)
        with ZarrReader.open(url, cache_dir=self.cache, memory_cache=memory,
                             allow_lossy_cache=True, offline=True) as reader:
            np.testing.assert_array_equal(reader[:], quantized)

    def test_unrelated_downloads_overlap_with_a_shared_cache(self):
        for processes in (False, True):
            with self.subTest(processes=processes):
                barrier = threading.Barrier(4, timeout=10)
                def hook(handler):
                    if not handler.path.startswith('/parallel/'):
                        return False
                    try:
                        barrier.wait()
                    except threading.BrokenBarrierError:
                        handler.send_error(400, 'downloads were serialized')
                        return True
                    handler.path = '/' + handler.path.rsplit('/', 1)[-1]
                    return False
                self.server.request_hook = hook
                for i in range(4):
                    Image.fromarray(self.data+i).save(self.root/f'{i}.png')
                cache = self.cache/str(processes)
                sources = [Source(str(i), Storage(self.url+f'/parallel/{i}.png', 'png')) for i in range(4)]
                options = dict(cache_dir=cache, cache_max_bytes=4096)
                if processes:
                    with multiprocessing.get_context('spawn').Pool(4) as pool:
                        results = pool.starmap(read_remote_source, [(s, options) for s in sources])
                else:
                    with ThreadPoolExecutor(4) as pool:
                        results = list(pool.map(lambda s: read_remote_source(s, options), sources))
                for i, result in enumerate(results):
                    np.testing.assert_array_equal(result, self.data+i)
                self.assertFalse(barrier.broken)
                self.assertEqual(list(cache.rglob('*.tmp.*')), [])
                with sqlite3.connect(cache/'.vc_reader_cache.sqlite3') as db:
                    self.assertEqual(db.execute('select count(*) from downloads').fetchone()[0], 0)

    def test_live_download_survives_another_cache_owner_and_budget_pressure(self):
        from vc_reader import _reader
        payload = (self.root/'image.png').read_bytes()
        release = threading.Event()
        def hook(handler):
            if handler.path != '/held.png':
                return False
            handler.send_response(200)
            handler.send_header('Content-Length', str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload[:len(payload)//2])
            handler.wfile.flush()
            if release.wait(15):
                handler.wfile.write(payload[len(payload)//2:])
            return True
        self.server.request_hook = hook
        options = dict(cache_dir=self.cache, cache_max_bytes=len(payload))
        held = Source('held', Storage(self.url+'/held.png', 'png'))
        other = Source('other', Storage(self.url+'/image.png', 'png'))
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(read_remote_source, held, options)
            try:
                deadline = time.monotonic()+10
                while time.monotonic() < deadline:
                    temporary = list(self.cache.rglob('*.tmp.*'))
                    if temporary and temporary[0].stat().st_size:
                        break
                    time.sleep(.01)
                else:
                    self.fail('download did not start writing')
                # A separately opened native owner runs the same reconciliation
                # as a newly started worker, while the first transfer is live.
                owner = _reader.DiskCache(str(self.cache), -1)
                self.assertTrue(temporary[0].exists())
                with sqlite3.connect(self.cache/'.vc_reader_cache.sqlite3') as db:
                    reserved = db.execute('select sum(size) from downloads').fetchone()[0]
                    self.assertEqual(reserved, len(payload))
                with self.assertRaisesRegex(RuntimeError, 'active download'):
                    owner.set_max_bytes(len(payload)-1)
                second = pool.submit(read_remote_source, other, options)
                # Other network requests may begin, but cannot consume the
                # reserved bytes until the first download publishes.
                deadline = time.monotonic()+5
                while self.counts.get('/image.png', 0) == 0 and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(self.counts.get('/image.png'), 1)
                self.assertFalse(second.done())
                with sqlite3.connect(self.cache/'.vc_reader_cache.sqlite3') as db:
                    total, limit = db.execute('select total,max_bytes from budget').fetchone()
                    reserved = db.execute('select sum(size) from downloads').fetchone()[0]
                    self.assertLessEqual(total+reserved, limit)
            finally:
                release.set()
            np.testing.assert_array_equal(first.result(10), self.data)
            np.testing.assert_array_equal(second.result(10), self.data)
        self.assertLessEqual(sum(p.stat().st_size for p in self.cache.rglob('*.blob')), len(payload))
        self.assertEqual(list(self.cache.rglob('*.tmp.*')), [])

    def test_abandoned_download_releases_its_reservation(self):
        payload = (self.root/'image.png').read_bytes()
        release = threading.Event()
        def hook(handler):
            if handler.path != '/abandoned.png':
                return False
            handler.send_response(200)
            handler.send_header('Content-Length', str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload[:len(payload)//2])
            handler.wfile.flush()
            release.wait(15)
            return True
        self.server.request_hook = hook
        disk_budget(str(self.cache), len(payload))
        source = Source('abandoned', Storage(self.url+'/abandoned.png', 'png'))
        process = multiprocessing.get_context('spawn').Process(
            target=read_remote_source, args=(source, dict(cache_dir=self.cache)))
        process.start()
        try:
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                temporary = list(self.cache.rglob('*.tmp.*'))
                if temporary and temporary[0].stat().st_size:
                    break
                time.sleep(.01)
            else:
                self.fail('download did not start writing')
            process.terminate()
            process.join(5)
            self.assertFalse(process.is_alive())
            with sqlite3.connect(self.cache/'.vc_reader_cache.sqlite3') as db:
                self.assertEqual(db.execute('select sum(size) from downloads').fetchone()[0], len(payload))
            other = Source('other', Storage(self.url+'/image.png', 'png'))
            # Reuse the already-open parent cache, so admission (not startup)
            # must reclaim the dead worker's reservation to make progress.
            np.testing.assert_array_equal(read_remote_source(other, dict(cache_dir=self.cache)), self.data)
            with sqlite3.connect(self.cache/'.vc_reader_cache.sqlite3') as db:
                self.assertEqual(db.execute('select count(*) from downloads').fetchone()[0], 0)
            self.assertEqual(list(self.cache.rglob('*.tmp.*')), [])
        finally:
            release.set()
            if process.is_alive():
                process.terminate()
            process.join(5)

    def test_unsized_and_http_compressed_images_respect_the_payload_budget(self):
        import gzip
        payload = (self.root/'image.png').read_bytes()
        def hook(handler):
            compressed = handler.path == '/gzip.png'
            body = gzip.compress(payload) if compressed else payload
            handler.send_response(200)
            if compressed:
                handler.send_header('Content-Encoding', 'gzip')
                handler.send_header('Content-Length', str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
            return True
        self.server.request_hook = hook
        for name in ('unsized', 'gzip'):
            source = Source(name, Storage(self.url+f'/{name}.png', 'png'))
            options = dict(cache_dir=self.cache/name, cache_max_bytes=len(payload))
            np.testing.assert_array_equal(read_remote_source(source, options), self.data)
            with self.assertRaisesRegex(RuntimeError, 'cache_max_bytes'):
                read_remote_source(source, dict(cache_dir=self.cache/(name+'-small'), cache_max_bytes=len(payload)-1))
        self.assertEqual(list(self.cache.rglob('*.tmp.*')), [])

    def test_remote_offline_validation_auth_s3_and_alternatives(self):
        source = Source('remote', Storage(self.url+'/image.png', 'png'))
        with source.open(cache_dir=self.cache) as reader:
            np.testing.assert_array_equal(reader.read((1, 2), (3, 4)), self.data[1:4, 2:6])
            self.assertEqual(reader.stats()['remote_bytes'], (self.root/'image.png').stat().st_size)
            before = dict(self.counts)
            reader.read((1, 2), (3, 4))
            self.assertEqual(before, self.counts)
        with source.open(cache_dir=self.cache, offline=True) as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data)
            self.assertEqual(before, self.counts)
        with source.open(cache_dir=self.cache) as reader:
            self.assertEqual(reader.stats()['remote_bytes'], 0)  # Conditional 304.
        missing = replace(source, storage=Storage(self.url+'/missing.png', 'png'))
        with self.assertRaises(CacheMissError):
            missing.open(cache_dir=self.cache, offline=True)
        alternate = replace(missing, storage=replace(missing.storage, alternatives=(source.storage.location,)))
        with alternate.open(cache_dir=self.cache) as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data)
        basic = replace(source, storage=Storage(self.url+'/basic/image.png', 'png', alternatives=(source.storage.location,)))
        with self.assertRaises(PermissionError):
            basic.open(cache_dir=self.cache)
        with basic.open(cache_dir=self.cache, auth=('user', 'password')) as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data)
        s3 = replace(source, storage=Storage('s3://bucket/image.png', 'png'))
        keys = S3Credentials('example', 'test-secret', region='us-east-1', endpoint_url=self.url)
        with s3.open(cache_dir=self.cache, auth=keys) as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data)
        with s3.open(cache_dir=self.cache, offline=True, auth='aws') as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data)

    def test_streaming_retries_reset_temporary_file_and_preserve_denial(self):
        for prefix in ('retry', 'truncated'):
            source = Source(prefix, Storage(self.url+f'/{prefix}/image.png', 'png'))
            with source.open(cache_dir=self.cache) as reader:
                np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data)
            self.assertEqual(self.counts[f'/{prefix}/image.png'], 2)
        denied = Source('denied', Storage(self.url+'/denied/image.png', 'png'))
        with self.assertRaisesRegex(PermissionError, 'user:denied.*403'):
            denied.open(cache_dir=self.cache)
        self.assertEqual(list(self.cache.rglob('*.tmp.*')), [])

    def test_disk_limit_atomic_population_eviction_and_reopen(self):
        size = (self.root/'image.png').stat().st_size
        source = Source('image', Storage(self.url+'/image.png', 'png'))
        with self.assertRaisesRegex(RuntimeError, 'cache_max_bytes'):
            source.open(cache_dir=self.cache, cache_max_bytes=size-1)
        self.assertEqual(list(self.cache.rglob('*.blob')), [])
        self.assertEqual(list(self.cache.rglob('*.tmp.*')), [])
        def read(_):
            with source.open(cache_dir=self.cache, cache_max_bytes=2*size) as reader:
                return reader.read((0, 0), (5, 7))
        with ThreadPoolExecutor(max_workers=4) as executor:
            for result in executor.map(read, range(8)):
                np.testing.assert_array_equal(result, self.data)
        self.assertEqual(len(list(self.cache.rglob('*.blob'))), 1)
        self.assertEqual(disk_budget(str(self.cache), None).max_bytes, 2*size)
        first = source.open(cache_dir=self.cache, cache_max_bytes=size)
        Image.fromarray(self.data+1).save(self.root/'other.png')
        other = replace(source, storage=Storage(self.url+'/other.png', 'png'))
        with other.open(cache_dir=self.cache, cache_max_bytes=size) as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data+1)
        # A previously opened reader can re-materialize an evicted file.
        np.testing.assert_array_equal(first.read((0, 0), (5, 7)), self.data)
        first.close()
        self.assertLessEqual(sum(p.stat().st_size for p in self.cache.rglob('*.blob')), size)
        restored = pickle.loads(pickle.dumps(source.open(cache_dir=self.cache)))
        np.testing.assert_array_equal(restored.read((0, 0), (5, 7)), self.data)
        restored.close()

    def test_remote_zarr_keeps_existing_cache_and_missing_chunk_semantics(self):
        path = self.root/'array.zarr'
        array(path, self.data, {'_ARRAY_DIMENSIONS': ['y', 'x']})
        source = Source('remote-zarr', Storage(self.url+'/array.zarr', 'zarr'))
        with source.open(cache_dir=self.cache, cache_max_bytes=1024) as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data)
        before = dict(self.counts)
        with source.open(cache_dir=self.cache, offline=True) as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data)
        self.assertEqual(before, self.counts)

    def test_remote_tiff_stack_lazy_reads_and_content_change(self):
        tifffile.imwrite(self.root/'image.tif', self.data)
        source = Source('tiff', Storage(self.url+'/image.tif', 'tiff'))
        with source.open(cache_dir=self.cache) as reader:
            np.testing.assert_array_equal(reader.read((1, 2), (3, 4)), self.data[1:4, 2:6])
        stack = Source('stack', Storage(format='stack', locations=(self.url+'/image.png',
                       self.url+'/missing.png', self.url+'/image.tif')), axes=('z', 'y', 'x'))
        with stack.open(cache_dir=self.cache) as reader:
            self.assertNotIn('/missing.png', self.counts)
            np.testing.assert_array_equal(reader.read((2, 0, 0), (1, 5, 7)), self.data[None])
            self.assertNotIn('/missing.png', self.counts)
        png = Source('changed', Storage(self.url+'/image.png', 'png'))
        old = png.open(cache_dir=self.cache)
        Image.fromarray(self.data+1).save(self.root/'image.png')
        with png.open(cache_dir=self.cache) as current:
            np.testing.assert_array_equal(current.read((0, 0), (5, 7)), self.data+1)
        with self.assertRaisesRegex(ValueError, 'source changed'):
            old.read((0, 0), (5, 7))
        old.close()

    def test_remote_process_population_temporary_limit_and_shared_chunk_budget(self):
        source = Source('workers', Storage(self.url+'/image.png', 'png'))
        with source.open(cache_dir=self.cache) as reader:
            for method in ('spawn', 'fork'):
                if method not in multiprocessing.get_all_start_methods():
                    continue
                with multiprocessing.get_context(method).Pool(2) as pool:
                    for result in pool.map(process_read, [reader]*4):
                        np.testing.assert_array_equal(result, self.data[:3, :4])
        self.assertEqual(len(list(self.cache.rglob('*.blob'))), 1)
        self.assertEqual(disk_budget(str(self.cache), None).max_bytes, 1 << 30)
        with self.assertRaisesRegex(RuntimeError, 'cache_max_bytes'):
            source.open(cache_dir=self.root/'unused', cache_writes=False, cache_max_bytes=1)
        with source.open(cache_writes=False, cache_max_bytes=1024) as reader:
            temporary = Path(reader._encoded._temp.name)
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data)
        self.assertFalse(temporary.exists())
        self.assertFalse((self.root/'unused').exists())
        # One directory budget counts both existing Zarr chunks and new files.
        array(self.root/'shared.zarr', self.data, {'_ARRAY_DIMENSIONS': ['y', 'x']})
        zarr = Source('shared', Storage(self.url+'/shared.zarr', 'zarr'))
        limit = (self.root/'image.png').stat().st_size
        disk_budget(str(self.cache), limit)
        with zarr.open(cache_dir=self.cache) as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data)
            self.assertEqual(reader._backend.reader.write_layer.budget_root, str(self.cache))
        with source.open(cache_dir=self.cache) as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), self.data)
        import sqlite3
        with sqlite3.connect(self.cache/'.vc_reader_cache.sqlite3') as db:
            total, maximum = db.execute('select total,max_bytes from budget').fetchone()
            self.assertLessEqual(total, maximum)
            self.assertEqual(maximum, limit)
        self.assertEqual(list(self.cache.rglob('*.tmp.*')), [])


if __name__ == '__main__':
    unittest.main()
