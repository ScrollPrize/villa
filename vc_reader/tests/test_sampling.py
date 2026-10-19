"""Independent small-array and analytic checks for aligned CPU/CUDA reads."""
import itertools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np
from PIL import Image
import tifffile

if os.environ.get('VC_READER_BUILD_DIR'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from extension_build import load_extensions
    load_extensions(Path(os.environ['VC_READER_BUILD_DIR']))

from vc_reader import (Affine, Catalog, Companion, Encoding, Frame, Plane, QueryContext, Selection,
                       Source, Storage, ValueFrame, VolumeRegistration)
from vc_reader.sampling import open_sample
from vc_reader.cuda import is_available
from test_source_storage import array, pyramid


def worker_read(reader):
    with reader:
        return reader.read((0, 0, 0), (2, 2, 2))['image'].data


def host(a):
    return a.numpy() if hasattr(a, 'numpy') else a


def reference(values, pull, start, shape, order, fill=0, plane=False):
    """Scalar double-precision oracle: no production planning/indexing helpers."""
    if values.ndim == 3:
        values = values[None]
    out = np.empty((values.shape[0], *shape), dtype=values.dtype if not order else 'float32')
    valid = np.ones((1, *shape), dtype=bool)
    for index in np.ndindex(shape):
        xyz = pull.apply(tuple(reversed(tuple(a+b for a, b in zip(index, start)))))
        q = tuple(reversed(xyz))
        supported = not plane or -.5 <= q[0] <= .5
        if plane:
            q = (0., *q[1:])
        neighbours = []
        if supported:
            if not order:
                at = tuple(round(v) for v in q)
                neighbours = [(at, 1.)]
            else:
                lo = tuple(int(np.floor(v)) for v in q)
                for bits in itertools.product((0, 1), repeat=3):
                    weight = np.prod([v-l if b else 1-(v-l) for v, l, b in zip(q, lo, bits)])
                    if weight:
                        neighbours.append((tuple(a+b for a, b in zip(lo, bits)), weight))
        result = np.zeros(values.shape[0], dtype=np.float64) if order else np.full(values.shape[0], fill, dtype=values.dtype)
        if not neighbours:
            result[...] = fill
        for at, weight in neighbours:
            inside = all(0 <= a < n for a, n in zip(at, values.shape[1:]))
            supported &= inside
            value = values[(slice(None), *at)] if inside else np.full(values.shape[0], fill, dtype=values.dtype)
            if order:
                result += weight*value
            else:
                result = value
        out[(slice(None), *index)] = result
        valid[(0, *index)] = supported
    return out, valid


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref = self.source('reference', np.arange(7*8*9, dtype='uint16').reshape(7, 8, 9), role='image')

    def source(self, name, data, role='label', axes=None, affine=None, **kwargs):
        path = self.root/(name+'.zarr')
        axes = axes or (('z', 'y', 'x') if data.ndim == 3 else ('c', 'z', 'y', 'x'))
        array(path, data, {'_ARRAY_DIMENSIONS': axes})
        return Source(name, Storage(str(path), 'zarr'), role=role,
                      registration=VolumeRegistration('s', 'v', affine=affine or Affine.identity()), **kwargs)

    def backends(self):
        return ('cpu', 'cuda') if is_available() else ('cpu',)

    def test_time_selected_zarr_direct_and_affine_sampling(self):
        values = np.arange(3*4*5, dtype=np.uint8).reshape(3, 4, 5)
        data = np.stack((np.zeros_like(values), values))
        for scale in (1, 2):
            source = self.source(f'time-{scale}', data, role='image', axes=('t', 'z', 'y', 'x'),
                                 affine=Affine.scale(scale)).with_selection(time=1)
            start, shape = (1, 1, 1), (2, 2, 3)
            expected, valid = reference(values, Affine.scale(1/scale), start, shape, 1)
            for backend in self.backends():
                with self.subTest(scale=scale, backend=backend), open_sample(
                        self.ref, {'selected': source}, backend=backend) as reader:
                    result = reader.read(start, shape)['selected']
                    np.testing.assert_allclose(host(result.data), expected, rtol=0, atol=1e-5)
                    np.testing.assert_array_equal(host(result.valid), valid)

    def test_sampling_cache_distinguishes_complete_file_stacks(self):
        paths = {}
        for name, value in (('common', 0), ('a', 10), ('b', 20)):
            path = self.root / f'{name}.png'
            Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(path)
            paths[name] = str(path)
        targets = {name: Source(name, Storage(format='stack', locations=(paths['common'], paths[name])),
                               role='image', registration=VolumeRegistration('s', 'v', affine=Affine.scale(2)))
                   for name in ('a', 'b')}
        for backend in self.backends():
            for cache_bytes in (0, 1 << 20):
                with self.subTest(backend=backend, cache_bytes=cache_bytes), open_sample(
                        self.ref, targets, backend=backend, sampling_cache_bytes=cache_bytes) as reader:
                    for _ in range(2):
                        result = reader.read((1, 0, 0), (1, 2, 2))
                        for name, value in (('a', 5), ('b', 10)):
                            np.testing.assert_array_equal(host(result[name].data), value)
                            self.assertTrue(host(result[name].valid).all())
                    if cache_bytes:
                        self.assertGreater(reader.stats()['tile_cache_hits'], 0)

    def test_numpy_indexing_shared_space_channels_and_validity(self):
        values = np.arange(7*8*9, dtype='uint16').reshape(7, 8, 9)
        channels = np.stack((values, values+1000, values+2000))
        mask = self.source('index-mask', (values % 2).astype('uint8'))
        label = self.source('index-label', values, companions=(Companion('quality', mask),))
        scalar = self.source('index-channels', channels, role='scalar')
        keys = [Ellipsis, (), np.s_[1:4, 2:6, 3:8], np.s_[-2:, ..., -1],
                np.s_[2], np.s_[1, 2, 3], np.s_[-100:100, 5:99, :], np.s_[3:1], np.s_[99:]]
        for backend in self.backends():
            with self.subTest(backend=backend), open_sample(self.ref, {'label': label, 'field': scalar}, backend=backend) as reader:
                self.assertEqual(reader.shape, values.shape)
                for key in keys:
                    result = reader[key]
                    for name, data in [('image', values[None]), ('label', values[None]), ('field', channels)]:
                        index = (slice(None), *(key if isinstance(key, tuple) else (key,)))
                        np.testing.assert_array_equal(host(result[name].data), data[index])
                        expected_valid = np.ones((1, *values.shape), dtype=bool)[index]
                        np.testing.assert_array_equal(host(result[name].valid), expected_valid)
                    np.testing.assert_array_equal(host(result['label'].masks['quality']), (values[None] % 2)[index])
                view = reader['field']
                self.assertEqual(view.shape, channels.shape)
                for key in [Ellipsis, np.s_[1, 1:4, 2:5, 3:7], np.s_[-1, ..., -1],
                            np.s_[1:3, 2, :, :], np.s_[0, 1, 2, 3], np.s_[2:1], np.s_[:, 99:]]:
                    np.testing.assert_array_equal(host(view[key]), channels[key])
                # Named target reads do not sample unrelated targets or masks.
                with patch.object(reader, '_sample_many', wraps=reader._sample_many) as sample_many:
                    saved = view[1, 1:3, 2:4, 3:5]
                    self.assertEqual(sample_many.call_count, 1)
                for key in [np.s_[7], np.s_[-8], np.s_[::2], np.s_[..., ...],
                            np.s_[:, :, :, :], None, True, [0, 1], .5]:
                    with self.subTest(invalid=key), self.assertRaises(IndexError):
                        reader[key]
                with self.assertRaises(KeyError):
                    reader['missing']
                with self.assertRaises(IndexError):
                    view[3]
            np.testing.assert_array_equal(host(saved), channels[1, 1:3, 2:4, 3:5])
            with self.assertRaisesRegex(ValueError, 'closed'):
                view[:]

    def test_channel_indexing_follows_geometric_transform(self):
        values = np.broadcast_to(np.array([1, 2, 3], 'float32')[:, None, None, None], (3, 7, 8, 9)).copy()
        field = self.source('index-vector', values, role='vector', component_order=('x', 'y', 'z'),
                            normalize=False, value_frame=ValueFrame(sample_id='s', volume_id='v',
                            affine=Affine.from_rows(((1, 2, 0, 0), (0, 1, 3, 0), (0, 0, 1, 0)))))
        for backend in self.backends():
            with open_sample(self.ref, {'vector': field}, backend=backend) as reader:
                np.testing.assert_array_equal(host(reader['vector'][0, 1:3, 2:4, 3:5]), np.full((2, 2, 2), 5, 'float32'))

    def test_zarr_indexing_cpu_and_cuda(self):
        from vc_reader.reader import open_zarr
        values = np.arange(4*5*6, dtype='uint8').reshape(4, 5, 6)
        source = self.source('index-u8', values, role='image')
        with open_zarr(source.storage.location) as reader:
            for key in [Ellipsis, np.s_[1:3, 2:4, 3:], np.s_[-1, ..., -2:], np.s_[1, 2, 3], np.s_[99:]]:
                np.testing.assert_array_equal(reader[key], values[key])
                if is_available():
                    with reader.cuda() as cuda:
                        np.testing.assert_array_equal(cuda[key].numpy(), values[key])
            np.testing.assert_array_equal(reader[[0, 2], :, 1:3], values[[0, 2], :, 1:3])
        with source.open() as reader:
            np.testing.assert_array_equal(reader[-2:, 1:4, -3:], values[-2:, 1:4, -3:])

    def test_affines_against_independent_reference(self):
        values = np.arange(5*6*7, dtype='uint64').reshape(5, 6, 7) + 2**55
        matrices = [Affine.identity(), Affine.from_rows(((1, 0, 0, .5), (0, 1, 0, -.5), (0, 0, 1, 1.25))),
                    Affine.from_rows(((0, -1, 0, 5), (1, 0, 0, 0), (0, 0, 1, 0))),
                    Affine.from_rows(((1, .25, 0, 0), (0, 1, .5, 0), (0, 0, 1, 0))),
                    Affine.from_rows(((-1, 0, 0, 6), (0, .5, 0, 0), (0, 0, 2, 0)))]
        for order in (0, 1):
            data = (values-2**55).astype('float32') if order else values
            source = self.source('data'+str(order), data, role='scalar' if order else 'label', fill_value=3)
            for pull in matrices:
                selected = replace(source, registration=replace(source.registration, affine=pull.inverse()))
                expected, mask = reference(data, pull, (-1, -1, -1), (7, 8, 9), order, 3)
                for backend in self.backends():
                    with self.subTest(order=order, pull=pull, backend=backend), open_sample(self.ref, {'target': selected}, backend=backend) as reader:
                        result = reader.read((-1, -1, -1), (7, 8, 9))
                        np.testing.assert_array_equal(host(result['target'].valid), mask)
                        if order:
                            np.testing.assert_allclose(host(result['target'].data), expected, atol=2e-5, rtol=2e-6)
                        else:
                            np.testing.assert_array_equal(host(result['target'].data), expected)
                            self.assertEqual(result['target'].data.dtype, data.dtype)
                        self.assertEqual(result['target'].valid.dtype, np.dtype(bool))

    def test_slice_footprint_and_partial_fill(self):
        path = self.root/'slice.png'
        Image.fromarray(np.full((3, 4), 10, 'uint8')).save(path)
        plane = Plane((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 2))
        source = Source('slice', Storage(str(path), 'png'), role='scalar', registration=VolumeRegistration('s', 'v', plane=plane))
        for backend in self.backends():
            with open_sample(self.ref, {'slice': source}, backend=backend) as reader:
                result = reader.read((-2, 0, 0), (5, 1, 1))
                np.testing.assert_array_equal(host(result['slice'].data).ravel(), [0, 10, 10, 10, 0])
                np.testing.assert_array_equal(host(result['slice'].valid).ravel(), [False, True, True, True, False])
            shifted = replace(source, registration=replace(source.registration, plane=replace(plane, origin=(.5, 0, 0))))
            with open_sample(self.ref, {'slice': shifted}, backend=backend) as reader:
                result = reader.read((0, 0, 0), (1, 1, 2))
                np.testing.assert_array_equal(host(result['slice'].data).ravel(), [5, 10])
                np.testing.assert_array_equal(host(result['slice'].valid).ravel(), [False, True])

    def test_large_origins_ties_singletons_and_exact_encodings(self):
        path = self.root/'labels.png'
        Image.fromarray(np.array([[0, 1, 2, 3]], 'uint8')).save(path)
        origin = 2**26+1
        source = Source('encoded', Storage(str(path), 'png'), role='label',
                        registration=VolumeRegistration('s', 'v', plane=Plane((origin+.5, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))),
                        encoding=Encoding('integer', tuple((i, 2**60+i) for i in range(4)), 'uint64'), fill_value=2**60+9)
        for backend in self.backends():
            with open_sample(self.ref, {'label': source}, backend=backend) as reader:
                result = reader.read((0, 0, origin), (1, 1, 5))
                np.testing.assert_array_equal(host(result['label'].data).ravel(), np.array([2**60, 2**60, 2**60+2, 2**60+2, 2**60+9], dtype='uint64'))
                np.testing.assert_array_equal(host(result['label'].valid).ravel(), [True, True, True, True, False])

    def test_fields_frames_order_normals_and_metadata(self):
        values = np.broadcast_to(np.array([3, 2, 1], 'float32')[:, None, None, None], (3, 7, 8, 9)).copy()
        frame = ValueFrame(sample_id='s', volume_id='v', affine=Affine.from_rows(((-2, 0, 0, 10), (0, 3, 0, 20), (0, 0, 4, 30))))
        vec = self.source('vector', values, role='vector', component_order=('z', 'y', 'x'), value_frame=frame, normalize=False)
        normal = replace(vec, role='normal', normalize=None)
        coord = replace(vec, role='coordinate', normalize=None)
        external = replace(coord, value_frame=ValueFrame(kind='external', name='instrument', units='um'))
        for backend in self.backends():
            with open_sample(self.ref, {'v': vec, 'n': normal, 'c': coord, 'e': external}, backend=backend) as reader:
                result = reader.read((1, 2, 3), (1, 1, 1))
                np.testing.assert_array_equal(host(result['v'].data).ravel(), [12, 6, -2])
                expected = np.array([.75, 2/3, -.5]); expected /= np.linalg.norm(expected)
                np.testing.assert_allclose(host(result['n'].data).ravel(), expected, rtol=2e-7)
                np.testing.assert_array_equal(host(result['c'].data).ravel(), [42, 26, 8])
                np.testing.assert_array_equal(host(result['e'].data).ravel(), [3, 2, 1])
                self.assertEqual(result.metadata['value_frames']['e']['name'], 'instrument')
                self.assertEqual(result.metadata['value_frames']['v']['component_order'], ('z', 'y', 'x'))
        values[:] = 0; values[:, 0, 0, 1] = np.nan
        bad = self.source('bad', values, role='normal', component_order=('x', 'y', 'z'), value_frame=frame)
        for backend in self.backends():
            with open_sample(self.ref, {'n': bad}, backend=backend) as reader:
                result = reader.read((0, 0, 0), (1, 1, 2))
                self.assertFalse(host(result['n'].valid).any())
                np.testing.assert_array_equal(host(result['n'].data), 0)

    def test_masks_channel_time_selection_and_buffer_lifetime(self):
        data = np.arange(2*7*8*9*3, dtype='int32').reshape(2, 7, 8, 9, 3)
        source = self.source('selected', data, axes=('t', 'z', 'y', 'x', 'c')).with_selection(time=1, channels=(2, 0))
        mask = self.source('mask', np.full((7, 8, 9), 7, 'uint8'))
        mask = replace(mask, registration=None)
        source = replace(source, companions=(Companion('supervision', mask, True),))
        for backend in self.backends():
            with open_sample(self.ref, {'label': source}, backend=backend) as reader:
                first, second = reader.read_many([((1, 2, 3), (2, 2, 2)), ((0, 0, 0), (1, 1, 1))])
                reader.read((0, 0, 0), (1, 1, 1))
                self.assertIsNone(first.metadata)
            expected = np.moveaxis(data[1, 1:3, 2:4, 3:5][:, :, :, [2, 0]], -1, 0)
            np.testing.assert_array_equal(host(first['label'].data), expected)
            np.testing.assert_array_equal(host(first['label'].masks['supervision']), 7)
            self.assertEqual(first['label'].masks['supervision'].shape, (1, 2, 2, 2))
            self.assertTrue(host(first['label'].valid).all())

    def test_volume_level_uses_resolved_storage_and_value_level(self):
        path = self.root/'pyramid.zarr'
        pyramid(path, scales=(1, 3, 6), units='micrometer')
        cat = Catalog({'samples': {'s': {'sample': {'id': 's'}, 'volumes': {'v': {
            'id': 'v', 'properties': {'pixel_size_um': 1}, 'data': [{'id': 'image', 'type': 'ome-zarr', 'origins': [
                {'path': str(path), 'access_roots': [{'url': '', 'type': 'local'}]}]}]}}}}})
        level = cat.samples['s'].volumes['v'].level(1)
        values = np.ones((3, 7, 8, 9), dtype='float32')
        field = self.source('level-field', values, role='vector', affine=Affine.scale(3),
                            component_order=('x', 'y', 'z'), normalize=False,
                            value_frame=ValueFrame(sample_id='s', volume_id='v', level=1))
        with open_sample(level, {'label': replace(self.ref, role='label'), 'vector': field}) as reader:
            np.testing.assert_array_equal(reader.read((0, 0, 0), (2, 2, 2))['vector'].data, 1)
            self.assertEqual(reader._reference_map, Affine.scale(3))
            np.testing.assert_array_equal(reader.read((0, 0, 0), (2, 2, 2))['image'].data, 2)

    def test_budget_eviction_repeated_requests_and_worker_reopening(self):
        values = np.arange(20*22*24, dtype='uint16').reshape(20, 22, 24)
        pull = Affine.from_rows(((1, .25, 0, .5), (0, 1, 0, 0), (0, 0, 1, 0)))
        source = self.source('bounded', values, affine=pull.inverse())
        expected, mask = reference(values, pull, (0, 0, 0), (12, 14, 16), 0)
        for backend in self.backends():
            with open_sample(self.ref, {'t': source}, backend=backend, sampling_cache_bytes=8192,
                             sampling_work_bytes=8192*3*8, plan_cache_bytes=512) as reader:
                first = reader.read((0, 0, 0), (12, 14, 16))
                reader.read((5, 5, 5), (12, 14, 16))
                np.testing.assert_array_equal(host(first['t'].data), expected)
                np.testing.assert_array_equal(host(first['t'].valid), mask)
                stats = reader.stats()
                self.assertLessEqual(stats['tile_cache_bytes'], 8192)
                self.assertLessEqual(stats['plan_cache_bytes'], 512)
                self.assertLessEqual(stats['peak_tile_work_bytes'], 8192*3*8)
            with open_sample(self.ref, {'t': source}, backend=backend, sampling_work_bytes=1) as reader:
                with self.assertRaisesRegex(MemoryError, 'sampling_work_bytes'):
                    reader.read((0, 0, 0), (1, 1, 1))
        with open_sample(self.ref) as reader:
            reader.read((0, 0, 0), (2, 2, 2))
            with patch('vc_reader.sources.Source.open', side_effect=AssertionError('opened during unpickle')):
                restored = pickle.loads(pickle.dumps(reader))
            np.testing.assert_array_equal(worker_read(restored), worker_read(pickle.loads(pickle.dumps(reader))))
            for method in ('spawn', 'fork'):
                if method in multiprocessing.get_all_start_methods():
                    with multiprocessing.get_context(method).Pool(1) as pool:
                        result = pool.apply(worker_read, (reader,))
                        self.assertEqual(result.shape, (1, 2, 2, 2))
            with ThreadPoolExecutor(2) as pool:
                results = list(pool.map(lambda _: reader.read((0, 0, 0), (2, 2, 2)), range(4)))
            for result in results:
                np.testing.assert_array_equal(result['image'].data, results[0]['image'].data)

    def test_color_encoding_missing_chunks_and_external_value_frame(self):
        path = self.root/'rgb.png'
        Image.fromarray(np.full((4, 5, 3), (1, 2, 3), 'uint8')).save(path)
        label = Source('rgb', Storage(str(path), 'png'), role='label',
                       registration=VolumeRegistration('s', 'v', plane=Plane((.25, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))),
                       encoding=Encoding('color', (((1, 2, 3), 2**63+17),), 'uint64'))
        missing = self.source('missing', np.ones((7, 8, 9), 'uint8'))
        (Path(missing.storage.location)/'0.0.0').unlink()
        missing = replace(missing, registration=replace(missing.registration, affine=Affine.scale(2)),
                          encoding=Encoding('integer', ((0, 2**63+19),), 'uint64'), fill_value=2**63+21)
        for backend in self.backends():
            with open_sample(self.ref, {'rgb': label, 'missing': missing}, backend=backend) as reader:
                result = reader.read((0, 0, 0), (1, 3, 4))
                np.testing.assert_array_equal(host(result['rgb'].data), np.uint64(2**63+17))
                np.testing.assert_array_equal(host(result['missing'].data), np.uint64(2**63+19))
                self.assertTrue(host(result['missing'].valid).all())


    def test_dtypes_oblique_sampling_and_geometric_padding(self):
        pull = Affine.from_rows(((.866025403784, -.5, 0, 2.25), (.5, .866025403784, 0, .125), (0, 0, 1, 0)))
        for dtype in ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', '>u2'):
            values = (np.arange(4*5*6).reshape(4, 5, 6) % 17).astype(dtype)
            source = self.source('dtype'+dtype.replace('>', 'be'), values, affine=pull.inverse())
            expected, valid = reference(values, pull, (-1, -1, -1), (6, 7, 8), 0)
            for backend in self.backends():
                with open_sample(self.ref, {'t': source}, backend=backend) as reader:
                    result = reader.read((-1, -1, -1), (6, 7, 8))
                    np.testing.assert_array_equal(host(result['t'].data), expected)
                    np.testing.assert_array_equal(host(result['t'].valid), valid)
        values = np.ones((3, 7, 8, 9), dtype='float32')
        frame = ValueFrame(sample_id='s', volume_id='v', affine=Affine.from_rows(((2, 0, 0, 10), (0, 3, 0, 20), (0, 0, 4, 30))))
        source = self.source('padding', values, role='coordinate', component_order=('x', 'y', 'z'), value_frame=frame)
        for affine in (Affine.identity(), Affine.from_rows(((1, 0, 0, .5), (0, 1, 0, 0), (0, 0, 1, 0)))):
            selected = replace(source, registration=replace(source.registration, affine=affine))
            for backend in self.backends():
                with open_sample(self.ref, {'t': selected}, backend=backend) as reader:
                    result = reader.read((0, 0, -2), (1, 1, 4))
                    np.testing.assert_array_equal(host(result['t'].data)[:, 0, 0, 0], 0)
                    np.testing.assert_array_equal(host(result['t'].data)[:, 0, 0, -1], [12, 23, 34])
                    self.assertFalse(host(result['t'].valid)[0, 0, 0, 0])


    def test_u8_fast_path_partial_support_and_independent_companions(self):
        values = np.arange(5*6*7, dtype='uint8').reshape(5, 6, 7)
        pull = Affine.from_rows(((1, .125, 0, .25), (0, 1, .125, .25), (0, 0, 1, .25)))
        source = self.source('u8-linear', values, role='scalar', affine=pull.inverse())
        mask = self.source('separate-mask', np.full((1, 1, 1), 5, dtype='uint8'))
        source = replace(source, companions=(Companion('quality', mask),))
        expected, valid = reference(values, pull, (0, 0, 0), (7, 8, 9), 1)
        for backend in self.backends():
            with open_sample(self.ref, {'s': source}, backend=backend) as reader:
                result = reader.read((0, 0, 0), (7, 8, 9))
                np.testing.assert_allclose(host(result['s'].data), expected, rtol=2e-6, atol=2e-5)
                np.testing.assert_array_equal(host(result['s'].valid), valid)
                self.assertEqual(host(result['s'].masks['quality'])[0, 0, 0, 0], 5)
                self.assertEqual(host(result['s'].masks['quality'])[0, 1, 1, 1], 0)
                self.assertTrue(host(result['s'].valid)[0, 1, 1, 1])

    @unittest.skipUnless(is_available(), 'CUDA build/device required')
    def test_cuda_stream_dlpack_and_worker_guard(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is optional')
        from vc_reader import _reader
        stream = torch.cuda.Stream()
        try:
            _reader._set_stream(stream.cuda_stream)
            with open_sample(self.ref, backend='cuda') as reader:
                result = reader.read((-1, 0, 0), (3, 2, 2))
            value = torch.from_dlpack(result['image'].data)
            valid = torch.from_dlpack(result['image'].valid)
            self.assertEqual(valid.dtype, torch.bool)
            self.assertFalse(valid[0, 0].any().item())
            self.assertTrue(valid[0, 1:].all().item())
            np.testing.assert_array_equal(value.cpu().numpy(), host(result['image'].data))
            with patch('torch.utils.data.get_worker_info', return_value=object()):
                with self.assertRaisesRegex(RuntimeError, 'DataLoader'):
                    open_sample(self.ref, backend='cuda')
        finally:
            _reader._set_stream(0)


    def test_sparse_large_array_origins_and_multiblock_exact_labels(self):
        path = self.root/'large.zarr'
        path.mkdir()
        base = 2**26
        (path/'.zarray').write_text(json.dumps(dict(zarr_format=2, shape=[1, 1, base+8], chunks=[1, 1, 4],
            dtype='<u8', compressor=None, filters=None, fill_value=0, order='C')))
        (path/'.zattrs').write_text(json.dumps({'_ARRAY_DIMENSIONS': ['z', 'y', 'x']}))
        values = np.arange(8, dtype='uint64') + np.uint64(2**60)
        for i in range(2):
            (path/f'0.0.{base//4+i}').write_bytes(values[i*4:i*4+4].tobytes())
        source = Source('large', Storage(str(path), 'zarr'), role='label',
                        registration=VolumeRegistration('s', 'v', affine=Affine.from_rows(((1, 0, 0, -.5), (0, 1, 0, 0), (0, 0, 1, 0)))))
        data = (np.arange(2*3*140, dtype='uint64').reshape(2, 3, 140)+np.uint64(2**60))
        block_source = self.source('blocks', data, affine=Affine.from_rows(((1, 0, 0, -.5), (0, 1, 0, 0), (0, 0, 1, 0))))
        expected, valid = reference(data, block_source.registration.affine.inverse(), (0, 0, 0), (2, 3, 129), 0)
        for backend in self.backends():
            with open_sample(self.ref, {'label': source}, backend=backend) as reader:
                result = reader.read((0, 0, base), (1, 1, 5))
                np.testing.assert_array_equal(host(result['label'].data).ravel(), values[[0, 2, 2, 4, 4]])
            with open_sample(self.ref, {'label': block_source}, backend=backend) as reader:
                result = reader.read((0, 0, 0), (2, 3, 129))
                np.testing.assert_array_equal(host(result['label'].data), expected)
                np.testing.assert_array_equal(host(result['label'].valid), valid)


    def test_tiff_and_jpeg_categorical_integration(self):
        values = np.arange(5*7, dtype='uint16').reshape(5, 7)*1000
        tif = self.root/'labels16.tif'
        tifffile.imwrite(tif, values)
        jpeg = self.root/'decoded.jpg'
        Image.fromarray((values % 251).astype('uint8')).save(jpeg, quality=93)
        plane = Plane((.5, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        with Image.open(jpeg) as decoded:
            jpeg_values = np.asarray(decoded).copy()
        for path, fmt, data in ((tif, 'tiff', values), (jpeg, 'jpeg', jpeg_values)):
            source = Source(path.name, Storage(str(path), fmt), role='label',
                            registration=VolumeRegistration('s', 'v', plane=plane))
            expected, valid = reference(data[None], Affine.from_rows(((1, 0, 0, -.5), (0, 1, 0, 0), (0, 0, 1, 0))),
                                        (0, -1, -1), (1, 7, 9), 0, plane=True)
            for backend in self.backends():
                with open_sample(self.ref, {'label': source}, backend=backend) as reader:
                    result = reader.read((0, -1, -1), (1, 7, 9))
                    np.testing.assert_array_equal(host(result['label'].data), expected)
                    np.testing.assert_array_equal(host(result['label'].valid), valid)


    def test_contract_errors_and_lazy_import(self):
        with self.assertRaisesRegex(ValueError, 'reserved'):
            open_sample(self.ref, {'image': self.ref})
        with self.assertRaisesRegex(ValueError, 'placement'):
            open_sample(replace(self.ref, registration=VolumeRegistration('s', 'v')))
        with self.assertRaisesRegex(ValueError, 'singular'):
            open_sample(self.ref, {'bad': replace(self.ref, registration=VolumeRegistration('s', 'v', affine=Affine.scale(0)))})
        with self.assertRaisesRegex(ValueError, 'relationship'):
            open_sample(self.ref, {'bad': replace(self.ref, registration=VolumeRegistration('s', 'other', affine=Affine.identity()))})
        with open_sample(self.ref) as reader:
            self.assertEqual(reader.read_many([]), [])
            result = reader.read((0, 0, 0), (0, 1, 1))
            self.assertEqual(result['image'].data.shape, (1, 0, 1, 1))
        with self.assertRaisesRegex(ValueError, 'closed'):
            reader.read((0, 0, 0), (1, 1, 1))
        code = "import sys; import vc_reader.sampling; assert not {'numpy', 'torch', 'PIL', 'vc_reader._reader'} & sys.modules.keys()"
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_catalog_contexts_and_source_reference_coordinates(self):
        cat = Catalog({'samples': {'s': {'sample': {'id': 's', 'properties': {'volume_transforms': [
            {'from_volume_id': 'a', 'transforms': [{'to_volume_id': 'v', 'matrix': Affine.scale(2).matrix}]}]}},
            'volumes': {'a': {'id': 'a'}, 'v': {'id': 'v'}}}}})
        other = replace(self.ref, registration=VolumeRegistration('s', 'a', affine=Affine.identity()))
        for backend in self.backends():
            with open_sample(self.ref, {'other': other}, catalog=cat, backend=backend) as reader:
                self.assertTrue(host(reader.read((0, 0, 0), (2, 2, 2))['other'].valid).all())
            query = replace(other, query=QueryContext(Frame('s', 'v'), 'transform', Affine.scale(2)))
            with open_sample(self.ref, {'other': query}, backend=backend) as reader:
                self.assertTrue(host(reader.read((0, 0, 0), (2, 2, 2))['other'].valid).all())
        inverse_reference = replace(self.ref, query=QueryContext(Frame('s', 'a'), 'transform', Affine.scale(.5)))
        with open_sample(inverse_reference, {'other': other}) as reader:
            self.assertTrue(reader.read((0, 0, 0), (2, 2, 2))['other'].valid.all())
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            open_sample(self.ref, {'bad': replace(query, query=replace(query.query, transform=Affine.scale(3)))}, catalog=cat)
        with self.assertRaisesRegex(ValueError, 'relationship'):
            open_sample(self.ref, {'bad': replace(query, query=replace(query.query, reference=Frame('s', 'elsewhere')))})
        mapping = Affine.from_rows(((0, -2, 0, 10), (3, 0, 0, 20), (0, 0, 4, 30)))
        ref = replace(self.ref, registration=replace(self.ref.registration, affine=mapping))
        xyz = np.stack(np.meshgrid(np.arange(7), np.arange(8), np.arange(9), indexing='ij')[::-1]).astype('float32')
        xyz = np.einsum('ij,jzyx->izyx', np.asarray(mapping.linear, dtype='float32'), xyz) + np.array(mapping.translation, dtype='float32')[:, None, None, None]
        coord = self.source('positions', xyz, role='coordinate', affine=mapping, component_order=('x', 'y', 'z'),
                            value_frame=ValueFrame(sample_id='s', volume_id='v', affine=Affine.identity()))
        for backend in self.backends():
            with open_sample(ref, {'p': coord}, backend=backend) as reader:
                a, b = reader.read_many([((1, 2, 3), (2, 2, 2)), ((2, 3, 4), (1, 1, 1))])
                np.testing.assert_allclose(host(a['p'].data)[:, 1, 1, 1], [4, 3, 2], atol=2e-6)
                np.testing.assert_array_equal(host(a['p'].data)[:, 1:, 1:, 1:], host(b['p'].data))


if __name__ == '__main__':
    unittest.main()
