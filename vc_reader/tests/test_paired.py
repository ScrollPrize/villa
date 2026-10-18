"""Flattened pair indexing, stored grids, exact categories, and worker lifetime."""
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from vc_reader import Affine, Catalog, Companion, Plane, Selection, Source, Storage, VolumeRegistration
from test_source_storage import array
from vc_reader.cuda import is_available


def host(value):
    return value.numpy() if hasattr(value, 'numpy') else value


def read_worker(reader):
    with reader:
        return reader[1:3, 2:5]


def pyramid(path, values, scales):
    path.mkdir()
    axes = ('z', 'y', 'x') if values[0].ndim == 3 else ('y', 'x')
    datasets = []
    for level, (value, scale) in enumerate(zip(values, scales)):
        array(path/str(level), value)
        datasets.append(dict(path=str(level), coordinateTransformations=[dict(type='scale', scale=([1.] if 'z' in axes else []) + [scale, scale])]))
    (path/'.zgroup').write_text('{"zarr_format":2}')
    (path/'.zattrs').write_text(json.dumps(dict(multiscales=[dict(axes=[dict(name=a, unit='micrometer') for a in axes], datasets=datasets)])))
    return Source(path.name, Storage(str(path), 'zarr', pyramid=True, selection=Selection(level=0)), axes=axes)


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scroll = Catalog(dict(samples=dict(s=dict(sample=dict(id='s', properties=dict(type='scroll')), volumes=dict(v=dict(id='v')))))).scrolls['s']

    def source(self, name, values, role):
        path = self.root/name
        axes = {2: ('y', 'x'), 3: ('z', 'y', 'x'), 4: ('c', 'z', 'y', 'x')}[values.ndim]
        array(path, values, {'_ARRAY_DIMENSIONS': axes})
        return Source(name, Storage(str(path), 'zarr'), role=role, axes=axes)

    def pair(self, render, label, **mapping):
        scroll = self.scroll.with_source(render, kind='image', volume='v')
        scroll = scroll.with_source(label, kind='label', volume='v', image=render.id, **(mapping or {'same_grid': True}))
        return scroll.labels[label.id]

    def backends(self):
        return ('cpu', 'cuda') if is_available() else ('cpu',)

    def test_shapes_indexing_channels_layers_and_ownership(self):
        image = np.arange(2*5*7*9, dtype='uint16').reshape(2, 5, 7, 9)
        labels = np.arange(7*9, dtype='uint8').reshape(7, 9)
        render = self.source('image', image, 'image')
        label = self.source('labels', labels, 'label')
        mask = self.source('mask', np.full((7, 9), 128, 'uint8'), 'label')
        label = replace(label, companions=(Companion('supervision', mask, True),))
        asset = self.pair(render, label)
        for backend in self.backends():
            with asset.open_paired(image=asset.images.zarr(), layers=slice(1, 4), backend=backend) as reader:
                patch = reader[-6:-2, -7:-2]
                self.assertEqual(patch.metadata['image_id'], asset.images.zarr().id)
                self.assertIs(patch.image, patch['image'])
                self.assertIs(patch.label, patch['label'])
                self.assertEqual(patch.image.masks, {})
                np.testing.assert_array_equal(host(patch['image'].data), image[:, 1:4, 1:5, 2:7])
                np.testing.assert_array_equal(host(patch['label'].data), labels[None, 1:5, 2:7])
                np.testing.assert_array_equal(host(patch['label'].masks['supervision']), 128)
                self.assertEqual(patch['label'].valid.shape, (1, 4, 5))
                np.testing.assert_array_equal(host(reader['image'][1, -1, 2:5, 3:6]), image[1, 3, 2:5, 3:6])
                np.testing.assert_array_equal(host(reader['label'][0, -1, -1]), labels[-1, -1])
                self.assertEqual(reader[3, 2:4]['image'].data.shape, (2, 3, 2))
                self.assertEqual(reader[3, 2:4]['label'].data.shape, (1, 2))
                row = reader[3, 2:4].label
                self.assertEqual(row.valid.shape, (1, 2))
                np.testing.assert_array_equal(host(row.masks['supervision']), [[128, 128]])
                crops = reader.read_many([((1, 2), (2, 3)), ((4, 5), (1, 2))])
                crops[0].label.masks.clear()
                self.assertEqual(crops[1].label.masks['supervision'].shape, (1, 1, 2))
                self.assertEqual(patch.label.masks['supervision'].shape, (1, 4, 5))
                self.assertEqual(reader[:0, :]['image'].data.shape, (2, 3, 0, 9))
                self.assertEqual(reader['image'][:0, :, :, :].shape, (0, 3, 7, 9))
                with self.assertRaises(IndexError):
                    reader[99, :]
                with self.assertRaises(IndexError):
                    reader[::2, :]
                clipped = reader[-100:100, :]
                self.assertEqual(clipped['label'].data.shape, (1, 7, 9))
                reader[0:1, 0:1]
            np.testing.assert_array_equal(host(patch['label'].data), labels[None, 1:5, 2:7])
            with asset.open_paired(layers=-1, backend=backend) as reader:
                np.testing.assert_array_equal(host(reader[:2, :3]['image'].data), image[:, -1, :2, :3])
            with asset.open_paired(layers=slice(0, 0), backend=backend) as reader:
                self.assertEqual(reader[:2, :3]['image'].data.shape, (2, 0, 2, 3))
                self.assertEqual(reader[:2, :3]['label'].data.shape, (1, 2, 3))

    def test_affine_exact_categories_validity_padding_and_cache(self):
        image = np.zeros((3, 8, 10), 'uint8')
        # Large categorical integers verify that nearest sampling never converts
        # class IDs through floating point.
        labels = np.arange(8*10, dtype='uint64').reshape(8, 10) + 2**55
        asset = self.pair(self.source('render', image, 'image'), self.source('labels', labels, 'label'),
                          pixel_affine=((2., 0., 1.), (0., 2., -1.)))
        expected = np.zeros((1, 8, 10), dtype='uint64')
        valid = np.zeros((1, 8, 10), dtype='bool')
        for y in range(8):
            for x in range(10):
                sy, sx = round((y+1)/2), round((x-1)/2)
                if 0 <= sy < 8 and 0 <= sx < 10:
                    expected[0, y, x], valid[0, y, x] = labels[sy, sx], True
        for backend in self.backends():
            with asset.open_paired(backend=backend, sampling_cache_bytes=1024, plan_cache_bytes=1024) as reader:
                for _ in range(2):
                    result = reader[:, :]
                    np.testing.assert_array_equal(host(result['label'].data), expected)
                    np.testing.assert_array_equal(host(result['label'].valid), valid)
                stats = reader.stats()
                self.assertLessEqual(stats['tile_cache_bytes'], 1024)
                self.assertLessEqual(stats['plan_cache_bytes'], 1024)
                padded = reader.read((-3, -2), (5, 4))
                self.assertFalse(host(padded['image'].valid)[..., :3, :].any())

    def test_small_affine_difference_on_large_canvas_is_not_snapped(self):
        width = 20000001
        for name, shape, chunks, axes in (('large-render', (1,1,width), (1,1,16), ('z','y','x')),
                                         ('large-label', (1,width), (1,16), ('y','x'))):
            path = self.root/name
            path.mkdir()
            (path/'.zarray').write_text(json.dumps(dict(zarr_format=2, shape=shape, chunks=chunks,
                dtype='|u1', compressor=None, filters=None, fill_value=0, order='C')))
            (path/'.zattrs').write_text(json.dumps({'_ARRAY_DIMENSIONS': axes}))
        path = self.root/'large-label'
        (path/'0.624999').write_bytes(bytes([99]*16))
        (path/'0.625000').write_bytes(bytes([100]*16))
        render = Source('large-render', Storage(str(self.root/'large-render'), 'zarr'), role='image', axes=('z','y','x'))
        label = Source('large-label', Storage(str(path), 'zarr'), role='label', axes=('y','x'))
        asset = self.pair(render, label, pixel_affine=((1.00000006,0,0),(0,1,0)))
        # 10,000,000 / 1.00000006 rounds to 9,999,999. Snapping the
        # small scale difference would instead select the next chunk's class.
        for backend in self.backends():
            with asset.open_paired(backend=backend) as reader:
                np.testing.assert_array_equal(host(reader[:,10000000:10000001]['label'].data), [[[99]]])

    def test_independent_pyramids_choose_matching_grid_or_sample_base(self):
        image = np.arange(3*8*10, dtype='uint16').reshape(3, 8, 10)
        labels = np.arange(8*10, dtype='uint16').reshape(8, 10)
        render = replace(pyramid(self.root/'r', [image, image[:, ::2, ::2]], [1, 2]), role='image')
        # Label level 1 is 4x; level 2 is 2x. Masks have only the base array.
        label = replace(pyramid(self.root/'l', [labels, labels[::4, ::4], labels[::2, ::2]], [1, 4, 2]), role='label')
        mask = self.source('mask', np.full((8, 10), 255, 'uint8'), 'label')
        label = replace(label, companions=(Companion('validation', mask, True),))
        asset = self.pair(render, label)
        for backend in self.backends():
            with asset.open_paired(level=1, backend=backend) as reader:
                self.assertEqual(reader.metadata['label_selection'].array_path, '2')
                result = reader[:, :]
                np.testing.assert_array_equal(host(result['image'].data), image[None, :, ::2, ::2])
                np.testing.assert_array_equal(host(result['label'].data), labels[None, ::2, ::2])
                np.testing.assert_array_equal(host(result['label'].masks['validation']), 255)
        # With no matching stored label level, resample categorical base values.
        label2 = self.source('labels2', labels, 'label')
        asset = self.pair(render, label2)
        with asset.open_paired(level=1) as reader:
            np.testing.assert_array_equal(reader[:, :]['label'].data, labels[None, ::2, ::2])
        with self.assertRaises((ValueError, FileNotFoundError)):
            asset.open_paired(level=3)

    def test_no_implicit_pairing_and_reject_contradictory_grids(self):
        render = self.source('r', np.zeros((2, 4, 5), 'uint8'), 'image')
        label = self.source('l', np.zeros((4, 5), 'uint8'), 'label')
        scroll = self.scroll.with_source(render, kind='image', volume='v').with_source(label, kind='label', volume='v')
        self.assertFalse(scroll.labels.get().images)
        with self.assertRaisesRegex(ValueError, 'absent'):
            scroll.labels.get().open_paired(image=scroll.images.get())
        bad = self.source('bad', np.zeros((3, 5), 'uint8'), 'label')
        with self.assertRaisesRegex(ValueError, 'shapes'):
            self.pair(render, bad).open_paired()
        render2 = replace(pyramid(self.root/'rp', [np.zeros((2, 4, 5), 'uint8')], [2]), role='image')
        label2 = replace(pyramid(self.root/'lp', [np.zeros((4, 5), 'uint8')], [1]), role='label')
        with self.assertRaisesRegex(ValueError, 'contradict'):
            self.pair(render2, label2).open_paired()

    def test_registered_companion_plane_and_local_image_formats(self):
        from PIL import Image
        import tifffile
        image = np.arange(3*7*9, dtype='uint16').reshape(3, 7, 9)
        labels = np.arange(7*9, dtype='uint8').reshape(7, 9)
        tifffile.imwrite(self.root/'render.tif', image, photometric='minisblack', metadata={'axes': 'ZYX'})
        Image.fromarray(labels).save(self.root/'label.png')
        render = Source('tiff-render', Storage(str(self.root/'render.tif'), 'tiff'), role='image', axes=('z', 'y', 'x'))
        registration = VolumeRegistration('s', 'v', plane=Plane((0,0,0), (1,0,0), (0,1,0), (0,0,1)))
        label = Source('png-label', Storage(str(self.root/'label.png'), 'png'), role='label', axes=('y', 'x'), registration=registration)
        mask = self.source('mask-registered', np.full((2, 3), 128, 'uint8'), 'label')
        mask = replace(mask, registration=replace(registration, plane=Plane((2,1,0), (1,0,0), (0,1,0), (0,0,1))))
        label = replace(label, companions=(Companion('supervision', mask),))
        expected = np.zeros((1,7,9), 'uint8')
        expected[:,1:3,2:5] = 128
        for backend in self.backends():
            with self.pair(render, label).open_paired(backend=backend) as reader:
                result = reader[:, :]
                np.testing.assert_array_equal(host(result['image'].data), image[None])
                np.testing.assert_array_equal(host(result['label'].data), labels[None])
                np.testing.assert_array_equal(host(result['label'].masks['supervision']), expected)
                self.assertTrue(host(result['label'].valid).all())

    def test_custom_volume_and_representation_alignment(self):
        data = np.arange(3*7*9, dtype='uint16').reshape(3,7,9)
        source = self.source('ct', data, 'image')
        scroll = self.scroll.with_source(source, kind='volume')
        volume = scroll.volumes['user:ct']
        representation = self.source('representation', data, 'scalar')
        representation = replace(representation, registration=VolumeRegistration('s', volume.id, affine=Affine.identity()))
        scroll = scroll.with_source(representation, kind='representation', volume=volume)
        with volume.open() as reader:
            np.testing.assert_array_equal(reader[:], data)
        with volume.open_aligned({'representation': scroll.representations[representation.id]}) as reader:
            np.testing.assert_array_equal(reader[:]['representation'].data, data[None])

    def test_pickle_and_spawn_reopen_cpu_only(self):
        image = np.arange(3*7*9, dtype='uint16').reshape(3, 7, 9)
        labels = np.arange(7*9, dtype='uint8').reshape(7, 9)
        asset = self.pair(self.source('r', image, 'image'), self.source('l', labels, 'label'))
        with asset.open_paired() as reader:
            restored = pickle.loads(pickle.dumps(reader))
            self.assertTrue(restored._lazy)
            np.testing.assert_array_equal(read_worker(restored).label.data, labels[None, 1:3, 2:5])
            with multiprocessing.get_context('spawn').Pool(1) as pool:
                result = pool.apply(read_worker, (reader,))
            np.testing.assert_array_equal(result.image.data, image[None, :, 1:3, 2:5])
        if is_available():
            with asset.open_paired(backend='cuda') as reader:
                with self.assertRaisesRegex(TypeError, 'CUDA'):
                    pickle.dumps(reader)


class ResolutionTests(unittest.TestCase):
    def test_header_resolution_absent_levels_and_contradictions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = pyramid(root/'ct', [np.zeros((8, 8, 8), 'uint8'), np.zeros((4, 4, 4), 'uint8')], [1, 2])
            # CT spacing scales in all three dimensions.
            attrs = json.loads((root/'ct'/'.zattrs').read_text())
            attrs['multiscales'][0]['datasets'][1]['coordinateTransformations'][0]['scale'][0] = 2
            (root/'ct'/'.zattrs').write_text(json.dumps(attrs))
            data = dict(samples=dict(s=dict(sample=dict(id='s', properties=dict(type='scroll')), volumes=dict(v=dict(id='v', properties=dict(pixel_size_um=1, shape=[8, 8, 8]), data=[dict(type='ome-zarr', id='ct', origins=[dict(path=str(root/'ct'), access_roots=[dict(url='')])])])))))
            c = Catalog(data)
            candidate = c.scrolls.volumes.at_resolution(2).get()
            resolved = candidate.resolve()
            self.assertTrue(resolved.verified)
            self.assertEqual(resolved.shape, (4, 4, 4))
            self.assertEqual(resolved.spacing_um, (2., 2., 2.))
            with candidate.open() as reader:
                self.assertEqual(reader.shape, (4, 4, 4))
            with self.assertRaises(ValueError):
                c.scrolls.volumes.at_resolution(4).get().resolve()
            data['samples']['s']['volumes']['v']['properties']['pixel_size_um'] = 2
            with self.assertRaisesRegex(ValueError, 'contradicts'):
                Catalog(data).scrolls.volumes.at_resolution(2).get().open()
