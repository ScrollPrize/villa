"""Local source-reading contracts, exact labels, memory limits and worker lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, PropertyMock

import numpy as np
try:
    from PIL import Image
    import tifffile
except ImportError as error:
    raise unittest.SkipTest('source reader tests require the optional labels extra') from error

from vc_reader import Affine, Encoding, Selection, Source, Storage, VolumeRegistration
from vc_reader.reader import DecodedCache


def worker_read(reader, connection):
    try:
        connection.send((reader.read((0, 0), (2, 3)), reader.decoded_cache.stats()))
    except BaseException as error:
        connection.send(error)
    finally:
        connection.close()


class SourceReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def png(self, name='image.png', values=None, **source_options):
        if values is None:
            values = np.arange(35, dtype=np.uint8).reshape(5, 7)
        path = self.root / name
        Image.fromarray(values).save(path)
        return Source(name, Storage(str(path), 'png'), **source_options), values

    def tiff(self, values, *, axes=None, filename='image.tif', **write_options):
        path = self.root / filename
        tifffile.imwrite(path, values, metadata={'axes': axes} if axes else None, **write_options)
        return Source(filename, Storage(str(path), 'tiff'))

    def test_spatial_crop_padding_and_buffer_ownership(self):
        source, data = self.png(fill_value=83)
        reader = source.open()
        self.assertEqual((reader.shape, reader.axes, reader.dtype), (data.shape, ('y', 'x'), data.dtype))
        result = reader.read((-2, 3), (6, 6))
        expected = np.full((6, 6), 83, dtype=np.uint8)
        expected[2:, :4] = data[:4, 3:]
        np.testing.assert_array_equal(result, expected)
        result[:] = 42
        np.testing.assert_array_equal(reader.read((0, 0), data.shape), data)
        self.assertGreater(reader.decoded_cache.stats()['hits'], 0)
        reader.close()
        reader.close()
        np.testing.assert_array_equal(result, 42)
        with self.assertRaisesRegex(ValueError, 'closed'):
            reader.read((0, 0), (1, 1))
        with source.open() as reader:
            self.assertEqual(reader.read((0, 0), (0, 3)).shape, (0, 3))
            np.testing.assert_array_equal(reader.read((99, 99), (2, 2)), 83)
            with self.assertRaisesRegex(ValueError, 'spatial'):
                reader.read((0, 0, 0), (1, 1, 1))
            with self.assertRaises(TypeError):
                reader.read((0.5, 0), (1, 1))
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                reader.read((0, 0), (-1, 2))

    def test_rgb_channel_selection_and_color_mapping(self):
        rgb = np.zeros((5, 7, 3), dtype=np.uint8)
        rgb[:, :3] = (255, 0, 0)
        rgb[:, 3:] = (0, 128, 255)
        source, _ = self.png(values=rgb)
        with source.with_selection(channels=(2, 0)).open() as reader:
            self.assertEqual(reader.shape, (5, 7, 2))
            np.testing.assert_array_equal(reader.read((1, 2), (2, 4)), rgb[1:3, 2:6, [2, 0]])
        mapped = replace(source, encoding=Encoding('color', (((255, 0, 0), 2**40+1),
                                                           ((0, 128, 255), 2**40+2)), 'uint64'), fill_value=9)
        with mapped.open() as reader:
            self.assertEqual(reader.axes, ('y', 'x'))
            self.assertEqual(reader.shape, (5, 7))
            self.assertEqual(reader.dtype, np.dtype('uint64'))
            out = reader.read((-1, 2), (3, 4))
            expected = np.full((3, 4), 9, dtype=np.uint64)
            expected[1:, 0] = 2**40+1
            expected[1:, 1:] = 2**40+2
            np.testing.assert_array_equal(out, expected)

    def test_numpy_indexing_axes_channels_encoding_and_bounds(self):
        rgb = np.arange(5*7*3, dtype='uint8').reshape(5, 7, 3)
        source, _ = self.png(values=rgb)
        keys = [Ellipsis, (), np.s_[1:4, 2:6], np.s_[..., -1], np.s_[-2, :, 0:2],
                np.s_[1, 2, 0], np.s_[-100:100, 5:99, :], np.s_[3:1, :, :],
                np.s_[:, :, 2:1], np.s_[99:, :, :]]
        with source.open() as reader:
            for key in keys:
                with self.subTest(key=key):
                    np.testing.assert_array_equal(reader[key], rgb[key])
                    self.assertEqual(reader[key].shape, rgb[key].shape)
            for key in [np.s_[5, :, :], np.s_[-6, :, :], np.s_[..., ...],
                        np.s_[:, :, :, :], np.s_[::2], None, True, np.bool_(False), [0, 1], .5]:
                with self.subTest(invalid=key), self.assertRaises(IndexError):
                    reader[key]
            with self.assertRaises(ValueError):
                reader[::0]
            saved = reader[1:3, 2:5, 1]
        np.testing.assert_array_equal(saved, rgb[1:3, 2:5, 1])
        with self.assertRaisesRegex(ValueError, 'closed'):
            reader[:]

        data = np.moveaxis(rgb, -1, 0)
        channel_first = self.tiff(data, axes='CYX', photometric='rgb', planarconfig='separate')
        with channel_first.with_selection(channels=(2, 0)).open() as reader:
            np.testing.assert_array_equal(reader[1, 1:4, -3:], data[0, 1:4, -3:])
        encoded = replace(source, encoding=Encoding('color', tuple((tuple(map(int, color)), i)
                           for i, color in enumerate(rgb.reshape(-1, 3))), 'uint64'))
        with encoded.open() as reader:
            expected = np.arange(35, dtype='uint64').reshape(5, 7)
            np.testing.assert_array_equal(reader[-2:, 1:4], expected[-2:, 1:4])

    def test_palette_indices_alpha_and_png16(self):
        indices = np.arange(35, dtype=np.uint8).reshape(5, 7) % 2
        path = self.root / 'palette.png'
        image = Image.fromarray(indices).convert('P')
        image.putpalette([255, 0, 0, 0, 255, 0] + [0]*762)
        image.save(path, transparency=0)
        source = Source('palette', Storage(str(path), 'png'))
        with source.open() as reader:
            np.testing.assert_array_equal(reader.read((0, 0), indices.shape), indices)
            self.assertEqual(reader.palette['values'][:6], (255, 0, 0, 0, 255, 0))
            self.assertEqual(reader.palette['transparency'], 0)
        mapped = replace(source, encoding=Encoding('integer', ((0, 17), (1, 23)), 'uint16'))
        with mapped.open() as reader:
            np.testing.assert_array_equal(reader.read((0, 0), indices.shape), np.where(indices, 23, 17))
        rgba = np.arange(140, dtype=np.uint8).reshape(5, 7, 4)
        source, _ = self.png('rgba.png', rgba)
        with source.open() as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), rgba)
        gray = np.arange(35, dtype=np.uint16).reshape(5, 7) * 1000
        source, _ = self.png('gray16.png', gray)
        with source.open() as reader:
            self.assertEqual(reader.dtype, np.dtype('uint16'))
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), gray)

    def test_unknown_encoding_and_exact_large_integer_values(self):
        data = np.array([[2**63+1, 2**63+2, 2**63+3]], dtype=np.uint64)
        source = self.tiff(data, axes='YX')
        mapped = replace(source, encoding=Encoding('integer', ((2**63+1, 2**62+1),
                                                             (2**63+2, 2**62+2)), 'uint64'))
        with mapped.open() as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (1, 2)), [[2**62+1, 2**62+2]])
            with self.assertRaisesRegex(ValueError, 'user:image.tif: unknown encoding'):
                reader.read((0, 2), (1, 1))
            np.testing.assert_array_equal(reader.read((-1, 0), (1, 2)), 0)
        mapped = replace(mapped, encoding=replace(mapped.encoding, fallback=99))
        with mapped.open() as reader:
            self.assertEqual(reader.read((0, 2), (1, 1)).item(), 99)
        with self.assertRaisesRegex(ValueError, 'representable'):
            replace(source, fill_value=-1).open()

    def test_jpeg_values_are_decoded_without_conversion(self):
        path = self.root / 'image.jpg'
        rgb = np.arange(105, dtype=np.uint8).reshape(5, 7, 3)
        Image.fromarray(rgb).save(path, quality=85)
        with Image.open(path) as image:
            expected = np.array(image)
        with Source('jpeg', Storage(str(path), 'jpeg')).open() as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), expected)

    def test_tiff_pages_channel_first_and_time(self):
        data = np.arange(3*5*7, dtype=np.uint16).reshape(3, 5, 7)
        source = self.tiff(data, axes='ZYX', photometric='minisblack')
        with source.open() as reader:
            self.assertEqual(reader.axes, ('z', 'y', 'x'))
            np.testing.assert_array_equal(reader.read((1, 1, 2), (2, 3, 4)), data[1:, 1:4, 2:6])
        channel_first = self.tiff(data, axes='CYX', filename='channels.tif', photometric='rgb', planarconfig='separate')
        with channel_first.with_selection(channels=(2, 0)).open() as reader:
            self.assertEqual(reader.axes, ('c', 'y', 'x'))
            np.testing.assert_array_equal(reader.read((1, 2), (3, 4)), data[[2, 0], 1:4, 2:6])
        time = self.tiff(data, axes='TYX', filename='time.tif', photometric='minisblack')
        with self.assertRaisesRegex(ValueError, 'time selection'):
            time.open()
        with time.with_selection(time=2).open() as reader:
            self.assertEqual(reader.axes, ('y', 'x'))
            np.testing.assert_array_equal(reader.read((0, 0), (5, 7)), data[2])

    def test_ambiguous_tiff_axes_series_and_missing_level(self):
        data = np.arange(3*5*7, dtype=np.uint16).reshape(3, 5, 7)
        source = self.tiff(data, photometric='minisblack')
        with self.assertRaisesRegex(ValueError, 'ambiguous storage axes'):
            source.open()
        with replace(source, axes=('z', 'y', 'x')).open() as reader:
            np.testing.assert_array_equal(reader.read((0, 0, 0), data.shape), data)
        path = self.root / 'series.tif'
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.ones((5, 7), dtype=np.uint8))
            writer.write(np.full((9, 11), 7, dtype=np.uint8))
        source = Source('series', Storage(str(path), 'tiff'))
        with self.assertRaisesRegex(ValueError, 'select series'):
            source.open()
        with source.with_selection(series=1).open() as reader:
            np.testing.assert_array_equal(reader.read((0, 0), (2, 3)), 7)
        with self.assertRaisesRegex(ValueError, 'level 4 does not exist'):
            source.with_selection(series=1, level=4).open()
        with self.assertRaisesRegex(ValueError, 'contradict storage axes'):
            replace(source.with_selection(series=1), axes=('x', 'y')).open()

    def test_stack_is_ordered_and_validates_only_requested_planes(self):
        first, data = self.png('first.png')
        second, other = self.png('second.png', np.full((5, 7), 99, dtype=np.uint8))
        missing = str(self.root / 'missing.png')
        source = Source('stack', Storage(format='stack', locations=(second.storage.location, first.storage.location, missing)))
        with source.open() as reader:
            self.assertEqual(reader.shape, (3, 5, 7))
            result = reader.read((0, 0, 0), (2, 5, 7))
            np.testing.assert_array_equal(result, np.stack([other, data]))
            self.assertNotIn(2, reader._files)
            with self.assertRaisesRegex(FileNotFoundError, 'user:stack'):
                reader.read((2, 0, 0), (1, 1, 1))
        wrong, _ = self.png('wrong.png', np.zeros((3, 4), dtype=np.uint8))
        source = replace(source, storage=replace(source.storage, locations=(first.storage.location, wrong.storage.location)))
        with source.open() as reader:
            with self.assertRaisesRegex(ValueError, 'plane 1 shape/dtype'):
                reader.read((1, 0, 0), (1, 2, 2))

    def test_limits_precede_decode_and_shared_cache_survives_close(self):
        source, data = self.png()
        with source.open(max_decode_bytes=32) as reader:
            with patch('PIL.Image.Image.load', side_effect=AssertionError('decoded before limit check')):
                with self.assertRaisesRegex(MemoryError, 'max_decode_bytes'):
                    reader.read((0, 0), (1, 1))
        cache = DecodedCache(max_bytes=data.nbytes, max_decode_bytes=2 << 20)
        first, second = source.open(decoded_cache=cache), source.open(decoded_cache=cache)
        out = first.read((0, 0), (2, 3))
        first.close()
        np.testing.assert_array_equal(second.read((0, 0), (2, 3)), out)
        self.assertEqual(cache.stats()['hits'], 1)
        self.assertEqual(cache.stats()['misses'], 1)
        self.assertEqual(cache.stats()['nbytes'], data.nbytes)
        cache.clear()
        second.close()
        np.testing.assert_array_equal(out, data[:2, :3])
        with self.assertRaisesRegex(ValueError, 'configure byte budgets'):
            source.open(decoded_cache=cache, decoded_cache_bytes=10)
        with source.open(decoded_cache_bytes=1) as reader:
            out = reader.read((0, 0), (1, 1))
            self.assertEqual(reader.decoded_cache.stats()['nbytes'], 0)
            self.assertIsNone(out.base)

    def test_compressed_tiff_budget_and_mmap_small_crop(self):
        values = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        compressed = self.tiff(values, axes='YX', compression='deflate')
        with compressed.open(max_decode_bytes=64) as reader:
            with patch.object(tifffile.TiffPage, 'asarray', side_effect=AssertionError('decoded too early')):
                with self.assertRaises(MemoryError):
                    reader.read((0, 0), (1, 1))
        with compressed.open() as reader:
            np.testing.assert_array_equal(reader.read((4, 7), (3, 5)), values[4:7, 7:12])
        mapped = self.tiff(values, axes='YX', filename='mapped.tif')
        with mapped.open(max_decode_bytes=0) as reader:
            np.testing.assert_array_equal(reader.read((4, 7), (3, 5)), values[4:7, 7:12])
            self.assertEqual(reader.decoded_cache.stats()['nbytes'], 0)

    def test_raw_unregistered_and_declared_metadata(self):
        source, data = self.png()
        with source.open() as reader:
            self.assertIsNone(reader.registration)
        registered = replace(source, registration=VolumeRegistration('s', 'v', affine=Affine.scale(2)))
        with registered.open() as reader:
            self.assertEqual(reader.registration, registered.registration)
        for kwargs in ({'shape': (7, 5)}, {'dtype': 'uint16'}, {'axes': ('x', 'y')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(source, **kwargs).open()

    def test_float32_singleton_channels_and_missing_codec(self):
        values = np.array([[[0., -0., 1.25], [np.nan, np.inf, -2.]]], dtype=np.float32)
        source = self.tiff(values, axes='CYX', photometric='minisblack')
        with source.open() as reader:
            self.assertEqual(reader.shape, values.shape)
            output = reader.read((0, 0), (2, 3))
            self.assertEqual(output.dtype, np.dtype('float32'))
            np.testing.assert_array_equal(output.view(np.uint32), values.view(np.uint32))
        source = self.tiff(np.zeros((5, 7), dtype=np.uint8), axes='YX',
                           filename='compressed.tif', compression='deflate')
        with source.open() as reader, patch.object(tifffile.TiffPage, 'decode', new_callable=PropertyMock,
                                                   side_effect=ValueError('requires imagecodecs')):
            with self.assertRaisesRegex(ValueError, 'TIFF codec.*imagecodecs'):
                reader.read((0, 0), (1, 1))

    @unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'fork is unavailable')
    def test_fork_reopens_without_inherited_cache_lock(self):
        source, data = self.png()
        reader = source.open()
        reader.read((0, 0), (2, 3))
        context = multiprocessing.get_context('fork')
        parent, child = context.Pipe()
        process = context.Process(target=worker_read, args=(reader, child))
        with reader.decoded_cache._lock:
            process.start()
            child.close()
            try:
                self.assertTrue(parent.poll(10), 'forked worker did not return')
                result = parent.recv()
                if isinstance(result, BaseException):
                    raise result
                output, stats = result
                np.testing.assert_array_equal(output, data[:2, :3])
                self.assertEqual(stats['misses'], 1)
                self.assertEqual(stats['hits'], 0)
            finally:
                parent.close()
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join()
        self.assertEqual(process.exitcode, 0)
        reader.close()

    def test_pickle_spawn_and_process_local_cache(self):
        source, data = self.png()
        cache = DecodedCache()
        reader = source.open(decoded_cache=cache)
        reader.read((0, 0), (2, 3))
        payload = pickle.dumps((reader, source.open(decoded_cache=cache)))
        first, second = pickle.loads(payload)
        self.assertIs(first.decoded_cache, second.decoded_cache)
        self.assertEqual(first.decoded_cache.stats()['entries'], 0)
        self.assertEqual(first.shape, data.shape)
        np.testing.assert_array_equal(first.read((0, 0), (2, 3)), data[:2, :3])
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe()
        process = context.Process(target=worker_read, args=(reader, child))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(20), 'worker did not return')
            result = parent.recv()
            if isinstance(result, BaseException):
                raise result
            output, stats = result
            np.testing.assert_array_equal(output, data[:2, :3])
            self.assertEqual(stats['misses'], 1)
            self.assertEqual(stats['hits'], 0)
        finally:
            parent.close()
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join()
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(cache.stats()['misses'], 1)
        reader.close()
        closed = pickle.loads(pickle.dumps(reader))
        with self.assertRaisesRegex(ValueError, 'closed'):
            closed.read((0, 0), (1, 1))

    def test_concurrent_readers_and_eviction(self):
        source, data = self.png()
        other, _ = self.png('other.png', data + 1)
        cache = DecodedCache(max_bytes=data.nbytes)
        readers = [source.open(decoded_cache=cache) for _ in range(8)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(lambda r: r.read((0, 0), (5, 7)), readers))
        self.assertEqual(cache.stats()['misses'], 1)
        self.assertEqual(cache.stats()['hits'], 7)
        with other.open(decoded_cache=cache) as reader:
            reader.read((0, 0), (5, 7))
        self.assertEqual(cache.stats()['entries'], 1)
        for reader, output in zip(readers, outputs):
            reader.close()
            np.testing.assert_array_equal(output, data)

    def test_modified_files_corruption_and_lazy_imports(self):
        source, data = self.png()
        with source.open() as reader:
            reader.read((0, 0), (1, 1))
            Image.fromarray(data + 1).save(source.storage.location)
            with self.assertRaisesRegex(ValueError, 'source changed'):
                reader.read((0, 0), (1, 1))
        path = self.root / 'bad.png'
        path.write_bytes(b'not an image')
        with self.assertRaises(OSError):
            Source('bad', Storage(str(path), 'png')).open()
        code = '''
import sys
from vc_reader import Source, Storage
from vc_reader.reader import DecodedCache
s = Source('unopened', Storage('/absent.png', 'png'))
assert callable(s.open)
assert not any(n in sys.modules for n in ('numpy', 'PIL', 'tifffile', 'torch', 'vc_reader._reader'))
'''
        code = f'import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})\n' + code
        result = subprocess.run([sys.executable, '-S', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
