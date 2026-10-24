"""Slice selection, direct destination writes and shared CPU/CUDA batches."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_source_storage import array
from vc_reader.cuda import DeviceArray, is_available
from vc_reader.reader import ZarrReader


class ZarrSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'ct'
        self.values = np.arange(5*7*9, dtype=np.uint16).reshape(5, 7, 9).astype(np.uint8)
        array(self.path, self.values)
        self.reader = ZarrReader.open(self.path, memory_cache=4096)
        self.addCleanup(self.reader.close)

    def test_numpy_clipping_integer_axes_ellipsis_and_empty_selections(self):
        selections = [np.s_[1:4, 2:6, 3:8], np.s_[-2:, ...], np.s_[2, :, -1],
                      np.s_[2, 3, 4], np.s_[:99, 5:20, :], np.s_[3:1, :, :],
                      np.s_[99:100, :, :], Ellipsis]
        with patch.object(self.reader, '_read_regions', wraps=self.reader._read_regions) as batch:
            results = self.reader.read_many(iter(selections))
            self.assertEqual(batch.call_count, 1)
        for selection, actual in zip(selections, results):
            with self.subTest(selection=selection):
                np.testing.assert_array_equal(actual, self.values[selection])
                np.testing.assert_array_equal(self.reader.read(selection), self.reader[selection])
        self.assertEqual(self.reader.read_many([]), [])

    def test_cast_into_existing_contiguous_strided_and_squeezed_buffers(self):
        selections = [np.s_[1:4, 2:6, 3:8], np.s_[2, 1:6, 3:8], np.s_[2, 3, 4]]
        for selection in selections:
            expected = self.values[selection].astype(np.float32)
            for order in ('C', 'F'):
                output = np.empty(expected.shape, dtype=np.float32, order=order)
                result = self.reader.read(selection, out=output, dtype=np.float32)
                self.assertIs(result, output)
                np.testing.assert_array_equal(result, expected)
        storage = np.full((3, 4, 10), -1, np.float32)
        output = storage[..., ::2]
        result = self.reader.read(selections[0], out=output, dtype=np.float32)
        self.assertIs(result, output)
        np.testing.assert_array_equal(output, self.values[selections[0]])
        np.testing.assert_array_equal(storage[..., 1::2], -1)
        first = np.empty((3, 4, 5), np.float32)
        results = self.reader.read_many(selections[:2], out=[first, None], dtype=np.float32)
        self.assertIs(results[0], first)
        for selection, result in zip(selections, results):
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(result, self.values[selection])
        self.assertEqual(self.reader.dtype, np.uint8)

    def test_invalid_selections_and_buffers(self):
        with self.assertRaises(ValueError):
            self.reader.read(np.s_[::0, :, :])
        for selection in (np.s_[::-1, :, :], np.s_[..., None], np.s_[9, :, :],
                          np.s_[True, :, :], np.s_[1.5, :, :]):
            with self.subTest(selection=selection), self.assertRaises(IndexError):
                self.reader.read(selection)
        selection = np.s_[1:4, 2:6, 3:8]
        for output in (np.empty((3, 4, 4), np.float32), np.empty((3, 4, 5), np.uint8)):
            with self.assertRaises(ValueError):
                self.reader.read(selection, out=output, dtype=np.float32)
        output = np.empty((3, 4, 5), np.float32)
        with self.assertRaises(ValueError):
            self.reader.read_many([selection], out=[])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.reader.read_many([selection, selection], out=[output, output], dtype=np.float32)
        output.flags.writeable = False
        with self.assertRaisesRegex(ValueError, 'writable'):
            self.reader.read(selection, out=output, dtype=np.float32)

    def test_four_dimensional_selection_and_destination(self):
        values = np.stack((self.values, self.values + 1))
        path = self.path.parent / 'channels'
        array(path, values)
        selection = np.s_[1, -3:, 2:6, 1:8]
        with ZarrReader.open(path) as reader:
            output = np.empty(values[selection].shape, np.float32)
            self.assertIs(reader.read(selection, out=output, dtype=np.float32), output)
            np.testing.assert_array_equal(output, values[selection])

    @unittest.skipUnless(is_available(), 'CUDA build and device required')
    def test_cuda_batch_matches_numpy_and_stays_on_device(self):
        selections = [np.s_[1:4, 2:6, 3:8], np.s_[-2:, ...], np.s_[2, :, -1],
                      np.s_[2, 3, 4], np.s_[3:1, :, :]]
        with self.reader.cuda(cache_bytes=4096) as reader:
            for source in ('cpu', 'chunks'):
                with self.subTest(source=source):
                    with patch.object(reader, '_read_regions', wraps=reader._read_regions) as batch:
                        values, masks = reader.read_many(selections, dtype=np.float32, source=source, valid=True)
                        self.assertEqual(batch.call_count, 1)
                    for selection, value, mask in zip(selections, values, masks):
                        self.assertIsInstance(value, DeviceArray)
                        self.assertEqual(value.dtype, np.float32)
                        np.testing.assert_array_equal(value.numpy(), self.values[selection])
                        np.testing.assert_array_equal(mask.numpy(), np.ones(np.shape(self.values[selection]), np.uint8))
                    value, mask = reader.read(selections[0], source=source, valid=True)
                    np.testing.assert_array_equal(value.numpy(), self.values[selections[0]])
                    np.testing.assert_array_equal(mask.numpy(), 1)
                    self.assertEqual(reader.read_many([], valid=True), ([], []))
                    self.assertEqual(reader.read_many([]), [])
                    with self.assertRaises(IndexError):
                        reader.read(np.s_[::-2, :, :])


def chunked(path, data, chunks):
    """Uncompressed Zarr v2 array with several (edge-padded) chunks."""
    path.mkdir(parents=True, exist_ok=True)
    (path / '.zarray').write_text(json.dumps(dict(zarr_format=2, shape=data.shape, chunks=chunks,
        dtype=data.dtype.str, compressor=None, filters=None, fill_value=0, order='C')))
    for index in np.ndindex(*[-(-n // c) for n, c in zip(data.shape, chunks)]):
        block = np.zeros(chunks, data.dtype)
        part = data[tuple(slice(i * c, (i + 1) * c) for i, c in zip(index, chunks))]
        block[tuple(slice(0, n) for n in part.shape)] = part
        (path / '.'.join(map(str, index))).write_bytes(block.tobytes())


STEPPED = [np.s_[::2, ::3, ::4], np.s_[1::3, 2:17:5, 3::7], np.s_[::9, :, ::13], np.s_[-5::2, 4, 1:-1:3],
           np.s_[2, ::4, ...], np.s_[::50, ::50, ::50], np.s_[3:4:2, 0:20:19, :1:5], np.s_[5:2:2, ::2, :]]


class SteppedSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        rng = np.random.default_rng(7)
        self.values = rng.integers(0, 255, size=(13, 20, 27), dtype=np.uint8)
        chunked(self.root / 'u8', self.values, (4, 5, 6))
        self.reader = ZarrReader.open(self.root / 'u8')
        self.addCleanup(self.reader.close)

    def test_stepped_selections_match_numpy(self):
        results = self.reader.read_many(STEPPED)
        for selection, actual in zip(STEPPED, results):
            with self.subTest(selection=selection):
                expected = self.values[selection]
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(self.reader[selection], expected)
                floating = self.reader.read(selection, dtype=np.float32)
                self.assertEqual(floating.dtype, np.float32)
                np.testing.assert_array_equal(floating, expected)
                output = np.full(expected.shape, -1.0, np.float64)
                self.assertIs(self.reader.read(selection, out=output, dtype=np.float64), output)
                np.testing.assert_array_equal(output, expected)

    def test_multibyte_and_four_dimensional_stepped_reads(self):
        rng = np.random.default_rng(8)
        values = rng.integers(-30000, 30000, size=(3, 11, 9, 14), dtype=np.int16)
        chunked(self.root / 'i16', values, (2, 4, 3, 5))
        with ZarrReader.open(self.root / 'i16') as reader:
            for selection in (np.s_[::2, 1::3, ::2, 2::4], np.s_[1, ::5, 3, ::3], np.s_[:, ::10, ::1, ::13]):
                with self.subTest(selection=selection):
                    np.testing.assert_array_equal(reader[selection], values[selection])
                    np.testing.assert_array_equal(reader.read(selection, dtype=np.float32), values[selection])

    def test_stepped_reads_skip_unsampled_chunks(self):
        # Chunks holding no sampled element are corrupt: reading one would raise.
        path = self.root / 'sparse'
        chunked(path, self.values, (4, 5, 6))
        selection = np.s_[1::8, 2::10, 0::12]
        sampled = [set(range(n)[s.start::s.step]) for n, s in zip(self.values.shape, selection)]
        for index in np.ndindex(4, 4, 5):
            if not all(any(i * c <= v < (i + 1) * c for v in keep) for i, c, keep in zip(index, (4, 5, 6), sampled)):
                (path / '.'.join(map(str, index))).write_bytes(b'corrupt')
        with ZarrReader.open(path) as reader:
            np.testing.assert_array_equal(reader[selection], self.values[selection])
            with self.assertRaises(Exception):
                reader[:, :, :]

    @unittest.skipUnless(is_available(), 'CUDA build and device required')
    def test_cuda_stepped_selections_match_numpy(self):
        with self.reader.cuda(cache_bytes=1 << 20) as reader:
            for source in ('cpu', 'chunks'):
                values, masks = reader.read_many(STEPPED, dtype=np.float32, source=source, valid=True)
                for selection, value, mask in zip(STEPPED, values, masks):
                    with self.subTest(source=source, selection=selection):
                        self.assertIsInstance(value, DeviceArray)
                        np.testing.assert_array_equal(value.numpy(), self.values[selection])
                        np.testing.assert_array_equal(mask.numpy(), np.ones(np.shape(self.values[selection]), np.uint8))
                        np.testing.assert_array_equal(reader.read(selection, source=source).numpy(), self.values[selection])

    @unittest.skipUnless(is_available(), 'CUDA build and device required')
    def test_cuda_stepped_reads_load_only_sampled_bricks(self):
        path = self.root / 'sparse_cuda'
        chunked(path, self.values, (4, 5, 6))
        selection = np.s_[1::8, 2::10, 0::12]
        sampled = [set(range(n)[s.start::s.step]) for n, s in zip(self.values.shape, selection)]
        for index in np.ndindex(4, 4, 5):
            if not all(any(i * c <= v < (i + 1) * c for v in keep) for i, c, keep in zip(index, (4, 5, 6), sampled)):
                (path / '.'.join(map(str, index))).write_bytes(b'corrupt')
        with ZarrReader.open(path) as cpu, cpu.cuda(cache_bytes=1 << 20) as reader:
            np.testing.assert_array_equal(reader.read(selection, source='chunks').numpy(), self.values[selection])
            self.assertEqual(reader.stats['chunk_uploads'], 2 * 2 * 3)


if __name__ == '__main__':
    unittest.main()
