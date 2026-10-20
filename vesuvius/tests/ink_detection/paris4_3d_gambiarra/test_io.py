"""CPU failure-path tests: local zarr, mocked HTTP (no network), the row loader and atomic writes."""
import hashlib
import http.client
import io
import json
import tempfile
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vesuvius.ink_detection.paris4_3d_gambiarra import ct, runtime as rt, volume as V


def metadata(shape=(256, 256, 256)):
    return dict(zarr_format=2, shape=list(shape), chunks=[128] * 3, dtype='|u1', order='C',
                fill_value=0, filters=None, compressor=None, dimension_separator='/')


class Store(unittest.TestCase):
    def test_metadata_retry_and_chunk_errors(self):
        meta = json.dumps(metadata()).encode()
        with patch('urllib.request.urlopen', side_effect=[http.client.IncompleteRead(b'x'), io.BytesIO(meta)]), \
                patch('time.sleep'):
            store = ct.ChunkStore('https://example.invalid/', retries=2)
        self.addCleanup(store.close)
        self.assertEqual(store.stats['retries'], 1)
        for code in (403, 500):
            error = urllib.error.HTTPError('test', code, 'test', {}, None)
            with patch('urllib.request.urlopen', side_effect=error), patch('time.sleep'):
                with self.assertRaises(urllib.error.HTTPError):
                    store.exists((0, 0, 0))
                with self.assertRaises(urllib.error.HTTPError):
                    store.chunk((0, 0, 0))
        with patch('urllib.request.urlopen', side_effect=urllib.error.HTTPError('test', 404, 'test', {}, None)):
            self.assertFalse(store.exists((0, 0, 0)))
            self.assertIsNone(store.chunk((0, 0, 0)))

    def test_listing_pages_and_rejects_error_body(self):
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(metadata((128,) * 3)).encode())):
            store = ct.ChunkStore('https://bucket.s3.amazonaws.com/prefix/')
        self.addCleanup(store.close)
        with patch('urllib.request.urlopen', return_value=io.BytesIO(b'<Error><Code>AccessDenied</Code></Error>')):
            with self.assertRaisesRegex(ValueError, 'ListObjectsV2'):
                store.presence()
        first = b'<ListBucketResult><IsTruncated>true</IsTruncated><NextContinuationToken>a+b/c=</NextContinuationToken></ListBucketResult>'
        last = b'<ListBucketResult><IsTruncated>false</IsTruncated><Contents><Key>prefix/0/0/0</Key></Contents></ListBucketResult>'
        with patch('urllib.request.urlopen', side_effect=[io.BytesIO(first), io.BytesIO(last)]) as urlopen:
            self.assertTrue(store.presence().all())
            self.assertIn('continuation-token=a%2Bb%2Fc%3D', urlopen.call_args.args[0])

    def test_edge_padding_ignores_bytes_outside_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '.zarray').write_text(json.dumps(metadata((2, 3, 4))))
            (root / '0' / '0').mkdir(parents=True)
            (root / '0' / '0' / '0').write_bytes(bytes([7]) * 128 ** 3)
            store = ct.ChunkStore(root)
            try:
                got = store.box((0, 0, 0), (5, 5, 5), strict=False)
                want = np.zeros((5,) * 3, np.uint8)
                want[:2, :3, :4] = 7
                np.testing.assert_array_equal(got, want)
            finally:
                store.close()


class Loader(unittest.TestCase):
    def make_store(self, fetch):
        pool = ThreadPoolExecutor(2)
        self.addCleanup(pool.shutdown)
        return SimpleNamespace(pool=pool, chunk=fetch)

    def test_cancel_when_buffers_are_exhausted(self):
        store = self.make_store(lambda key: np.full((128,) * 3, key[0], np.uint8))
        plan = dict(c0=[0, 0, 0], c1=[5, 1, 1], presence=np.ones((5, 1, 1), bool))
        loader = V.RowLoader(store, plan, buffers=1)
        loader.get()  # deliberately never recycle the only buffer
        loader.close()
        self.assertFalse(loader.is_alive())

    def test_presence_controls_fetches_and_errors_propagate(self):
        def unexpected(key):
            raise AssertionError('absent chunks must not be fetched')
        loader = V.RowLoader(self.make_store(unexpected), dict(c0=[0, 0, 0], c1=[1, 1, 1],
                                                              presence=np.zeros((1,) * 3, bool)), buffers=1)
        self.assertEqual(int(loader.get()[1].sum()), 0)
        loader.close()
        loader = V.RowLoader(self.make_store(lambda key: None), dict(c0=[0, 0, 0], c1=[1, 1, 1],
                                                                     presence=np.ones((1,) * 3, bool)), buffers=1)
        try:
            with self.assertRaisesRegex(RuntimeError, 'chunk loader failed'):
                loader.get()
        finally:
            loader.close()


class Transactions(unittest.TestCase):
    def test_atomic_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.pth'
            rt.save(path, {'step': 1, 'tensor': torch.arange(4)})
            with self.assertRaises(RuntimeError):
                with rt.atomic_file(path) as f:
                    f.write(b'broken')
                    raise RuntimeError('injected failure')
            self.assertEqual(rt.load(path)['step'], 1)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_writer_partial_edge_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = V.ZarrWriter(root, V.zarray((5, 7, 6), chunk=4, compressor=None), threads=2)
            plan = dict(T0=[0, 0, 0], T1=[5, 7, 6], chunks=[(0, 0, 0), (1, 1, 1)])
            writer.begin(plan)
            codes = np.arange(5 * 7 * 6, dtype=np.uint8).reshape(5, 7, 6)
            writer.rows(0, codes[:3])
            writer.rows(3, codes[3:])
            record = writer.end()
            self.assertEqual(set(record['files']), {'0 0 0', '1 1 1'})  # only the planned chunks
            edge = np.frombuffer((root / '1' / '1' / '1').read_bytes(), np.uint8).reshape(4, 4, 4)
            expected = np.zeros((4,) * 3, np.uint8)
            expected[:1, :3, :2] = codes[4:, 4:, 4:]
            np.testing.assert_array_equal(edge, expected)

    def test_exclusive_file_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'RUN.json'
            self.assertTrue(rt.create_file(path, b'first'))
            self.assertFalse(rt.create_file(path, b'second'))
            self.assertEqual(path.read_bytes(), b'first')
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_writer_tile_cleanup_and_failed_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = V.ZarrWriter(root, V.zarray((8, 8, 16), chunk=4, compressor=None), threads=2)
            self.addCleanup(writer.close)
            (root / '0' / '0').mkdir(parents=True)
            names = ['0', '1', '2', '3', '.1.77.part', '.2.77.part', '01', 'other']  # tile x 0..1, neighbour x 2..3
            for name in names:
                (root / '0' / '0' / name).write_bytes(b'x')
            chunks, temps = writer.stored([0, 0, 0], [4, 4, 8])
            self.assertEqual((sorted(chunks), [p.name for p in temps]), (['0 0 0', '0 0 1'], ['.1.77.part']))
            self.assertEqual(writer.verify([0, 0, 0], [4, 4, 8], {'0 0 0': dict(bytes=1, sha256=hashlib.sha256(b'x').hexdigest())}),
                             ['chunk 0 0 1 not recorded'])
            writer.clean([0, 0, 0], [4, 4, 8], keep=[(0, 0, 0)])
            self.assertEqual(sorted(p.name for p in (root / '0' / '0').iterdir()), ['.2.77.part', '0', '01', '2', '3', 'other'])
            with patch('os.fsync', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    writer._write((1, 1, 1), np.zeros((4,) * 3, np.uint8))
            self.assertEqual(list((root / '1' / '1').iterdir()), [])  # the temporary file is removed

    def test_curve_truncation_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            curve = Path(tmp) / 'curve.jsonl'
            curve.write_text('{"step": 1}\n{"step": 2}\n{"step": 3}\n{"step":')
            rt.truncate_jsonl(curve, 2)
            self.assertEqual(curve.read_text(), '{"step": 1}\n{"step": 2}\n')


if __name__ == '__main__':
    unittest.main()
