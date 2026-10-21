import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vc_reader.reader import cache_conversion as conversion
from vc_reader.reader.disk_cache import _decoded_layer, decoded_cache_path, open_layers
from vc_reader.reader.zarr_meta import ArrayMeta
from vc_reader.reader import open_zarr
from vc_reader.cli import main


class CacheConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def array(self, dtype='u1', separator='/', name='array'):
        path = self.root / name
        path.mkdir()
        path.chmod(0o750)
        meta = dict(zarr_format=2, shape=[3, 4, 4], chunks=[2, 4, 4], dtype=np.dtype(dtype).str,
                    compressor=None, filters=None, order='C', fill_value=0,
                    dimension_separator=separator)
        (path / '.zarray').write_text(json.dumps(meta))
        (path / '.zattrs').write_text(json.dumps({'custom': {'retain': True}}))
        chunk = path / ('0/0/0' if separator == '/' else '0.0.0')
        chunk.parent.mkdir(parents=True, exist_ok=True)
        raw = (np.arange(32).reshape(2, 4, 4) * 7).astype(dtype)
        chunk.write_bytes(raw.tobytes())
        marker = path / ('1/0/0.missing' if separator == '/' else '1.0.0.missing')
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return path, meta, raw, chunk, marker

    def assert_original(self, path, meta, raw, chunk):
        self.assertEqual(json.loads((path / '.zarray').read_text()), meta)
        self.assertEqual(chunk.read_bytes(), raw.tobytes())
        self.assertFalse(list(self.root.glob('.array.delta3d-*')))

    def test_roundtrip_byteorders_separators_markers_and_cache_metadata(self):
        for i, (dtype, sep) in enumerate([('u1', '/'), ('<u2', '.'), ('>u2', '/')]):
            with self.subTest(dtype=dtype, separator=sep):
                path, meta, raw, chunk, marker = self.array(dtype, sep, f'array{i}')
                result = conversion.convert_array(path, workers=2, progress=None)
                self.assertEqual(result['before_bytes'], raw.nbytes)
                self.assertEqual(path.stat().st_mode & 0o777, 0o750)
                self.assertEqual(result['after_bytes'], chunk.stat().st_size)
                self.assertEqual(chunk.read_bytes()[:4], b'D3D1')
                self.assertEqual(chunk.read_bytes()[6], 1)
                self.assertEqual(json.loads((path / '.zarray').read_text())['dtype'], meta['dtype'])
                self.assertTrue(marker.exists())
                self.assertEqual(json.loads((path / '.zattrs').read_text()), {'custom': {'retain': True}})
                with open_zarr(str(path), threads=1) as reader:
                    np.testing.assert_array_equal(reader[:2], raw)
                # Independent Delta3D stores are usable as remote cache layers.
                layer = _decoded_layer(path, ArrayMeta.from_zarray(meta), 'http://example/array', False)
                self.assertIsNotNone(layer)
                self.assertEqual(layer.stored.compressor['id'], 'vc-delta3d')
                self.assertIsNone(conversion.convert_array(path, progress=None))

    def test_dry_run_and_cli_do_not_fetch_catalog_or_modify_array(self):
        path, meta, raw, chunk, _ = self.array()
        with patch('vc_reader.cli.Catalog.load', side_effect=AssertionError('network')):
            self.assertEqual(main(['compress-cache', str(self.root), '--dry-run']), 0)
        self.assert_original(path, meta, raw, chunk)

    def test_failure_and_interruption_keep_original(self):
        for failure in [RuntimeError('encode failed'), KeyboardInterrupt()]:
            path, meta, raw, chunk, _ = self.array(name='array' + type(failure).__name__)
            with patch.object(conversion, '_convert_chunk', side_effect=failure):
                with self.assertRaises(type(failure)):
                    conversion.convert_array(path, workers=1, progress=None)
            self.assert_original(path, meta, raw, chunk)

    def test_truncated_chunk_is_rejected_before_conversion(self):
        path, meta, raw, chunk, _ = self.array()
        chunk.write_bytes(b'short')
        with self.assertRaisesRegex(ValueError, 'expected 32'):
            conversion.convert_array(path, progress=None)
        self.assertEqual(chunk.read_bytes(), b'short')
        self.assertEqual(json.loads((path / '.zarray').read_text()), meta)

    def test_changed_source_aborts_before_publication(self):
        path, meta, raw, chunk, _ = self.array()
        original = conversion._convert_chunk
        def changed(*args):
            result = original(*args)
            (path / 'new.missing').touch()
            return result
        with patch.object(conversion, '_convert_chunk', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'source array changed'):
                conversion.convert_array(path, workers=1, progress=None)
        self.assert_original(path, meta, raw, chunk)

    def test_failed_exchange_keeps_original(self):
        path, meta, raw, chunk, _ = self.array()
        real_exchange = conversion._exchange
        def exchange(a, b):
            if a == path:
                raise OSError('exchange failed')
            real_exchange(a, b)
        with patch.object(conversion, '_exchange', side_effect=exchange):
            with self.assertRaisesRegex(OSError, 'exchange failed'):
                conversion.convert_array(path, workers=1, progress=None)
        self.assert_original(path, meta, raw, chunk)

    def test_empty_array_report_and_second_run(self):
        path, meta, _, chunk, marker = self.array()
        chunk.unlink()
        report = self.root / 'report.json'
        rows = conversion.compress_cache(self.root, report=report, progress=None)
        self.assertEqual(rows[0]['chunks'], 0)
        self.assertEqual(rows[0]['after_bytes'], 0)
        self.assertTrue(marker.exists())
        self.assertEqual(json.loads(report.read_text())['arrays'], rows)
        self.assertEqual(conversion.compress_cache(self.root, progress=None), [])

    def test_lossy_independent_cache_requires_opt_in(self):
        path, meta, _, _, _ = self.array()
        stored = dict(meta, compressor={'id': 'vc-delta3d', 'quant': 3})
        (path / '.zarray').write_text(json.dumps(stored))
        self.assertIsNone(_decoded_layer(path, ArrayMeta.from_zarray(meta), 'url', False))
        self.assertIsNotNone(_decoded_layer(path, ArrayMeta.from_zarray(meta), 'url', False, allow_lossy=True))

    def test_cache_acceptance_uses_format_not_provenance(self):
        path, meta, _, _, _ = self.array()
        for attrs in [None, {}, {'vc_reader': {'source_zarray': {'different': True}}}]:
            for codec in [None, {'id': 'vc-delta3d', 'quant': 1}, {'id': 'blosc'},
                          {'id': 'zstd'}, {'id': 'lzw'}, {'id': 'imagecodecs_lzw'}]:
                with self.subTest(attrs=attrs, codec=codec):
                    if attrs is None:
                        (path / '.zattrs').unlink(missing_ok=True)
                    else:
                        (path / '.zattrs').write_text(json.dumps(attrs))
                    (path / '.zarray').write_text(json.dumps(dict(meta, compressor=codec)))
                    self.assertIsNotNone(_decoded_layer(path, ArrayMeta.from_zarray(meta), 'url', False))
        (path / '.zarray').write_text(json.dumps(dict(meta, chunks=[1, 4, 4])))
        with self.assertRaisesRegex(ValueError, 'different array'):
            _decoded_layer(path, ArrayMeta.from_zarray(meta), 'url', False)
        (path / '.zarray').write_text(json.dumps(dict(meta, compressor={'id': 'unknown'})))
        with self.assertRaisesRegex(ValueError, 'cannot read'):
            _decoded_layer(path, ArrayMeta.from_zarray(meta), 'url', False)

    def test_read_only_codecs_are_not_used_for_raw_cache_writes(self):
        path, meta, _, _, _ = self.array()
        url = 'http://example/array'
        cached = Path(decoded_cache_path(self.root / 'cache', url))
        cached.parent.mkdir(parents=True)
        path.rename(cached)
        (cached / '.zarray').write_text(json.dumps(dict(meta, compressor={'id': 'zstd'})))
        for budget in [None, 1 << 20]:
            layers, writer = open_layers([self.root / 'cache'], url, ArrayMeta.from_zarray(meta),
                                         group_url=None, level_key=None, cache_max_bytes=budget)
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0].stored.codec_id, 'zstd')
            self.assertIsNone(writer)

    def test_budgeted_and_consolidated_stores_are_rejected(self):
        path, _, _, _, _ = self.array()
        for name in ['.vc_reader_cache.sqlite3', '.zmetadata']:
            file = self.root / name
            file.touch()
            with self.assertRaises(ValueError):
                conversion.convert_array(path, progress=None)
            file.unlink()


if __name__ == '__main__':
    unittest.main()
